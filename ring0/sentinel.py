"""Sentinel — Ring 0 main loop (pure stdlib).

Launches and supervises Ring 2.  On success (survived max_runtime_sec),
triggers Ring 1 evolution to mutate the code.  On failure, rolls back
to the last known-good commit, evolves from that base, and restarts.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
import threading
import time
import tomllib

from ring0.fitness import FitnessTracker
from ring0.git_manager import GitManager
from ring0.heartbeat import HeartbeatMonitor
from ring0.memory import MemoryStore
from ring0.parameter_seed import generate_params, params_to_dict
from ring0.resource_monitor import check_resources

log = logging.getLogger("protea.sentinel")


def _load_config(project_root: pathlib.Path) -> dict:
    cfg_path = project_root / "config" / "config.toml"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def _start_ring2(ring2_path: pathlib.Path, heartbeat_path: pathlib.Path) -> subprocess.Popen:
    """Launch the Ring 2 process and return its Popen handle."""
    env = {**os.environ, "PROTEA_HEARTBEAT": str(heartbeat_path)}
    proc = subprocess.Popen(
        [sys.executable, str(ring2_path / "main.py")],
        cwd=str(ring2_path),
        env=env,
    )
    log.info("Ring 2 started  pid=%d", proc.pid)
    return proc


def _stop_ring2(proc: subprocess.Popen | None) -> None:
    """Terminate the Ring 2 process if it is still running."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log.info("Ring 2 stopped  pid=%d", proc.pid)


def _try_evolve(project_root, fitness, ring2_path, generation, params, survived, notifier, directive="", memory_store=None):
    """Best-effort evolution.  Returns True if new code was written."""
    try:
        from ring1.config import load_ring1_config
        from ring1.evolver import Evolver

        r1_config = load_ring1_config(project_root)
        if not r1_config.claude_api_key:
            log.warning("CLAUDE_API_KEY not set — skipping evolution")
            return False

        memories = memory_store.get_recent(5) if memory_store else []
        evolver = Evolver(r1_config, fitness, memory_store=memory_store)
        result = evolver.evolve(
            ring2_path=ring2_path,
            generation=generation,
            params=params_to_dict(params),
            survived=survived,
            directive=directive,
            memories=memories,
        )
        if result.success:
            log.info("Evolution succeeded: %s", result.reason)
            return True
        else:
            log.warning("Evolution failed: %s", result.reason)
            if notifier:
                notifier.notify_error(generation, result.reason)
            return False
    except Exception as exc:
        log.error("Evolution error (non-fatal): %s", exc)
        if notifier:
            notifier.notify_error(generation, str(exc))
        return False


def _create_memory_store(db_path):
    """Best-effort MemoryStore creation.  Returns None on any error."""
    try:
        return MemoryStore(db_path)
    except Exception as exc:
        log.debug("MemoryStore not available: %s", exc)
        return None


def _create_notifier(project_root):
    """Best-effort Telegram notifier creation.  Returns None on any error."""
    try:
        from ring1.config import load_ring1_config
        from ring1.telegram import create_notifier

        r1_config = load_ring1_config(project_root)
        return create_notifier(r1_config)
    except Exception as exc:
        log.debug("Telegram notifier not available: %s", exc)
        return None


def _create_bot(project_root, state, fitness, ring2_path):
    """Best-effort Telegram bot creation.  Returns None on any error."""
    try:
        from ring1.config import load_ring1_config
        from ring1.telegram_bot import create_bot, start_bot_thread

        r1_config = load_ring1_config(project_root)
        bot = create_bot(r1_config, state, fitness, ring2_path)
        if bot:
            start_bot_thread(bot)
            log.info("Telegram bot started")
        return bot
    except Exception as exc:
        log.debug("Telegram bot not available: %s", exc)
        return None


def _create_executor(project_root, state, ring2_path, reply_fn, memory_store=None):
    """Best-effort task executor creation.  Returns None on any error."""
    try:
        from ring1.config import load_ring1_config
        from ring1.task_executor import create_executor, start_executor_thread

        r1_config = load_ring1_config(project_root)
        executor = create_executor(r1_config, state, ring2_path, reply_fn, memory_store=memory_store)
        if executor:
            start_executor_thread(executor)
            log.info("Task executor started")
        return executor
    except Exception as exc:
        log.debug("Task executor not available: %s", exc)
        return None


def run(project_root: pathlib.Path) -> None:
    """Sentinel main loop — run until interrupted."""
    cfg = _load_config(project_root)
    r0 = cfg["ring0"]

    ring2_path = project_root / r0["git"]["ring2_path"]
    heartbeat_path = ring2_path / ".heartbeat"
    db_path = project_root / r0["fitness"]["db_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)

    interval = r0["heartbeat_interval_sec"]
    timeout = r0["heartbeat_timeout_sec"]
    seed = r0["evolution"]["seed"]

    git = GitManager(ring2_path)
    git.init_repo()
    fitness = FitnessTracker(db_path)
    memory_store = _create_memory_store(db_path)
    hb = HeartbeatMonitor(heartbeat_path, timeout_sec=timeout)
    notifier = _create_notifier(project_root)

    # Shared state for Telegram bot interaction.
    from ring1.telegram_bot import SentinelState
    state = SentinelState()
    state.memory_store = memory_store
    bot = _create_bot(project_root, state, fitness, ring2_path)

    # Task executor for P0 user tasks.
    reply_fn = bot._send_reply if bot else (lambda text: None)
    executor = _create_executor(project_root, state, ring2_path, reply_fn, memory_store=memory_store)

    generation = 0
    last_good_hash: str | None = None
    proc: subprocess.Popen | None = None

    # Initial snapshot of seed code.
    try:
        last_good_hash = git.snapshot(f"gen-{generation} seed")
    except subprocess.CalledProcessError:
        pass

    log.info("Sentinel online — heartbeat every %ds, timeout %ds", interval, timeout)

    try:
        params = generate_params(generation, seed)
        proc = _start_ring2(ring2_path, heartbeat_path)
        start_time = time.time()
        hb.wait_for_heartbeat(startup_timeout=timeout)

        while True:
            state.p0_event.wait(timeout=interval)
            state.p0_event.clear()

            # --- resource check ---
            ok, msg = check_resources(
                r0["max_cpu_percent"],
                r0["max_memory_percent"],
                r0["max_disk_percent"],
            )
            if not ok:
                log.warning("Resource alert: %s", msg)

            elapsed = time.time() - start_time

            # --- update shared state for bot ---
            with state.lock:
                state.generation = generation
                state.start_time = start_time
                state.alive = hb.is_alive()
                state.mutation_rate = params.mutation_rate
                state.max_runtime_sec = params.max_runtime_sec

            # --- pause check (bot can set this) ---
            if state.pause_event.is_set():
                continue

            # --- kill check (bot can set this) ---
            if state.kill_event.is_set():
                state.kill_event.clear()
                log.info("Kill signal received — restarting Ring 2 (gen-%d)", generation)
                _stop_ring2(proc)
                proc = _start_ring2(ring2_path, heartbeat_path)
                start_time = time.time()
                hb.wait_for_heartbeat(startup_timeout=timeout)
                continue

            # --- success check: survived max_runtime_sec ---
            if elapsed >= params.max_runtime_sec and hb.is_alive():
                log.info(
                    "Ring 2 survived gen-%d (%.1fs >= %ds)",
                    generation, elapsed, params.max_runtime_sec,
                )
                _stop_ring2(proc)

                # Record success.
                commit_hash = last_good_hash or "unknown"
                fitness.record(
                    generation=generation,
                    commit_hash=commit_hash,
                    score=1.0,
                    runtime_sec=elapsed,
                    survived=True,
                )

                with state.lock:
                    state.last_score = 1.0
                    state.last_survived = True

                # Snapshot the surviving code.
                try:
                    last_good_hash = git.snapshot(f"gen-{generation} survived")
                except subprocess.CalledProcessError:
                    pass

                # Record observation in memory.
                if memory_store:
                    memory_store.add(
                        generation, "observation",
                        f"Gen {generation} survived {elapsed:.0f}s (max {params.max_runtime_sec}s). "
                        f"Code: {len((ring2_path / 'main.py').read_text())} bytes.",
                    )

                # Evolve (best-effort) — skip if P0 task active.
                if state.p0_active.is_set():
                    log.info("P0 task active — skipping evolution")
                    evolved = False
                else:
                    with state.lock:
                        directive = state.evolution_directive
                        state.evolution_directive = ""
                    if directive and memory_store:
                        memory_store.add(generation, "directive", directive)
                    evolved = _try_evolve(
                        project_root, fitness, ring2_path,
                        generation, params, True, notifier,
                        directive=directive,
                        memory_store=memory_store,
                    )
                if evolved:
                    try:
                        git.snapshot(f"gen-{generation} evolved")
                    except subprocess.CalledProcessError:
                        pass

                # Notify.
                if notifier:
                    notifier.notify_generation_complete(
                        generation, 1.0, True, last_good_hash or "unknown",
                    )

                # Next generation.
                generation += 1
                params = generate_params(generation, seed)
                log.info("Starting generation %d (params: %s)", generation, params)
                proc = _start_ring2(ring2_path, heartbeat_path)
                start_time = time.time()
                hb.wait_for_heartbeat(startup_timeout=timeout)
                continue

            # --- heartbeat check ---
            if hb.is_alive():
                continue

            # Ring 2 is dead — failure path.
            log.warning("Ring 2 lost heartbeat after %.1fs (gen-%d)", elapsed, generation)
            _stop_ring2(proc)

            score = min(elapsed / params.max_runtime_sec, 0.99) if params.max_runtime_sec > 0 else 0.0
            commit_hash = last_good_hash or "unknown"
            fitness.record(
                generation=generation,
                commit_hash=commit_hash,
                score=score,
                runtime_sec=elapsed,
                survived=False,
            )

            with state.lock:
                state.last_score = score
                state.last_survived = False

            # Rollback to last known-good code.
            if last_good_hash:
                log.info("Rolling back to %s", last_good_hash[:12])
                git.rollback(last_good_hash)

            # Record observation in memory.
            if memory_store:
                memory_store.add(
                    generation, "observation",
                    f"Gen {generation} died after {elapsed:.0f}s / {params.max_runtime_sec}s. Heartbeat lost.",
                )

            # Evolve from the good base (best-effort) — skip if P0 active.
            if state.p0_active.is_set():
                log.info("P0 task active — skipping evolution")
                evolved = False
            else:
                with state.lock:
                    directive = state.evolution_directive
                    state.evolution_directive = ""
                if directive and memory_store:
                    memory_store.add(generation, "directive", directive)
                evolved = _try_evolve(
                    project_root, fitness, ring2_path,
                    generation, params, False, notifier,
                    directive=directive,
                    memory_store=memory_store,
                )
            if evolved:
                try:
                    git.snapshot(f"gen-{generation} evolved-from-rollback")
                except subprocess.CalledProcessError:
                    pass

            # Notify.
            if notifier:
                notifier.notify_generation_complete(
                    generation, score, False, commit_hash,
                )

            # Next generation.
            generation += 1
            params = generate_params(generation, seed)
            log.info("Restarting Ring 2 — generation %d (params: %s)", generation, params)
            proc = _start_ring2(ring2_path, heartbeat_path)
            start_time = time.time()
            hb.wait_for_heartbeat(startup_timeout=timeout)

    except KeyboardInterrupt:
        log.info("Sentinel shutting down (KeyboardInterrupt)")
    finally:
        if executor:
            executor.stop()
        if bot:
            bot.stop()
        _stop_ring2(proc)
        log.info("Sentinel offline")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    project_root = pathlib.Path(__file__).resolve().parent.parent
    run(project_root)
