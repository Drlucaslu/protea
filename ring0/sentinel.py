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

from ring0.commit_watcher import CommitWatcher
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
    log_file = ring2_path / ".output.log"
    fh = open(log_file, "w")
    env = {**os.environ, "PROTEA_HEARTBEAT": str(heartbeat_path)}
    proc = subprocess.Popen(
        [sys.executable, str(ring2_path / "main.py")],
        cwd=str(ring2_path),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    proc._log_fh = fh          # keep reference for later close
    proc._log_path = log_file  # keep path for later read
    log.info("Ring 2 started  pid=%d", proc.pid)
    return proc


def _stop_ring2(proc: subprocess.Popen | None) -> None:
    """Terminate the Ring 2 process if it is still running."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.info("Ring 2 stopped  pid=%d", proc.pid)
    fh = getattr(proc, "_log_fh", None)
    if fh:
        fh.close()


def _read_ring2_output(proc, max_lines: int = 100) -> str:
    """Read the last *max_lines* from Ring 2's captured output log."""
    log_path = getattr(proc, "_log_path", None)
    if not log_path or not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _classify_failure(proc, output: str) -> str:
    """Determine why Ring 2 failed based on return code and output."""
    rc = proc.returncode
    if rc is None:
        return "heartbeat timeout (process still running)"
    if rc < 0:
        import signal as _signal
        sig = _signal.Signals(-rc).name if -rc in _signal.Signals._value2member_map_ else str(-rc)
        return f"killed by signal {sig}"
    if rc != 0:
        # Extract the last Traceback from output.
        lines = output.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Traceback"):
                return "\n".join(lines[i:])
        return f"exit code {rc}"
    return "clean exit but heartbeat lost"


def _should_evolve(state, cooldown_sec: int) -> bool:
    """Check whether evolution should proceed based on priority and cooldown."""
    if state.p0_active.is_set():
        return False
    if state.p1_active.is_set():
        return False
    if time.time() - state.last_evolution_time < cooldown_sec:
        return False
    return True


def _try_evolve(project_root, fitness, ring2_path, generation, params, survived, notifier, directive="", memory_store=None, skill_store=None, crash_logs=None):
    """Best-effort evolution.  Returns True if new code was written."""
    try:
        from ring1.config import load_ring1_config
        from ring1.evolver import Evolver

        r1_config = load_ring1_config(project_root)
        if not r1_config.claude_api_key:
            log.warning("CLAUDE_API_KEY not set — skipping evolution")
            return False

        memories = memory_store.get_recent(5) if memory_store else []

        # Gather task history and skills for directed evolution.
        task_history = []
        if memory_store:
            try:
                task_history = memory_store.get_by_type("task", limit=10)
            except Exception:
                pass

        skills = []
        if skill_store:
            try:
                skills = skill_store.get_active(20)
            except Exception:
                pass

        evolver = Evolver(r1_config, fitness, memory_store=memory_store)
        result = evolver.evolve(
            ring2_path=ring2_path,
            generation=generation,
            params=params_to_dict(params),
            survived=survived,
            directive=directive,
            memories=memories,
            task_history=task_history,
            skills=skills,
            crash_logs=crash_logs,
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


def _try_crystallize(project_root, skill_store, source_code, output, generation, skill_cap=100):
    """Best-effort crystallization.  Returns action string or None."""
    try:
        from ring1.config import load_ring1_config
        from ring1.crystallizer import Crystallizer

        r1_config = load_ring1_config(project_root)
        if not r1_config.claude_api_key:
            log.warning("CLAUDE_API_KEY not set — skipping crystallization")
            return None

        crystallizer = Crystallizer(r1_config, skill_store)
        result = crystallizer.crystallize(
            source_code=source_code,
            output=output,
            generation=generation,
            skill_cap=skill_cap,
        )
        log.info("Crystallization result: action=%s skill=%s reason=%s",
                 result.action, result.skill_name, result.reason)
        return result.action
    except Exception as exc:
        log.error("Crystallization error (non-fatal): %s", exc)
        return None


def _create_memory_store(db_path):
    """Best-effort MemoryStore creation.  Returns None on any error."""
    try:
        return MemoryStore(db_path)
    except Exception as exc:
        log.debug("MemoryStore not available: %s", exc)
        return None


def _create_skill_store(db_path):
    """Best-effort SkillStore creation.  Returns None on any error."""
    try:
        from ring0.skill_store import SkillStore
        return SkillStore(db_path)
    except Exception as exc:
        log.debug("SkillStore not available: %s", exc)
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


def _create_executor(project_root, state, ring2_path, reply_fn, memory_store=None, skill_store=None):
    """Best-effort task executor creation.  Returns None on any error."""
    try:
        from ring1.config import load_ring1_config
        from ring1.task_executor import create_executor, start_executor_thread

        r1_config = load_ring1_config(project_root)
        executor = create_executor(r1_config, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store)
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
    cooldown_sec = r0["evolution"].get("cooldown_sec", 1800)

    git = GitManager(ring2_path)
    git.init_repo()
    fitness = FitnessTracker(db_path)
    memory_store = _create_memory_store(db_path)
    skill_store = _create_skill_store(db_path)
    hb = HeartbeatMonitor(heartbeat_path, timeout_sec=timeout)
    notifier = _create_notifier(project_root)

    # Shared state for Telegram bot interaction.
    from ring1.telegram_bot import SentinelState
    state = SentinelState()
    state.memory_store = memory_store
    state.skill_store = skill_store
    bot = _create_bot(project_root, state, fitness, ring2_path)

    # Task executor for P0 user tasks.
    reply_fn = bot._send_reply if bot else (lambda text: None)
    executor = _create_executor(project_root, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store)

    # Commit watcher — auto-restart on new commits.
    commit_watcher = CommitWatcher(project_root, state.restart_event)
    threading.Thread(target=commit_watcher.run, name="commit-watcher", daemon=True).start()

    generation = 0
    last_good_hash: str | None = None
    last_crystallized_hash: str | None = None
    skill_cap = r0["evolution"].get("skill_max_count", 100)
    proc: subprocess.Popen | None = None

    # Initial snapshot of seed code.
    try:
        last_good_hash = git.snapshot(f"gen-{generation} seed")
    except subprocess.CalledProcessError:
        pass

    log.info("Sentinel online — heartbeat every %ds, timeout %ds, cooldown %ds", interval, timeout, cooldown_sec)

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

            # --- restart check (commit watcher sets this) ---
            if state.restart_event.is_set():
                log.info("New commit detected — restarting Protea")
                break

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

                # Record observation in memory (with output).
                output = _read_ring2_output(proc, max_lines=50)
                source = (ring2_path / "main.py").read_text()
                if memory_store:
                    memory_store.add(
                        generation, "observation",
                        f"Gen {generation} survived {elapsed:.0f}s (max {params.max_runtime_sec}s). "
                        f"Code: {len(source)} bytes.\n"
                        f"Output (last 50 lines):\n{output[-1000:] if output else '(no output)'}",
                    )

                # Crystallize skill (best-effort) — skip if source unchanged.
                if skill_store:
                    import hashlib
                    source_hash = hashlib.sha256(source.encode()).hexdigest()
                    if source_hash != last_crystallized_hash:
                        log.info("Crystallizing gen-%d (hash=%s…)", generation, source_hash[:12])
                        _try_crystallize(
                            project_root, skill_store, source, output,
                            generation, skill_cap=skill_cap,
                        )
                        last_crystallized_hash = source_hash
                    else:
                        log.debug("Skipping crystallization — source unchanged (hash=%s…)", source_hash[:12])

                # Evolve (best-effort) — skip if busy or cooling down.
                if not _should_evolve(state, cooldown_sec):
                    log.info("Skipping evolution (busy or cooldown)")
                    evolved = False
                else:
                    with state.lock:
                        directive = state.evolution_directive
                        state.evolution_directive = ""
                    if directive and memory_store:
                        memory_store.add(generation, "directive", directive)
                    crash_logs = []
                    if memory_store:
                        try:
                            crash_logs = memory_store.get_by_type("crash_log", limit=3)
                        except Exception:
                            pass
                    evolved = _try_evolve(
                        project_root, fitness, ring2_path,
                        generation, params, True, notifier,
                        directive=directive,
                        memory_store=memory_store,
                        skill_store=skill_store,
                        crash_logs=crash_logs,
                    )
                if evolved:
                    state.last_evolution_time = time.time()
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
            output = _read_ring2_output(proc)
            _stop_ring2(proc)

            failure_reason = _classify_failure(proc, output)
            log.warning("Failure reason: %s", failure_reason.splitlines()[0])

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

            # Record crash log and observation in memory.
            if memory_store:
                memory_store.add(
                    generation, "crash_log",
                    f"Gen {generation} died after {elapsed:.0f}s.\n"
                    f"Reason: {failure_reason}\n\n"
                    f"--- Last output ---\n{output[-2000:] if output else '(no output)'}",
                )

            # Evolve from the good base (best-effort) — skip if busy or cooling down.
            if not _should_evolve(state, cooldown_sec):
                log.info("Skipping evolution (busy or cooldown)")
                evolved = False
            else:
                with state.lock:
                    directive = state.evolution_directive
                    state.evolution_directive = ""
                if directive and memory_store:
                    memory_store.add(generation, "directive", directive)
                crash_logs = []
                if memory_store:
                    try:
                        crash_logs = memory_store.get_by_type("crash_log", limit=3)
                    except Exception:
                        pass
                evolved = _try_evolve(
                    project_root, fitness, ring2_path,
                    generation, params, False, notifier,
                    directive=directive,
                    memory_store=memory_store,
                    skill_store=skill_store,
                    crash_logs=crash_logs,
                )
            if evolved:
                state.last_evolution_time = time.time()
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
        commit_watcher.stop()
        if executor:
            executor.stop()
        if bot:
            bot.stop()
        _stop_ring2(proc)
        log.info("Sentinel offline")

    # Restart the entire process if triggered by CommitWatcher.
    if state.restart_event.is_set():
        log.info("Restarting via os.execv()")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    project_root = pathlib.Path(__file__).resolve().parent.parent
    run(project_root)
