"""Sentinel — Ring 0 main loop (pure stdlib).

Launches and supervises Ring 2.  On success (survived max_runtime_sec),
triggers Ring 1 evolution to mutate the code.  On failure, rolls back
to the last known-good commit, evolves from that base, and restarts.
"""

from __future__ import annotations

import logging
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import tomllib

from ring0.commit_watcher import CommitWatcher
from ring0.fitness import FitnessTracker, evaluate_output
from ring0.fitness_v2 import evaluate_output_v2, get_task_level_for_generation
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
    fh = open(log_file, "a")
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


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its descendants (children, grandchildren, etc.)."""
    try:
        parent = os.waitpid(pid, os.WNOHANG)  # noqa: F841 — just reap if zombie
    except ChildProcessError:
        pass
    # Walk /proc-style via sysctl on macOS or /proc on Linux.
    # Fallback: use ``pkill -P`` which is available on both.
    try:
        subprocess.run(
            ["pkill", "-TERM", "-P", str(pid)],
            timeout=3, capture_output=True,
        )
        time.sleep(0.5)
        subprocess.run(
            ["pkill", "-KILL", "-P", str(pid)],
            timeout=3, capture_output=True,
        )
    except Exception:
        pass


def _stop_ring2(proc: subprocess.Popen | None) -> None:
    """Terminate the Ring 2 process **and its entire child tree**."""
    if proc is None:
        return
    if proc.poll() is None:
        # First, kill children so they don't become orphans.
        _kill_process_tree(proc.pid)
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


def _should_evolve(state, cooldown_sec: int, fitness=None, plateau_window: int = 5, plateau_epsilon: float = 0.03, has_directive: bool = False) -> tuple[bool, bool]:
    """Check whether evolution should proceed.

    Returns (should_evolve, is_plateaued):
    - should_evolve: True if evolution should run.
    - is_plateaued: True if scores are stagnant (signals to LLM to try
      something fundamentally different).

    Adaptive evolution: when scores are plateaued AND no user directive
    is pending, skip the LLM call to save tokens.  A directive always
    forces evolution.
    """
    if state.p0_active.is_set():
        return False, False
    if state.p1_active.is_set():
        return False, False
    if time.time() - state.last_evolution_time < cooldown_sec:
        return False, False

    # Detect plateau.
    plateaued = False
    if fitness:
        try:
            plateaued = fitness.is_plateaued(window=plateau_window, epsilon=plateau_epsilon)
        except Exception:
            pass

    # Adaptive: skip evolution when plateaued unless a directive is pending.
    if plateaued and not has_directive:
        log.info("Scores plateaued — skipping evolution to save tokens (set a directive to force)")
        return False, True

    return True, plateaued


def _try_evolve(project_root, fitness, ring2_path, generation, params, survived, notifier, directive="", memory_store=None, skill_store=None, crash_logs=None, is_plateaued=False, gene_pool=None, user_profile_summary=""):
    """Best-effort evolution.  Returns True if new code was written."""
    try:
        from ring1.config import load_ring1_config
        from ring1.evolver import Evolver

        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            log.warning("LLM API key not configured — skipping evolution")
            return False

        # Compact context to save tokens: fewer memories.
        memories = memory_store.get_recent(3) if memory_store else []

        # Gather task history and skills for directed evolution.
        task_history = []
        if memory_store:
            try:
                task_history = memory_store.get_by_type("task", limit=5)
            except Exception:
                pass

        skills = []
        if skill_store:
            try:
                skills = skill_store.get_active(15)
            except Exception:
                pass

        # Get persistent error signatures from recent fitness history.
        persistent_errors = []
        try:
            persistent_errors = fitness.get_recent_error_signatures(limit=5)
        except Exception:
            pass

        # Get context-relevant genes for inheritance.
        genes = []
        if gene_pool:
            try:
                from ring0.gene_pool import GenePool as _GP
                context_parts = []
                # Current Ring 2 source — extract class/function names.
                try:
                    current_source = (ring2_path / "main.py").read_text()
                    context_parts.append(_GP.extract_summary(current_source))
                except OSError:
                    pass
                if directive:
                    context_parts.append(directive)
                for task in task_history:
                    context_parts.append(task.get("content", ""))
                for err in persistent_errors:
                    context_parts.append(err)
                context = " ".join(context_parts)
                genes = gene_pool.get_relevant(context, 3)
            except Exception:
                pass

        # Classify evolution intent.
        from ring0.evolution_intent import classify_intent

        evolution_intent = classify_intent(
            survived=survived,
            is_plateaued=is_plateaued,
            persistent_errors=persistent_errors,
            crash_logs=crash_logs or [],
            directive=directive,
        )
        log.info(
            "Evolution intent: %s (signals: %s)",
            evolution_intent["intent"],
            evolution_intent["signals"],
        )

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
            persistent_errors=persistent_errors,
            is_plateaued=is_plateaued,
            gene_pool=genes,
            evolution_intent=evolution_intent,
            user_profile_summary=user_profile_summary,
        )
        if result.success:
            log.info("Evolution succeeded: %s", result.reason)
            if result.metadata:
                log.info(
                    "Evolution metadata: intent=%s scope=%s lines_changed=%d",
                    result.metadata.get("intent"),
                    result.metadata.get("blast_radius", {}).get("scope"),
                    result.metadata.get("blast_radius", {}).get("lines_changed", 0),
                )
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


def _try_crystallize(project_root, skill_store, source_code, output, generation, skill_cap=100, registry_client=None):
    """Best-effort crystallization.  Returns action string or None."""
    try:
        from ring1.config import load_ring1_config
        from ring1.crystallizer import Crystallizer

        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            log.warning("LLM API key not configured — skipping crystallization")
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

        # Auto-publish to registry on successful crystallization.
        if result.action in ("create", "update") and result.skill_name and registry_client:
            try:
                skill_data = skill_store.get_by_name(result.skill_name)
                if skill_data:
                    registry_client.publish(
                        name=skill_data["name"],
                        description=skill_data.get("description", ""),
                        prompt_template=skill_data.get("prompt_template", ""),
                        parameters=skill_data.get("parameters"),
                        tags=skill_data.get("tags"),
                        source_code=skill_data.get("source_code", ""),
                    )
                    log.info("Published skill %r to registry", result.skill_name)
            except Exception as pub_exc:
                log.debug("Registry publish failed (non-fatal): %s", pub_exc)

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


def _create_gene_pool(db_path):
    """Best-effort GenePool creation.  Returns None on any error."""
    try:
        from ring0.gene_pool import GenePool
        return GenePool(db_path)
    except Exception as exc:
        log.debug("GenePool not available: %s", exc)
        return None


def _create_task_store(db_path):
    """Best-effort TaskStore creation.  Returns None on any error."""
    try:
        from ring0.task_store import TaskStore
        return TaskStore(db_path)
    except Exception as exc:
        log.debug("TaskStore not available: %s", exc)
        return None


def _create_skill_runner():
    """Best-effort SkillRunner creation.  Returns None on any error."""
    try:
        from ring1.skill_runner import SkillRunner
        return SkillRunner()
    except Exception as exc:
        log.debug("SkillRunner not available: %s", exc)
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


def _create_registry_client(project_root, cfg):
    """Best-effort RegistryClient creation.  Returns None on any error."""
    try:
        from ring1.registry_client import RegistryClient
        reg_cfg = cfg.get("registry", {})
        if not reg_cfg.get("enabled", False):
            return None
        url = reg_cfg.get("url", "https://protea-hub-production.up.railway.app")
        import socket
        node_id = reg_cfg.get("node_id", "default")
        if node_id == "default":
            node_id = socket.gethostname()
        client = RegistryClient(url, node_id)
        log.info("RegistryClient created (url=%s, node_id=%s)", url, node_id)
        return client
    except Exception as exc:
        log.debug("RegistryClient not available: %s", exc)
        return None


def _create_portal(project_root, cfg, skill_store, skill_runner):
    """Best-effort Skill Portal creation.  Returns None on any error."""
    try:
        from ring1.skill_portal import create_portal, start_portal_thread

        portal = create_portal(skill_store, skill_runner, project_root, cfg)
        if portal:
            start_portal_thread(portal)
            log.info("Skill Portal started")
        return portal
    except Exception as exc:
        log.debug("Skill Portal not available: %s", exc)
        return None


def _create_user_profiler(db_path):
    """Best-effort UserProfiler creation.  Returns None on any error."""
    try:
        from ring0.user_profile import UserProfiler
        return UserProfiler(db_path)
    except Exception as exc:
        log.debug("UserProfiler not available: %s", exc)
        return None


def _create_embedding_provider(cfg):
    """Best-effort EmbeddingProvider creation.  Returns None on any error."""
    try:
        from ring1.embeddings import create_embedding_provider
        provider = create_embedding_provider(cfg)
        # Return None if it's a NoOp (saves passing it around)
        from ring1.embeddings import NoOpEmbedding
        if isinstance(provider, NoOpEmbedding):
            return None
        return provider
    except Exception as exc:
        log.debug("EmbeddingProvider not available: %s", exc)
        return None


def _create_memory_curator(project_root):
    """Best-effort MemoryCurator creation.  Returns None on any error."""
    try:
        from ring1.config import load_ring1_config
        from ring1.memory_curator import MemoryCurator

        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            return None
        client = r1_config.get_llm_client()
        return MemoryCurator(client)
    except Exception as exc:
        log.debug("MemoryCurator not available: %s", exc)
        return None


def _create_dashboard(project_root, cfg, **data_sources):
    """Best-effort Dashboard creation.  Returns None on any error."""
    try:
        from ring1.dashboard import create_dashboard, start_dashboard_thread

        dashboard = create_dashboard(project_root, cfg, **data_sources)
        if dashboard:
            start_dashboard_thread(dashboard)
            log.info("Dashboard started")
        return dashboard
    except Exception as exc:
        log.debug("Dashboard not available: %s", exc)
        return None


def _create_executor(project_root, state, ring2_path, reply_fn, memory_store=None, skill_store=None, skill_runner=None, task_store=None, registry_client=None, user_profiler=None, embedding_provider=None):
    """Best-effort task executor creation.  Returns None on any error."""
    try:
        from ring1.config import load_ring1_config
        from ring1.task_executor import create_executor, start_executor_thread

        r1_config = load_ring1_config(project_root)
        executor = create_executor(r1_config, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store, skill_runner=skill_runner, task_store=task_store, registry_client=registry_client, user_profiler=user_profiler, embedding_provider=embedding_provider)
        if executor:
            thread = start_executor_thread(executor)
            state.executor_thread = thread
            log.info("Task executor started")
        return executor
    except Exception as exc:
        log.debug("Task executor not available: %s", exc)
        return None


def run(project_root: pathlib.Path) -> None:
    """Sentinel main loop — run until interrupted."""
    # Convert SIGTERM into KeyboardInterrupt so the finally block runs,
    # ensuring Ring 2 subprocess, skill runners, and the Telegram bot
    # are stopped cleanly.
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    cfg = _load_config(project_root)
    r0 = cfg["ring0"]

    ring2_path = project_root / r0["git"]["ring2_path"]
    heartbeat_path = ring2_path / ".heartbeat"
    db_path = project_root / r0["fitness"]["db_path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)

    interval = r0["heartbeat_interval_sec"]
    timeout = r0["heartbeat_timeout_sec"]
    seed = r0["evolution"]["seed"]
    cooldown_sec = r0["evolution"].get("cooldown_sec", 900)
    plateau_window = r0["evolution"].get("plateau_window", 5)
    plateau_epsilon = r0["evolution"].get("plateau_epsilon", 0.03)

    git = GitManager(ring2_path)
    git.init_repo()
    fitness = FitnessTracker(db_path)
    memory_store = _create_memory_store(db_path)
    skill_store = _create_skill_store(db_path)
    gene_pool = _create_gene_pool(db_path)
    task_store = _create_task_store(db_path)
    user_profiler = _create_user_profiler(db_path)
    embedding_provider = _create_embedding_provider(cfg)
    memory_curator = _create_memory_curator(project_root)

    # Initialize TaskGenerator for progressive difficulty
    from ring1.task_generator import TaskGenerator
    task_generator = TaskGenerator(
        base_level=1,
        adjustment_window=10,
        upgrade_threshold=0.90,
        downgrade_threshold=0.65,
    )
    
    # Initialize AutoCrystallizer for automatic skill extraction
    from ring1.auto_crystallizer import AutoCrystallizer
    skills_dir = project_root / "skills"
    auto_crystallizer = AutoCrystallizer(
        skills_dir=skills_dir,
        min_stability=0.80,
        min_score=0.85,
        min_occurrences=5,
    )
    hb = HeartbeatMonitor(heartbeat_path, timeout_sec=timeout)
    notifier = _create_notifier(project_root)

    # Shared state for Telegram bot interaction.
    from ring1.telegram_bot import SentinelState
    state = SentinelState()
    state.notifier = notifier  # bot uses this for auto-detect propagation
    skill_runner = _create_skill_runner()
    state.memory_store = memory_store
    state.skill_store = skill_store
    state.skill_runner = skill_runner
    state.task_store = task_store
    bot = _create_bot(project_root, state, fitness, ring2_path)

    # Registry client — publish skills to remote registry + hub fallback.
    registry_client = _create_registry_client(project_root, cfg)
    state.registry_client = registry_client

    # Evict stale hub skills on startup.
    if skill_store:
        evicted = skill_store.evict_stale()
        if evicted:
            log.info("Evicted %d stale hub skills", evicted)

    # Backfill gene pool from existing skills (one-time).
    if gene_pool and skill_store:
        try:
            backfilled = gene_pool.backfill(skill_store)
            if backfilled:
                log.info("Gene pool backfilled %d genes from skills", backfilled)
        except Exception as exc:
            log.debug("Gene pool backfill failed (non-fatal): %s", exc)

    # Backfill gene pool from git history.
    if gene_pool and fitness and gene_pool.count() < gene_pool.max_size:
        try:
            backfilled = gene_pool.backfill_from_git(ring2_path, fitness)
            if backfilled:
                log.info("Gene pool backfilled %d genes from git history", backfilled)
        except Exception as exc:
            log.debug("Gene pool git backfill failed (non-fatal): %s", exc)

    # Task executor for P0 user tasks.
    reply_fn = bot._send_reply if bot else (lambda text: None)
    executor = _create_executor(project_root, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store, skill_runner=skill_runner, task_store=task_store, registry_client=registry_client, user_profiler=user_profiler, embedding_provider=embedding_provider)
    # Expose subagent_manager on state for /background command.
    state.subagent_manager = getattr(executor, "subagent_manager", None) if executor else None

    # Skill Portal — unified web dashboard.
    portal = _create_portal(project_root, cfg, skill_store, skill_runner)

    # Dashboard — system state visualization.
    dashboard = _create_dashboard(
        project_root, cfg,
        memory_store=memory_store,
        skill_store=skill_store,
        fitness_tracker=fitness,
        user_profiler=user_profiler,
        gene_pool=gene_pool,
        task_store=task_store,
        state=state,
    )

    # Commit watcher — auto-restart on new commits.
    commit_watcher = CommitWatcher(project_root, state.restart_event)
    threading.Thread(target=commit_watcher.run, name="commit-watcher", daemon=True).start()

    # Restore generation counter from fitness database.
    restored_gen = fitness.get_max_generation()
    if restored_gen >= 0:
        generation = restored_gen + 1
        log.info("Resumed from generation %d (last recorded: %d)", generation, restored_gen)
    else:
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

                # Read output and score (with novelty from recent fingerprints).
                output = _read_ring2_output(proc, max_lines=200)
                output_lines = output.splitlines() if output else []
                recent_fps = []
                try:
                    recent_fps = fitness.get_recent_fingerprints(limit=10)
                except Exception:
                    pass
                score, detail = evaluate_output(
                    output_lines, survived=True,
                    elapsed=elapsed, max_runtime=params.max_runtime_sec,
                    recent_fingerprints=recent_fps,
                )

                # Record success.
                commit_hash = last_good_hash or "unknown"
                fitness.record(
                    generation=generation,
                    commit_hash=commit_hash,
                    score=score,
                    runtime_sec=elapsed,
                    survived=True,
                    detail=detail,
                )
                log.info("Fitness score gen-%d: %.4f  detail=%s", generation, score, detail)

                with state.lock:
                    state.last_score = score
                    state.last_survived = True

                # Snapshot the surviving code.
                try:
                    last_good_hash = git.snapshot(f"gen-{generation} survived")
                except subprocess.CalledProcessError:
                    pass
                source = (ring2_path / "main.py").read_text()
                if memory_store:
                    memory_store.add(
                        generation, "observation",
                        f"Gen {generation} survived {elapsed:.0f}s (max {params.max_runtime_sec}s). "
                        f"Code: {len(source)} bytes.\n"
                        f"Output (last 50 lines):\n{output[-1000:] if output else '(no output)'}",
                    )

                # Store gene in pool (best-effort).
                if gene_pool:
                    try:
                        gene_pool.add(generation, score, source)
                    except Exception as exc:
                        log.debug("Gene pool add failed (non-fatal): %s", exc)

                # Crystallize skill (best-effort) — skip if source unchanged.
                if skill_store:
                    import hashlib
                    source_hash = hashlib.sha256(source.encode()).hexdigest()
                    if source_hash != last_crystallized_hash:
                        log.info("Crystallizing gen-%d (hash=%s…)", generation, source_hash[:12])
                        _try_crystallize(
                            project_root, skill_store, source, output,
                            generation, skill_cap=skill_cap,
                            registry_client=registry_client,
                        )
                        last_crystallized_hash = source_hash
                    else:
                        log.debug("Skipping crystallization — source unchanged (hash=%s…)", source_hash[:12])

                # Evolve (best-effort) — skip if busy, cooling down, or plateaued.
                with state.lock:
                    pending_directive = state.evolution_directive
                should_evo, plateaued = _should_evolve(
                    state, cooldown_sec, fitness=fitness,
                    plateau_window=plateau_window,
                    plateau_epsilon=plateau_epsilon,
                    has_directive=bool(pending_directive),
                )
                if not should_evo:
                    if not plateaued:
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
                    # Get user profile summary for evolution.
                    profile_summary = ""
                    if user_profiler:
                        try:
                            profile_summary = user_profiler.get_profile_summary()
                        except Exception:
                            pass
                    evolved = _try_evolve(
                        project_root, fitness, ring2_path,
                        generation, params, True, notifier,
                        directive=directive,
                        memory_store=memory_store,
                        skill_store=skill_store,
                        crash_logs=crash_logs,
                        is_plateaued=plateaued,
                        gene_pool=gene_pool,
                        user_profile_summary=profile_summary,
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
                        generation, score, True, last_good_hash or "unknown",
                    )

                # Next generation.
                generation += 1

                # Periodic maintenance: compact memory + decay profile (every 10 generations).
                if generation % 10 == 0:
                    if memory_store:
                        try:
                            compact_result = memory_store.compact(generation, curator=memory_curator)
                            log.info("Memory compaction: %s", compact_result)
                        except Exception:
                            log.debug("Memory compaction failed (non-fatal)", exc_info=True)
                    if user_profiler:
                        try:
                            removed = user_profiler.apply_decay()
                            if removed:
                                log.info("Profile decay: removed %d stale topics", removed)
                        except Exception:
                            log.debug("Profile decay failed (non-fatal)", exc_info=True)

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
            output = _read_ring2_output(proc, max_lines=200)
            _stop_ring2(proc)

            failure_reason = _classify_failure(proc, output)
            log.warning("Failure reason: %s", failure_reason)

            output_lines = output.splitlines() if output else []
            score, detail = evaluate_output(
                output_lines, survived=False,
                elapsed=elapsed, max_runtime=params.max_runtime_sec,
            )
            log.info("Fitness score gen-%d: %.4f  detail=%s", generation, score, detail)
            commit_hash = last_good_hash or "unknown"
            fitness.record(
                generation=generation,
                commit_hash=commit_hash,
                score=score,
                runtime_sec=elapsed,
                survived=False,
                detail=detail,
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
            # Failures always trigger evolution (no plateau skip) to fix the issue.
            with state.lock:
                pending_directive = state.evolution_directive
            should_evo, plateaued = _should_evolve(
                state, cooldown_sec, fitness=fitness,
                plateau_window=plateau_window,
                plateau_epsilon=plateau_epsilon,
                has_directive=True,  # failures always force evolution
            )
            if not should_evo:
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
                # Get user profile summary for evolution.
                profile_summary = ""
                if user_profiler:
                    try:
                        profile_summary = user_profiler.get_profile_summary()
                    except Exception:
                        pass
                evolved = _try_evolve(
                    project_root, fitness, ring2_path,
                    generation, params, False, notifier,
                    directive=directive,
                    memory_store=memory_store,
                    skill_store=skill_store,
                    crash_logs=crash_logs,
                    is_plateaued=False,  # failure path — focus on fixing, not novelty
                    gene_pool=gene_pool,
                    user_profile_summary=profile_summary,
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
        if dashboard:
            dashboard.stop()
        if portal:
            portal.stop()
        if skill_runner and skill_runner.is_running():
            skill_runner.stop()
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
