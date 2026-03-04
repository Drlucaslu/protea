"""Sentinel — Ring 0 main loop (pure stdlib).

Launches and supervises Ring 2.  On success (survived max_runtime_sec),
records fitness and advances generation.  On failure, rolls back to the
last known-good commit and restarts.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import pathlib
import signal
import random
import subprocess
import sys
import threading
import time
import tomllib

from ring0.commit_watcher import CommitWatcher
from ring0.fitness import FitnessTracker, evaluate_output
from ring0.git_manager import GitManager
from ring0.heartbeat import HeartbeatMonitor
from ring0.memory import MemoryStore
from ring0.output_filter import filter_ring2_output
from ring0.parameter_seed import generate_params, params_to_dict
from ring0.resource_monitor import check_resources
from typing import NamedTuple

log = logging.getLogger("protea.sentinel")


class Ring0Config(NamedTuple):
    """Typed configuration extracted from config.toml [ring0] section."""
    ring2_path: str
    db_path: str
    heartbeat_interval_sec: int
    heartbeat_timeout_sec: int
    max_cpu_percent: float
    max_memory_percent: float
    max_disk_percent: float
    # Reflection config
    reflection_idle_threshold_sec: int
    reflection_auto_confidence: float
    reflection_cooldown_sec: int
    reflection_min_tasks: int

    @classmethod
    def from_dict(cls, r0: dict) -> "Ring0Config":
        """Parse from the raw [ring0] config dict."""
        refl = r0.get("reflection", {})
        return cls(
            ring2_path=r0["git"]["ring2_path"],
            db_path=r0["fitness"]["db_path"],
            heartbeat_interval_sec=r0["heartbeat_interval_sec"],
            heartbeat_timeout_sec=r0["heartbeat_timeout_sec"],
            max_cpu_percent=r0["max_cpu_percent"],
            max_memory_percent=r0["max_memory_percent"],
            max_disk_percent=r0["max_disk_percent"],
            reflection_idle_threshold_sec=refl.get("idle_threshold_sec", 7200),
            reflection_auto_confidence=refl.get("auto_confidence", 0.8),
            reflection_cooldown_sec=refl.get("cooldown_sec", 1800),
            reflection_min_tasks=refl.get("min_tasks_before_reflect", 3),
        )


def _load_config(project_root: pathlib.Path) -> dict:
    cfg_path = project_root / "config" / "config.toml"
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)


def _start_ring2(ring2_path: pathlib.Path, heartbeat_path: pathlib.Path) -> subprocess.Popen:
    """Launch the Ring 2 process and return its Popen handle."""
    log_file = ring2_path / ".output.log"
    # Truncate log to last 200 lines before new generation.
    # Use tail instead of read_text() to avoid OOM on huge log files.
    if log_file.exists():
        try:
            tmp = log_file.with_suffix(".log.tmp")
            subprocess.run(
                ["tail", "-200", str(log_file)],
                stdout=open(tmp, "w"), stderr=subprocess.DEVNULL,
            )
            tmp.replace(log_file)
        except Exception:
            pass
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
    try:
        result = subprocess.run(
            ["tail", f"-{max_lines}", str(log_path)],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout
    except Exception:
        return ""


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
        lines = output.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Traceback"):
                return "\n".join(lines[i:])
        return f"exit code {rc}"
    return "clean exit but heartbeat lost"


def _best_effort(label, factory):
    """Run *factory*; return its result or None on any error."""
    try:
        return factory()
    except Exception as exc:
        log.debug("%s not available: %s", label, exc)
        return None


def _compute_skill_hit_ratio(memory_store, limit: int = 50) -> dict:
    """Compute skill hit ratio from recent task metadata.

    Returns {"total": int, "skill": int, "ratio": float, "top_skills": dict[str,int]}.
    """
    entries = []
    for t in ("task",):
        try:
            entries.extend(memory_store.get_by_type(t, limit))
        except Exception:
            pass
    if not entries:
        return {"total": 0, "skill": 0, "ratio": 0.0, "top_skills": {}}

    total = len(entries)
    skill_count = 0
    skill_freq: dict[str, int] = {}
    for e in entries:
        used = e.get("metadata", {}).get("skills_used", [])
        if used:
            skill_count += 1
            for s in used:
                skill_freq[s] = skill_freq.get(s, 0) + 1

    top = dict(sorted(skill_freq.items(), key=lambda x: -x[1])[:5])
    ratio = skill_count / total if total else 0.0
    return {"total": total, "skill": skill_count, "ratio": ratio, "top_skills": top}


def _create_notifier(project_root):
    """Best-effort Telegram notifier creation."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.telegram import create_notifier
        return create_notifier(load_ring1_config(project_root))
    return _best_effort("Telegram notifier", _factory)


def _create_bot(project_root, state, fitness, ring2_path):
    """Best-effort Telegram bot creation."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.telegram_bot import create_bot, start_bot_thread
        r1_config = load_ring1_config(project_root)
        bot = create_bot(r1_config, state, fitness, ring2_path)
        if bot:
            start_bot_thread(bot)
            log.info("Telegram bot started")
        return bot
    return _best_effort("Telegram bot", _factory)


def _create_registry_client(project_root, cfg):
    """Best-effort RegistryClient creation."""
    def _factory():
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
    return _best_effort("RegistryClient", _factory)


def _create_task_syncer(scheduled_store, registry_client, user_profiler, cfg):
    """Best-effort TaskSyncer creation."""
    if not scheduled_store or not registry_client:
        return None
    def _factory():
        from ring1.task_sync import TaskSyncer
        sync_cfg = cfg.get("ring1", {}).get("task_sync", cfg.get("ring1", {}).get("skill_sync", {}))
        if not sync_cfg.get("enabled", True):
            return None
        max_discover = sync_cfg.get("max_discover_per_sync", 5)

        syncer = TaskSyncer(
            scheduled_store=scheduled_store,
            registry_client=registry_client,
            user_profiler=user_profiler,
            max_discover=max_discover,
        )
        log.info("TaskSyncer created (max_discover=%d)", max_discover)
        return syncer
    return _best_effort("TaskSyncer", _factory)


def _create_portal(project_root, cfg, skill_store, skill_runner):
    """Best-effort Skill Portal creation."""
    def _factory():
        from ring1.skill_portal import create_portal, start_portal_thread
        portal = create_portal(skill_store, skill_runner, project_root, cfg)
        if portal:
            start_portal_thread(portal)
            log.info("Skill Portal started")
        return portal
    return _best_effort("Skill Portal", _factory)


def _create_task_api(project_root, cfg, executor):
    """Best-effort Task API creation."""
    def _factory():
        from ring1.task_api import create_task_api
        registry = getattr(executor, "registry", None) if executor else None
        api = create_task_api(cfg, registry, project_root)
        if api:
            api.start()
            log.info("Task API started")
        return api
    return _best_effort("Task API", _factory)


def _create_matrix_bot(project_root, state):
    """Best-effort Matrix bot creation."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.matrix_bot import MatrixBot
        r1_config = load_ring1_config(project_root)
        if not r1_config.matrix_enabled or not r1_config.matrix_homeserver:
            return None
        if not r1_config.matrix_access_token:
            log.warning("Matrix bot: enabled but MATRIX_ACCESS_TOKEN missing — disabled")
            return None
        bot = MatrixBot(
            homeserver=r1_config.matrix_homeserver,
            access_token=r1_config.matrix_access_token,
            room_id=r1_config.matrix_room_id,
            state=state,
        )
        threading.Thread(target=bot.run, name="matrix-bot", daemon=True).start()
        log.info("Matrix bot started")
        return bot
    return _best_effort("Matrix bot", _factory)


def _create_embedding_provider(cfg):
    """Best-effort EmbeddingProvider creation."""
    def _factory():
        from ring1.embeddings import create_embedding_provider
        provider = create_embedding_provider(cfg)
        from ring1.embeddings import NoOpEmbedding
        if isinstance(provider, NoOpEmbedding):
            return None
        return provider
    return _best_effort("EmbeddingProvider", _factory)


def _create_memory_curator(project_root):
    """Best-effort MemoryCurator creation."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.memory_curator import MemoryCurator
        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            return None
        client = r1_config.get_llm_client()
        return MemoryCurator(client)
    return _best_effort("MemoryCurator", _factory)


def _create_dashboard(project_root, cfg, **data_sources):
    """Best-effort Dashboard creation."""
    def _factory():
        from ring1.dashboard import create_dashboard, start_dashboard_thread
        dashboard = create_dashboard(project_root, cfg, **data_sources)
        if dashboard:
            start_dashboard_thread(dashboard)
            log.info("Dashboard started")
        return dashboard
    return _best_effort("Dashboard", _factory)


def _create_executor(project_root, state, ring2_path, reply_fn, memory_store=None, skill_store=None, skill_runner=None, task_store=None, user_profiler=None, embedding_provider=None, scheduled_store=None, send_file_fn=None, preference_store=None, reply_fn_factory=None):
    """Best-effort task executor creation."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.task_executor import create_executor, start_executor_thread
        r1_config = load_ring1_config(project_root)
        executor = create_executor(r1_config, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store, skill_runner=skill_runner, task_store=task_store, user_profiler=user_profiler, embedding_provider=embedding_provider, scheduled_store=scheduled_store, send_file_fn=send_file_fn, preference_store=preference_store, reply_fn_factory=reply_fn_factory)
        if executor:
            thread = start_executor_thread(executor)
            state.executor_thread = thread
            log.info("Task executor started")
        return executor
    return _best_effort("Task executor", _factory)


def _acquire_lock(project_root: pathlib.Path):
    """Acquire exclusive PID lock. Returns file handle, or None if locked."""
    lock_path = project_root / "data" / "sentinel.pid"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def run(project_root: pathlib.Path) -> None:
    """Sentinel main loop — run until interrupted."""
    lock_fh = _acquire_lock(project_root)
    if lock_fh is None:
        log.error("Another sentinel is already running (lock: %s/data/sentinel.pid)", project_root)
        return

    # Convert SIGTERM into KeyboardInterrupt so the finally block runs.
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    cfg = _load_config(project_root)
    r0 = Ring0Config.from_dict(cfg["ring0"])

    ring2_path = project_root / r0.ring2_path
    heartbeat_path = ring2_path / ".heartbeat"
    db_path = project_root / r0.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    interval = r0.heartbeat_interval_sec
    timeout = r0.heartbeat_timeout_sec

    git = GitManager(ring2_path)
    git.init_repo()
    fitness = FitnessTracker(db_path)
    memory_store = _best_effort("MemoryStore", lambda: MemoryStore(db_path))
    skill_store = _best_effort("SkillStore", lambda: __import__("ring0.skill_store", fromlist=["SkillStore"]).SkillStore(db_path))
    embedding_provider = _create_embedding_provider(cfg)
    task_store = _best_effort("TaskStore", lambda: __import__("ring0.task_store", fromlist=["TaskStore"]).TaskStore(db_path))
    scheduled_store = _best_effort("ScheduledTaskStore", lambda: __import__("ring0.scheduled_task_store", fromlist=["ScheduledTaskStore"]).ScheduledTaskStore(db_path))
    user_profiler = _best_effort("UserProfiler", lambda: __import__("ring0.user_profile", fromlist=["UserProfiler"]).UserProfiler(db_path))
    preference_store = _best_effort("PreferenceStore", lambda: __import__("ring0.preference_store", fromlist=["PreferenceStore"]).PreferenceStore(db_path, cfg.get("ring1", {}).get("user_profile", {})))
    output_queue = _best_effort("OutputQueue", lambda: __import__("ring0.output_queue", fromlist=["OutputQueue"]).OutputQueue(db_path))
    memory_curator = _create_memory_curator(project_root)

    # Capability skill sandbox — venv manager.
    venv_manager = None
    sandbox_cfg = cfg.get("ring1", {}).get("skill_sandbox", {})
    if sandbox_cfg.get("enabled", True):
        try:
            from ring1.skill_sandbox import VenvManager
            base_dir = project_root / sandbox_cfg.get("base_dir", "data/skill_envs")
            max_envs = sandbox_cfg.get("max_envs", 10)
            venv_manager = VenvManager(base_dir, max_envs)
            log.info("VenvManager created (base_dir=%s, max_envs=%d)", base_dir, max_envs)
        except Exception as exc:
            log.debug("VenvManager not available: %s", exc)

    # Load soul profile (centralized identity/preferences).
    _soul_text = ""
    try:
        from ring1.soul import load as _soul_load, get as _soul_get
        _soul_load(project_root)
        _soul_text = _soul_get()
    except Exception as exc:
        log.debug("Soul profile not loaded: %s", exc)

    # Sync soul rules to preference store.
    if preference_store and _soul_text:
        try:
            from ring1.soul import get_rules
            synced = preference_store.sync_soul_rules(get_rules())
            if synced:
                log.info("Soul rules synced to preference store (%d rules)", synced)
        except Exception as exc:
            log.debug("Soul sync failed: %s", exc)

    hb = HeartbeatMonitor(heartbeat_path, timeout_sec=timeout)
    notifier = _create_notifier(project_root)

    # Shared state for Telegram bot interaction.
    from ring1.telegram_bot import SentinelState
    state = SentinelState()
    state.notifier = notifier
    skill_runner = _best_effort("SkillRunner", lambda: __import__("ring1.skill_runner", fromlist=["SkillRunner"]).SkillRunner(venv_manager=venv_manager))
    state.memory_store = memory_store
    state.skill_store = skill_store
    state.skill_runner = skill_runner
    state.task_store = task_store
    state.scheduled_store = scheduled_store
    state._preference_store = preference_store
    state.output_queue = output_queue
    state._nudge_context_path = project_root / "data" / "nudge_context.json"
    state._load_nudge_context()
    bot = _create_bot(project_root, state, fitness, ring2_path)
    matrix_bot = _create_matrix_bot(project_root, state)

    # Registry client — publish skills to remote registry + hub fallback.
    registry_client = _create_registry_client(project_root, cfg)
    state.registry_client = registry_client

    # Evict stale hub skills and clean up unused evolved skills on startup.
    if skill_store:
        evicted = skill_store.evict_stale()
        if evicted:
            log.info("Evicted %d stale hub skills", evicted)
        cleaned = skill_store.cleanup_unused()
        if cleaned:
            log.info("Deactivated %d unused evolved skills", cleaned)

    # Task syncer — periodic publish + discover (task templates).
    task_syncer = _create_task_syncer(
        scheduled_store, registry_client, user_profiler, cfg,
    )
    sync_interval = cfg.get("ring1", {}).get("task_sync", cfg.get("ring1", {}).get("skill_sync", {})).get("interval_sec", 7200)

    # Reclassify 'general' topics and preferences to specific categories.
    if user_profiler:
        try:
            reclass = user_profiler.reclassify_general()
            if reclass["total"]:
                log.info("Reclassified %d 'general' topics: %s", reclass["total"], reclass)
        except Exception as exc:
            log.debug("Profile reclassify failed (non-fatal): %s", exc)
    if preference_store and user_profiler:
        try:
            cats = user_profiler.get_category_distribution()
            top_cat = next(iter(cats), "lifestyle")
            pref_reclass = preference_store.reclassify_general(fallback_category=top_cat)
            if pref_reclass["moments"] or pref_reclass["preferences"]:
                log.info("Preference reclassified: %s", pref_reclass)
        except Exception as exc:
            log.debug("Preference reclassify failed (non-fatal): %s", exc)

    # Task executor for P0 user tasks.
    reply_fn = bot._send_reply if bot else lambda text: None
    send_file_fn = bot._send_document if bot else None
    reply_fn_factory = bot.make_reply_fn if bot else None
    executor = _create_executor(project_root, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store, skill_runner=skill_runner, task_store=task_store, user_profiler=user_profiler, embedding_provider=embedding_provider, scheduled_store=scheduled_store, send_file_fn=send_file_fn, preference_store=preference_store, reply_fn_factory=reply_fn_factory)
    # Feedback prompt after task completion (not on intermediate messages).
    if executor and bot:
        executor.feedback_fn = bot.send_feedback_prompt
    # Expose subagent_manager on state for /background command.
    state.subagent_manager = getattr(executor, "subagent_manager", None) if executor else None

    # Nudge engine — contextual suggestions after tasks + proactive engagement.
    nudge_cfg = cfg.get("ring1", {}).get("nudge", {})
    if nudge_cfg.get("enabled", False):
        def _create_nudge():
            from ring1.nudge import NudgeEngine
            from ring1.config import load_ring1_config
            r1_config = load_ring1_config(project_root)
            if not r1_config.has_llm_config():
                return None
            client = r1_config.get_llm_client()
            return NudgeEngine(
                llm_client=client,
                memory_store=memory_store,
                scheduled_store=scheduled_store,
                user_profiler=user_profiler,
                preference_store=preference_store,
                config=nudge_cfg,
            )
        nudge_engine = _best_effort("NudgeEngine", _create_nudge)
        if nudge_engine and executor:
            executor.nudge_fn = nudge_engine.post_task_nudge
        if nudge_engine and bot:
            bot._nudge_engine = nudge_engine
            bot._nudge_interval = nudge_cfg.get("interval_sec", 600)
            bot._last_nudge_proactive = 0.0

    # Proactive loop — morning briefings, evening summaries, periodic checks.
    proactive_loop = None
    proactive_cfg = cfg.get("ring1", {}).get("proactive", {})
    if proactive_cfg.get("enabled", False):
        def _create_proactive():
            from ring1.config import load_ring1_config
            from ring1.proactive_loop import ProactiveLoop
            r1_config = load_ring1_config(project_root)
            if not r1_config.has_llm_config():
                return None
            client = r1_config.get_llm_client()
            return ProactiveLoop(
                llm_client=client,
                memory_store=memory_store,
                preference_store=preference_store,
                user_profiler=user_profiler,
                notifier=notifier,
                config=proactive_cfg,
                project_root=project_root,
                state=state,
            )
        proactive_loop = _best_effort("ProactiveLoop", _create_proactive)
        if proactive_loop:
            log.info("ProactiveLoop enabled (morning=%d, evening=%d)",
                     proactive_cfg.get("morning_hour", 9),
                     proactive_cfg.get("evening_hour", 21))

    # Task API — HTTP endpoint for external bot delegation.
    task_api = _create_task_api(project_root, cfg, executor)

    # Skill Portal — unified web dashboard.
    portal = _create_portal(project_root, cfg, skill_store, skill_runner)

    # Dashboard — system state visualization.
    dashboard = _create_dashboard(
        project_root, cfg,
        memory_store=memory_store,
        skill_store=skill_store,
        fitness_tracker=fitness,
        user_profiler=user_profiler,
        task_store=task_store,
        scheduled_store=scheduled_store,
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
    # Jitter: initial delay of 0–50% of interval so nodes don't all sync at once.
    last_skill_sync_time: float = time.time() - sync_interval + random.uniform(0, sync_interval * 0.5)
    last_consolidation_date: str = ""  # YYYY-MM-DD — nightly consolidation
    proc: subprocess.Popen | None = None

    # Initial snapshot of seed code.
    try:
        last_good_hash = git.snapshot(f"gen-{generation} seed")
    except subprocess.CalledProcessError:
        pass

    log.info("Sentinel online — heartbeat every %ds, timeout %ds", interval, timeout)

    # Notify Telegram that sentinel is online
    if notifier:
        notifier.notify_sentinel_online(generation)

    # Trigger initial soul onboarding check (after bot is ready).
    if proactive_loop:
        try:
            proactive_loop._check_onboarding()
        except Exception:
            log.debug("Initial onboarding check failed", exc_info=True)

    # Reflection system — replaces evolution.
    reflector = None
    last_reflection_time: float = 0.0
    try:
        from ring1.reflector import Reflector
        from ring1.config import load_ring1_config
        r1_config = load_ring1_config(project_root)
        if r1_config.has_llm_config():
            reflector = Reflector(
                config=r0,
                fitness_tracker=fitness,
                memory_store=memory_store,
                skill_store=skill_store,
                notifier=notifier,
                auto_confidence=r0.reflection_auto_confidence,
            )
            log.info("Reflector initialized (auto_confidence=%.2f, cooldown=%ds)",
                     r0.reflection_auto_confidence, r0.reflection_cooldown_sec)
    except Exception as exc:
        log.debug("Reflector not available: %s", exc)

    # Pass reflector to executor for task context augmentation.
    if reflector and executor:
        executor.reflector = reflector

    try:
        params = generate_params(generation, 42)
        proc = _start_ring2(ring2_path, heartbeat_path)
        start_time = time.time()
        hb.wait_for_heartbeat(startup_timeout=timeout)

        while True:
            state.p0_event.wait(timeout=interval)
            state.p0_event.clear()

            # --- proactive loop check ---
            if proactive_loop:
                try:
                    proactive_loop.check_and_send()
                except Exception:
                    log.debug("Proactive check failed (non-fatal)", exc_info=True)

            # --- resource check ---
            ok, msg = check_resources(
                r0.max_cpu_percent,
                r0.max_memory_percent,
                r0.max_disk_percent,
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
                if state.p0_active.is_set():
                    log.info("New commit detected — waiting for active task (max 30s)")
                    for _ in range(15):  # 15 * 2s = 30s
                        if not state.p0_active.is_set():
                            break
                        time.sleep(2)
                log.info("New commit detected — restarting Protea")
                break

            # --- scheduled task check ---
            if scheduled_store:
                try:
                    due = scheduled_store.get_due(time.time())
                    for sched in due:
                        from ring1.telegram_bot import Task
                        task = Task(text=sched["task_text"], chat_id=sched["chat_id"],
                                    exec_mode=sched.get("exec_mode", "llm"))
                        state.task_queue.put(task)
                        state.p0_event.set()
                        if sched["schedule_type"] == "cron":
                            from ring0.cron import next_run as _cron_next
                            from datetime import datetime
                            nxt = _cron_next(sched["cron_expr"], datetime.now())
                            scheduled_store.update_after_run(sched["schedule_id"], nxt.timestamp())
                        else:
                            scheduled_store.disable(sched["schedule_id"])
                            scheduled_store.update_after_run(sched["schedule_id"], None)
                        log.info("Scheduled task fired: %s (%s)", sched["name"], sched["schedule_id"])
                except Exception:
                    log.debug("Scheduled task check failed", exc_info=True)

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

                # Task alignment bonus: reward output that matches user interests.
                if user_profiler:
                    try:
                        user_cats = user_profiler.get_category_distribution()
                        topic_kw = user_profiler.get_top_topic_names(limit=30)
                        if user_cats:
                            alignment_bonus = fitness.score_task_alignment(
                                output_lines, user_cats, topic_keywords=topic_kw,
                            )
                            if alignment_bonus > 0:
                                score = min(score + alignment_bonus, 1.0)
                                detail["task_alignment"] = alignment_bonus
                    except Exception:
                        pass

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

                # Notify.
                if notifier:
                    notifier.notify_generation_complete(
                        generation, score, True, last_good_hash or "unknown",
                    )

                # Reflection trigger: after task completion (event-driven).
                if reflector and state.last_task_completion > last_reflection_time:
                    if time.time() - last_reflection_time > r0.reflection_cooldown_sec:
                        try:
                            proposals = reflector.reflect_after_task()
                            if proposals:
                                reflector.process_proposals(proposals)
                            last_reflection_time = time.time()
                        except Exception:
                            log.debug("Post-task reflection failed", exc_info=True)

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
                    if preference_store:
                        try:
                            aggregated = preference_store.aggregate_moments()
                            if aggregated:
                                log.info("Preference aggregation: %d preferences updated", aggregated)
                            decayed = preference_store.apply_confidence_decay()
                            if decayed:
                                log.info("Preference decay: removed %d low-confidence entries", decayed)
                        except Exception:
                            log.debug("Preference maintenance failed (non-fatal)", exc_info=True)

                    if output_queue:
                        try:
                            expired = output_queue.expire_old(max_age_hours=24)
                            if expired:
                                log.info("Output queue: expired %d stale items", expired)
                        except Exception:
                            log.debug("Output queue expiry failed (non-fatal)", exc_info=True)

                    # Nightly consolidation: cross-task correlation + insights (once per day).
                    from datetime import datetime as _dt
                    _today = _dt.now().strftime("%Y-%m-%d")
                    _hour = _dt.now().hour
                    if (memory_store and memory_curator and _today != last_consolidation_date
                            and _hour >= 23):
                        try:
                            cons_result = memory_curator.nightly_consolidate(
                                memory_store, preference_store=preference_store,
                            )
                            last_consolidation_date = _today
                            if cons_result.get("moments_stored", 0):
                                log.info("Nightly consolidation: %s", cons_result)
                        except Exception:
                            log.debug("Nightly consolidation failed (non-fatal)", exc_info=True)

                # Periodic task template sync (interval + jitter to avoid thundering herd).
                if task_syncer and (time.time() - last_skill_sync_time) >= sync_interval:
                    try:
                        sync_result = task_syncer.sync()
                        log.info("Task sync: %s", sync_result)
                    except Exception:
                        log.debug("Task sync failed (non-fatal)", exc_info=True)
                    # Add 0–50% jitter so nodes don't re-sync in lockstep.
                    last_skill_sync_time = time.time() + random.uniform(0, sync_interval * 0.5)

                params = generate_params(generation, 42)
                log.info("Starting generation %d (params: %s)", generation, params)
                proc = _start_ring2(ring2_path, heartbeat_path)
                start_time = time.time()
                hb.wait_for_heartbeat(startup_timeout=timeout)
                continue

            # --- heartbeat check ---
            if hb.is_alive():
                # Idle reflection check.
                if reflector:
                    idle_since_task = time.time() - state.last_task_completion
                    if (idle_since_task > r0.reflection_idle_threshold_sec
                            and time.time() - last_reflection_time > r0.reflection_idle_threshold_sec):
                        try:
                            proposals = reflector.reflect_on_idle()
                            if proposals:
                                reflector.process_proposals(proposals)
                            last_reflection_time = time.time()
                        except Exception:
                            log.debug("Idle reflection failed", exc_info=True)
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

            # Record crash log in memory (with deduplication).
            if memory_store:
                filtered_output = filter_ring2_output(output, max_lines=50) if output else "(no output)"
                crash_content = (
                    f"Gen {generation} died after {elapsed:.0f}s.\n"
                    f"Reason: {failure_reason}\n\n"
                    f"--- Last output ---\n{filtered_output}"
                )

                should_store = True
                try:
                    is_duplicate = memory_store.is_duplicate_content(
                        "crash_log",
                        crash_content,
                        lookback=5,
                        similarity_threshold=0.90
                    )
                    if is_duplicate:
                        should_store = False
                        log.debug("Skipping duplicate crash reason: %s", failure_reason)
                except Exception:
                    pass

                if should_store:
                    memory_store.add(generation, "crash_log", crash_content)

            # Notify.
            if notifier:
                notifier.notify_generation_complete(
                    generation, score, False, commit_hash,
                )

            # Next generation.
            generation += 1
            params = generate_params(generation, 42)
            log.info("Restarting Ring 2 — generation %d (params: %s)", generation, params)
            proc = _start_ring2(ring2_path, heartbeat_path)
            start_time = time.time()
            hb.wait_for_heartbeat(startup_timeout=timeout)

    except KeyboardInterrupt:
        log.info("Sentinel shutting down (KeyboardInterrupt)")
    finally:
        # Disable SIGTERM during cleanup to prevent interrupt-chaining.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        commit_watcher.stop()
        if task_api:
            task_api.stop()
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
        if matrix_bot:
            matrix_bot.stop()

        # Wait for executor thread to finish in-flight work.
        if state.executor_thread and state.executor_thread.is_alive():
            state.executor_thread.join(timeout=5)

        _stop_ring2(proc)
        log.info("Sentinel offline")

    # Restart the entire process if triggered by CommitWatcher.
    if state.restart_event.is_set():
        log.info("Restarting via os.execv()")
        os.execv(sys.executable, [sys.executable] + sys.argv)


def main() -> None:
    debug = os.environ.get("PROTEA_DEBUG", "").strip() not in ("", "0", "false")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if debug:
        logging.getLogger("protea").info("Debug mode enabled (PROTEA_DEBUG=1)")
    project_root = pathlib.Path(__file__).resolve().parent.parent
    run(project_root)
