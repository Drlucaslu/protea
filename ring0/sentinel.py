"""Sentinel — Ring 0 main loop (pure stdlib).

Launches and supervises Ring 2.  On success (survived max_runtime_sec),
triggers Ring 1 evolution to mutate the code.  On failure, rolls back
to the last known-good commit, evolves from that base, and restarts.
"""

from __future__ import annotations

import json
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
    seed: int
    cooldown_sec: int
    plateau_window: int
    plateau_epsilon: float
    skill_max_count: int
    max_cpu_percent: float
    max_memory_percent: float
    max_disk_percent: float

    @classmethod
    def from_dict(cls, r0: dict) -> "Ring0Config":
        """Parse from the raw [ring0] config dict."""
        evo = r0.get("evolution", {})
        return cls(
            ring2_path=r0["git"]["ring2_path"],
            db_path=r0["fitness"]["db_path"],
            heartbeat_interval_sec=r0["heartbeat_interval_sec"],
            heartbeat_timeout_sec=r0["heartbeat_timeout_sec"],
            seed=evo["seed"],
            cooldown_sec=evo.get("cooldown_sec", 900),
            plateau_window=evo.get("plateau_window", 5),
            plateau_epsilon=evo.get("plateau_epsilon", 0.03),
            skill_max_count=evo.get("skill_max_count", 100),
            max_cpu_percent=r0["max_cpu_percent"],
            max_memory_percent=r0["max_memory_percent"],
            max_disk_percent=r0["max_disk_percent"],
        )


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


def _best_effort(label, factory):
    """Run *factory*; return its result or None on any error."""
    try:
        return factory()
    except Exception as exc:
        log.debug("%s not available: %s", label, exc)
        return None


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


def _try_install_capability(proposal, skill_store, venv_manager, allowed_packages):
    """Validate and install a capability skill proposed by evolution.

    Returns True if installed successfully.
    """
    from ring1.skill_validator import validate_skill_local, validate_dependencies

    name = proposal.get("name", "")
    source_code = proposal.get("source_code", "")
    dependencies = proposal.get("dependencies", [])

    if not name or not source_code:
        log.warning("Capability proposal missing name or source_code")
        return False

    # 1. Validate source code (local/lenient — evolved skills are trusted).
    code_result = validate_skill_local(source_code)
    if not code_result.safe:
        log.warning("Capability '%s' rejected: unsafe code — %s", name, code_result.errors)
        return False

    # 2. Validate dependencies.
    dep_result = validate_dependencies(dependencies, allowed_packages)
    if not dep_result.safe:
        log.warning("Capability '%s' rejected: bad deps — %s", name, dep_result.errors)
        return False

    # 3. Install to skill store.
    try:
        skill_store.add(
            name=name,
            description=proposal.get("description", ""),
            prompt_template=proposal.get("description", ""),
            tags=proposal.get("tags", []),
            source_code=source_code,
            source="evolved",
            dependencies=dependencies,
        )
    except Exception as exc:
        log.warning("Capability '%s' install failed: %s", name, exc)
        return False

    # 4. Pre-create venv (best-effort).
    if venv_manager and dependencies:
        try:
            venv_manager.ensure_env(name, dependencies)
            log.info("Capability '%s' installed with deps: %s", name, dependencies)
        except Exception as exc:
            log.warning("Capability '%s' venv setup failed (will retry on first run): %s", name, exc)

    return True


def _try_evolve(project_root, fitness, ring2_path, generation, params, survived, notifier, directive="", memory_store=None, skill_store=None, crash_logs=None, is_plateaued=False, gene_pool=None, user_profile_summary="", venv_manager=None, allowed_packages=None):
    """Best-effort evolution.  Returns True if new code was written."""
    try:
        from ring1.config import load_ring1_config
        from ring1.evolver import Evolver

        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            log.warning("LLM API key not configured — skipping evolution")
            return False

        # Compact context: directives and 1 reflection only.
        # Reflections/crash_logs are machine-generated with low priority;
        # user tasks are the primary evolution signal.
        memories = []
        if memory_store:
            try:
                for t in ("directive", "reflection"):
                    memories.extend(memory_store.get_by_type(t, limit=1))
                memories.sort(key=lambda m: m.get("id", 0), reverse=True)
                memories = memories[:2]
            except Exception:
                memories = []

        # User task history — primary evolution signal, higher limit.
        task_history = []
        if memory_store:
            try:
                task_history = memory_store.get_by_type("task", limit=8)
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

        # Persist intent to memory so the Dashboard intent timeline works.
        if memory_store:
            try:
                intent_content = f"{evolution_intent['intent']}: {', '.join(evolution_intent.get('signals', []))}"
                memory_store.add(
                    generation, "evolution_intent", intent_content,
                    metadata=evolution_intent,
                )
            except Exception:
                pass

        # Collect permanent capabilities for the evolution prompt.
        permanent_caps = []
        if skill_store:
            try:
                permanent_caps = skill_store.get_permanent()
            except Exception:
                pass

        # Build allowed packages list.
        allowed_pkg_set = allowed_packages

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
            permanent_capabilities=permanent_caps or None,
            allowed_packages=list(allowed_pkg_set) if allowed_pkg_set else None,
        )
        if result.success:
            log.info("Evolution succeeded: %s", result.reason)
            if result.metadata:
                blast = result.metadata.get("blast_radius", {})
                log.info(
                    "Evolution metadata: intent=%s scope=%s lines_changed=%d",
                    result.metadata.get("intent"),
                    blast.get("scope"),
                    blast.get("lines_changed", 0),
                )
                # Update the intent memory entry with blast_radius.
                if memory_store and blast:
                    try:
                        intents = memory_store.get_by_type("evolution_intent", limit=1)
                        if intents and intents[0].get("generation") == generation:
                            meta = intents[0].get("metadata", {})
                            meta["blast_radius"] = blast
                            with memory_store._connect() as con:
                                con.execute(
                                    "UPDATE memory SET metadata = ? WHERE id = ?",
                                    (json.dumps(meta), intents[0]["id"]),
                                )
                    except Exception:
                        pass

                # Handle capability proposal from evolution.
                proposal = result.metadata.get("capability_proposal")
                if proposal and skill_store:
                    installed = _try_install_capability(
                        proposal, skill_store, venv_manager, allowed_pkg_set,
                    )
                    if installed and gene_pool:
                        try:
                            gene_pool.add(
                                generation=generation,
                                score=0.85,
                                source_code=proposal["source_code"],
                            )
                        except Exception:
                            pass

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
                    resp = registry_client.publish(
                        name=skill_data["name"],
                        description=skill_data.get("description", ""),
                        prompt_template=skill_data.get("prompt_template", ""),
                        parameters=skill_data.get("parameters"),
                        tags=skill_data.get("tags"),
                        source_code=skill_data.get("source_code", ""),
                    )
                    if resp is not None:
                        skill_store.mark_published(result.skill_name)
                    log.info("Published skill %r to registry", result.skill_name)
            except Exception as pub_exc:
                log.debug("Registry publish failed (non-fatal): %s", pub_exc)

        return result.action
    except Exception as exc:
        log.error("Crystallization error (non-fatal): %s", exc)
        return None


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


def _create_skill_syncer(skill_store, registry_client, user_profiler, cfg):
    """Best-effort SkillSyncer creation."""
    if not skill_store or not registry_client:
        return None
    def _factory():
        from ring1.skill_sync import SkillSyncer
        sync_cfg = cfg.get("ring1", {}).get("skill_sync", {})
        if not sync_cfg.get("enabled", True):
            return None
        max_discover = sync_cfg.get("max_discover_per_sync", 5)
        syncer = SkillSyncer(
            skill_store=skill_store,
            registry_client=registry_client,
            user_profiler=user_profiler,
            max_discover=max_discover,
        )
        log.info("SkillSyncer created (max_discover=%d)", max_discover)
        return syncer
    return _best_effort("SkillSyncer", _factory)


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


def _create_executor(project_root, state, ring2_path, reply_fn, memory_store=None, skill_store=None, skill_runner=None, task_store=None, registry_client=None, user_profiler=None, embedding_provider=None):
    """Best-effort task executor creation."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.task_executor import create_executor, start_executor_thread
        r1_config = load_ring1_config(project_root)
        executor = create_executor(r1_config, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store, skill_runner=skill_runner, task_store=task_store, registry_client=registry_client, user_profiler=user_profiler, embedding_provider=embedding_provider)
        if executor:
            thread = start_executor_thread(executor)
            state.executor_thread = thread
            log.info("Task executor started")
        return executor
    return _best_effort("Task executor", _factory)


def run(project_root: pathlib.Path) -> None:
    """Sentinel main loop — run until interrupted."""
    # Convert SIGTERM into KeyboardInterrupt so the finally block runs,
    # ensuring Ring 2 subprocess, skill runners, and the Telegram bot
    # are stopped cleanly.
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
    seed = r0.seed
    cooldown_sec = r0.cooldown_sec
    plateau_window = r0.plateau_window
    plateau_epsilon = r0.plateau_epsilon

    git = GitManager(ring2_path)
    git.init_repo()
    fitness = FitnessTracker(db_path)
    memory_store = _best_effort("MemoryStore", lambda: MemoryStore(db_path))
    skill_store = _best_effort("SkillStore", lambda: __import__("ring0.skill_store", fromlist=["SkillStore"]).SkillStore(db_path))
    gene_pool = _best_effort("GenePool", lambda: __import__("ring0.gene_pool", fromlist=["GenePool"]).GenePool(db_path))
    task_store = _best_effort("TaskStore", lambda: __import__("ring0.task_store", fromlist=["TaskStore"]).TaskStore(db_path))
    user_profiler = _best_effort("UserProfiler", lambda: __import__("ring0.user_profile", fromlist=["UserProfiler"]).UserProfiler(db_path))
    embedding_provider = _create_embedding_provider(cfg)
    memory_curator = _create_memory_curator(project_root)

    # Capability skill sandbox — venv manager + allowed packages.
    venv_manager = None
    allowed_packages = None
    sandbox_cfg = cfg.get("ring1", {}).get("skill_sandbox", {})
    if sandbox_cfg.get("enabled", True):
        try:
            from ring1.skill_sandbox import VenvManager
            from ring1.skill_validator import _DEFAULT_ALLOWED_PACKAGES
            base_dir = project_root / sandbox_cfg.get("base_dir", "data/skill_envs")
            max_envs = sandbox_cfg.get("max_envs", 10)
            venv_manager = VenvManager(base_dir, max_envs)
            allowed_packages = _DEFAULT_ALLOWED_PACKAGES | frozenset(
                sandbox_cfg.get("extra_allowed_packages", [])
            )
            log.info("VenvManager created (base_dir=%s, max_envs=%d)", base_dir, max_envs)
        except Exception as exc:
            log.debug("VenvManager not available: %s", exc)

    hb = HeartbeatMonitor(heartbeat_path, timeout_sec=timeout)
    notifier = _create_notifier(project_root)

    # Shared state for Telegram bot interaction.
    from ring1.telegram_bot import SentinelState
    state = SentinelState()
    state.notifier = notifier  # bot uses this for auto-detect propagation
    skill_runner = _best_effort("SkillRunner", lambda: __import__("ring1.skill_runner", fromlist=["SkillRunner"]).SkillRunner(venv_manager=venv_manager))
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

    # Skill syncer — periodic publish + discover.
    skill_syncer = _create_skill_syncer(skill_store, registry_client, user_profiler, cfg)
    sync_interval = cfg.get("ring1", {}).get("skill_sync", {}).get("interval_sec", 7200)

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
    last_skill_sync_time: float = 0.0  # epoch — triggers sync on first eligible moment
    skill_cap = r0.skill_max_count
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
                # Note: We no longer store observations in memory (see line 234 comment).
                # They're noisy per-generation logs that crowd out useful memories.

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
                        venv_manager=venv_manager,
                        allowed_packages=allowed_packages,
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

                # Periodic skill sync (every sync_interval seconds, default 2 hours).
                if skill_syncer and (time.time() - last_skill_sync_time) >= sync_interval:
                    try:
                        sync_result = skill_syncer.sync()
                        log.info("Skill sync: %s", sync_result)
                        last_skill_sync_time = time.time()
                    except Exception:
                        log.debug("Skill sync failed (non-fatal)", exc_info=True)
                        last_skill_sync_time = time.time()  # avoid retry storm

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

            # Record crash log and observation in memory (with deduplication).
            if memory_store:
                # Filter output to remove system noise before storing
                filtered_output = filter_ring2_output(output, max_lines=50) if output else "(no output)"
                crash_content = (
                    f"Gen {generation} died after {elapsed:.0f}s.\n"
                    f"Reason: {failure_reason}\n\n"
                    f"--- Last output ---\n{filtered_output}"
                )
                
                # Check for duplicate crashes using fuzzy matching on failure reason
                should_store = True
                try:
                    # Use fuzzy deduplication for crash logs (90% threshold for stricter matching)
                    is_duplicate = memory_store.is_duplicate_content(
                        "crash_log",
                        crash_content,
                        lookback=5,
                        similarity_threshold=0.90
                    )
                    
                    if is_duplicate:
                        should_store = False
                        log.debug(f"Skipping duplicate crash reason: {failure_reason}")
                except Exception:
                    pass  # If query fails, store anyway
                
                if should_store:
                    memory_store.add(generation, "crash_log", crash_content)

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
                    venv_manager=venv_manager,
                    allowed_packages=allowed_packages,
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
