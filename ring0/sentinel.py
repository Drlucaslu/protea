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


def _compute_skill_hit_ratio(memory_store, limit: int = 50) -> dict:
    """Compute skill hit ratio from recent task metadata.

    Returns {"total": int, "skill": int, "ratio": float, "top_skills": dict[str,int]}.
    """
    entries = []
    for t in ("task", "p1_task"):
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


def _effective_cooldown(base_cooldown: int, skill_ratio: float) -> int:
    """Scale cooldown by skill hit ratio: 1.0x at 0%, up to 3.0x at 100%.

    Linear: multiplier = 1.0 + 2.0 * ratio.
    """
    multiplier = 1.0 + 2.0 * min(skill_ratio, 1.0)
    return int(base_cooldown * multiplier)


def _should_evolve(state, cooldown_sec: int, fitness=None, plateau_window: int = 5, plateau_epsilon: float = 0.03, has_directive: bool = False, last_task_time: float = 0) -> tuple[bool, bool]:
    """Check whether evolution should proceed.

    Returns (should_evolve, is_plateaued):
    - should_evolve: True if evolution should run.
    - is_plateaued: True if scores are stagnant (signals to LLM to try
      something fundamentally different).

    Adaptive evolution: when scores are plateaued AND no user directive
    is pending, skip the LLM call to save tokens.  A directive always
    forces evolution.

    Task idle decay: when no tasks have been completed recently, extend
    the effective cooldown to save tokens during inactive periods.
    """
    if state.p0_active.is_set():
        return False, False
    if state.p1_active.is_set():
        return False, False
    if time.time() - state.last_evolution_time < cooldown_sec:
        return False, False

    # Task idle decay: extend cooldown when no tasks are coming in.
    if last_task_time > 0:
        idle_hours = (time.time() - last_task_time) / 3600
        if idle_hours >= 2:
            idle_multiplier = min(1.0 + idle_hours / 2, 4.0)
            idle_cooldown = int(cooldown_sec * idle_multiplier)
            if time.time() - state.last_evolution_time < idle_cooldown:
                log.info("Task idle %.1fh — extended cooldown %ds", idle_hours, idle_cooldown)
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


def _try_auto_directive(memory_store, user_profiler, project_root):
    """Best-effort: generate a directive from recent tasks + user profile.

    Returns a directive string or None on failure / insufficient data.
    """
    try:
        from ring1.config import load_ring1_config
        from ring1.directive_generator import DirectiveGenerator

        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            return None

        task_history = []
        if memory_store:
            try:
                task_history = memory_store.get_by_type("task", limit=30)
            except Exception:
                pass

        profile_summary = ""
        if user_profiler:
            try:
                profile_summary = user_profiler.get_profile_summary()
            except Exception:
                pass

        client = r1_config.get_llm_client()
        generator = DirectiveGenerator(client)
        return generator.generate(task_history, profile_summary)
    except Exception as exc:
        log.debug("Auto-directive generation failed: %s", exc)
        return None


def _build_evolution_direction(gene_pool, user_profiler, skill_store, memory_store):
    """Build dynamic evolution direction from genes + user interests."""
    parts = []

    # 1. From user_profile: top categories + topics.
    if user_profiler:
        try:
            cats = user_profiler.get_category_distribution()
            if cats:
                top_cats = list(cats.items())[:5]
                parts.append("### User Interest Areas (by frequency)")
                for cat, weight in top_cats:
                    parts.append(f"- {cat} (weight: {weight:.0f})")
            topics = user_profiler.get_top_topic_names(limit=10)
            if topics:
                parts.append("")
                parts.append("### Specific Topics of Interest")
                parts.append(", ".join(topics))
        except Exception:
            pass

    # 2. From gene_pool: top scoring genes' summaries (DNA).
    if gene_pool:
        try:
            top_genes = gene_pool.get_top(5)
            if top_genes:
                parts.append("")
                parts.append("### Successful Gene Patterns (DNA)")
                parts.append("These patterns scored well in past generations. Build on them:")
                for g in top_genes:
                    summary = g.get("gene_summary", "")
                    score = g.get("score", 0)
                    task_hits = g.get("task_hit_count", 0) or 0
                    if summary:
                        parts.append(f"- [score={score:.2f}, task_hits={task_hits}] {summary[:120]}")
        except Exception:
            pass

    # 3. From recent tasks: uncovered task types (evolution opportunities).
    if memory_store and skill_store:
        try:
            tasks = memory_store.get_by_type("task", limit=20)
            uncovered = []
            for t in tasks:
                meta = t.get("metadata", {})
                if isinstance(meta, str):
                    meta = json.loads(meta) if meta else {}
                if not meta.get("skills_used") and not meta.get("skills_matched"):
                    content = t.get("content", "")[:80]
                    if content:
                        uncovered.append(content)
            if uncovered:
                parts.append("")
                parts.append("### Uncovered Task Types (evolution opportunities)")
                parts.append(
                    "These recent user tasks had NO skill coverage — "
                    "evolve toward capabilities that serve these needs:"
                )
                for task_desc in uncovered[:5]:
                    parts.append(f"- {task_desc}")
        except Exception:
            pass

    return "\n".join(parts) if parts else ""


def _try_evolve(project_root, fitness, ring2_path, generation, params, survived, notifier, directive="", memory_store=None, skill_store=None, crash_logs=None, is_plateaued=False, gene_pool=None, user_profile_summary="", structured_preferences="", venv_manager=None, allowed_packages=None, skill_hit_summary=None, evolution_direction=""):
    """Best-effort evolution.  Returns (success, gene_ids, adopted_ids) tuple."""
    try:
        from ring1.config import load_ring1_config
        from ring1.evolver import Evolver

        r1_config = load_ring1_config(project_root)
        if not r1_config.has_llm_config():
            log.warning("LLM API key not configured — skipping evolution")
            return False, [], []

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
        _injected_gene_ids: list[int] = []
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
                if genes:
                    _injected_gene_ids = [g["id"] for g in genes if "id" in g]
                    if _injected_gene_ids:
                        gene_pool.record_hits(_injected_gene_ids, generation)
                        gene_pool.record_hypothesis(generation, _injected_gene_ids)
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

        # Fetch semantic rules for evolution context.
        semantic_rules = []
        if memory_store:
            try:
                semantic_rules = memory_store.get_semantic_rules(limit=10)
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
            persistent_errors=persistent_errors,
            is_plateaued=is_plateaued,
            gene_pool=genes,
            evolution_intent=evolution_intent,
            user_profile_summary=user_profile_summary,
            structured_preferences=structured_preferences,
            permanent_capabilities=permanent_caps or None,
            allowed_packages=list(allowed_pkg_set) if allowed_pkg_set else None,
            skill_hit_summary=skill_hit_summary,
            semantic_rules=semantic_rules or None,
            evolution_direction=evolution_direction,
        )
        if result.success:
            log.info("Evolution succeeded: %s", result.reason)
            # Record LLM token usage.
            if result.metadata and result.metadata.get("llm_usage"):
                usage = result.metadata["llm_usage"]
                try:
                    fitness.record_llm_usage(
                        generation, "evolution",
                        usage["input_tokens"], usage["output_tokens"],
                    )
                except Exception:
                    pass
                log.info(
                    "Evolution tokens: in=%d out=%d",
                    usage["input_tokens"], usage["output_tokens"],
                )
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

            # Verify gene adoption: only genes actually used in new code get hits.
            _adopted_gene_ids: list[int] = []
            if gene_pool and genes and result.new_source:
                try:
                    gene_ids = [g["id"] for g in genes if "id" in g]
                    if gene_ids:
                        _adopted_gene_ids = gene_pool.verify_adoption(
                            result.new_source, gene_ids, generation,
                        )
                except Exception:
                    pass

            return True, _injected_gene_ids, _adopted_gene_ids
        else:
            log.warning("Evolution failed: %s", result.reason)
            if notifier:
                notifier.notify_error(generation, result.reason)
            return False, [], []
    except Exception as exc:
        log.error("Evolution error (non-fatal): %s", exc)
        if notifier:
            notifier.notify_error(generation, str(exc))
        return False, [], []


def _try_crystallize(project_root, skill_store, source_code, output, generation, skill_cap=100, fitness=None, gene_ids=None):
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
        # Record LLM token usage.
        if result.llm_usage and result.llm_usage.get("input_tokens"):
            if fitness:
                try:
                    fitness.record_llm_usage(
                        generation, "crystallization",
                        result.llm_usage["input_tokens"],
                        result.llm_usage["output_tokens"],
                    )
                except Exception:
                    pass
            log.info(
                "Crystallization tokens: in=%d out=%d",
                result.llm_usage["input_tokens"],
                result.llm_usage["output_tokens"],
            )

        # Record gene → skill lineage for task-hit attribution.
        if result.action in ("create", "update") and result.skill_name and gene_ids:
            try:
                skill_store.record_lineage(result.skill_name, gene_ids, generation)
            except Exception:
                pass

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


def _create_executor(project_root, state, ring2_path, reply_fn, memory_store=None, skill_store=None, skill_runner=None, task_store=None, user_profiler=None, embedding_provider=None, scheduled_store=None, send_file_fn=None, preference_store=None, gene_pool=None):
    """Best-effort task executor creation."""
    def _factory():
        from ring1.config import load_ring1_config
        from ring1.task_executor import create_executor, start_executor_thread
        r1_config = load_ring1_config(project_root)
        executor = create_executor(r1_config, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store, skill_runner=skill_runner, task_store=task_store, user_profiler=user_profiler, embedding_provider=embedding_provider, scheduled_store=scheduled_store, send_file_fn=send_file_fn, preference_store=preference_store, gene_pool=gene_pool)
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
    embedding_provider = _create_embedding_provider(cfg)
    gene_pool = _best_effort("GenePool", lambda: __import__("ring0.gene_pool", fromlist=["GenePool"]).GenePool(db_path, embedding_provider=embedding_provider))
    task_store = _best_effort("TaskStore", lambda: __import__("ring0.task_store", fromlist=["TaskStore"]).TaskStore(db_path))
    scheduled_store = _best_effort("ScheduledTaskStore", lambda: __import__("ring0.scheduled_task_store", fromlist=["ScheduledTaskStore"]).ScheduledTaskStore(db_path))
    user_profiler = _best_effort("UserProfiler", lambda: __import__("ring0.user_profile", fromlist=["UserProfiler"]).UserProfiler(db_path))
    preference_store = _best_effort("PreferenceStore", lambda: __import__("ring0.preference_store", fromlist=["PreferenceStore"]).PreferenceStore(db_path, cfg.get("ring1", {}).get("user_profile", {})))
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
    state.scheduled_store = scheduled_store
    state._preference_store = preference_store
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

    # Backfill skill lineage (one-time heuristic).
    lineage_backfilled = 0
    if skill_store and gene_pool:
        try:
            lineage_backfilled = skill_store.backfill_lineage(gene_pool)
            if lineage_backfilled:
                log.info("Skill lineage backfilled for %d skills", lineage_backfilled)
        except Exception as exc:
            log.debug("Skill lineage backfill failed (non-fatal): %s", exc)

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

    # Post-backfill attribution: credit historical tasks.
    if lineage_backfilled and memory_store:
        try:
            recent_tasks = memory_store.get_by_type("task", limit=50)
            attributed_gene_ids: set[int] = set()
            for task in recent_tasks:
                meta = task.get("metadata", {})
                if isinstance(meta, str):
                    meta = json.loads(meta) if meta else {}
                for skill_name in meta.get("skills_used", []):
                    for entry in skill_store.get_lineage(skill_name):
                        attributed_gene_ids.add(entry["gene_id"])
                for skill_name in meta.get("skills_matched", []):
                    for entry in skill_store.get_lineage(skill_name):
                        attributed_gene_ids.add(entry["gene_id"])
            if attributed_gene_ids:
                gene_pool.record_task_hits(list(attributed_gene_ids), 0)
                log.info("Post-backfill attribution: %d genes credited", len(attributed_gene_ids))
        except Exception as exc:
            log.debug("Post-backfill attribution failed (non-fatal): %s", exc)

    # Task executor for P0 user tasks.
    reply_fn = bot._send_reply if bot else lambda text: None
    send_file_fn = bot._send_document if bot else None
    executor = _create_executor(project_root, state, ring2_path, reply_fn, memory_store=memory_store, skill_store=skill_store, skill_runner=skill_runner, task_store=task_store, user_profiler=user_profiler, embedding_provider=embedding_provider, scheduled_store=scheduled_store, send_file_fn=send_file_fn, preference_store=preference_store, gene_pool=gene_pool)
    # Feedback prompt after task completion (not on intermediate messages).
    if executor and bot:
        executor.feedback_fn = bot.send_feedback_prompt
    # Expose subagent_manager on state for /background command.
    state.subagent_manager = getattr(executor, "subagent_manager", None) if executor else None

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
            )
        proactive_loop = _best_effort("ProactiveLoop", _create_proactive)
        if proactive_loop:
            log.info("ProactiveLoop enabled (morning=%d, evening=%d)",
                     proactive_cfg.get("morning_hour", 9),
                     proactive_cfg.get("evening_hour", 21))

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
    last_crystallized_hash: str | None = None
    # Jitter: initial delay of 0–50% of interval so nodes don't all sync at once.
    last_skill_sync_time: float = time.time() - sync_interval + random.uniform(0, sync_interval * 0.5)
    last_consolidation_date: str = ""  # YYYY-MM-DD — nightly consolidation
    skill_cap = r0.skill_max_count
    proc: subprocess.Popen | None = None

    # Initial snapshot of seed code.
    try:
        last_good_hash = git.snapshot(f"gen-{generation} seed")
    except subprocess.CalledProcessError:
        pass

    log.info("Sentinel online — heartbeat every %ds, timeout %ds, cooldown %ds", interval, timeout, cooldown_sec)
    
    # Notify Telegram that sentinel is online
    if notifier:
        notifier.notify_sentinel_online(generation)

    last_injected_gene_ids: list[int] = []
    last_attributed_task_id: int = 0
    directive_remaining_cycles: int = 0

    try:
        params = generate_params(generation, seed)
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
                        task = Task(text=sched["task_text"], chat_id=sched["chat_id"])
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
                        # Build gene_ids: current gene + any injected parent genes.
                        crystallize_gene_ids = list(last_injected_gene_ids)
                        if gene_pool:
                            current_gene_id = gene_pool.get_id_by_hash(source_hash)
                            if current_gene_id is not None and current_gene_id not in crystallize_gene_ids:
                                crystallize_gene_ids.append(current_gene_id)
                        log.info("Crystallizing gen-%d (hash=%s…)", generation, source_hash[:12])
                        _try_crystallize(
                            project_root, skill_store, source, output,
                            generation, skill_cap=skill_cap,
                            fitness=fitness,
                            gene_ids=crystallize_gene_ids,
                        )
                        last_crystallized_hash = source_hash
                    else:
                        log.debug("Skipping crystallization — source unchanged (hash=%s…)", source_hash[:12])

                # Evolve (best-effort) — skip if busy, cooling down, or plateaued.
                # Dynamic cooldown based on skill hit ratio.
                skill_hit = {"total": 0, "skill": 0, "ratio": 0.0, "top_skills": {}}
                if memory_store:
                    try:
                        skill_hit = _compute_skill_hit_ratio(memory_store)
                    except Exception:
                        pass
                eff_cooldown = _effective_cooldown(cooldown_sec, skill_hit["ratio"])

                with state.lock:
                    pending_directive = state.evolution_directive
                should_evo, plateaued = _should_evolve(
                    state, eff_cooldown, fitness=fitness,
                    plateau_window=plateau_window,
                    plateau_epsilon=plateau_epsilon,
                    has_directive=bool(pending_directive),
                    last_task_time=state.last_task_completion,
                )
                if not should_evo:
                    if plateaued and not pending_directive:
                        auto_dir = _try_auto_directive(memory_store, user_profiler, project_root)
                        if auto_dir:
                            with state.lock:
                                state.evolution_directive = auto_dir
                            should_evo = True
                            pending_directive = auto_dir
                            log.info("Auto-directive: %s", auto_dir[:80])
                        else:
                            log.info("Plateau, no auto-directive — skipping")
                    elif not plateaued:
                        log.info("Skipping evolution (busy or cooldown %.0fs, skill ratio %.0f%%)",
                                 eff_cooldown, skill_hit["ratio"] * 100)
                if not should_evo:
                    evolved = False
                else:
                    with state.lock:
                        directive = state.evolution_directive
                        if directive:
                            if directive_remaining_cycles <= 0:
                                directive_remaining_cycles = 3  # new directive lives for 3 cycles
                            directive_remaining_cycles -= 1
                            if directive_remaining_cycles <= 0:
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
                    # Get structured preferences for evolution.
                    pref_summary = ""
                    if preference_store:
                        try:
                            pref_summary = preference_store.get_preference_summary_text()
                        except Exception:
                            pass
                    # Build dynamic evolution direction.
                    evo_direction = _build_evolution_direction(
                        gene_pool, user_profiler, skill_store, memory_store,
                    )
                    evolved, last_injected_gene_ids, adopted_gene_ids = _try_evolve(
                        project_root, fitness, ring2_path,
                        generation, params, True, notifier,
                        directive=directive,
                        memory_store=memory_store,
                        skill_store=skill_store,
                        crash_logs=crash_logs,
                        is_plateaued=plateaued,
                        gene_pool=gene_pool,
                        user_profile_summary=profile_summary,
                        structured_preferences=pref_summary,
                        venv_manager=venv_manager,
                        allowed_packages=allowed_packages,
                        skill_hit_summary=skill_hit,
                        evolution_direction=evo_direction,
                    )
                    if gene_pool and last_injected_gene_ids:
                        try:
                            gene_pool.close_hypothesis(
                                generation, survived=True, score=score,
                                adopted_ids=adopted_gene_ids,
                            )
                        except Exception:
                            pass
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

                # Task hit attribution: skill usage → lineage → gene scoring.
                # Runs every generation so hits are never missed.
                if gene_pool and skill_store and memory_store:
                    try:
                        recent_tasks = memory_store.get_by_type("task", limit=30)
                        new_tasks = [t for t in recent_tasks if t.get("id", 0) > last_attributed_task_id]
                        attributed_gene_ids: set[int] = set()
                        for task in new_tasks:
                            meta = task.get("metadata", {})
                            if isinstance(meta, str):
                                meta = json.loads(meta) if meta else {}
                            for skill_name in meta.get("skills_used", []):
                                for entry in skill_store.get_lineage(skill_name):
                                    attributed_gene_ids.add(entry["gene_id"])
                            for skill_name in meta.get("skills_matched", []):
                                for entry in skill_store.get_lineage(skill_name):
                                    attributed_gene_ids.add(entry["gene_id"])
                        if attributed_gene_ids:
                            gene_pool.record_task_hits(list(attributed_gene_ids), generation)
                        if new_tasks:
                            last_attributed_task_id = max(t.get("id", 0) for t in new_tasks)
                    except Exception:
                        log.debug("Task hit attribution failed (non-fatal)", exc_info=True)

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
                        # Detect preference drift and inject evolution directive if significant.
                        try:
                            drifts = preference_store.detect_drift()
                            rising = [d for d in drifts if d["drift_direction"] == "rising"]
                            if rising:
                                top_drift = rising[0]
                                drift_msg = (
                                    f"User interest rising in '{top_drift['preference_key']}' "
                                    f"(confidence {top_drift['old_confidence']:.2f} → "
                                    f"{top_drift['new_confidence']:.2f})"
                                )
                                log.info("Preference drift detected: %s", drift_msg)
                                with state.lock:
                                    if not state.evolution_directive:
                                        state.evolution_directive = (
                                            f"Adapt to user's increasing interest: {drift_msg}"
                                        )
                        except Exception:
                            log.debug("Drift detection failed (non-fatal)", exc_info=True)

                    if gene_pool:
                        try:
                            recent_tasks = []
                            if memory_store:
                                recent_tasks = [
                                    t for t in memory_store.get_by_type("task", limit=20)
                                    if t.get("generation", 0) > generation - 10
                                ]
                                if recent_tasks:
                                    task_ctx = " ".join(t.get("content", "") for t in recent_tasks)
                                    matched = gene_pool.get_relevant(task_ctx, 5)
                                    if matched:
                                        ids = [g["id"] for g in matched if "id" in g]
                                        if ids:
                                            gene_pool.record_hits(ids, generation)
                            task_boosted = gene_pool.apply_task_boost()
                            boosted = gene_pool.apply_boost()
                            decayed = gene_pool.apply_decay(generation)
                            if task_boosted or boosted or decayed:
                                log.info("Gene scoring: task_boost=%d code_boost=%d decayed=%d",
                                         task_boosted, boosted, decayed)
                        except Exception:
                            log.debug("Gene scoring failed (non-fatal)", exc_info=True)

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
            # Dynamic cooldown based on skill hit ratio.
            skill_hit = {"total": 0, "skill": 0, "ratio": 0.0, "top_skills": {}}
            if memory_store:
                try:
                    skill_hit = _compute_skill_hit_ratio(memory_store)
                except Exception:
                    pass
            eff_cooldown = _effective_cooldown(cooldown_sec, skill_hit["ratio"])

            with state.lock:
                pending_directive = state.evolution_directive
            should_evo, plateaued = _should_evolve(
                state, eff_cooldown, fitness=fitness,
                plateau_window=plateau_window,
                plateau_epsilon=plateau_epsilon,
                has_directive=True,  # failures always force evolution
                last_task_time=state.last_task_completion,
            )
            if not should_evo:
                log.info("Skipping evolution (busy or cooldown %.0fs, skill ratio %.0f%%)",
                         eff_cooldown, skill_hit["ratio"] * 100)
                evolved = False
            else:
                with state.lock:
                    directive = state.evolution_directive
                    if directive:
                        if directive_remaining_cycles <= 0:
                            directive_remaining_cycles = 3
                        directive_remaining_cycles -= 1
                        if directive_remaining_cycles <= 0:
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
                # Get structured preferences for evolution.
                pref_summary = ""
                if preference_store:
                    try:
                        pref_summary = preference_store.get_preference_summary_text()
                    except Exception:
                        pass
                # Build dynamic evolution direction.
                evo_direction = _build_evolution_direction(
                    gene_pool, user_profiler, skill_store, memory_store,
                )
                evolved, last_injected_gene_ids, _ = _try_evolve(
                    project_root, fitness, ring2_path,
                    generation, params, False, notifier,
                    directive=directive,
                    memory_store=memory_store,
                    skill_store=skill_store,
                    crash_logs=crash_logs,
                    is_plateaued=False,  # failure path — focus on fixing, not novelty
                    gene_pool=gene_pool,
                    user_profile_summary=profile_summary,
                    structured_preferences=pref_summary,
                    venv_manager=venv_manager,
                    allowed_packages=allowed_packages,
                    skill_hit_summary=skill_hit,
                    evolution_direction=evo_direction,
                )
                if gene_pool and last_injected_gene_ids:
                    try:
                        gene_pool.close_hypothesis(
                            generation, survived=False, score=score,
                            adopted_ids=[],
                        )
                    except Exception:
                        pass
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
        # Disable SIGTERM during cleanup to prevent interrupt-chaining.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

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
        if matrix_bot:
            matrix_bot.stop()

        # Wait for executor thread to finish in-flight work.
        if state.executor_thread and state.executor_thread.is_alive():
            state.executor_thread.join(timeout=5)

        _stop_ring2(proc)
        log.info("Sentinel offline")

    # Restart the entire process if triggered by CommitWatcher.
    # Python 3 sets O_CLOEXEC on open() fds by default, so os.execv()
    # automatically closes them — no manual fd cleanup needed.
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
