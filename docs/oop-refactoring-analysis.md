# Protea OOP & Modular Refactoring Analysis

> Date: 2026-02-17

## Current Strengths

1. **Ring isolation** — ring0 has zero top-level imports from ring1; all cross-ring calls are lazy + try/except. ring1 has zero imports from ring0. No circular dependencies.
2. **LLM abstraction** — `LLMClient` ABC + `ClaudeClient` / `OpenAIClient` is a clean Strategy pattern.
3. **Value objects** — `EvolutionParams`, `Ring1Config`, `EvolutionResult`, `CrystallizationResult` are all immutable NamedTuples.
4. **Defensive resilience** — every optional component degrades gracefully to `None`.
5. **Good test coverage** — ~30 test files for ~30 source modules (924 tests).

---

## Improvement Areas

### 1. Store Layer: Missing Common Base Class (Biggest DRY Violation)

**Problem:** 5 stores (`FitnessTracker`, `MemoryStore`, `SkillStore`, `GenePool`, `TaskStore`) each independently implement ~30 lines of identical SQLite boilerplate:

```python
# Duplicated across all 5 stores
def __init__(self, db_path):
    self.db_path = db_path
    with self._connect() as con:
        con.execute(_CREATE_TABLE)

def _connect(self):
    con = sqlite3.connect(str(self.db_path))
    con.row_factory = sqlite3.Row
    return con

@staticmethod
def _row_to_dict(row):
    return dict(row)
```

**Proposal:** Extract `SQLiteStore` base class:

```python
class SQLiteStore:
    _CREATE_TABLE: str  # subclass provides

    def __init__(self, db_path: pathlib.Path):
        self.db_path = db_path
        with self._connect() as con:
            con.execute(self._CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection: ...
    def _row_to_dict(self, row) -> dict: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...
```

Each store then only defines `_CREATE_TABLE` and its business methods.

**Files affected:** `ring0/fitness.py`, `ring0/memory.py`, `ring0/skill_store.py`, `ring0/task_store.py`, `ring0/gene_pool.py` + new `ring0/sqlite_store.py`

---

### 2. Ring 0 Config: Raw Dict vs Typed Object

**Problem:** Ring 1 has typed `Ring1Config` (NamedTuple, 17 fields), but Ring 0 uses raw dict:

```python
# Ring 0 — no type safety, key typos are silent failures
r0 = cfg["ring0"]
interval = r0["heartbeat_interval_sec"]  # typo? runtime crash
```

**Proposal:** Create `Ring0Config` NamedTuple:

```python
class Ring0Config(NamedTuple):
    heartbeat_interval_sec: float
    heartbeat_timeout_sec: float
    max_runtime_sec: float
    cooldown_sec: float
    plateau_window: int
    plateau_epsilon: float
    ring2_path: str
    db_path: str
    seed: int
```

**Files affected:** `ring0/sentinel.py` + new loader function

---

### 3. `SentinelState` — Universal State Bag

**Problem:** `SentinelState` in `ring1/telegram_bot.py` holds 20+ fields spanning unrelated concerns:

| Category | Fields |
|----------|--------|
| Lifecycle | `generation`, `start_time`, `alive` |
| Evolution | `mutation_rate`, `max_runtime_sec`, `last_score` |
| Events | `pause_event`, `kill_event`, `restart_event` |
| Components | `skill_runner`, `executor`, `task_store` |
| Tasks | `task_queue`, `evolution_directive` |

**Proposal:** Split into focused dataclasses:

```python
@dataclass
class LifecycleState:
    generation: int
    alive: bool
    start_time: float

@dataclass
class EvolutionState:
    mutation_rate: float
    last_score: float
    last_survived: bool

@dataclass
class ControlEvents:
    pause: threading.Event
    kill: threading.Event
    restart: threading.Event
```

**Files affected:** `ring1/telegram_bot.py`, `ring0/sentinel.py`, `ring1/task_executor.py`

---

### 4. `sentinel.run()` — 400-Line God Function

**Problem:** `run()` mixes initialization, main loop, success path, failure path, evolution trigger, crystallization, and cleanup — 6+ concerns in one function.

**Proposal:** Refactor into a `Sentinel` class:

```python
class Sentinel:
    def __init__(self, project_root: pathlib.Path):
        """Initialize all components."""
        ...

    def run(self):
        """Main loop — clean and concise."""
        self._install_signal_handler()
        try:
            while True:
                self._tick()
        except KeyboardInterrupt:
            log.info("Shutting down")
        finally:
            self._shutdown()

    def _tick(self):
        """Single generation lifecycle."""
        ...

    def _handle_success(self, generation, output, elapsed): ...
    def _handle_failure(self, generation, output, elapsed): ...
    def _try_evolve(self, generation, ...): ...
    def _shutdown(self): ...
```

**Files affected:** `ring0/sentinel.py`, `run.py` (minimal)

---

### 5. Factory Function Repetition

**Problem:** 10 `_create_*` functions in sentinel.py share identical structure:

```python
def _create_X(...):
    try:
        from ring1.X import X
        return X(...)
    except Exception as exc:
        log.debug("X not available: %s", exc)
        return None
```

**Proposal:** Extract generic helper:

```python
def _best_effort(label: str, factory: Callable[[], T]) -> T | None:
    try:
        return factory()
    except Exception as exc:
        log.debug("%s not available: %s", label, exc)
        return None

# Usage
memory_store = _best_effort("MemoryStore", lambda: MemoryStore(db_path))
```

**Files affected:** `ring0/sentinel.py`

---

### 6. LLM Client Duplicated Retry Logic

**Problem:** `ClaudeClient._call_api()` and `OpenAIClient._call_api()` have nearly identical retry/backoff code.

**Proposal:** Extract to `LLMClient` base class:

```python
class LLMClient(ABC):
    def _retry(self, fn: Callable, max_retries: int = 3) -> Any:
        """Shared retry with exponential backoff."""
        delay = 1.0
        for attempt in range(max_retries):
            try:
                return fn()
            except urllib.error.HTTPError as exc:
                if exc.code in self._RETRYABLE_CODES and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(...) from exc
```

**Files affected:** `ring1/llm_base.py`, `ring1/llm_client.py`, `ring1/llm_openai.py`

---

### 7. `TelegramBot` Monolith (600+ Lines)

**Problem:** 18+ command handlers, file uploads, inline keyboard, authorization — all in one class.

**Proposal:** Extract command dispatcher:

```python
class CommandDispatcher:
    """Routes /commands to handler methods."""
    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, cmd: str, handler: Callable): ...
    def dispatch(self, cmd: str, args: str, chat_id: int): ...
```

**Files affected:** `ring1/telegram_bot.py`

---

## Priority Matrix

| Priority | Item | Effort | Benefit |
|----------|------|--------|---------|
| **P0** | Store base class extraction | Small | Eliminate ~150 lines of duplication, unified SQLite behavior |
| **P0** | `sentinel.run()` → `Sentinel` class | Medium | Major readability & testability improvement |
| **P1** | Ring0Config typed object | Small | Eliminate typo risk, IDE support |
| **P1** | SentinelState split | Small | Clear responsibilities |
| **P2** | LLM retry extraction | Small | DRY |
| **P2** | Factory function consolidation | Small | DRY |
| **P3** | TelegramBot split | Medium | Maintainability |

---

## Class Inventory (Current State)

### Ring 0

| Module | Class | Role | Base |
|--------|-------|------|------|
| `fitness.py` | `FitnessTracker` | Fitness score storage & queries | — |
| `memory.py` | `MemoryStore` | Experiential memory storage | — |
| `heartbeat.py` | `HeartbeatMonitor` | File-based heartbeat monitor | — |
| `git_manager.py` | `GitManager` | Git CLI wrapper | — |
| `skill_store.py` | `SkillStore` | Skill persistence | — |
| `task_store.py` | `TaskStore` | Task persistence | — |
| `gene_pool.py` | `GenePool` | Top-N gene inheritance storage | — |
| `commit_watcher.py` | `CommitWatcher` | Polls git HEAD for auto-restart | — |
| `parameter_seed.py` | `EvolutionParams` | Deterministic param generation | `NamedTuple` |

### Ring 1

| Module | Class | Role | Base |
|--------|-------|------|------|
| `llm_base.py` | `LLMClient` | LLM provider interface | `ABC` |
| `llm_client.py` | `ClaudeClient` | Anthropic implementation | `LLMClient` |
| `llm_openai.py` | `OpenAIClient` | OpenAI/DeepSeek implementation | `LLMClient` |
| `config.py` | `Ring1Config` | Configuration container | `NamedTuple` |
| `evolver.py` | `Evolver` | LLM-driven code mutation | — |
| `crystallizer.py` | `Crystallizer` | Skill extraction | — |
| `task_executor.py` | `TaskExecutor` | User task processing | — |
| `subagent.py` | `SubagentManager` | Background LLM tasks | — |
| `telegram_bot.py` | `TelegramBot` | Bidirectional Telegram bot | — |
| `telegram_bot.py` | `SentinelState` | Shared state container | — |
| `telegram.py` | `TelegramNotifier` | Fire-and-forget notifications | — |
| `skill_runner.py` | `SkillRunner` | Subprocess management | — |
| `skill_portal.py` | `SkillPortal` | Web dashboard | — |
| `tool_registry.py` | `ToolRegistry` | LLM tool dispatch | — |
