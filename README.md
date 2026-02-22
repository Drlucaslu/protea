# Protea

Self-evolving artificial life system. The program is a living organism — it can self-restructure, self-reproduce, and self-evolve.

## Architecture

Three-ring design running on a single Mac mini:

- **Ring 0 (Sentinel)** — Immutable physics layer. Supervises Ring 2, performs heartbeat monitoring, git snapshots, rollback on failure, fitness tracking, and persistent storage (SQLite). Pure Python stdlib.
- **Ring 1 (Intelligence)** — LLM-driven evolution engine, task executor, Telegram/Matrix bots, skill crystallizer, dashboard. Supports multiple LLM providers (Anthropic, OpenAI, DeepSeek, Qwen).
- **Ring 2 (Evolvable Code)** — The living code that evolves each generation. Managed by Ring 0.

## Prerequisites

- Python 3.11+
- Git

## Quick Start

```bash
# Remote install (clones repo, creates venv, configures .env, runs tests)
curl -sSL https://raw.githubusercontent.com/Drlucaslu/protea/main/setup.sh | bash
cd protea && .venv/bin/python run.py
```

Or manually:

```bash
git clone https://github.com/Drlucaslu/protea.git && cd protea
bash setup.sh
.venv/bin/python run.py
```

For background operation with automatic watchdog restart:

```bash
bash run_with_nohup.sh
```

## Project Structure

```
protea/
├── ring0/                      # Ring 0 — Sentinel (pure stdlib)
│   ├── sentinel.py             # Main supervisor loop
│   ├── heartbeat.py            # Ring 2 heartbeat monitoring
│   ├── git_manager.py          # Git snapshot + rollback
│   ├── fitness.py              # Fitness scoring + plateau detection
│   ├── memory.py               # Tiered memory store (hot/warm/cold) + vector search
│   ├── skill_store.py          # Crystallized skill store
│   ├── gene_pool.py            # Gene pool — evolutionary inheritance
│   ├── task_store.py           # Task persistence store
│   ├── scheduled_task_store.py # Cron / one-shot scheduled tasks
│   ├── user_profile.py         # User profiling — topic extraction + interest decay
│   ├── preference_store.py     # Structured preference store
│   ├── evolution_intent.py     # Intent classification + blast radius
│   ├── commit_watcher.py       # Auto-restart on new commits
│   ├── cron.py                 # Cron expression parser
│   ├── resource_monitor.py     # CPU/memory/disk monitoring
│   └── sqlite_store.py         # Base SQLite store mixin
│
├── ring1/                      # Ring 1 — Intelligence layer
│   ├── evolver.py              # LLM-driven code evolution + blast radius gate
│   ├── prompts.py              # Evolution + crystallization prompts
│   ├── crystallizer.py         # Skill crystallization from surviving code
│   ├── auto_crystallizer.py    # Automatic crystallization triggers
│   ├── llm_base.py             # LLM client ABC + factory + thread-safe timeout
│   ├── llm_client.py           # Anthropic Claude client
│   ├── llm_openai.py           # OpenAI / DeepSeek / Qwen client
│   ├── task_executor.py        # P0 user tasks + P1 autonomous tasks
│   ├── task_generator.py       # Autonomous task generation
│   ├── preference_extractor.py # Implicit preference signal extraction
│   ├── habit_detector.py       # User habit pattern detection
│   ├── proactive_loop.py       # Proactive suggestion engine
│   ├── directive_generator.py  # Dynamic evolution direction
│   ├── convergence_detector.py # Evolution convergence detection
│   ├── memory_curator.py       # LLM-assisted memory curation
│   ├── embeddings.py           # Embedding provider (OpenAI / local hash)
│   ├── telegram_bot.py         # Telegram bot (commands + free-text)
│   ├── telegram.py             # Telegram notifier (one-way)
│   ├── matrix_bot.py           # Matrix bot via Client-Server API
│   ├── dashboard.py            # Web dashboard (memory, skills, profile, intent)
│   ├── skill_portal.py         # Skill web UI
│   ├── skill_runner.py         # Skill process manager
│   ├── skill_sandbox.py        # Skill venv + dependency manager
│   ├── skill_sync.py           # Skill ↔ profile sync
│   ├── skill_validator.py      # Skill code validation
│   ├── registry_client.py      # Hub registry client
│   ├── subagent.py             # Background task subagents
│   ├── tool_registry.py        # Tool dispatch framework
│   ├── tools/                  # Tool implementations
│   │   ├── filesystem.py       # read_file, write_file, edit_file, list_dir
│   │   ├── shell.py            # exec (sandboxed shell)
│   │   ├── web.py              # web_search, web_fetch
│   │   ├── message.py          # Progress messages to user
│   │   ├── skill.py            # run_skill, view_skill, edit_skill
│   │   ├── spawn.py            # Background task spawning
│   │   ├── schedule.py         # Scheduled task management
│   │   ├── send_file.py        # File sending to chat
│   │   ├── report.py           # Report generation
│   │   └── progress_monitor.py # Task progress monitoring
│   ├── web_tools.py            # DuckDuckGo + URL fetch
│   └── pdf_utils.py            # PDF text extraction
│
├── ring2/                      # Ring 2 — Evolvable code
│   └── main.py                 # The living program (auto-evolved, never commit manually)
│
├── config/config.toml          # Configuration
├── data/                       # SQLite databases (auto-created)
├── tests/                      # 1952 tests
├── run.py                      # Entry point
├── run_with_nohup.sh           # Background launcher with watchdog
└── setup.sh                    # One-step installer
```

## How It Works

1. **Sentinel** starts Ring 2 as a subprocess
2. Ring 2 writes a `.heartbeat` file every 2s; Sentinel checks freshness
3. If Ring 2 **survives** `max_runtime_sec`: score fitness, crystallize skills, evolve code, advance generation
4. If Ring 2 **dies**: rollback to last good commit, evolve from rollback base, restart
5. Each generation gets deterministic parameters from a seeded PRNG
6. **CommitWatcher** detects new git commits and triggers `os.execv()` restart
7. **TaskStore** persists queued tasks to SQLite — they survive restarts

## Evolution & Fitness

Fitness is scored 0.0–1.0 with six components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Base survival | 0.50 | Survived max_runtime_sec |
| Output volume | 0.10 | Meaningful non-empty lines (saturates at 50) |
| Output diversity | 0.10 | Unique lines / total lines |
| Novelty | 0.05 | Jaccard distance vs recent generations' output |
| Structured output | 0.10 | JSON, tables, key:value patterns |
| Functional bonus | 0.05 | Real I/O, HTTP, file operations detected |
| Error penalty | −0.10 | Traceback/error/exception lines |

### Incremental Evolution

Evolution prefers **incremental modification** over full rewrites. A blast radius gate in the evolver rejects full rewrites (>70% lines changed) when the code is working fine (`optimize` intent). Larger changes are allowed when the code is broken (`repair`), stagnant (`explore`), or a user directive is pending (`adapt`).

### Evolution Intent

Each evolution is classified by intent (priority order):

1. **adapt** — User directive pending (highest priority)
2. **repair** — Code crashed or has persistent errors
3. **explore** — Scores plateaued
4. **optimize** — Code survived, make incremental improvements

When scores plateau and no directive is pending, LLM evolution calls are **skipped** to save tokens.

## Gene Pool

Evolved code patterns are preserved across generations via a **gene pool**. When Ring 2 survives, its source code is analysed (AST) to extract a compact gene summary (~200–500 tokens). The top 100 genes (by fitness score) are stored in SQLite.

During evolution, the best 2–3 gene summaries are injected into the LLM prompt as **Inherited Patterns**. During task execution, genes are filtered by a **semantic relevance threshold** (`min_semantic=1.0`) to avoid injecting unrelated patterns.

## Multi-LLM Provider Support

| Provider | Config | Default Model |
|----------|--------|---------------|
| Anthropic (default) | `CLAUDE_API_KEY` env var | claude-sonnet-4-5 |
| OpenAI | `[ring1.llm]` section | gpt-4o |
| DeepSeek | `[ring1.llm]` section | deepseek-chat |
| Qwen (千问) | `[ring1.llm]` section | qwen3.5-plus |

```toml
[ring1.llm]
provider = "qwen"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-plus"
max_tokens = 8192
```

Each HTTP request runs in a daemon thread with a 90-second hard timeout to guard against socket hangs.

## Long-Term Memory

Three-tier memory system with importance scoring and selective forgetting:

| Tier | Retention | Description |
|------|-----------|-------------|
| Hot | Recent 10 generations | Active memories, full fidelity |
| Warm | 10–30 generations | Compacted by type, top-3 per group kept |
| Cold | 30–100 generations | LLM-curated (keep / summarize / discard) |
| Forgotten | >100 generations | Deleted if importance < 0.3 |

Optional **semantic search**: when an embedding provider is configured, memories are stored with 256-dim vectors. Retrieval uses hybrid scoring (0.4 keyword + 0.6 cosine similarity).

## Dashboard

Local web UI at `http://localhost:8899`:

| Page | Content |
|------|---------|
| Overview | Stat cards + SVG fitness chart |
| Memory | Browsable table with tier/type filters |
| Skills | Card grid with usage counts and tags |
| Intent | Vertical timeline of evolution intents |
| Profile | Category bar chart + interaction stats |

All pages have JSON API counterparts (`/api/memory`, `/api/skills`, etc.).

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Status panel (generation, uptime, executor health) |
| `/history` | Recent 10 generations |
| `/top` | Top 5 fitness scores |
| `/code` | View current Ring 2 source |
| `/pause` / `/resume` | Pause/resume evolution |
| `/kill` | Restart Ring 2 |
| `/direct <text>` | Set evolution directive |
| `/tasks` | Task queue + recent history |
| `/memory` | Recent memories |
| `/forget` | Clear all memories |
| `/skills` | List crystallized skills |
| `/skill <name>` | View skill details |
| `/run <name>` | Start a skill process |
| `/stop` | Stop running skill |
| `/running` | Running skill status |
| `/background` | Background subagent tasks |
| `/files` | List uploaded files |
| `/find <prefix>` | Search files by name |
| *free text* | Submit as P0 task to LLM |

## Configuration

All settings in `config/config.toml`:

- **ring0**: heartbeat intervals, resource limits, evolution seed/cooldown, plateau detection, skill cap
- **ring1**: `CLAUDE_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (env vars)
- **ring1.llm**: multi-provider LLM config (provider, model, api_key_env, api_url)
- **ring1.dashboard**: local dashboard (enabled, host, port)
- **ring1.embeddings**: vector search (provider, model, dimensions)

## Registry

Protea skills can be published to and installed from [protea-hub](https://github.com/lianglu/protea-hub).
