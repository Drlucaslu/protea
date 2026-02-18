# Protea

Self-evolving artificial life system. The program is a living organism — it can self-restructure, self-reproduce, and self-evolve.

## Architecture

Three-ring design running on a single Mac mini:

- **Ring 0 (Sentinel)** — Immutable physics layer. Supervises Ring 2, performs heartbeat monitoring, git snapshots, rollback on failure, fitness tracking, and persistent storage (SQLite). Pure Python stdlib.
- **Ring 1 (Intelligence)** — LLM-driven evolution engine, task executor, Telegram bot, skill crystallizer, web portal, dashboard. Supports multiple LLM providers (Anthropic, OpenAI, DeepSeek, Qwen) for mutations, user tasks, and autonomous P1 work.
- **Ring 2 (Evolvable Code)** — The living code that evolves. Managed in its own git repo by Ring 0.

## Prerequisites

- Python 3.11+
- Git

## Quick Start

```bash
# Remote install (clones repo, creates venv, configures .env, runs tests)
curl -sSL https://raw.githubusercontent.com/Drlucaslu/protea/main/setup.sh | bash
cd protea && .venv/bin/python run.py
```

Or if you already cloned the repo:

```bash
bash setup.sh
.venv/bin/python run.py
```

## Project Structure

```
protea/
├── ring0/                      # Ring 0 — Sentinel (pure stdlib)
│   ├── sentinel.py             # Main supervisor loop
│   ├── heartbeat.py            # Ring 2 heartbeat monitoring
│   ├── git_manager.py          # Git snapshot + rollback
│   ├── fitness.py              # Fitness scoring + novelty + plateau detection
│   ├── memory.py               # Tiered memory store (hot/warm/cold) + vector search
│   ├── skill_store.py          # Crystallized skill store (SQLite)
│   ├── gene_pool.py            # Gene pool — top-N evolutionary inheritance (SQLite)
│   ├── task_store.py           # Task persistence store (SQLite)
│   ├── user_profile.py         # User profiling — topic extraction + interest decay
│   ├── parameter_seed.py       # Deterministic parameter generation
│   ├── resource_monitor.py     # CPU/memory/disk monitoring
│   └── commit_watcher.py       # Auto-restart on new commits
│
├── ring1/                      # Ring 1 — Intelligence layer
│   ├── config.py               # Ring 1 configuration loader
│   ├── evolver.py              # LLM-driven code evolution
│   ├── crystallizer.py         # Skill crystallization from surviving code
│   ├── llm_base.py             # LLM client ABC + factory
│   ├── llm_client.py           # Anthropic Claude client
│   ├── llm_openai.py           # OpenAI / DeepSeek / Qwen client
│   ├── task_executor.py        # P0 user tasks + P1 autonomous tasks
│   ├── dashboard.py            # System dashboard (memory, skills, profile, intent)
│   ├── memory_curator.py       # LLM-assisted memory curation (warm→cold)
│   ├── embeddings.py           # Embedding provider (OpenAI / NoOp)
│   ├── telegram_bot.py         # Telegram bot (commands + free-text)
│   ├── telegram.py             # Telegram notifier (one-way)
│   ├── skill_portal.py         # Web dashboard for skills
│   ├── skill_runner.py         # Skill process manager
│   ├── subagent.py             # Background task subagents
│   ├── registry_client.py      # Hub registry client
│   ├── tool_registry.py        # Tool dispatch framework
│   ├── tools/                  # Tool implementations
│   │   ├── filesystem.py       # read_file, write_file, edit_file, list_dir
│   │   ├── shell.py            # exec (sandboxed shell)
│   │   ├── web.py              # web_search, web_fetch
│   │   ├── message.py          # Progress messages to user
│   │   ├── skill.py            # run_skill, view_skill, edit_skill
│   │   ├── spawn.py            # Background task spawning
│   │   └── report.py           # Report generation
│   ├── web_tools.py            # DuckDuckGo + URL fetch
│   ├── pdf_utils.py            # PDF text extraction
│   └── prompts.py              # Evolution + crystallization prompts
│
├── ring2/                      # Ring 2 — Evolvable code
│   └── main.py                 # The living program
│
├── config/config.toml          # Configuration
├── data/                       # SQLite databases (auto-created)
├── tests/                      # 1098 tests
│   ├── test_ring0/             # Ring 0 unit tests
│   └── test_ring1/             # Ring 1 unit tests
├── .github/workflows/ci.yml   # CI (Python 3.11 + 3.13)
└── run.py                      # Entry point
```

## How It Works

1. **Sentinel** starts Ring 2 as a subprocess
2. Ring 2 writes a `.heartbeat` file every 2s; Sentinel checks freshness
3. If Ring 2 **survives** `max_runtime_sec`: record success, score fitness, crystallize skills, evolve code, advance generation
4. If Ring 2 **dies**: record failure, rollback to last good commit, evolve from rollback base, restart
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
| **Novelty** | 0.10 | Jaccard distance vs recent generations' output |
| Structured output | 0.10 | JSON, tables, key:value patterns |
| **Functional bonus** | 0.05 | Real I/O, HTTP, file operations detected |
| Error penalty | −0.10 | Traceback/error/exception lines |

**Adaptive evolution** saves tokens: when scores plateau (last N within epsilon), LLM evolution calls are skipped unless a user directive is pending. Persistent bugs (errors recurring across 2+ generations) are flagged as "must fix" in evolution prompts.

## Gene Pool — Evolutionary Inheritance

Evolved code patterns are preserved across generations via a **gene pool**. When Ring 2 survives, its source code is analysed (AST with regex fallback) to extract a compact gene summary (~200–500 tokens) containing class/function signatures and docstrings, with heartbeat boilerplate filtered out.

The top 100 genes (by fitness score) are stored in SQLite. During evolution, the best 2–3 gene summaries are injected into the LLM prompt as **Inherited Patterns**, so the evolver can build upon proven code instead of starting from scratch. On first startup, existing crystallized skills are backfilled into the gene pool.

## Multi-LLM Provider Support

Protea supports multiple LLM providers via a unified client interface:

| Provider | Config | Default Model |
|----------|--------|---------------|
| Anthropic (default) | `CLAUDE_API_KEY` env var | claude-sonnet-4-5 |
| OpenAI | `[ring1.llm]` section | gpt-4o |
| DeepSeek | `[ring1.llm]` section | deepseek-chat |
| Qwen (千问) | `[ring1.llm]` section | qwen3.5-plus |

To use a non-Anthropic provider, add to `config/config.toml`:

```toml
# DeepSeek
[ring1.llm]
provider = "deepseek"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-chat"

# Qwen 3.5 (千问 3.5)
[ring1.llm]
provider = "qwen"
api_key_env = "DASHSCOPE_API_KEY"
model = "qwen3.5-plus"
max_tokens = 8192
```

## Long-Term Memory

Three-tier memory system with importance scoring and selective forgetting:

| Tier | Retention | Description |
|------|-----------|-------------|
| Hot | Recent 10 generations | Active memories, full fidelity |
| Warm | 10–30 generations | Compacted by type, top-3 per group kept |
| Cold | 30–100 generations | LLM-curated (keep / summarize / discard) |
| Forgotten | >100 generations | Deleted if importance < 0.3 |

Importance is scored automatically by entry type (directive=0.9, crash_log=0.8, task=0.7, etc.). Compaction runs every 10 generations. The warm→cold transition uses **LLM-assisted curation** — a structured prompt asks the model which memories to keep, summarize, or discard, with rule-based fallback on failure.

**Semantic search** (optional): when an OpenAI embedding provider is configured, memories are stored with 256-dim vectors. Retrieval uses hybrid scoring (0.4 keyword + 0.6 cosine similarity).

## User Profiling

Keyword-based topic extraction from task history across 9 categories (coding, math, data, web, ai, system, creative, finance, research). Topic weights decay over time (`0.95^cycle`), so the profile reflects recent interests. The profile summary is injected into evolution prompts to guide mutations toward the user's active domains.

## Dashboard

Local web UI at `http://localhost:8899` with 5 pages:

| Page | Content |
|------|---------|
| Overview | Stat cards (memory/skills/intent/profile) + SVG fitness chart |
| Memory | Browsable table with tier/type filters |
| Skills | Card grid with usage counts and tags |
| Intent | Vertical timeline of evolution intents |
| Profile | Category bar chart + interaction stats |

All pages have JSON API counterparts (`/api/memory`, `/api/skills`, etc.). Auto-refreshes every 10 seconds.

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
| `/tasks` | Task queue + recent task history |
| `/memory` | Recent experiential memories |
| `/forget` | Clear all memories |
| `/skills` | List crystallized skills |
| `/skill <name>` | View skill details |
| `/run <name>` | Start a skill process |
| `/stop` | Stop running skill |
| `/running` | Running skill status |
| `/background` | Background subagent tasks |
| `/files` | List uploaded files |
| `/find <prefix>` | Search files by name |
| *free text* | Submit as P0 task to Claude |

## Web Portal

Skill Portal runs on a configurable HTTP port, providing a web dashboard for browsing, running, and monitoring crystallized skills.

## Configuration

All settings live in `config/config.toml`:

- **ring0**: heartbeat intervals, resource limits, evolution seed/cooldown, plateau detection, skill cap
- **ring1** (via env): `CLAUDE_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- **ring1.llm**: optional multi-provider LLM config (provider, model, api_key_env, api_url)
- **ring1.dashboard**: local dashboard (enabled, host, port)
- **ring1.embeddings**: optional vector search (provider, model, dimensions)
- P1 autonomous tasks: idle threshold, check interval, enable/disable

## Status

- [x] Ring 0 Sentinel — heartbeat, git, fitness (novelty + functional scoring), memory, skills, task persistence
- [x] Ring 1 Evolution — LLM mutations, adaptive evolution, crystallization, P0/P1 tasks
- [x] Multi-LLM — Anthropic, OpenAI, DeepSeek, Qwen via unified client
- [x] Telegram Bot — bidirectional commands + free-text tasks
- [x] Skill Portal — web dashboard
- [x] Dashboard — system state visualization (memory, skills, profile, intent)
- [x] Long-term memory — tiered storage, importance scoring, LLM-curated compaction
- [x] User profiling — interest extraction, weight decay, evolution guidance
- [x] Semantic search — optional vector embeddings + hybrid retrieval
- [x] CommitWatcher — auto-restart on deploy
- [x] Gene Pool — AST-based code inheritance across generations
- [x] Task persistence — survives restarts via SQLite
- [x] CI — GitHub Actions (Python 3.11 + 3.13)
- [x] 1098 tests passing

## Registry

Protea skills can be published to and installed from [protea-hub](https://github.com/lianglu/protea-hub), a skill registry deployed on Railway.
