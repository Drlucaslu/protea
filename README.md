# Protea

Self-evolving artificial life system. The program is a living organism — it can self-restructure, self-reproduce, and self-evolve.

## Architecture

Three-ring design running on a single Mac mini:

- **Ring 0 (Sentinel)** — Immutable physics layer. Supervises Ring 2, performs heartbeat monitoring, git snapshots, rollback on failure, fitness tracking, and persistent storage (SQLite). Pure Python stdlib.
- **Ring 1 (Intelligence)** — LLM-driven evolution engine, task executor, Telegram bot, skill crystallizer, web portal. Calls Claude API for mutations, user tasks, and autonomous P1 work.
- **Ring 2 (Evolvable Code)** — The living code that evolves. Managed in its own git repo by Ring 0.

## Prerequisites

- Python 3.11+
- Git

## Quick Start

```bash
# Remote install (clones repo, creates venv, .env, ring2 git, runs tests)
curl -sSL https://raw.githubusercontent.com/Drlucaslu/protea/main/setup.sh | bash
cd protea && source .venv/bin/activate && python run.py
```

Or if you already cloned the repo:

```bash
bash setup.sh
source .venv/bin/activate && python run.py
```

## Project Structure

```
protea/
├── ring0/                      # Ring 0 — Sentinel (pure stdlib)
│   ├── sentinel.py             # Main supervisor loop
│   ├── heartbeat.py            # Ring 2 heartbeat monitoring
│   ├── git_manager.py          # Git snapshot + rollback
│   ├── fitness.py              # Fitness scoring (SQLite)
│   ├── memory.py               # Experiential memory store (SQLite)
│   ├── skill_store.py          # Crystallized skill store (SQLite)
│   ├── task_store.py           # Task persistence store (SQLite)
│   ├── parameter_seed.py       # Deterministic parameter generation
│   ├── resource_monitor.py     # CPU/memory/disk monitoring
│   └── commit_watcher.py       # Auto-restart on new commits
│
├── ring1/                      # Ring 1 — Intelligence layer
│   ├── config.py               # Ring 1 configuration loader
│   ├── evolver.py              # LLM-driven code evolution
│   ├── crystallizer.py         # Skill crystallization from surviving code
│   ├── llm_client.py           # Claude API client
│   ├── task_executor.py        # P0 user tasks + P1 autonomous tasks
│   ├── telegram_bot.py         # Telegram bot (commands + free-text)
│   ├── telegram.py             # Telegram notifier (one-way)
│   ├── skill_portal.py         # Web dashboard for skills
│   ├── skill_runner.py         # Skill process manager
│   ├── subagent.py             # Background task subagents
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
│   └── prompts.py              # Evolution prompt templates
│
├── ring2/                      # Ring 2 — Evolvable code
│   └── main.py                 # The living program
│
├── config/config.toml          # Configuration
├── data/                       # SQLite databases (auto-created)
├── tests/                      # 797+ tests
│   ├── test_ring0/             # Ring 0 unit tests
│   └── test_ring1/             # Ring 1 unit tests
└── run.py                      # Entry point
```

## How It Works

1. **Sentinel** starts Ring 2 as a subprocess
2. Ring 2 writes a `.heartbeat` file every 2s; Sentinel checks freshness
3. If Ring 2 **survives** `max_runtime_sec`: record success, crystallize skills, evolve code, advance generation
4. If Ring 2 **dies**: record failure, rollback to last good commit, evolve from rollback base, restart
5. Each generation gets deterministic parameters from a seeded PRNG
6. **CommitWatcher** detects new git commits and triggers `os.execv()` restart
7. **TaskStore** persists queued tasks to SQLite — they survive restarts

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

- **ring0**: heartbeat intervals, resource limits, evolution seed, skill cap
- **ring1** (via env): `CLAUDE_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- P1 autonomous tasks: idle threshold, check interval, enable/disable

## Status

- [x] Ring 0 Sentinel — heartbeat, git, fitness, memory, skills, task persistence
- [x] Ring 1 Evolution — LLM mutations, crystallization, P0/P1 tasks
- [x] Telegram Bot — bidirectional commands + free-text tasks
- [x] Skill Portal — web dashboard
- [x] CommitWatcher — auto-restart on deploy
- [x] Task persistence — survives restarts via SQLite
- [x] 797+ tests passing

## Registry

Protea skills can be published to and installed from [protea-hub](https://github.com/lianglu/protea-hub), a skill registry deployed on Railway.
