"""Telegram Bot — bidirectional interaction via getUpdates long polling.

Pure stdlib (urllib.request + json + threading).  Runs as a daemon thread
alongside the Sentinel main loop.  Errors never propagate to the caller.
"""

from __future__ import annotations

import json
import logging
import pathlib
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("protea.telegram_bot")

_API_BASE = "https://api.telegram.org/bot{token}/{method}"


# ---------------------------------------------------------------------------
# Shared state between Sentinel thread and Bot thread
# ---------------------------------------------------------------------------

class SentinelState:
    """Thread-safe container for Sentinel runtime state.

    Sentinel writes fields under the lock each loop iteration.
    Bot reads fields under the lock on command.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pause_event = threading.Event()
        self.kill_event = threading.Event()
        # Mutable fields — protected by self.lock
        self.generation: int = 0
        self.start_time: float = time.time()
        self.alive: bool = False
        self.mutation_rate: float = 0.0
        self.max_runtime_sec: float = 0.0
        self.last_score: float = 0.0
        self.last_survived: bool = False
        # Task / scheduling fields (Phase 3.5)
        self.task_queue: queue.Queue = queue.Queue()
        self.p0_active = threading.Event()    # P0 task executing
        self.p0_event = threading.Event()     # pulse to wake sentinel
        self.evolution_directive: str = ""    # user directive (lock-protected)
        self.memory_store = None              # set by Sentinel after creation

    def snapshot(self) -> dict:
        """Return a consistent copy of all fields."""
        with self.lock:
            return {
                "generation": self.generation,
                "start_time": self.start_time,
                "alive": self.alive,
                "mutation_rate": self.mutation_rate,
                "max_runtime_sec": self.max_runtime_sec,
                "last_score": self.last_score,
                "last_survived": self.last_survived,
                "paused": self.pause_event.is_set(),
                "p0_active": self.p0_active.is_set(),
                "evolution_directive": self.evolution_directive,
                "task_queue_size": self.task_queue.qsize(),
            }


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A user task submitted via free-text Telegram message."""
    text: str
    chat_id: str
    created_at: float = field(default_factory=time.time)
    task_id: str = field(default_factory=lambda: f"t-{int(time.time() * 1000) % 1_000_000}")


# ---------------------------------------------------------------------------
# Telegram Bot
# ---------------------------------------------------------------------------

class TelegramBot:
    """Telegram Bot that reads commands via getUpdates long polling."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        state: SentinelState,
        fitness,
        ring2_path: pathlib.Path,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.state = state
        self.fitness = fitness
        self.ring2_path = ring2_path
        self._offset: int = 0
        self._running = threading.Event()
        self._running.set()

    # -- low-level API helpers --

    def _api_call(self, method: str, params: dict | None = None) -> dict | None:
        """Call a Telegram Bot API method.  Returns parsed JSON or None."""
        url = _API_BASE.format(token=self.bot_token, method=method)
        payload = json.dumps(params or {}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = 35 if method == "getUpdates" else 10
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if body.get("ok"):
                    return body
                return None
        except Exception:
            log.debug("API call %s failed", method, exc_info=True)
            return None

    def _get_updates(self) -> list[dict]:
        """Fetch new updates via long polling."""
        params = {"offset": self._offset, "timeout": 30}
        result = self._api_call("getUpdates", params)
        if not result:
            return []
        updates = result.get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    def _send_reply(self, text: str) -> None:
        """Send a text reply (fire-and-forget)."""
        self._api_call("sendMessage", {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        })

    def _is_authorized(self, update: dict) -> bool:
        """Check if the update comes from the authorized chat."""
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        return str(chat.get("id", "")) == self.chat_id

    # -- command handlers --

    def _cmd_status(self) -> str:
        snap = self.state.snapshot()
        elapsed = time.time() - snap["start_time"]
        status = "PAUSED" if snap["paused"] else ("ALIVE" if snap["alive"] else "DEAD")
        return (
            f"*Protea Status*\n"
            f"Generation: {snap['generation']}\n"
            f"Status: {status}\n"
            f"Uptime: {elapsed:.0f}s\n"
            f"Mutation rate: {snap['mutation_rate']:.2f}\n"
            f"Max runtime: {snap['max_runtime_sec']:.0f}s"
        )

    def _cmd_history(self) -> str:
        rows = self.fitness.get_history(limit=10)
        if not rows:
            return "No history yet."
        lines = ["*Recent 10 generations:*"]
        for r in rows:
            surv = "OK" if r["survived"] else "FAIL"
            lines.append(
                f"Gen {r['generation']}  score={r['score']:.2f}  "
                f"{surv}  {r['runtime_sec']:.0f}s"
            )
        return "\n".join(lines)

    def _cmd_top(self) -> str:
        rows = self.fitness.get_best(n=5)
        if not rows:
            return "No fitness data yet."
        lines = ["*Top 5 generations:*"]
        for r in rows:
            surv = "OK" if r["survived"] else "FAIL"
            lines.append(
                f"Gen {r['generation']}  score={r['score']:.2f}  "
                f"{surv}  `{r['commit_hash'][:8]}`"
            )
        return "\n".join(lines)

    def _cmd_code(self) -> str:
        code_path = self.ring2_path / "main.py"
        try:
            source = code_path.read_text()
        except FileNotFoundError:
            return "ring2/main.py not found."
        if len(source) > 3000:
            source = source[:3000] + "\n... (truncated)"
        return f"```python\n{source}\n```"

    def _cmd_pause(self) -> str:
        if self.state.pause_event.is_set():
            return "Already paused."
        self.state.pause_event.set()
        return "Evolution paused."

    def _cmd_resume(self) -> str:
        if not self.state.pause_event.is_set():
            return "Not paused."
        self.state.pause_event.clear()
        return "Evolution resumed."

    def _cmd_kill(self) -> str:
        self.state.kill_event.set()
        return "Kill signal sent — Ring 2 will restart."

    def _cmd_help(self) -> str:
        return (
            "*Protea Bot Commands:*\n"
            "/status — current generation, uptime, state\n"
            "/history — recent 10 generations\n"
            "/top — top 5 by fitness\n"
            "/code — current Ring 2 source\n"
            "/pause — pause evolution loop\n"
            "/resume — resume evolution loop\n"
            "/kill — restart Ring 2 (no generation advance)\n"
            "/direct <text> — set evolution directive\n"
            "/tasks — show task queue and directive\n"
            "/memory — view recent memories\n"
            "/forget — clear all memories\n\n"
            "Or send any text to ask Protea a question (P0 task)."
        )

    def _cmd_direct(self, full_text: str) -> str:
        """Set an evolution directive from /direct <text>."""
        # Strip the /direct prefix (and optional @botname)
        parts = full_text.strip().split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            return "Usage: /direct <directive text>\nExample: /direct 变成贪吃蛇"
        directive = parts[1].strip()
        with self.state.lock:
            self.state.evolution_directive = directive
        self.state.p0_event.set()  # wake sentinel
        return f"Evolution directive set: {directive}"

    def _cmd_tasks(self) -> str:
        """Show task queue status and current directive."""
        snap = self.state.snapshot()
        lines = ["*Task Queue Status:*"]
        lines.append(f"Queued tasks: {snap['task_queue_size']}")
        lines.append(f"P0 active: {'Yes' if snap['p0_active'] else 'No'}")
        directive = snap["evolution_directive"]
        lines.append(f"Directive: {directive if directive else '(none)'}")
        return "\n".join(lines)

    def _cmd_memory(self) -> str:
        """Show recent memories."""
        ms = self.state.memory_store
        if not ms:
            return "Memory not available."
        entries = ms.get_recent(5)
        if not entries:
            return "No memories yet."
        lines = [f"*Recent Memories ({ms.count()} total):*"]
        for e in entries:
            lines.append(
                f"[Gen {e['generation']}, {e['entry_type']}] {e['content']}"
            )
        return "\n".join(lines)

    def _cmd_forget(self) -> str:
        """Clear all memories."""
        ms = self.state.memory_store
        if not ms:
            return "Memory not available."
        ms.clear()
        return "All memories cleared."

    def _enqueue_task(self, text: str, chat_id: str) -> str:
        """Create a Task, enqueue it, pulse p0_event, return ack."""
        task = Task(text=text, chat_id=chat_id)
        self.state.task_queue.put(task)
        self.state.p0_event.set()  # wake sentinel for P0 scheduling
        return f"Got it — processing your request ({task.task_id})..."

    # -- dispatch --

    _COMMANDS: dict[str, str] = {
        "/status": "_cmd_status",
        "/history": "_cmd_history",
        "/top": "_cmd_top",
        "/code": "_cmd_code",
        "/pause": "_cmd_pause",
        "/resume": "_cmd_resume",
        "/kill": "_cmd_kill",
        "/help": "_cmd_help",
        "/start": "_cmd_help",
        "/tasks": "_cmd_tasks",
        "/memory": "_cmd_memory",
        "/forget": "_cmd_forget",
    }

    def _handle_command(self, text: str, chat_id: str = "") -> str:
        """Dispatch a command or free-text message and return the response."""
        stripped = text.strip()
        if not stripped:
            return self._cmd_help()

        # Free text (not a command) → enqueue as P0 task
        if not stripped.startswith("/"):
            return self._enqueue_task(stripped, chat_id)

        # /direct needs special handling (passes full text)
        first_word = stripped.split()[0].lower().split("@")[0]
        if first_word == "/direct":
            return self._cmd_direct(stripped)

        # Standard command dispatch
        method_name = self._COMMANDS.get(first_word)
        if method_name is None:
            return self._cmd_help()
        return getattr(self, method_name)()

    # -- main loop --

    def run(self) -> None:
        """Long-polling loop.  Intended to run in a daemon thread."""
        log.info("Telegram bot started (chat_id=%s)", self.chat_id)
        while self._running.is_set():
            try:
                updates = self._get_updates()
                for update in updates:
                    try:
                        if not self._is_authorized(update):
                            log.debug("Ignoring unauthorized update")
                            continue
                        msg = update.get("message", {})
                        text = msg.get("text", "")
                        if not text:
                            continue
                        msg_chat_id = str(msg.get("chat", {}).get("id", ""))
                        reply = self._handle_command(text, chat_id=msg_chat_id)
                        self._send_reply(reply)
                    except Exception:
                        log.debug("Error handling update", exc_info=True)
            except Exception:
                log.debug("Error in polling loop", exc_info=True)
                # Back off on repeated errors.
                if self._running.is_set():
                    time.sleep(5)
        log.info("Telegram bot stopped")

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._running.clear()


# ---------------------------------------------------------------------------
# Factory + thread launcher
# ---------------------------------------------------------------------------

def create_bot(config, state: SentinelState, fitness, ring2_path: pathlib.Path) -> TelegramBot | None:
    """Create a TelegramBot from Ring1Config, or None if disabled/missing."""
    if not config.telegram_enabled:
        return None
    if not config.telegram_bot_token or not config.telegram_chat_id:
        log.warning("Telegram bot: enabled but token/chat_id missing — disabled")
        return None
    return TelegramBot(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
        state=state,
        fitness=fitness,
        ring2_path=ring2_path,
    )


def start_bot_thread(bot: TelegramBot) -> threading.Thread:
    """Start the bot in a daemon thread and return the thread handle."""
    thread = threading.Thread(target=bot.run, name="telegram-bot", daemon=True)
    thread.start()
    return thread
