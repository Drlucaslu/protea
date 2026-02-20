"""Matrix Bot — bidirectional interaction via Client-Server API long-polling.

Pure stdlib (urllib.request + json + threading).  Runs as a daemon thread
alongside the Sentinel main loop.  Errors never propagate to the caller.

Uses the Matrix Client-Server API v1.11:
  - Sync:  GET /_matrix/client/v3/sync?since={token}&timeout=30000
  - Send:  PUT /_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn_id}
  - Auth:  Authorization: Bearer {access_token}
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request

log = logging.getLogger("protea.matrix_bot")


class MatrixBot:
    """Matrix bot that reads commands via /sync long polling."""

    def __init__(
        self,
        homeserver: str,
        access_token: str,
        room_id: str,
        state,
    ) -> None:
        self.homeserver = homeserver.rstrip("/")
        self.access_token = access_token
        self.room_id = room_id
        self.state = state
        self._since: str = ""
        self._running = threading.Event()
        self._running.set()
        self._txn_id: int = int(time.time() * 1000)
        # Track our own user_id to ignore our own messages
        self._user_id: str = ""

    # -- low-level API helpers --

    def _api_request(self, method: str, path: str, body: dict | None = None, timeout: int = 35) -> dict | None:
        """Make an authenticated Matrix API request."""
        url = f"{self.homeserver}{path}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            log.debug("Matrix API %s %s failed", method, path, exc_info=True)
            return None

    def _sync(self) -> list[dict]:
        """Perform a /sync request and return room timeline events."""
        params = "timeout=30000"
        if self._since:
            params += f"&since={self._since}"
        else:
            # On first sync, use a filter to only get recent messages
            params += "&filter={\"room\":{\"timeline\":{\"limit\":0}}}"

        result = self._api_request("GET", f"/_matrix/client/v3/sync?{params}", timeout=35)
        if not result:
            return []

        self._since = result.get("next_batch", self._since)

        # Extract timeline events from our room
        rooms = result.get("rooms", {}).get("join", {})
        room_data = rooms.get(self.room_id, {})
        events = room_data.get("timeline", {}).get("events", [])
        return events

    def _send_reply(self, text: str) -> None:
        """Send a text message to the configured room."""
        self._txn_id += 1
        path = f"/_matrix/client/v3/rooms/{self.room_id}/send/m.room.message/{self._txn_id}"
        body = {
            "msgtype": "m.text",
            "body": text,
        }
        self._api_request("PUT", path, body=body, timeout=10)

    def _whoami(self) -> str:
        """Get our own user_id to filter out our own messages."""
        result = self._api_request("GET", "/_matrix/client/v3/account/whoami", timeout=10)
        if result:
            return result.get("user_id", "")
        return ""

    # -- command handlers --

    def _cmd_status(self) -> str:
        snap = self.state.snapshot()
        elapsed = time.time() - snap["start_time"]
        raw = "PAUSED" if snap["paused"] else ("ALIVE" if snap["alive"] else "DEAD")
        lines = [
            "Protea Status",
            f"Generation: {snap['generation']}",
            f"Status: {raw}",
            f"Uptime: {elapsed:.0f}s",
            f"Queued tasks: {snap['task_queue_size']}",
        ]
        executor_alive = snap.get("executor_alive", False)
        lines.append(f"Executor: {'online' if executor_alive else 'offline'}")
        return "\n".join(lines)

    def _cmd_tasks(self) -> str:
        snap = self.state.snapshot()
        lines = [
            "Task Queue",
            f"Queued: {snap['task_queue_size']}",
            f"P0 active: {'yes' if snap['p0_active'] else 'no'}",
        ]
        ts = self.state.task_store
        if ts:
            recent = ts.get_recent(5)
            if recent:
                lines.append("\nRecent:")
                for t in recent:
                    icon = {"pending": "P", "executing": "R", "completed": "D", "failed": "F"}.get(t["status"], "?")
                    text_preview = t["text"][:40] + ("..." if len(t["text"]) > 40 else "")
                    lines.append(f"[{icon}] {t['task_id']}: {text_preview}")
        return "\n".join(lines)

    def _cmd_skills(self) -> str:
        ss = self.state.skill_store
        if not ss:
            return "Skills module not available."
        skills = ss.get_active(500)
        if not skills:
            return "No saved skills."
        lines = [f"Skills ({len(skills)}):"]
        for s in skills:
            lines.append(f"- {s['name']}: {s['description']} (used {s['usage_count']}x)")
        return "\n".join(lines)

    def _cmd_calendar(self) -> str:
        ss = self.state.scheduled_store
        if not ss:
            return "Schedule module not available."
        tasks = ss.get_all()
        if not tasks:
            return "No scheduled tasks."

        from datetime import datetime
        try:
            from ring0.cron import describe as _cron_desc
        except ImportError:
            _cron_desc = lambda x: x

        lines = ["Calendar:"]
        for t in tasks:
            icon = "[ON]" if t["enabled"] else "[OFF]"
            name = t["name"]
            if t["schedule_type"] == "cron":
                schedule_desc = _cron_desc(t["cron_expr"])
            else:
                schedule_desc = f"once {t['cron_expr']}"
            next_at = ""
            if t["next_run_at"]:
                next_dt = datetime.fromtimestamp(t["next_run_at"])
                next_at = f" — next: {next_dt.strftime('%Y-%m-%d %H:%M')}"
            disabled_tag = " (disabled)" if not t["enabled"] else ""
            lines.append(f"{icon} {name} — {schedule_desc}{next_at}{disabled_tag}")
        return "\n".join(lines)

    def _cmd_schedule(self, full_text: str) -> str:
        """Handle /schedule subcommands via shared state.scheduled_store."""
        ss = self.state.scheduled_store
        if not ss:
            return "Schedule module not available."

        parts = full_text.strip().split(None, 1)
        args = parts[1].strip() if len(parts) > 1 else ""

        if not args or args == "list":
            return self._cmd_calendar()

        tokens = args.split(None, 2)
        subcmd = tokens[0].lower()

        if subcmd == "remove":
            name = tokens[1] if len(tokens) > 1 else ""
            if not name:
                return "Usage: /schedule remove <name>"
            task = ss.get_by_name(name)
            if not task:
                return f"Task '{name}' not found."
            ss.remove(task["schedule_id"])
            return f"Removed: {name}"

        if subcmd == "enable":
            name = tokens[1] if len(tokens) > 1 else ""
            if not name:
                return "Usage: /schedule enable <name>"
            task = ss.get_by_name(name)
            if not task:
                return f"Task '{name}' not found."
            ss.enable(task["schedule_id"])
            return f"Enabled: {name}"

        if subcmd == "disable":
            name = tokens[1] if len(tokens) > 1 else ""
            if not name:
                return "Usage: /schedule disable <name>"
            task = ss.get_by_name(name)
            if not task:
                return f"Task '{name}' not found."
            ss.disable(task["schedule_id"])
            return f"Disabled: {name}"

        return (
            "Usage:\n"
            "/schedule list — list all tasks\n"
            "/schedule remove <name> — remove task\n"
            "/schedule enable <name> — enable task\n"
            "/schedule disable <name> — disable task"
        )

    def _cmd_help(self) -> str:
        return (
            "Protea Matrix Bot Commands:\n"
            "/status — system status\n"
            "/tasks — task queue\n"
            "/skills — skill list\n"
            "/schedule — manage scheduled tasks\n"
            "/calendar — view scheduled tasks\n"
            "/help — this help\n\n"
            "Send any text to create a P0 task."
        )

    def _enqueue_task(self, text: str) -> str:
        from ring1.telegram_bot import Task
        task = Task(text=text, chat_id="")
        ts = self.state.task_store
        if ts:
            try:
                ts.add(task.task_id, task.text, task.chat_id, task.created_at)
            except Exception:
                log.debug("Failed to persist task", exc_info=True)
        self.state.task_queue.put(task)
        self.state.p0_event.set()
        return f"Received — processing ({task.task_id})..."

    # -- dispatch --

    def _handle_command(self, text: str) -> str:
        """Dispatch a command or free-text message and return the response."""
        stripped = text.strip()
        if not stripped:
            return self._cmd_help()

        if not stripped.startswith("/"):
            return self._enqueue_task(stripped)

        first_word = stripped.split()[0].lower()

        if first_word == "/status":
            return self._cmd_status()
        if first_word == "/tasks":
            return self._cmd_tasks()
        if first_word == "/skills":
            return self._cmd_skills()
        if first_word == "/calendar":
            return self._cmd_calendar()
        if first_word == "/schedule":
            return self._cmd_schedule(stripped)
        if first_word == "/help":
            return self._cmd_help()

        return self._cmd_help()

    # -- main loop --

    def run(self) -> None:
        """Long-polling sync loop.  Intended to run in a daemon thread."""
        log.info("Matrix bot started (homeserver=%s, room=%s)", self.homeserver, self.room_id)

        # Get our own user_id
        self._user_id = self._whoami()
        if self._user_id:
            log.info("Matrix bot user_id: %s", self._user_id)

        while self._running.is_set():
            try:
                events = self._sync()
                for event in events:
                    try:
                        # Only handle m.room.message events
                        if event.get("type") != "m.room.message":
                            continue
                        # Skip our own messages
                        sender = event.get("sender", "")
                        if sender == self._user_id:
                            continue
                        content = event.get("content", {})
                        if content.get("msgtype") != "m.text":
                            continue
                        body = content.get("body", "")
                        if not body:
                            continue

                        reply = self._handle_command(body)
                        if reply:
                            self._send_reply(reply)
                    except Exception:
                        log.debug("Error handling Matrix event", exc_info=True)
            except Exception:
                log.debug("Error in Matrix sync loop", exc_info=True)
                if self._running.is_set():
                    time.sleep(5)

        log.info("Matrix bot stopped")

    def stop(self) -> None:
        """Signal the sync loop to stop."""
        self._running.clear()
