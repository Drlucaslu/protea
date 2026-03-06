"""Telegram Bot — bidirectional interaction via getUpdates long polling.

Pure stdlib (urllib.request + json + threading).  Runs as a daemon thread
alongside the Sentinel main loop.  Errors never propagate to the caller.
"""

from __future__ import annotations

import html as _html
import json
import logging
import mimetypes
import os
import pathlib
import queue
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from uuid import uuid4

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

    __slots__ = (
        # Synchronisation primitives
        "lock", "pause_event", "kill_event", "p0_event", "restart_event",
        "p0_active", "p1_active",
        # Generation state
        "generation", "start_time", "alive", "mutation_rate",
        "max_runtime_sec", "last_score", "last_survived",
        # Task / scheduling
        "task_queue", "evolution_directive", "last_evolution_time",
        "last_task_completion", "executor_thread",
        "pending_reflection_reason",
        # Store references
        "memory_store", "skill_store", "task_store", "scheduled_store",
        # Service references
        "notifier", "skill_runner", "registry_client", "subagent_manager",
        # Convergence detection
        "convergence_proposals", "_convergence_context",
        # Preference store reference (for feedback)
        "_preference_store",
        # Output queue
        "output_queue",
        # Nudge engine
        "nudge_queue", "_nudge_context", "_nudge_context_path",
        # Soul onboarding
        "_pending_soul_question",
        # Habit detection
        "_pending_habits", "_habit_dismissed", "_habit_context",
    )

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pause_event = threading.Event()
        self.kill_event = threading.Event()
        self.p0_event = threading.Event()
        self.restart_event = threading.Event()
        self.p0_active = threading.Event()
        self.p1_active = threading.Event()
        # Mutable fields — protected by self.lock
        self.generation: int = 0
        self.start_time: float = time.time()
        self.alive: bool = False
        self.mutation_rate: float = 0.0
        self.max_runtime_sec: float = 0.0
        self.last_score: float = 0.0
        self.last_survived: bool = False
        # Task / scheduling
        self.task_queue: queue.Queue = queue.Queue()
        self.evolution_directive: str = ""
        self.last_evolution_time: float = 0.0
        self.last_task_completion: float = 0.0
        self.executor_thread: threading.Thread | None = None
        self.pending_reflection_reason: str = ""  # Set by executor for urgent reflection
        # Store references (set by Sentinel after creation)
        self.memory_store = None
        self.skill_store = None
        self.task_store = None
        self.scheduled_store = None
        # Service references (set by Sentinel after creation)
        self.notifier = None
        self.skill_runner = None
        self.registry_client = None
        self.subagent_manager = None
        # Convergence detection
        self.convergence_proposals: queue.Queue = queue.Queue()
        self._convergence_context: dict[str, dict] = {}
        self._preference_store = None  # Set by Sentinel after creation
        # Output queue
        self.output_queue = None  # Set by Sentinel after creation
        # Nudge engine
        self.nudge_queue: queue.Queue = queue.Queue()
        self._nudge_context: dict[str, dict] = {}
        self._nudge_context_path = None
        # Soul onboarding
        self._pending_soul_question: dict | None = None
        # Habit detection
        self._pending_habits: dict[str, dict] = {}
        self._habit_dismissed: set[str] = set()
        self._habit_context: dict[str, dict] = {}

    def _save_nudge_context(self):
        if not self._nudge_context_path:
            return
        try:
            self._nudge_context_path.write_text(
                json.dumps(self._nudge_context), encoding="utf-8")
        except Exception:
            pass

    def _load_nudge_context(self):
        if not self._nudge_context_path or not self._nudge_context_path.exists():
            return
        try:
            data = json.loads(self._nudge_context_path.read_text(encoding="utf-8"))
            now = time.time()
            self._nudge_context = {
                k: v for k, v in data.items()
                if now - v.get("created_at", 0) < 86400
            }
        except Exception:
            pass

    def set_nudge_context(self, key, meta):
        meta.setdefault("created_at", time.time())
        self._nudge_context[key] = meta
        self._save_nudge_context()

    def pop_nudge_context(self, key):
        val = self._nudge_context.pop(key, None)
        if val is not None:
            self._save_nudge_context()
        return val

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
                "p1_active": self.p1_active.is_set(),  # kept for compat
                "task_queue_size": self.task_queue.qsize(),
                "executor_alive": (
                    self.executor_thread is not None
                    and self.executor_thread.is_alive()
                ),
                "last_task_completion": self.last_task_completion,
            }


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A user task submitted via free-text Telegram message."""
    text: str
    chat_id: str
    reply_to_message_id: int | None = None
    created_at: float = field(default_factory=time.time)
    task_id: str = field(default_factory=lambda: f"t-{int(time.time() * 1000) % 1_000_000}")
    exec_mode: str = "llm"  # "llm" (full LLM pipeline) or "shell" (direct script execution)


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
        self.bot_username: str = ""  # set by _fetch_bot_info()
        self._fetch_bot_info()
        
        self._triage_llm = self._create_triage_llm()

        # 对话上下文追踪 - 记住最近的 bot 消息以支持回复关联
        self._last_bot_messages = []  # 保留最近 5 条消息
        self._max_context_messages = 5
        self._context_window_seconds = 120  # 2分钟内的回复视为有上下文

        # Feedback collection: 👍/👎 after completed tasks (~20% probability).

    # -- low-level API helpers --

    def _fetch_bot_info(self) -> None:
        """Fetch bot username via getMe (best-effort)."""
        result = self._api_call("getMe")
        if result:
            self.bot_username = result["result"].get("username", "")
            log.info("Bot username: @%s", self.bot_username)

    # ------------------------------------------------------------------
    # Markdown → Telegram HTML conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _md_to_tg_html(text: str) -> str:
        """Convert Markdown to Telegram-compatible HTML.

        Handles both standard Markdown (``**bold**``) and Telegram-legacy
        Markdown (``*bold*``), converting everything to HTML so we can use
        ``parse_mode=HTML`` which is the most forgiving Telegram mode.
        """
        preserved: list[str] = []

        def _hold(s: str) -> str:
            idx = len(preserved)
            preserved.append(s)
            return f"\x00\x01{idx}\x00"

        # 1. Fenced code blocks: ```lang\ncode\n```
        def _block(m: re.Match) -> str:
            lang = m.group(1) or ""
            code = _html.escape(m.group(2).strip())
            if lang:
                return _hold(f'<pre><code class="language-{lang}">'
                             f"{code}</code></pre>")
            return _hold(f"<pre>{code}</pre>")

        text = re.sub(r"```(\w*)\n?(.*?)```", _block, text, flags=re.DOTALL)

        # 2. Inline code: `code`
        text = re.sub(
            r"`([^`\n]+)`",
            lambda m: _hold(f"<code>{_html.escape(m.group(1))}</code>"),
            text,
        )

        # 3. Links: [text](url)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            lambda m: _hold(
                f'<a href="{_html.escape(m.group(2))}">'
                f"{_html.escape(m.group(1))}</a>"
            ),
            text,
        )

        # 4. Escape HTML entities in remaining text
        text = _html.escape(text)

        # 5. Apply formatting (order matters: longest match first)
        # ***bold italic***
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
        # **bold**  (standard Markdown)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        # *bold*  (Telegram-legacy — treat as bold, not italic)
        text = re.sub(
            r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<b>\1</b>", text
        )
        # _italic_
        text = re.sub(
            r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"<i>\1</i>", text
        )
        # ### headers → bold
        text = re.sub(
            r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE
        )
        # ~~strikethrough~~
        text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

        # 6. Restore preserved content
        for i, content in enumerate(preserved):
            text = text.replace(f"\x00\x01{i}\x00", content)

        return text

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
        except urllib.error.HTTPError as exc:
            # One-line log for HTTP errors (no traceback) — reduces noise
            log.warning("API call %s failed: HTTP %d", method, exc.code)
            return None
        except Exception:
            log.warning("API call %s failed", method, exc_info=True)
            return None

    def _get_updates(self) -> list[dict]:
        """Fetch new updates via long polling."""
        params = {
            "offset": self._offset,
            "timeout": 30,
            "allowed_updates": ["message", "callback_query"],
        }
        result = self._api_call("getUpdates", params)
        if not result:
            return []
        updates = result.get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    _TG_MSG_LIMIT = 4000  # Telegram hard limit ~4096, leave margin

    def _split_message(self, text: str) -> list[str]:
        """Split *text* into segments that fit within Telegram's message limit."""
        limit = self._TG_MSG_LIMIT
        if len(text) <= limit:
            return [text]
        segments: list[str] = []
        while text:
            if len(text) <= limit:
                segments.append(text)
                break
            # Find last newline within limit for a clean break.
            cut = text.rfind("\n", 0, limit)
            if cut <= 0:
                cut = limit  # no good break point — hard cut
            segments.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return segments

    def _send_reply(self, text: str, chat_id: str | None = None,
                     reply_to_message_id: int | None = None) -> None:
        """Send a text reply (fire-and-forget).

        Args:
            text: Message text.
            chat_id: Target chat.  Defaults to owner private chat.
            reply_to_message_id: If set, reply to this message (thread linking).
        """
        target = chat_id or self.chat_id
        segments = self._split_message(text)
        for i, seg in enumerate(segments):
            if len(segments) > 1:
                seg = f"[{i + 1}/{len(segments)}]\n{seg}"
            html_seg = self._md_to_tg_html(seg)
            params: dict = {"chat_id": target, "text": html_seg, "parse_mode": "HTML"}
            if reply_to_message_id:
                params["reply_to_message_id"] = reply_to_message_id
            result = self._api_call("sendMessage", params)
            if result is None:
                # HTML was rejected — retry as plain text.
                params["text"] = seg  # original text, no conversion
                params.pop("parse_mode", None)
                params.pop("reply_to_message_id", None)  # might fail if original deleted
                result = self._api_call("sendMessage", params)

        # 记录发送的消息以支持上下文追踪 (only for owner private chat)
        if result and result.get("ok") and (not chat_id or chat_id == self.chat_id):
            msg_data = result.get("result", {})
            message_id = msg_data.get("message_id")
            if message_id:
                self._last_bot_messages.append({
                    "text": text,
                    "message_id": message_id,
                    "timestamp": time.time()
                })
                # 只保留最近的 N 条消息
                if len(self._last_bot_messages) > self._max_context_messages:
                    self._last_bot_messages.pop(0)

    def make_reply_fn(self, chat_id: str, reply_to_message_id: int | None = None):
        """Create a reply callable bound to a specific chat (and optional thread)."""
        def reply(text: str) -> None:
            self._send_reply(text, chat_id=chat_id, reply_to_message_id=reply_to_message_id)
        return reply

    def _send_message_with_keyboard(self, text: str, buttons: list[list[dict]]) -> dict | None:
        """Send a message with an inline keyboard.

        *buttons* is a list of rows, each row a list of dicts with
        ``text`` and ``callback_data`` keys.
        Returns the Telegram API response dict, or None on failure.
        """
        return self._api_call("sendMessage", {
            "chat_id": self.chat_id,
            "text": self._md_to_tg_html(text),
            "parse_mode": "HTML",
            "reply_markup": json.dumps({"inline_keyboard": buttons}),
        })

    def deliver_evolution_outputs(self) -> None:
        """Check output queue and send pending evolution outputs to user."""
        oq = getattr(self.state, 'output_queue', None)
        if not oq:
            return
        try:
            pending = oq.get_pending(limit=2)
        except Exception:
            log.debug("Output queue get_pending failed", exc_info=True)
            return
        for item in pending:
            text = f"\U0001f9ec 新进化能力: **{item['capability']}**\n\n{item['summary'][:300]}"
            buttons = [[
                {"text": "\U0001f44d 不错", "callback_data": f"evo:accept:{item['id']}"},
                {"text": "\U0001f4cc 定期执行", "callback_data": f"evo:schedule:{item['id']}"},
                {"text": "\U0001f44e 不要了", "callback_data": f"evo:reject:{item['id']}"},
            ]]
            result = self._send_message_with_keyboard(text, buttons)
            msg_id = result.get("result", {}).get("message_id") if result else None
            try:
                oq.mark_delivered(item['id'], msg_id)
            except Exception:
                log.debug("Output queue mark_delivered failed", exc_info=True)

    def send_feedback_prompt(self) -> None:
        """Send a quick feedback prompt after a completed task.

        Called by the task executor after the full response is sent.
        Triggers with ~20% probability to avoid annoying the user.
        """
        import random
        if random.random() > 0.2:
            return
        buttons = [[
            {"text": "\U0001f44d", "callback_data": "feedback:positive"},
            {"text": "\U0001f44e", "callback_data": "feedback:negative"},
        ]]
        self._send_message_with_keyboard(
            "这次回答还行吗？", buttons,
        )

    def _handle_nudge_callback(self, data: str) -> str:
        """Handle nudge-related inline keyboard callbacks."""
        if data == "nudge:dismiss":
            return "好的 \U0001f44c"

        # nudge:schedule:<hash>, nudge:execute:<hash>, nudge:expand:<hash>
        parts = data.split(":")
        if len(parts) < 3:
            return "操作无效。"
        action, h = parts[1], parts[2]
        ctx = self.state._nudge_context.get(h)
        if not ctx:
            return "建议已过期。"

        if action == "schedule":
            # Ask for frequency — store hash for follow-up.
            self.state.set_nudge_context(f"_sched_{h}", ctx)
            buttons = [[
                {"text": "每天", "callback_data": f"nudge:cron:daily:{h}"},
                {"text": "每小时", "callback_data": f"nudge:cron:hourly:{h}"},
            ]]
            self._send_message_with_keyboard(
                "你想多久执行一次？", buttons,
            )
            return ""

        if action == "cron":
            # parts: nudge:cron:<freq>:<hash>
            if len(parts) < 4:
                return "操作无效。"
            freq, cron_h = parts[2], parts[3]
            sched_ctx = self.state.pop_nudge_context(f"_sched_{cron_h}")
            if not sched_ctx:
                sched_ctx = self.state._nudge_context.get(cron_h)
            if not sched_ctx:
                return "建议已过期。"
            cron_map = {"daily": "0 9 * * *", "hourly": "0 * * * *"}
            cron_expr = cron_map.get(freq, "0 9 * * *")
            task_text = sched_ctx.get("suggested_task") or sched_ctx.get("source_text", "")
            if not task_text:
                return "没有找到要定期执行的任务。"
            ss = self.state.scheduled_store
            if ss:
                try:
                    ss.add(task_text[:200], cron_expr)
                    return f"已创建定期任务：{task_text[:60]}（{cron_expr}）"
                except Exception:
                    log.debug("Failed to create scheduled task", exc_info=True)
                    return "创建定期任务失败，请稍后重试。"
            return "定期任务功能暂不可用。"

        if action == "execute":
            task_text = ctx.get("suggested_task", "")
            if not task_text:
                return "没有找到要执行的任务。"
            from ring1.telegram_bot import Task
            task = Task(text=task_text, chat_id=self.chat_id)
            self.state.task_queue.put(task)
            self.state.p0_event.set()
            return f"好的，正在执行：{task_text[:60]}"

        if action == "expand":
            source = ctx.get("source_text", "")
            if not source:
                return "没有找到要展开的内容。"
            expand_text = f"请详细展开: {source[:200]}"
            from ring1.telegram_bot import Task
            task = Task(text=expand_text, chat_id=self.chat_id)
            self.state.task_queue.put(task)
            self.state.p0_event.set()
            return f"好的，正在展开..."

        return "未知操作。"

    def _send_document(self, file_path: str, caption: str = "", target_chat_id: str = "") -> bool:
        """Send a file to the authorized chat via sendDocument (multipart).

        Args:
            file_path: Path to the file on disk.
            caption: Optional caption text (max 1024 chars).

        Returns:
            True on success, False on any error.
        """
        path = pathlib.Path(file_path)
        if not path.is_file():
            log.warning("_send_document: file not found: %s", file_path)
            return False
        file_size = path.stat().st_size
        if file_size > 50 * 1024 * 1024:  # 50 MB Telegram limit
            log.warning("_send_document: file too large (%d bytes): %s", file_size, file_path)
            return False

        boundary = uuid4().hex
        content_type, _ = mimetypes.guess_type(path.name)
        if content_type is None:
            content_type = "application/octet-stream"

        # Build multipart/form-data body
        parts: list[bytes] = []

        # chat_id field
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
        chat_id_to_use = target_chat_id if target_chat_id else self.chat_id
        parts.append(f"{chat_id_to_use}\r\n".encode())

        # caption field (optional)
        if caption:
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(b'Content-Disposition: form-data; name="caption"\r\n\r\n')
            parts.append(f"{caption[:1024]}\r\n".encode())

        # document field (file binary)
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="document"; filename="{path.name}"\r\n'.encode()
        )
        parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        parts.append(path.read_bytes())
        parts.append(b"\r\n")

        # closing boundary
        parts.append(f"--{boundary}--\r\n".encode())

        body = b"".join(parts)
        url = _API_BASE.format(token=self.bot_token, method="sendDocument")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("ok"):
                    log.info("Sent document: %s", path.name)
                    return True
                log.warning("sendDocument failed: %s", result)
                return False
        except Exception:
            log.debug("sendDocument error", exc_info=True)
            return False

    def _download_file(self, file_id: str) -> bytes | None:
        """Download a file from Telegram servers and return its bytes."""
        try:
            # Step 1: Get file path from Telegram
            result = self._api_call("getFile", {"file_id": file_id})
            if not result or "result" not in result:
                return None
            file_path = result["result"].get("file_path")
            if not file_path:
                return None
            
            # Step 2: Download the file
            download_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            req = urllib.request.Request(download_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception:
            log.debug("File download failed", exc_info=True)
            return None

    def _handle_file(self, file_info: dict, file_type: str, msg_chat_id: str, caption: str = "") -> tuple[str, pathlib.Path | None]:
        """Handle any file upload (document, photo, audio, video, voice).

        Args:
            file_info: dict containing file_id, file_name (or generated), file_size
            file_type: "document", "photo", "audio", "video", "voice"
            msg_chat_id: chat ID
            caption: optional caption from message

        Returns:
            Tuple of (response_text, saved_path_or_None).
        """
        file_id = file_info.get("file_id")

        # Generate filename based on type if not provided
        if "file_name" in file_info:
            file_name = file_info["file_name"]
        else:
            # Generate filename with timestamp
            timestamp = int(time.time() * 1000) % 1_000_000
            ext_map = {
                "photo": "jpg",
                "audio": "mp3",
                "video": "mp4",
                "voice": "ogg",
            }
            ext = ext_map.get(file_type, "bin")
            file_name = f"{file_type}_{timestamp}.{ext}"

        file_size = file_info.get("file_size", 0)

        if not file_id:
            return "⚠️ 文件 ID 缺失。", None

        # Download file
        file_bytes = self._download_file(file_id)
        if file_bytes is None:
            return "⚠️ 文件下载失败。", None

        # Save to telegram_output directory
        output_dir = pathlib.Path("telegram_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / file_name

        # Handle duplicate names
        counter = 1
        while output_path.exists():
            name_parts = file_name.rsplit(".", 1)
            if len(name_parts) == 2:
                output_path = output_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
            else:
                output_path = output_dir / f"{file_name}_{counter}"
            counter += 1

        try:
            output_path.write_bytes(file_bytes)

            # Type-specific emoji
            emoji_map = {
                "document": "📄",
                "photo": "🖼",
                "audio": "🎵",
                "video": "🎬",
                "voice": "🎤",
            }
            emoji = emoji_map.get(file_type, "📎")

            type_name_map = {
                "document": "文档",
                "photo": "图片",
                "audio": "音频",
                "video": "视频",
                "voice": "语音",
            }
            type_name = type_name_map.get(file_type, "文件")

            response = (
                f"✅ {emoji} {type_name}已接收并保存！\n\n"
                f"📄 文件名: {file_name}\n"
                f"💾 大小: {file_size / 1024:.1f} KB\n"
                f"📂 保存路径: {output_path}\n"
            )

            if caption:
                response += f"💬 说明: {caption}\n"

            response += "\n💡 现在可以用其他命令处理这个文件了。"

            return response, output_path
        except Exception as e:
            log.error("Failed to save file", exc_info=True)
            return f"⚠️ 保存文件失败: {str(e)}", None

    def _answer_callback_query(self, callback_query_id: str) -> None:
        """Acknowledge a callback query so Telegram stops showing a spinner."""
        self._api_call("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
        })

    def _is_authorized(self, update: dict) -> bool:
        """Check if the update comes from an authorized chat.

        - Group/supergroup messages are always accepted (filtered later by
          ``_should_respond_in_group``).
        - Private messages use the existing owner-lock logic.
        """
        if "callback_query" in update:
            chat = update["callback_query"].get("message", {}).get("chat", {})
        else:
            chat = update.get("message", {}).get("chat", {})
        msg_chat_id = str(chat.get("id", ""))
        if not msg_chat_id:
            return False
        chat_type = chat.get("type", "private")
        if chat_type in ("group", "supergroup"):
            return True  # group messages accepted; _should_respond_in_group filters later
        # Private chat: auto-detect or check owner
        if not self.chat_id:
            self._lock_chat_id(msg_chat_id)
            return True
        return msg_chat_id == self.chat_id

    def _lock_chat_id(self, chat_id: str) -> None:
        """Lock to *chat_id*, persist to ``.env``, and update the notifier."""
        self.chat_id = chat_id
        log.info("Auto-detected chat_id=%s", chat_id)
        # Propagate to TelegramNotifier if available on state.
        notifier = getattr(self.state, "notifier", None)
        if notifier and hasattr(notifier, "set_chat_id"):
            notifier.set_chat_id(chat_id)
        self._persist_chat_id(chat_id)

    def _persist_chat_id(self, chat_id: str) -> None:
        """Write ``TELEGRAM_CHAT_ID`` into the ``.env`` file."""
        env_path = self.ring2_path.parent / ".env"
        try:
            if env_path.is_file():
                lines = env_path.read_text().splitlines()
                new_lines = []
                found = False
                for line in lines:
                    if line.strip().startswith("TELEGRAM_CHAT_ID"):
                        new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
                        found = True
                    else:
                        new_lines.append(line)
                if not found:
                    new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
                env_path.write_text("\n".join(new_lines) + "\n")
            else:
                env_path.write_text(f"TELEGRAM_CHAT_ID={chat_id}\n")
            log.info("Persisted chat_id to %s", env_path)
        except Exception:
            log.debug("Failed to persist chat_id to .env", exc_info=True)

    def _should_respond_in_group(self, msg: dict) -> bool:
        """Return True if the bot should respond to this group message.

        Two-layer filter: fast rules first, then LLM triage.
        """
        text = msg.get("text", "") or msg.get("caption", "")
        # 1. @mention
        if self.bot_username and f"@{self.bot_username}" in text:
            return True
        # 2. Bot name mentioned (case-insensitive)
        if self._bot_name_mentioned(text):
            return True
        # 3. Reply to bot's own message
        reply = msg.get("reply_to_message", {})
        if reply.get("from", {}).get("username") == self.bot_username and self.bot_username:
            return True
        # 4. Slash command
        if text.strip().startswith("/"):
            return True
        # 5. LLM triage: can we answer with high confidence?
        return self._llm_triage_group_message(text)

    def _bot_name_mentioned(self, text: str) -> bool:
        """Check if the bot's name is mentioned in the text (case-insensitive)."""
        text_lower = text.lower()
        names = {"protea"}
        if self.bot_username:
            clean = self.bot_username.lower().replace("bot", "").replace("test", "")
            if clean:
                names.add(clean)
            names.add(self.bot_username.lower())
        return any(name in text_lower for name in names)

    def _llm_triage_group_message(self, text: str) -> bool:
        """Use a fast LLM to decide if the bot can confidently answer this message."""
        if not text.strip() or not self._triage_llm:
            return False
        prompt = (
            "A message was sent in a group chat. Decide: is this message a question "
            "or request where an AI assistant could be helpful?\n\n"
            "YES: questions, requests for info, how-to, translation, calculation, "
            "weather, time, recommendations, anything seeking an answer\n"
            "NO: casual chat between humans, greetings, opinions with no question, "
            "inside jokes, messages clearly addressed to another person\n\n"
            f"Message: \"{text}\"\n\nReply ONLY YES or NO."
        )
        try:
            result = self._triage_llm.send_message("You are a triage classifier.", prompt)
            answer = result.strip().upper()
            should = answer.startswith("YES")
            log.info("LLM triage: %s → %s", text[:60], answer)
            return should
        except Exception:
            log.debug("LLM triage failed", exc_info=True)
            return False

    def _create_triage_llm(self):
        """Create a lightweight LLM client for group message triage."""
        api_key = os.environ.get("CLAUDE_API_KEY", "")
        if not api_key:
            log.info("Triage LLM not created: CLAUDE_API_KEY not set")
            return None
        try:
            from ring1.llm_base import create_llm_client
            client = create_llm_client(
                provider="anthropic", api_key=api_key,
                model="claude-haiku-4-5-20251001", max_tokens=16,
            )
            log.info("Triage LLM created (haiku)")
            return client
        except Exception:
            log.warning("Failed to create triage LLM", exc_info=True)
            return None

    # -- command handlers --

    def _get_ring2_description(self) -> str:
        """Extract the first line of Ring 2's module docstring."""
        try:
            source = (self.ring2_path / "main.py").read_text()
            for quote in ('"""', "'''"):
                idx = source.find(quote)
                if idx == -1:
                    continue
                end = source.find(quote, idx + 3)
                if end == -1:
                    continue
                doc = source[idx + 3:end].strip().splitlines()[0]
                return doc
        except Exception:
            pass
        return ""

    def _cmd_status(self) -> str:
        snap = self.state.snapshot()
        elapsed = time.time() - snap["start_time"]
        status_map = {
            "PAUSED": "PAUSED (已暂停)",
            "ALIVE": "ALIVE (运行中)",
            "DEAD": "DEAD (已停止)",
        }
        raw = "PAUSED" if snap["paused"] else ("ALIVE" if snap["alive"] else "DEAD")
        status = status_map[raw]
        desc = self._get_ring2_description()
        lines = [
            f"*Protea 状态面板*",
            f"🧬 代 (Generation): {snap['generation']}",
            f"📡 状态 (Status): {status}",
            f"⏱ 运行时长 (Uptime): {elapsed:.0f}s",
            f"🎲 变异率 (Mutation rate): {snap['mutation_rate']:.2f}",
            f"⏳ 最大运行时间 (Max runtime): {snap['max_runtime_sec']:.0f}s",
        ]
        if desc:
            lines.append(f"🧠 当前程序 (Program): {desc}")
        # Executor health
        executor_alive = snap.get("executor_alive", False)
        executor_status = "🟢 正常" if executor_alive else "🔴 离线"
        lines.append(f"🤖 执行器 (Executor): {executor_status}")
        lines.append(f"📋 排队任务 (Queued): {snap['task_queue_size']}")
        last_comp = snap.get("last_task_completion", 0.0)
        if last_comp > 0:
            ago = time.time() - last_comp
            lines.append(f"✅ 上次完成 (Last done): {ago:.0f}s ago")
        return "\n".join(lines)

    def _cmd_history(self) -> str:
        rows = self.fitness.get_history(limit=10)
        if not rows:
            return "暂无历史记录。"
        lines = ["*最近 10 代历史 (Recent 10 generations):*"]
        for r in rows:
            surv = "✅ 存活" if r["survived"] else "❌ 失败"
            lines.append(
                f"第 {r['generation']} 代  适应度={r['score']:.2f}  "
                f"{surv}  {r['runtime_sec']:.0f}s"
            )
        return "\n".join(lines)

    def _cmd_top(self) -> str:
        rows = self.fitness.get_best(n=5)
        if not rows:
            return "暂无适应度数据。"
        lines = ["*适应度排行 Top 5 (Top 5 generations):*"]
        for r in rows:
            surv = "✅ 存活" if r["survived"] else "❌ 失败"
            lines.append(
                f"第 {r['generation']} 代  适应度={r['score']:.2f}  "
                f"{surv}  `{r['commit_hash'][:8]}`"
            )
        return "\n".join(lines)

    def _cmd_code(self) -> str:
        code_path = self.ring2_path / "main.py"
        try:
            source = code_path.read_text()
        except FileNotFoundError:
            return "ring2/main.py 未找到。"
        if len(source) > 3000:
            source = source[:3000] + "\n... (已截断)"
        return f"```python\n{source}\n```"

    def _cmd_pause(self) -> str:
        if self.state.pause_event.is_set():
            return "已经处于暂停状态。"
        self.state.pause_event.set()
        return "进化已暂停。"

    def _cmd_resume(self) -> str:
        if not self.state.pause_event.is_set():
            return "当前未暂停。"
        self.state.pause_event.clear()
        return "进化已恢复。"

    def _cmd_kill(self) -> str:
        self.state.kill_event.set()
        return "终止信号已发送 — Ring 2 将重启。"

    def _cmd_help(self) -> str:
        return (
            "*Protea 指令列表:*\n"
            "/status — 查看状态 (代数、运行时间、状态)\n"
            "/history — 最近 10 代历史\n"
            "/top — 适应度排行 Top 5\n"
            "/code — 查看当前 Ring 2 源码\n"
            "/pause — 暂停进化循环\n"
            "/resume — 恢复进化循环\n"
            "/kill — 重启 Ring 2 (不推进代数)\n"
            "/direct <文本> — 设置进化指令\n"
            "/tasks — 查看任务队列与指令\n"
            "/memory — 查看最近记忆\n"
            "/forget — 清除所有记忆\n"
            "/skills — 列出已保存的技能\n"
            "/skill <名称> — 查看技能详情\n"
            "/run <名称> — 启动一个技能进程\n"
            "/stop — 停止正在运行的技能\n"
            "/running — 查看技能运行状态\n"
            "/background — 查看后台任务\n"
            "/files — 列出已上传的文件\n"
            "/find <前缀> — 查找文件\n"
            "/schedule — 管理定时任务\n"
            "/calendar — 查看定时任务日历\n\n"
            "💬 直接发送文字即可向 Protea 提问 (P0 任务)\n\n"
            "📎 *支持的文件类型:*\n"
            "📄 文档 (Document) - Excel, PDF, Word 等\n"
            "🖼 图片 (Photo) - JPG, PNG 等\n"
            "🎵 音频 (Audio) - MP3, M4A 等\n"
            "🎬 视频 (Video) - MP4, MOV 等\n"
            "🎤 语音 (Voice) - 语音消息\n"
            "💾 所有文件自动保存到 telegram_output/ 目录"
        )

    def _cmd_direct(self, full_text: str) -> str:
        """Set an evolution directive from /direct <text>."""
        # Strip the /direct prefix (and optional @botname)
        parts = full_text.strip().split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            return "用法: /direct <指令文本>\n示例: /direct 变成贪吃蛇"
        directive = parts[1].strip()
        with self.state.lock:
            self.state.evolution_directive = directive
        self.state.p0_event.set()  # wake sentinel
        return f"进化指令已设置: {directive}"

    def _cmd_tasks(self) -> str:
        """Show task queue status, current directive, and recent tasks."""
        snap = self.state.snapshot()
        lines = ["*任务队列 (Task Queue):*"]
        lines.append(f"排队中 (Queued): {snap['task_queue_size']}")
        p0 = "是" if snap["p0_active"] else "否"
        lines.append(f"P0 执行中 (Active): {p0}")
        directive = self.state.evolution_directive
        if directive:
            lines.append(f"指令 (Directive): {directive}")
        # Recent tasks from store
        ts = self.state.task_store
        if ts:
            recent = ts.get_recent(5)
            if recent:
                lines.append("")
                lines.append("*最近任务 (Recent):*")
                for t in recent:
                    status_icon = {"pending": "⏳", "executing": "🔄", "completed": "✅", "failed": "❌"}.get(t["status"], "❓")
                    text_preview = t["text"][:40] + ("…" if len(t["text"]) > 40 else "")
                    lines.append(f"{status_icon} {t['task_id']}: {text_preview}")
        return "\n".join(lines)

    def _cmd_memory(self) -> str:
        """Show recent memories."""
        ms = self.state.memory_store
        if not ms:
            return "记忆模块不可用。"
        entries = ms.get_recent(5)
        if not entries:
            return "暂无记忆。"
        lines = [f"*最近记忆 (共 {ms.count()} 条):*"]
        for e in entries:
            lines.append(
                f"[第 {e['generation']} 代, {e['entry_type']}] {e['content']}"
            )
        return "\n".join(lines)

    def _cmd_forget(self) -> str:
        """Clear all memories."""
        ms = self.state.memory_store
        if not ms:
            return "记忆模块不可用。"
        ms.clear()
        return "所有记忆已清除。"

    def _cmd_skills(self) -> str:
        """List saved skills."""
        ss = self.state.skill_store
        if not ss:
            return "技能库不可用。"
        skills = ss.get_active(500)
        if not skills:
            return "暂无已保存的技能。"
        lines = [f"*已保存技能 (共 {len(skills)} 个):*"]
        for s in skills:
            lines.append(f"- *{s['name']}*: {s['description']} (已使用 {s['usage_count']} 次)")
        return "\n".join(lines)

    def _cmd_skill(self, full_text: str) -> str | None:
        """Show skill details: /skill <name>.  No args → inline keyboard."""
        ss = self.state.skill_store
        if not ss:
            return "技能库不可用。"
        parts = full_text.strip().split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            skills = ss.get_active(500)
            if not skills:
                return "暂无已保存的技能。"
            buttons = [
                [{"text": s["name"], "callback_data": f"skill:{s['name']}"}]
                for s in skills
            ]
            self._send_message_with_keyboard("选择一个技能:", buttons)
            return None
        name = parts[1].strip()
        skill = ss.get_by_name(name)
        if not skill:
            return f"技能 '{name}' 未找到。"
        lines = [
            f"*技能: {skill['name']}*",
            f"描述 (Description): {skill['description']}",
            f"来源 (Source): {skill['source']}",
            f"已使用 (Used): {skill['usage_count']} 次",
            f"激活 (Active): {'是' if skill['active'] else '否'}",
            "",
            "提示词模板 (Prompt template):",
            f"```\n{skill['prompt_template']}\n```",
        ]
        return "\n".join(lines)

    def _cmd_run(self, full_text: str) -> str | None:
        """Start a skill: /run <name>.  No args → inline keyboard."""
        sr = self.state.skill_runner
        if not sr:
            return "技能运行器不可用。"
        ss = self.state.skill_store
        if not ss:
            return "技能库不可用。"

        parts = full_text.strip().split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            skills = ss.get_active(500)
            if not skills:
                return "暂无已保存的技能。"
            buttons = [
                [{"text": s["name"], "callback_data": f"run:{s['name']}"}]
                for s in skills
            ]
            self._send_message_with_keyboard("选择要运行的技能:", buttons)
            return None
        name = parts[1].strip()

        skill = ss.get_by_name(name)
        if not skill:
            return f"技能 '{name}' 未找到。"
        source_code = skill.get("source_code", "")
        if not source_code:
            return f"技能 '{name}' 没有源码。"

        pid, msg = sr.run(name, source_code)
        ss.update_usage(name)
        return msg

    def _cmd_stop_skill(self) -> str:
        """Stop the running skill."""
        sr = self.state.skill_runner
        if not sr:
            return "技能运行器不可用。"
        if sr.stop():
            return "技能已停止。"
        return "当前没有运行中的技能。"

    def _cmd_running(self) -> str:
        """Show running skill status and recent output."""
        sr = self.state.skill_runner
        if not sr:
            return "技能运行器不可用。"
        info = sr.get_info()
        if not info:
            return "暂无已启动的技能。"
        status = "运行中 (RUNNING)" if info["running"] else "已停止 (STOPPED)"
        lines = [
            f"*技能: {info['skill_name']}*",
            f"状态 (Status): {status}",
            f"进程 (PID): {info['pid']}",
        ]
        if info["running"]:
            lines.append(f"运行时长 (Uptime): {info['uptime']:.0f}s")
        if info["port"]:
            lines.append(f"端口 (Port): {info['port']}")
        output = sr.get_output(max_lines=15)
        if output:
            lines.append(f"\n*最近输出:*\n```\n{output}\n```")
        else:
            lines.append("\n(无输出)")
        return "\n".join(lines)

    def _cmd_background(self) -> str:
        """Show background subagent tasks."""
        mgr = getattr(self.state, "subagent_manager", None)
        if not mgr:
            return "后台任务不可用。"
        tasks = mgr.get_active()
        if not tasks:
            return "暂无后台任务。"
        lines = [f"*后台任务 (共 {len(tasks)} 个):*"]
        for t in tasks:
            status = "✅ 完成" if t["done"] else "⏳ 运行中"
            lines.append(
                f"- {t['task_id']} [{status}] {t['duration']:.0f}s — {t['description'][:60]}"
            )
        return "\n".join(lines)

    def _cmd_files(self) -> str:
        """List files in telegram_output directory."""
        output_dir = pathlib.Path("telegram_output")
        if not output_dir.exists():
            return "telegram_output 目录不存在。"
        
        files = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return "telegram_output 目录为空。"
        
        lines = [f"*已上传文件 (共 {len(files)} 个):*"]
        for f in files[:20]:  # Show only 20 most recent
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                lines.append(f"📄 {f.name} ({size_kb:.1f} KB)")
        
        return "\n".join(lines)

    def _cmd_find(self, full_text: str) -> str:
        """Find files by prefix: /find <prefix>."""
        parts = full_text.strip().split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            return "用法: /find <文件名前缀>\n示例: /find 13OB"
        
        prefix = parts[1].strip()
        
        # Search in multiple directories
        search_dirs = [
            pathlib.Path("telegram_output"),
            pathlib.Path("."),
            pathlib.Path("ring2_output"),
        ]
        
        found_files = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for f in search_dir.rglob("*"):
                if f.is_file() and f.name.startswith(prefix):
                    found_files.append(f)
        
        if not found_files:
            return f"未找到以 '{prefix}' 开头的文件。"
        
        lines = [f"*找到 {len(found_files)} 个匹配文件:*"]
        for f in found_files[:20]:  # Limit to 20 results
            size_kb = f.stat().st_size / 1024
            lines.append(f"📄 {f} ({size_kb:.1f} KB)")
        
        if len(found_files) > 20:
            lines.append(f"\n... 还有 {len(found_files) - 20} 个文件未显示")
        
        return "\n".join(lines)

    def _cmd_schedule(self, full_text: str, chat_id: str = "") -> str:
        """Handle /schedule subcommands: list|add|once|remove|enable|disable."""
        ss = self.state.scheduled_store
        if not ss:
            return "定时任务模块不可用。"
        parts = full_text.strip().split(None, 1)
        args = parts[1].strip() if len(parts) > 1 else ""

        if not args or args == "list":
            return self._cmd_calendar()

        tokens = args.split(None, 2)
        subcmd = tokens[0].lower()

        if subcmd == "add":
            # /schedule add <name> <cron> <task>
            # Parse: name is first token, cron is next 5 tokens, rest is task
            rest = tokens[1] if len(tokens) > 1 else ""
            if not rest:
                return (
                    "用法: /schedule add <名称> <cron> <任务>\n"
                    '示例: /schedule add 每日新闻 "30 9 * * *" 获取今日新闻摘要'
                )
            # Support quoted cron: /schedule add name "30 9 * * *" task text
            rest_parts = rest.strip()
            if len(tokens) > 2:
                rest_parts = tokens[1] + " " + tokens[2]
            else:
                rest_parts = tokens[1] if len(tokens) > 1 else ""
            return self._schedule_add(rest_parts, chat_id)

        if subcmd == "once":
            # /schedule once <name> <datetime> <task>
            rest = args[len("once"):].strip()
            if not rest:
                return (
                    "用法: /schedule once <名称> <日期时间> <任务>\n"
                    "示例: /schedule once 提醒 2026-02-20T14:00 开会提醒"
                )
            return self._schedule_once(rest, chat_id)

        if subcmd == "remove":
            name = tokens[1] if len(tokens) > 1 else ""
            if not name:
                return "用法: /schedule remove <名称>"
            task = ss.get_by_name(name)
            if not task:
                return f"定时任务 '{name}' 未找到。"
            ss.remove(task["schedule_id"])
            return f"已删除定时任务: {name}"

        if subcmd == "enable":
            name = tokens[1] if len(tokens) > 1 else ""
            if not name:
                return "用法: /schedule enable <名称>"
            task = ss.get_by_name(name)
            if not task:
                return f"定时任务 '{name}' 未找到。"
            ss.enable(task["schedule_id"])
            return f"已启用定时任务: {name}"

        if subcmd == "disable":
            name = tokens[1] if len(tokens) > 1 else ""
            if not name:
                return "用法: /schedule disable <名称>"
            task = ss.get_by_name(name)
            if not task:
                return f"定时任务 '{name}' 未找到。"
            ss.disable(task["schedule_id"])
            return f"已禁用定时任务: {name}"

        return (
            "用法:\n"
            "/schedule list — 列出所有定时任务\n"
            "/schedule add <名称> <cron> <任务> — 添加 cron 任务\n"
            "/schedule once <名称> <日期时间> <任务> — 添加一次性任务\n"
            "/schedule remove <名称> — 删除\n"
            "/schedule enable <名称> — 启用\n"
            "/schedule disable <名称> — 禁用"
        )

    def _schedule_add(self, text: str, chat_id: str) -> str:
        """Parse and add a cron scheduled task.

        Expected formats:
          name "cron_expr" task text
          name cron_expr task text  (when cron is 5 space-separated fields)
        """
        ss = self.state.scheduled_store

        # Try quoted cron first: name "30 9 * * *" task text
        import re
        m = re.match(r'(\S+)\s+"([^"]+)"\s+(.*)', text, re.DOTALL)
        if m:
            name, cron_expr, task_text = m.group(1), m.group(2), m.group(3)
        else:
            # Unquoted: name field1 field2 field3 field4 field5 task text
            parts = text.split()
            if len(parts) < 7:
                return (
                    "用法: /schedule add <名称> <cron 5字段> <任务>\n"
                    '示例: /schedule add 每日新闻 "30 9 * * *" 获取今日新闻摘要\n'
                    "或: /schedule add 每日新闻 30 9 * * * 获取今日新闻摘要"
                )
            name = parts[0]
            cron_expr = " ".join(parts[1:6])
            task_text = " ".join(parts[6:])

        # Validate cron
        try:
            from ring0.cron import next_run as _cron_next, describe as _cron_desc
            from datetime import datetime
            _cron_next(cron_expr, datetime.now())
        except Exception as e:
            return f"无效的 cron 表达式: {cron_expr}\n错误: {e}"

        # Check duplicate name
        if ss.get_by_name(name):
            return f"已存在同名定时任务: {name}"

        sid = ss.add(name, task_text, cron_expr, schedule_type="cron", chat_id=chat_id)
        desc = _cron_desc(cron_expr)
        return f"已添加定时任务: {name}\n计划: {desc} ({cron_expr})\n任务: {task_text}"

    def _schedule_once(self, text: str, chat_id: str) -> str:
        """Parse and add a one-shot scheduled task."""
        ss = self.state.scheduled_store
        parts = text.split(None, 2)
        if len(parts) < 3:
            return (
                "用法: /schedule once <名称> <日期时间> <任务>\n"
                "示例: /schedule once 提醒 2026-02-20T14:00 开会提醒"
            )
        name, dt_str, task_text = parts[0], parts[1], parts[2]

        # Validate datetime
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            return f"无效的日期时间: {dt_str}\n格式: YYYY-MM-DDTHH:MM"

        if dt.timestamp() < time.time():
            return "指定的时间已过去。"

        if ss.get_by_name(name):
            return f"已存在同名定时任务: {name}"

        sid = ss.add(name, task_text, dt_str, schedule_type="once", chat_id=chat_id)
        return f"已添加一次性任务: {name}\n时间: {dt_str}\n任务: {task_text}"

    def _cmd_calendar(self) -> str:
        """List all scheduled tasks, ordered by next_run_at."""
        ss = self.state.scheduled_store
        if not ss:
            return "定时任务模块不可用。"
        tasks = ss.get_all()
        if not tasks:
            return "暂无定时任务。"

        from datetime import datetime
        try:
            from ring0.cron import describe as _cron_desc
        except ImportError:
            _cron_desc = lambda x: x

        lines = ["*日历 (Calendar):*"]
        for t in tasks:
            icon = "🟢" if t["enabled"] else "⚪"
            name = t["name"]
            if t["schedule_type"] == "cron":
                schedule_desc = _cron_desc(t["cron_expr"])
            else:
                schedule_desc = f"一次性 {t['cron_expr']}"
            disabled_tag = " (已禁用)" if not t["enabled"] else ""
            next_at = ""
            if t["next_run_at"]:
                next_dt = datetime.fromtimestamp(t["next_run_at"])
                next_at = f" — 下次: {next_dt.strftime('%Y-%m-%d %H:%M')}"
            runs = f" [{t['run_count']}次]" if t["run_count"] > 0 else ""
            lines.append(f"{icon} {name} — {schedule_desc}{next_at}{runs}{disabled_tag}")
        return "\n".join(lines)

    def _enqueue_task(self, text: str, chat_id: str,
                      reply_to_message_id: int | None = None) -> str:
        """Create a Task, enqueue it, pulse p0_event, return ack."""
        task = Task(text=text, chat_id=chat_id, reply_to_message_id=reply_to_message_id)
        ts = self.state.task_store
        if ts:
            try:
                ts.add(task.task_id, task.text, task.chat_id, task.created_at)
            except Exception:
                log.debug("Failed to persist task", exc_info=True)
        self.state.task_queue.put(task)
        self.state.p0_event.set()  # wake sentinel for P0 scheduling
        return f"收到 — 正在处理你的请求 ({task.task_id})..."

    def _handle_callback(self, data: str) -> str:
        """Handle an inline keyboard callback by prefix.

        ``data`` format: ``run:<name>``, ``skill:<name>``,
        ``convergence:…``, or ``feedback:positive|negative``.
        Returns a text reply.
        """
        # --- nudge callbacks ---
        if data.startswith("nudge:"):
            return self._handle_nudge_callback(data)

        # --- feedback callbacks ---
        if data == "feedback:positive":
            return "\U0001f44d 谢谢反馈！"
        if data == "feedback:negative":
            # Record negative feedback as a drift event if preference_store is available.
            try:
                ms = self.state.memory_store
                if ms:
                    recent = ms.get_by_type("task", limit=1)
                    if recent:
                        content = recent[0].get("content", "")[:100]
                        log.info("Negative feedback on task: %s", content)
                        # Store as a preference moment of type 'feedback'.
                        ps = getattr(self.state, "_preference_store", None)
                        if ps:
                            ps.store_moment(
                                moment_type="feedback",
                                content=f"negative feedback on: {content}",
                                category="general",
                                extracted_signal="user dissatisfied with response",
                            )
            except Exception:
                log.debug("Feedback processing failed", exc_info=True)
            return "\U0001f44e 收到，我会改进的！"

        if data.startswith("convergence:confirm:"):
            rule_key = data[len("convergence:confirm:"):]
            ctx = self.state._convergence_context.pop(rule_key, None)
            if not ctx:
                return "规则已过期。"
            ms = self.state.memory_store
            if ms:
                ms.add(
                    generation=self.state.snapshot().get("generation", 0),
                    entry_type="semantic_rule",
                    content=ctx["rule_text"],
                    importance=0.9,
                    metadata={"source": "convergence", "cluster_size": ctx["cluster_size"]},
                )
            return f"已固化规则：{ctx['rule_text'][:80]}"

        if data.startswith("convergence:dismiss:"):
            rule_key = data[len("convergence:dismiss:"):]
            self.state._convergence_context.pop(rule_key, None)
            return "好的，不保存这条规则。"

        # --- output feedback callbacks ---
        if data.startswith("evo:accept:"):
            item_id = int(data.split(":")[-1])
            oq = getattr(self.state, 'output_queue', None)
            if oq:
                oq.mark_feedback(item_id, "accepted")
            return "\U0001f44d Noted!"

        if data.startswith("evo:reject:"):
            item_id = int(data.split(":")[-1])
            oq = getattr(self.state, 'output_queue', None)
            if oq:
                oq.mark_feedback(item_id, "rejected")
            return "\U0001f44e Noted, won't repeat."

        # --- habit callbacks ---
        if data.startswith("habit:schedule:"):
            # Format: habit:schedule:<template_key>:<cron_expr>[|<auto_stop_hours>]
            rest = data[len("habit:schedule:"):]
            # Split template key (template:xxx) from cron+options.
            # Template key always starts with "template:", so find the second "template:" boundary
            # or split on the cron part. Format: template:<name>:<cron>[|<hours>]
            # e.g. "template:flight_price_tracker:*/10 * * * *|2"
            parts = rest.split(":")
            # parts[0] = "template", parts[1] = name, parts[2:] = cron+options
            template_key = f"{parts[0]}:{parts[1]}"
            cron_and_opts = ":".join(parts[2:])
            auto_stop = 0
            if "|" in cron_and_opts:
                cron_expr, stop_str = cron_and_opts.rsplit("|", 1)
                try:
                    auto_stop = int(stop_str)
                except ValueError:
                    pass
            else:
                cron_expr = cron_and_opts
            self.state._pending_habits[template_key] = {
                "cron": cron_expr,
                "auto_stop_hours": auto_stop,
                "timestamp": time.time(),
            }
            ctx = self.state._habit_context.get(template_key, {})
            prompt = ctx.get("clarification_prompt", "请具体描述一下你想要自动执行的内容：")
            return prompt

        if data.startswith("habit:dismiss:"):
            template_key = data[len("habit:dismiss:"):]
            self.state._habit_dismissed.add(template_key)
            return "好的，不再提醒这个习惯。"

        if data.startswith("run:"):
            name = data[4:]
            sr = self.state.skill_runner
            if not sr:
                return "技能运行器不可用。"
            ss = self.state.skill_store
            if not ss:
                return "技能库不可用。"
            skill = ss.get_by_name(name)
            if not skill:
                return f"技能 '{name}' 未找到。"
            source_code = skill.get("source_code", "")
            if not source_code:
                return f"技能 '{name}' 没有源码。"
            pid, msg = sr.run(name, source_code)
            ss.update_usage(name)
            return msg
        if data.startswith("skill:"):
            name = data[6:]
            ss = self.state.skill_store
            if not ss:
                return "技能库不可用。"
            skill = ss.get_by_name(name)
            if not skill:
                return f"技能 '{name}' 未找到。"
            lines = [
                f"*技能: {skill['name']}*",
                f"描述 (Description): {skill['description']}",
                f"来源 (Source): {skill['source']}",
                f"已使用 (Used): {skill['usage_count']} 次",
                f"激活 (Active): {'是' if skill['active'] else '否'}",
                "",
                "提示词模板 (Prompt template):",
                f"```\n{skill['prompt_template']}\n```",
            ]
            return "\n".join(lines)
        return "未知操作。"

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
        "/skills": "_cmd_skills",
        "/skill": "_cmd_skill",  # Added missing command
        "/run": "_cmd_run",      # Added missing command
        "/stop": "_cmd_stop_skill",
        "/running": "_cmd_running",
        "/background": "_cmd_background",
        "/files": "_cmd_files",
        "/find": "_cmd_find",    # Added missing command
        "/calendar": "_cmd_calendar",
    }

    def _detect_conversation_context(self, msg: dict) -> dict | None:
        """检测用户消息是否在回复 bot 之前的消息，返回上下文信息。
        
        检测规则：
        1. 用户在 Telegram 中明确 reply 了 bot 的某条消息（reply_to_message）
        2. 用户发送的消息在时间窗口内（紧接着 bot 的消息）
        
        返回格式：
        {
            "replied_text": "bot 之前说的话",
            "replied_message_id": 123,
            "time_delta": 5.2,  # 秒
            "match_type": "explicit_reply" | "time_window"
        }
        """
        current_time = time.time()
        
        # 规则1: 检查是否明确回复了某条消息
        reply_to = msg.get("reply_to_message")
        if reply_to:
            # 确认是回复 bot 的消息（不是回复其他用户）
            reply_from = reply_to.get("from", {})
            if reply_from.get("is_bot"):
                replied_msg_id = reply_to.get("message_id")
                replied_text = reply_to.get("text", "")
                
                # 在我们的历史记录中找到这条消息
                for bot_msg in self._last_bot_messages:
                    if bot_msg["message_id"] == replied_msg_id:
                        return {
                            "replied_text": bot_msg["text"],
                            "replied_message_id": replied_msg_id,
                            "time_delta": current_time - bot_msg["timestamp"],
                            "match_type": "explicit_reply"
                        }
                
                # 即使不在历史记录中，也返回基本信息
                if replied_text:
                    return {
                        "replied_text": replied_text,
                        "replied_message_id": replied_msg_id,
                        "time_delta": 0,
                        "match_type": "explicit_reply"
                    }
        
        # 规则2: 检查时间窗口（最近一条消息）
        if self._last_bot_messages:
            last_msg = self._last_bot_messages[-1]
            time_delta = current_time - last_msg["timestamp"]
            
            if time_delta <= self._context_window_seconds:
                return {
                    "replied_text": last_msg["text"],
                    "replied_message_id": last_msg["message_id"],
                    "time_delta": time_delta,
                    "match_type": "time_window"
                }
        
        return None
    
    def _enrich_text_with_context(self, user_text: str, context: dict) -> str:
        """将上下文信息注入到用户输入中，帮助 LLM 理解对话连续性。"""
        match_type = context.get("match_type", "unknown")
        replied_text = context.get("replied_text", "")
        time_delta = context.get("time_delta", 0)
        
        # 截取回复内容的前 200 个字符（避免过长）
        replied_preview = replied_text[:200]
        if len(replied_text) > 200:
            replied_preview += "..."
        
        if match_type == "explicit_reply":
            # 用户明确回复了某条消息
            prefix = (
                f"[Context: User is replying to your previous message]\n"
                f"Your message: \"{replied_preview}\"\n"
                f"User's reply: "
            )
        else:
            # 时间窗口内的连续对话
            prefix = (
                f"[Context: User sent this {int(time_delta)}s after your last message]\n"
                f"Your previous message: \"{replied_preview}\"\n"
                f"User now says: "
            )
        
        return prefix + user_text

    def _handle_command(self, text: str, chat_id: str = "",
                        reply_to_message_id: int | None = None) -> str:
        """Dispatch a command or free-text message and return the response."""
        stripped = text.strip()
        if not stripped:
            return self._cmd_help()

        # Free text (not a command) → enqueue as P0 task
        if not stripped.startswith("/"):
            return self._enqueue_task(stripped, chat_id, reply_to_message_id)

        # /direct, /skill, /run, /find need special handling (passes full text)
        first_word = stripped.split()[0].lower().split("@")[0]
        if first_word == "/direct":
            return self._cmd_direct(stripped)
        if first_word == "/skill":
            return self._cmd_skill(stripped)
        if first_word == "/run":
            return self._cmd_run(stripped)
        if first_word == "/find":
            return self._cmd_find(stripped)
        if first_word == "/schedule":
            return self._cmd_schedule(stripped, chat_id=chat_id)

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
                        msg_dbg = update.get("message", {})
                        chat_dbg = msg_dbg.get("chat", {})
                        log.info("Update received: chat_id=%s type=%s text=%s",
                                 chat_dbg.get("id"), chat_dbg.get("type"),
                                 (msg_dbg.get("text", "") or "")[:80])
                        if not self._is_authorized(update):
                            log.info("Ignoring unauthorized update from chat_id=%s", chat_dbg.get("id"))
                            continue

                        # --- callback_query (inline keyboard press) ---
                        cb = update.get("callback_query")
                        if cb:
                            self._answer_callback_query(str(cb["id"]))
                            reply = self._handle_callback(cb.get("data", ""))
                            if reply:
                                cb_chat = cb.get("message", {}).get("chat", {})
                                cb_type = cb_chat.get("type", "private")
                                if cb_type in ("group", "supergroup"):
                                    self._send_reply(reply,
                                                     chat_id=str(cb_chat.get("id", "")),
                                                     reply_to_message_id=cb.get("message", {}).get("message_id"))
                                else:
                                    self._send_reply(reply)
                            continue

                        # --- regular message ---
                        msg = update.get("message", {})
                        chat_info = msg.get("chat", {})
                        msg_chat_id = str(chat_info.get("id", ""))
                        chat_type = chat_info.get("type", "private")
                        is_group = chat_type in ("group", "supergroup")

                        # Group filter: fast rules + LLM triage
                        if is_group and not self._should_respond_in_group(msg):
                            log.info("Group msg filtered out: chat_id=%s text=%s",
                                     msg_chat_id, (msg.get("text", "") or "")[:80])
                            continue

                        caption = msg.get("caption", "")
                        
                        # Check for various file types
                        handled = False
                        
                        # Group-aware reply helper
                        _reply_kw = {}
                        if is_group:
                            _reply_kw = {"chat_id": msg_chat_id,
                                         "reply_to_message_id": msg.get("message_id")}

                        # Helper: process file result tuple and optionally enqueue caption as task
                        def _process_file(file_info, file_type):
                            reply, saved_path = self._handle_file(file_info, file_type, msg_chat_id, caption)
                            if reply:
                                self._send_reply(reply, **_reply_kw)
                            if caption and saved_path and not caption.strip().startswith("/"):
                                task_text = f"[文件已上传: {saved_path}]\n\n{caption}"
                                ack = self._enqueue_task(task_text, msg_chat_id,
                                                         reply_to_message_id=msg.get("message_id") if is_group else None)
                                self._send_reply(ack, **_reply_kw)

                        # 1. Document (any file uploaded as document)
                        document = msg.get("document")
                        if document:
                            _process_file(document, "document")
                            handled = True

                        # 2. Photo (images)
                        if not handled and "photo" in msg:
                            # Telegram sends multiple sizes, get the largest
                            photos = msg["photo"]
                            if photos:
                                largest_photo = max(photos, key=lambda p: p.get("file_size", 0))
                                _process_file(largest_photo, "photo")
                                handled = True

                        # 3. Audio (music files with metadata)
                        if not handled and "audio" in msg:
                            _process_file(msg["audio"], "audio")
                            handled = True

                        # 4. Video
                        if not handled and "video" in msg:
                            _process_file(msg["video"], "video")
                            handled = True

                        # 5. Voice message
                        if not handled and "voice" in msg:
                            _process_file(msg["voice"], "voice")
                            handled = True

                        # 6. Video note (circular video)
                        if not handled and "video_note" in msg:
                            _process_file(msg["video_note"], "video_note")
                            handled = True

                        # 7. Sticker
                        if not handled and "sticker" in msg:
                            _process_file(msg["sticker"], "sticker")
                            handled = True
                        
                        # If file was handled, skip text processing
                        if handled:
                            continue
                        
                        # Check for text message
                        text = msg.get("text", "")
                        if not text:
                            continue

                        # Check for pending soul onboarding reply
                        pending_soul = getattr(self.state, '_pending_soul_question', None)
                        if (pending_soul and text.strip()
                                and not text.strip().startswith("/")):
                            answer = text.strip()
                            if answer.lower() in ("跳过", "skip"):
                                self._send_reply("好的，跳过。", **_reply_kw)
                            else:
                                try:
                                    from ring1 import soul
                                    soul.write_field(pending_soul["field"], answer)
                                    self._send_reply("✅ 已记录，谢谢！", **_reply_kw)
                                except Exception:
                                    log.debug("Soul write_field failed", exc_info=True)
                                    self._send_reply("记录失败，请稍后重试。", **_reply_kw)
                            self.state._pending_soul_question = None
                            handled = True
                            continue

                        # Strip @botname from group messages
                        if is_group and self.bot_username:
                            text = text.replace(f"@{self.bot_username}", "").strip()

                        # 上下文增强：检测用户是否在回复 bot 的消息 (private chat only)
                        if not is_group:
                            context_info = self._detect_conversation_context(msg)
                            if context_info:
                                text = self._enrich_text_with_context(text, context_info)

                        msg_reply_to = msg.get("message_id") if is_group else None
                        reply = self._handle_command(
                            text, chat_id=msg_chat_id,
                            reply_to_message_id=msg_reply_to,
                        )
                        if reply is not None:
                            if is_group:
                                self._send_reply(reply, chat_id=msg_chat_id,
                                                 reply_to_message_id=msg.get("message_id"))
                            else:
                                self._send_reply(reply)
                    except Exception:
                        log.debug("Error handling update", exc_info=True)

                # Consume convergence proposals from the executor.
                while not self.state.convergence_proposals.empty():
                    try:
                        text, buttons = self.state.convergence_proposals.get_nowait()
                        self._send_message_with_keyboard(text, buttons)
                    except queue.Empty:
                        break
                    except Exception:
                        log.debug("Error sending convergence proposal", exc_info=True)
                        break

                # Consume nudge suggestions from the executor.
                while not self.state.nudge_queue.empty():
                    try:
                        text, buttons, nudge_meta = self.state.nudge_queue.get_nowait()
                        # Store context for callback handling.
                        h = nudge_meta.get("task_hash", "")
                        if h:
                            self.state.set_nudge_context(h, nudge_meta)
                        self._send_message_with_keyboard(text, buttons)
                    except queue.Empty:
                        break
                    except Exception:
                        log.debug("Error sending nudge", exc_info=True)
                        break

                # Proactive nudge check (periodic, during active usage).
                # Runs in a background thread to avoid blocking the poll loop
                # when the LLM API is slow or timing out.
                _nudge_engine = getattr(self, "_nudge_engine", None)
                if _nudge_engine and not getattr(self, "_proactive_nudge_running", False):
                    _nudge_interval = getattr(self, "_nudge_interval", 600)
                    _last_proactive = getattr(self, "_last_nudge_proactive", 0.0)
                    if time.time() - _last_proactive >= _nudge_interval:
                        last_active = getattr(self.state, "last_task_completion", 0)
                        if time.time() - last_active < 1800:
                            self._last_nudge_proactive = time.time()
                            self._proactive_nudge_running = True
                            def _run_proactive(bot_self, engine):
                                try:
                                    result = engine.proactive_nudge()
                                    if result:
                                        text, buttons, nudge_meta = result
                                        h = nudge_meta.get("task_hash", "")
                                        if h:
                                            bot_self.state.set_nudge_context(h, nudge_meta)
                                        bot_self._send_message_with_keyboard(text, buttons)
                                except Exception:
                                    log.debug("Proactive nudge failed", exc_info=True)
                                finally:
                                    bot_self._proactive_nudge_running = False
                            t = threading.Thread(
                                target=_run_proactive, args=(self, _nudge_engine),
                                daemon=True,
                            )
                            t.start()

                # Deliver pending evolution outputs to user.
                try:
                    self.deliver_evolution_outputs()
                except Exception:
                    log.debug("Error delivering evolution outputs", exc_info=True)
            except Exception:
                log.warning("Error in polling loop", exc_info=True)
                # Back off on repeated errors.
                if self._running.is_set():
                    time.sleep(5)
        log.info("Telegram bot stopped")

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._running.clear()


# ---------------------------------------------------------------------------
# Schedule text parser
# ---------------------------------------------------------------------------

def _parse_schedule_text(text: str) -> str | None:
    """Parse Chinese/English schedule text into a cron expression.

    Returns a cron string or None if unrecognized.
    """
    t = text.strip().lower()
    _MAP = {
        "每小时": "0 * * * *",
        "每天": "0 9 * * *",
        "每日": "0 9 * * *",
        "每周": "0 9 * * 1",
        "每周一": "0 9 * * 1",
        "每周二": "0 9 * * 2",
        "每周三": "0 9 * * 3",
        "每周四": "0 9 * * 4",
        "每周五": "0 9 * * 5",
        "每周六": "0 9 * * 6",
        "每周日": "0 9 * * 0",
        "每月": "0 9 1 * *",
        "hourly": "0 * * * *",
        "daily": "0 9 * * *",
        "weekly": "0 9 * * 1",
        "monthly": "0 9 1 * *",
    }
    return _MAP.get(t)


# ---------------------------------------------------------------------------
# Factory + thread launcher
# ---------------------------------------------------------------------------

def create_bot(config, state: SentinelState, fitness, ring2_path: pathlib.Path) -> TelegramBot | None:
    """Create a TelegramBot from Ring1Config, or None if disabled/missing.

    ``chat_id`` may be empty — the bot will auto-detect it from the first
    incoming message.
    """
    if not config.telegram_enabled:
        return None
    if not config.telegram_bot_token:
        log.warning("Telegram bot: enabled but token missing — disabled")
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
