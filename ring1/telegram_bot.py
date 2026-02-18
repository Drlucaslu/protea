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
        # Store references
        "memory_store", "skill_store", "task_store", "scheduled_store",
        # Service references
        "notifier", "skill_runner", "registry_client", "subagent_manager",
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
                "p1_active": self.p1_active.is_set(),
                "evolution_directive": self.evolution_directive,
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
        """Send a text reply (fire-and-forget).

        Tries Markdown first; falls back to plain text if Telegram rejects it
        (e.g. LLM responses often contain ``##`` headers that are invalid in
        Telegram's legacy Markdown mode).
        """
        result = self._api_call("sendMessage", {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        })
        if result is None:
            # Markdown was rejected — retry as plain text.
            self._api_call("sendMessage", {
                "chat_id": self.chat_id,
                "text": text,
            })

    def _send_message_with_keyboard(self, text: str, buttons: list[list[dict]]) -> None:
        """Send a message with an inline keyboard (fire-and-forget).

        *buttons* is a list of rows, each row a list of dicts with
        ``text`` and ``callback_data`` keys.
        """
        self._api_call("sendMessage", {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": json.dumps({"inline_keyboard": buttons}),
        })

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

    def _handle_file(self, file_info: dict, file_type: str, msg_chat_id: str, caption: str = "") -> str:
        """Handle any file upload (document, photo, audio, video, voice).
        
        Args:
            file_info: dict containing file_id, file_name (or generated), file_size
            file_type: "document", "photo", "audio", "video", "voice"
            msg_chat_id: chat ID
            caption: optional caption from message
        
        Returns: Success or error message
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
            return "⚠️ 文件 ID 缺失。"
        
        # Download file
        file_bytes = self._download_file(file_id)
        if file_bytes is None:
            return "⚠️ 文件下载失败。"
        
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
            
            return response
        except Exception as e:
            log.error("Failed to save file", exc_info=True)
            return f"⚠️ 保存文件失败: {str(e)}"

    def _answer_callback_query(self, callback_query_id: str) -> None:
        """Acknowledge a callback query so Telegram stops showing a spinner."""
        self._api_call("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
        })

    def _is_authorized(self, update: dict) -> bool:
        """Check if the update comes from the authorized chat.

        When ``chat_id`` is empty (not yet configured), the first incoming
        message is accepted and its chat ID is locked as the authorized chat.
        """
        if "callback_query" in update:
            chat = update["callback_query"].get("message", {}).get("chat", {})
        else:
            chat = update.get("message", {}).get("chat", {})
        msg_chat_id = str(chat.get("id", ""))
        if not msg_chat_id:
            return False
        # Auto-detect: lock to the first sender when chat_id is not configured.
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
        directive = snap["evolution_directive"]
        lines.append(f"进化指令 (Directive): {directive if directive else '(无)'}")
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

    def _enqueue_task(self, text: str, chat_id: str) -> str:
        """Create a Task, enqueue it, pulse p0_event, return ack."""
        task = Task(text=text, chat_id=chat_id)
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

        ``data`` format: ``run:<name>`` or ``skill:<name>``.
        Returns a text reply.
        """
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

    def _handle_command(self, text: str, chat_id: str = "") -> str:
        """Dispatch a command or free-text message and return the response."""
        stripped = text.strip()
        if not stripped:
            return self._cmd_help()

        # Free text (not a command) → enqueue as P0 task
        if not stripped.startswith("/"):
            return self._enqueue_task(stripped, chat_id)

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
                        if not self._is_authorized(update):
                            log.debug("Ignoring unauthorized update")
                            continue

                        # --- callback_query (inline keyboard press) ---
                        cb = update.get("callback_query")
                        if cb:
                            self._answer_callback_query(str(cb["id"]))
                            reply = self._handle_callback(cb.get("data", ""))
                            if reply:
                                self._send_reply(reply)
                            continue

                        # --- regular message ---
                        msg = update.get("message", {})
                        msg_chat_id = str(msg.get("chat", {}).get("id", ""))
                        caption = msg.get("caption", "")
                        
                        # Check for various file types
                        handled = False
                        
                        # 1. Document (any file uploaded as document)
                        document = msg.get("document")
                        if document:
                            reply = self._handle_file(document, "document", msg_chat_id, caption)
                            if reply:
                                self._send_reply(reply)
                            handled = True
                        
                        # 2. Photo (images)
                        if not handled and "photo" in msg:
                            # Telegram sends multiple sizes, get the largest
                            photos = msg["photo"]
                            if photos:
                                largest_photo = max(photos, key=lambda p: p.get("file_size", 0))
                                reply = self._handle_file(largest_photo, "photo", msg_chat_id, caption)
                                if reply:
                                    self._send_reply(reply)
                                handled = True
                        
                        # 3. Audio (music files with metadata)
                        if not handled and "audio" in msg:
                            audio = msg["audio"]
                            reply = self._handle_file(audio, "audio", msg_chat_id, caption)
                            if reply:
                                self._send_reply(reply)
                            handled = True
                        
                        # 4. Video
                        if not handled and "video" in msg:
                            video = msg["video"]
                            reply = self._handle_file(video, "video", msg_chat_id, caption)
                            if reply:
                                self._send_reply(reply)
                            handled = True
                        
                        # 5. Voice message
                        if not handled and "voice" in msg:
                            voice = msg["voice"]
                            reply = self._handle_file(voice, "voice", msg_chat_id, caption)
                            if reply:
                                self._send_reply(reply)
                            handled = True
                        
                        # 6. Video note (circular video)
                        if not handled and "video_note" in msg:
                            video_note = msg["video_note"]
                            reply = self._handle_file(video_note, "video_note", msg_chat_id, caption)
                            if reply:
                                self._send_reply(reply)
                            handled = True
                        
                        # 7. Sticker
                        if not handled and "sticker" in msg:
                            sticker = msg["sticker"]
                            reply = self._handle_file(sticker, "sticker", msg_chat_id, caption)
                            if reply:
                                self._send_reply(reply)
                            handled = True
                        
                        # If file was handled, skip text processing
                        if handled:
                            continue
                        
                        # Check for text message
                        text = msg.get("text", "")
                        if not text:
                            continue
                        reply = self._handle_command(text, chat_id=msg_chat_id)
                        if reply is not None:
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
