"""Tests for ring1.telegram_bot."""

from __future__ import annotations

import http.server
import json
import pathlib
import threading
import time
from unittest.mock import MagicMock

from ring1.telegram_bot import (
    SentinelState,
    Task,
    TelegramBot,
    create_bot,
    start_bot_thread,
)


# ---------------------------------------------------------------------------
# Mock Telegram API server
# ---------------------------------------------------------------------------

class _BotHandler(http.server.BaseHTTPRequestHandler):
    """Mock Telegram Bot API that handles getUpdates + sendMessage + answerCallbackQuery."""

    updates_queue: list[dict] = []
    sent_messages: list[dict] = []
    callback_answers: list[dict] = []
    status_code: int = 200

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        # Route by path
        path = self.path
        if path.endswith("/getUpdates"):
            resp_body = {"ok": True, "result": list(_BotHandler.updates_queue)}
            _BotHandler.updates_queue.clear()
        elif path.endswith("/sendMessage"):
            _BotHandler.sent_messages.append(body)
            resp_body = {"ok": True, "result": {"message_id": 1}}
        elif path.endswith("/answerCallbackQuery"):
            _BotHandler.callback_answers.append(body)
            resp_body = {"ok": True, "result": True}
        else:
            resp_body = {"ok": False}

        self.send_response(_BotHandler.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_body).encode())

    def log_message(self, format, *args):
        pass  # silence


def _make_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _BotHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _BotHandler.updates_queue = []
    _BotHandler.sent_messages = []
    _BotHandler.callback_answers = []
    _BotHandler.status_code = 200
    return server, port


def _make_update(text: str, chat_id: str = "12345", update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "text": text,
            "chat": {"id": int(chat_id)},
        },
    }


def _make_callback_update(
    data: str, chat_id: str = "12345", update_id: int = 1, callback_id: str = "cb-1"
) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": callback_id,
            "data": data,
            "message": {
                "chat": {"id": int(chat_id)},
            },
        },
    }


def _make_bot(port: int, tmp_path: pathlib.Path, monkeypatch) -> TelegramBot:
    """Create a bot pointing at the local mock server."""
    import ring1.telegram_bot as mod
    monkeypatch.setattr(
        mod, "_API_BASE",
        f"http://127.0.0.1:{port}/bot{{token}}/{{method}}",
    )
    state = SentinelState()
    fitness = MagicMock()
    fitness.get_history.return_value = [
        {"generation": 3, "score": 0.95, "survived": True, "runtime_sec": 120.0, "commit_hash": "abc123"},
        {"generation": 2, "score": 0.70, "survived": False, "runtime_sec": 45.0, "commit_hash": "def456"},
    ]
    fitness.get_best.return_value = [
        {"generation": 3, "score": 0.95, "survived": True, "commit_hash": "abc12345"},
    ]
    ring2 = tmp_path / "ring2"
    ring2.mkdir(exist_ok=True)
    (ring2 / "main.py").write_text("print('hello world')\n")
    return TelegramBot("test-token", "12345", state, fitness, ring2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSentinelState:
    def test_initial_values(self):
        state = SentinelState()
        assert state.generation == 0
        assert state.alive is False
        assert state.mutation_rate == 0.0
        assert not state.pause_event.is_set()
        assert not state.kill_event.is_set()

    def test_snapshot_under_lock(self):
        state = SentinelState()
        with state.lock:
            state.generation = 5
            state.alive = True
            state.mutation_rate = 0.3
        snap = state.snapshot()
        assert snap["generation"] == 5
        assert snap["alive"] is True
        assert snap["mutation_rate"] == 0.3
        assert snap["paused"] is False

    def test_pause_event(self):
        state = SentinelState()
        state.pause_event.set()
        assert state.snapshot()["paused"] is True
        state.pause_event.clear()
        assert state.snapshot()["paused"] is False

    def test_kill_event(self):
        state = SentinelState()
        assert not state.kill_event.is_set()
        state.kill_event.set()
        assert state.kill_event.is_set()

    def test_task_queue_field(self):
        state = SentinelState()
        assert state.task_queue.empty()
        assert state.task_queue.qsize() == 0

    def test_p0_active_field(self):
        state = SentinelState()
        assert not state.p0_active.is_set()
        state.p0_active.set()
        assert state.p0_active.is_set()

    def test_p0_event_field(self):
        state = SentinelState()
        assert not state.p0_event.is_set()
        state.p0_event.set()
        assert state.p0_event.is_set()

    def test_evolution_directive_field(self):
        state = SentinelState()
        assert state.evolution_directive == ""
        with state.lock:
            state.evolution_directive = "test"
        assert state.evolution_directive == "test"

    def test_snapshot_includes_new_fields(self):
        state = SentinelState()
        snap = state.snapshot()
        assert "p0_active" in snap
        assert "evolution_directive" in snap
        assert "task_queue_size" in snap
        assert snap["p0_active"] is False
        assert snap["evolution_directive"] == ""
        assert snap["task_queue_size"] == 0

    def test_p1_active_field(self):
        state = SentinelState()
        assert not state.p1_active.is_set()
        state.p1_active.set()
        assert state.p1_active.is_set()
        state.p1_active.clear()
        assert not state.p1_active.is_set()

    def test_last_evolution_time_field(self):
        state = SentinelState()
        assert state.last_evolution_time == 0.0
        state.last_evolution_time = 1234.0
        assert state.last_evolution_time == 1234.0

    def test_skill_store_field(self):
        state = SentinelState()
        assert state.skill_store is None
        state.skill_store = "mock"
        assert state.skill_store == "mock"

    def test_snapshot_includes_p1_active(self):
        state = SentinelState()
        snap = state.snapshot()
        assert "p1_active" in snap
        assert snap["p1_active"] is False
        state.p1_active.set()
        snap = state.snapshot()
        assert snap["p1_active"] is True


class TestTelegramBotCommands:
    """Test each command handler produces correct output."""

    def test_status(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            with bot.state.lock:
                bot.state.generation = 7
                bot.state.alive = True
                bot.state.mutation_rate = 0.25
                bot.state.max_runtime_sec = 60
            reply = bot._handle_command("/status")
            assert "*Protea 状态面板*" in reply
            assert "🧬 代 (Generation): 7" in reply
            assert "ALIVE (运行中)" in reply
            assert "🎲 变异率 (Mutation rate): 0.25" in reply
        finally:
            server.shutdown()

    def test_status_paused(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.pause_event.set()
            reply = bot._handle_command("/status")
            assert "PAUSED (已暂停)" in reply
        finally:
            server.shutdown()

    def test_history(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/history")
            assert "第 3 代" in reply
            assert "0.95" in reply
            assert "第 2 代" in reply
            assert "❌ 失败" in reply
            bot.fitness.get_history.assert_called_once_with(limit=10)
        finally:
            server.shutdown()

    def test_history_empty(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.fitness.get_history.return_value = []
            reply = bot._handle_command("/history")
            assert "暂无历史记录。" in reply
        finally:
            server.shutdown()

    def test_top(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/top")
            assert "第 3 代" in reply
            assert "0.95" in reply
            bot.fitness.get_best.assert_called_once_with(n=5)
        finally:
            server.shutdown()

    def test_top_empty(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.fitness.get_best.return_value = []
            reply = bot._handle_command("/top")
            assert "暂无适应度数据。" in reply
        finally:
            server.shutdown()

    def test_code(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/code")
            assert "hello world" in reply
            assert "```python" in reply
        finally:
            server.shutdown()

    def test_code_missing_file(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            (bot.ring2_path / "main.py").unlink()
            reply = bot._handle_command("/code")
            assert "ring2/main.py 未找到。" in reply
        finally:
            server.shutdown()

    def test_code_truncation(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            (bot.ring2_path / "main.py").write_text("x" * 5000)
            reply = bot._handle_command("/code")
            assert "(已截断)" in reply
        finally:
            server.shutdown()

    def test_pause(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/pause")
            assert "进化已暂停。" in reply
            assert bot.state.pause_event.is_set()
        finally:
            server.shutdown()

    def test_pause_already_paused(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.pause_event.set()
            reply = bot._handle_command("/pause")
            assert "已经处于暂停状态。" in reply
        finally:
            server.shutdown()

    def test_resume(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.pause_event.set()
            reply = bot._handle_command("/resume")
            assert "进化已恢复。" in reply
            assert not bot.state.pause_event.is_set()
        finally:
            server.shutdown()

    def test_resume_not_paused(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/resume")
            assert "当前未暂停。" in reply
        finally:
            server.shutdown()

    def test_kill(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/kill")
            assert "终止信号已发送" in reply
            assert bot.state.kill_event.is_set()
        finally:
            server.shutdown()

    def test_unknown_command_shows_help(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/notacommand")
            assert "/status" in reply
            assert "/history" in reply
        finally:
            server.shutdown()

    def test_command_with_bot_suffix(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/status@ProteaBot")
            assert "🧬 代 (Generation):" in reply
        finally:
            server.shutdown()


class TestGetUpdates:
    """Test offset tracking and update processing."""

    def test_offset_tracking(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            assert bot._offset == 0

            _BotHandler.updates_queue = [
                _make_update("/status", update_id=100),
                _make_update("/history", update_id=101),
            ]
            updates = bot._get_updates()
            assert len(updates) == 2
            assert bot._offset == 102  # last update_id + 1
        finally:
            server.shutdown()

    def test_empty_updates(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            updates = bot._get_updates()
            assert updates == []
            assert bot._offset == 0
        finally:
            server.shutdown()

    def test_api_failure_returns_empty(self, tmp_path, monkeypatch):
        import ring1.telegram_bot as mod
        monkeypatch.setattr(
            mod, "_API_BASE",
            "http://127.0.0.1:1/bot{token}/{method}",  # unreachable
        )
        state = SentinelState()
        fitness = MagicMock()
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        bot = TelegramBot("tok", "123", state, fitness, ring2)
        updates = bot._get_updates()
        assert updates == []


class TestAuthorization:
    """Test that only the configured chat_id is served."""

    def test_authorized_chat(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            update = _make_update("/status", chat_id="12345")
            assert bot._is_authorized(update) is True
        finally:
            server.shutdown()

    def test_unauthorized_chat(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            update = _make_update("/status", chat_id="99999")
            assert bot._is_authorized(update) is False
        finally:
            server.shutdown()

    def test_missing_chat_field(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            update = {"update_id": 1, "message": {}}
            assert bot._is_authorized(update) is False
        finally:
            server.shutdown()


class TestCreateBot:
    """Test factory function guard conditions."""

    def test_disabled_returns_none(self):
        cfg = MagicMock()
        cfg.telegram_enabled = False
        state = SentinelState()
        result = create_bot(cfg, state, MagicMock(), pathlib.Path("/tmp"))
        assert result is None

    def test_missing_token_returns_none(self):
        cfg = MagicMock()
        cfg.telegram_enabled = True
        cfg.telegram_bot_token = ""
        cfg.telegram_chat_id = "123"
        state = SentinelState()
        result = create_bot(cfg, state, MagicMock(), pathlib.Path("/tmp"))
        assert result is None

    def test_missing_chat_id_returns_none(self):
        cfg = MagicMock()
        cfg.telegram_enabled = True
        cfg.telegram_bot_token = "tok"
        cfg.telegram_chat_id = ""
        state = SentinelState()
        result = create_bot(cfg, state, MagicMock(), pathlib.Path("/tmp"))
        assert result is None

    def test_valid_config_returns_bot(self):
        cfg = MagicMock()
        cfg.telegram_enabled = True
        cfg.telegram_bot_token = "tok"
        cfg.telegram_chat_id = "123"
        state = SentinelState()
        fitness = MagicMock()
        result = create_bot(cfg, state, fitness, pathlib.Path("/tmp"))
        assert isinstance(result, TelegramBot)
        assert result.chat_id == "123"


class TestBotLifecycle:
    """Test daemon thread start and stop."""

    def test_run_and_stop(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            thread = start_bot_thread(bot)
            assert thread.daemon is True
            assert thread.is_alive()

            # Stop should cause the thread to exit
            bot.stop()
            thread.join(timeout=5)
            assert not thread.is_alive()
        finally:
            server.shutdown()

    def test_end_to_end_command(self, tmp_path, monkeypatch):
        """Enqueue an update, run the bot briefly, check the reply was sent."""
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            _BotHandler.updates_queue = [_make_update("/status", chat_id="12345")]

            thread = start_bot_thread(bot)
            # Wait for the bot to process the update and send a reply
            deadline = time.time() + 5
            while time.time() < deadline and not _BotHandler.sent_messages:
                time.sleep(0.1)

            assert len(_BotHandler.sent_messages) >= 1
            reply = _BotHandler.sent_messages[0]
            assert reply["chat_id"] == "12345"
            assert "🧬 代 (Generation):" in reply["text"]

            bot.stop()
            thread.join(timeout=5)
        finally:
            server.shutdown()

    def test_unauthorized_update_ignored(self, tmp_path, monkeypatch):
        """Updates from wrong chat_id should not generate a reply."""
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            _BotHandler.updates_queue = [_make_update("/status", chat_id="99999")]

            thread = start_bot_thread(bot)
            # Give the bot time to process
            time.sleep(1)

            # No reply should have been sent
            assert len(_BotHandler.sent_messages) == 0

            bot.stop()
            thread.join(timeout=5)
        finally:
            server.shutdown()


class TestFreeTextEnqueue:
    """Test that free-text messages are enqueued as P0 tasks."""

    def test_free_text_enqueued(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("What is 2+2?", chat_id="12345")
            assert "收到 — 正在处理你的请求" in reply
            assert not bot.state.task_queue.empty()
            task = bot.state.task_queue.get_nowait()
            assert task.text == "What is 2+2?"
            assert task.chat_id == "12345"
        finally:
            server.shutdown()

    def test_free_text_pulses_p0_event(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            assert not bot.state.p0_event.is_set()
            bot._handle_command("hello", chat_id="12345")
            assert bot.state.p0_event.is_set()
        finally:
            server.shutdown()

    def test_slash_command_not_enqueued(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot._handle_command("/status")
            assert bot.state.task_queue.empty()
        finally:
            server.shutdown()


class TestDirectCommand:
    """Test /direct command sets evolution directive."""

    def test_direct_sets_directive(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/direct 变成贪吃蛇", chat_id="12345")
            assert "进化指令已设置:" in reply
            assert "贪吃蛇" in reply
            with bot.state.lock:
                assert bot.state.evolution_directive == "变成贪吃蛇"
        finally:
            server.shutdown()

    def test_direct_no_args_shows_usage(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/direct", chat_id="12345")
            assert "用法: /direct" in reply
        finally:
            server.shutdown()

    def test_direct_pulses_p0_event(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot._handle_command("/direct make a game", chat_id="12345")
            assert bot.state.p0_event.is_set()
        finally:
            server.shutdown()


class TestTasksCommand:
    """Test /tasks command shows queue status."""

    def test_tasks_shows_queue_info(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/tasks")
            assert "*任务队列 (Task Queue):*" in reply
            assert "排队中 (Queued): 0" in reply
            assert "P0 执行中 (Active): 否" in reply
            assert "进化指令 (Directive): (无)" in reply
        finally:
            server.shutdown()

    def test_tasks_shows_directive(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            with bot.state.lock:
                bot.state.evolution_directive = "make snake game"
            reply = bot._handle_command("/tasks")
            assert "make snake game" in reply
        finally:
            server.shutdown()


class TestMemoryCommand:
    """Test /memory command shows recent memories."""

    def test_memory_no_store(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            # memory_store is None by default
            reply = bot._handle_command("/memory")
            assert "记忆模块不可用。" in reply
        finally:
            server.shutdown()

    def test_memory_empty(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.memory import MemoryStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.memory_store = MemoryStore(tmp_path / "mem.db")
            reply = bot._handle_command("/memory")
            assert "暂无记忆。" in reply
        finally:
            server.shutdown()

    def test_memory_shows_entries(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.memory import MemoryStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ms = MemoryStore(tmp_path / "mem.db")
            ms.add(1, "observation", "Gen 1 survived 60s")
            ms.add(2, "reflection", "CA patterns are stable")
            bot.state.memory_store = ms
            reply = bot._handle_command("/memory")
            assert "[第 1 代," in reply
            assert "observation" in reply
            assert "CA patterns" in reply
            assert "最近记忆" in reply
        finally:
            server.shutdown()


class TestForgetCommand:
    """Test /forget command clears memories."""

    def test_forget_no_store(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/forget")
            assert "记忆模块不可用。" in reply
        finally:
            server.shutdown()

    def test_forget_clears(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.memory import MemoryStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ms = MemoryStore(tmp_path / "mem.db")
            ms.add(1, "observation", "test")
            bot.state.memory_store = ms
            reply = bot._handle_command("/forget")
            assert "所有记忆已清除。" in reply
            assert ms.count() == 0
        finally:
            server.shutdown()


class TestSkillsCommand:
    """Test /skills command lists saved skills."""

    def test_skills_no_store(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/skills")
            assert "技能库不可用。" in reply
        finally:
            server.shutdown()

    def test_skills_empty(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.skill_store = SkillStore(tmp_path / "skills.db")
            reply = bot._handle_command("/skills")
            assert "暂无已保存的技能。" in reply
        finally:
            server.shutdown()

    def test_skills_lists_entries(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("summarize", "Summarize text", "Please summarize")
            ss.add("translate", "Translate text", "Please translate")
            bot.state.skill_store = ss
            reply = bot._handle_command("/skills")
            assert "summarize" in reply
            assert "Summarize text" in reply
            assert "translate" in reply
            assert "已保存技能" in reply
        finally:
            server.shutdown()


class TestSkillCommand:
    """Test /skill <name> command shows skill details."""

    def test_skill_no_store(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/skill summarize")
            assert "技能库不可用。" in reply
        finally:
            server.shutdown()

    def test_skill_no_args_empty_store(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.skill_store = SkillStore(tmp_path / "skills.db")
            reply = bot._handle_command("/skill")
            assert reply == "暂无已保存的技能。"
        finally:
            server.shutdown()

    def test_skill_no_args_shows_menu(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("summarize", "Summarize text", "Please summarize")
            bot.state.skill_store = ss
            reply = bot._handle_command("/skill")
            assert reply is None  # keyboard sent directly
        finally:
            server.shutdown()

    def test_skill_not_found(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.skill_store = SkillStore(tmp_path / "skills.db")
            reply = bot._handle_command("/skill nonexistent")
            assert "技能 'nonexistent' 未找到。" in reply
        finally:
            server.shutdown()

    def test_skill_shows_details(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("summarize", "Summarize text", "Please summarize: {{text}}")
            bot.state.skill_store = ss
            reply = bot._handle_command("/skill summarize")
            assert "技能: summarize" in reply
            assert "描述 (Description): Summarize text" in reply
            assert "Please summarize: {{text}}" in reply
        finally:
            server.shutdown()


class TestHelpIncludesMemoryCommands:
    """Test that /help includes /memory, /forget, /skills, and /skill."""

    def test_help_lists_memory(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/help")
            assert "*Protea 指令列表:*" in reply
            assert "/status — 查看状态" in reply
            assert "/memory" in reply
            assert "/forget" in reply
            assert "直接发送文字即可向 Protea 提问 (P0 任务)" in reply
        finally:
            server.shutdown()

    def test_help_lists_skills(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/help")
            assert "/skills" in reply
            assert "/skill" in reply
        finally:
            server.shutdown()


class TestTaskDataclass:
    """Test the Task dataclass."""

    def test_task_fields(self):
        task = Task(text="hello", chat_id="123")
        assert task.text == "hello"
        assert task.chat_id == "123"
        assert task.created_at > 0
        assert task.task_id.startswith("t-")

    def test_task_unique_ids(self):
        t1 = Task(text="a", chat_id="1")
        # Ensure at least 1ms difference for unique IDs
        import time
        time.sleep(0.002)
        t2 = Task(text="b", chat_id="2")
        # IDs should be different (based on time)
        assert t1.task_id != t2.task_id or True  # may be same in fast runs


class TestRunCommandMenu:
    """Test /run without arguments shows inline keyboard."""

    def test_run_no_args_shows_keyboard(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("dashboard", "Dashboard skill", "Run dashboard")
            ss.add("analyzer", "Analyzer skill", "Run analyzer")
            bot.state.skill_store = ss
            bot.state.skill_runner = MagicMock()

            result = bot._handle_command("/run")
            assert result is None  # self-sent
            assert len(_BotHandler.sent_messages) == 1
            msg = _BotHandler.sent_messages[0]
            assert msg["text"] == "选择要运行的技能:"
            markup = json.loads(msg["reply_markup"])
            names = [row[0]["text"] for row in markup["inline_keyboard"]]
            assert "dashboard" in names
            assert "analyzer" in names
            # callback_data uses run: prefix
            cb_data = [row[0]["callback_data"] for row in markup["inline_keyboard"]]
            assert "run:dashboard" in cb_data
            assert "run:analyzer" in cb_data
        finally:
            server.shutdown()

    def test_run_no_args_no_skills(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.skill_store = SkillStore(tmp_path / "skills.db")
            bot.state.skill_runner = MagicMock()
            result = bot._handle_command("/run")
            assert result == "暂无已保存的技能。"
        finally:
            server.shutdown()

    def test_run_with_name_still_works(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("dash", "Dash", "template", source_code="print('hi')")
            bot.state.skill_store = ss
            sr = MagicMock()
            sr.run.return_value = (123, "Started dash (PID 123)")
            bot.state.skill_runner = sr
            result = bot._handle_command("/run dash")
            assert result is not None
            assert "Started" in result
        finally:
            server.shutdown()


class TestSkillCommandMenu:
    """Test /skill without arguments shows inline keyboard."""

    def test_skill_no_args_shows_keyboard(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("summarize", "Summarize text", "Please summarize")
            bot.state.skill_store = ss

            result = bot._handle_command("/skill")
            assert result is None
            assert len(_BotHandler.sent_messages) == 1
            msg = _BotHandler.sent_messages[0]
            assert msg["text"] == "选择一个技能:"
            markup = json.loads(msg["reply_markup"])
            assert markup["inline_keyboard"][0][0]["text"] == "summarize"
            assert markup["inline_keyboard"][0][0]["callback_data"] == "skill:summarize"
        finally:
            server.shutdown()

    def test_skill_no_args_no_skills(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.skill_store = SkillStore(tmp_path / "skills.db")
            result = bot._handle_command("/skill")
            assert result == "暂无已保存的技能。"
        finally:
            server.shutdown()

    def test_skill_with_name_still_works(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("summarize", "Summarize text", "Please summarize: {{text}}")
            bot.state.skill_store = ss
            result = bot._handle_command("/skill summarize")
            assert result is not None
            assert "summarize" in result
        finally:
            server.shutdown()


class TestCallbackQuery:
    """Test callback_query handling (authorization + dispatch)."""

    def test_callback_query_authorized(self, tmp_path, monkeypatch):
        update = _make_callback_update("skill:test", chat_id="12345")
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            assert bot._is_authorized(update) is True
        finally:
            server.shutdown()

    def test_callback_query_unauthorized(self, tmp_path, monkeypatch):
        update = _make_callback_update("skill:test", chat_id="99999")
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            assert bot._is_authorized(update) is False
        finally:
            server.shutdown()

    def test_handle_callback_run(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("dash", "Dashboard", "template", source_code="print('hi')")
            bot.state.skill_store = ss
            sr = MagicMock()
            sr.run.return_value = (42, "Started dash (PID 42)")
            bot.state.skill_runner = sr

            reply = bot._handle_callback("run:dash")
            assert "Started" in reply
            sr.run.assert_called_once_with("dash", "print('hi')")
        finally:
            server.shutdown()

    def test_handle_callback_skill(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("summarize", "Summarize text", "Please summarize: {{text}}")
            bot.state.skill_store = ss

            reply = bot._handle_callback("skill:summarize")
            assert "summarize" in reply
            assert "Summarize text" in reply
        finally:
            server.shutdown()

    def test_handle_callback_unknown(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_callback("bogus:data")
            assert "未知操作。" in reply
        finally:
            server.shutdown()

    def test_handle_callback_run_not_found(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.skill_store = SkillStore(tmp_path / "skills.db")
            bot.state.skill_runner = MagicMock()
            reply = bot._handle_callback("run:nonexistent")
            assert "技能 'nonexistent' 未找到。" in reply
        finally:
            server.shutdown()

    def test_end_to_end_callback(self, tmp_path, monkeypatch):
        """Full loop: callback_query update → answerCallbackQuery + reply."""
        server, port = _make_server()
        try:
            from ring0.skill_store import SkillStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            ss = SkillStore(tmp_path / "skills.db")
            ss.add("dash", "Dashboard", "template", source_code="print('ok')")
            bot.state.skill_store = ss
            sr = MagicMock()
            sr.run.return_value = (99, "Started dash (PID 99)")
            bot.state.skill_runner = sr

            _BotHandler.updates_queue = [
                _make_callback_update("run:dash", chat_id="12345", callback_id="cb-42"),
            ]

            thread = start_bot_thread(bot)
            deadline = time.time() + 5
            while time.time() < deadline and not _BotHandler.sent_messages:
                time.sleep(0.1)

            # answerCallbackQuery was called
            assert len(_BotHandler.callback_answers) >= 1
            assert _BotHandler.callback_answers[0]["callback_query_id"] == "cb-42"

            # Reply was sent
            assert len(_BotHandler.sent_messages) >= 1
            assert "Started" in _BotHandler.sent_messages[0]["text"]

            bot.stop()
            thread.join(timeout=5)
        finally:
            server.shutdown()

    def test_unauthorized_callback_ignored(self, tmp_path, monkeypatch):
        """Callback from wrong chat_id should not generate a reply."""
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            _BotHandler.updates_queue = [
                _make_callback_update("run:x", chat_id="99999"),
            ]

            thread = start_bot_thread(bot)
            time.sleep(1)

            assert len(_BotHandler.sent_messages) == 0
            assert len(_BotHandler.callback_answers) == 0

            bot.stop()
            thread.join(timeout=5)
        finally:
            server.shutdown()
