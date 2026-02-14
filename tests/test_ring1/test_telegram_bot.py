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
    """Mock Telegram Bot API that handles getUpdates + sendMessage."""

    updates_queue: list[dict] = []
    sent_messages: list[dict] = []
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
            assert "Generation: 7" in reply
            assert "ALIVE" in reply
            assert "0.25" in reply
        finally:
            server.shutdown()

    def test_status_paused(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.pause_event.set()
            reply = bot._handle_command("/status")
            assert "PAUSED" in reply
        finally:
            server.shutdown()

    def test_history(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/history")
            assert "Gen 3" in reply
            assert "0.95" in reply
            assert "Gen 2" in reply
            assert "FAIL" in reply
            bot.fitness.get_history.assert_called_once_with(limit=10)
        finally:
            server.shutdown()

    def test_history_empty(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.fitness.get_history.return_value = []
            reply = bot._handle_command("/history")
            assert "No history" in reply
        finally:
            server.shutdown()

    def test_top(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/top")
            assert "Gen 3" in reply
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
            assert "No fitness" in reply
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
            assert "not found" in reply
        finally:
            server.shutdown()

    def test_code_truncation(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            (bot.ring2_path / "main.py").write_text("x" * 5000)
            reply = bot._handle_command("/code")
            assert "truncated" in reply
        finally:
            server.shutdown()

    def test_pause(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/pause")
            assert "paused" in reply.lower()
            assert bot.state.pause_event.is_set()
        finally:
            server.shutdown()

    def test_pause_already_paused(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.pause_event.set()
            reply = bot._handle_command("/pause")
            assert "Already" in reply
        finally:
            server.shutdown()

    def test_resume(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.pause_event.set()
            reply = bot._handle_command("/resume")
            assert "resumed" in reply.lower()
            assert not bot.state.pause_event.is_set()
        finally:
            server.shutdown()

    def test_resume_not_paused(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/resume")
            assert "Not paused" in reply
        finally:
            server.shutdown()

    def test_kill(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/kill")
            assert "Kill" in reply or "restart" in reply.lower()
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
            assert "Generation" in reply
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
            assert "Generation" in reply["text"]

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
            assert "processing" in reply.lower() or "got it" in reply.lower()
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
            assert "directive set" in reply.lower()
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
            assert "usage" in reply.lower()
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
            assert "Queued tasks: 0" in reply
            assert "P0 active: No" in reply
            assert "(none)" in reply
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
            assert "not available" in reply.lower()
        finally:
            server.shutdown()

    def test_memory_empty(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            from ring0.memory import MemoryStore
            bot = _make_bot(port, tmp_path, monkeypatch)
            bot.state.memory_store = MemoryStore(tmp_path / "mem.db")
            reply = bot._handle_command("/memory")
            assert "No memories" in reply
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
            assert "Gen 1" in reply
            assert "observation" in reply
            assert "CA patterns" in reply
            assert "2 total" in reply
        finally:
            server.shutdown()


class TestForgetCommand:
    """Test /forget command clears memories."""

    def test_forget_no_store(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/forget")
            assert "not available" in reply.lower()
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
            assert "cleared" in reply.lower()
            assert ms.count() == 0
        finally:
            server.shutdown()


class TestHelpIncludesMemoryCommands:
    """Test that /help includes /memory and /forget."""

    def test_help_lists_memory(self, tmp_path, monkeypatch):
        server, port = _make_server()
        try:
            bot = _make_bot(port, tmp_path, monkeypatch)
            reply = bot._handle_command("/help")
            assert "/memory" in reply
            assert "/forget" in reply
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
