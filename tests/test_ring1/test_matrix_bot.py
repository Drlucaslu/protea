"""Tests for ring1.matrix_bot."""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from ring1.matrix_bot import MatrixBot
from ring1.telegram_bot import SentinelState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    s = SentinelState()
    s.memory_store = None
    s.skill_store = None
    s.task_store = None
    s.scheduled_store = None
    return s


@pytest.fixture
def bot(state):
    return MatrixBot(
        homeserver="https://matrix.example.com",
        access_token="test-token",
        room_id="!test:example.com",
        state=state,
    )


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------

class TestInit:
    def test_homeserver_trailing_slash_stripped(self, state):
        bot = MatrixBot("https://matrix.org/", "tok", "!r:m.org", state)
        assert bot.homeserver == "https://matrix.org"

    def test_attributes_set(self, bot):
        assert bot.access_token == "test-token"
        assert bot.room_id == "!test:example.com"
        assert bot._since == ""

    def test_running_initially_set(self, bot):
        assert bot._running.is_set()


# ---------------------------------------------------------------------------
# TestCommandDispatch
# ---------------------------------------------------------------------------

class TestCommandDispatch:
    def test_help(self, bot):
        reply = bot._handle_command("/help")
        assert "Protea" in reply
        assert "/status" in reply

    def test_status(self, bot):
        reply = bot._handle_command("/status")
        assert "Generation" in reply
        assert "Status" in reply

    def test_tasks(self, bot):
        reply = bot._handle_command("/tasks")
        assert "Queue" in reply

    def test_skills_no_store(self, bot):
        reply = bot._handle_command("/skills")
        assert "not available" in reply

    def test_skills_with_store(self, bot):
        mock_store = MagicMock()
        mock_store.get_active.return_value = [
            {"name": "test-skill", "description": "A test", "usage_count": 3},
        ]
        bot.state.skill_store = mock_store
        reply = bot._handle_command("/skills")
        assert "test-skill" in reply

    def test_calendar_no_store(self, bot):
        reply = bot._handle_command("/calendar")
        assert "not available" in reply

    def test_calendar_with_store(self, bot):
        mock_store = MagicMock()
        mock_store.get_all.return_value = [
            {
                "name": "daily-news",
                "cron_expr": "30 9 * * *",
                "schedule_type": "cron",
                "enabled": 1,
                "next_run_at": time.time() + 3600,
                "run_count": 5,
            },
        ]
        bot.state.scheduled_store = mock_store
        reply = bot._handle_command("/calendar")
        assert "daily-news" in reply

    def test_unknown_command_returns_help(self, bot):
        reply = bot._handle_command("/unknown")
        assert "/status" in reply

    def test_free_text_enqueues_task(self, bot):
        reply = bot._handle_command("hello world")
        assert "Received" in reply
        assert not bot.state.task_queue.empty()

    def test_empty_returns_help(self, bot):
        reply = bot._handle_command("")
        assert "/status" in reply


# ---------------------------------------------------------------------------
# TestScheduleCommand
# ---------------------------------------------------------------------------

class TestScheduleCommand:
    def test_schedule_no_store(self, bot):
        reply = bot._handle_command("/schedule list")
        assert "not available" in reply

    def test_schedule_list(self, bot):
        mock_store = MagicMock()
        mock_store.get_all.return_value = []
        bot.state.scheduled_store = mock_store
        reply = bot._handle_command("/schedule list")
        assert "No scheduled" in reply

    def test_schedule_remove(self, bot):
        mock_store = MagicMock()
        mock_store.get_by_name.return_value = {"schedule_id": "sched-123", "name": "test"}
        bot.state.scheduled_store = mock_store
        reply = bot._handle_command("/schedule remove test")
        assert "Removed" in reply
        mock_store.remove.assert_called_once_with("sched-123")

    def test_schedule_remove_not_found(self, bot):
        mock_store = MagicMock()
        mock_store.get_by_name.return_value = None
        bot.state.scheduled_store = mock_store
        reply = bot._handle_command("/schedule remove nonexistent")
        assert "not found" in reply

    def test_schedule_enable(self, bot):
        mock_store = MagicMock()
        mock_store.get_by_name.return_value = {"schedule_id": "sched-123", "name": "test"}
        bot.state.scheduled_store = mock_store
        reply = bot._handle_command("/schedule enable test")
        assert "Enabled" in reply

    def test_schedule_disable(self, bot):
        mock_store = MagicMock()
        mock_store.get_by_name.return_value = {"schedule_id": "sched-123", "name": "test"}
        bot.state.scheduled_store = mock_store
        reply = bot._handle_command("/schedule disable test")
        assert "Disabled" in reply

    def test_schedule_unknown_subcmd(self, bot):
        mock_store = MagicMock()
        bot.state.scheduled_store = mock_store
        reply = bot._handle_command("/schedule foobar")
        assert "Usage" in reply


# ---------------------------------------------------------------------------
# TestEnqueueTask
# ---------------------------------------------------------------------------

class TestEnqueueTask:
    def test_enqueue_sets_p0_event(self, bot):
        assert not bot.state.p0_event.is_set()
        bot._enqueue_task("test task")
        assert bot.state.p0_event.is_set()

    def test_enqueue_with_task_store(self, bot):
        mock_ts = MagicMock()
        bot.state.task_store = mock_ts
        bot._enqueue_task("stored task")
        assert mock_ts.add.called

    def test_enqueue_returns_ack(self, bot):
        reply = bot._enqueue_task("hello")
        assert "processing" in reply.lower() or "received" in reply.lower()


# ---------------------------------------------------------------------------
# TestStop
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_clears_running(self, bot):
        assert bot._running.is_set()
        bot.stop()
        assert not bot._running.is_set()


# ---------------------------------------------------------------------------
# TestApiRequest
# ---------------------------------------------------------------------------

class TestApiRequest:
    @patch("ring1.matrix_bot.urllib.request.urlopen")
    def test_api_request_success(self, mock_urlopen, bot):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = bot._api_request("GET", "/test")
        assert result == {"ok": True}

    @patch("ring1.matrix_bot.urllib.request.urlopen")
    def test_api_request_failure(self, mock_urlopen, bot):
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        result = bot._api_request("GET", "/test")
        assert result is None

    def test_send_reply_constructs_correct_path(self, bot):
        with patch.object(bot, "_api_request") as mock:
            bot._send_reply("hello")
            call_args = mock.call_args
            assert "PUT" == call_args[0][0]
            assert "!test:example.com" in call_args[0][1]
            body = call_args[1]["body"]  # keyword arg
            assert body["body"] == "hello"
            assert body["msgtype"] == "m.text"
