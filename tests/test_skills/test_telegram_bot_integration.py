"""Tests for skills.telegram_bot_integration — send_audio and audio support."""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import patch

# skills/ is a runtime directory that may not exist in CI.
skill_mod = __import__("pytest").importorskip(
    "skills.telegram_bot_integration",
    reason="skills/ directory not present (runtime artifact)",
)
TelegramBot = skill_mod.TelegramBot
TelegramMessage = skill_mod.TelegramMessage


# ---------------------------------------------------------------------------
# TestTelegramMessageAudioField
# ---------------------------------------------------------------------------

class TestTelegramMessageAudioField:
    """TelegramMessage dataclass has audio-related fields."""

    def test_audio_fields_default_none(self):
        msg = TelegramMessage(chat_id="123")
        assert msg.audio is None
        assert msg.audio_title is None
        assert msg.audio_performer is None
        assert msg.audio_duration is None

    def test_audio_fields_set(self):
        msg = TelegramMessage(
            chat_id="123",
            audio=b"fake-mp3",
            audio_title="My Song",
            audio_performer="Artist",
            audio_duration=180,
        )
        assert msg.audio == b"fake-mp3"
        assert msg.audio_title == "My Song"
        assert msg.audio_performer == "Artist"
        assert msg.audio_duration == 180


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(tmp_path: pathlib.Path) -> TelegramBot:
    """Create a TelegramBot with a fake token pointing at a temp DB."""
    return TelegramBot(tmp_path / "test.db", bot_token="fake-token")


def _ok_response(*_args, **_kwargs):
    return {"ok": True, "result": {"message_id": 1}}


def _fail_response(*_args, **_kwargs):
    return {"ok": False, "error": "test-error"}


# ---------------------------------------------------------------------------
# TestSendAudio
# ---------------------------------------------------------------------------

class TestSendAudio:
    """TelegramBot.send_audio() sends audio via the Telegram API."""

    def test_success(self, tmp_path):
        bot = _make_bot(tmp_path)
        with patch.object(bot, "_api_request", side_effect=_ok_response) as mock_req:
            result = bot.send_audio("42", b"audio-bytes")

        assert result is True
        assert bot.stats["audios_sent"] == 1
        assert bot.stats["failures"] == 0

        mock_req.assert_called_once()
        method, data, files = mock_req.call_args[0]
        assert method == "sendAudio"
        assert data["chat_id"] == "42"
        assert "audio" in files
        assert files["audio"][1] == b"audio-bytes"

    def test_failure_increments_failures(self, tmp_path):
        bot = _make_bot(tmp_path)
        with patch.object(bot, "_api_request", side_effect=_fail_response):
            result = bot.send_audio("42", b"audio-bytes")

        assert result is False
        assert bot.stats["audios_sent"] == 0
        assert bot.stats["failures"] == 1

    def test_with_metadata(self, tmp_path):
        bot = _make_bot(tmp_path)
        with patch.object(bot, "_api_request", side_effect=_ok_response) as mock_req:
            result = bot.send_audio(
                "42", b"mp3-data",
                filename="song.mp3",
                caption="Listen!",
                title="Cool Song",
                performer="DJ Bot",
                duration=240,
            )

        assert result is True
        _, data, files = mock_req.call_args[0]
        assert data["caption"] == "Listen!"
        assert data["title"] == "Cool Song"
        assert data["performer"] == "DJ Bot"
        assert data["duration"] == 240
        assert files["audio"][0] == "song.mp3"

    def test_default_filename(self, tmp_path):
        bot = _make_bot(tmp_path)
        with patch.object(bot, "_api_request", side_effect=_ok_response) as mock_req:
            bot.send_audio("42", b"data")

        _, _, files = mock_req.call_args[0]
        assert files["audio"][0] == "audio.mp3"

    def test_optional_fields_omitted_when_none(self, tmp_path):
        bot = _make_bot(tmp_path)
        with patch.object(bot, "_api_request", side_effect=_ok_response) as mock_req:
            bot.send_audio("42", b"data")

        _, data, _ = mock_req.call_args[0]
        assert "caption" not in data
        assert "title" not in data
        assert "performer" not in data
        assert "duration" not in data


# ---------------------------------------------------------------------------
# TestProcessQueueAudio
# ---------------------------------------------------------------------------

class TestProcessQueueAudio:
    """process_queue dispatches audio messages correctly."""

    def test_audio_message_dispatched(self, tmp_path):
        bot = _make_bot(tmp_path)
        msg = TelegramMessage(
            chat_id="42",
            audio=b"queue-audio",
            audio_title="Queued Song",
            audio_performer="Queue Artist",
            audio_duration=120,
            filename="queued.mp3",
            caption="From queue",
        )
        bot.queue_message(msg)

        with patch.object(bot, "send_audio", return_value=True) as mock_send:
            processed = bot.process_queue()

        assert processed == 1
        mock_send.assert_called_once_with(
            "42", b"queue-audio",
            filename="queued.mp3",
            caption="From queue",
            title="Queued Song",
            performer="Queue Artist",
            duration=120,
        )

    def test_audio_message_retried_on_failure(self, tmp_path):
        bot = _make_bot(tmp_path)
        msg = TelegramMessage(chat_id="42", audio=b"retry-audio")
        bot.queue_message(msg)

        with patch.object(bot, "send_audio", return_value=False):
            processed = bot.process_queue()

        assert processed == 0
        assert bot.stats["queue_size"] == 1  # re-queued


# ---------------------------------------------------------------------------
# TestStats
# ---------------------------------------------------------------------------

class TestStats:
    """Stats dict includes audios_sent counter."""

    def test_audios_sent_in_stats(self, tmp_path):
        bot = _make_bot(tmp_path)
        stats = bot.get_stats()
        assert "audios_sent" in stats
        assert stats["audios_sent"] == 0

    def test_audios_sent_incremented(self, tmp_path):
        bot = _make_bot(tmp_path)
        with patch.object(bot, "_api_request", side_effect=_ok_response):
            bot.send_audio("42", b"data")
        assert bot.get_stats()["audios_sent"] == 1
