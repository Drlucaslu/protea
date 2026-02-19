"""Tests for ring1.directive_generator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ring1.directive_generator import DirectiveGenerator, MIN_TASKS, MAX_DIRECTIVE_LEN
from ring1.llm_base import LLMError


def _make_tasks(n: int) -> list[dict]:
    """Create n fake task history entries."""
    return [
        {"content": f"Task {i}: do something useful", "generation": i}
        for i in range(n)
    ]


class TestDirectiveGenerator:

    def test_valid_directive(self):
        """LLM returns a valid directive → parsed successfully."""
        client = MagicMock()
        client.send_message.return_value = "Add natural language date parsing for calendar tasks"
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(5), "User interested in productivity")
        assert result == "Add natural language date parsing for calendar tasks"
        client.send_message.assert_called_once()

    def test_too_few_tasks_returns_none(self):
        """< MIN_TASKS → returns None without calling LLM."""
        client = MagicMock()
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(MIN_TASKS - 1))
        assert result is None
        client.send_message.assert_not_called()

    def test_exactly_min_tasks(self):
        """Exactly MIN_TASKS → proceeds normally."""
        client = MagicMock()
        client.send_message.return_value = "Build file search skill"
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(MIN_TASKS))
        assert result == "Build file search skill"

    def test_llm_exception_returns_none(self):
        """LLM raises → returns None."""
        client = MagicMock()
        client.send_message.side_effect = LLMError("API timeout")
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(10))
        assert result is None

    def test_llm_returns_empty_string(self):
        """LLM returns empty → None."""
        client = MagicMock()
        client.send_message.return_value = ""
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(5))
        assert result is None

    def test_directive_truncated_to_max_length(self):
        """Directive longer than MAX_DIRECTIVE_LEN → truncated."""
        long_directive = "A" * 150
        client = MagicMock()
        client.send_message.return_value = long_directive
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(5))
        assert result is not None
        assert len(result) <= MAX_DIRECTIVE_LEN

    def test_markdown_headers_skipped(self):
        """Response with markdown headers → skips them, takes first content line."""
        client = MagicMock()
        client.send_message.return_value = "## Directive\nAdd PDF text extraction skill"
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(5))
        assert result == "Add PDF text extraction skill"

    def test_quoted_directive_stripped(self):
        """Response wrapped in quotes → quotes stripped."""
        client = MagicMock()
        client.send_message.return_value = '"Build a weather query skill"'
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(5))
        assert result == "Build a weather query skill"

    def test_no_user_profile(self):
        """Works without user profile summary."""
        client = MagicMock()
        client.send_message.return_value = "Improve Telegram command parsing"
        gen = DirectiveGenerator(client)

        result = gen.generate(_make_tasks(5), user_profile_summary="")
        assert result == "Improve Telegram command parsing"
        # Verify profile section not in prompt
        call_args = client.send_message.call_args
        user_msg = call_args[0][1]
        assert "User Profile" not in user_msg

    def test_task_content_truncated_in_prompt(self):
        """Long task content is truncated to 100 chars in prompt."""
        tasks = [{"content": "X" * 200, "generation": 0}] * 5
        client = MagicMock()
        client.send_message.return_value = "Some directive"
        gen = DirectiveGenerator(client)

        gen.generate(tasks)
        call_args = client.send_message.call_args
        user_msg = call_args[0][1]
        # Each task line should have truncated content with "..."
        assert "..." in user_msg
