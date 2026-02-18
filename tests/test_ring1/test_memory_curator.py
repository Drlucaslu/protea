"""Tests for ring1.memory_curator — MemoryCurator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from ring1.memory_curator import MemoryCurator


def _make_candidates():
    return [
        {"id": 1, "entry_type": "task", "content": "Help me debug python", "importance": 0.7},
        {"id": 2, "entry_type": "observation", "content": "Gen 10 survived 120s", "importance": 0.5},
        {"id": 3, "entry_type": "reflection", "content": "CA patterns work best", "importance": 0.6},
    ]


class TestCurate:
    def test_returns_decisions(self):
        mock_client = MagicMock()
        mock_client.send_message.return_value = json.dumps([
            {"id": 1, "action": "keep"},
            {"id": 2, "action": "discard"},
            {"id": 3, "action": "summarize", "summary": "CA patterns effective"},
        ])

        curator = MemoryCurator(mock_client)
        decisions = curator.curate(_make_candidates())

        assert len(decisions) == 3
        assert decisions[0] == {"id": 1, "action": "keep"}
        assert decisions[1] == {"id": 2, "action": "discard"}
        assert decisions[2]["action"] == "summarize"
        assert decisions[2]["summary"] == "CA patterns effective"

    def test_llm_failure_returns_empty(self):
        from ring1.llm_base import LLMError

        mock_client = MagicMock()
        mock_client.send_message.side_effect = LLMError("API down")

        curator = MemoryCurator(mock_client)
        decisions = curator.curate(_make_candidates())
        assert decisions == []

    def test_empty_candidates(self):
        mock_client = MagicMock()
        curator = MemoryCurator(mock_client)
        assert curator.curate([]) == []
        mock_client.send_message.assert_not_called()

    def test_invalid_json_returns_empty(self):
        mock_client = MagicMock()
        mock_client.send_message.return_value = "not valid json at all"

        curator = MemoryCurator(mock_client)
        decisions = curator.curate(_make_candidates())
        assert decisions == []

    def test_filters_invalid_ids(self):
        mock_client = MagicMock()
        mock_client.send_message.return_value = json.dumps([
            {"id": 1, "action": "keep"},
            {"id": 999, "action": "discard"},  # Invalid ID
        ])

        curator = MemoryCurator(mock_client)
        decisions = curator.curate(_make_candidates())
        assert len(decisions) == 1
        assert decisions[0]["id"] == 1

    def test_filters_invalid_actions(self):
        mock_client = MagicMock()
        mock_client.send_message.return_value = json.dumps([
            {"id": 1, "action": "keep"},
            {"id": 2, "action": "delete"},  # Invalid action
        ])

        curator = MemoryCurator(mock_client)
        decisions = curator.curate(_make_candidates())
        assert len(decisions) == 1

    def test_handles_markdown_fences(self):
        mock_client = MagicMock()
        mock_client.send_message.return_value = (
            '```json\n[{"id": 1, "action": "keep"}]\n```'
        )

        curator = MemoryCurator(mock_client)
        decisions = curator.curate(_make_candidates())
        assert len(decisions) == 1

    def test_non_list_response_returns_empty(self):
        mock_client = MagicMock()
        mock_client.send_message.return_value = '{"id": 1, "action": "keep"}'

        curator = MemoryCurator(mock_client)
        decisions = curator.curate(_make_candidates())
        assert decisions == []
