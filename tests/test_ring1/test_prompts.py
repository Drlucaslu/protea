"""Tests for ring1.prompts."""

from ring1.prompts import (
    build_memory_curation_prompt,
    MEMORY_CURATION_SYSTEM_PROMPT,
)


class TestMemoryCurationPrompt:
    """Verify build_memory_curation_prompt."""

    def test_returns_tuple(self):
        candidates = [
            {"id": 1, "entry_type": "task", "content": "test", "importance": 0.7},
        ]
        system, user = build_memory_curation_prompt(candidates)
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_has_criteria(self):
        assert "keep" in MEMORY_CURATION_SYSTEM_PROMPT
        assert "discard" in MEMORY_CURATION_SYSTEM_PROMPT
        assert "summarize" in MEMORY_CURATION_SYSTEM_PROMPT

    def test_user_contains_entries(self):
        candidates = [
            {"id": 1, "entry_type": "task", "content": "debug python", "importance": 0.7},
            {"id": 2, "entry_type": "observation", "content": "survived 120s", "importance": 0.5},
        ]
        _, user = build_memory_curation_prompt(candidates)
        assert "ID 1" in user
        assert "ID 2" in user
        assert "debug python" in user
        assert "Total: 2" in user

    def test_truncates_long_content(self):
        candidates = [
            {"id": 1, "entry_type": "task", "content": "x" * 500, "importance": 0.5},
        ]
        _, user = build_memory_curation_prompt(candidates)
        assert "..." in user
        assert len(user) < 500
