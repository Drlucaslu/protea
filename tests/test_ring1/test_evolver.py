"""Tests for ring1.evolver."""

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from ring1.evolver import Evolver, EvolutionResult, validate_ring2_code


# --- A valid Ring 2 source for testing ---
VALID_SOURCE = '''\
import os, pathlib, time

def write_heartbeat(path, pid):
    path.write_text(f"{pid}\\n{time.time()}\\n")

def main():
    hb = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    while True:
        write_heartbeat(hb, pid)
        time.sleep(2)

if __name__ == "__main__":
    main()
'''

INVALID_SOURCE_NO_HB = '''\
def main():
    print("no heartbeat")

if __name__ == "__main__":
    main()
'''

INVALID_SOURCE_NO_MAIN = '''\
import os
hb = os.environ.get("PROTEA_HEARTBEAT", ".heartbeat")
print(hb)
'''

SYNTAX_ERROR_SOURCE = "def main(\n"


class TestValidateRing2Code:
    def test_valid(self):
        ok, reason = validate_ring2_code(VALID_SOURCE)
        assert ok is True
        assert reason == "OK"

    def test_syntax_error(self):
        ok, reason = validate_ring2_code(SYNTAX_ERROR_SOURCE)
        assert ok is False
        assert "Syntax error" in reason

    def test_missing_heartbeat(self):
        ok, reason = validate_ring2_code(INVALID_SOURCE_NO_HB)
        assert ok is False
        assert "PROTEA_HEARTBEAT" in reason

    def test_missing_main(self):
        ok, reason = validate_ring2_code(INVALID_SOURCE_NO_MAIN)
        assert ok is False
        assert "main()" in reason


def _make_config(**overrides):
    cfg = MagicMock()
    cfg.claude_api_key = overrides.get("api_key", "sk-test")
    cfg.claude_model = overrides.get("model", "test-model")
    cfg.claude_max_tokens = overrides.get("max_tokens", 4096)
    cfg.max_prompt_history = overrides.get("max_prompt_history", 10)
    return cfg


def _make_fitness(history=None, best=None):
    ft = MagicMock()
    ft.get_history.return_value = history or []
    ft.get_best.return_value = best or []
    return ft


class TestEvolver:
    def test_evolve_success(self, tmp_path):
        """Full happy path: LLM returns valid code."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"Here's the mutated code:\n```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.return_value = llm_response
            evolver = Evolver(config, fitness)
            result = evolver.evolve(ring2, generation=1, params={}, survived=True)

        assert result.success is True
        assert result.reason == "OK"
        assert len(result.new_source) > 0

    def test_evolve_missing_main_py(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        # No main.py.
        config = _make_config()
        fitness = _make_fitness()
        evolver = Evolver(config, fitness)
        result = evolver.evolve(ring2, generation=0, params={}, survived=False)
        assert result.success is False
        assert "not found" in result.reason

    def test_evolve_llm_error(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        config = _make_config()
        fitness = _make_fitness()

        from ring1.llm_client import LLMError
        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.side_effect = LLMError("timeout")
            evolver = Evolver(config, fitness)
            result = evolver.evolve(ring2, generation=0, params={}, survived=True)

        assert result.success is False
        assert "LLM error" in result.reason

    def test_evolve_no_code_block(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        config = _make_config()
        fitness = _make_fitness()

        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.return_value = "Sorry, no code today."
            evolver = Evolver(config, fitness)
            result = evolver.evolve(ring2, generation=0, params={}, survived=True)

        assert result.success is False
        assert "No code block" in result.reason

    def test_evolve_invalid_code(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        bad_code = "```python\ndef main():\n    print('no heartbeat')\n```"
        config = _make_config()
        fitness = _make_fitness()

        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.return_value = bad_code
            evolver = Evolver(config, fitness)
            result = evolver.evolve(ring2, generation=0, params={}, survived=True)

        assert result.success is False
        assert "Validation" in result.reason

    def test_evolve_writes_to_file(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text("old content")

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.return_value = llm_response
            evolver = Evolver(config, fitness)
            result = evolver.evolve(ring2, generation=0, params={}, survived=True)

        assert result.success is True
        written = (ring2 / "main.py").read_text()
        assert "PROTEA_HEARTBEAT" in written
        assert "old content" not in written

    def test_evolve_queries_fitness(self, tmp_path):
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config(max_prompt_history=7)
        fitness = _make_fitness()

        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.return_value = llm_response
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=3, params={}, survived=True)

        fitness.get_history.assert_called_once_with(limit=7)
        fitness.get_best.assert_called_once_with(n=5)

    def test_evolution_result_namedtuple(self):
        r = EvolutionResult(success=True, reason="OK", new_source="code")
        assert r.success is True
        assert r.reason == "OK"
        assert r.new_source == "code"

    def test_directive_passed_to_prompt_builder(self, tmp_path):
        """Directive should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        with patch("ring1.evolver.ClaudeClient") as MockClient, \
             patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            MockClient.return_value.send_message.return_value = llm_response
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True,
                           directive="make a game")

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("directive") == "make a game" or \
               (len(call_kwargs[0]) >= 8 and call_kwargs[0][7] == "make a game")

    def test_directive_default_empty(self, tmp_path):
        """Calling evolve without directive should use empty string."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        with patch("ring1.evolver.ClaudeClient") as MockClient, \
             patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            MockClient.return_value.send_message.return_value = llm_response
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("directive") == "" or \
               (len(call_kwargs[0]) >= 8 and call_kwargs[0][7] == "")

    def test_memories_passed_to_prompt_builder(self, tmp_path):
        """Memories should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()
        memories = [{"generation": 1, "entry_type": "observation", "content": "test"}]

        with patch("ring1.evolver.ClaudeClient") as MockClient, \
             patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            MockClient.return_value.send_message.return_value = llm_response
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True,
                           memories=memories)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("memories") == memories

    def test_reflection_extracted_and_stored(self, tmp_path):
        """Reflection should be extracted from LLM response and stored."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = (
            "## Reflection\n"
            "Single-thread heartbeat is most stable.\n\n"
            f"```python\n{VALID_SOURCE}```"
        )
        config = _make_config()
        fitness = _make_fitness()
        memory_store = MagicMock()

        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.return_value = llm_response
            evolver = Evolver(config, fitness, memory_store=memory_store)
            result = evolver.evolve(ring2, generation=5, params={}, survived=True)

        assert result.success is True
        memory_store.add.assert_called_once_with(
            5, "reflection", "Single-thread heartbeat is most stable."
        )

    def test_no_reflection_no_store(self, tmp_path):
        """No memory.add call if LLM response has no reflection."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()
        memory_store = MagicMock()

        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.return_value = llm_response
            evolver = Evolver(config, fitness, memory_store=memory_store)
            evolver.evolve(ring2, generation=1, params={}, survived=True)

        memory_store.add.assert_not_called()

    def test_no_memory_store_no_error(self, tmp_path):
        """Evolver should work fine without memory_store."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = (
            "## Reflection\nSome reflection.\n\n"
            f"```python\n{VALID_SOURCE}```"
        )
        config = _make_config()
        fitness = _make_fitness()

        with patch("ring1.evolver.ClaudeClient") as MockClient:
            MockClient.return_value.send_message.return_value = llm_response
            evolver = Evolver(config, fitness)  # no memory_store
            result = evolver.evolve(ring2, generation=1, params={}, survived=True)

        assert result.success is True
