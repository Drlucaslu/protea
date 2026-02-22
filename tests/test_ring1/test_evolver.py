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

        config.get_llm_client.return_value.send_message.return_value = llm_response
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

        from ring1.llm_base import LLMError
        config.get_llm_client.return_value.send_message.side_effect = LLMError("timeout")
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

        config.get_llm_client.return_value.send_message.return_value = "Sorry, no code today."
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

        config.get_llm_client.return_value.send_message.return_value = bad_code
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

        config.get_llm_client.return_value.send_message.return_value = llm_response
        evolver = Evolver(config, fitness)
        # Use adapt intent so blast radius gate allows full rewrite.
        result = evolver.evolve(
            ring2, generation=0, params={}, survived=True,
            evolution_intent={"intent": "adapt", "signals": ["test"]},
        )

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

        config.get_llm_client.return_value.send_message.return_value = llm_response
        evolver = Evolver(config, fitness)
        evolver.evolve(ring2, generation=3, params={}, survived=True)

        fitness.get_history.assert_called_once_with(limit=5)  # capped at 5 to save tokens
        fitness.get_best.assert_called_once_with(n=3)

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

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
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

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
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

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
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

        config.get_llm_client.return_value.send_message.return_value = llm_response
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

        config.get_llm_client.return_value.send_message.return_value = llm_response
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

        config.get_llm_client.return_value.send_message.return_value = llm_response
        evolver = Evolver(config, fitness)  # no memory_store
        result = evolver.evolve(ring2, generation=1, params={}, survived=True)

        assert result.success is True

    def test_task_history_passed_to_prompt_builder(self, tmp_path):
        """task_history should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()
        task_history = [{"content": "test task"}]

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True,
                           task_history=task_history)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("task_history") == task_history

    def test_skills_passed_to_prompt_builder(self, tmp_path):
        """skills should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()
        skills = [{"name": "greet", "description": "Greeting"}]

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True,
                           skills=skills)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("skills") == skills

    def test_task_history_and_skills_default_none(self, tmp_path):
        """Calling evolve without task_history/skills should pass None."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("task_history") is None
        assert call_kwargs[1].get("skills") is None

    def test_crash_logs_passed_to_prompt_builder(self, tmp_path):
        """crash_logs should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()
        crash_logs = [{"generation": 1, "content": "Gen 1 died."}]

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=2, params={}, survived=False,
                           crash_logs=crash_logs)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("crash_logs") == crash_logs

    def test_crash_logs_default_none(self, tmp_path):
        """Calling evolve without crash_logs should pass None."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("crash_logs") is None

    def test_gene_pool_passed_to_prompt_builder(self, tmp_path):
        """gene_pool should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()
        gene_pool = [
            {"generation": 5, "score": 0.90, "gene_summary": "class Foo: pass"},
        ]

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True,
                           gene_pool=gene_pool)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("gene_pool") == gene_pool

    def test_gene_pool_default_none(self, tmp_path):
        """Calling evolve without gene_pool should pass None."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True)

        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("gene_pool") is None

    def test_capability_proposal_in_metadata(self, tmp_path):
        """If LLM response contains a capability block, it should appear in metadata."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        capability_json = '{"name": "web_scraper", "description": "Scrape pages", "dependencies": ["requests"], "source_code": "print(1)", "tags": ["web"]}'
        llm_response = (
            f"## Reflection\nAdding capability.\n\n"
            f"```python\n{VALID_SOURCE}```\n\n"
            f"```capability\n{capability_json}\n```"
        )
        config = _make_config()
        fitness = _make_fitness()

        config.get_llm_client.return_value.send_message.return_value = llm_response
        evolver = Evolver(config, fitness)
        result = evolver.evolve(ring2, generation=1, params={}, survived=True)

        assert result.success is True
        assert "capability_proposal" in result.metadata
        assert result.metadata["capability_proposal"]["name"] == "web_scraper"
        assert result.metadata["capability_proposal"]["dependencies"] == ["requests"]

    def test_no_capability_proposal_not_in_metadata(self, tmp_path):
        """Without capability block, metadata should not have capability_proposal."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        config.get_llm_client.return_value.send_message.return_value = llm_response
        evolver = Evolver(config, fitness)
        result = evolver.evolve(ring2, generation=1, params={}, survived=True)

        assert result.success is True
        assert "capability_proposal" not in result.metadata

    def test_permanent_capabilities_passed_to_prompt(self, tmp_path):
        """permanent_capabilities should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()
        caps = [{"name": "test_cap", "description": "d", "dependencies": [], "usage_count": 1}]

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True,
                           permanent_capabilities=caps)

        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("permanent_capabilities") == caps

    def test_allowed_packages_passed_to_prompt(self, tmp_path):
        """allowed_packages should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True,
                           allowed_packages=["requests", "pandas"])

        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("allowed_packages") == ["requests", "pandas"]

    def test_skill_hit_summary_passed_to_prompt(self, tmp_path):
        """skill_hit_summary should be forwarded to build_evolution_prompt."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()
        hit_summary = {"total": 10, "skill": 7, "ratio": 0.7, "top_skills": {"s1": 5}}

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True,
                           skill_hit_summary=hit_summary)

        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("skill_hit_summary") == hit_summary

    def test_skill_hit_summary_default_none(self, tmp_path):
        """Calling evolve without skill_hit_summary should pass None."""
        ring2 = tmp_path / "ring2"
        ring2.mkdir()
        (ring2 / "main.py").write_text(VALID_SOURCE)

        llm_response = f"```python\n{VALID_SOURCE}```"
        config = _make_config()
        fitness = _make_fitness()

        config.get_llm_client.return_value.send_message.return_value = llm_response
        with patch("ring1.evolver.build_evolution_prompt") as mock_prompt:
            mock_prompt.return_value = ("system", "user")
            evolver = Evolver(config, fitness)
            evolver.evolve(ring2, generation=1, params={}, survived=True)

        call_kwargs = mock_prompt.call_args
        assert call_kwargs[1].get("skill_hit_summary") is None
