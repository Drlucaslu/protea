"""Tests for ring1.prompts."""

from ring1.prompts import build_evolution_prompt, extract_python_code, extract_reflection


class TestBuildEvolutionPrompt:
    def test_returns_tuple(self):
        system, user = build_evolution_prompt(
            current_source="print('hello')",
            fitness_history=[],
            best_performers=[],
            params={"mutation_rate": 0.1, "max_runtime_sec": 60},
            generation=0,
            survived=True,
        )
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_has_constraints(self):
        system, _ = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
        )
        assert "heartbeat" in system.lower()
        assert "main()" in system
        assert "PROTEA_HEARTBEAT" in system

    def test_user_prompt_contains_source(self):
        _, user = build_evolution_prompt(
            current_source="print('unique_marker_42')",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=5,
            survived=False,
        )
        assert "unique_marker_42" in user
        assert "Generation 5" in user
        assert "DIED" in user

    def test_survived_instructions(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=1,
            survived=True,
        )
        assert "SURVIVED" in user
        assert "creative" in user.lower() or "interesting" in user.lower()

    def test_died_instructions(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=1,
            survived=False,
        )
        assert "DIED" in user
        assert "robust" in user.lower() or "fix" in user.lower()

    def test_includes_fitness_history(self):
        history = [
            {"generation": 0, "score": 0.5, "runtime_sec": 30.0, "survived": False},
            {"generation": 1, "score": 1.0, "runtime_sec": 60.0, "survived": True},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=history,
            best_performers=[],
            params={},
            generation=2,
            survived=True,
        )
        assert "Gen 0" in user
        assert "Gen 1" in user
        assert "SURVIVED" in user

    def test_includes_best_performers(self):
        best = [
            {"generation": 3, "score": 0.95, "commit_hash": "abc123def456"},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=best,
            params={},
            generation=4,
            survived=True,
        )
        assert "abc123de" in user
        assert "0.95" in user

    def test_includes_params(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={"mutation_rate": 0.42, "max_runtime_sec": 120},
            generation=0,
            survived=True,
        )
        assert "0.42" in user
        assert "120" in user

    def test_directive_included(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            directive="变成贪吃蛇",
        )
        assert "User Directive" in user
        assert "变成贪吃蛇" in user

    def test_no_directive_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            directive="",
        )
        assert "User Directive" not in user

    def test_directive_default_empty(self):
        """Calling without directive arg should not include directive section."""
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
        )
        assert "User Directive" not in user

    def test_memories_included(self):
        memories = [
            {"generation": 5, "entry_type": "reflection", "content": "CA patterns survive best"},
            {"generation": 3, "entry_type": "observation", "content": "Gen 3 survived 120s"},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=6,
            survived=True,
            memories=memories,
        )
        assert "Learned Patterns" in user
        assert "CA patterns survive best" in user
        assert "Gen 5, reflection" in user
        assert "Gen 3, observation" in user

    def test_no_memories_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            memories=None,
        )
        assert "Learned Patterns" not in user

    def test_empty_memories_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            memories=[],
        )
        assert "Learned Patterns" not in user

    def test_memories_default_none(self):
        """Calling without memories arg should not include section."""
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
        )
        assert "Learned Patterns" not in user

    def test_system_prompt_has_reflection_format(self):
        system, _ = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
        )
        assert "## Reflection" in system
        assert "reflection" in system.lower()


class TestExtractPythonCode:
    def test_extracts_code_block(self):
        response = 'Some text\n```python\nprint("hello")\n```\nMore text'
        code = extract_python_code(response)
        assert code == 'print("hello")'

    def test_multiline_code(self):
        response = '```python\ndef main():\n    pass\n```'
        code = extract_python_code(response)
        assert "def main():" in code
        assert "pass" in code

    def test_no_code_block(self):
        response = "Just some text without code"
        code = extract_python_code(response)
        assert code is None

    def test_empty_code_block(self):
        response = "```python\n\n```"
        code = extract_python_code(response)
        assert code is None

    def test_non_python_block_ignored(self):
        response = "```javascript\nconsole.log('hi')\n```"
        code = extract_python_code(response)
        assert code is None

    def test_first_block_wins(self):
        response = (
            '```python\nfirst()\n```\n'
            '```python\nsecond()\n```'
        )
        code = extract_python_code(response)
        assert code == "first()"

    def test_preserves_indentation(self):
        response = '```python\ndef f():\n    for i in range(10):\n        print(i)\n```'
        code = extract_python_code(response)
        assert "    for i" in code
        assert "        print" in code


class TestExtractReflection:
    def test_extracts_reflection(self):
        response = (
            "## Reflection\n"
            "Single-thread heartbeat is most stable.\n\n"
            "```python\ndef main():\n    pass\n```"
        )
        reflection = extract_reflection(response)
        assert reflection == "Single-thread heartbeat is most stable."

    def test_multiline_reflection(self):
        response = (
            "## Reflection\n"
            "Line one.\n"
            "Line two.\n\n"
            "```python\ncode\n```"
        )
        reflection = extract_reflection(response)
        assert "Line one." in reflection
        assert "Line two." in reflection

    def test_no_reflection_section(self):
        response = "```python\ndef main():\n    pass\n```"
        reflection = extract_reflection(response)
        assert reflection is None

    def test_empty_reflection(self):
        response = "## Reflection\n\n```python\ncode\n```"
        reflection = extract_reflection(response)
        assert reflection is None

    def test_reflection_with_code_only(self):
        response = "Just some text without reflection"
        reflection = extract_reflection(response)
        assert reflection is None
