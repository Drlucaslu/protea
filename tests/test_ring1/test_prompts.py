"""Tests for ring1.prompts."""

from ring1.prompts import (
    build_crystallize_prompt,
    build_evolution_prompt,
    build_memory_curation_prompt,
    extract_python_code,
    extract_reflection,
    parse_crystallize_response,
    MEMORY_CURATION_SYSTEM_PROMPT,
)


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
        assert "new" in user.lower() or "different" in user.lower()

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
        assert "Gen 3" in user
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
        assert "[Gen 5]" in user
        assert "[Gen 3]" in user

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

    def test_task_history_included(self):
        task_history = [
            {"content": "What is the weather?"},
            {"content": "Summarize this article"},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            task_history=task_history,
        )
        assert "Recent User Tasks" in user
        assert "What is the weather?" in user
        assert "Summarize this article" in user

    def test_no_task_history_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            task_history=None,
        )
        assert "Recent User Tasks" not in user

    def test_empty_task_history_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            task_history=[],
        )
        assert "Recent User Tasks" not in user

    def test_skills_included(self):
        skills = [
            {"name": "summarize", "description": "Summarize text", "usage_count": 0},
            {"name": "translate", "description": "Translate text", "usage_count": 3},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            skills=skills,
        )
        assert "Existing Skills" in user
        # Used skills shown with usage count.
        assert "translate" in user
        assert "used 3x" in user
        # Unused skills listed in "Never used" section.
        assert "summarize" in user
        assert "Never used" in user

    def test_no_skills_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            skills=None,
        )
        assert "Existing Skills" not in user

    def test_empty_skills_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            skills=[],
        )
        assert "Existing Skills" not in user

    def test_system_prompt_has_evolution_strategy(self):
        system, _ = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
        )
        assert "Evolution Strategy" in system
        assert "user" in system.lower()

    def test_crash_logs_included(self):
        crash_logs = [
            {"generation": 2, "content": "Gen 2 died after 5s.\nReason: exit code 1\n\n--- Last output ---\nKeyError: 'foo'"},
            {"generation": 1, "content": "Gen 1 died after 3s.\nReason: killed by signal SIGKILL"},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=3,
            survived=False,
            crash_logs=crash_logs,
        )
        assert "Recent Crashes" in user
        assert "Gen 2" in user
        assert "KeyError" in user
        assert "Gen 1" in user
        assert "SIGKILL" in user

    def test_no_crash_logs_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            crash_logs=None,
        )
        assert "Recent Crashes" not in user

    def test_empty_crash_logs_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            crash_logs=[],
        )
        assert "Recent Crashes" not in user

    def test_crash_logs_before_instructions(self):
        crash_logs = [
            {"generation": 1, "content": "Gen 1 died."},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=2,
            survived=False,
            crash_logs=crash_logs,
        )
        crash_pos = user.index("Recent Crashes")
        instructions_pos = user.index("## Instructions")
        assert crash_pos < instructions_pos

    def test_persistent_errors_included(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=5,
            survived=True,
            persistent_errors=["attributeerror: 'list' has no attribute 'get'"],
        )
        assert "PERSISTENT BUGS" in user
        assert "list" in user

    def test_plateau_warning(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=5,
            survived=True,
            is_plateaued=True,
        )
        assert "PLATEAUED" in user
        assert "fundamentally different" in user.lower()

    def test_gene_pool_included(self):
        gene_pool = [
            {"generation": 42, "score": 0.92, "gene_summary": "class StreamAnalyzer:\n    \"\"\"Real-time anomaly detection\"\"\""},
            {"generation": 38, "score": 0.89, "gene_summary": "class PackageScanner:\n    \"\"\"PyPI dependency graph\"\"\""},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=50,
            survived=True,
            gene_pool=gene_pool,
        )
        assert "Inherited Patterns" in user
        assert "StreamAnalyzer" in user
        assert "PackageScanner" in user
        assert "Gen 42" in user
        assert "0.92" in user

    def test_no_gene_pool_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            gene_pool=None,
        )
        assert "Inherited Patterns" not in user

    def test_empty_gene_pool_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=0,
            survived=True,
            gene_pool=[],
        )
        assert "Inherited Patterns" not in user

    def test_gene_pool_before_instructions(self):
        gene_pool = [
            {"generation": 1, "score": 0.80, "gene_summary": "class Foo:\n    pass"},
        ]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=2,
            survived=True,
            gene_pool=gene_pool,
        )
        gene_pos = user.index("Inherited Patterns")
        instructions_pos = user.index("## Instructions")
        assert gene_pos < instructions_pos


class TestEvolutionIntentInPrompt:
    """Verify ## Evolution Intent section when evolution_intent is provided."""

    def test_repair_intent_section(self):
        intent = {"intent": "repair", "signals": ["TypeError", "KeyError"]}
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=5,
            survived=False,
            evolution_intent=intent,
        )
        assert "## Evolution Intent: REPAIR" in user
        assert "FIX" in user
        assert "- TypeError" in user
        assert "- KeyError" in user
        # Legacy ## Instructions should NOT appear.
        assert "## Instructions" not in user

    def test_explore_intent_section(self):
        intent = {"intent": "explore", "signals": ["plateau"]}
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=10,
            survived=True,
            is_plateaued=True,
            evolution_intent=intent,
        )
        assert "## Evolution Intent: EXPLORE" in user
        assert "PLATEAUED" in user
        assert "## Instructions" not in user

    def test_adapt_intent_section(self):
        intent = {"intent": "adapt", "signals": ["directive: make a game"]}
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=3,
            survived=True,
            directive="make a game",
            evolution_intent=intent,
        )
        assert "## Evolution Intent: ADAPT" in user
        assert "directive" in user.lower()
        assert "## Instructions" not in user
        # User Directive section should still appear.
        assert "## User Directive" in user
        assert "make a game" in user

    def test_optimize_intent_section(self):
        intent = {"intent": "optimize", "signals": ["survived"]}
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=7,
            survived=True,
            evolution_intent=intent,
        )
        assert "## Evolution Intent: OPTIMIZE" in user
        assert "survived" in user.lower()
        assert "## Instructions" not in user

    def test_no_intent_falls_back_to_instructions(self):
        """Without evolution_intent, legacy ## Instructions section is used."""
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=1,
            survived=True,
        )
        assert "## Instructions" in user
        assert "## Evolution Intent" not in user

    def test_intent_before_user_directive(self):
        """Evolution Intent section should appear before User Directive."""
        intent = {"intent": "adapt", "signals": ["directive: test"]}
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=1,
            survived=True,
            directive="test",
            evolution_intent=intent,
        )
        intent_pos = user.index("## Evolution Intent")
        directive_pos = user.index("## User Directive")
        assert intent_pos < directive_pos


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


class TestBuildCrystallizePrompt:
    """build_crystallize_prompt() should produce valid (system, user) pair."""

    def test_returns_tuple(self):
        system, user = build_crystallize_prompt(
            source_code="print('hello')",
            output="hello",
            generation=5,
            existing_skills=[],
        )
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_contains_source_code(self):
        _, user = build_crystallize_prompt(
            source_code="unique_marker_99",
            output="",
            generation=1,
            existing_skills=[],
        )
        assert "unique_marker_99" in user

    def test_contains_output(self):
        _, user = build_crystallize_prompt(
            source_code="x=1",
            output="output_marker_42",
            generation=1,
            existing_skills=[],
        )
        assert "output_marker_42" in user

    def test_empty_output_no_section(self):
        _, user = build_crystallize_prompt(
            source_code="x=1",
            output="",
            generation=1,
            existing_skills=[],
        )
        assert "Program Output" not in user

    def test_contains_existing_skills(self):
        skills = [
            {"name": "web_dashboard", "description": "Web dashboard", "tags": ["web"]},
            {"name": "game", "description": "Snake game", "tags": ["game"]},
        ]
        _, user = build_crystallize_prompt(
            source_code="x=1",
            output="",
            generation=1,
            existing_skills=skills,
        )
        assert "web_dashboard" in user
        assert "Snake game" in user
        assert "Existing Skills" in user

    def test_capacity_display(self):
        skills = [{"name": f"s{i}", "description": "d", "tags": []} for i in range(5)]
        _, user = build_crystallize_prompt(
            source_code="x=1",
            output="",
            generation=1,
            existing_skills=skills,
            skill_cap=100,
        )
        assert "5/100" in user

    def test_full_capacity_warning(self):
        skills = [{"name": f"s{i}", "description": "d", "tags": []} for i in range(100)]
        _, user = build_crystallize_prompt(
            source_code="x=1",
            output="",
            generation=1,
            existing_skills=skills,
            skill_cap=100,
        )
        assert "FULL" in user


class TestParseCrystallizeResponse:
    """parse_crystallize_response() should parse JSON from LLM responses."""

    def test_parse_create(self):
        resp = '{"action": "create", "name": "web_dashboard", "description": "Web dash", "prompt_template": "tmpl", "tags": ["web"]}'
        data = parse_crystallize_response(resp)
        assert data["action"] == "create"
        assert data["name"] == "web_dashboard"

    def test_parse_update(self):
        resp = '{"action": "update", "existing_name": "web_dashboard", "description": "Updated", "prompt_template": "tmpl", "tags": []}'
        data = parse_crystallize_response(resp)
        assert data["action"] == "update"
        assert data["existing_name"] == "web_dashboard"

    def test_parse_skip(self):
        resp = '{"action": "skip", "reason": "already covered"}'
        data = parse_crystallize_response(resp)
        assert data["action"] == "skip"
        assert data["reason"] == "already covered"

    def test_invalid_json(self):
        assert parse_crystallize_response("not json at all") is None

    def test_invalid_action(self):
        resp = '{"action": "delete", "name": "x"}'
        assert parse_crystallize_response(resp) is None

    def test_markdown_wrapped(self):
        resp = '```json\n{"action": "create", "name": "test", "description": "d", "prompt_template": "t", "tags": []}\n```'
        data = parse_crystallize_response(resp)
        assert data is not None
        assert data["action"] == "create"

    def test_non_dict_returns_none(self):
        assert parse_crystallize_response("[1, 2, 3]") is None

    def test_missing_action_returns_none(self):
        assert parse_crystallize_response('{"name": "x"}') is None


class TestUserProfileInEvolution:
    """Verify user_profile_summary parameter in build_evolution_prompt."""

    def test_profile_included(self):
        summary = "User interests: coding (45%), data (25%)\nTop topics: python, pandas"
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=10,
            survived=True,
            user_profile_summary=summary,
        )
        assert "## User Profile" in user
        assert "coding (45%)" in user
        assert "Align evolution" in user

    def test_empty_profile_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=10,
            survived=True,
            user_profile_summary="",
        )
        assert "## User Profile" not in user

    def test_default_profile_no_section(self):
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=10,
            survived=True,
        )
        assert "## User Profile" not in user

    def test_profile_after_tasks_before_skills(self):
        task_history = [{"content": "test task"}]
        skills = [{"name": "s", "description": "d", "usage_count": 0}]
        _, user = build_evolution_prompt(
            current_source="x=1",
            fitness_history=[],
            best_performers=[],
            params={},
            generation=10,
            survived=True,
            task_history=task_history,
            skills=skills,
            user_profile_summary="User interests: coding (100%)",
        )
        task_pos = user.index("Recent User Tasks")
        profile_pos = user.index("User Profile")
        skills_pos = user.index("Existing Skills")
        assert task_pos < profile_pos < skills_pos


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
