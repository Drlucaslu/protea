"""Tests for registry.importer — SkillImporter."""

from __future__ import annotations

import json
import pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time

import pytest

from registry.importer import SkillImporter, _parse_frontmatter


# ---------------------------------------------------------------------------
# Sample SKILL.md content
# ---------------------------------------------------------------------------

_VALID_SKILL_MD = """\
---
name: llm-council
description: "Run a council of LLMs for multi-perspective analysis"
metadata: {"openclaw":{"requires":{"env":["API_KEY"],"bins":["curl"]},"tags":["ai","council"]}}
license: MIT
---
# LLM Council

You are an AI council facilitator. When given a topic, query multiple
perspectives and synthesize a unified response.

## Steps
1. Analyze the topic
2. Generate perspectives
3. Synthesize results
"""

_MINIMAL_SKILL_MD = """\
---
name: simple-tool
description: A simple tool
---
Do the simple thing.
"""

_NO_FRONTMATTER_MD = """\
# My Unnamed Skill

This is a skill without frontmatter.
"""

_EMPTY_TEXT = ""

_NO_NAME_MD = """\
---
description: "Missing name field"
---
Some content.
"""


# ---------------------------------------------------------------------------
# Fake stores
# ---------------------------------------------------------------------------


class _FakeSkillStore:
    def __init__(self):
        self._skills: dict[str, dict] = {}

    def get_by_name(self, name: str) -> dict | None:
        return self._skills.get(name)

    def add(self, name, description="", prompt_template="", parameters=None, tags=None, source="user", source_code=""):
        self._skills[name] = {
            "name": name,
            "description": description,
            "prompt_template": prompt_template,
            "parameters": parameters or {},
            "tags": tags or [],
            "source": source,
            "source_code": source_code,
        }
        return len(self._skills)


class _FakeRegistryClient:
    def __init__(self):
        self.published: list[dict] = []

    def publish(self, name, description="", prompt_template="", parameters=None, tags=None, source_code=""):
        self.published.append({"name": name, "description": description})
        return {"name": name, "version": 1}


# ---------------------------------------------------------------------------
# Tests: _parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_standard_format(self):
        fm, body = _parse_frontmatter(_VALID_SKILL_MD)
        assert fm["name"] == "llm-council"
        assert fm["description"] == "Run a council of LLMs for multi-perspective analysis"
        assert fm["license"] == "MIT"
        assert "LLM Council" in body

    def test_metadata_json(self):
        fm, body = _parse_frontmatter(_VALID_SKILL_MD)
        metadata = fm["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["openclaw"]["tags"] == ["ai", "council"]
        assert metadata["openclaw"]["requires"]["env"] == ["API_KEY"]

    def test_no_frontmatter(self):
        fm, body = _parse_frontmatter(_NO_FRONTMATTER_MD)
        assert fm == {}
        assert "My Unnamed Skill" in body

    def test_minimal(self):
        fm, body = _parse_frontmatter(_MINIMAL_SKILL_MD)
        assert fm["name"] == "simple-tool"
        assert "Do the simple thing." in body


# ---------------------------------------------------------------------------
# Tests: parse_skill_md
# ---------------------------------------------------------------------------


class TestParseSkillMd:
    def test_full_parse(self):
        importer = SkillImporter()
        skill = importer.parse_skill_md(_VALID_SKILL_MD)
        assert skill is not None
        assert skill["name"] == "llm-council"
        assert skill["description"] == "Run a council of LLMs for multi-perspective analysis"
        assert "ai" in skill["tags"]
        assert "council" in skill["tags"]
        assert "requires" in skill["parameters"]
        assert "LLM Council" in skill["prompt_template"]
        assert skill["source"] == "clawhub"

    def test_minimal_parse(self):
        importer = SkillImporter()
        skill = importer.parse_skill_md(_MINIMAL_SKILL_MD)
        assert skill is not None
        assert skill["name"] == "simple-tool"
        assert skill["tags"] == []

    def test_no_frontmatter_extracts_heading(self):
        importer = SkillImporter()
        skill = importer.parse_skill_md(_NO_FRONTMATTER_MD)
        assert skill is not None
        assert skill["name"] == "my-unnamed-skill"

    def test_empty_returns_none(self):
        importer = SkillImporter()
        assert importer.parse_skill_md(_EMPTY_TEXT) is None

    def test_no_name_no_heading_returns_none(self):
        importer = SkillImporter()
        text = "---\ndescription: test\n---\nNo heading here, just plain text."
        assert importer.parse_skill_md(text) is None


# ---------------------------------------------------------------------------
# Tests: import_file
# ---------------------------------------------------------------------------


class TestImportFile:
    def test_valid_file(self, tmp_path):
        md = tmp_path / "SKILL.md"
        md.write_text(_VALID_SKILL_MD, encoding="utf-8")
        store = _FakeSkillStore()
        importer = SkillImporter(skill_store=store)
        result = importer.import_file(md)
        assert result is not None
        assert result["name"] == "llm-council"
        assert "llm-council" in store._skills

    def test_file_not_found(self, tmp_path):
        importer = SkillImporter()
        result = importer.import_file(tmp_path / "nonexistent.md")
        assert result is None

    def test_invalid_format_skipped(self, tmp_path):
        md = tmp_path / "SKILL.md"
        md.write_text("", encoding="utf-8")
        importer = SkillImporter()
        result = importer.import_file(md)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: import_directory
# ---------------------------------------------------------------------------


class TestImportDirectory:
    def test_recursive_scan(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "SKILL.md").write_text(_VALID_SKILL_MD, encoding="utf-8")
        (tmp_path / "b" / "SKILL.md").write_text(_MINIMAL_SKILL_MD, encoding="utf-8")

        store = _FakeSkillStore()
        importer = SkillImporter(skill_store=store)
        result = importer.import_directory(tmp_path)
        assert result.imported == 2
        assert result.skipped == 0
        assert len(result.errors) == 0

    def test_mixed_valid_invalid(self, tmp_path):
        (tmp_path / "good").mkdir()
        (tmp_path / "bad").mkdir()
        (tmp_path / "good" / "SKILL.md").write_text(_VALID_SKILL_MD, encoding="utf-8")
        (tmp_path / "bad" / "SKILL.md").write_text("", encoding="utf-8")

        store = _FakeSkillStore()
        importer = SkillImporter(skill_store=store)
        result = importer.import_directory(tmp_path)
        assert result.imported == 1
        assert result.skipped == 1

    def test_empty_directory(self, tmp_path):
        importer = SkillImporter()
        result = importer.import_directory(tmp_path)
        assert result.imported == 0

    def test_not_a_directory(self, tmp_path):
        importer = SkillImporter()
        result = importer.import_directory(tmp_path / "nonexistent")
        assert result.imported == 0
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# Tests: import_from_url (with local HTTP server mock)
# ---------------------------------------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    response_text = _VALID_SKILL_MD

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        data = self.response_text.encode("utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


class TestImportFromUrl:
    def test_url_import(self):
        srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        port = srv.server_address[1]
        try:
            store = _FakeSkillStore()
            importer = SkillImporter(skill_store=store)
            url = f"http://127.0.0.1:{port}/SKILL.md"
            result = importer.import_from_url(url)
            assert result is not None
            assert result["name"] == "llm-council"
            assert "llm-council" in store._skills
        finally:
            srv.shutdown()

    def test_url_unreachable(self):
        importer = SkillImporter()
        result = importer.import_from_url("http://127.0.0.1:1/SKILL.md")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: duplicate skip
# ---------------------------------------------------------------------------


class TestDuplicateSkip:
    def test_existing_skill_not_overwritten(self, tmp_path):
        md = tmp_path / "SKILL.md"
        md.write_text(_VALID_SKILL_MD, encoding="utf-8")
        store = _FakeSkillStore()
        store.add("llm-council", "Already exists", "old template")

        importer = SkillImporter(skill_store=store)
        result = importer.import_file(md)
        assert result is not None
        # The existing skill should NOT be overwritten.
        assert store._skills["llm-council"]["description"] == "Already exists"


# ---------------------------------------------------------------------------
# Tests: registry client integration
# ---------------------------------------------------------------------------


class TestRegistryClientIntegration:
    def test_publish_to_registry(self, tmp_path):
        md = tmp_path / "SKILL.md"
        md.write_text(_VALID_SKILL_MD, encoding="utf-8")
        registry = _FakeRegistryClient()
        importer = SkillImporter(registry_client=registry)
        result = importer.import_file(md)
        assert result is not None
        assert len(registry.published) == 1
        assert registry.published[0]["name"] == "llm-council"
