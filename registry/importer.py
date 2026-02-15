"""SKILL.md importer — parses OpenClaw SKILL.md format and imports into Protea.

Converts SKILL.md frontmatter + Markdown body into Protea skills.
Pure stdlib — no external dependencies (uses re + json for frontmatter parsing).
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import urllib.error
import urllib.request
from typing import NamedTuple

log = logging.getLogger("protea.importer")


class ImportResult(NamedTuple):
    imported: int
    skipped: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Frontmatter parsing (pure stdlib)
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_KV_RE = re.compile(r'^(\w[\w-]*):\s*(.+)$', re.MULTILINE)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML-like frontmatter and return (metadata, body).

    Only supports simple ``key: value`` and ``key: "value"`` format.
    The ``metadata`` field is parsed as JSON (OpenClaw stores it as inline JSON).
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    fm_text = m.group(1)
    body = text[m.end():]

    data: dict = {}
    for kv in _KV_RE.finditer(fm_text):
        key = kv.group(1)
        val = kv.group(2).strip()
        # Strip surrounding quotes.
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        # Try parsing as JSON for complex values (e.g. metadata).
        if val.startswith("{") or val.startswith("["):
            try:
                val = json.loads(val)
            except json.JSONDecodeError:
                pass
        data[key] = val

    return data, body


# ---------------------------------------------------------------------------
# SkillImporter
# ---------------------------------------------------------------------------


class SkillImporter:
    """Import OpenClaw SKILL.md files into Protea SkillStore and/or Registry."""

    def __init__(self, skill_store=None, registry_client=None) -> None:
        self._skill_store = skill_store
        self._registry_client = registry_client

    def parse_skill_md(self, text: str) -> dict | None:
        """Parse a single SKILL.md text and return a skill dict.

        Returns None if the text cannot be parsed into a valid skill.
        """
        if not text or not text.strip():
            return None

        fm, body = _parse_frontmatter(text)
        name = fm.get("name", "")
        if not name:
            # Try to extract name from first heading.
            heading = re.match(r"^#\s+(.+)", body.strip())
            if heading:
                name = heading.group(1).strip().lower().replace(" ", "-")
            else:
                return None

        description = fm.get("description", "")
        if isinstance(description, str):
            description = description.strip()

        # Extract tags and requirements from metadata.
        metadata = fm.get("metadata", {})
        tags: list[str] = []
        parameters: dict = {}
        if isinstance(metadata, dict):
            openclaw = metadata.get("openclaw", {})
            if isinstance(openclaw, dict):
                tags = openclaw.get("tags", [])
                requires = openclaw.get("requires", {})
                if requires:
                    parameters = {"requires": requires}

        prompt_template = body.strip()

        return {
            "name": name,
            "description": description,
            "prompt_template": prompt_template,
            "tags": tags if isinstance(tags, list) else [],
            "parameters": parameters,
            "source_code": text,
            "source": "clawhub",
        }

    def import_file(self, path: pathlib.Path) -> dict | None:
        """Read and import a single SKILL.md file.  Returns the skill dict or None."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, IOError) as exc:
            log.warning("Cannot read %s: %s", path, exc)
            return None

        skill = self.parse_skill_md(text)
        if skill is None:
            log.debug("Skipping %s — could not parse", path)
            return None

        self._persist(skill)
        return skill

    def import_directory(self, dir_path: pathlib.Path) -> ImportResult:
        """Recursively scan a directory for SKILL.md files and import them."""
        imported = 0
        skipped = 0
        errors: list[str] = []

        if not dir_path.is_dir():
            return ImportResult(0, 0, [f"Not a directory: {dir_path}"])

        for md_path in sorted(dir_path.rglob("SKILL.md")):
            try:
                result = self.import_file(md_path)
                if result:
                    imported += 1
                    log.info("Imported: %s (%s)", result["name"], md_path)
                else:
                    skipped += 1
            except Exception as exc:
                errors.append(f"{md_path}: {exc}")
                log.warning("Error importing %s: %s", md_path, exc)

        return ImportResult(imported, skipped, errors)

    def import_from_url(self, url: str) -> dict | None:
        """Fetch a SKILL.md from a URL and import it."""
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            log.warning("Cannot fetch %s: %s", url, exc)
            return None

        skill = self.parse_skill_md(text)
        if skill is None:
            log.debug("Skipping URL %s — could not parse", url)
            return None

        self._persist(skill)
        return skill

    def import_from_urls(self, urls: list[str]) -> ImportResult:
        """Batch import from multiple URLs."""
        imported = 0
        skipped = 0
        errors: list[str] = []

        for url in urls:
            try:
                result = self.import_from_url(url)
                if result:
                    imported += 1
                    log.info("Imported from URL: %s", result["name"])
                else:
                    skipped += 1
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                log.warning("Error importing %s: %s", url, exc)

        return ImportResult(imported, skipped, errors)

    def _persist(self, skill: dict) -> None:
        """Save skill to local store and/or remote registry."""
        name = skill["name"]

        if self._skill_store:
            try:
                existing = self._skill_store.get_by_name(name)
                if existing:
                    log.debug("Skill %r already exists locally — skipping", name)
                else:
                    self._skill_store.add(
                        name=name,
                        description=skill.get("description", ""),
                        prompt_template=skill.get("prompt_template", ""),
                        parameters=skill.get("parameters"),
                        tags=skill.get("tags"),
                        source=skill.get("source", "clawhub"),
                        source_code=skill.get("source_code", ""),
                    )
            except Exception as exc:
                log.warning("Local store error for %r: %s", name, exc)

        if self._registry_client:
            try:
                self._registry_client.publish(
                    name=name,
                    description=skill.get("description", ""),
                    prompt_template=skill.get("prompt_template", ""),
                    parameters=skill.get("parameters"),
                    tags=skill.get("tags"),
                    source_code=skill.get("source_code", ""),
                )
            except Exception as exc:
                log.debug("Registry publish error for %r: %s", name, exc)
