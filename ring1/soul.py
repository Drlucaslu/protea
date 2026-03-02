"""Soul Profile — centralized identity, user info, and preferences for Protea.

Loads config/soul.md, caches it, and injects it into all system prompts.
Manages onboarding: asks the user questions via Telegram to fill in the
Owner section progressively.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from datetime import date

log = logging.getLogger("protea.soul")

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_soul_text: str = ""
_project_root: pathlib.Path | None = None
_soul_path: pathlib.Path | None = None

# ---------------------------------------------------------------------------
# Onboarding questions
# ---------------------------------------------------------------------------

_QUESTIONS: list[tuple[str, str]] = [
    ("owner_name", "你好！我还不太了解你。你叫什么名字？（回复姓名，或发「跳过」）"),
    ("owner_interests", "你平时对什么感兴趣？主要想用我来帮你做什么？"),
    ("owner_style", "你希望我回复的风格是怎样的？（简洁/详细/随意）"),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load(project_root: pathlib.Path) -> None:
    """Load config/soul.md and cache it. Call once at startup."""
    global _soul_text, _project_root, _soul_path
    _project_root = project_root
    _soul_path = project_root / "config" / "soul.md"
    if _soul_path.exists():
        _soul_text = _soul_path.read_text(encoding="utf-8")
        log.info("Soul profile loaded (%d chars)", len(_soul_text))
    else:
        _soul_text = ""
        log.warning("Soul profile not found: %s", _soul_path)


def reload() -> None:
    """Re-read soul.md from disk (call after onboarding writes)."""
    global _soul_text
    if _soul_path and _soul_path.exists():
        _soul_text = _soul_path.read_text(encoding="utf-8")


def get() -> str:
    """Return the cached soul profile text."""
    return _soul_text


def inject(base_prompt: str) -> str:
    """Prepend the soul profile to a base system prompt.

    If the soul profile is empty, returns the base prompt unchanged.
    """
    if not _soul_text:
        return base_prompt
    return _soul_text + "\n\n---\n\n" + base_prompt


def parse_sections() -> dict[str, str]:
    """Parse soul.md into {heading: content} by ## headings."""
    sections: dict[str, str] = {}
    current_heading = ""
    current_lines: list[str] = []

    for line in _soul_text.splitlines():
        m = re.match(r"^##\s+(.+)$", line)
        if m:
            if current_heading:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def is_section_empty(section_name: str) -> bool:
    """Check if a section has no meaningful content (only comments/whitespace)."""
    sections = parse_sections()
    content = sections.get(section_name, "")
    # Strip HTML comments and whitespace.
    cleaned = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL).strip()
    return not cleaned


# ---------------------------------------------------------------------------
# Onboarding state (persisted to data/soul_onboarding.json)
# ---------------------------------------------------------------------------

def _onboarding_path() -> pathlib.Path | None:
    if _project_root is None:
        return None
    return _project_root / "data" / "soul_onboarding.json"


def _load_onboarding_state() -> dict:
    path = _onboarding_path()
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_asked_date": "", "completed_fields": []}


def _save_onboarding_state(state: dict) -> None:
    path = _onboarding_path()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_next_question() -> tuple[str, str] | None:
    """Return (field_name, question_text) for the next unanswered question.

    Returns None if all questions are answered or onboarding is complete.
    """
    ob = _load_onboarding_state()
    completed = set(ob.get("completed_fields", []))
    for field, question in _QUESTIONS:
        if field not in completed:
            return (field, question)
    return None


def should_ask_today() -> bool:
    """Check if we can ask an onboarding question today (max 1/day)."""
    ob = _load_onboarding_state()
    return ob.get("last_asked_date", "") != date.today().isoformat()


def record_asked() -> None:
    """Record that we asked a question today."""
    ob = _load_onboarding_state()
    ob["last_asked_date"] = date.today().isoformat()
    _save_onboarding_state(ob)


def write_field(field: str, value: str, project_root: pathlib.Path | None = None) -> None:
    """Write a user's answer into the ## Owner section of soul.md.

    Also marks the field as completed in onboarding state.
    """
    root = project_root or _project_root
    if root is None:
        log.warning("write_field: project_root not set")
        return

    soul_path = root / "config" / "soul.md"
    if not soul_path.exists():
        log.warning("write_field: soul.md not found")
        return

    # Map field names to display labels.
    labels = {
        "owner_name": "名字",
        "owner_interests": "兴趣",
        "owner_style": "回复风格",
    }
    label = labels.get(field, field)
    entry = f"- {label}：{value}"

    text = soul_path.read_text(encoding="utf-8")

    # Find the ## Owner section and append the entry.
    owner_pattern = re.compile(r"(## Owner\s*\n)(.*?)(\n## |\Z)", re.DOTALL)
    m = owner_pattern.search(text)
    if m:
        section_content = m.group(2)
        # Remove HTML comment placeholder if present.
        section_content = re.sub(r"<!--.*?-->", "", section_content, flags=re.DOTALL).strip()
        if section_content:
            new_section = section_content + "\n" + entry + "\n"
        else:
            new_section = entry + "\n"
        replacement = m.group(1) + new_section
        if m.group(3).startswith("\n## "):
            replacement += "\n" + m.group(3)[1:]  # preserve next section header
        text = text[:m.start()] + replacement + (text[m.end():] if not m.group(3).startswith("\n## ") else text[m.end():])
    else:
        # No Owner section found — append one.
        text += f"\n## Owner\n{entry}\n"

    soul_path.write_text(text, encoding="utf-8")

    # Mark field as completed.
    ob = _load_onboarding_state()
    if field not in ob.get("completed_fields", []):
        ob.setdefault("completed_fields", []).append(field)
    _save_onboarding_state(ob)

    # Refresh cache.
    reload()
    log.info("Soul field '%s' written: %s", field, value)
