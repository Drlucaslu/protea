"""Federated skill discovery — adapters for external skill registries.

Each adapter normalizes results to a common dict format so SkillSyncer
can treat all sources uniformly.  Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

log = logging.getLogger("protea.skill_sources")

_MAX_RETRIES = 2
_BASE_DELAY = 1.0
_TIMEOUT = 5


def _http_get(url: str, timeout: int = _TIMEOUT) -> dict | list | None:
    """Best-effort HTTP GET with retry.  Returns parsed JSON or None."""
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503} and attempt < _MAX_RETRIES - 1:
                time.sleep(_BASE_DELAY * (2 ** attempt))
                continue
            return None
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BASE_DELAY * (2 ** attempt))
                continue
            return None
    return None


class SkillSource:
    """Base class for external skill discovery sources."""

    name: str = ""

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search for skills matching *query*.  Returns normalized dicts."""
        return []

    def download(self, identifier: str) -> dict | None:
        """Download a skill by name/identifier.  Returns normalized dict or None."""
        return None


class ProteaHubSource(SkillSource):
    """Wraps RegistryClient for the unified source interface."""

    name = "protea-hub"

    def __init__(self, registry_client) -> None:
        self._client = registry_client

    def search(self, query: str, limit: int = 10) -> list[dict]:
        results = self._client.search(query=query, limit=limit)
        return [self._normalize(r) for r in results]

    def download(self, identifier: str) -> dict | None:
        # identifier is "name" — use any-node download.
        # Try to find node_id from a prior search; fallback to download-by-name.
        result = self._client._request("GET", f"/api/download/{identifier}")
        if result is None:
            return None
        return self._normalize(result)

    @staticmethod
    def _normalize(raw: dict) -> dict:
        return {
            "name": raw.get("name", ""),
            "node_id": raw.get("node_id", ""),
            "description": raw.get("description", ""),
            "prompt_template": raw.get("prompt_template", ""),
            "tags": raw.get("tags", []),
            "source_code": raw.get("source_code", ""),
            "source": "hub:protea-hub",
        }


class ClawHubSource(SkillSource):
    """ClawHub registry (OpenClaw SKILL.md format)."""

    name = "clawhub"

    def __init__(self, base_url: str = "https://openclawskill.ai") -> None:
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, limit: int = 10) -> list[dict]:
        q = urllib.request.quote(query)
        url = f"{self.base_url}/api/skills?q={q}&limit={limit}"
        data = _http_get(url)
        if not data or not isinstance(data, list):
            return []
        return [self._normalize(r) for r in data[:limit]]

    def download(self, identifier: str) -> dict | None:
        url = f"{self.base_url}/api/skills/{urllib.request.quote(identifier)}"
        data = _http_get(url)
        if not data or not isinstance(data, dict):
            return None
        return self._normalize(data)

    @staticmethod
    def _normalize(raw: dict) -> dict:
        return {
            "name": raw.get("name", ""),
            "node_id": raw.get("node_id", raw.get("author", "")),
            "description": raw.get("description", ""),
            "prompt_template": raw.get("prompt_template", raw.get("prompt", "")),
            "tags": raw.get("tags", []),
            "source_code": raw.get("source_code", raw.get("code", "")),
            "source": "hub:clawhub",
        }


class SkillsShSource(SkillSource):
    """Vercel Skills.sh registry (Agent Skills format)."""

    name = "skills.sh"

    def __init__(self, base_url: str = "https://skills.sh") -> None:
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, limit: int = 10) -> list[dict]:
        q = urllib.request.quote(query)
        url = f"{self.base_url}/api/skills?q={q}&limit={limit}"
        data = _http_get(url)
        if not data or not isinstance(data, list):
            return []
        return [self._normalize(r) for r in data[:limit]]

    def download(self, identifier: str) -> dict | None:
        url = f"{self.base_url}/api/skills/{urllib.request.quote(identifier)}"
        data = _http_get(url)
        if not data or not isinstance(data, dict):
            return None
        return self._normalize(data)

    @staticmethod
    def _normalize(raw: dict) -> dict:
        return {
            "name": raw.get("name", ""),
            "node_id": raw.get("node_id", raw.get("author", "")),
            "description": raw.get("description", ""),
            "prompt_template": raw.get("prompt_template", raw.get("prompt", "")),
            "tags": raw.get("tags", []),
            "source_code": raw.get("source_code", raw.get("code", "")),
            "source": "hub:skills.sh",
        }
