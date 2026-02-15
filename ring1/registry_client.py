"""Registry HTTP client — pure stdlib (urllib.request + json).

Allows a Protea instance to publish, search, and download skills from a
remote Skill Registry.  Follows the same retry + best-effort pattern as
ring1/llm_client.py.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

log = logging.getLogger("protea.registry_client")

_MAX_RETRIES = 2
_BASE_DELAY = 1.0  # seconds


class RegistryClient:
    """HTTP client for the Skill Registry."""

    def __init__(
        self,
        registry_url: str = "http://127.0.0.1:8761",
        node_id: str = "default",
        timeout: int = 10,
    ) -> None:
        self.registry_url = registry_url.rstrip("/")
        self.node_id = node_id
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
    ) -> dict | None:
        """Send an HTTP request with retry.  Returns parsed JSON or None."""
        url = f"{self.registry_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        headers = {"Content-Type": "application/json"} if data else {}

        for attempt in range(_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=headers, method=method,
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                code = exc.code
                if code in {429, 500, 502, 503} and attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "Registry %s %s → %d — retry %d/%d in %.1fs",
                        method, path, code, attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                log.warning("Registry %s %s failed: HTTP %d", method, path, code)
                return None
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1:
                    delay = _BASE_DELAY * (2 ** attempt)
                    log.warning(
                        "Registry %s %s error — retry %d/%d in %.1fs: %s",
                        method, path, attempt + 1, _MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                    continue
                log.warning("Registry %s %s failed: %s", method, path, exc)
                return None
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish(
        self,
        name: str,
        description: str,
        prompt_template: str,
        parameters: dict | None = None,
        tags: list[str] | None = None,
        source_code: str = "",
    ) -> dict | None:
        """Publish a skill to the registry."""
        return self._request("POST", "/api/skills", body={
            "node_id": self.node_id,
            "name": name,
            "description": description,
            "prompt_template": prompt_template,
            "parameters": parameters or {},
            "tags": tags or [],
            "source_code": source_code,
        })

    def search(
        self,
        query: str | None = None,
        tag: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Search for skills.  Returns a list of skill dicts."""
        params = []
        if query:
            params.append(f"q={urllib.request.quote(query)}")
        if tag:
            params.append(f"tag={urllib.request.quote(tag)}")
        params.append(f"limit={limit}")
        qs = "&".join(params)
        result = self._request("GET", f"/api/skills?{qs}")
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return []

    def download(self, node_id: str, name: str) -> dict | None:
        """Download a skill (auto-increments downloads)."""
        return self._request("GET", f"/api/skills/{node_id}/{name}")

    def rate(self, node_id: str, name: str, up: bool = True) -> bool:
        """Rate a skill."""
        result = self._request(
            "POST", f"/api/skills/{node_id}/{name}/rate",
            body={"up": up},
        )
        return result is not None

    def unpublish(self, name: str) -> bool:
        """Delete own skill from the registry."""
        result = self._request("DELETE", f"/api/skills/{self.node_id}/{name}")
        return result is not None and result.get("ok", False)

    def get_stats(self) -> dict | None:
        """Get registry statistics."""
        return self._request("GET", "/api/stats")
