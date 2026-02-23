"""Registry HTTP client — pure stdlib (urllib.request + json).

Allows a Protea instance to publish, search, and download task templates
from a remote registry.  Follows the same retry + best-effort pattern as
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
    """HTTP client for the Protea Hub registry."""

    def __init__(
        self,
        registry_url: str = "https://protea-hub-production.up.railway.app",
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

    def get_stats(self) -> dict | None:
        """Get registry statistics."""
        return self._request("GET", "/api/stats")

    # ------------------------------------------------------------------
    # Task Template API
    # ------------------------------------------------------------------

    def publish_task_template(
        self,
        name: str,
        task_text: str,
        cron_expr: str,
        schedule_type: str = "cron",
        tags: list[str] | None = None,
        template_hash: str = "",
        category: str = "",
    ) -> dict | None:
        """Publish a task template to the registry."""
        body: dict = {
            "node_id": self.node_id,
            "name": name,
            "task_text": task_text,
            "cron_expr": cron_expr,
            "schedule_type": schedule_type,
            "tags": tags or [],
            "template_hash": template_hash,
            "category": category,
        }
        return self._request("POST", "/api/task-templates", body=body)

    def search_task_templates(
        self,
        query: str | None = None,
        tag: str | None = None,
        category: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search for task templates.  Returns a list of template dicts."""
        params = []
        if query:
            params.append(f"q={urllib.request.quote(query)}")
        if tag:
            params.append(f"tag={urllib.request.quote(tag)}")
        if category:
            params.append(f"category={urllib.request.quote(category)}")
        params.append(f"limit={limit}")
        qs = "&".join(params)
        result = self._request("GET", f"/api/task-templates?{qs}")
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return []

    def download_task_template(self, node_id: str, name: str) -> dict | None:
        """Download a task template (auto-increments downloads)."""
        return self._request("GET", f"/api/task-templates/{node_id}/{name}")
