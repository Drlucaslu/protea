"""Task API — lightweight HTTP server for external tool delegation.

Allows external bots (e.g. hive) to delegate tasks that require web search,
file operations, or shell commands to Protea's LLM + tool infrastructure.

Endpoints:
  POST /api/task  — run a task through LLM + tools, return result
  GET  /api/health — health check
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("protea.task_api")

TASK_SYSTEM_PROMPT = """\
You are a research assistant. You have tools for web search, web browsing, \
file operations, and shell commands.

Execute the user's task and return a concise, data-focused result. \
Do NOT include pleasantries, disclaimers, or filler — just the information requested.

If the task requires multiple steps, use tools as needed and synthesize the result.
Respond in the same language as the task.
"""


class TaskAPIHandler(BaseHTTPRequestHandler):
    """Handle /api/task and /api/health requests."""

    # Silence per-request logging from BaseHTTPRequestHandler.
    def log_message(self, format, *args):
        pass

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        """Validate Bearer token. Returns True if OK."""
        secret = self.server.api_secret
        if not secret:
            return True  # no secret configured = open
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {secret}":
            return True
        self._send_json(401, {"status": "error", "error": "unauthorized"})
        return False

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"status": "error", "error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/task":
            self._send_json(404, {"status": "error", "error": "not found"})
            return

        if not self._check_auth():
            return

        # Read body.
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"status": "error", "error": "empty body"})
            return
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"status": "error", "error": "invalid JSON"})
            return

        task_text = body.get("task", "").strip()
        source = body.get("source", "unknown")
        if not task_text:
            self._send_json(400, {"status": "error", "error": "missing 'task' field"})
            return

        log.info("Task from %s: %s", source, task_text[:120])

        # Execute via LLM + tools.
        try:
            result = self.server.execute_task(task_text)
            log.info("Task complete (%s): %d chars", source, len(result))
            self._send_json(200, {"status": "ok", "result": result})
        except Exception as exc:
            log.error("Task failed (%s): %s", source, exc, exc_info=True)
            self._send_json(500, {"status": "error", "error": str(exc)})


class TaskAPI:
    """HTTP server wrapper for the task API."""

    def __init__(
        self,
        client,
        registry,
        host: str = "127.0.0.1",
        port: int = 8877,
        secret: str = "",
        max_tool_rounds: int = 25,
    ) -> None:
        self._client = client
        self._registry = registry
        self._host = host
        self._port = port
        self._secret = secret
        self._max_tool_rounds = max_tool_rounds
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def execute_task(self, task_text: str) -> str:
        """Run a task through LLM + tools and return the text result."""
        if self._registry:
            response = self._client.send_message_with_tools(
                TASK_SYSTEM_PROMPT,
                task_text,
                tools=self._registry.get_schemas(),
                tool_executor=self._registry.execute,
                max_rounds=self._max_tool_rounds,
            )
        else:
            response = self._client.send_message(
                TASK_SYSTEM_PROMPT,
                task_text,
            )
        return response

    def start(self) -> None:
        """Start the HTTP server in a daemon thread."""
        server = ThreadingHTTPServer((self._host, self._port), TaskAPIHandler)
        server.api_secret = self._secret
        server.execute_task = self.execute_task
        self._server = server

        self._thread = threading.Thread(
            target=server.serve_forever,
            name="task-api",
            daemon=True,
        )
        self._thread.start()
        log.info("Task API started on %s:%d", self._host, self._port)

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server:
            self._server.shutdown()
            log.info("Task API stopped")


def create_task_api(config, registry, project_root) -> TaskAPI | None:
    """Create a TaskAPI from config, or None if disabled.

    config: the full parsed config dict (from config.toml).
    registry: ToolRegistry instance (will be cloned to strip messaging tools).
    """
    import pathlib

    cfg = config.get("ring1", {}).get("task_api", {})
    if not cfg.get("enabled", False):
        return None

    # Load LLM client.
    try:
        from ring1.config import load_ring1_config
        r1_config = load_ring1_config(pathlib.Path(project_root))
        client = r1_config.get_llm_client()
    except Exception as exc:
        log.warning("Task API: cannot create LLM client: %s", exc)
        return None

    # Clone registry without Telegram-specific and dangerous tools.
    task_registry = None
    if registry:
        task_registry = registry.clone_without(
            "message", "spawn", "send_file", "edit_skill", "schedule",
        )

    host = cfg.get("host", "127.0.0.1")
    port = cfg.get("port", 8877)
    secret_env = cfg.get("secret_env", "PROTEA_TASK_API_SECRET")
    secret = os.environ.get(secret_env, "")
    max_rounds = cfg.get("max_tool_rounds", 25)

    return TaskAPI(
        client=client,
        registry=task_registry,
        host=host,
        port=port,
        secret=secret,
        max_tool_rounds=max_rounds,
    )
