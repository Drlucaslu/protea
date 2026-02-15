"""Skill Registry HTTP server — REST API for multi-node skill sharing.

Follows the same ThreadingHTTPServer + BaseHTTPRequestHandler pattern as
ring1/skill_portal.py.  Class attributes are injected before server start.
Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import logging
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

log = logging.getLogger("protea.registry_server")

# ---------------------------------------------------------------------------
# HTML templates (dark theme — matches skill_portal)
# ---------------------------------------------------------------------------

_BASE_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0e27; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; }
a { color: #667eea; text-decoration: none; }
a:hover { text-decoration: underline; }
.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
}
.header h1 { color: #fff; font-size: 1.5rem; }
.header nav a { color: rgba(255,255,255,0.85); margin-left: 1.5rem; font-size: 0.95rem; }
.header nav a:hover { color: #fff; text-decoration: none; }
.container { max-width: 1200px; margin: 2rem auto; padding: 0 1.5rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.2rem; }
.card {
    background: #151a3a; border: 1px solid #252a4a; border-radius: 10px;
    padding: 1.2rem; transition: border-color 0.2s;
}
.card:hover { border-color: #667eea; }
.card h3 { font-size: 1.1rem; margin-bottom: 0.4rem; }
.card .desc { color: #999; font-size: 0.85rem; margin-bottom: 0.6rem; }
.card .tags { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.6rem; }
.card .tag { background: #252a4a; color: #aaa; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 4px; }
.card .meta { display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #777; }
.stats { display: flex; gap: 2rem; margin-bottom: 2rem; }
.stat { background: #151a3a; border: 1px solid #252a4a; border-radius: 10px; padding: 1rem 1.5rem; text-align: center; }
.stat .value { font-size: 2rem; font-weight: 700; color: #667eea; }
.stat .label { font-size: 0.85rem; color: #777; margin-top: 0.3rem; }
"""

_NAV_HTML = '<nav><a href="/">Dashboard</a><a href="/api/skills">API</a><a href="/api/stats">Stats</a></nav>'


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title} — Protea Registry</title>"
        '<meta http-equiv="refresh" content="10">'
        f"<style>{_BASE_CSS}</style></head><body>"
        f'<div class="header"><h1>Protea Skill Registry</h1>{_NAV_HTML}</div>'
        f'<div class="container">{body}</div>'
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Route pattern for /api/skills/<node_id>/<name>
# ---------------------------------------------------------------------------

_SKILL_DETAIL_RE = re.compile(r"^/api/skills/([^/]+)/([^/]+)$")
_SKILL_RATE_RE = re.compile(r"^/api/skills/([^/]+)/([^/]+)/rate$")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class RegistryHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Skill Registry.

    Class attributes are injected before starting the server:
      store (RegistryStore)
    """

    store = None  # type: ignore[assignment]

    def log_message(self, format, *args):  # noqa: A002
        pass

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._serve_dashboard()
        elif path == "/api/skills":
            self._serve_list(parsed.query)
        elif path == "/api/stats":
            self._serve_stats()
        else:
            m = _SKILL_DETAIL_RE.match(path)
            if m:
                self._serve_detail(m.group(1), m.group(2))
            else:
                self._send_error(404, "Not Found")

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/skills":
            self._handle_publish()
        else:
            m = _SKILL_RATE_RE.match(path)
            if m:
                self._handle_rate(m.group(1), m.group(2))
            else:
                self._send_error(404, "Not Found")

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        m = _SKILL_DETAIL_RE.match(path)
        if m:
            self._handle_delete(m.group(1), m.group(2))
        else:
            self._send_error(404, "Not Found")

    # ------------------------------------------------------------------
    # Route implementations
    # ------------------------------------------------------------------

    def _serve_dashboard(self) -> None:
        stats = self.store.stats()
        skills = self.store.get_all(limit=100)

        stats_html = (
            '<div class="stats">'
            f'<div class="stat"><div class="value">{stats["total_skills"]}</div><div class="label">Skills</div></div>'
            f'<div class="stat"><div class="value">{stats["total_nodes"]}</div><div class="label">Nodes</div></div>'
            f'<div class="stat"><div class="value">{stats["total_downloads"]}</div><div class="label">Downloads</div></div>'
            '</div>'
        )

        cards = []
        for s in skills:
            tags_html = "".join(
                f'<span class="tag">{t}</span>' for t in (s.get("tags") or [])
            )
            name = s.get("name", "unknown")
            desc = (s.get("description") or "")[:120]
            node = s.get("node_id", "")
            dl = s.get("downloads", 0)
            ver = s.get("version", 1)
            cards.append(
                f'<div class="card">'
                f"<h3>{name}</h3>"
                f'<div class="desc">{desc}</div>'
                f'<div class="tags">{tags_html}</div>'
                f'<div class="meta"><span>{node} · v{ver}</span>'
                f"<span>↓{dl}</span></div>"
                f"</div>"
            )
        grid = '<div class="grid">' + "".join(cards) + "</div>"
        if not cards:
            grid = '<p style="color:#777">No skills registered yet.</p>'
        self._send_html(_page("Dashboard", stats_html + grid))

    def _serve_list(self, query_string: str) -> None:
        qs = parse_qs(query_string)
        q = qs.get("q", [None])[0]
        tag = qs.get("tag", [None])[0]
        tags = [tag] if tag else None
        limit = 50
        try:
            limit = int(qs.get("limit", ["50"])[0])
        except (ValueError, IndexError):
            pass
        skills = self.store.search(query=q, tags=tags, limit=limit)
        self._send_json(skills)

    def _serve_detail(self, node_id: str, name: str) -> None:
        skill = self.store.get(node_id, name)
        if skill is None:
            self._send_error(404, f"Skill '{node_id}/{name}' not found")
            return
        self.store.increment_downloads(node_id, name)
        # Re-read to get updated count.
        skill = self.store.get(node_id, name)
        self._send_json(skill)

    def _serve_stats(self) -> None:
        self._send_json(self.store.stats())

    def _handle_publish(self) -> None:
        body = self._read_json_body()
        if body is None:
            return
        node_id = body.get("node_id", "")
        name = body.get("name", "")
        if not node_id or not name:
            self._send_error(400, "Missing required fields: node_id, name")
            return
        description = body.get("description", "")
        prompt_template = body.get("prompt_template", "")
        parameters = body.get("parameters")
        tags = body.get("tags")
        source_code = body.get("source_code", "")
        rowid = self.store.publish(
            node_id=node_id,
            name=name,
            description=description,
            prompt_template=prompt_template,
            parameters=parameters,
            tags=tags,
            source_code=source_code,
        )
        skill = self.store.get(node_id, name)
        self._send_json(skill or {"id": rowid}, code=201)

    def _handle_rate(self, node_id: str, name: str) -> None:
        body = self._read_json_body()
        if body is None:
            return
        up = body.get("up", True)
        self.store.rate(node_id, name, up=up)
        self._send_json({"ok": True})

    def _handle_delete(self, node_id: str, name: str) -> None:
        deleted = self.store.delete(node_id, name)
        if deleted:
            self._send_json({"ok": True, "deleted": f"{node_id}/{name}"})
        else:
            self._send_error(404, f"Skill '{node_id}/{name}' not found")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_json_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            self._send_error(400, f"Invalid JSON body: {exc}")
            return None

    def _send_json(self, obj: object, code: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        self._send_json({"error": message}, code=code)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class RegistryServer:
    """HTTP server for the Skill Registry."""

    def __init__(self, store, host: str = "127.0.0.1", port: int = 8761) -> None:
        self._store = store
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None

    def run(self) -> None:
        """Start the HTTP server (blocking)."""
        handler = type(
            "InjectedRegistryHandler",
            (RegistryHandler,),
            {"store": self._store},
        )
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        log.info("Skill Registry listening on %s:%d", self._host, self.actual_port)
        self._server.serve_forever()

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server:
            self._server.shutdown()
            log.info("Skill Registry stopped")

    @property
    def actual_port(self) -> int:
        """Return the actual port (useful when port=0 for OS-assigned)."""
        if self._server:
            return self._server.server_address[1]
        return self._port
