"""Skill Portal — unified web dashboard for all skills and reports.

Provides a persistent HTTP server (default port 8888) that acts as the
single entry point for browsing skills, viewing running skill UIs via
iframe, and serving analysis reports.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

log = logging.getLogger("protea.skill_portal")

# ---------------------------------------------------------------------------
# HTML templates (dark theme)
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
"""

_CARD_CSS = """\
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
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.badge-running { background: rgba(78,205,196,0.15); color: #4ecdc4; }
.badge-stopped { background: rgba(255,107,107,0.15); color: #ff6b6b; }
"""

_NAV_HTML = """\
<nav><a href="/">Dashboard</a><a href="/reports">Reports</a><a href="/api/skills">API</a></nav>
"""


def _page(title: str, body: str, extra_css: str = "", refresh: bool = False) -> str:
    meta = '<meta http-equiv="refresh" content="10">' if refresh else ""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title} — Protea</title>{meta}"
        f"<style>{_BASE_CSS}{extra_css}</style></head><body>"
        f'<div class="header"><h1>Protea Skill Portal</h1>{_NAV_HTML}</div>'
        f'<div class="container">{body}</div>'
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class PortalHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Skill Portal.

    Class attributes are injected before starting the server:
      skill_store, skill_runner, project_root, reports_dir
    """

    skill_store = None
    skill_runner = None
    project_root: pathlib.Path | None = None
    reports_dir: pathlib.Path | None = None

    # Silence per-request logs.
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(self.path).split("?")[0].rstrip("/") or "/"

        if path == "/":
            self._serve_dashboard()
        elif path == "/reports":
            self._serve_reports_list()
        elif path.startswith("/reports/"):
            self._serve_report_file(path[len("/reports/"):])
        elif path == "/api/skills":
            self._serve_api_skills()
        elif path == "/api/status":
            self._serve_api_status()
        elif path.startswith("/skill/"):
            self._serve_skill_page(path[len("/skill/"):])
        else:
            self._send_error(404, "Not Found")

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    def _serve_dashboard(self) -> None:
        skills = self._get_skills_with_status()
        cards = []
        for s in skills:
            tags_html = "".join(
                f'<span class="tag">{t}</span>' for t in (s.get("tags") or [])
            )
            running = s.get("running", False)
            badge_cls = "badge-running" if running else "badge-stopped"
            badge_txt = "running" if running else "stopped"
            name = s.get("name", "unknown")
            desc = s.get("description", "")[:120]
            usage = s.get("usage_count", 0)
            cards.append(
                f'<a href="/skill/{name}" class="card">'
                f"<h3>{name}</h3>"
                f'<div class="desc">{desc}</div>'
                f'<div class="tags">{tags_html}</div>'
                f'<div class="meta"><span>usage: {usage}</span>'
                f'<span class="badge {badge_cls}">{badge_txt}</span></div>'
                f"</a>"
            )
        body = '<div class="grid">' + "".join(cards) + "</div>"
        if not cards:
            body = '<p style="color:#777">No skills registered yet.</p>'
        self._send_html(_page("Dashboard", body, _CARD_CSS, refresh=True))

    def _serve_skill_page(self, name: str) -> None:
        # Check running info first.
        runner_info = None
        if self.skill_runner:
            info = self.skill_runner.get_info()
            if info and info.get("skill_name") == name and info.get("running"):
                runner_info = info

        # Get skill metadata from store.
        skill_data = None
        if self.skill_store:
            skill_data = self.skill_store.get_by_name(name)

        if skill_data is None and runner_info is None:
            self._send_error(404, f"Skill '{name}' not found")
            return

        if runner_info and runner_info.get("port"):
            port = runner_info["port"]
            body = (
                f"<h2>{name} <span class='badge badge-running'>running</span></h2>"
                f'<p style="margin:0.8rem 0;color:#777">Port {port} · '
                f'PID {runner_info["pid"]} · '
                f'uptime {runner_info["uptime"]:.0f}s</p>'
                f'<iframe src="http://localhost:{port}" '
                f'style="width:100%;height:80vh;border:1px solid #252a4a;'
                f'border-radius:8px;background:#fff;"></iframe>'
            )
        else:
            desc = skill_data.get("description", "") if skill_data else ""
            tags = skill_data.get("tags", []) if skill_data else []
            source = skill_data.get("source", "") if skill_data else ""
            tags_html = " ".join(f'<span class="tag">{t}</span>' for t in tags)
            body = (
                f"<h2>{name} <span class='badge badge-stopped'>not running</span></h2>"
                f'<p style="margin:0.8rem 0;color:#999">{desc}</p>'
                f'<div class="tags" style="margin-bottom:1rem">{tags_html}</div>'
                f'<p style="color:#777">Source: {source}</p>'
                f'<p style="color:#ff6b6b;margin-top:1rem">'
                f"This skill is not currently running. "
                f"Start it via Telegram or the task executor.</p>"
            )

        self._send_html(_page(name, body, _CARD_CSS))

    _REPORT_EXTENSIONS = {".html", ".md", ".pdf"}

    def _serve_reports_list(self) -> None:
        reports_dir = self.reports_dir
        if reports_dir is None or not reports_dir.is_dir():
            body = '<p style="color:#777">No reports directory found.</p>'
            self._send_html(_page("Reports", body))
            return

        # Collect all supported report files
        files: list[pathlib.Path] = []
        for ext in sorted(self._REPORT_EXTENSIONS):
            files.extend(reports_dir.glob(f"*{ext}"))

        if not files:
            body = '<p style="color:#777">No reports available yet.</p>'
            self._send_html(_page("Reports", body, _CARD_CSS))
            return

        # Group by stem (filename without extension)
        groups: dict[str, list[pathlib.Path]] = {}
        for f in files:
            groups.setdefault(f.stem, []).append(f)

        # Sort groups by newest file mtime descending
        sorted_groups = sorted(
            groups.items(),
            key=lambda kv: max(f.stat().st_mtime for f in kv[1]),
            reverse=True,
        )

        badge_colors = {
            ".html": ("HTML", "#e74c3c"),
            ".md": ("MD", "#3498db"),
            ".pdf": ("PDF", "#27ae60"),
        }

        items = []
        for stem, group_files in sorted_groups:
            newest = max(group_files, key=lambda f: f.stat().st_mtime)
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(newest.stat().st_mtime))
            # Format badges / links for each available format
            format_links = []
            for f in sorted(group_files, key=lambda f: f.suffix):
                label, color = badge_colors.get(f.suffix, (f.suffix.upper(), "#999"))
                format_links.append(
                    f'<a href="/reports/{f.name}" class="badge" '
                    f'style="background:rgba(255,255,255,0.05);color:{color};'
                    f'border:1px solid {color};margin-right:0.3rem">{label}</a>'
                )
            formats_html = "".join(format_links)
            items.append(
                f'<div class="card">'
                f"<h3>{stem}</h3>"
                f'<div class="tags" style="margin:0.5rem 0">{formats_html}</div>'
                f'<div class="meta"><span>{mtime}</span></div>'
                f"</div>"
            )
        body = '<div class="grid">' + "".join(items) + "</div>"
        self._send_html(_page("Reports", body, _CARD_CSS))

    _CONTENT_TYPES = {
        ".html": "text/html; charset=utf-8",
        ".md": "text/plain; charset=utf-8",
        ".pdf": "application/pdf",
    }

    def _serve_report_file(self, filename: str) -> None:
        suffix = pathlib.Path(filename).suffix.lower()
        if suffix not in self._REPORT_EXTENSIONS:
            self._send_error(403, f"Only {', '.join(sorted(self._REPORT_EXTENSIONS))} files are served")
            return
        reports_dir = self.reports_dir
        if reports_dir is None:
            self._send_error(404, "Reports directory not configured")
            return
        filepath = (reports_dir / filename).resolve()
        # Path traversal protection.
        if not str(filepath).startswith(str(reports_dir.resolve())):
            self._send_error(403, "Forbidden")
            return
        if not filepath.is_file():
            self._send_error(404, "Report not found")
            return
        content_type = self._CONTENT_TYPES.get(suffix, "application/octet-stream")
        if suffix == ".pdf":
            data = filepath.read_bytes()
        else:
            data = filepath.read_text(errors="replace").encode("utf-8")
        self._send_response(200, content_type, data)

    def _serve_api_skills(self) -> None:
        self._send_json(self._get_skills_with_status())

    def _serve_api_status(self) -> None:
        self._send_json({
            "portal": "running",
            "timestamp": time.time(),
            "project_root": str(self.project_root) if self.project_root else None,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_skills_with_status(self) -> list[dict]:
        if self.skill_store is None:
            return []
        try:
            skills = self.skill_store.get_active()
        except Exception:
            return []

        # Annotate with running status from skill_runner.
        runner_info = None
        if self.skill_runner:
            runner_info = self.skill_runner.get_info()

        result = []
        for s in skills:
            entry = {
                "name": s.get("name", ""),
                "description": s.get("description", ""),
                "tags": s.get("tags", []),
                "usage_count": s.get("usage_count", 0),
                "source": s.get("source", ""),
                "active": s.get("active", True),
                "running": False,
                "port": None,
            }
            if runner_info and runner_info.get("skill_name") == entry["name"] and runner_info.get("running"):
                entry["running"] = True
                entry["port"] = runner_info.get("port")
            result.append(entry)
        return result

    def _send_response(self, code: int, content_type: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self._send_response(200, "text/html; charset=utf-8", data)

    def _send_json(self, obj: object) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        body = _page("Error", f'<h2>{code}</h2><p style="color:#ff6b6b">{message}</p>')
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Portal server
# ---------------------------------------------------------------------------


class SkillPortal:
    """Persistent web portal for skill management and reports."""

    def __init__(
        self,
        skill_store,
        skill_runner,
        project_root: pathlib.Path,
        host: str = "127.0.0.1",
        port: int = 8888,
    ) -> None:
        self._skill_store = skill_store
        self._skill_runner = skill_runner
        self._project_root = project_root
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None

    def run(self) -> None:
        """Start the HTTP server (blocking — call from a thread)."""
        handler = type(
            "InjectedPortalHandler",
            (PortalHandler,),
            {
                "skill_store": self._skill_store,
                "skill_runner": self._skill_runner,
                "project_root": self._project_root,
                "reports_dir": self._project_root / "output" / "reports",
            },
        )
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        log.info("Skill Portal listening on %s:%d", self._host, self.actual_port)
        self._server.serve_forever()

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server:
            self._server.shutdown()
            log.info("Skill Portal stopped")

    @property
    def actual_port(self) -> int:
        """Return the actual port (useful when port=0 for OS-assigned)."""
        if self._server:
            return self._server.server_address[1]
        return self._port


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_portal(skill_store, skill_runner, project_root: pathlib.Path, config: dict) -> SkillPortal | None:
    """Create a SkillPortal from configuration.  Returns None if disabled."""
    portal_cfg = config.get("ring1", {}).get("portal", {})
    if not portal_cfg.get("enabled", False):
        return None
    host = portal_cfg.get("host", "127.0.0.1")
    port = portal_cfg.get("port", 8888)
    return SkillPortal(skill_store, skill_runner, project_root, host=host, port=port)


def start_portal_thread(portal: SkillPortal) -> threading.Thread:
    """Start the portal in a daemon thread and return the thread."""
    t = threading.Thread(target=portal.run, name="skill-portal", daemon=True)
    t.start()
    return t
