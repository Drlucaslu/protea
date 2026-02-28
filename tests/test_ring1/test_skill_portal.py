"""Tests for ring1.skill_portal."""

from __future__ import annotations

import json
import pathlib
import threading
import time
import urllib.request

from ring1.skill_portal import SkillPortal, create_portal


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeSkillStore:
    """Minimal stub for SkillStore."""

    def __init__(self, skills: list[dict] | None = None) -> None:
        self._skills = skills or []

    def get_active(self, limit: int = 50) -> list[dict]:
        return self._skills[:limit]

    def get_by_name(self, name: str) -> dict | None:
        for s in self._skills:
            if s["name"] == name:
                return s
        return None


class _FakeSkillRunner:
    """Minimal stub for SkillRunner."""

    def __init__(self, info: dict | None = None) -> None:
        self._info = info

    def get_info(self) -> dict | None:
        return self._info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_SKILLS = [
    {
        "name": "web_scraper",
        "description": "Scrapes web pages",
        "tags": ["web", "scraping"],
        "usage_count": 5,
        "source": "crystallized",
        "active": True,
    },
    {
        "name": "data_viz",
        "description": "Data visualization tool",
        "tags": ["data"],
        "usage_count": 3,
        "source": "user",
        "active": True,
    },
]


def _start_portal(
    tmp_path: pathlib.Path,
    skill_store=None,
    skill_runner=None,
) -> SkillPortal:
    """Start a portal on an OS-assigned port and wait until ready."""
    portal = SkillPortal(
        skill_store=skill_store,
        skill_runner=skill_runner,
        project_root=tmp_path,
        host="127.0.0.1",
        port=0,
    )
    t = threading.Thread(target=portal.run, daemon=True)
    t.start()
    # Wait for the server to bind.
    for _ in range(50):
        if portal._server is not None:
            break
        time.sleep(0.05)
    return portal


def _get(portal: SkillPortal, path: str) -> tuple[int, str]:
    """Make a GET request and return (status_code, body)."""
    url = f"http://127.0.0.1:{portal.actual_port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# TestDashboard
# ---------------------------------------------------------------------------


class TestDashboard:
    def test_root_returns_html(self, tmp_path):
        store = _FakeSkillStore(_SAMPLE_SKILLS)
        portal = _start_portal(tmp_path, skill_store=store)
        try:
            code, body = _get(portal, "/")
            assert code == 200
            assert "Protea" in body
            assert "text/html" in body or "web_scraper" in body
        finally:
            portal.stop()

    def test_dashboard_shows_skills(self, tmp_path):
        store = _FakeSkillStore(_SAMPLE_SKILLS)
        portal = _start_portal(tmp_path, skill_store=store)
        try:
            code, body = _get(portal, "/")
            assert code == 200
            assert "web_scraper" in body
            assert "data_viz" in body
        finally:
            portal.stop()

    def test_dashboard_empty_skills(self, tmp_path):
        store = _FakeSkillStore([])
        portal = _start_portal(tmp_path, skill_store=store)
        try:
            code, body = _get(portal, "/")
            assert code == 200
            assert "No skills" in body
        finally:
            portal.stop()


# ---------------------------------------------------------------------------
# TestApiSkills
# ---------------------------------------------------------------------------


class TestApiSkills:
    def test_returns_json_array(self, tmp_path):
        store = _FakeSkillStore(_SAMPLE_SKILLS)
        portal = _start_portal(tmp_path, skill_store=store)
        try:
            code, body = _get(portal, "/api/skills")
            assert code == 200
            data = json.loads(body)
            assert isinstance(data, list)
            assert len(data) == 2
        finally:
            portal.stop()

    def test_running_status(self, tmp_path):
        store = _FakeSkillStore(_SAMPLE_SKILLS)
        runner = _FakeSkillRunner({
            "skill_name": "web_scraper",
            "pid": 12345,
            "running": True,
            "uptime": 60.0,
            "port": 9000,
        })
        portal = _start_portal(tmp_path, skill_store=store, skill_runner=runner)
        try:
            code, body = _get(portal, "/api/skills")
            data = json.loads(body)
            ws = next(s for s in data if s["name"] == "web_scraper")
            assert ws["running"] is True
            assert ws["port"] == 9000
            dv = next(s for s in data if s["name"] == "data_viz")
            assert dv["running"] is False
        finally:
            portal.stop()

    def test_store_none_returns_empty(self, tmp_path):
        portal = _start_portal(tmp_path, skill_store=None)
        try:
            code, body = _get(portal, "/api/skills")
            assert code == 200
            data = json.loads(body)
            assert data == []
        finally:
            portal.stop()


# ---------------------------------------------------------------------------
# TestSkillPage
# ---------------------------------------------------------------------------


class TestSkillPage:
    def test_running_skill_has_iframe(self, tmp_path):
        store = _FakeSkillStore(_SAMPLE_SKILLS)
        runner = _FakeSkillRunner({
            "skill_name": "web_scraper",
            "pid": 100,
            "running": True,
            "uptime": 10.0,
            "port": 7777,
        })
        portal = _start_portal(tmp_path, skill_store=store, skill_runner=runner)
        try:
            code, body = _get(portal, "/skill/web_scraper")
            assert code == 200
            assert "iframe" in body
            assert "7777" in body
        finally:
            portal.stop()

    def test_stopped_skill_shows_hint(self, tmp_path):
        store = _FakeSkillStore(_SAMPLE_SKILLS)
        portal = _start_portal(tmp_path, skill_store=store)
        try:
            code, body = _get(portal, "/skill/web_scraper")
            assert code == 200
            assert "not currently running" in body
        finally:
            portal.stop()

    def test_unknown_skill_404(self, tmp_path):
        store = _FakeSkillStore(_SAMPLE_SKILLS)
        portal = _start_portal(tmp_path, skill_store=store)
        try:
            code, _ = _get(portal, "/skill/nonexistent")
            assert code == 404
        finally:
            portal.stop()


# ---------------------------------------------------------------------------
# TestReportsPage
# ---------------------------------------------------------------------------


class TestReportsPage:
    def test_lists_html_files(self, tmp_path):
        reports_dir = tmp_path / "output" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "analysis.html").write_text("<h1>Report</h1>")
        (reports_dir / "summary.html").write_text("<h1>Summary</h1>")
        portal = _start_portal(tmp_path)
        try:
            code, body = _get(portal, "/reports")
            assert code == 200
            # Grouped by stem — card titles show stem names
            assert "analysis" in body
            assert "summary" in body
        finally:
            portal.stop()

    def test_lists_md_and_pdf(self, tmp_path):
        reports_dir = tmp_path / "output" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "research.md").write_text("# Research")
        (reports_dir / "research.pdf").write_bytes(b"%PDF-1.4 fake")
        portal = _start_portal(tmp_path)
        try:
            code, body = _get(portal, "/reports")
            assert code == 200
            assert "research" in body
            assert "MD" in body
            assert "PDF" in body
        finally:
            portal.stop()

    def test_multi_format_grouped(self, tmp_path):
        """Same stem with .html, .md, .pdf should appear as one card."""
        reports_dir = tmp_path / "output" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "report.html").write_text("<h1>HTML</h1>")
        (reports_dir / "report.md").write_text("# MD")
        (reports_dir / "report.pdf").write_bytes(b"%PDF-1.4 fake")
        portal = _start_portal(tmp_path)
        try:
            code, body = _get(portal, "/reports")
            assert code == 200
            # One card for "report" with all three format badges
            assert "HTML" in body
            assert "MD" in body
            assert "PDF" in body
            # Links to all three files
            assert "/reports/report.html" in body
            assert "/reports/report.md" in body
            assert "/reports/report.pdf" in body
        finally:
            portal.stop()

    def test_no_reports_dir_graceful(self, tmp_path):
        # No reports/ directory at all.
        portal = _start_portal(tmp_path)
        try:
            code, body = _get(portal, "/reports")
            assert code == 200
            assert "No reports" in body
        finally:
            portal.stop()

    def test_serve_report_content(self, tmp_path):
        reports_dir = tmp_path / "output" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "test.html").write_text("<h1>Test Report</h1>")
        portal = _start_portal(tmp_path)
        try:
            code, body = _get(portal, "/reports/test.html")
            assert code == 200
            assert "<h1>Test Report</h1>" in body
        finally:
            portal.stop()

    def test_serve_markdown(self, tmp_path):
        reports_dir = tmp_path / "output" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "notes.md").write_text("# Notes\n\nHello world")
        portal = _start_portal(tmp_path)
        try:
            code, body = _get(portal, "/reports/notes.md")
            assert code == 200
            assert "# Notes" in body
        finally:
            portal.stop()

    def test_serve_pdf(self, tmp_path):
        reports_dir = tmp_path / "output" / "reports"
        reports_dir.mkdir(parents=True)
        pdf_content = b"%PDF-1.4 fake pdf content"
        (reports_dir / "doc.pdf").write_bytes(pdf_content)
        portal = _start_portal(tmp_path)
        try:
            url = f"http://127.0.0.1:{portal.actual_port}/reports/doc.pdf"
            with urllib.request.urlopen(url, timeout=5) as resp:
                assert resp.status == 200
                assert resp.headers["Content-Type"] == "application/pdf"
                data = resp.read()
                assert data == pdf_content
        finally:
            portal.stop()

    def test_reject_unsupported_extension(self, tmp_path):
        reports_dir = tmp_path / "output" / "reports"
        reports_dir.mkdir(parents=True)
        (reports_dir / "data.json").write_text("{}")
        portal = _start_portal(tmp_path)
        try:
            code, _ = _get(portal, "/reports/data.json")
            assert code == 403
        finally:
            portal.stop()

    def test_reject_path_traversal(self, tmp_path):
        reports_dir = tmp_path / "output" / "reports"
        reports_dir.mkdir(parents=True)
        # Create a file outside reports dir.
        (tmp_path / "secret.html").write_text("secret")
        portal = _start_portal(tmp_path)
        try:
            code, _ = _get(portal, "/reports/../secret.html")
            assert code == 403
        finally:
            portal.stop()


# ---------------------------------------------------------------------------
# TestCreatePortal
# ---------------------------------------------------------------------------


class TestCreatePortal:
    def test_enabled(self):
        cfg = {"ring1": {"portal": {"enabled": True, "host": "127.0.0.1", "port": 9999}}}
        portal = create_portal(None, None, pathlib.Path("/tmp"), cfg)
        assert portal is not None
        assert portal._port == 9999

    def test_disabled(self):
        cfg = {"ring1": {"portal": {"enabled": False}}}
        portal = create_portal(None, None, pathlib.Path("/tmp"), cfg)
        assert portal is None

    def test_default_port(self):
        cfg = {"ring1": {"portal": {"enabled": True}}}
        portal = create_portal(None, None, pathlib.Path("/tmp"), cfg)
        assert portal is not None
        assert portal._port == 8888

    def test_missing_section(self):
        cfg = {"ring1": {}}
        portal = create_portal(None, None, pathlib.Path("/tmp"), cfg)
        assert portal is None


# ---------------------------------------------------------------------------
# TestPortalLifecycle
# ---------------------------------------------------------------------------


class TestPortalLifecycle:
    def test_start_and_stop(self, tmp_path):
        portal = _start_portal(tmp_path)
        try:
            # Should respond to requests.
            code, body = _get(portal, "/api/status")
            assert code == 200
            data = json.loads(body)
            assert data["portal"] == "running"
        finally:
            portal.stop()

    def test_stop_joins_cleanly(self, tmp_path):
        portal = SkillPortal(None, None, tmp_path, port=0)
        t = threading.Thread(target=portal.run, daemon=True)
        t.start()
        for _ in range(50):
            if portal._server is not None:
                break
            time.sleep(0.05)
        portal.stop()
        t.join(timeout=5)
        assert not t.is_alive()
