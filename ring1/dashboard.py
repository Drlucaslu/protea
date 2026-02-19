"""Dashboard — local web UI for Protea system state visualization.

Provides an HTTP server (default port 8899) with pages for memory browsing,
skill gallery, intent timeline, user profile, and a system overview with
SVG fitness charts.

Architecture mirrors skill_portal.py: DashboardHandler + type()-injected
class attributes + factory + thread helper.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import html
import json
import logging
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote

log = logging.getLogger("protea.dashboard")

# ---------------------------------------------------------------------------
# CSS
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
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 1.2rem; margin-bottom: 2rem; }
.card {
    background: #151a3a; border: 1px solid #252a4a; border-radius: 10px;
    padding: 1.2rem; transition: border-color 0.2s;
}
.card:hover { border-color: #667eea; }
.card h3 { font-size: 1rem; color: #667eea; margin-bottom: 0.4rem; }
.card .value { font-size: 1.8rem; font-weight: 700; color: #fff; }
.card .detail { font-size: 0.8rem; color: #777; margin-top: 0.3rem; }
table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #1a1f3a; font-size: 0.85rem; }
th { color: #667eea; font-weight: 600; }
tr:hover { background: #151a3a; }
.tier-hot { color: #ff6b6b; }
.tier-warm { color: #feca57; }
.tier-cold { color: #48dbfb; }
.filters { margin: 1rem 0; display: flex; gap: 0.8rem; flex-wrap: wrap; }
.filters a { padding: 0.3rem 0.8rem; border-radius: 4px; background: #151a3a; color: #999; font-size: 0.85rem; }
.filters a.active, .filters a:hover { background: #252a4a; color: #667eea; }
.timeline { margin: 1rem 0; }
.timeline-item { position: relative; padding-left: 2rem; padding-bottom: 1.5rem; border-left: 2px solid #252a4a; }
.timeline-item:last-child { border-left-color: transparent; }
.timeline-item::before {
    content: ''; position: absolute; left: -6px; top: 0.3rem;
    width: 10px; height: 10px; border-radius: 50%; background: #667eea;
}
.timeline-item .gen { color: #667eea; font-weight: 600; font-size: 0.9rem; }
.timeline-item .info { color: #999; font-size: 0.8rem; margin-top: 0.2rem; }
.bar-container { display: flex; align-items: center; gap: 0.5rem; margin: 0.4rem 0; }
.bar-label { width: 80px; text-align: right; font-size: 0.85rem; color: #999; }
.bar { height: 20px; border-radius: 3px; background: linear-gradient(90deg, #667eea, #764ba2); }
.bar-value { font-size: 0.8rem; color: #777; min-width: 50px; }
.badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1.2rem; }
.skill-card { background: #151a3a; border: 1px solid #252a4a; border-radius: 10px; padding: 1.2rem; }
.skill-card:hover { border-color: #667eea; }
.skill-card h3 { font-size: 1rem; margin-bottom: 0.3rem; }
.skill-card .desc { color: #999; font-size: 0.8rem; margin-bottom: 0.5rem; }
.skill-card .tags { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.5rem; }
.skill-card .tag { background: #252a4a; color: #aaa; font-size: 0.7rem; padding: 0.1rem 0.4rem; border-radius: 3px; }
.skill-card .meta { font-size: 0.75rem; color: #666; }
.profile-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
@media (max-width: 768px) { .profile-grid { grid-template-columns: 1fr; } }
.stat-item { margin: 0.5rem 0; font-size: 0.9rem; }
.stat-item .label { color: #999; }
.stat-item .val { color: #fff; font-weight: 600; }
.importance-bar { display: inline-block; height: 12px; border-radius: 2px; background: #667eea; vertical-align: middle; }
.intent-repair .gen { color: #ff6b6b; }
.intent-repair::before { background: #ff6b6b; }
.intent-optimize .gen { color: #4ecdc4; }
.intent-optimize::before { background: #4ecdc4; }
.intent-explore .gen { color: #667eea; }
.intent-explore::before { background: #667eea; }
.intent-adapt .gen { color: #feca57; }
.intent-adapt::before { background: #feca57; }
"""

_NAV_HTML = """\
<nav>
<a href="/">Overview</a>
<a href="/memory">Memory</a>
<a href="/skills">Skills</a>
<a href="/genes">Genes</a>
<a href="/intent">Intent</a>
<a href="/profile">Profile</a>
<a href="/schedule">Schedule</a>
</nav>
"""


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text))


def _page(title: str, body: str, refresh: bool = True) -> str:
    meta = '<meta http-equiv="refresh" content="10">' if refresh else ""
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)} — Protea Dashboard</title>{meta}"
        f"<style>{_BASE_CSS}</style></head><body>"
        f'<div class="header"><h1>Protea Dashboard</h1>{_NAV_HTML}</div>'
        f'<div class="container">{body}</div>'
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# SVG Charts
# ---------------------------------------------------------------------------

def _render_fitness_svg(history: list[dict], width: int = 800, height: int = 200) -> str:
    """Server-side SVG line chart for fitness history."""
    if not history:
        return '<p style="color:#777">No fitness data yet.</p>'

    margin = 40
    plot_w = width - margin * 2
    plot_h = height - margin * 2

    scores = [h.get("score", 0) for h in history]
    gens = [h.get("generation", i) for i, h in enumerate(history)]
    survived = [h.get("survived", False) for h in history]

    n = len(scores)
    if n == 1:
        xs = [margin + plot_w / 2]
    else:
        xs = [margin + i * plot_w / (n - 1) for i in range(n)]

    min_s, max_s = 0.0, 1.0
    ys = [margin + plot_h - (s - min_s) / (max_s - min_s) * plot_h for s in scores]

    # Build polyline
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="#0d1230" rx="8"/>',
        # Grid lines
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_h}" stroke="#1a1f3a" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin + plot_h}" x2="{margin + plot_w}" y2="{margin + plot_h}" stroke="#1a1f3a" stroke-width="1"/>',
    ]

    # Y-axis labels
    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin + plot_h - val * plot_h
        parts.append(f'<text x="{margin - 5}" y="{y + 4}" fill="#555" font-size="10" text-anchor="end">{val:.2f}</text>')
        parts.append(f'<line x1="{margin}" y1="{y}" x2="{margin + plot_w}" y2="{y}" stroke="#1a1f3a" stroke-width="0.5" stroke-dasharray="4"/>')

    # Line
    parts.append(f'<polyline points="{points}" fill="none" stroke="#667eea" stroke-width="2"/>')

    # Dots
    for i, (x, y) in enumerate(zip(xs, ys)):
        fill = "#667eea" if survived[i] else "#ff6b6b"
        r = 4
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#0d1230" stroke-width="1"/>')
        # Generation label (sparse to avoid clutter)
        if n <= 20 or i % max(1, n // 10) == 0:
            parts.append(f'<text x="{x:.1f}" y="{margin + plot_h + 15}" fill="#555" font-size="9" text-anchor="middle">G{gens[i]}</text>')

    # Legend
    lx = margin + plot_w - 140
    ly = margin + 8
    parts.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="#667eea"/>')
    parts.append(f'<text x="{lx + 8}" y="{ly + 4}" fill="#999" font-size="10">survived</text>')
    parts.append(f'<circle cx="{lx + 70}" cy="{ly}" r="4" fill="#ff6b6b"/>')
    parts.append(f'<text x="{lx + 78}" y="{ly + 4}" fill="#999" font-size="10">died</text>')

    parts.append("</svg>")
    return "".join(parts)


def _render_category_bars_svg(categories: dict[str, float], width: int = 400, height: int = 0) -> str:
    """Horizontal bar chart SVG for category distribution."""
    if not categories:
        return '<p style="color:#777">No profile data yet.</p>'

    total = sum(categories.values())
    if total == 0:
        return '<p style="color:#777">No profile data yet.</p>'

    bar_h = 24
    gap = 6
    label_w = 80
    value_w = 60
    bar_area = width - label_w - value_w - 20
    n = len(categories)
    if height <= 0:
        height = n * (bar_h + gap) + 10

    colors = ["#667eea", "#764ba2", "#4ecdc4", "#ff6b6b", "#feca57", "#48dbfb", "#a29bfe", "#fd79a8", "#00cec9"]

    parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

    for i, (cat, weight) in enumerate(categories.items()):
        pct = weight / total
        y = i * (bar_h + gap) + 5
        bar_w = max(2, pct * bar_area)
        color = colors[i % len(colors)]
        parts.append(f'<text x="{label_w - 5}" y="{y + bar_h / 2 + 4}" fill="#999" font-size="12" text-anchor="end">{_esc(cat)}</text>')
        parts.append(f'<rect x="{label_w}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="0.8"/>')
        parts.append(f'<text x="{label_w + bar_w + 5}" y="{y + bar_h / 2 + 4}" fill="#777" font-size="11">{pct:.0%}</text>')

    parts.append("</svg>")
    return "".join(parts)


def _compute_daily_skill_hit_ratio(memory_store, days: int = 14) -> list[dict]:
    """Compute daily skill hit ratio from task memory entries.

    Returns a list of {date, total, skill, ratio} dicts for the last *days* days.
    """
    from datetime import datetime, timedelta, timezone

    entries = []
    for entry_type in ("task", "p1_task"):
        try:
            entries.extend(memory_store.get_by_type(entry_type, 200))
        except Exception:
            pass

    # Bucket by date
    buckets: dict[str, dict] = {}
    for e in entries:
        ts = e.get("timestamp", "")
        if not ts:
            continue
        day = ts[:10]  # "YYYY-MM-DD"
        if day not in buckets:
            buckets[day] = {"total": 0, "skill": 0}
        buckets[day]["total"] += 1
        meta = e.get("metadata", {})
        if meta.get("skills_used"):
            buckets[day]["skill"] += 1

    # Fill in missing days
    today = datetime.now(timezone.utc).date()
    result = []
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        key = d.isoformat()
        b = buckets.get(key, {"total": 0, "skill": 0})
        ratio = b["skill"] / b["total"] if b["total"] > 0 else 0.0
        result.append({"date": key, "total": b["total"], "skill": b["skill"], "ratio": ratio})
    return result


def _render_skill_hit_ratio_svg(daily_data: list[dict], width: int = 800, height: int = 200) -> str:
    """Bar chart SVG for daily skill hit ratio."""
    if not daily_data or all(d["total"] == 0 for d in daily_data):
        return '<p style="color:#777">No task data yet.</p>'

    margin = 40
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    n = len(daily_data)
    bar_w = max(4, plot_w / n - 4)

    parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="#0d1230" rx="8"/>',
        # Axes
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_h}" stroke="#1a1f3a" stroke-width="1"/>',
        f'<line x1="{margin}" y1="{margin + plot_h}" x2="{margin + plot_w}" y2="{margin + plot_h}" stroke="#1a1f3a" stroke-width="1"/>',
    ]

    # Y-axis labels (0% - 100%)
    for pct in (0, 25, 50, 75, 100):
        y = margin + plot_h - pct / 100 * plot_h
        parts.append(f'<text x="{margin - 5}" y="{y + 4}" fill="#555" font-size="10" text-anchor="end">{pct}%</text>')
        parts.append(f'<line x1="{margin}" y1="{y}" x2="{margin + plot_w}" y2="{y}" stroke="#1a1f3a" stroke-width="0.5" stroke-dasharray="4"/>')

    # Bars
    for i, d in enumerate(daily_data):
        x = margin + i * (plot_w / n) + 2
        ratio = d["ratio"]
        bar_h = max(0, ratio * plot_h)
        y = margin + plot_h - bar_h
        if d["total"] > 0:
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2" fill="#667eea" opacity="0.8"/>')
            # Label: skill/total
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" fill="#999" font-size="9" text-anchor="middle">{d["skill"]}/{d["total"]}</text>')
        # X-axis: day number
        day_num = d["date"][8:10]  # DD
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{margin + plot_h + 14}" fill="#555" font-size="9" text-anchor="middle">{day_num}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the Protea Dashboard.

    Class attributes injected before starting:
        memory_store, skill_store, fitness_tracker, user_profiler,
        gene_pool, task_store, scheduled_store, state
    """

    memory_store = None
    skill_store = None
    fitness_tracker = None
    user_profiler = None
    gene_pool = None
    task_store = None
    scheduled_store = None
    state = None

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        raw_path = unquote(self.path)
        path = raw_path.split("?")[0].rstrip("/") or "/"
        query = {}
        if "?" in raw_path:
            query = parse_qs(raw_path.split("?", 1)[1])

        # Page routes
        if path == "/":
            self._serve_overview()
        elif path == "/memory":
            self._serve_memory(query)
        elif path == "/skills":
            self._serve_skills()
        elif path == "/genes":
            self._serve_genes()
        elif path == "/intent":
            self._serve_intent()
        elif path == "/profile":
            self._serve_profile()
        elif path == "/schedule":
            self._serve_schedule()
        # API routes
        elif path == "/api/memory":
            self._api_memory(query)
        elif path == "/api/memory/stats":
            self._api_memory_stats()
        elif path == "/api/skills":
            self._api_skills()
        elif path == "/api/genes":
            self._api_genes()
        elif path == "/api/intent":
            self._api_intent()
        elif path == "/api/profile":
            self._api_profile()
        elif path == "/api/fitness":
            self._api_fitness()
        elif path == "/api/status":
            self._api_status()
        elif path == "/api/schedules":
            self._api_schedules()
        elif path == "/api/skill_hit_ratio":
            self._api_skill_hit_ratio()
        else:
            self._send_error(404, "Not Found")

    # ------------------------------------------------------------------
    # Page handlers
    # ------------------------------------------------------------------

    def _serve_overview(self) -> None:
        """Overview page with stat cards + fitness chart."""
        # Memory stats
        mem_stats = {}
        if self.memory_store:
            try:
                mem_stats = self.memory_store.get_stats()
            except Exception:
                pass
        by_tier = mem_stats.get("by_tier", {})
        active_total = sum(v for k, v in by_tier.items() if k != "archive")
        mem_card = (
            f'<div class="card"><h3>Memory</h3>'
            f'<div class="value">{active_total}</div>'
            f'<div class="detail">hot: {by_tier.get("hot", 0)} warm: {by_tier.get("warm", 0)} cold: {by_tier.get("cold", 0)}</div>'
            f'</div>'
        )

        # Skills stats
        skill_count = 0
        if self.skill_store:
            try:
                skill_count = len(self.skill_store.get_active(100))
            except Exception:
                pass
        skill_card = (
            f'<div class="card"><h3>Skills</h3>'
            f'<div class="value">{skill_count}</div>'
            f'<div class="detail">active skills</div></div>'
        )

        # Intent
        intent_text = "—"
        if self.memory_store:
            try:
                intents = self.memory_store.get_by_type("evolution_intent", limit=1)
                if intents:
                    content = intents[0].get("content", "")
                    intent_text = content.split(":")[0] if ":" in content else content[:20]
            except Exception:
                pass
        intent_card = (
            f'<div class="card"><h3>Intent</h3>'
            f'<div class="value">{_esc(intent_text)}</div>'
            f'<div class="detail">current direction</div></div>'
        )

        # Profile
        top_cat = "—"
        if self.user_profiler:
            try:
                dist = self.user_profiler.get_category_distribution()
                if dist:
                    top_cat = next(iter(dist))
            except Exception:
                pass
        profile_card = (
            f'<div class="card"><h3>Profile</h3>'
            f'<div class="value">{_esc(top_cat)}</div>'
            f'<div class="detail">top interest</div></div>'
        )

        # Gene count
        gene_count = 0
        if self.gene_pool:
            try:
                gene_count = self.gene_pool.count()
            except Exception:
                pass
        gene_card = (
            f'<div class="card"><h3>Genes</h3>'
            f'<div class="value">{gene_count}</div>'
            f'<div class="detail">in gene pool</div></div>'
        )

        # Generation / Ring 2 status
        gen_num = 0
        r2_alive = False
        if self.state:
            try:
                snap = self.state.snapshot()
                gen_num = snap.get("generation", 0)
                r2_alive = snap.get("alive", False)
            except Exception:
                pass
        status_label = '<span style="color:#4ecdc4">alive</span>' if r2_alive else '<span style="color:#ff6b6b">dead</span>'
        gen_card = (
            f'<div class="card"><h3>Generation</h3>'
            f'<div class="value">{gen_num}</div>'
            f'<div class="detail">Ring 2: {status_label}</div></div>'
        )

        # Fitness chart
        fitness_svg = ""
        if self.fitness_tracker:
            try:
                history = self.fitness_tracker.get_history(limit=30)
                history.reverse()
                fitness_svg = _render_fitness_svg(history)
            except Exception:
                pass

        # Cooldown status card
        recent_ratio = 0.0
        if self.memory_store:
            try:
                from ring0.sentinel import _compute_skill_hit_ratio
                hit = _compute_skill_hit_ratio(self.memory_store)
                recent_ratio = hit["ratio"]
            except Exception:
                pass
        multiplier = 1.0 + 2.0 * recent_ratio
        cooldown_card = (
            f'<div class="card"><h3>Evo Cooldown</h3>'
            f'<div class="value">{multiplier:.1f}x</div>'
            f'<div class="detail">skill coverage: {recent_ratio:.0%}</div></div>'
        )

        # Skill hit ratio chart
        hit_ratio_svg = ""
        if self.memory_store:
            try:
                daily_data = _compute_daily_skill_hit_ratio(self.memory_store)
                hit_ratio_svg = _render_skill_hit_ratio_svg(daily_data)
            except Exception:
                pass

        body = (
            f'<div class="cards">{gen_card}{mem_card}{skill_card}{gene_card}{intent_card}{profile_card}{cooldown_card}</div>'
            f'<h2 style="margin-bottom:1rem">Fitness Trend</h2>'
            f'{fitness_svg}'
            f'<h2 style="margin:2rem 0 1rem">Skill Hit Ratio</h2>'
            f'{hit_ratio_svg}'
        )
        self._send_html(_page("Overview", body))

    def _serve_memory(self, query: dict) -> None:
        """Memory browser with tier/type filters."""
        tier_filter = query.get("tier", [None])[0]
        type_filter = query.get("type", [None])[0]

        # Build filter links
        tiers = ["all", "hot", "warm", "cold"]
        tier_links = []
        for t in tiers:
            active = "active" if (t == "all" and not tier_filter) or t == tier_filter else ""
            href = "/memory" if t == "all" else f"/memory?tier={t}"
            if type_filter and t != "all":
                href += f"&type={type_filter}"
            elif type_filter:
                href += f"?type={type_filter}"
            tier_links.append(f'<a href="{href}" class="{active}">{t}</a>')

        types = ["all", "task", "crash_log", "reflection", "observation", "directive", "evolution_intent", "p1_task"]
        type_links = []
        for tp in types:
            active = "active" if (tp == "all" and not type_filter) or tp == type_filter else ""
            base = f"/memory?tier={tier_filter}" if tier_filter else "/memory"
            href = base if tp == "all" else f"{base}{'&' if tier_filter else '?'}type={tp}"
            type_links.append(f'<a href="{href}" class="{active}">{tp}</a>')

        filters_html = (
            f'<div class="filters">Tier: {" ".join(tier_links)}</div>'
            f'<div class="filters">Type: {" ".join(type_links)}</div>'
        )

        # Fetch entries
        entries = []
        if self.memory_store:
            try:
                if tier_filter and tier_filter != "all":
                    entries = self.memory_store.get_by_tier(tier_filter, limit=100)
                else:
                    entries = self.memory_store.get_recent(limit=100)
                if type_filter and type_filter != "all":
                    entries = [e for e in entries if e.get("entry_type") == type_filter]
            except Exception:
                pass

        # Build table
        rows_html = []
        for e in entries:
            tier = e.get("tier", "hot")
            tier_cls = f"tier-{tier}"
            importance = e.get("importance", 0.5)
            bar_w = int(importance * 80)
            content = _esc(e.get("content", "")[:100])
            ts = e.get("timestamp", "")[:16]
            rows_html.append(
                f'<tr>'
                f'<td>{e.get("id", "")}</td>'
                f'<td>{e.get("generation", "")}</td>'
                f'<td>{_esc(e.get("entry_type", ""))}</td>'
                f'<td class="{tier_cls}">{tier}</td>'
                f'<td><span class="importance-bar" style="width:{bar_w}px"></span> {importance:.2f}</td>'
                f'<td>{content}</td>'
                f'<td>{ts}</td>'
                f'</tr>'
            )

        table_html = (
            "<table><thead><tr>"
            "<th>ID</th><th>Gen</th><th>Type</th><th>Tier</th><th>Importance</th><th>Content</th><th>Time</th>"
            "</tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table>"
        )

        body = f"<h2>Memory Browser</h2>{filters_html}{table_html}"
        self._send_html(_page("Memory", body))

    def _serve_skills(self) -> None:
        """Skill gallery with cards."""
        skills = []
        if self.skill_store:
            try:
                skills = self.skill_store.get_active(50)
            except Exception:
                pass

        cards = []
        for s in skills:
            tags_html = "".join(f'<span class="tag">{_esc(t)}</span>' for t in (s.get("tags") or []))
            name = _esc(s.get("name", "unknown"))
            desc = _esc((s.get("description", ""))[:120])
            usage = s.get("usage_count", 0)
            source = _esc(s.get("source", ""))
            # Permanent badge
            perm_badge = ' <span class="badge" style="background:#4ecdc4;color:#000">permanent</span>' if s.get("permanent") else ""
            # Dependencies
            deps = s.get("dependencies") or []
            deps_html = ""
            if deps:
                deps_html = f'<div class="meta" style="margin-top:0.3rem">deps: {_esc(", ".join(deps))}</div>'
            cards.append(
                f'<div class="skill-card">'
                f'<h3>{name}{perm_badge}</h3>'
                f'<div class="desc">{desc}</div>'
                f'<div class="tags">{tags_html}</div>'
                f'<div class="meta">usage: {usage} | source: {source}</div>'
                f'{deps_html}'
                f'</div>'
            )

        # Leaderboard table — top skills by usage
        top_skills = [s for s in skills if s.get("usage_count", 0) > 0][:15]
        if top_skills:
            lb_rows = []
            for rank, s in enumerate(top_skills, 1):
                name = _esc(s.get("name", "unknown"))
                usage = s.get("usage_count", 0)
                last_used = _esc(str(s.get("last_used", ""))[:16])
                source = _esc(s.get("source", ""))
                perm = "yes" if s.get("permanent") else "no"
                lb_rows.append(
                    f'<tr><td>{rank}</td><td>{name}</td><td>{usage}</td>'
                    f'<td>{last_used}</td><td>{source}</td><td>{perm}</td></tr>'
                )
            leaderboard_html = (
                '<h2 style="margin-bottom:1rem">Top Skills by Usage</h2>'
                "<table><thead><tr>"
                "<th>#</th><th>Name</th><th>Usage</th><th>Last Used</th><th>Source</th><th>Permanent</th>"
                "</tr></thead><tbody>"
                + "".join(lb_rows)
                + "</tbody></table>"
                + '<div style="margin-bottom:2rem"></div>'
            )
        else:
            leaderboard_html = ""

        body_content = '<div class="grid">' + "".join(cards) + "</div>" if cards else '<p style="color:#777">No skills registered yet.</p>'
        body = f"{leaderboard_html}<h2>Skills Gallery</h2>{body_content}"
        self._send_html(_page("Skills", body))

    def _serve_intent(self) -> None:
        """Evolution intent timeline."""
        intents = []
        if self.memory_store:
            try:
                intents = self.memory_store.get_by_type("evolution_intent", limit=30)
            except Exception:
                pass

        items = []
        for entry in intents:
            gen = entry.get("generation", "?")
            content = entry.get("content", "")
            meta = entry.get("metadata", {})
            blast = meta.get("blast_radius", {})
            scope = blast.get("scope", "")
            lines = blast.get("lines_changed", 0)

            # Parse intent from content (format: "INTENT: signal1, signal2")
            intent_name = content.split(":")[0].strip() if ":" in content else content[:30]
            signals = content.split(":", 1)[1].strip() if ":" in content else ""
            intent_cls = f"intent-{intent_name.lower()}" if intent_name.lower() in ("repair", "optimize", "explore", "adapt") else ""

            items.append(
                f'<div class="timeline-item {intent_cls}">'
                f'<div class="gen">Gen {gen}: {_esc(intent_name.upper())}</div>'
                f'<div class="info">signals: {_esc(signals)}</div>'
                + (f'<div class="info">scope: {_esc(scope)} ({lines} lines)</div>' if scope else "")
                + f'</div>'
            )

        timeline = '<div class="timeline">' + "".join(items) + "</div>" if items else '<p style="color:#777">No evolution intents yet.</p>'
        body = f"<h2>Intent Timeline</h2>{timeline}"
        self._send_html(_page("Intent", body))

    def _serve_profile(self) -> None:
        """User profile page with category chart and stats."""
        categories: dict[str, float] = {}
        stats: dict = {}
        top_topics: list[dict] = []

        if self.user_profiler:
            try:
                categories = self.user_profiler.get_category_distribution()
                stats = self.user_profiler.get_stats()
                top_topics = self.user_profiler.get_top_topics(10)
            except Exception:
                pass

        bars_svg = _render_category_bars_svg(categories, width=450)

        # Stats panel
        stat_items = []
        if stats:
            stat_items.append(f'<div class="stat-item"><span class="label">Total interactions:</span> <span class="val">{stats.get("interaction_count", 0)}</span></div>')
            if stats.get("earliest_interaction"):
                stat_items.append(f'<div class="stat-item"><span class="label">First seen:</span> <span class="val">{stats["earliest_interaction"][:10]}</span></div>')
            if stats.get("latest_interaction"):
                stat_items.append(f'<div class="stat-item"><span class="label">Last seen:</span> <span class="val">{stats["latest_interaction"][:10]}</span></div>')
            stat_items.append(f'<div class="stat-item"><span class="label">Active topics:</span> <span class="val">{stats.get("topic_count", 0)}</span></div>')

        # Top topics
        if top_topics:
            stat_items.append('<h3 style="margin-top:1rem;margin-bottom:0.5rem">Top Topics</h3>')
            for t in top_topics:
                stat_items.append(f'<div class="stat-item">{_esc(t["topic"])} <span class="label">({t["category"]}, weight: {t["weight"]:.1f})</span></div>')

        stats_html = "".join(stat_items) if stat_items else '<p style="color:#777">No profile data yet.</p>'

        body = (
            f'<h2>User Profile</h2>'
            f'<div class="profile-grid">'
            f'<div><h3 style="margin-bottom:1rem">Interest Distribution</h3>{bars_svg}</div>'
            f'<div><h3 style="margin-bottom:1rem">Statistics</h3>{stats_html}</div>'
            f'</div>'
        )
        self._send_html(_page("Profile", body))

    def _serve_schedule(self) -> None:
        """Scheduled tasks page with table."""
        tasks: list[dict] = []
        if self.scheduled_store:
            try:
                tasks = self.scheduled_store.get_all()
            except Exception:
                pass

        try:
            from ring0.cron import describe as _cron_desc
        except ImportError:
            _cron_desc = None

        rows_html = []
        for t in tasks:
            enabled = t.get("enabled", 0)
            enabled_html = '<span style="color:#4ecdc4">enabled</span>' if enabled else '<span style="color:#777">disabled</span>'
            name = _esc(t.get("name", ""))
            stype = _esc(t.get("schedule_type", "cron"))
            cron_raw = _esc(t.get("cron_expr", ""))

            # Human-readable schedule description
            if t.get("schedule_type") == "cron" and _cron_desc:
                schedule_desc = _esc(_cron_desc(t["cron_expr"]))
            else:
                schedule_desc = cron_raw

            last_run = ""
            if t.get("last_run_at"):
                from datetime import datetime
                last_run = datetime.fromtimestamp(t["last_run_at"]).strftime("%Y-%m-%d %H:%M")
            next_run = ""
            if t.get("next_run_at"):
                from datetime import datetime
                next_run = datetime.fromtimestamp(t["next_run_at"]).strftime("%Y-%m-%d %H:%M")

            run_count = t.get("run_count", 0)
            task_text = _esc((t.get("task_text", ""))[:60])

            rows_html.append(
                f'<tr>'
                f'<td><strong>{name}</strong></td>'
                f'<td>{schedule_desc}<br><span style="color:#555;font-size:0.75rem">{cron_raw}</span></td>'
                f'<td>{stype}</td>'
                f'<td>{enabled_html}</td>'
                f'<td>{task_text}</td>'
                f'<td>{last_run or "—"}</td>'
                f'<td>{next_run or "—"}</td>'
                f'<td>{run_count}</td>'
                f'</tr>'
            )

        table_html = (
            "<table><thead><tr>"
            "<th>Name</th><th>Schedule</th><th>Type</th><th>Enabled</th>"
            "<th>Task</th><th>Last Run</th><th>Next Run</th><th>Runs</th>"
            "</tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table>"
        )

        count = len(tasks)
        body = f"<h2>Scheduled Tasks ({count})</h2>{table_html}"
        self._send_html(_page("Schedule", body))

    def _serve_genes(self) -> None:
        """Gene leaderboard page — top genes by score."""
        genes = []
        if self.gene_pool:
            try:
                genes = self.gene_pool.get_top(0)
            except Exception:
                pass

        rows_html = []
        for i, g in enumerate(genes, 1):
            score = g.get("score", 0)
            gen = g.get("generation", "?")
            summary = _esc(str(g.get("gene_summary", ""))[:80])
            tags = _esc(str(g.get("tags", ""))[:60])
            hits = g.get("hit_count", 0)
            last_hit = g.get("last_hit_gen", 0)
            rows_html.append(
                f'<tr><td>{i}</td><td>{score:.2f}</td><td>{gen}</td>'
                f'<td>{summary}</td><td>{tags}</td>'
                f'<td>{hits}</td><td>{last_hit}</td></tr>'
            )

        table = (
            "<table><thead><tr>"
            "<th>#</th><th>Score</th><th>Gen</th><th>Summary</th><th>Tags</th>"
            "<th>Hits</th><th>Last Hit</th>"
            "</tr></thead><tbody>"
            + "".join(rows_html)
            + "</tbody></table>"
        ) if rows_html else '<p style="color:#777">No genes in pool yet.</p>'

        body = f"<h2>Gene Leaderboard</h2>{table}"
        self._send_html(_page("Genes", body))

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------

    def _api_memory(self, query: dict) -> None:
        tier = query.get("tier", [None])[0]
        entry_type = query.get("type", [None])[0]
        entries = []
        if self.memory_store:
            try:
                if tier:
                    entries = self.memory_store.get_by_tier(tier, limit=100)
                else:
                    entries = self.memory_store.get_recent(limit=100)
                if entry_type:
                    entries = [e for e in entries if e.get("entry_type") == entry_type]
            except Exception:
                pass
        self._send_json(entries)

    def _api_memory_stats(self) -> None:
        stats = {}
        if self.memory_store:
            try:
                stats = self.memory_store.get_stats()
            except Exception:
                pass
        self._send_json(stats)

    def _api_genes(self) -> None:
        genes = []
        if self.gene_pool:
            try:
                genes = self.gene_pool.get_top(0)
            except Exception:
                pass
        self._send_json(genes)

    def _api_skills(self) -> None:
        skills = []
        if self.skill_store:
            try:
                skills = self.skill_store.get_active(50)
            except Exception:
                pass
        self._send_json(skills)

    def _api_intent(self) -> None:
        intents = []
        if self.memory_store:
            try:
                intents = self.memory_store.get_by_type("evolution_intent", limit=30)
            except Exception:
                pass
        self._send_json(intents)

    def _api_profile(self) -> None:
        data: dict = {"categories": {}, "stats": {}, "topics": []}
        if self.user_profiler:
            try:
                data["categories"] = self.user_profiler.get_category_distribution()
                data["stats"] = self.user_profiler.get_stats()
                data["topics"] = self.user_profiler.get_top_topics(20)
            except Exception:
                pass
        self._send_json(data)

    def _api_schedules(self) -> None:
        tasks = []
        if self.scheduled_store:
            try:
                tasks = self.scheduled_store.get_all()
            except Exception:
                pass
        self._send_json(tasks)

    def _api_skill_hit_ratio(self) -> None:
        data = []
        if self.memory_store:
            try:
                data = _compute_daily_skill_hit_ratio(self.memory_store)
            except Exception:
                pass
        self._send_json(data)

    def _api_fitness(self) -> None:
        history = []
        if self.fitness_tracker:
            try:
                history = self.fitness_tracker.get_history(limit=50)
            except Exception:
                pass
        self._send_json(history)

    def _api_status(self) -> None:
        gen = 0
        alive = False
        if self.state:
            try:
                snap = self.state.snapshot()
                gen = snap.get("generation", 0)
                alive = snap.get("alive", False)
            except Exception:
                pass
        self._send_json({
            "dashboard": "running",
            "generation": gen,
            "alive": alive,
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _send_html(self, html_str: str) -> None:
        data = html_str.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: object) -> None:
        data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        body = _page("Error", f'<h2>{code}</h2><p style="color:#ff6b6b">{_esc(message)}</p>', refresh=False)
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Dashboard server
# ---------------------------------------------------------------------------

class Dashboard:
    """Persistent web dashboard for system state visualization."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8899,
        **data_sources,
    ) -> None:
        self._host = host
        self._port = port
        self._data_sources = data_sources
        self._server: ThreadingHTTPServer | None = None

    def run(self) -> None:
        """Start the HTTP server (blocking — call from a thread)."""
        handler = type(
            "InjectedDashboardHandler",
            (DashboardHandler,),
            self._data_sources,
        )
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        log.info("Dashboard listening on %s:%d", self._host, self.actual_port)
        self._server.serve_forever()

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server:
            self._server.shutdown()
            log.info("Dashboard stopped")

    @property
    def actual_port(self) -> int:
        if self._server:
            return self._server.server_address[1]
        return self._port


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_dashboard(project_root: pathlib.Path, config: dict, **data_sources) -> Dashboard | None:
    """Create a Dashboard from configuration.  Returns None if disabled."""
    dash_cfg = config.get("ring1", {}).get("dashboard", {})
    if not dash_cfg.get("enabled", False):
        return None
    host = dash_cfg.get("host", "127.0.0.1")
    port = dash_cfg.get("port", 8899)
    return Dashboard(host=host, port=port, **data_sources)


def start_dashboard_thread(dashboard: Dashboard) -> threading.Thread:
    """Start the dashboard in a daemon thread and return the thread."""
    t = threading.Thread(target=dashboard.run, name="dashboard", daemon=True)
    t.start()
    return t
