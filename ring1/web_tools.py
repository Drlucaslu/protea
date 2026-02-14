"""Web tools — pure stdlib web search and fetch for LLM tool_use.

Provides web_search (DuckDuckGo HTML) and web_fetch (URL content extraction)
using only urllib.request and html.parser.  Neither function raises exceptions;
errors are returned as empty results or error strings.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

log = logging.getLogger("protea.web_tools")

_TIMEOUT = 10  # seconds
_MAX_FETCH_BYTES = 100_000  # 100 KB read limit for web_fetch
_DDG_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = "Mozilla/5.0 (compatible; Protea/1.0)"


# ---------------------------------------------------------------------------
# DuckDuckGo HTML result parser
# ---------------------------------------------------------------------------

class _DDGResultParser(HTMLParser):
    """Parse DuckDuckGo HTML search results.

    Extracts titles + URLs from <a class="result__a"> and snippets from
    <a class="result__snippet">.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._in_title = False
        self._in_snippet = False
        self._current: dict = {}
        self._text_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "")
        if "result__a" in cls:
            self._in_title = True
            self._text_buf = []
            href = attr_dict.get("href", "")
            # DDG wraps URLs in a redirect; extract the real URL.
            if "uddg=" in href:
                try:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    href = parsed.get("uddg", [href])[0]
                except Exception:
                    pass
            self._current = {"title": "", "url": href, "snippet": ""}
        elif "result__snippet" in cls:
            self._in_snippet = True
            self._text_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a":
            return
        if self._in_title:
            self._in_title = False
            self._current["title"] = "".join(self._text_buf).strip()
        elif self._in_snippet:
            self._in_snippet = False
            self._current["snippet"] = "".join(self._text_buf).strip()
            self.results.append(self._current)
            self._current = {}

    def handle_data(self, data: str) -> None:
        if self._in_title or self._in_snippet:
            self._text_buf.append(data)


# ---------------------------------------------------------------------------
# HTML text extractor
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style/noscript."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join("".join(self.parts).split())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return up to *max_results* result dicts.

    Each dict: {"title": str, "url": str, "snippet": str}.
    Returns [] on any error.
    """
    try:
        form_data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            _DDG_URL,
            data=form_data,
            headers={"User-Agent": _USER_AGENT},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        parser = _DDGResultParser()
        parser.feed(html)
        return parser.results[:max_results]
    except Exception as exc:
        log.warning("web_search failed: %s", exc)
        return []


def web_fetch(url: str, max_chars: int = 5000) -> str:
    """Fetch a URL and return extracted plain text (up to *max_chars*).

    Returns an error string on failure (never raises).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read(_MAX_FETCH_BYTES)
        html = raw.decode("utf-8", errors="replace")
        extractor = _TextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text
    except Exception as exc:
        log.warning("web_fetch failed for %s: %s", url, exc)
        return f"Error fetching {url}: {exc}"
