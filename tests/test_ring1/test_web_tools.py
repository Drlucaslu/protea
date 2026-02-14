"""Tests for ring1.web_tools."""

from __future__ import annotations

import io
import urllib.error

import pytest

from ring1.web_tools import (
    _DDGResultParser,
    _TextExtractor,
    web_fetch,
    web_search,
)


# ---------------------------------------------------------------------------
# _DDGResultParser
# ---------------------------------------------------------------------------

class TestDDGResultParser:
    def test_parses_single_result(self):
        html = (
            '<a class="result__a" href="https://example.com">Example Title</a>'
            '<a class="result__snippet">A short snippet.</a>'
        )
        parser = _DDGResultParser()
        parser.feed(html)
        assert len(parser.results) == 1
        assert parser.results[0]["title"] == "Example Title"
        assert parser.results[0]["url"] == "https://example.com"
        assert parser.results[0]["snippet"] == "A short snippet."

    def test_parses_multiple_results(self):
        html = (
            '<a class="result__a" href="https://a.com">A</a>'
            '<a class="result__snippet">Snippet A</a>'
            '<a class="result__a" href="https://b.com">B</a>'
            '<a class="result__snippet">Snippet B</a>'
        )
        parser = _DDGResultParser()
        parser.feed(html)
        assert len(parser.results) == 2
        assert parser.results[0]["url"] == "https://a.com"
        assert parser.results[1]["url"] == "https://b.com"

    def test_extracts_uddg_url(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.com&rut=abc"
        html = (
            f'<a class="result__a" href="{href}">Title</a>'
            '<a class="result__snippet">Snip</a>'
        )
        parser = _DDGResultParser()
        parser.feed(html)
        assert parser.results[0]["url"] == "https://real.com"

    def test_empty_html(self):
        parser = _DDGResultParser()
        parser.feed("<html><body>No results</body></html>")
        assert parser.results == []

    def test_nested_tags_in_title(self):
        html = (
            '<a class="result__a" href="https://x.com"><b>Bold</b> Title</a>'
            '<a class="result__snippet">Snip</a>'
        )
        parser = _DDGResultParser()
        parser.feed(html)
        assert parser.results[0]["title"] == "Bold Title"


# ---------------------------------------------------------------------------
# _TextExtractor
# ---------------------------------------------------------------------------

class TestTextExtractor:
    def test_extracts_visible_text(self):
        html = "<html><body><p>Hello world</p></body></html>"
        ext = _TextExtractor()
        ext.feed(html)
        assert "Hello world" in ext.get_text()

    def test_skips_script(self):
        html = "<p>Before</p><script>var x=1;</script><p>After</p>"
        ext = _TextExtractor()
        ext.feed(html)
        text = ext.get_text()
        assert "Before" in text
        assert "After" in text
        assert "var x" not in text

    def test_skips_style(self):
        html = "<p>Text</p><style>body{color:red}</style>"
        ext = _TextExtractor()
        ext.feed(html)
        text = ext.get_text()
        assert "Text" in text
        assert "color" not in text

    def test_skips_noscript(self):
        html = "<p>Yes</p><noscript>Enable JS</noscript>"
        ext = _TextExtractor()
        ext.feed(html)
        text = ext.get_text()
        assert "Yes" in text
        assert "Enable JS" not in text

    def test_nested_skip_tags(self):
        html = "<script>outer<script>inner</script>still skip</script><p>Visible</p>"
        ext = _TextExtractor()
        ext.feed(html)
        text = ext.get_text()
        assert "Visible" in text
        assert "outer" not in text
        assert "inner" not in text

    def test_collapses_whitespace(self):
        html = "<p>  Hello   \n  world  </p>"
        ext = _TextExtractor()
        ext.feed(html)
        assert ext.get_text() == "Hello world"


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_success(self, monkeypatch):
        fake_html = (
            '<a class="result__a" href="https://example.com">Example</a>'
            '<a class="result__snippet">A snippet</a>'
            '<a class="result__a" href="https://other.com">Other</a>'
            '<a class="result__snippet">Another snippet</a>'
        )

        def fake_urlopen(req, timeout=None):
            return io.BytesIO(fake_html.encode("utf-8"))

        monkeypatch.setattr("ring1.web_tools.urllib.request.urlopen", fake_urlopen)
        results = web_search("test query")
        assert len(results) == 2
        assert results[0]["title"] == "Example"

    def test_max_results(self, monkeypatch):
        items = ""
        for i in range(10):
            items += (
                f'<a class="result__a" href="https://r{i}.com">R{i}</a>'
                f'<a class="result__snippet">S{i}</a>'
            )

        def fake_urlopen(req, timeout=None):
            return io.BytesIO(items.encode("utf-8"))

        monkeypatch.setattr("ring1.web_tools.urllib.request.urlopen", fake_urlopen)
        results = web_search("test", max_results=3)
        assert len(results) == 3

    def test_network_error_returns_empty(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("no network")

        monkeypatch.setattr("ring1.web_tools.urllib.request.urlopen", fake_urlopen)
        results = web_search("test")
        assert results == []


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------

class TestWebFetch:
    def test_success(self, monkeypatch):
        fake_html = "<html><body><p>Hello world</p></body></html>"

        def fake_urlopen(req, timeout=None):
            buf = io.BytesIO(fake_html.encode("utf-8"))
            buf.read_orig = buf.read
            return buf

        monkeypatch.setattr("ring1.web_tools.urllib.request.urlopen", fake_urlopen)
        text = web_fetch("https://example.com")
        assert "Hello world" in text

    def test_truncates_to_max_chars(self, monkeypatch):
        long_text = "A" * 10000
        fake_html = f"<html><body><p>{long_text}</p></body></html>"

        def fake_urlopen(req, timeout=None):
            return io.BytesIO(fake_html.encode("utf-8"))

        monkeypatch.setattr("ring1.web_tools.urllib.request.urlopen", fake_urlopen)
        text = web_fetch("https://example.com", max_chars=100)
        assert len(text) == 103  # 100 + "..."
        assert text.endswith("...")

    def test_strips_script_tags(self, monkeypatch):
        fake_html = "<p>Visible</p><script>alert('xss')</script>"

        def fake_urlopen(req, timeout=None):
            return io.BytesIO(fake_html.encode("utf-8"))

        monkeypatch.setattr("ring1.web_tools.urllib.request.urlopen", fake_urlopen)
        text = web_fetch("https://example.com")
        assert "Visible" in text
        assert "alert" not in text

    def test_network_error_returns_error_string(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("ring1.web_tools.urllib.request.urlopen", fake_urlopen)
        text = web_fetch("https://bad.example.com")
        assert text.startswith("Error fetching https://bad.example.com:")
