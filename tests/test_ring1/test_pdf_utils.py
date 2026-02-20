"""Tests for ring1.pdf_utils — Chinese PDF generation."""

from __future__ import annotations

import pathlib

import pytest

reportlab = pytest.importorskip("reportlab", reason="reportlab not installed")

from ring1.pdf_utils import markdown_to_pdf, _ensure_font


class TestEnsureFont:
    def test_returns_string(self):
        name = _ensure_font()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_idempotent(self):
        name1 = _ensure_font()
        name2 = _ensure_font()
        assert name1 == name2


class TestMarkdownToPdf:
    def test_basic_conversion(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nThis is a test.", encoding="utf-8")
        pdf = tmp_path / "test.pdf"

        result = markdown_to_pdf(md, pdf)
        assert not result.startswith("Error:")
        assert pdf.is_file()
        assert pdf.stat().st_size > 0
        # PDF should start with %PDF header
        assert pdf.read_bytes()[:5] == b"%PDF-"

    def test_chinese_content(self, tmp_path):
        md = tmp_path / "chinese.md"
        md.write_text(
            "# 中文标题\n\n这是一个包含中文的测试文档。\n\n"
            "## 第二级标题\n\n- 列表项一\n- 列表项二\n",
            encoding="utf-8",
        )
        pdf = tmp_path / "chinese.pdf"

        result = markdown_to_pdf(md, pdf)
        assert not result.startswith("Error:")
        assert pdf.is_file()
        assert pdf.stat().st_size > 100

    def test_headings_and_lists(self, tmp_path):
        md = tmp_path / "elements.md"
        md.write_text(
            "# H1 Title\n\n## H2 Title\n\n### H3 Title\n\n"
            "Paragraph text here.\n\n"
            "- Item A\n- Item B\n- Item C\n\n"
            "1. First\n2. Second\n",
            encoding="utf-8",
        )
        pdf = tmp_path / "elements.pdf"

        result = markdown_to_pdf(md, pdf)
        assert not result.startswith("Error:")
        assert pdf.is_file()

    def test_missing_input(self, tmp_path):
        result = markdown_to_pdf(tmp_path / "nonexistent.md", tmp_path / "out.pdf")
        assert result.startswith("Error:")

    def test_creates_output_dir(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Test\n\nContent.", encoding="utf-8")
        pdf = tmp_path / "subdir" / "nested" / "out.pdf"

        result = markdown_to_pdf(md, pdf)
        assert not result.startswith("Error:")
        assert pdf.is_file()

    def test_empty_markdown(self, tmp_path):
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        pdf = tmp_path / "empty.pdf"

        result = markdown_to_pdf(md, pdf)
        assert not result.startswith("Error:")
        assert pdf.is_file()

    def test_returns_absolute_path(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Test", encoding="utf-8")
        pdf = tmp_path / "test.pdf"

        result = markdown_to_pdf(md, pdf)
        assert pathlib.Path(result).is_absolute()
