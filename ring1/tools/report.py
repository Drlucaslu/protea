"""Report tool — generate Chinese-friendly PDF from Markdown.

Wraps ring1.pdf_utils.markdown_to_pdf as a Tool for LLM tool_use.
"""

from __future__ import annotations

import pathlib
import shutil

from ring1.tool_registry import Tool


def make_report_tool(workspace_path: str) -> Tool:
    """Create a Tool instance for PDF report generation."""

    ws = pathlib.Path(workspace_path)

    def _exec_generate_pdf(inp: dict) -> str:
        from ring1.pdf_utils import markdown_to_pdf

        md_path = inp["markdown_path"]
        out_path = inp.get("output_path", "")

        md_full = (ws / md_path) if not pathlib.Path(md_path).is_absolute() else pathlib.Path(md_path)

        if not md_full.is_file():
            return f"Error: markdown file not found: {md_full}"

        if out_path:
            pdf_full = (ws / out_path) if not pathlib.Path(out_path).is_absolute() else pathlib.Path(out_path)
        else:
            pdf_full = md_full.with_suffix(".pdf")

        result = markdown_to_pdf(str(md_full), str(pdf_full))
        if result.startswith("Error:"):
            return result

        # Copy to output/reports/ directory if not already there
        reports_dir = ws / "output" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        dest = reports_dir / pdf_full.name
        if pdf_full.resolve() != dest.resolve():
            shutil.copy2(str(pdf_full), str(dest))
            return f"PDF generated: {result}\nCopied to: {dest}"
        return f"PDF generated: {result}"

    return Tool(
        name="generate_pdf",
        description=(
            "Generate a PDF report from a Markdown file.  Supports Chinese "
            "and other CJK characters.  The PDF is also copied to the "
            "output/reports/ directory for web portal access."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "markdown_path": {
                    "type": "string",
                    "description": "Path to the input Markdown file (relative to workspace or absolute).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path for the output PDF file (optional; defaults to same name with .pdf extension).",
                },
            },
            "required": ["markdown_path"],
        },
        execute=_exec_generate_pdf,
    )
