"""Tests for ring1.tools.send_file."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, ANY

from ring1.tools.send_file import make_send_file_tool


class TestSendFileTool:
    """Test send_file tool behaviour."""

    def setup_method(self):
        # Clean up thread context to avoid test pollution.
        for attr in ("task_chat_id", "reply_to_message_id"):
            if hasattr(threading.current_thread(), attr):
                delattr(threading.current_thread(), attr)

    def test_send_success(self, tmp_path):
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        send_fn = MagicMock(return_value=True)
        tool = make_send_file_tool(send_fn)

        result = tool.execute({"file_path": str(f)})
        assert "File sent" in result
        send_fn.assert_called_once_with(str(f), "", "")

    def test_send_with_caption(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        send_fn = MagicMock(return_value=True)
        tool = make_send_file_tool(send_fn)

        result = tool.execute({"file_path": str(f), "caption": "Your data"})
        assert "File sent" in result
        send_fn.assert_called_once_with(str(f), "Your data", "")

    def test_file_not_found(self):
        send_fn = MagicMock()
        tool = make_send_file_tool(send_fn)

        result = tool.execute({"file_path": "/nonexistent/file.txt"})
        assert "Error" in result
        assert "not found" in result
        send_fn.assert_not_called()

    def test_send_returns_false(self, tmp_path):
        f = tmp_path / "fail.txt"
        f.write_text("content")
        send_fn = MagicMock(return_value=False)
        tool = make_send_file_tool(send_fn)

        result = tool.execute({"file_path": str(f)})
        assert "Error" in result
        assert "failed to send" in result

    def test_send_raises_exception(self, tmp_path):
        f = tmp_path / "err.txt"
        f.write_text("content")
        send_fn = MagicMock(side_effect=RuntimeError("network down"))
        tool = make_send_file_tool(send_fn)

        result = tool.execute({"file_path": str(f)})
        assert "Error" in result
        assert "network down" in result

    def test_file_too_large(self, tmp_path, monkeypatch):
        f = tmp_path / "huge.bin"
        f.write_bytes(b"x")
        # Mock stat to report >50MB
        import ring1.tools.send_file as mod
        monkeypatch.setattr(mod, "_MAX_FILE_SIZE", 10)
        send_fn = MagicMock()
        tool = make_send_file_tool(send_fn)

        result = tool.execute({"file_path": str(f)})
        # File is 1 byte but limit is 10 — should succeed
        # Now test with actual large size
        f2 = tmp_path / "big.bin"
        f2.write_bytes(b"x" * 20)
        result2 = tool.execute({"file_path": str(f2)})
        assert "too large" in result2
        send_fn.assert_called_once()  # only the first call

    def test_resolves_output_prefix(self, tmp_path, monkeypatch):
        """If file_path is relative and not found, try output/ prefix."""
        import os
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        f = out_dir / "report.pdf"
        f.write_bytes(b"data")

        send_fn = MagicMock(return_value=True)
        tool = make_send_file_tool(send_fn)

        result = tool.execute({"file_path": "report.pdf"})
        assert "File sent" in result
        # Should have resolved to output/report.pdf
        call_args = send_fn.call_args[0]
        assert "output" in call_args[0]

    def test_schema_structure(self):
        send_fn = MagicMock()
        tool = make_send_file_tool(send_fn)
        assert tool.name == "send_file"
        assert "file_path" in tool.input_schema["properties"]
        assert "caption" in tool.input_schema["properties"]
        assert tool.input_schema["required"] == ["file_path"]

    def test_caption_defaults_to_empty(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        send_fn = MagicMock(return_value=True)
        tool = make_send_file_tool(send_fn)

        tool.execute({"file_path": str(f)})
        send_fn.assert_called_once_with(str(f), "", "")
