"""Tests for _detect_fabrication() and related task_executor functions."""

from ring1.task_executor import _detect_fabrication, TaskExecutor


class TestDetectFabrication:
    """Unit tests for fabrication detection."""

    def test_clean_response_no_tools(self):
        """Plain text response with no tool claims → no signals."""
        resp = "这是一段普通的回复，没有任何工具调用。"
        assert _detect_fabrication(resp, []) == []

    def test_clean_response_with_tools(self):
        """Normal response with actual tool usage → no signals."""
        resp = "I searched the web and found the latest prices."
        assert _detect_fabrication(resp, ["web_search", "web_fetch"]) == []

    def test_claimed_tool_not_called(self):
        """LLM claims '已调用 send_file(...)' but never called it."""
        resp = '已调用 send_file("report.pdf") 发送给用户。'
        signals = _detect_fabrication(resp, [])
        assert any("tool_call_claim:send_file" in s for s in signals)

    def test_claimed_tool_actually_called(self):
        """LLM claims '已调用 web_fetch(...)' and actually called it → no signal."""
        resp = '已调用 web_fetch("https://example.com") 获取数据。'
        assert _detect_fabrication(resp, ["web_fetch"]) == []

    def test_fake_api_data_no_tools(self):
        """Fake JSON data embedded in text with no data tools → signal."""
        resp = '{"binance": 42000, "coingecko": 41900}'
        signals = _detect_fabrication(resp, [])
        assert "fake_api_data" in signals

    def test_fake_api_data_with_data_tool(self):
        """JSON data in text but web_search was called → no signal."""
        resp = '{"binance": 42000, "coingecko": 41900}'
        assert _detect_fabrication(resp, ["web_search"]) == []

    def test_fake_phase_no_message_tool(self):
        """'Phase 1 Complete' without message tool → signal."""
        resp = "✅ Phase 1 Complete: Found 10 sources\n🔄 Phase 2: Analyzing..."
        signals = _detect_fabrication(resp, [])
        assert "fake_phase" in signals

    def test_fake_phase_with_message_tool(self):
        """'Phase 1 Complete' with message tool → no signal."""
        resp = "✅ Phase 1 Complete: Found 10 sources"
        assert _detect_fabrication(resp, ["message", "web_search"]) == []

    def test_fake_delivery_no_send_file(self):
        """'报告已发送给用户' without send_file → signal."""
        resp = "报告已发送给用户，请查收。"
        signals = _detect_fabrication(resp, [])
        assert "fake_delivery" in signals

    def test_fake_delivery_with_send_file(self):
        """'文件已发送' with send_file → no signal."""
        resp = "文件已发送到您的 Telegram。"
        assert _detect_fabrication(resp, ["send_file"]) == []

    def test_fake_timestamp_no_data_tools(self):
        """Fake timestamp claim without data tools → signal."""
        resp = "Current UTC: 2025-03-15 14:30:00"
        signals = _detect_fabrication(resp, [])
        assert "fake_timestamp" in signals

    def test_fake_timestamp_with_data_tool(self):
        """Timestamp with exec tool → no signal (could be real)."""
        resp = "Current UTC: 2026-03-01 10:00:00"
        assert _detect_fabrication(resp, ["exec"]) == []

    def test_chinese_phase_complete(self):
        """'阶段 1 完成' without message tool → signal."""
        resp = "阶段 1 完成，共找到 5 条结果。"
        signals = _detect_fabrication(resp, [])
        assert "fake_phase" in signals

    def test_multiple_signals(self):
        """Multiple fabrication patterns in one response."""
        resp = (
            '已调用 send_file("report.pdf")\n'
            "Phase 1 Complete\n"
            '{"binance": 42000}\n'
            "报告已发送给用户"
        )
        signals = _detect_fabrication(resp, [])
        assert len(signals) >= 3


