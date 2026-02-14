"""Tests for ring1.llm_client."""

import json
import http.server
import threading

import pytest

from ring1.llm_client import ClaudeClient, LLMError


class TestClaudeClientInit:
    def test_missing_api_key_raises(self):
        with pytest.raises(LLMError, match="CLAUDE_API_KEY"):
            ClaudeClient(api_key="")

    def test_valid_init(self):
        client = ClaudeClient(api_key="sk-test", model="test-model", max_tokens=100)
        assert client.api_key == "sk-test"
        assert client.model == "test-model"
        assert client.max_tokens == 100

    def test_default_values(self):
        client = ClaudeClient(api_key="sk-test")
        assert client.model == "claude-sonnet-4-5-20250929"
        assert client.max_tokens == 4096


class _MockHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP handler that returns canned Claude API responses.

    Supports a single response_body or a response_bodies list (consumed in order).
    """

    # Class-level state for test control.
    response_body: dict = {}
    response_bodies: list[dict] = []  # if set, pops from front on each call
    status_code: int = 200
    call_count: int = 0

    def do_POST(self):
        _MockHandler.call_count += 1
        content_len = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_len)  # consume body

        # Pick response: from sequence or single body.
        if _MockHandler.response_bodies:
            body = _MockHandler.response_bodies.pop(0)
        else:
            body = _MockHandler.response_body

        self.send_response(_MockHandler.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        pass  # suppress output


@pytest.fixture
def mock_api(monkeypatch):
    """Start a local HTTP server and patch API_URL to point to it."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    import ring1.llm_client as mod
    monkeypatch.setattr(mod, "API_URL", f"http://127.0.0.1:{port}/v1/messages")
    # Reset handler state.
    _MockHandler.call_count = 0
    _MockHandler.status_code = 200
    _MockHandler.response_bodies = []
    _MockHandler.response_body = {
        "content": [{"type": "text", "text": "Hello from Claude"}]
    }

    yield _MockHandler

    server.shutdown()


class TestSendMessage:
    def test_success(self, mock_api):
        client = ClaudeClient(api_key="sk-test")
        result = client.send_message("system", "hello")
        assert result == "Hello from Claude"
        assert mock_api.call_count == 1

    def test_custom_response(self, mock_api):
        mock_api.response_body = {
            "content": [{"type": "text", "text": "custom reply"}]
        }
        client = ClaudeClient(api_key="sk-test")
        result = client.send_message("system", "hello")
        assert result == "custom reply"

    def test_no_text_content_raises(self, mock_api):
        mock_api.response_body = {"content": [{"type": "image", "data": "..."}]}
        client = ClaudeClient(api_key="sk-test")
        with pytest.raises(LLMError, match="No text content"):
            client.send_message("system", "hello")

    def test_http_400_no_retry(self, mock_api):
        mock_api.status_code = 400
        mock_api.response_body = {"error": "bad request"}
        client = ClaudeClient(api_key="sk-test")
        with pytest.raises(LLMError, match="HTTP 400"):
            client.send_message("system", "hello")
        # 400 should NOT be retried.
        assert mock_api.call_count == 1

    def test_http_429_retries(self, mock_api, monkeypatch):
        import ring1.llm_client as mod
        monkeypatch.setattr(mod, "_BASE_DELAY", 0.01)

        mock_api.status_code = 429
        client = ClaudeClient(api_key="sk-test")
        with pytest.raises(LLMError):
            client.send_message("system", "hello")
        # Should have retried 3 times.
        assert mock_api.call_count == 3

    def test_http_500_retries(self, mock_api, monkeypatch):
        import ring1.llm_client as mod
        monkeypatch.setattr(mod, "_BASE_DELAY", 0.01)

        mock_api.status_code = 500
        client = ClaudeClient(api_key="sk-test")
        with pytest.raises(LLMError):
            client.send_message("system", "hello")
        assert mock_api.call_count == 3


# ---------------------------------------------------------------------------
# TestSendMessageWithTools
# ---------------------------------------------------------------------------

_DUMMY_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
]


def _dummy_executor(name: str, tool_input: dict) -> str:
    if name == "web_search":
        return json.dumps([{"title": "Result", "url": "https://r.com"}])
    return "unknown tool"


class TestSendMessageWithTools:
    def test_no_tool_use(self, mock_api):
        """When Claude returns text without tool_use, return it directly."""
        mock_api.response_body = {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "No tools needed"}],
        }
        client = ClaudeClient(api_key="sk-test")
        result = client.send_message_with_tools(
            "system", "hello", _DUMMY_TOOLS, _dummy_executor,
        )
        assert result == "No tools needed"
        assert mock_api.call_count == 1

    def test_single_tool_round(self, mock_api):
        """Claude calls a tool once, then returns final text."""
        mock_api.response_bodies = [
            # Round 1: tool_use
            {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "Let me search..."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "web_search",
                        "input": {"query": "test"},
                    },
                ],
            },
            # Round 2: final text
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Here are results"}],
            },
        ]
        client = ClaudeClient(api_key="sk-test")
        result = client.send_message_with_tools(
            "system", "search test", _DUMMY_TOOLS, _dummy_executor,
        )
        assert result == "Here are results"
        assert mock_api.call_count == 2

    def test_multi_round_tool_use(self, mock_api):
        """Claude calls tools over multiple rounds."""
        mock_api.response_bodies = [
            # Round 1: tool_use
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "web_search",
                        "input": {"query": "first"},
                    },
                ],
            },
            # Round 2: another tool_use
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_2",
                        "name": "web_search",
                        "input": {"query": "second"},
                    },
                ],
            },
            # Round 3: final text
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Final answer"}],
            },
        ]
        client = ClaudeClient(api_key="sk-test")
        result = client.send_message_with_tools(
            "system", "deep search", _DUMMY_TOOLS, _dummy_executor,
        )
        assert result == "Final answer"
        assert mock_api.call_count == 3

    def test_max_rounds_exhausted_with_text(self, mock_api):
        """When max_rounds is exhausted, return whatever text was last seen."""
        mock_api.response_bodies = [
            {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "Partial"},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "web_search",
                        "input": {"query": "q"},
                    },
                ],
            },
        ] * 2  # 2 identical tool_use rounds, but max_rounds=1
        client = ClaudeClient(api_key="sk-test")
        # max_rounds=1 means only one API call; it returns tool_use, so loop
        # processes that round and then exhausts max_rounds.
        result = client.send_message_with_tools(
            "system", "test", _DUMMY_TOOLS, _dummy_executor, max_rounds=1,
        )
        assert result == "Partial"

    def test_max_rounds_exhausted_no_text_raises(self, mock_api):
        """When max_rounds is exhausted with no text at all, raise LLMError."""
        mock_api.response_bodies = [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "web_search",
                        "input": {"query": "q"},
                    },
                ],
            },
        ]
        client = ClaudeClient(api_key="sk-test")
        with pytest.raises(LLMError, match="loop exhausted"):
            client.send_message_with_tools(
                "system", "test", _DUMMY_TOOLS, _dummy_executor, max_rounds=1,
            )

    def test_tool_executor_exception_handled(self, mock_api):
        """If tool_executor raises, error string is sent back to Claude."""
        def failing_executor(name, inp):
            raise RuntimeError("boom")

        mock_api.response_bodies = [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "web_search",
                        "input": {"query": "q"},
                    },
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Handled error"}],
            },
        ]
        client = ClaudeClient(api_key="sk-test")
        result = client.send_message_with_tools(
            "system", "test", _DUMMY_TOOLS, failing_executor,
        )
        assert result == "Handled error"
        assert mock_api.call_count == 2
