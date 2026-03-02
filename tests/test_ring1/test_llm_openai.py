"""Tests for ring1.llm_openai — OpenAI-compatible LLM client."""

import http.server
import json
import threading

import pytest

from ring1.llm_base import LLMClient, LLMError
from ring1.llm_openai import OpenAIClient, _convert_tool_schema


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


class TestConvertToolSchema:
    def test_basic_conversion(self):
        anthropic_tool = {
            "name": "web_search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        result = _convert_tool_schema(anthropic_tool)
        assert result == {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }

    def test_missing_input_schema(self):
        tool = {"name": "noop", "description": "Does nothing"}
        result = _convert_tool_schema(tool)
        assert result["function"]["parameters"] == {}


# ---------------------------------------------------------------------------
# Mock HTTP server
# ---------------------------------------------------------------------------


class _MockHandler(http.server.BaseHTTPRequestHandler):
    """Returns canned OpenAI chat completions responses."""

    response_body: dict = {}
    response_bodies: list[dict] = []
    status_code: int = 200
    call_count: int = 0
    last_payload: dict = {}

    def do_POST(self):
        _MockHandler.call_count += 1
        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len)
        _MockHandler.last_payload = json.loads(raw) if raw else {}

        if _MockHandler.response_bodies:
            body = _MockHandler.response_bodies.pop(0)
        else:
            body = _MockHandler.response_body

        self.send_response(_MockHandler.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        pass


@pytest.fixture
def mock_api():
    """Start a local HTTP server mimicking OpenAI chat completions."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    _MockHandler.call_count = 0
    _MockHandler.status_code = 200
    _MockHandler.response_bodies = []
    _MockHandler.last_payload = {}
    _MockHandler.response_body = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello from OpenAI"}, "finish_reason": "stop"}
        ]
    }

    yield _MockHandler, f"http://127.0.0.1:{port}/v1/chat/completions"

    server.shutdown()


# ---------------------------------------------------------------------------
# Client init
# ---------------------------------------------------------------------------


class TestOpenAIClientInit:
    def test_empty_api_key_allowed_for_self_hosted(self):
        """Empty API key is valid for self-hosted LLMs (e.g. Ollama)."""
        client = OpenAIClient(api_key="", model="gpt-4o")
        assert client.api_key == ""

    def test_valid_init(self):
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", max_tokens=100)
        assert client.api_key == "sk-test"
        assert client.model == "gpt-4o"
        assert client.max_tokens == 100

    def test_is_llm_client(self):
        client = OpenAIClient(api_key="sk-test", model="gpt-4o")
        assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------


class TestSendMessage:
    def test_success(self, mock_api):
        handler, url = mock_api
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message("system prompt", "hello")
        assert result == "Hello from OpenAI"
        assert handler.call_count == 1

    def test_usage_tracked(self, mock_api):
        handler, url = mock_api
        handler.response_body = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 35},
        }
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        client.send_message("system", "hello")
        assert client.last_usage == {"input_tokens": 120, "output_tokens": 35}

    def test_usage_zero_when_missing(self, mock_api):
        handler, url = mock_api
        handler.response_body = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
            ],
        }
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        client.send_message("system", "hello")
        assert client.last_usage == {"input_tokens": 0, "output_tokens": 0}

    def test_system_message_sent(self, mock_api):
        handler, url = mock_api
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        client.send_message("my system prompt", "my user message")
        messages = handler.last_payload["messages"]
        assert messages[0] == {"role": "system", "content": "my system prompt"}
        assert messages[1] == {"role": "user", "content": "my user message"}

    def test_no_choices_raises(self, mock_api):
        handler, url = mock_api
        handler.response_body = {"choices": []}
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        with pytest.raises(LLMError, match="No choices"):
            client.send_message("system", "hello")

    def test_no_content_raises(self, mock_api):
        handler, url = mock_api
        handler.response_body = {"choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]}
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        with pytest.raises(LLMError, match="No text content"):
            client.send_message("system", "hello")

    def test_http_400_no_retry(self, mock_api):
        handler, url = mock_api
        handler.status_code = 400
        handler.response_body = {"error": "bad request"}
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        with pytest.raises(LLMError, match="HTTP 400"):
            client.send_message("system", "hello")
        assert handler.call_count == 1

    def test_http_429_retries(self, mock_api, monkeypatch):
        monkeypatch.setattr(OpenAIClient, "_BASE_DELAY", 0.01)

        handler, url = mock_api
        handler.status_code = 429
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        with pytest.raises(LLMError):
            client.send_message("system", "hello")
        assert handler.call_count == 3

    def test_http_500_retries(self, mock_api, monkeypatch):
        monkeypatch.setattr(OpenAIClient, "_BASE_DELAY", 0.01)

        handler, url = mock_api
        handler.status_code = 500
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        with pytest.raises(LLMError):
            client.send_message("system", "hello")
        assert handler.call_count == 3


# ---------------------------------------------------------------------------
# send_message_with_tools
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
        handler, url = mock_api
        handler.response_body = {
            "choices": [{
                "message": {"role": "assistant", "content": "No tools needed"},
                "finish_reason": "stop",
            }]
        }
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message_with_tools(
            "system", "hello", _DUMMY_TOOLS, _dummy_executor,
        )
        assert result == "No tools needed"
        assert handler.call_count == 1

    def test_single_tool_round(self, mock_api):
        handler, url = mock_api
        handler.response_bodies = [
            # Round 1: tool call
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Let me search...",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query": "test"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
            # Round 2: final text
            {
                "choices": [{
                    "message": {"role": "assistant", "content": "Here are results"},
                    "finish_reason": "stop",
                }]
            },
        ]
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message_with_tools(
            "system", "search test", _DUMMY_TOOLS, _dummy_executor,
        )
        assert result == "Here are results"
        assert handler.call_count == 2

    def test_multi_round_tool_use(self, mock_api):
        handler, url = mock_api
        handler.response_bodies = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "first"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "second"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
            {
                "choices": [{
                    "message": {"role": "assistant", "content": "Final answer"},
                    "finish_reason": "stop",
                }]
            },
        ]
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message_with_tools(
            "system", "deep search", _DUMMY_TOOLS, _dummy_executor,
        )
        assert result == "Final answer"
        assert handler.call_count == 3

    def test_max_rounds_exhausted_with_text(self, mock_api):
        handler, url = mock_api
        handler.response_bodies = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Partial",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "q"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
        ] * 2
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message_with_tools(
            "system", "test", _DUMMY_TOOLS, _dummy_executor, max_rounds=1,
        )
        assert result == "Partial"

    def test_max_rounds_exhausted_no_text(self, mock_api):
        handler, url = mock_api
        handler.response_bodies = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "q"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
        ]
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message_with_tools(
            "system", "test", _DUMMY_TOOLS, _dummy_executor, max_rounds=1,
        )
        assert "ran out of tool-call budget" in result

    def test_tool_executor_exception_handled(self, mock_api):
        handler, url = mock_api

        def failing_executor(name, inp):
            raise RuntimeError("boom")

        handler.response_bodies = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "q"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
            {
                "choices": [{
                    "message": {"role": "assistant", "content": "Handled error"},
                    "finish_reason": "stop",
                }]
            },
        ]
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message_with_tools(
            "system", "test", _DUMMY_TOOLS, failing_executor,
        )
        assert result == "Handled error"
        assert handler.call_count == 2

    def test_tools_converted_to_openai_format(self, mock_api):
        """Verify tools are sent in OpenAI function-calling format."""
        handler, url = mock_api
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        client.send_message_with_tools(
            "system", "hello", _DUMMY_TOOLS, _dummy_executor,
        )
        tools_sent = handler.last_payload.get("tools", [])
        assert len(tools_sent) == 1
        assert tools_sent[0]["type"] == "function"
        assert tools_sent[0]["function"]["name"] == "web_search"
        assert "parameters" in tools_sent[0]["function"]

    def test_usage_accumulated_across_rounds(self, mock_api):
        handler, url = mock_api
        handler.response_bodies = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "q"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 80, "completion_tokens": 20},
            },
            {
                "choices": [{
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 150, "completion_tokens": 40},
            },
        ]
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        client.send_message_with_tools(
            "system", "test", _DUMMY_TOOLS, _dummy_executor,
        )
        assert client.last_usage == {"input_tokens": 230, "output_tokens": 60}

    def test_tool_results_compressed_after_api_call(self, mock_api):
        """Large tool results from earlier rounds should be compressed in messages."""
        handler, url = mock_api
        big_result = "X" * 3000

        def big_executor(name, inp):
            return big_result

        handler.response_bodies = [
            # Round 1: tool call
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "first"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
            # Round 2: another tool call — round 1 result should be compressed
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "second"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
            # Round 3: final text
            {
                "choices": [{
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }]
            },
        ]
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message_with_tools(
            "system", "test", _DUMMY_TOOLS, big_executor, max_rounds=5,
        )
        assert result == "Done"
        assert handler.call_count == 3

    def test_invalid_json_arguments(self, mock_api):
        """Bad JSON in tool arguments should not crash — fall back to empty dict."""
        handler, url = mock_api
        handler.response_bodies = [
            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "not json"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            },
            {
                "choices": [{
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }]
            },
        ]
        client = OpenAIClient(api_key="sk-test", model="gpt-4o", api_url=url)
        result = client.send_message_with_tools(
            "system", "test", _DUMMY_TOOLS, _dummy_executor,
        )
        assert result == "OK"
