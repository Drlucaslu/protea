"""Ring 1 configuration — load [ring1] from config.toml + .env secrets.

Pure stdlib.  Reads `.env` into os.environ for secrets (API keys, tokens).
"""

from __future__ import annotations

import os
import pathlib
from typing import NamedTuple


class Ring1Config(NamedTuple):
    claude_api_key: str
    claude_model: str
    claude_max_tokens: int
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_enabled: bool
    max_prompt_history: int
    p1_enabled: bool
    p1_idle_threshold_sec: int
    p1_check_interval_sec: int
    workspace_path: str = "."
    shell_timeout: int = 30
    max_tool_rounds: int = 25
    llm_provider: str = ""       # "anthropic"|"openai"|"deepseek"|"qwen" (empty = anthropic)
    llm_api_key_env: str = ""    # env var name for API key (empty = CLAUDE_API_KEY)
    llm_model: str = ""          # model name (empty = claude_model)
    llm_max_tokens: int = 0      # 0 = use claude_max_tokens
    llm_api_url: str = ""        # custom API URL override

    def has_llm_config(self) -> bool:
        """Check whether any LLM provider is configured with an API key."""
        provider = self.llm_provider or "anthropic"
        if provider == "anthropic":
            return bool(self.claude_api_key)
        return bool(self.llm_api_key_env and os.environ.get(self.llm_api_key_env))

    def get_llm_client(self):
        """Create an LLM client from config, with backward compatibility.

        When no ``[ring1.llm]`` section is configured (llm_provider is empty),
        falls back to the Anthropic client using ``claude_*`` fields.
        """
        from ring1.llm_base import LLMError, create_llm_client

        provider = self.llm_provider or "anthropic"

        if provider == "anthropic":
            api_key = self.claude_api_key
            model = self.llm_model or self.claude_model
            max_tokens = self.llm_max_tokens or self.claude_max_tokens
        else:
            env_var = self.llm_api_key_env
            if not env_var:
                raise LLMError(
                    f"llm_api_key_env must be set for provider {provider!r}"
                )
            api_key = os.environ.get(env_var, "")
            model = self.llm_model
            max_tokens = self.llm_max_tokens or self.claude_max_tokens

        return create_llm_client(
            provider=provider,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            api_url=self.llm_api_url or None,
        )


def _load_dotenv(project_root: pathlib.Path) -> None:
    """Parse a simple .env file and inject into os.environ.

    Handles KEY=VALUE, ignores comments (#) and blank lines.
    Strips optional surrounding quotes from values.
    """
    env_path = project_root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip matching quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


def load_ring1_config(project_root: pathlib.Path) -> Ring1Config:
    """Load Ring 1 config from config.toml + environment variables."""
    import tomllib

    _load_dotenv(project_root)

    cfg_path = project_root / "config" / "config.toml"
    with open(cfg_path, "rb") as f:
        toml = tomllib.load(f)

    r1 = toml.get("ring1", {})
    tg = r1.get("telegram", {})
    autonomy = r1.get("autonomy", {})
    tools = r1.get("tools", {})
    llm = r1.get("llm", {})

    return Ring1Config(
        claude_api_key=os.environ.get("CLAUDE_API_KEY", ""),
        claude_model=r1.get("claude_model", "claude-sonnet-4-5-20250929"),
        claude_max_tokens=r1.get("claude_max_tokens", 4096),
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        telegram_enabled=tg.get("enabled", False),
        max_prompt_history=r1.get("max_prompt_history", 10),
        p1_enabled=autonomy.get("enabled", True),
        p1_idle_threshold_sec=autonomy.get("idle_threshold_sec", 600),
        p1_check_interval_sec=autonomy.get("check_interval_sec", 60),
        workspace_path=tools.get("workspace_path", "."),
        shell_timeout=tools.get("shell_timeout", 30),
        max_tool_rounds=tools.get("max_tool_rounds", 25),
        # LLM provider config: env vars (LLM_*) override config.toml [ring1.llm].
        # This allows per-machine provider selection via .env while sharing
        # the same config.toml across nodes.
        llm_provider=os.environ.get("LLM_PROVIDER", "") or llm.get("provider", ""),
        llm_api_key_env=os.environ.get("LLM_API_KEY_ENV", "") or llm.get("api_key_env", ""),
        llm_model=os.environ.get("LLM_MODEL", "") or llm.get("model", ""),
        llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "0") or 0) or llm.get("max_tokens", 0),
        llm_api_url=os.environ.get("LLM_API_URL", "") or llm.get("api_url", ""),
    )
