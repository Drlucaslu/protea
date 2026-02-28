#!/usr/bin/env python3
"""Protea Setup Wizard — Web UI and CLI modes for first-time configuration.

Usage:
    python setup_wizard.py --web   # Launch Web UI on port 8899
    python setup_wizard.py --cli   # Interactive terminal setup

Pure stdlib — no external dependencies required.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent

# ───────────────────────────────────────────────────────────────────
# LLM Provider Definitions
# ───────────────────────────────────────────────────────────────────

PROVIDERS = {
    "anthropic": {
        "label": "Anthropic (Claude)",
        "env_key": "CLAUDE_API_KEY",
        "config_provider": "anthropic",
        "validate_url": "https://api.anthropic.com/v1/messages",
        "placeholder": "sk-ant-api03-...",
        "default_endpoint": "https://api.anthropic.com/v1/messages",
        "requires_key": True,
    },
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "config_provider": "openai",
        "validate_url": "https://api.openai.com/v1/models",
        "placeholder": "sk-...",
        "default_endpoint": "https://api.openai.com/v1/chat/completions",
        "requires_key": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "config_provider": "deepseek",
        "validate_url": "https://api.deepseek.com/v1/models",
        "placeholder": "sk-...",
        "default_endpoint": "https://api.deepseek.com/v1/chat/completions",
        "requires_key": True,
    },
    "qwen": {
        "label": "Qwen (\u5343\u95ee)",
        "env_key": "DASHSCOPE_API_KEY",
        "config_provider": "qwen",
        "validate_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
        "placeholder": "sk-...",
        "default_endpoint": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        "alt_endpoints": {
            "International": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
            "China (\u4e2d\u56fd\u5927\u9646)": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        },
        "requires_key": True,
    },
    "minimax": {
        "label": "MiniMax",
        "env_key": "MINIMAX_API_KEY",
        "config_provider": "minimax",
        "validate_url": "https://api.minimax.chat/v1/text/chatcompletion_v2",
        "placeholder": "eyJ...",
        "default_endpoint": "https://api.minimax.chat/v1/text/chatcompletion_v2",
        "alt_endpoints": {
            "International": "https://api.minimax.chat/v1/text/chatcompletion_v2",
            "China (\u4e2d\u56fd\u5927\u9646)": "https://api.minimax.cn/v1/text/chatcompletion_v2",
        },
        "requires_key": True,
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "env_key": "MOONSHOT_API_KEY",
        "config_provider": "kimi",
        "validate_url": "https://api.moonshot.cn/v1/models",
        "placeholder": "sk-...",
        "default_endpoint": "https://api.moonshot.cn/v1/chat/completions",
        "requires_key": True,
    },
    "gemini": {
        "label": "Gemini (Google)",
        "env_key": "GEMINI_API_KEY",
        "config_provider": "gemini",
        "validate_url": "https://generativelanguage.googleapis.com/v1beta/openai/models",
        "placeholder": "AIza...",
        "default_endpoint": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "requires_key": True,
    },
    "ollama": {
        "label": "Ollama (Local)",
        "env_key": "",
        "config_provider": "ollama",
        "validate_url": "http://localhost:11434/v1/models",
        "placeholder": "",
        "default_endpoint": "http://localhost:11434/v1/chat/completions",
        "requires_key": False,
    },
}

# ───────────────────────────────────────────────────────────────────
# API Key Validation
# ───────────────────────────────────────────────────────────────────


def validate_api_key(provider_id: str, api_key: str, endpoint: str = "") -> tuple[bool, str]:
    """Validate an API key by making a lightweight API call.

    Returns (success, message).
    """
    provider = PROVIDERS.get(provider_id)
    if not provider:
        return False, f"Unknown provider: {provider_id}"

    # Ollama: no key needed — just check the endpoint is reachable.
    if provider_id == "ollama":
        url = (endpoint.strip().rstrip("/") if endpoint else
               provider["validate_url"])
        # Normalize: if user gave chat/completions URL, derive models URL.
        if url.endswith("/chat/completions"):
            url = url.rsplit("/chat/completions", 1)[0] + "/models"
        elif not url.endswith("/models"):
            url = url.rstrip("/") + "/v1/models"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True, "Ollama is running"
        except urllib.error.URLError:
            return False, "Cannot connect — is Ollama running?"
        except Exception as exc:
            return False, f"Connection error: {exc}"
        return True, "Ollama is running"

    if not api_key or not api_key.strip():
        return False, "API key is empty"

    try:
        if provider_id == "anthropic":
            # Anthropic: POST /v1/messages with minimal payload.
            data = json.dumps({
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }).encode()
            req = urllib.request.Request(
                provider["validate_url"],
                data=data,
                headers={
                    "x-api-key": api_key.strip(),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
        elif provider_id == "gemini":
            # Gemini uses API key as query parameter.
            url = provider["validate_url"] + f"?key={api_key.strip()}"
            req = urllib.request.Request(url)
        else:
            # OpenAI-compatible: GET /v1/models.
            req = urllib.request.Request(
                provider["validate_url"],
                headers={"Authorization": f"Bearer {api_key.strip()}"},
            )

        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return True, "Valid"
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        if exc.code == 403:
            return False, "Access denied (403 Forbidden)"
        # 400/429 etc still means the key is recognized.
        if exc.code in (400, 429):
            return True, f"Key accepted (HTTP {exc.code})"
        return False, f"HTTP {exc.code}: {exc.reason}"
    except urllib.error.URLError as exc:
        return False, f"Connection failed: {exc.reason}"
    except Exception as exc:
        return False, f"Validation error: {exc}"

    return True, "Valid"


# ───────────────────────────────────────────────────────────────────
# Setup Actions (shared by Web and CLI)
# ───────────────────────────────────────────────────────────────────


def write_env_file(
    provider_id: str,
    api_key: str,
    telegram_token: str = "",
    telegram_chat_id: str = "",
    endpoint: str = "",
) -> None:
    """Write .env file with the provided configuration."""
    provider = PROVIDERS[provider_id]
    env_key = provider["env_key"]
    default_ep = provider.get("default_endpoint", "")

    lines = [
        f"# Protea configuration (generated by setup wizard)",
        f"",
    ]

    # Write the primary API key.
    if provider_id == "anthropic":
        lines.append(f"CLAUDE_API_KEY={api_key}")
    elif provider_id == "ollama":
        lines.append(f"CLAUDE_API_KEY=")
        lines.append(f"LLM_PROVIDER=ollama")
        lines.append(f"LLM_API_KEY_ENV=")
    else:
        lines.append(f"CLAUDE_API_KEY=")
        lines.append(f"{env_key}={api_key}")
        lines.append(f"LLM_PROVIDER={provider['config_provider']}")
        lines.append(f"LLM_API_KEY_ENV={env_key}")

    # Write endpoint if non-default.
    ep = endpoint.strip() if endpoint else ""
    if ep and ep != default_ep:
        lines.append(f"LLM_API_URL={ep}")
    elif provider_id == "ollama":
        # Always write Ollama URL since it's non-standard.
        lines.append(f"LLM_API_URL={ep or default_ep}")

    lines.extend([
        "",
        f"# Telegram Bot (optional)",
        f"TELEGRAM_BOT_TOKEN={telegram_token}",
        f"TELEGRAM_CHAT_ID={telegram_chat_id}",
    ])

    (ROOT / ".env").write_text("\n".join(lines) + "\n")


def init_ring2_git() -> bool:
    """Initialize Ring 2 git repo if needed. Returns True on success."""
    ring2 = ROOT / "ring2"
    if (ring2 / ".git").is_dir():
        return True
    try:
        subprocess.run(["git", "init", str(ring2)], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(ring2), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(ring2), "commit", "-m", "init"],
                       check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def create_directories() -> None:
    """Create data/ and output/ directories."""
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "output").mkdir(exist_ok=True)


def run_setup(
    provider_id: str,
    api_key: str,
    telegram_token: str = "",
    telegram_chat_id: str = "",
    endpoint: str = "",
) -> list[dict]:
    """Execute all setup steps. Returns list of {step, ok, msg}."""
    results = []

    # 1. Write .env
    try:
        write_env_file(provider_id, api_key, telegram_token, telegram_chat_id, endpoint)
        results.append({"step": "Write .env", "ok": True, "msg": "Created"})
    except Exception as exc:
        results.append({"step": "Write .env", "ok": False, "msg": str(exc)})

    # 2. Init Ring 2 git
    ok = init_ring2_git()
    results.append({
        "step": "Initialize Ring 2 git",
        "ok": ok,
        "msg": "Ready" if ok else "Failed (non-fatal)",
    })

    # 3. Create directories
    try:
        create_directories()
        results.append({"step": "Create data directories", "ok": True, "msg": "Ready"})
    except Exception as exc:
        results.append({"step": "Create data directories", "ok": False, "msg": str(exc)})

    # 4. Make scripts executable
    for script in ("run_with_nohup.sh", "stop_run.sh", "install.sh"):
        path = ROOT / script
        if path.exists():
            path.chmod(path.stat().st_mode | 0o755)
    results.append({"step": "Set script permissions", "ok": True, "msg": "Done"})

    return results


# ───────────────────────────────────────────────────────────────────
# CLI Mode
# ───────────────────────────────────────────────────────────────────

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _cli_ok(msg: str) -> None:
    print(f"  {_GREEN}[ok]{_RESET} {msg}")


def _cli_fail(msg: str) -> None:
    print(f"  {_RED}[!!]{_RESET} {msg}")


def _cli_input(prompt: str) -> str:
    try:
        return input(f"  {prompt}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)


def run_cli() -> None:
    """Interactive CLI setup wizard."""
    print()
    print(f"  {_BOLD}=== Protea Setup (CLI) ==={_RESET}")
    print()

    # Already configured?
    if (ROOT / ".env").exists():
        answer = _cli_input(".env already exists. Overwrite? [y/N]: ")
        if answer.lower() != "y":
            _cli_ok("Keeping existing .env")
            # Still run other setup steps.
            results = run_setup.__wrapped__(None, None) if hasattr(run_setup, "__wrapped__") else []
            print()
            print(f"  {_BOLD}Protea is ready.{_RESET}")
            _print_start_instructions()
            return

    # Step 1: LLM Provider
    print(f"  {_BOLD}[1/4] LLM Provider{_RESET}")
    provider_ids = list(PROVIDERS.keys())
    for i, pid in enumerate(provider_ids, 1):
        label = PROVIDERS[pid]["label"]
        rec = " (recommended)" if pid == "anthropic" else ""
        print(f"    {i}. {label}{_DIM}{rec}{_RESET}")
    choice = _cli_input("  Choose [1]: ") or "1"
    try:
        provider_id = provider_ids[int(choice) - 1]
    except (ValueError, IndexError):
        provider_id = "anthropic"
    provider = PROVIDERS[provider_id]
    _cli_ok(f"{provider['label']}")
    print()

    # Step 2: API Key (skip for Ollama)
    api_key = ""
    if provider.get("requires_key", True):
        print(f"  {_BOLD}[2/4] API Key{_RESET}")
        while True:
            api_key = _cli_input(f"  {provider['env_key']}: ")
            if not api_key:
                _cli_fail("API key is required.")
                continue
            print(f"  {_YELLOW}  Validating...{_RESET}", end="", flush=True)
            valid, msg = validate_api_key(provider_id, api_key)
            print(f"\r  {'  ' * 15}\r", end="")  # clear line
            if valid:
                _cli_ok(f"Key valid ({msg})")
                break
            else:
                _cli_fail(f"{msg}")
                retry = _cli_input("  Try again? [Y/n]: ") or "Y"
                if retry.lower() == "n":
                    _cli_fail("Aborted.")
                    sys.exit(1)
    else:
        print(f"  {_BOLD}[2/4] API Key{_RESET}")
        print(f"  {_DIM}  Not required for {provider['label']}{_RESET}")
    print()

    # Step 3: Endpoint (advanced, optional)
    print(f"  {_BOLD}[3/4] API Endpoint{_RESET}")
    default_ep = provider.get("default_endpoint", "")
    print(f"  {_DIM}  Default: {default_ep}{_RESET}")
    alt_eps = provider.get("alt_endpoints")
    if alt_eps:
        for name, url in alt_eps.items():
            print(f"  {_DIM}  {name}: {url}{_RESET}")
    endpoint = _cli_input("  Custom endpoint (Enter for default): ")
    if endpoint:
        _cli_ok(f"Endpoint: {endpoint}")
    else:
        print(f"  {_DIM}  Using default{_RESET}")
    if provider_id == "ollama" and not endpoint:
        # Validate Ollama is running.
        print(f"  {_YELLOW}  Checking Ollama...{_RESET}", end="", flush=True)
        valid, msg = validate_api_key("ollama", "", endpoint)
        print(f"\r  {'  ' * 20}\r", end="")
        if valid:
            _cli_ok(msg)
        else:
            _cli_fail(msg)
    print()

    # Step 4: Telegram (optional)
    print(f"  {_BOLD}[4/4] Telegram (optional){_RESET}")
    telegram_token = _cli_input("  Bot Token (Enter to skip): ")
    telegram_chat_id = ""
    if telegram_token:
        telegram_chat_id = _cli_input("  Chat ID (Enter for auto-detect): ")
    if not telegram_token:
        print(f"  {_DIM}  Skipped{_RESET}")
    else:
        _cli_ok("Telegram configured")
    print()

    # Run setup
    print(f"  {_BOLD}Setting up...{_RESET}")
    results = run_setup(provider_id, api_key, telegram_token, telegram_chat_id, endpoint)
    for r in results:
        if r["ok"]:
            _cli_ok(f"{r['step']}: {r['msg']}")
        else:
            _cli_fail(f"{r['step']}: {r['msg']}")

    print()
    print(f"  {_GREEN}{_BOLD}=== Setup Complete ==={_RESET}")
    _print_start_instructions()


def _print_start_instructions() -> None:
    print()
    print(f"  To start Protea:")
    print(f"    .venv/bin/python run.py        {_DIM}# foreground{_RESET}")
    print(f"    ./run_with_nohup.sh            {_DIM}# background{_RESET}")
    print(f"    http://localhost:8899           {_DIM}# dashboard{_RESET}")
    print()


# ───────────────────────────────────────────────────────────────────
# Web UI Mode
# ───────────────────────────────────────────────────────────────────

_SETUP_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Protea Setup</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0e27; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }
a { color: #667eea; text-decoration: none; }

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem; text-align: center;
}
.header h1 { color: #fff; font-size: 1.6rem; font-weight: 600; }
.header p { color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 0.3rem; }

.container { max-width: 560px; margin: 2rem auto; padding: 0 1.5rem; }

.step {
    background: #151a3a; border: 1px solid #252a4a; border-radius: 12px;
    padding: 1.5rem; margin-bottom: 1.2rem; transition: border-color 0.2s;
}
.step:hover { border-color: #667eea; }
.step h2 { font-size: 1rem; color: #667eea; margin-bottom: 1rem; font-weight: 600; }
.step-num {
    display: inline-block; background: #667eea; color: #fff; width: 24px; height: 24px;
    border-radius: 50%; text-align: center; line-height: 24px; font-size: 0.8rem;
    margin-right: 0.5rem; font-weight: 700;
}

/* Radio group */
.radio-group { display: flex; flex-direction: column; gap: 0.5rem; }
.radio-option {
    display: flex; align-items: center; padding: 0.7rem 1rem; border-radius: 8px;
    border: 1px solid #252a4a; cursor: pointer; transition: all 0.15s;
}
.radio-option:hover { border-color: #667eea; background: rgba(102,126,234,0.05); }
.radio-option.selected { border-color: #667eea; background: rgba(102,126,234,0.1); }
.radio-option input { display: none; }
.radio-dot {
    width: 18px; height: 18px; border: 2px solid #555; border-radius: 50%;
    margin-right: 0.8rem; position: relative; flex-shrink: 0;
}
.radio-option.selected .radio-dot {
    border-color: #667eea;
}
.radio-option.selected .radio-dot::after {
    content: ''; position: absolute; top: 3px; left: 3px; width: 8px; height: 8px;
    background: #667eea; border-radius: 50%;
}
.radio-label { font-size: 0.9rem; }
.radio-rec { color: #667eea; font-size: 0.75rem; margin-left: 0.5rem; }

/* Input fields */
.field { margin-bottom: 1rem; }
.field label { display: block; color: #999; font-size: 0.8rem; margin-bottom: 0.4rem; }
.field input {
    width: 100%; padding: 0.7rem 1rem; border-radius: 8px;
    border: 1px solid #252a4a; background: #0d1130; color: #e0e0e0;
    font-size: 0.9rem; font-family: 'SF Mono', 'Fira Code', monospace;
    transition: border-color 0.15s;
}
.field input:focus { outline: none; border-color: #667eea; }
.field input::placeholder { color: #555; }
.field .hint { font-size: 0.75rem; color: #555; margin-top: 0.3rem; }

/* Validate button */
.validate-btn {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 0.9rem; border-radius: 6px; border: 1px solid #667eea;
    background: transparent; color: #667eea; font-size: 0.8rem; cursor: pointer;
    transition: all 0.15s;
}
.validate-btn:hover { background: rgba(102,126,234,0.1); }
.validate-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.validate-result { font-size: 0.8rem; margin-left: 0.8rem; }
.validate-result.ok { color: #48bb78; }
.validate-result.err { color: #fc8181; }

/* Collapsible */
.collapse-toggle {
    display: flex; align-items: center; cursor: pointer; color: #999;
    font-size: 0.85rem; margin-bottom: 0.5rem;
}
.collapse-toggle .arrow { margin-right: 0.5rem; transition: transform 0.2s; }
.collapse-toggle.open .arrow { transform: rotate(90deg); }
.collapse-body { display: none; }
.collapse-body.open { display: block; }

/* Submit */
.submit-btn {
    width: 100%; padding: 0.9rem; border-radius: 10px; border: none;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff; font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: opacity 0.15s; margin-top: 0.5rem;
}
.submit-btn:hover { opacity: 0.9; }
.submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }

/* Progress / Results */
.progress { margin-top: 1.2rem; }
.progress-item {
    display: flex; align-items: center; padding: 0.4rem 0; font-size: 0.85rem;
}
.progress-item .icon { margin-right: 0.6rem; font-size: 1rem; }
.progress-item.ok .icon { color: #48bb78; }
.progress-item.err .icon { color: #fc8181; }
.progress-item.pending .icon { color: #667eea; }

.done-box {
    text-align: center; padding: 2rem; margin-top: 1.2rem;
    background: #151a3a; border-radius: 12px; border: 1px solid #252a4a;
}
.done-box h2 { color: #48bb78; font-size: 1.2rem; margin-bottom: 0.8rem; }
.done-box code {
    display: block; padding: 0.6rem 1rem; margin: 0.3rem auto; border-radius: 6px;
    background: #0d1130; font-size: 0.85rem; max-width: 380px; text-align: left;
    font-family: 'SF Mono', 'Fira Code', monospace; color: #e0e0e0;
}
.done-box .hint { color: #777; font-size: 0.8rem; margin-top: 1rem; }

.hidden { display: none; }
</style>
</head>
<body>

<div class="header">
    <h1>Protea Setup</h1>
    <p>Configure your self-evolving AI system</p>
</div>

<div class="container">
    <!-- Step 1: Provider -->
    <div class="step" id="step1">
        <h2><span class="step-num">1</span> LLM Provider</h2>
        <div class="radio-group" id="providerGroup">
            <label class="radio-option selected" data-provider="anthropic">
                <input type="radio" name="provider" value="anthropic" checked>
                <span class="radio-dot"></span>
                <span class="radio-label">Anthropic (Claude)</span>
                <span class="radio-rec">recommended</span>
            </label>
            <label class="radio-option" data-provider="openai">
                <input type="radio" name="provider" value="openai">
                <span class="radio-dot"></span>
                <span class="radio-label">OpenAI</span>
            </label>
            <label class="radio-option" data-provider="qwen">
                <input type="radio" name="provider" value="qwen">
                <span class="radio-dot"></span>
                <span class="radio-label">Qwen (\u5343\u95ee)</span>
            </label>
            <label class="radio-option" data-provider="deepseek">
                <input type="radio" name="provider" value="deepseek">
                <span class="radio-dot"></span>
                <span class="radio-label">DeepSeek</span>
            </label>
            <label class="radio-option" data-provider="minimax">
                <input type="radio" name="provider" value="minimax">
                <span class="radio-dot"></span>
                <span class="radio-label">MiniMax</span>
            </label>
            <label class="radio-option" data-provider="kimi">
                <input type="radio" name="provider" value="kimi">
                <span class="radio-dot"></span>
                <span class="radio-label">Kimi (Moonshot)</span>
            </label>
            <label class="radio-option" data-provider="gemini">
                <input type="radio" name="provider" value="gemini">
                <span class="radio-dot"></span>
                <span class="radio-label">Gemini (Google)</span>
            </label>
            <label class="radio-option" data-provider="ollama">
                <input type="radio" name="provider" value="ollama">
                <span class="radio-dot"></span>
                <span class="radio-label">Ollama (Local)</span>
            </label>
        </div>
    </div>

    <!-- Step 2: API Key -->
    <div class="step" id="step2">
        <h2><span class="step-num">2</span> API Key</h2>
        <div id="keySection">
            <div class="field">
                <label id="keyLabel">CLAUDE_API_KEY</label>
                <input type="password" id="apiKey" placeholder="sk-ant-api03-..." autocomplete="off">
                <div class="hint">Your API key is stored locally in .env and never sent anywhere except the LLM provider.</div>
            </div>
        </div>
        <div id="ollamaHint" class="hidden" style="color:#999; font-size:0.85rem; margin-bottom:0.8rem;">
            No API key needed for local models. Make sure Ollama is running.
        </div>
        <!-- Advanced: Endpoint -->
        <div style="margin-top: 0.8rem;">
            <div class="collapse-toggle" id="advToggle" onclick="toggleAdvanced()">
                <span class="arrow">&#9654;</span> Advanced: API Endpoint
            </div>
            <div class="collapse-body" id="advBody">
                <div class="field">
                    <label id="epLabel">API Endpoint</label>
                    <input type="text" id="endpoint" placeholder="" autocomplete="off">
                    <div class="hint" id="epHint">Leave empty to use the default endpoint.</div>
                </div>
                <div id="altEndpoints" class="hidden" style="margin-top:0.4rem;"></div>
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-top: 0.8rem;">
            <button class="validate-btn" id="validateBtn" onclick="validateKey()">Validate</button>
            <span class="validate-result" id="validateResult"></span>
        </div>
    </div>

    <!-- Step 3: Telegram -->
    <div class="step" id="step3">
        <h2><span class="step-num">3</span> Telegram <span style="color:#555; font-weight:400; font-size:0.85rem;">(optional)</span></h2>
        <div class="field">
            <label>Bot Token</label>
            <input type="text" id="tgToken" placeholder="123456:ABC-DEF..." autocomplete="off">
        </div>
        <div class="field">
            <label>Chat ID</label>
            <input type="text" id="tgChatId" placeholder="Auto-detected from first message">
            <div class="hint">Leave empty for auto-detection.</div>
        </div>
    </div>

    <!-- Submit -->
    <div id="setupForm">
        <button class="submit-btn" id="submitBtn" onclick="doSetup()">Complete Setup</button>
    </div>

    <!-- Progress -->
    <div class="progress hidden" id="progress"></div>

    <!-- Done -->
    <div class="done-box hidden" id="doneBox">
        <h2>Setup Complete</h2>
        <p style="color:#999; margin-bottom:1rem;">Start Protea with:</p>
        <code>.venv/bin/python run.py</code>
        <code>./run_with_nohup.sh</code>
        <p class="hint">Dashboard will be available at http://localhost:8899 after starting.</p>
    </div>
</div>

<script>
// Provider switching
const providerMeta = {
    anthropic: { label: 'CLAUDE_API_KEY', placeholder: 'sk-ant-api03-...', needsKey: true,
                 endpoint: 'https://api.anthropic.com/v1/messages' },
    openai:    { label: 'OPENAI_API_KEY', placeholder: 'sk-...', needsKey: true,
                 endpoint: 'https://api.openai.com/v1/chat/completions' },
    qwen:      { label: 'DASHSCOPE_API_KEY', placeholder: 'sk-...', needsKey: true,
                 endpoint: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions',
                 alts: {'International': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions',
                        'China (\u4e2d\u56fd\u5927\u9646)': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'} },
    deepseek:  { label: 'DEEPSEEK_API_KEY', placeholder: 'sk-...', needsKey: true,
                 endpoint: 'https://api.deepseek.com/v1/chat/completions' },
    minimax:   { label: 'MINIMAX_API_KEY', placeholder: 'eyJ...', needsKey: true,
                 endpoint: 'https://api.minimax.chat/v1/text/chatcompletion_v2',
                 alts: {'International': 'https://api.minimax.chat/v1/text/chatcompletion_v2',
                        'China (\u4e2d\u56fd\u5927\u9646)': 'https://api.minimax.cn/v1/text/chatcompletion_v2'} },
    kimi:      { label: 'MOONSHOT_API_KEY', placeholder: 'sk-...', needsKey: true,
                 endpoint: 'https://api.moonshot.cn/v1/chat/completions' },
    gemini:    { label: 'GEMINI_API_KEY', placeholder: 'AIza...', needsKey: true,
                 endpoint: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions' },
    ollama:    { label: '', placeholder: '', needsKey: false,
                 endpoint: 'http://localhost:11434/v1/chat/completions' },
};

function switchProvider(p) {
    const meta = providerMeta[p];
    // Key section
    const keySection = document.getElementById('keySection');
    const ollamaHint = document.getElementById('ollamaHint');
    if (meta.needsKey) {
        keySection.classList.remove('hidden');
        ollamaHint.classList.add('hidden');
        document.getElementById('keyLabel').textContent = meta.label;
        document.getElementById('apiKey').placeholder = meta.placeholder;
    } else {
        keySection.classList.add('hidden');
        ollamaHint.classList.remove('hidden');
    }
    // Endpoint
    document.getElementById('endpoint').placeholder = meta.endpoint;
    document.getElementById('endpoint').value = '';
    // Alt endpoints
    const altDiv = document.getElementById('altEndpoints');
    if (meta.alts) {
        altDiv.innerHTML = '';
        const row = document.createElement('div');
        row.style.cssText = 'display:flex; gap:0.5rem; flex-wrap:wrap;';
        for (const [label, url] of Object.entries(meta.alts)) {
            const btn = document.createElement('button');
            btn.className = 'validate-btn';
            btn.style.fontSize = '0.75rem';
            btn.textContent = label;
            btn.addEventListener('click', function() { document.getElementById('endpoint').value = url; });
            row.appendChild(btn);
        }
        altDiv.appendChild(row);
        altDiv.classList.remove('hidden');
    } else {
        altDiv.classList.add('hidden');
        altDiv.innerHTML = '';
    }
    document.getElementById('validateResult').textContent = '';
}

function toggleAdvanced() {
    const toggle = document.getElementById('advToggle');
    const body = document.getElementById('advBody');
    toggle.classList.toggle('open');
    body.classList.toggle('open');
}

document.querySelectorAll('.radio-option').forEach(opt => {
    opt.addEventListener('click', () => {
        document.querySelectorAll('.radio-option').forEach(o => o.classList.remove('selected'));
        opt.classList.add('selected');
        opt.querySelector('input').checked = true;
        switchProvider(opt.dataset.provider);
    });
});

// Validate
async function validateKey() {
    const btn = document.getElementById('validateBtn');
    const result = document.getElementById('validateResult');
    const provider = document.querySelector('input[name=provider]:checked').value;
    const meta = providerMeta[provider];
    const key = document.getElementById('apiKey').value.trim();
    const ep = document.getElementById('endpoint').value.trim();
    if (meta.needsKey && !key) { result.textContent = 'Enter a key first'; result.className = 'validate-result err'; return; }

    btn.disabled = true;
    result.textContent = 'Validating...';
    result.className = 'validate-result';

    try {
        const resp = await fetch('/api/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, api_key: key, endpoint: ep }),
        });
        const data = await resp.json();
        if (data.valid) {
            result.textContent = '\\u2713 ' + data.message;
            result.className = 'validate-result ok';
        } else {
            result.textContent = '\\u2717 ' + data.message;
            result.className = 'validate-result err';
        }
    } catch (e) {
        result.textContent = 'Connection error';
        result.className = 'validate-result err';
    }
    btn.disabled = false;
}

// Setup
async function doSetup() {
    const provider = document.querySelector('input[name=provider]:checked').value;
    const meta = providerMeta[provider];
    const apiKey = document.getElementById('apiKey').value.trim();
    const ep = document.getElementById('endpoint').value.trim();
    if (meta.needsKey && !apiKey) { alert('API key is required'); return; }

    const tgToken = document.getElementById('tgToken').value.trim();
    const tgChatId = document.getElementById('tgChatId').value.trim();

    document.getElementById('submitBtn').disabled = true;
    const progress = document.getElementById('progress');
    progress.classList.remove('hidden');
    progress.innerHTML = '<div class="progress-item pending"><span class="icon">&#9679;</span> Running setup...</div>';

    try {
        const resp = await fetch('/api/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, api_key: apiKey, endpoint: ep, telegram_token: tgToken, telegram_chat_id: tgChatId }),
        });
        const data = await resp.json();
        let html = '';
        for (const r of data.results) {
            const cls = r.ok ? 'ok' : 'err';
            const icon = r.ok ? '&#10003;' : '&#10007;';
            html += '<div class="progress-item ' + cls + '"><span class="icon">' + icon + '</span> ' + r.step + ': ' + r.msg + '</div>';
        }
        progress.innerHTML = html;

        if (data.results.every(r => r.ok)) {
            document.getElementById('setupForm').classList.add('hidden');
            document.getElementById('doneBox').classList.remove('hidden');
            // Signal server to shut down after a short delay.
            setTimeout(() => fetch('/api/shutdown').catch(() => {}), 2000);
        }
    } catch (e) {
        progress.innerHTML = '<div class="progress-item err"><span class="icon">&#10007;</span> Connection error: ' + e.message + '</div>';
    }
    document.getElementById('submitBtn').disabled = false;
}
</script>
</body>
</html>
"""


class SetupHandler(BaseHTTPRequestHandler):
    """HTTP handler for the setup wizard Web UI."""

    def log_message(self, format, *args):
        # Suppress default access logs.
        pass

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/shutdown":
            self._json_response({"ok": True})
            # Schedule server shutdown.
            import threading
            threading.Thread(target=self._shutdown, daemon=True).start()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        if self.path == "/api/validate":
            provider = data.get("provider", "anthropic")
            api_key = data.get("api_key", "")
            endpoint = data.get("endpoint", "")
            valid, message = validate_api_key(provider, api_key, endpoint)
            self._json_response({"valid": valid, "message": message})

        elif self.path == "/api/setup":
            provider = data.get("provider", "anthropic")
            api_key = data.get("api_key", "")
            tg_token = data.get("telegram_token", "")
            tg_chat_id = data.get("telegram_chat_id", "")
            endpoint = data.get("endpoint", "")
            results = run_setup(provider, api_key, tg_token, tg_chat_id, endpoint)
            self._json_response({"results": results})

        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        content = _SETUP_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json_response(self, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _shutdown(self) -> None:
        import time
        time.sleep(1)
        self.server.shutdown()


def run_web(port: int = 8899) -> None:
    """Start the Web UI setup wizard."""
    url = f"http://localhost:{port}"
    print()
    print(f"  {_BOLD}=== Protea Setup (Web UI) ==={_RESET}")
    print()
    print(f"  Setup page: {_GREEN}{url}{_RESET}")
    print(f"  {_DIM}(opening browser...){_RESET}")
    print()
    print(f"  Press Ctrl+C to cancel.")
    print()

    server = ThreadingHTTPServer(("127.0.0.1", port), SetupHandler)
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    print()
    print(f"  {_GREEN}{_BOLD}Setup server stopped.{_RESET}")
    if (ROOT / ".env").exists():
        print(f"  {_GREEN}Configuration saved.{_RESET}")
        _print_start_instructions()
    else:
        print(f"  {_YELLOW}Setup was not completed.{_RESET}")


# ───────────────────────────────────────────────────────────────────
# Entry Point
# ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Protea Setup Wizard")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--web", action="store_true", help="Launch Web UI on port 8899")
    group.add_argument("--cli", action="store_true", help="Interactive CLI setup")
    parser.add_argument("--port", type=int, default=8899, help="Web UI port (default: 8899)")
    args = parser.parse_args()

    if args.web:
        run_web(port=args.port)
    else:
        run_cli()


if __name__ == "__main__":
    main()
