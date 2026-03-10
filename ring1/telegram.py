"""Telegram Bot API notifications — fire-and-forget, never raises.

Pure stdlib (urllib.request + json).
"""

from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger("protea.telegram")

_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Send messages via Telegram Bot API.  All methods are fire-and-forget."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._url = _API_BASE.format(token=bot_token)

    def set_chat_id(self, chat_id: str) -> None:
        """Update the chat ID (e.g. after auto-detection by the bot)."""
        self.chat_id = chat_id

    def send(self, message: str) -> bool:
        """Send a text message.  Returns True on success, False on any error."""
        if not self.chat_id:
            return False
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("ok", False)
        except Exception:
            log.debug("Telegram send failed", exc_info=True)
            return False

    def send_with_keyboard(self, text: str, buttons: list[list[dict]]) -> int | None:
        """Send message with inline keyboard. Returns message_id or None."""
        if not self.chat_id:
            return None
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "reply_markup": {
                "inline_keyboard": buttons,
            },
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if body.get("ok"):
                    return body.get("result", {}).get("message_id")
                return None
        except Exception:
            log.debug("Telegram send_with_keyboard failed", exc_info=True)
            return None

    def notify_error(self, context: int, error: str) -> bool:
        """Send an error notification."""
        msg = f"*Protea ERROR*\n```\n{error[:500]}\n```"
        return self.send(msg)

    def notify_sentinel_online(self, cycle: int = 0) -> bool:
        """Notify that the sentinel (Ring 0) has started/restarted."""
        msg = (
            f"🛡️ *哨兵程序已上线*\n\n"
            f"Ring 0 已启动\n"
            f"当前周期: {cycle}\n"
            f"监控状态: ✅ 运行中"
        )
        return self.send(msg)


def create_notifier(config) -> TelegramNotifier | None:
    """Create a TelegramNotifier from Ring1Config, or None if disabled.

    ``chat_id`` may be empty — ``send()`` will silently no-op until the bot
    auto-detects a chat and calls ``set_chat_id()``.
    """
    if not config.telegram_enabled:
        return None
    if not config.telegram_bot_token:
        log.warning("Telegram enabled but token missing — disabled")
        return None
    return TelegramNotifier(config.telegram_bot_token, config.telegram_chat_id)
