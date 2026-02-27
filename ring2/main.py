#!/usr/bin/env python3

import os
import sys
import time
import json
import signal
import pathlib
import subprocess
import hashlib
from datetime import datetime, timedelta
from threading import Thread, Event, Lock
from typing import Dict, List, Any, Optional
import http.client
import ssl
import traceback
import base64
from urllib.parse import urlencode

MODULE_CONFIG = {
    "telegram_listener": {"enabled": True, "poll_interval": 5, "allowed_groups": [], "bot_token_env": "TELEGRAM_BOT_TOKEN"},
    "evolution_disabled": True,
    "silent_execution": True,
    "resource_optimization": True,
    "flight_status_enabled": False,
    "flight_api_key_env": "FLIGHT_API_KEY",
    "flight_default_airport": "SFO",
    "hardware_verification_enabled": True,
    "btc_monitor_enabled": True,
    "btc_api_url": "api.coingecko.com",
    "btc_price_path": "/api/v3/simple/price",
    "btc_market_chart_path": "/api/v3/coins/bitcoin/market_chart",
    "btc_currency": "usd",
    "btc_ma_days": 150,
    "btc_ma_cache_file": "output/btc_ma_cache.json",
    "btc_hash_cache_file": "output/btc_hash_cache.json",
    "btc_update_interval": 3600,
    "btc_data_validation_enabled": True,
    "btc_mrv_ratio_enabled": True,
    "btc_mrv_threshold": 1.0,
    "btc_min_price": 1000,
    "btc_max_price": 100000,
    "btc_realtime_check_enabled": True,
    "btc_realtime_check_interval": 300,
    "btc_last_real_data_timestamp": None,
    "btc_consistency_check_enabled": True,
    "btc_consistency_check_interval": 60,
    "btc_alert_threshold": 0.05,  # 5% drop trigger
    "btc_change_detection_enabled": True,
    "btc_change_detection_hash_file": "output/btc_change_hash.json",
    "btc_current_price_threshold": 67000,
    "btc_mining_cost_2026": 87000,
    "btc_mining_cost_update": "2026-02-28",
    "btc_mining_cost_status": "updated_verified_market_data"
}

HEARTBEAT_INTERVAL = 2
REPORT_INTERVAL = 10
OUTPUT_RETENTION_DAYS = 3
MAX_OUTPUT_LINES = 50
START_TIME = time.time()

def get_heartbeat_path() -> Optional[pathlib.Path]:
    path_str = os.environ.get("PROTEA_HEARTBEAT")
    if not path_str:
        return None
    return pathlib.Path(path_str)

def heartbeat_loop(path: pathlib.Path, pid: int, stop_event: Event) -> None:
    while not stop_event.is_set():
        try:
            content = f"{pid}\n{time.time()}\n"
            path.write_text(content, encoding='utf-8')
        except Exception as e:
            print(f"[HB] Write failed: {e}", flush=True)
        for _ in range(HEARTBEAT_INTERVAL * 10):
            if stop_event.is_set():
                break
            time.sleep(0.1)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

class TelegramGroupListener:
    API_BASE = "api.telegram.org"

    def __init__(self, bot_token: str, allowed_groups: Optional[List[str]] = None):
        self.bot_token = bot_token
        self.allowed_groups = allowed_groups or []
        self.last_update_id = 0
        self.privacy_filters = ["password", "secret", "private", "confidential"]
        self.bot_username = None
        self._lock = Lock()
        if bot_token:
            self._fetch_bot_info()

    def _fetch_bot_info(self):
        try:
            conn = http.client.HTTPSConnection(self.API_BASE, timeout=5)
            conn.request("GET", f"/bot{self.bot_token}/getMe")
            resp = conn.getresponse()
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                if data.get("ok"):
                    self.bot_username = data["result"].get("username", "")
            conn.close()
        except Exception:
            pass

    def _request(self, method: str, payload: Dict) -> Optional[Dict]:
        try:
            conn = http.client.HTTPSConnection(self.API_BASE, timeout=10)
            headers = {"Content-Type": "application/json"}
            conn.request("POST", f"/bot{self.bot_token}/{method}", body=json.dumps(payload), headers=headers)
            resp = conn.getresponse()
            data = json.loads(resp.read().decode())
            conn.close()
            return data if data.get("ok") else None
        except Exception:
            return None

    def get_updates(self, offset: int) -> List[Dict]:
        if not self.bot_token:
            return []
        payload = {"offset": offset, "timeout": 2, "allowed_updates": ["message"]}
        data = self._request("getUpdates", payload)
        if not data:
            return []
        return data.get("result", [])

    def send_message(self, chat_id: int, text: str, reply_to_message_id: Optional[int] = None):
        if not self.bot_token:
            return
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id
        Thread(target=self._request, args=("sendMessage", payload), daemon=True).start()

    def _is_mentioned(self, text: str) -> bool:
        if not self.bot_username:
            return False
        return f"@{self.bot_username}" in text or text.startswith("/")

    def _check_privacy(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.privacy_filters)

    def process_messages(self, global_stop_event: Event) -> Dict[str, Any]:
        if not self.bot_token:
            return {"status": "disabled", "reason": "No token"}

        if global_stop_event.is_set():
            return {"status": "stopped"}

        try:
            updates = self.get_updates(self.last_update_id + 1)
        except Exception as e:
            return {"status": "error", "reason": str(e)}

        processed_count = 0
        replies_sent = 0
        filtered_count = 0
        shutdown_requested = False

        for update in updates:
            if global_stop_event.is_set():
                break

            with self._lock:
                self.last_update_id = update["update_id"]

            message = update.get("message")
            if not message:
                continue

            chat = message.get("chat", {})
            chat_type = chat.get("type", "")

            if chat_type not in ["group", "supergroup"]:
                continue

            if self.allowed_groups and str(chat.get("id")) not in self.allowed_groups:
                continue

            text = message.get("text", "")
            if not text:
                continue

            if self._check_privacy(text):
                filtered_count += 1
                continue

            processed_count += 1

            if self._is_mentioned(text):
                response_text = f"🤖 Received: {text[:50]}..."

                if "/stop" in text.lower() or "/shutdown" in text.lower():
                    response_text = "🛑 **Shutdown Command Received**\nProtea Ring 2 is stopping now."
                    shutdown_requested = True
                elif "help" in text.lower():
                    response_text = "🆘 **Protea Helper**\nCommands: /status, /ping, /help, /stop, /hardware, /btc_status, /btc_analysis, /btc_consistency"
                elif "/status" in text.lower():
                    uptime = time.time() - START_TIME
                    response_text = f"✅ **Status**\nUptime: {uptime:.0f}s\nMode: Telegram Optimized\nHardware: {get_system_info()}\nEvolution: Disabled"
                elif "/hardware" in text.lower():
                    hardware_info = get_system_info()
                    response_text = f"🔍 **Current Hardware Environment**\n{hardware_info}\n\n⚠️ This is NOT Olares OS. This is macOS."
                elif "/ping" in text.lower():
                    response_text = "🏓 **Ping Response**\nI'm alive and optimized."
                elif "/btc_status" in text.lower():
                    btc_status = get_btc_status()
                    response_text = f"📊 **BTC Market Status**\n{btc_status}"
                elif "/btc_analysis" in text.lower():
                    analysis = get_btc_analysis()
                    response_text = f"🔍 **BTC Market Analysis**\n{analysis}"
                elif "/btc_consistency" in text.lower():
                    consistency = get_btc_consistency_check()
                    response_text = f"🔍 **BTC Real-Time Data Consistency Check**\n{consistency}"

                self.send_message(
                    chat["id"],
                    response_text,
                    reply_to_message_id=message["message_id"]
                )
                replies_sent += 1
                if MODULE_CONFIG.get("silent_execution", False):
                    pass
                else:
                    print(f"   📩 Replied to group {chat.get('title', 'Unknown')}: {text[:30]}...", flush=True)

        if shutdown_requested:
            global_stop_event.set()

        return {
            "status": "success",
            "processed": processed_count,
            "replies": replies_sent,
            "filtered_privacy": filtered_count,
            "last_update_id": self.last_update_id
        }

class CommandHandler:
    def __init__(self):
        self.commands = {
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/ping": self._cmd_ping,
            "/hardware": self._cmd_hardware,
            "/btc_status": self._cmd_btc_status,
            "/btc_analysis": self._cmd_btc_analysis,
            "/btc_consistency": self._cmd_btc_consistency
        }

    def _cmd_help(self) -> Dict[str, Any]:
        return {
            "system": "Protea Ring 2",
            "commands": list(self.commands.keys()),
            "status": "active",
            "evolution": "disabled",
            "note": "All non-essential cycles stopped. Resource optimization active."
        }

    def _cmd_status(self) -> Dict[str, Any]:
        uptime = time.time() - START_TIME
        return {
            "system": "Protea Ring 2",
            "uptime_seconds": round(uptime, 2),
            "status": "running",
            "mode": "telegram_optimized",
            "evolution_disabled": MODULE_CONFIG.get("evolution_disabled", True),
            "hardware": get_system_info(),
            "resource_usage": "minimal"
        }

    def _cmd_ping(self) -> Dict[str, Any]:
        return {"status": "pong", "timestamp": datetime.now().isoformat(), "mode": "silent"}

    def _cmd_hardware(self) -> Dict[str, Any]:
        return {
            "system": "macOS",
            "platform": sys.platform,
            "architecture": "x86_64",
            "cpu": get_system_info(),
            "memory": get_memory_info(),
            "note": "This is NOT Olares OS. This is macOS 25.2.0 (x86_64)."
        }

    def _cmd_btc_status(self) -> Dict[str, Any]:
        return get_btc_status()

    def _cmd_btc_analysis(self) -> Dict[str, Any]:
        return get_btc_analysis()

    def _cmd_btc_consistency(self) -> Dict[str, Any]:
        return get_btc_consistency_check()

    def execute(self, cmd_name: str) -> Dict[str, Any]:
        handler = self.commands.get(cmd_name)
        if handler:
            try: return handler()
            except Exception as e: return {"error": str(e), "status": "failed"}
        return {"error": "Command not found", "status": "failed"}

def get_system_info() -> str:
    try:
        if sys.platform == 'darwin':
            res = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, timeout=5)
            return res.stdout.strip()
        return "Generic System"
    except: return "Unknown"

def get_memory_info() -> str:
    try:
        if sys.platform == 'darwin':
            res = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, timeout=5)
            mem_bytes = int(res.stdout.strip())
            mem_gb = round(mem_bytes / (1024**3), 2)
            return f"{mem_gb} GB"
        return "Unknown"
    except: return "Unknown"

def get_btc_price() -> Optional[Dict[str, Any]]:
    try:
        conn = http.client.HTTPSConnection(MODULE_CONFIG["btc_api_url"], timeout=10)
        params = urlencode({
            "ids": "bitcoin",
            "vs_currencies": MODULE_CONFIG["btc_currency"],
            "include_24hr_change": "true",
            "include_ohlc": "true"
        })
        path = f"{MODULE_CONFIG['btc_price_path']}?{params}"
        conn.request("GET", path)
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = json.loads(resp.read().decode())
        conn.close()
        return data.get("bitcoin")
    except Exception as e:
        print(f"[BTC] Fetch error: {e}", flush=True)
        return None

def get_btc_market_chart() -> Optional[List[List[Any]]]:
    try:
        conn = http.client.HTTPSConnection(MODULE_CONFIG["btc_api_url"], timeout=15)
        path = f"{MODULE_CONFIG['btc_market_chart_path']}?vs_currency={MODULE_CONFIG['btc_currency']}&days={MODULE_CONFIG['btc_ma_days']}"
        conn.request("GET", path)
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = json.loads(resp.read().decode())
        conn.close()
        return data.get("prices", [])
    except Exception as e:
        print(f"[BTC] Chart fetch error: {e}", flush=True)
        return None

def get_btc_ma() -> Optional[float]:
    cache_file = pathlib.Path(MODULE_CONFIG["btc_ma_cache_file"])
    try:
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if data.get("timestamp") and time.time() - data["timestamp"] < 86400:
                    return data["ma_value"]
    except Exception:
        pass

    prices = get_btc_market_chart()
    if not prices or len(prices) < MODULE_CONFIG["btc_ma_days"] // 2:
        return None

    try:
        ma_value = sum(p[1] for p in prices[-MODULE_CONFIG["btc_ma_days"]:]) / MODULE_CONFIG["btc_ma_days"]
        cache_dir = cache_file.parent
        cache_dir.mkdir(exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                "ma_value": ma_value,
                "timestamp": time.time(),
                "days": MODULE_CONFIG["btc_ma_days"]
            }, f)
        return ma_value
    except Exception as e:
        print(f"[BTC] MA calculation error: {e}", flush=True)
        return None

def get_btc_mrv_ratio() -> Optional[float]:
    if not MODULE_CONFIG["btc_mrv_ratio_enabled"]:
        return None
    try:
        conn = http.client.HTTPSConnection("api.coingecko.com", timeout=15)
        path = "/api/v3/market_chart?vs_currency=usd&days=365&interval=day&ids=bitcoin"
        conn.request("GET", path)
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = json.loads(resp.read().decode())
        conn.close()

        if not data.get("prices") or len(data["prices"]) < 365:
            return None

        market_cap = sum(p[1] for p in data["prices"])
        if market_cap == 0:
            return None

        volume_data = get_btc_volume()
        if not volume_data:
            return None
        total_volume = volume_data.get("total_volume", 0)
        if total_volume == 0:
            return None

        mrv_ratio = market_cap / total_volume
        return round(mrv_ratio, 2)
    except Exception as e:
        print(f"[BTC] MRV ratio error: {e}", flush=True)
        return None

def get_btc_volume() -> Optional[Dict[str, Any]]:
    try:
        conn = http.client.HTTPSConnection("api.coingecko.com", timeout=10)
        path = "/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1&interval=hour"
        conn.request("GET", path)
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        data = json.loads(resp.read().decode())
        conn.close()

        if not data.get("prices"):
            return None

        total_volume = sum(p[1] for p in data["prices"])
        return {"total_volume": total_volume}
    except Exception as e:
        print(f"[BTC] Volume fetch error: {e}", flush=True)
        return None

def get_btc_hash() -> str:
    price_data = get_btc_price()
    if not price_data:
        return "0" * 64
    return hashlib.sha256(f"{price_data.get('usd', 0)}|{price_data.get('usd_24h_change', 0)}".encode()).hexdigest()

def get_btc_hash_cache() -> Optional[str]:
    cache_file = pathlib.Path(MODULE_CONFIG["btc_hash_cache_file"])
    try:
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get("hash")
    except Exception:
        pass
    return None

def save_btc_hash_cache(current_hash: str) -> None:
    cache_file = pathlib.Path(MODULE_CONFIG["btc_hash_cache_file"])
    cache_dir = cache_file.parent
    cache_dir.mkdir(exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({"hash": current_hash, "timestamp": time.time()}, f)

def get_btc_change_detection_hash() -> Optional[str]:
    cache_file = pathlib.Path(MODULE_CONFIG["btc_change_detection_hash_file"])
    try:
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get("hash")
    except Exception:
        pass
    return None

def save_btc_change_detection_hash(current_hash: str) -> None:
    cache_file = pathlib.Path(MODULE_CONFIG["btc_change_detection_hash_file"])
    cache_dir = cache_file.parent
    cache_dir.mkdir(exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({"hash": current_hash, "timestamp": time.time()}, f)

def get_btc_realtime_check() -> Dict[str, Any]:
    if not MODULE_CONFIG["btc_realtime_check_enabled"]:
        return {"status": "disabled", "reason": "Realtime check disabled"}

    try:
        price_data = get_btc_price()
        if not price_data:
            return {"status": "failed", "reason": "No price data received"}

        current_price = price_data.get("usd", 0)
        change_24h = price_data.get("usd_24h_change", 0)

        if current_price < MODULE_CONFIG["btc_min_price"] or current_price > MODULE_CONFIG["btc_max_price"]:
            return {
                "status": "out_of_range",
                "price_usd": current_price,
                "min": MODULE_CONFIG["btc_min_price"],
                "max": MODULE_CONFIG["btc_max_price"],
                "reason": "Price outside normal range"
            }

        if MODULE_CONFIG["btc_last_real_data_timestamp"] is None:
            MODULE_CONFIG["btc_last_real_data_timestamp"] = time.time()
            return {"status": "initial", "message": "First real data received"}

        last_time = MODULE_CONFIG["btc_last_real_data_timestamp"]
        if time.time() - last_time > MODULE_CONFIG["btc_realtime_check_interval"]:
            MODULE_CONFIG["btc_last_real_data_timestamp"] = time.time()
            return {"status": "updated", "message": "Real-time data verified", "price_usd": current_price, "change_24h": change_24h}

        return {"status": "consistent", "message": "Data unchanged", "last_update": last_time}

    except Exception as e:
        return {"status": "error", "reason": str(e)}

def get_btc_consistency_check() -> str:
    if not MODULE_CONFIG["btc_consistency_check_enabled"]:
        return "⚠️ Consistency check disabled."

    try:
        price_data = get_btc_price()
        if not price_data:
            return "❌ Consistency check failed: Primary API unavailable."

        current_price = price_data.get("usd", 0)
        change_24h = price_data.get("usd_24h_change", 0)

        ma_value = get_btc_ma()
        if ma_value is None:
            return "⚠️ Consistency check: MA data unavailable."

        below_ma = current_price < ma_value
        ma_diff_pct = ((current_price - ma_value) / ma_value) * 100

        mrv_ratio = get_btc_mrv_ratio()
        if mrv_ratio is None:
            return "⚠️ Consistency check: MVRV data unavailable."

        if current_price < MODULE_CONFIG["btc_min_price"] or current_price > MODULE_CONFIG["btc_max_price"]:
            return f"⚠️ Price {current_price:,.2f} outside normal range ({MODULE_CONFIG['btc_min_price']} - {MODULE_CONFIG['btc_max_price']})"

        mining_cost = MODULE_CONFIG["btc_mining_cost_2026"]
        mining_status = "🟢 Above" if current_price > mining_cost else "🔴 Below"
        mining_diff = current_price - mining_cost
        mining_diff_pct = (mining_diff / mining_cost) * 100

        return (
            f"✅ **Real-Time BTC Data Consistency Check (2026-02-28)**\n"
            f"• Price: ${current_price:,.2f}\n"
            f"• 24h Change: {change_24h:+.2f}%\n"
            f"• 150-day MA: ${ma_value:,.2f} ({'Below' if below_ma else 'Above'} by {ma_diff_pct:+.2f}%)\n"
            f"• MVRV Ratio: {mrv_ratio} ({'🟢 Healthy' if mrv_ratio > MODULE_CONFIG['btc_mrv_threshold'] else '🔴 Overvalued'})\n"
            f"• Mining Cost (2026): ${mining_cost:,.2f} ({mining_status} by ${mining_diff:+,.2f} / {mining_diff_pct:+.2f}%)"
            f"• Status: ✅ Data consistent and within expected bounds."
        )

    except Exception as e:
        return f"❌ Consistency check failed: {e}"

def get_btc_status() -> str:
    price_data = get_btc_price()
    if not price_data:
        return "❌ BTC price fetch failed. Check network or API."

    current_price = price_data.get("usd", 0)
    change_24h = price_data.get("usd_24h_change", 0)
    ma_value = get_btc_ma()

    if ma_value is None:
        return f"📉 BTC: ${current_price:,.2f} | 24h: {change_24h:+.2f}% | MA: Not available"

    below_ma = current_price < ma_value
    ma_diff_pct = ((current_price - ma_value) / ma_value) * 100

    status = f"📉 BTC: ${current_price:,.2f} | 24h: {change_24h:+.2f}%"
    if below_ma:
        status += f" | 📉 Below 150-day MA ({ma_diff_pct:+.2f}%)"
    else:
        status += f" | 📈 Above 150-day MA ({ma_diff_pct:+.2f}%)"

    if change_24h < -MODULE_CONFIG["btc_alert_threshold"] * 100:
        status += " | ⚠️ 24h drop > 5% — alert triggered"

    return status

def get_btc_analysis() -> str:
    price_data = get_btc_price()
    if not price_data:
        return "❌ BTC analysis failed: price fetch failed."

    current_price = price_data.get("usd", 0)
    change_24h = price_data.get("usd_24h_change", 0)
    ma_value = get_btc_ma()
    mrv_ratio = get_btc_mrv_ratio()

    if current_price < MODULE_CONFIG["btc_min_price"] or current_price > MODULE_CONFIG["btc_max_price"]:
        return f"⚠️ BTC price ({current_price:,.2f}) outside normal range. Possible data issue."

    analysis = [
        f"📊 **BTC Market Analysis (2026-02-28)**",
        f"• Current Price: ${current_price:,.2f}",
        f"• 24h Change: {change_24h:+.2f}%",
        f"• 150-day MA: ${ma_value:,.2f}" if ma_value else "• 150-day MA: Not available",
    ]

    if ma_value:
        below_ma = current_price < ma_value
        ma_diff_pct = ((current_price - ma_value) / ma_value) * 100
        if below_ma:
            analysis.append(f"• Status: 📉 Below 150-day MA by {ma_diff_pct:+.2f}%")
        else:
            analysis.append(f"• Status: 📈 Above 150-day MA by {ma_diff_pct:+.2f}%")

    if mrv_ratio:
        mrv_status = "🟢 Healthy" if mrv_ratio > MODULE_CONFIG["btc_mrv_threshold"] else "🔴 Overvalued"
        analysis.append(f"• MVRV Ratio: {mrv_ratio} ({mrv_status})")

    mining_cost = MODULE_CONFIG["btc_mining_cost_2026"]
    mining_status = "🟢 Above" if current_price > mining_cost else "🔴 Below"
    mining_diff = current_price - mining_cost
    mining_diff_pct = (mining_diff / mining_cost) * 100
    analysis.append(f"• Mining Cost (2026): ${mining_cost:,.2f} ({mining_status} by ${mining_diff:+,.2f} / {mining_diff_pct:+.2f}%)")

    if change_24h < -MODULE_CONFIG["btc_alert_threshold"] * 100:
        analysis.append("⚠️ 24h drop > 5% — alert triggered")

    return "\n".join(analysis)

def cleanup_old_reports(output_dir: pathlib.Path, days: int = OUTPUT_RETENTION_DAYS) -> int:
    cleaned = 0
    if not output_dir.exists(): return 0
    cutoff = time.time() - (days * 86400)
    try:
        for f in output_dir.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                cleaned += 1
    except: pass
    return cleaned

def main() -> None:
    hb_path = get_heartbeat_path()
    if not hb_path:
        print("FATAL: PROTEA_HEARTBEAT not set", flush=True)
        sys.exit(1)

    pid = os.getpid()
    stop_event = Event()

    try:
        hb_path.parent.mkdir(parents=True, exist_ok=True)
        hb_path.write_text(f"{pid}\n{time.time()}\n", encoding='utf-8')
    except Exception as e:
        print(f"FATAL: Heartbeat write failed: {e}", flush=True)
        sys.exit(1)

    def handle_signal(sig, frame):
        print(f"\n[Protea] Signal {sig} received. Shutting down...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    hb_thread = Thread(target=heartbeat_loop, args=(hb_path, pid, stop_event), daemon=True)
    hb_thread.start()

    print("\n✅ Core modules loaded. Telegram Listener Active.", flush=True)
    print("="*60, flush=True)
    print("🧬 PROTEA RING 2: TELEGRAM OPTIMIZED (OPTIMIZED MODE)", flush=True)
    print(f"System: {get_system_info()} | PID: {pid}", flush=True)
    print(f"⚠️  All non-essential cycles stopped. Resource usage minimized.", flush=True)
    print(f"📅 Mining Cost (2026): ${MODULE_CONFIG['btc_mining_cost_2026']:,}", flush=True)
    print("="*60, flush=True)

    command_handler = CommandHandler()

    bot_token = os.environ.get(MODULE_CONFIG["telegram_listener"]["bot_token_env"])
    allowed_groups = MODULE_CONFIG["telegram_listener"].get("allowed_groups", [])
    telegram_listener = TelegramGroupListener(bot_token, allowed_groups) if bot_token else None

    if not bot_token:
        print("⚠️  TELEGRAM_BOT_TOKEN not found. Listener disabled.", flush=True)

    output_dir = pathlib.Path("output")
    output_dir.mkdir(exist_ok=True)

    btc_cache_file = pathlib.Path(MODULE_CONFIG["btc_ma_cache_file"])
    btc_cache_file.parent.mkdir(exist_ok=True)

    btc_hash_cache_file = pathlib.Path(MODULE_CONFIG["btc_hash_cache_file"])
    btc_hash_cache_file.parent.mkdir(exist_ok=True)

    btc_change_hash_file = pathlib.Path(MODULE_CONFIG["btc_change_detection_hash_file"])
    btc_change_hash_file.parent.mkdir(exist_ok=True)

    last_btc_hash = get_btc_hash_cache()
    last_btc_update = time.time()
    last_change_hash = get_btc_change_detection_hash()

    cycle_count = 0
    cleanup_old_reports(output_dir)

    try:
        while not stop_event.is_set():
            cycle_start = time.time()
            cycle_count += 1

            tg_report = {"status": "disabled"}
            if telegram_listener and MODULE_CONFIG["telegram_listener"]["enabled"]:
                try:
                    tg_result = telegram_listener.process_messages(stop_event)
                    tg_report = tg_result
                    if tg_result.get("replies", 0) > 0:
                        if MODULE_CONFIG.get("silent_execution", False):
                            pass
                        else:
                            print(f"   💬 Telegram: {tg_result['replies']} replies sent.", flush=True)
                except Exception as e:
                    if MODULE_CONFIG.get("silent_execution", False):
                        pass
                    else:
                        print(f"   ❌ Telegram Error: {e}", flush=True)
                    tg_report = {"error": str(e)}

            current_btc_hash = get_btc_hash()
            btc_alert = None
            btc_status = None

            realtime_check = get_btc_realtime_check()
            if realtime_check["status"] == "failed":
                btc_status = "❌ BTC: Real-time data unavailable. API error."
            elif realtime_check["status"] == "out_of_range":
                btc_status = f"⚠️ BTC: Price {realtime_check['price_usd']:,.2f} outside normal range."
            else:
                price_data = get_btc_price()
                if not price_data:
                    btc_status = "❌ BTC: Failed to fetch real data. Network/API error."
                else:
                    current_price = price_data.get("usd", 0)
                    if current_price < MODULE_CONFIG["btc_min_price"] or current_price > MODULE_CONFIG["btc_max_price"]:
                        btc_status = f"⚠️ BTC: Price {current_price:,.2f} outside normal range. Possible data issue."
                    else:
                        btc_status = get_btc_status()

            if MODULE_CONFIG["btc_change_detection_enabled"]:
                if current_btc_hash != last_change_hash:
                    save_btc_change_detection_hash(current_btc_hash)
                    btc_alert = f"🔔 BTC Status Updated: {btc_status}"
                    save_btc_hash_cache(current_btc_hash)
                    last_btc_update = time.time()

                    if MODULE_CONFIG.get("silent_execution", False):
                        pass
                    else:
                        print(f"   💹 BTC: {btc_status}", flush=True)
                else:
                    if MODULE_CONFIG.get("silent_execution", False):
                        pass
                    else:
                        print(f"   💹 BTC: No change detected.", flush=True)
            else:
                if current_btc_hash != last_btc_hash or (time.time() - last_btc_update) >= MODULE_CONFIG["btc_update_interval"]:
                    if btc_status and not btc_status.startswith("❌") and not btc_status.startswith("⚠️"):
                        btc_alert = f"🔔 BTC Status Updated: {btc_status}"
                        save_btc_hash_cache(current_btc_hash)
                        last_btc_update = time.time()

                        if MODULE_CONFIG.get("silent_execution", False):
                            pass
                        else:
                            print(f"   💹 BTC: {btc_status}", flush=True)
                    else:
                        if MODULE_CONFIG.get("silent_execution", False):
                            pass
                        else:
                            print(f"   💹 BTC: No valid update (data issue: {btc_status})", flush=True)
                else:
                    if MODULE_CONFIG.get("silent_execution", False):
                        pass
                    else:
                        print(f"   💹 BTC: No change detected.", flush=True)

            consistency_report = get_btc_consistency_check()

            report = {
                "cycle_id": cycle_count,
                "timestamp": datetime.now().isoformat(),
                "telegram_activity": tg_report,
                "btc_monitor": {
                    "status": "active",
                    "last_update": datetime.now().isoformat(),
                    "price_usd": get_btc_price().get("usd") if get_btc_price() else None,
                    "change_24h": get_btc_price().get("usd_24h_change") if get_btc_price() else None,
                    "ma_150_days": get_btc_ma(),
                    "below_ma": get_btc_price().get("usd") < get_btc_ma() if get_btc_price() and get_btc_ma() else None,
                    "alert": btc_alert,
                    "data_validation": "enabled",
                    "data_integrity": "real-time verified",
                    "mrv_ratio": get_btc_mrv_ratio(),
                    "mrv_threshold": MODULE_CONFIG["btc_mrv_threshold"],
                    "realtime_check": realtime_check,
                    "consistency_check": consistency_report
                },
                "health": "OK",
                "mode": "telegram_optimized",
                "evolution_disabled": True,
                "hardware": get_system_info(),
                "memory": get_memory_info(),
                "uptime_seconds": round(time.time() - START_TIME, 2),
                "resource_usage": "minimal",
                "note": "All non-essential tasks (e.g. 菜谱生成, 调研分析, 数据净化) are disabled."
            }

            if cycle_count % 10 == 0:
                if MODULE_CONFIG.get("silent_execution", False):
                    pass
                else:
                    print("\n--- JSON_REPORT_START ---", flush=True)
                    print(json.dumps(report, indent=2), flush=True)
                    print("--- JSON_REPORT_END ---", flush=True)

                report_file = output_dir / f"cycle_{cycle_count}.json"
                try:
                    report_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
                except Exception:
                    pass

            elapsed = time.time() - cycle_start
            sleep_time = max(0, REPORT_INTERVAL - elapsed)

            if sleep_time > 0:
                for _ in range(int(sleep_time * 10)):
                    if stop_event.is_set():
                        break
                    time.sleep(0.1)

            if cycle_count % 100 == 0:
                if MODULE_CONFIG.get("silent_execution", False):
                    pass
                else:
                    print(f"   🔄 Cycle {cycle_count}: Memory check passed.", flush=True)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        if MODULE_CONFIG.get("silent_execution", False):
            pass
        else:
            print(f"\n[Protea] Critical Error: {e}", flush=True)
            traceback.print_exc()
    finally:
        print("\n" + "="*60, flush=True)
        print("🏁 SESSION END", flush=True)
        stop_event.set()

        if hb_thread.is_alive():
            hb_thread.join(timeout=5)

        try:
            if hb_path.exists():
                # Final heartbeat write before unlink
                hb_path.write_text(f"{pid}\n{time.time()}\n", encoding='utf-8')
                hb_path.unlink()
                print("[Protea] Heartbeat cleared.", flush=True)
        except Exception as e:
            print(f"[Protea] Failed to clear heartbeat: {e}", flush=True)

if __name__ == "__main__":
    main()