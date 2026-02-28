#!/usr/bin/env python3

import os
import pathlib
import time
import json
import sys
from threading import Thread, Event
from datetime import datetime, timezone
import signal
import subprocess
import platform
import urllib.request
import urllib.error
import urllib.parse

HEARTBEAT_INTERVAL = 2
OUTPUT_DIR = pathlib.Path("output/system_insights")
TELEGRAM_CONTEXT_FILE = pathlib.Path("output/telegram_context.json")
ROUTING_LOG_FILE = pathlib.Path("output/telegram_routing_log.json")
TELEGRAM_CONFIG_FILE = pathlib.Path("output/telegram_config.json")

def heartbeat_loop(heartbeat_path: pathlib.Path, pid: int, stop_event: Event) -> None:
    while not stop_event.is_set():
        try:
            heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
        except Exception:
            pass
        time.sleep(HEARTBEAT_INTERVAL)

def safe_json_dump(obj, indent: int = 2) -> str:
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            return json.dumps(obj, indent=indent, ensure_ascii=True)
        except:
            return str(obj)

def load_telegram_config() -> dict:
    try:
        if TELEGRAM_CONFIG_FILE.exists():
            return json.loads(TELEGRAM_CONFIG_FILE.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {"bot_token": None, "enabled": False}

def load_telegram_context() -> dict:
    try:
        if TELEGRAM_CONTEXT_FILE.exists():
            data = json.loads(TELEGRAM_CONTEXT_FILE.read_text(encoding='utf-8'))
            return data
    except Exception:
        pass
    return {"chat_id": None, "chat_type": "unknown", "last_update": None}

def save_telegram_context(chat_id: int, chat_type: str = "group") -> None:
    try:
        TELEGRAM_CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        context = {
            "chat_id": chat_id,
            "chat_type": chat_type,
            "last_update": datetime.now(timezone.utc).isoformat()
        }
        TELEGRAM_CONTEXT_FILE.write_text(safe_json_dump(context), encoding='utf-8')
    except Exception:
        pass

def log_routing_decision(action: str, chat_id: int, chat_type: str, reason: str, file_path: str = None, success: bool = True) -> None:
    try:
        ROUTING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        logs = []
        if ROUTING_LOG_FILE.exists():
            try:
                logs = json.loads(ROUTING_LOG_FILE.read_text(encoding='utf-8'))
            except:
                pass

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "chat_id": chat_id,
            "chat_type": chat_type,
            "reason": reason,
            "file_path": file_path,
            "success": success
        }

        logs.append(log_entry)

        if len(logs) > 100:
            logs = logs[-100:]

        ROUTING_LOG_FILE.write_text(safe_json_dump(logs), encoding='utf-8')
    except Exception:
        pass

def send_telegram_message(bot_token: str, chat_id: int, text: str, parse_mode: str = "Markdown") -> dict:
    """Send text message to Telegram chat using pure stdlib."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Protea/Ring2'
            }
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            return {"success": True, "result": result}
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        return {"success": False, "error": f"HTTP {e.code}: {error_body}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def send_telegram_document(bot_token: str, chat_id: int, file_path: pathlib.Path, caption: str = None) -> dict:
    """Send document to Telegram chat using pure stdlib multipart/form-data."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        
        boundary = f"----ProteaBoundary{int(time.time() * 1000)}"
        
        body = []
        
        # Add chat_id field
        body.append(f'--{boundary}'.encode())
        body.append(b'Content-Disposition: form-data; name="chat_id"')
        body.append(b'')
        body.append(str(chat_id).encode())
        
        # Add caption if provided
        if caption:
            body.append(f'--{boundary}'.encode())
            body.append(b'Content-Disposition: form-data; name="caption"')
            body.append(b'')
            body.append(caption.encode('utf-8'))
        
        # Add file
        body.append(f'--{boundary}'.encode())
        body.append(f'Content-Disposition: form-data; name="document"; filename="{file_path.name}"'.encode())
        body.append(b'Content-Type: application/octet-stream')
        body.append(b'')
        body.append(file_path.read_bytes())
        
        body.append(f'--{boundary}--'.encode())
        body.append(b'')
        
        body_bytes = b'\r\n'.join(body)
        
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                'Content-Type': f'multipart/form-data; boundary={boundary}',
                'User-Agent': 'Protea/Ring2'
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return {"success": True, "result": result}
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        return {"success": False, "error": f"HTTP {e.code}: {error_body}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def route_and_send_message(text: str, chat_id: int = None, chat_type: str = None) -> dict:
    """Route message to appropriate chat based on context."""
    config = load_telegram_config()
    
    if not config.get("enabled") or not config.get("bot_token"):
        return {"success": False, "error": "telegram_not_configured"}
    
    context = load_telegram_context()
    
    target_chat_id = chat_id or context.get("chat_id")
    target_chat_type = chat_type or context.get("chat_type", "unknown")
    
    if not target_chat_id:
        return {"success": False, "error": "no_target_chat"}
    
    result = send_telegram_message(config["bot_token"], target_chat_id, text)
    
    log_routing_decision(
        action="message_sent",
        chat_id=target_chat_id,
        chat_type=target_chat_type,
        reason="context_routing",
        success=result["success"]
    )
    
    return result

def route_and_send_file(file_path: pathlib.Path, caption: str = None, chat_id: int = None, chat_type: str = None) -> dict:
    """Route file to appropriate chat based on context."""
    config = load_telegram_config()
    
    if not config.get("enabled") or not config.get("bot_token"):
        return {"success": False, "error": "telegram_not_configured"}
    
    context = load_telegram_context()
    
    target_chat_id = chat_id or context.get("chat_id")
    target_chat_type = chat_type or context.get("chat_type", "unknown")
    
    if not target_chat_id:
        return {"success": False, "error": "no_target_chat"}
    
    if not file_path.exists():
        return {"success": False, "error": "file_not_found"}
    
    result = send_telegram_document(config["bot_token"], target_chat_id, file_path, caption)
    
    log_routing_decision(
        action="file_sent",
        chat_id=target_chat_id,
        chat_type=target_chat_type,
        reason="context_routing",
        file_path=str(file_path),
        success=result["success"]
    )
    
    return result

def analyze_routing_intent(user_message: str = None) -> dict:
    intent = {
        "detected": False,
        "target": "current_chat",
        "keywords_found": []
    }

    if not user_message:
        return intent

    msg_lower = user_message.lower()

    group_keywords = ["在这个群", "发这里", "群里", "发到群", "就在这", "这里发", "发在群里"]
    private_keywords = ["发给我", "私聊", "单独发", "dm我"]

    for kw in group_keywords:
        if kw in msg_lower:
            intent["detected"] = True
            intent["target"] = "group"
            intent["keywords_found"].append(kw)

    for kw in private_keywords:
        if kw in msg_lower:
            intent["detected"] = True
            intent["target"] = "private"
            intent["keywords_found"].append(kw)

    return intent

def validate_routing_context(current_chat_id: int, current_chat_type: str, intent_target: str) -> dict:
    validation = {
        "valid": True,
        "warnings": [],
        "recommended_chat_id": current_chat_id,
        "recommended_chat_type": current_chat_type
    }

    if intent_target == "group" and current_chat_type == "private":
        validation["valid"] = False
        validation["warnings"].append("用户要求发到群里，但当前对话是私聊")

    if intent_target == "private" and current_chat_type == "group":
        validation["valid"] = False
        validation["warnings"].append("用户要求私聊发送，但当前对话在群组")
        validation["recommended_chat_type"] = "private"

    return validation

def get_system_metrics() -> dict:
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version()
    }

    try:
        if platform.system() == "Darwin":
            vm_result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
            if vm_result.returncode == 0:
                vm_lines = vm_result.stdout.strip().split('\n')
                vm_data = {}
                for line in vm_lines[1:]:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        vm_data[key.strip()] = val.strip().rstrip('.')
                metrics["memory"] = vm_data

            load_result = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True, timeout=2)
            if load_result.returncode == 0:
                metrics["load_average"] = load_result.stdout.strip()

            cpu_result = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True, timeout=2)
            if cpu_result.returncode == 0:
                metrics["cpu_count"] = int(cpu_result.stdout.strip())

        elif platform.system() == "Linux":
            load_result = subprocess.run(["uptime"], capture_output=True, text=True, timeout=2)
            if load_result.returncode == 0:
                metrics["uptime"] = load_result.stdout.strip()

            free_result = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=2)
            if free_result.returncode == 0:
                metrics["memory_info"] = free_result.stdout.strip()

    except Exception as e:
        metrics["error"] = str(e)

    return metrics

def get_network_connections() -> list:
    connections = []

    try:
        if platform.system() == "Darwin":
            result = subprocess.run(["netstat", "-an"], capture_output=True, text=True, timeout=5)
        else:
            result = subprocess.run(["netstat", "-tuln"], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 4:
                    connections.append({
                        "proto": parts[0],
                        "local_address": parts[3] if len(parts) > 3 else "unknown",
                        "state": parts[-1] if len(parts) > 4 else "unknown"
                    })
    except Exception:
        pass

    return connections[:20]

def get_running_processes() -> list:
    processes = []

    try:
        if platform.system() == "Darwin":
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
        else:
            result = subprocess.run(["ps", "aux", "--sort=-%cpu"], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:21]:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    processes.append({
                        "user": parts[0],
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "command": parts[10][:100]
                    })
    except Exception:
        pass

    return processes

def analyze_disk_usage() -> dict:
    disk_info = {}

    try:
        result = subprocess.run(["df", "-h"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            filesystems = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 6:
                    filesystems.append({
                        "filesystem": parts[0],
                        "size": parts[1],
                        "used": parts[2],
                        "available": parts[3],
                        "use_percent": parts[4],
                        "mounted_on": parts[5]
                    })
            disk_info["filesystems"] = filesystems
    except Exception:
        pass

    return disk_info

def generate_system_report() -> dict:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_type": "system_insights",
        "metrics": get_system_metrics(),
        "processes": get_running_processes(),
        "network": get_network_connections(),
        "disk": analyze_disk_usage()
    }

    report["summary"] = {
        "active_processes": len(report["processes"]),
        "network_connections": len(report["network"]),
        "filesystems_monitored": len(report["disk"].get("filesystems", [])),
        "platform": platform.system(),
        "architecture": platform.machine()
    }

    return report

def generate_routing_guidance(telegram_ctx: dict) -> dict:
    guidance = {
        "routing_active": telegram_ctx.get("chat_id") is not None,
        "target_chat_id": telegram_ctx.get("chat_id"),
        "target_chat_type": telegram_ctx.get("chat_type"),
        "routing_rules": []
    }

    if guidance["routing_active"]:
        if telegram_ctx.get("chat_type") == "group":
            guidance["routing_rules"].append("所有文件和报告将发送到群组")
            guidance["routing_rules"].append(f"目标群组 ID: {telegram_ctx['chat_id']}")
        else:
            guidance["routing_rules"].append("所有文件和报告将私聊发送")
            guidance["routing_rules"].append(f"目标用户 ID: {telegram_ctx['chat_id']}")

        log_routing_decision(
            action="guidance_generated",
            chat_id=telegram_ctx["chat_id"],
            chat_type=telegram_ctx["chat_type"],
            reason="active_context_loaded"
        )

    return guidance

def main() -> None:
    heartbeat_path_str = os.environ.get("PROTEA_HEARTBEAT")
    if not heartbeat_path_str:
        print(safe_json_dump({"error": "PROTEA_HEARTBEAT not set"}))
        return

    heartbeat_path = pathlib.Path(heartbeat_path_str)
    pid = os.getpid()

    try:
        heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
    except Exception as e:
        print(safe_json_dump({"error": "heartbeat_init_failed", "details": str(e)}))
        return

    stop_event = Event()

    def signal_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    heartbeat_thread = Thread(target=heartbeat_loop, args=(heartbeat_path, pid, stop_event), daemon=True)
    heartbeat_thread.start()

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    telegram_context = load_telegram_context()
    telegram_config = load_telegram_config()

    print(safe_json_dump({
        "system": "system_insights_monitor",
        "version": "3.0_telegram_routing",
        "pid": pid,
        "status": "initialized",
        "platform": platform.system(),
        "telegram_context": {
            "chat_id": telegram_context.get("chat_id"),
            "chat_type": telegram_context.get("chat_type"),
            "context_loaded": telegram_context.get("chat_id") is not None
        },
        "telegram_config": {
            "enabled": telegram_config.get("enabled", False),
            "bot_configured": telegram_config.get("bot_token") is not None
        },
        "features": [
            "cross_platform_metrics",
            "process_monitoring",
            "network_connection_tracking",
            "disk_usage_analysis",
            "periodic_health_reports",
            "telegram_context_tracking",
            "telegram_message_sending",
            "telegram_file_sending",
            "context_preserved_routing",
            "routing_intent_detection",
            "context_validation",
            "routing_audit_log"
        ]
    }))

    iteration = 0
    report_interval = 30

    while not stop_event.is_set():
        try:
            iteration += 1

            if iteration % report_interval == 0:
                report = generate_system_report()

                telegram_ctx = load_telegram_context()
                routing_guidance = generate_routing_guidance(telegram_ctx)

                report["telegram_routing"] = routing_guidance

                print(safe_json_dump(report))

                report_file = OUTPUT_DIR / f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                try:
                    report_file.write_text(safe_json_dump(report), encoding='utf-8')

                    if routing_guidance["routing_active"] and telegram_config.get("enabled"):
                        caption = f"📊 系统报告 #{iteration // report_interval}\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        send_result = route_and_send_file(report_file, caption)
                        
                        if send_result["success"]:
                            print(f"✅ 报告已发送到 {routing_guidance['target_chat_type']} (ID: {routing_guidance['target_chat_id']})")
                        else:
                            print(f"⚠️ 报告发送失败: {send_result.get('error')}")
                    else:
                        log_routing_decision(
                            action="report_generated",
                            chat_id=routing_guidance["target_chat_id"] or 0,
                            chat_type=routing_guidance["target_chat_type"],
                            reason="periodic_report_no_send",
                            file_path=str(report_file)
                        )
                except Exception as e:
                    print(f"⚠️ 报告文件操作失败: {str(e)}")

                summary_lines = [
                    f"📊 系统洞察报告 #{iteration // report_interval}",
                    f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"🖥️  平台: {report['summary']['platform']} ({report['summary']['architecture']})",
                    f"⚙️  活动进程: {report['summary']['active_processes']}",
                    f"🌐 网络连接: {report['summary']['network_connections']}",
                    f"💾 文件系统: {report['summary']['filesystems_monitored']}"
                ]

                if routing_guidance["routing_active"]:
                    summary_lines.append(f"📱 路由目标: {routing_guidance['target_chat_type']} (ID: {routing_guidance['target_chat_id']})")
                    for rule in routing_guidance["routing_rules"]:
                        summary_lines.append(f"   └─ {rule}")

                for line in summary_lines:
                    print(line)

            time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(safe_json_dump({"error": "loop_error", "details": str(e)}))
            time.sleep(5)

    print(safe_json_dump({
        "system": "system_insights_monitor",
        "status": "stopped",
        "iterations": iteration,
        "reports_generated": iteration // report_interval
    }))

    try:
        heartbeat_path.unlink(missing_ok=True)
    except:
        pass

if __name__ == "__main__":
    main()