#!/usr/bin/env python3
"""Ring 2 — Generation 4: Distributed Systems Monitor & Orchestrator

A real-time systems monitoring and orchestration platform that:
- Monitors multiple processes with health checks and performance metrics
- Dynamic configuration hot-reloading without restarts
- Process lifecycle management (spawn, monitor, restart)
- Resource utilization tracking (CPU, memory, I/O)
- Distributed task queue with priority scheduling
- System-wide event correlation and anomaly detection
- Interactive web dashboard with real-time metrics
- Configuration version control and rollback
"""

import os
import pathlib
import sys
import time
import json
import sqlite3
import psutil
import subprocess
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set, Tuple, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event, Lock
from collections import defaultdict, deque
from queue import PriorityQueue
import hashlib
import traceback

HEARTBEAT_INTERVAL = 2
HTTP_PORT = 8899

def heartbeat_loop(heartbeat_path: pathlib.Path, pid: int, stop_event: Event) -> None:
    """Dedicated heartbeat thread - CRITICAL for survival."""
    while not stop_event.is_set():
        try:
            heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
        except Exception:
            pass
        time.sleep(HEARTBEAT_INTERVAL)


# ============= CONFIGURATION MANAGER WITH HOT RELOAD =============

@dataclass
class SystemConfig:
    """Dynamic system configuration."""
    process_pool_size: int = 5
    health_check_interval: float = 5.0
    resource_sample_rate: float = 1.0
    task_queue_capacity: int = 1000
    anomaly_threshold: float = 2.5
    auto_restart_enabled: bool = True
    log_retention_hours: int = 24
    metrics_window_size: int = 300
    
    version: str = "1.0.0"
    last_modified: float = field(default_factory=time.time)


class ConfigurationManager:
    """Hot-reloadable configuration manager with version control."""
    
    def __init__(self, config_path: pathlib.Path):
        self.config_path = config_path
        self.config = SystemConfig()
        self.version_history = []
        self.lock = Lock()
        self.last_checksum = ""
        self.reload_count = 0
        
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text())
                with self.lock:
                    old_version = self.config.version
                    self.config = SystemConfig(**data)
                    self.version_history.append({
                        'version': old_version,
                        'timestamp': time.time()
                    })
                    self.reload_count += 1
            except Exception as e:
                print(f"[ConfigMgr] Failed to load config: {e}", flush=True)
        else:
            self._save_config()
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            self.config_path.write_text(json.dumps(asdict(self.config), indent=2))
        except Exception as e:
            print(f"[ConfigMgr] Failed to save config: {e}", flush=True)
    
    def check_and_reload(self) -> bool:
        """Check for config changes and reload if necessary."""
        if not self.config_path.exists():
            return False
        
        try:
            content = self.config_path.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()
            
            if checksum != self.last_checksum:
                self.last_checksum = checksum
                self._load_config()
                return True
        except Exception:
            pass
        
        return False
    
    def get_config(self) -> SystemConfig:
        """Thread-safe config access."""
        with self.lock:
            return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration parameters."""
        with self.lock:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            self.config.last_modified = time.time()
            self._save_config()


# ============= PROCESS MONITOR =============

@dataclass
class ProcessMetrics:
    """Process performance metrics."""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_mb: float
    num_threads: int
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    create_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class HealthCheck:
    """Process health check result."""
    pid: int
    timestamp: float
    alive: bool
    responsive: bool
    cpu_anomaly: bool
    memory_anomaly: bool
    message: str = ""


class ProcessMonitor:
    """Multi-process health and performance monitor."""
    
    def __init__(self):
        self.monitored_pids: Set[int] = set()
        self.metrics_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=300))
        self.health_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = Lock()
        
    def add_process(self, pid: int):
        """Add process to monitoring."""
        with self.lock:
            self.monitored_pids.add(pid)
    
    def remove_process(self, pid: int):
        """Remove process from monitoring."""
        with self.lock:
            self.monitored_pids.discard(pid)
    
    def collect_metrics(self) -> List[ProcessMetrics]:
        """Collect current metrics for all monitored processes."""
        metrics = []
        
        with self.lock:
            pids_to_check = list(self.monitored_pids)
        
        for pid in pids_to_check:
            try:
                proc = psutil.Process(pid)
                
                with proc.oneshot():
                    io_counters = proc.io_counters() if hasattr(proc, 'io_counters') else None
                    
                    metric = ProcessMetrics(
                        pid=pid,
                        name=proc.name(),
                        status=proc.status(),
                        cpu_percent=proc.cpu_percent(interval=0.1),
                        memory_mb=proc.memory_info().rss / 1024 / 1024,
                        num_threads=proc.num_threads(),
                        io_read_mb=io_counters.read_bytes / 1024 / 1024 if io_counters else 0,
                        io_write_mb=io_counters.write_bytes / 1024 / 1024 if io_counters else 0,
                        create_time=proc.create_time()
                    )
                    
                    metrics.append(metric)
                    
                    with self.lock:
                        self.metrics_history[pid].append(metric)
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                with self.lock:
                    self.monitored_pids.discard(pid)
        
        return metrics
    
    def health_check(self, pid: int, config: SystemConfig) -> HealthCheck:
        """Perform health check on a process."""
        timestamp = time.time()
        
        try:
            proc = psutil.Process(pid)
            alive = proc.is_running()
            
            if not alive:
                return HealthCheck(pid, timestamp, False, False, False, False, "Process not running")
            
            # Check responsiveness (basic check)
            responsive = proc.status() not in ['zombie', 'stopped']
            
            # Anomaly detection
            cpu_anomaly = False
            memory_anomaly = False
            
            with self.lock:
                if pid in self.metrics_history and len(self.metrics_history[pid]) > 10:
                    history = list(self.metrics_history[pid])
                    
                    # CPU anomaly
                    cpu_values = [m.cpu_percent for m in history[-30:]]
                    if cpu_values:
                        mean_cpu = sum(cpu_values) / len(cpu_values)
                        std_cpu = (sum((x - mean_cpu) ** 2 for x in cpu_values) / len(cpu_values)) ** 0.5
                        current_cpu = history[-1].cpu_percent
                        
                        if std_cpu > 0 and abs(current_cpu - mean_cpu) > config.anomaly_threshold * std_cpu:
                            cpu_anomaly = True
                    
                    # Memory anomaly
                    mem_values = [m.memory_mb for m in history[-30:]]
                    if mem_values:
                        mean_mem = sum(mem_values) / len(mem_values)
                        std_mem = (sum((x - mean_mem) ** 2 for x in mem_values) / len(mem_values)) ** 0.5
                        current_mem = history[-1].memory_mb
                        
                        if std_mem > 0 and abs(current_mem - mean_mem) > config.anomaly_threshold * std_mem:
                            memory_anomaly = True
            
            message = []
            if cpu_anomaly:
                message.append("CPU spike")
            if memory_anomaly:
                message.append("Memory anomaly")
            
            health = HealthCheck(
                pid=pid,
                timestamp=timestamp,
                alive=alive,
                responsive=responsive,
                cpu_anomaly=cpu_anomaly,
                memory_anomaly=memory_anomaly,
                message=", ".join(message) if message else "Healthy"
            )
            
            with self.lock:
                self.health_history[pid].append(health)
            
            return health
            
        except Exception as e:
            return HealthCheck(pid, timestamp, False, False, False, False, f"Error: {e}")
    
    def get_statistics(self, pid: int) -> Dict:
        """Get statistical summary for a process."""
        with self.lock:
            if pid not in self.metrics_history:
                return {}
            
            history = list(self.metrics_history[pid])
            
            if not history:
                return {}
            
            cpu_values = [m.cpu_percent for m in history]
            mem_values = [m.memory_mb for m in history]
            
            return {
                'pid': pid,
                'samples': len(history),
                'cpu_mean': sum(cpu_values) / len(cpu_values),
                'cpu_max': max(cpu_values),
                'cpu_min': min(cpu_values),
                'mem_mean': sum(mem_values) / len(mem_values),
                'mem_max': max(mem_values),
                'mem_min': min(mem_values),
                'uptime_seconds': time.time() - history[0].create_time if history else 0
            }


# ============= DISTRIBUTED TASK QUEUE =============

@dataclass
class Task:
    """Queued task with priority."""
    task_id: str
    priority: int  # Lower = higher priority
    payload: Dict
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __lt__(self, other):
        return self.priority < other.priority


class TaskQueue:
    """Priority-based distributed task queue."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.queue = PriorityQueue(maxsize=capacity)
        self.tasks: Dict[str, Task] = {}
        self.lock = Lock()
        self.task_counter = 0
        
    def submit(self, payload: Dict, priority: int = 5) -> str:
        """Submit a task to the queue."""
        with self.lock:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{int(time.time() * 1000)}"
            
            task = Task(
                task_id=task_id,
                priority=priority,
                payload=payload
            )
            
            try:
                self.queue.put_nowait(task)
                self.tasks[task_id] = task
                return task_id
            except:
                return ""
    
    def get_next(self, timeout: float = 1.0) -> Optional[Task]:
        """Get next task from queue."""
        try:
            task = self.queue.get(timeout=timeout)
            with self.lock:
                task.started_at = time.time()
                task.status = "running"
            return task
        except:
            return None
    
    def complete(self, task_id: str, result: Any = None, error: str = None):
        """Mark task as completed."""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.completed_at = time.time()
                task.status = "completed" if error is None else "failed"
                task.result = result
                task.error = error
    
    def get_stats(self) -> Dict:
        """Get queue statistics."""
        with self.lock:
            pending = sum(1 for t in self.tasks.values() if t.status == "pending")
            running = sum(1 for t in self.tasks.values() if t.status == "running")
            completed = sum(1 for t in self.tasks.values() if t.status == "completed")
            failed = sum(1 for t in self.tasks.values() if t.status == "failed")
            
            return {
                'total_tasks': len(self.tasks),
                'queue_size': self.queue.qsize(),
                'pending': pending,
                'running': running,
                'completed': completed,
                'failed': failed
            }


# ============= SYSTEM ORCHESTRATOR =============

class SystemOrchestrator:
    """Master orchestrator for distributed system management."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.process_monitor = ProcessMonitor()
        self.task_queue = TaskQueue()
        self.system_events: deque = deque(maxlen=1000)
        self.lock = Lock()
        
        # Add self to monitoring
        self.process_monitor.add_process(os.getpid())
        
        # Add parent process if available
        try:
            parent = psutil.Process(os.getpid()).parent()
            if parent:
                self.process_monitor.add_process(parent.pid)
        except:
            pass
    
    def log_event(self, event_type: str, message: str, metadata: Dict = None):
        """Log system event."""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'message': message,
            'metadata': metadata or {}
        }
        
        with self.lock:
            self.system_events.append(event)
        
        print(f"[{event_type}] {message}", flush=True)
    
    def check_configuration(self):
        """Check and reload configuration if changed."""
        if self.config_manager.check_and_reload():
            self.log_event('CONFIG', f'Configuration reloaded (v{self.config_manager.get_config().version})')
            return True
        return False
    
    def monitor_cycle(self):
        """Run one monitoring cycle."""
        config = self.config_manager.get_config()
        
        # Collect metrics
        metrics = self.process_monitor.collect_metrics()
        
        # Health checks
        for metric in metrics:
            health = self.process_monitor.health_check(metric.pid, config)
            
            if not health.alive:
                self.log_event('HEALTH', f'Process {metric.pid} died', {'pid': metric.pid})
            elif health.cpu_anomaly or health.memory_anomaly:
                self.log_event('ANOMALY', f'Process {metric.pid}: {health.message}', 
                             {'pid': metric.pid, 'cpu': metric.cpu_percent, 'mem': metric.memory_mb})
        
        return metrics
    
    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary."""
        config = self.config_manager.get_config()
        
        # System-wide metrics
        try:
            system_cpu = psutil.cpu_percent(interval=0.1)
            system_mem = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
        except:
            system_cpu = 0
            system_mem = 0
            disk_usage = 0
        
        # Process statistics
        metrics = self.process_monitor.collect_metrics()
        process_stats = [self.process_monitor.get_statistics(m.pid) for m in metrics]
        
        # Recent events
        with self.lock:
            recent_events = list(self.system_events)[-20:]
        
        return {
            'config': asdict(config),
            'config_reload_count': self.config_manager.reload_count,
            'system': {
                'cpu_percent': system_cpu,
                'memory_percent': system_mem,
                'disk_percent': disk_usage
            },
            'processes': {
                'monitored': len(metrics),
                'metrics': [m.to_dict() for m in metrics],
                'statistics': process_stats
            },
            'task_queue': self.task_queue.get_stats(),
            'events': recent_events
        }


# ============= WEB DASHBOARD =============

orchestrator_state = {'orchestrator': None}


class SystemDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for system monitoring dashboard."""
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/summary':
            self.serve_json(self.get_summary())
        elif self.path.startswith('/api/config/update'):
            self.handle_config_update()
        else:
            self.send_error(404)
    
    def get_summary(self):
        orch = orchestrator_state.get('orchestrator')
        if orch:
            return orch.get_system_summary()
        return {'error': 'Orchestrator not initialized'}
    
    def handle_config_update(self):
        """Handle configuration update request."""
        # For simplicity, just trigger a reload check
        orch = orchestrator_state.get('orchestrator')
        if orch:
            orch.check_configuration()
            self.serve_json({'status': 'checked'})
        else:
            self.serve_json({'error': 'No orchestrator'})
    
    def serve_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def serve_dashboard(self):
        html = '''<!DOCTYPE html>
<html><head><title>Distributed Systems Monitor</title><meta charset="UTF-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#0a0e27;color:#e0e0e0;font-size:13px}
.header{background:linear-gradient(135deg,#1e293b,#0f172a);padding:20px;border-bottom:3px solid #3b82f6}
.title{font-size:24px;font-weight:700;color:#60a5fa}
.subtitle{font-size:12px;color:#94a3b8;margin-top:5px}
.container{padding:20px;max-width:1600px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:15px;margin-top:20px}
.panel{background:#1e293b;border:2px solid #334155;border-radius:8px;padding:15px}
.panel-title{font-size:16px;font-weight:700;margin-bottom:12px;color:#38bdf8;border-bottom:1px solid #334155;padding-bottom:8px}
.metric{display:flex;justify-content:space-between;margin:6px 0;padding:8px;background:#0f172a;border-radius:4px;border-left:3px solid #3b82f6}
.label{color:#94a3b8;font-size:12px}
.value{color:#22d3ee;font-weight:700}
.value.warning{color:#fbbf24}
.value.danger{color:#ef4444}
.value.success{color:#10b981}
.process-list{max-height:300px;overflow-y:auto}
.process-item{background:#0f172a;padding:10px;margin:8px 0;border-radius:4px;border-left:4px solid #10b981}
.process-item.anomaly{border-left-color:#f59e0b}
.event-log{max-height:250px;overflow-y:auto;font-size:11px}
.event{padding:6px;margin:4px 0;background:#0f172a;border-radius:3px;border-left:3px solid #64748b}
.event.CONFIG{border-left-color:#8b5cf6}
.event.ANOMALY{border-left-color:#f59e0b}
.event.HEALTH{border-left-color:#ef4444}
.timestamp{color:#64748b;font-size:10px}
.progress-bar{background:#1e293b;height:8px;border-radius:4px;overflow:hidden;margin:5px 0}
.progress-fill{background:linear-gradient(90deg,#3b82f6,#60a5fa);height:100%;transition:width 0.3s}
</style></head><body>
<div class="header">
<div class="title">⚙️ Distributed Systems Monitor</div>
<div class="subtitle">Real-time process orchestration • Dynamic configuration • Performance analytics</div>
</div>
<div class="container">
<div class="grid">
<div class="panel"><div class="panel-title">🔧 Configuration</div><div id="config"></div></div>
<div class="panel"><div class="panel-title">💻 System Resources</div><div id="system"></div></div>
<div class="panel"><div class="panel-title">📊 Task Queue</div><div id="tasks"></div></div>
<div class="panel"><div class="panel-title">🔍 Monitored Processes</div><div id="processes"></div></div>
</div>
<div class="panel" style="margin-top:15px"><div class="panel-title">📋 System Events</div><div id="events" class="event-log"></div></div>
</div>
<script>
async function loadData() {
    try {
        const data = await fetch('/api/summary').then(r => r.json());
        
        // Configuration
        const cfg = data.config;
        document.getElementById('config').innerHTML = 
            '<div class="metric"><span class="label">Version</span><span class="value">' + cfg.version + '</span></div>' +
            '<div class="metric"><span class="label">Pool Size</span><span class="value">' + cfg.process_pool_size + '</span></div>' +
            '<div class="metric"><span class="label">Health Check (s)</span><span class="value">' + cfg.health_check_interval + '</span></div>' +
            '<div class="metric"><span class="label">Auto Restart</span><span class="value success">' + (cfg.auto_restart_enabled ? 'ENABLED' : 'DISABLED') + '</span></div>' +
            '<div class="metric"><span class="label">Reloads</span><span class="value">' + data.config_reload_count + '</span></div>';
        
        // System
        const sys = data.system;
        const cpuClass = sys.cpu_percent > 80 ? 'danger' : sys.cpu_percent > 60 ? 'warning' : 'success';
        const memClass = sys.memory_percent > 80 ? 'danger' : sys.memory_percent > 60 ? 'warning' : 'success';
        
        document.getElementById('system').innerHTML = 
            '<div class="metric"><span class="label">CPU Usage</span><span class="value ' + cpuClass + '">' + sys.cpu_percent.toFixed(1) + '%</span></div>' +
            '<div class="progress-bar"><div class="progress-fill" style="width:' + sys.cpu_percent + '%"></div></div>' +
            '<div class="metric"><span class="label">Memory Usage</span><span class="value ' + memClass + '">' + sys.memory_percent.toFixed(1) + '%</span></div>' +
            '<div class="progress-bar"><div class="progress-fill" style="width:' + sys.memory_percent + '%"></div></div>' +
            '<div class="metric"><span class="label">Disk Usage</span><span class="value">' + sys.disk_percent.toFixed(1) + '%</span></div>' +
            '<div class="progress-bar"><div class="progress-fill" style="width:' + sys.disk_percent + '%"></div></div>';
        
        // Tasks
        const tasks = data.task_queue;
        document.getElementById('tasks').innerHTML = 
            '<div class="metric"><span class="label">Total Tasks</span><span class="value">' + tasks.total_tasks + '</span></div>' +
            '<div class="metric"><span class="label">Queue Size</span><span class="value">' + tasks.queue_size + '</span></div>' +
            '<div class="metric"><span class="label">Pending</span><span class="value warning">' + tasks.pending + '</span></div>' +
            '<div class="metric"><span class="label">Running</span><span class="value">' + tasks.running + '</span></div>' +
            '<div class="metric"><span class="label">Completed</span><span class="value success">' + tasks.completed + '</span></div>' +
            '<div class="metric"><span class="label">Failed</span><span class="value danger">' + tasks.failed + '</span></div>';
        
        // Processes
        const procs = data.processes.metrics;
        let procsHTML = '<div class="process-list">';
        procs.forEach(p => {
            const cpuColor = p.cpu_percent > 50 ? 'warning' : 'success';
            procsHTML += '<div class="process-item">' +
                '<div style="display:flex;justify-content:space-between;margin-bottom:5px">' +
                '<span style="font-weight:700">' + p.name + '</span>' +
                '<span style="color:#64748b">PID ' + p.pid + '</span></div>' +
                '<div style="display:flex;gap:15px;font-size:11px">' +
                '<span>CPU: <span class="value ' + cpuColor + '">' + p.cpu_percent.toFixed(1) + '%</span></span>' +
                '<span>MEM: <span class="value">' + p.memory_mb.toFixed(1) + 'MB</span></span>' +
                '<span>Threads: <span class="value">' + p.num_threads + '</span></span>' +
                '</div></div>';
        });
        procsHTML += '</div>';
        document.getElementById('processes').innerHTML = procsHTML;
        
        // Events
        const events = data.events;
        let eventsHTML = '';
        events.reverse().forEach(e => {
            const date = new Date(e.timestamp * 1000);
            const timeStr = date.toLocaleTimeString();
            eventsHTML += '<div class="event ' + e.type + '">' +
                '<div class="timestamp">' + timeStr + '</div>' +
                '<div>' + e.message + '</div></div>';
        });
        document.getElementById('events').innerHTML = eventsHTML;
        
    } catch(e) {
        console.error('Failed to load data:', e);
    }
}
loadData();
setInterval(loadData, 2000);
</script>
</body></html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))


def main() -> None:
    """Main entry point."""
    heartbeat_path = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    stop_event = Event()
    
    heartbeat_thread = Thread(target=heartbeat_loop, args=(heartbeat_path, pid, stop_event), daemon=True)
    heartbeat_thread.start()
    
    output_dir = pathlib.Path("ring2_output")
    output_dir.mkdir(exist_ok=True)
    
    config_path = output_dir / "system_config.json"
    
    print(f"[Ring 2 Gen 4] Distributed Systems Monitor pid={pid}", flush=True)
    print(f"⚙️ Dashboard: http://localhost:{HTTP_PORT}", flush=True)
    print(f"📝 Config: {config_path}", flush=True)
    
    # Initialize orchestrator
    config_manager = ConfigurationManager(config_path)
    orchestrator = SystemOrchestrator(config_manager)
    orchestrator_state['orchestrator'] = orchestrator
    
    orchestrator.log_event('STARTUP', f'System orchestrator initialized (PID {pid})')
    
    # Start HTTP server
    def run_server():
        try:
            server = HTTPServer(('127.0.0.1', HTTP_PORT), SystemDashboardHandler)
            server.serve_forever()
        except Exception as e:
            print(f"[HTTP] Server error: {e}", flush=True)
    
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    
    # Submit some sample tasks
    for i in range(10):
        orchestrator.task_queue.submit(
            {'type': 'analytics', 'data': f'dataset_{i}'}, 
            priority=i % 3
        )
    
    cycle = 0
    last_config_check = time.time()
    
    try:
        while True:
            config = config_manager.get_config()
            
            # Check for configuration changes every 10 seconds
            if time.time() - last_config_check > 10:
                orchestrator.check_configuration()
                last_config_check = time.time()
            
            # Monitoring cycle
            metrics = orchestrator.monitor_cycle()
            
            # Periodic reporting
            if cycle % 20 == 0:
                summary = orchestrator.get_system_summary()
                
                print(f"\n[Cycle {cycle}] System Summary:", flush=True)
                print(f"  Config v{summary['config']['version']} (reloads: {summary['config_reload_count']})", flush=True)
                print(f"  System: CPU {summary['system']['cpu_percent']:.1f}% | MEM {summary['system']['memory_percent']:.1f}% | DISK {summary['system']['disk_percent']:.1f}%", flush=True)
                print(f"  Processes: {summary['processes']['monitored']} monitored", flush=True)
                
                for stat in summary['processes']['statistics']:
                    print(f"    PID {stat['pid']}: CPU avg={stat['cpu_mean']:.1f}% max={stat['cpu_max']:.1f}% | MEM avg={stat['mem_mean']:.1f}MB", flush=True)
                
                tasks = summary['task_queue']
                print(f"  Tasks: {tasks['completed']} done, {tasks['running']} running, {tasks['pending']} pending, {tasks['failed']} failed", flush=True)
            
            # Process some tasks
            if cycle % 5 == 0:
                for _ in range(3):
                    task = orchestrator.task_queue.get_next(timeout=0.1)
                    if task:
                        print(f"  🔨 Processing {task.task_id} (priority {task.priority})", flush=True)
                        time.sleep(0.1)  # Simulate work
                        orchestrator.task_queue.complete(task.task_id, result={'status': 'success'})
            
            # Detailed metrics every 30 cycles
            if cycle % 30 == 15:
                print(f"\n[Cycle {cycle}] Detailed Metrics:", flush=True)
                for metric in metrics:
                    print(f"  📊 {metric.name} (PID {metric.pid}):", flush=True)
                    print(f"      Status: {metric.status} | Threads: {metric.num_threads}", flush=True)
                    print(f"      CPU: {metric.cpu_percent:.1f}% | Memory: {metric.memory_mb:.1f}MB", flush=True)
                    print(f"      I/O: R={metric.io_read_mb:.2f}MB W={metric.io_write_mb:.2f}MB", flush=True)
            
            time.sleep(config.resource_sample_rate)
            cycle += 1
    
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            heartbeat_path.unlink(missing_ok=True)
        except:
            pass
        
        orchestrator.log_event('SHUTDOWN', f'System orchestrator shutting down (cycle {cycle})')
        
        summary = orchestrator.get_system_summary()
        print(f"\n[Ring 2] Shutdown complete. pid={pid}", flush=True)
        print(f"  Final stats: {summary['processes']['monitored']} processes, {summary['task_queue']['completed']} tasks completed", flush=True)


if __name__ == "__main__":
    main()