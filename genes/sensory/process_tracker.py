"""
GENE: PROCESS_TRACKER
Category: sensory
Function: Tracks monitored processes and their metrics
Dependencies: [psutil]
Resource Cost: CPU=0.3%, MEM=5MB
"""

import psutil
import time
from typing import Dict, List, Optional
from collections import defaultdict, deque

class ProcessTrackerGene:
    """Tracks external processes and their resource usage"""
    
    METADATA = {
        'name': 'PROCESS_TRACKER',
        'category': 'sensory',
        'essential': False,
        'priority': 60
    }
    
    def __init__(self, history_size: int = 100):
        self.monitored_pids: List[int] = []
        self.process_history = defaultdict(lambda: deque(maxlen=history_size))
        self.active = False
    
    def express(self):
        """Gene expression - start tracking"""
        self.active = True
    
    def add_process(self, pid: int):
        """Add a process to monitor"""
        if pid not in self.monitored_pids:
            self.monitored_pids.append(pid)
    
    def remove_process(self, pid: int):
        """Remove a process from monitoring"""
        if pid in self.monitored_pids:
            self.monitored_pids.remove(pid)
    
    def sense(self) -> Dict[int, Dict]:
        """Scan all monitored processes"""
        if not self.active:
            return {}
        
        results = {}
        for pid in self.monitored_pids[:]:  # Copy list to avoid modification during iteration
            try:
                proc = psutil.Process(pid)
                with proc.oneshot():
                    metrics = {
                        'timestamp': time.time(),
                        'cpu_percent': proc.cpu_percent(),
                        'memory_mb': proc.memory_info().rss / 1024 / 1024,
                        'num_threads': proc.num_threads(),
                        'status': proc.status(),
                        'name': proc.name()
                    }
                    self.process_history[pid].append(metrics)
                    results[pid] = metrics
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process died or not accessible
                self.remove_process(pid)
        
        return results
    
    def get_process_stats(self, pid: int) -> Optional[Dict]:
        """Get statistics for a specific process"""
        if pid not in self.process_history or not self.process_history[pid]:
            return None
        
        history = list(self.process_history[pid])
        cpu_values = [h['cpu_percent'] for h in history]
        mem_values = [h['memory_mb'] for h in history]
        
        return {
            'pid': pid,
            'samples': len(history),
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg': sum(mem_values) / len(mem_values),
            'memory_max': max(mem_values),
            'last_status': history[-1]['status']
        }
    
    def shutdown(self):
        """Stop tracking"""
        self.active = False
