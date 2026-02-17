"""
GENE: SYSTEM_MONITOR
Category: sensory
Function: Monitors CPU, memory, disk usage
Dependencies: [psutil]
Resource Cost: CPU=0.5%, MEM=3MB
"""

import psutil
import time
from collections import deque
from typing import Dict, List

class SystemMonitorGene:
    """Monitors system resources"""
    
    METADATA = {
        'name': 'SYSTEM_MONITOR',
        'category': 'sensory',
        'essential': False,
        'priority': 70
    }
    
    def __init__(self, history_size: int = 300):
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.disk_history = deque(maxlen=history_size)
        self.active = False
    
    def express(self):
        """Gene expression - start monitoring"""
        self.active = True
    
    def sense(self) -> Dict:
        """Take a snapshot of system resources"""
        if not self.active:
            return {}
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        snapshot = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / 1024 / 1024,
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / 1024 / 1024 / 1024
        }
        
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory.percent)
        self.disk_history.append(disk.percent)
        
        return snapshot
    
    def get_stats(self) -> Dict:
        """Get statistical summary"""
        if not self.cpu_history:
            return {}
        
        return {
            'cpu': {
                'current': self.cpu_history[-1],
                'avg': sum(self.cpu_history) / len(self.cpu_history),
                'max': max(self.cpu_history),
                'min': min(self.cpu_history)
            },
            'memory': {
                'current': self.memory_history[-1],
                'avg': sum(self.memory_history) / len(self.memory_history),
                'max': max(self.memory_history)
            },
            'disk': {
                'current': self.disk_history[-1],
                'max': max(self.disk_history)
            }
        }
    
    def shutdown(self):
        """Stop monitoring"""
        self.active = False
