"""
GENE: HEARTBEAT
Category: core
Function: Maintains survival signal to Ring 1
Dependencies: []
Resource Cost: CPU=0.1%, MEM=1MB
"""

import time
import pathlib
from threading import Event

class HeartbeatGene:
    """Critical survival gene - writes heartbeat to shared file"""
    
    METADATA = {
        'name': 'HEARTBEAT',
        'category': 'core',
        'essential': True,
        'priority': 100
    }
    
    def __init__(self, heartbeat_path: str, pid: int, interval: float = 2.0):
        self.heartbeat_path = pathlib.Path(heartbeat_path)
        self.pid = pid
        self.interval = interval
        self.stop_event = Event()
    
    def express(self):
        """Gene expression - runs heartbeat loop"""
        while not self.stop_event.is_set():
            try:
                self.heartbeat_path.write_text(f"{self.pid}\n{time.time()}\n")
            except Exception:
                pass  # Silent failure for robustness
            time.sleep(self.interval)
    
    def shutdown(self):
        """Stop gene expression"""
        self.stop_event.set()
