"""
GENE: SIGNAL_HANDLER
Category: core
Function: Handles SIGTERM/SIGINT for graceful shutdown
Dependencies: []
Resource Cost: CPU=0.01%, MEM=0.5MB
"""

import signal
import sys
from typing import Callable

class SignalHandlerGene:
    """Handles OS signals for graceful shutdown"""
    
    METADATA = {
        'name': 'SIGNAL_HANDLER',
        'category': 'core',
        'essential': True,
        'priority': 99
    }
    
    def __init__(self, shutdown_callback: Callable):
        self.shutdown_callback = shutdown_callback
        self.active = False
    
    def express(self):
        """Gene expression - register signal handlers"""
        def handle_signal(signum, frame):
            print(f"\n[SIGNAL] Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_callback()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        self.active = True
    
    def shutdown(self):
        """Deactivate signal handling"""
        self.active = False
