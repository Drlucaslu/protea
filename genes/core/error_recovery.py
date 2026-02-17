"""
GENE: ERROR_RECOVERY
Category: core
Function: Catches and recovers from runtime errors
Dependencies: []
Resource Cost: CPU=0.1%, MEM=2MB
"""

import traceback
import time
from collections import deque
from typing import Dict

class ErrorRecoveryGene:
    """Handles errors and maintains error history"""
    
    METADATA = {
        'name': 'ERROR_RECOVERY',
        'category': 'core',
        'essential': True,
        'priority': 98
    }
    
    def __init__(self, max_history: int = 100):
        self.error_history = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = {}
        self.active = False
    
    def express(self):
        """Gene expression - activate error tracking"""
        self.active = True
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """
        Handle an error and decide if recovery is possible
        Returns: True if recovered, False if fatal
        """
        if not self.active:
            return False
        
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_record = {
            'timestamp': time.time(),
            'type': error_type,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        self.error_history.append(error_record)
        
        # Recovery logic: if same error happens > 10 times, it's fatal
        if self.error_counts[error_type] > 10:
            print(f"[ERROR_RECOVERY] Fatal: {error_type} occurred {self.error_counts[error_type]} times")
            return False
        
        print(f"[ERROR_RECOVERY] Recovered from {error_type} in {context}")
        return True
    
    def get_stats(self) -> dict:
        """Get error statistics"""
        return {
            'total_errors': len(self.error_history),
            'by_type': dict(self.error_counts),
            'recent': list(self.error_history)[-5:]
        }
    
    def shutdown(self):
        """Deactivate error recovery"""
        self.active = False
