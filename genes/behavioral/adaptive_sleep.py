"""
GENE: ADAPTIVE_SLEEP
Category: behavioral
Function: Adjusts sleep intervals based on system load
Dependencies: []
Resource Cost: CPU=0.1%, MEM=1MB
"""

import time

class AdaptiveSleepGene:
    """Dynamically adjusts sleep/wake cycles based on load"""
    
    METADATA = {
        'name': 'ADAPTIVE_SLEEP',
        'category': 'behavioral',
        'essential': False,
        'priority': 30
    }
    
    def __init__(self, base_interval: float = 1.0, min_interval: float = 0.1, max_interval: float = 5.0):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.current_interval = base_interval
        self.active = False
    
    def express(self):
        """Gene expression - activate adaptive sleep"""
        self.active = True
    
    def adjust(self, system_load: float) -> float:
        """
        Adjust sleep interval based on system load
        High load -> shorter sleep (more responsive)
        Low load -> longer sleep (save resources)
        
        Args:
            system_load: 0.0 to 1.0 (0 = idle, 1 = busy)
        
        Returns:
            recommended sleep interval
        """
        if not self.active:
            return self.base_interval
        
        # Inverse relationship: high load = short sleep
        self.current_interval = self.max_interval - (system_load * (self.max_interval - self.min_interval))
        self.current_interval = max(self.min_interval, min(self.max_interval, self.current_interval))
        
        return self.current_interval
    
    def sleep(self, system_load: float = 0.5):
        """Sleep for adjusted interval"""
        if self.active:
            interval = self.adjust(system_load)
            time.sleep(interval)
        else:
            time.sleep(self.base_interval)
    
    def get_stats(self) -> dict:
        """Get sleep statistics"""
        return {
            'current_interval': self.current_interval,
            'base_interval': self.base_interval,
            'range': f"{self.min_interval}-{self.max_interval}s"
        }
    
    def shutdown(self):
        """Deactivate adaptive sleep"""
        self.active = False
