"""
GENE: STATE_REPORTER
Category: social
Function: Periodically reports organism state to console
Dependencies: []
Resource Cost: CPU=0.2%, MEM=2MB
"""

import time
from typing import Dict, Callable

class StateReporterGene:
    """Reports organism state periodically"""
    
    METADATA = {
        'name': 'STATE_REPORTER',
        'category': 'social',
        'essential': False,
        'priority': 10
    }
    
    def __init__(self, interval: int = 30, state_callback: Callable = None):
        self.interval = interval
        self.state_callback = state_callback or (lambda: {})
        self.active = False
        self.cycle_count = 0
    
    def express(self):
        """Gene expression - activate reporting"""
        self.active = True
    
    def report(self) -> str:
        """Generate and print status report"""
        if not self.active:
            return ""
        
        self.cycle_count += 1
        state = self.state_callback()
        
        report_lines = [
            f"\n{'='*60}",
            f"[Cycle {self.cycle_count}] Silicon Life Status Report",
            f"{'='*60}"
        ]
        
        # Format state information
        for category, data in state.items():
            report_lines.append(f"\n📊 {category.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, float):
                        report_lines.append(f"  {key}: {value:.2f}")
                    else:
                        report_lines.append(f"  {key}: {value}")
            else:
                report_lines.append(f"  {data}")
        
        report = "\n".join(report_lines)
        print(report)
        return report
    
    def should_report(self, elapsed: float) -> bool:
        """Check if enough time has passed to report"""
        return self.active and (elapsed % self.interval < 1)
    
    def shutdown(self):
        """Stop reporting"""
        self.active = False
