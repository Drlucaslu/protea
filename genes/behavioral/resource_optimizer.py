"""
GENE: RESOURCE_OPTIMIZER
Category: behavioral
Function: Optimizes resource usage based on environmental pressure
Dependencies: []
Resource Cost: CPU=0.5%, MEM=3MB
"""

from typing import Dict, List

class ResourceOptimizerGene:
    """Optimizes behavior based on resource constraints"""
    
    METADATA = {
        'name': 'RESOURCE_OPTIMIZER',
        'category': 'behavioral',
        'essential': False,
        'priority': 35
    }
    
    def __init__(self):
        self.optimization_history: List[Dict] = []
        self.active = False
        self.current_mode = 'balanced'  # 'aggressive', 'balanced', 'conservative'
    
    def express(self):
        """Gene expression - activate optimization"""
        self.active = True
    
    def optimize(self, resource_state: Dict) -> Dict:
        """
        Determine optimal behavior based on resource state
        
        Args:
            resource_state: {cpu_percent, memory_percent, disk_percent}
        
        Returns:
            optimization recommendations
        """
        if not self.active:
            return {'mode': 'balanced'}
        
        cpu = resource_state.get('cpu_percent', 0)
        memory = resource_state.get('memory_percent', 0)
        
        # Decision logic
        if cpu > 80 or memory > 80:
            mode = 'conservative'
            recommendations = {
                'mode': mode,
                'reduce_polling': True,
                'increase_sleep': True,
                'disable_non_essential': True
            }
        elif cpu < 20 and memory < 40:
            mode = 'aggressive'
            recommendations = {
                'mode': mode,
                'increase_polling': True,
                'decrease_sleep': True,
                'enable_all_features': True
            }
        else:
            mode = 'balanced'
            recommendations = {
                'mode': mode,
                'maintain_current': True
            }
        
        self.current_mode = mode
        self.optimization_history.append({
            'mode': mode,
            'cpu': cpu,
            'memory': memory,
            'recommendations': recommendations
        })
        
        return recommendations
    
    def get_stats(self) -> Dict:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {'current_mode': self.current_mode}
        
        modes = [h['mode'] for h in self.optimization_history[-100:]]
        return {
            'current_mode': self.current_mode,
            'mode_distribution': {
                'aggressive': modes.count('aggressive'),
                'balanced': modes.count('balanced'),
                'conservative': modes.count('conservative')
            },
            'total_optimizations': len(self.optimization_history)
        }
    
    def shutdown(self):
        """Stop optimization"""
        self.active = False
