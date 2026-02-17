"""
GENE: PATTERN_LEARNER
Category: cognitive
Function: Learns patterns from sensory data using simple Q-learning
Dependencies: []
Resource Cost: CPU=1%, MEM=10MB
"""

import random
from collections import defaultdict
from typing import Dict, Tuple, Any

class PatternLearnerGene:
    """Simple reinforcement learning for pattern recognition"""
    
    METADATA = {
        'name': 'PATTERN_LEARNER',
        'category': 'cognitive',
        'essential': False,
        'priority': 50
    }
    
    def __init__(self, learning_rate: float = 0.1, discount: float = 0.9, epsilon: float = 0.1):
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon  # Exploration rate
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        self.state_history = []
        self.active = False
    
    def express(self):
        """Gene expression - activate learning"""
        self.active = True
    
    def discretize_state(self, state: Dict) -> Tuple:
        """Convert continuous state to discrete for Q-table"""
        # Simple discretization: bucket values
        discrete = []
        for key in sorted(state.keys()):
            value = state[key]
            if isinstance(value, (int, float)):
                # Bucket into ranges
                bucket = int(value / 10) * 10
                discrete.append((key, bucket))
        return tuple(discrete)
    
    def choose_action(self, state: Dict, actions: list) -> Any:
        """Choose action using epsilon-greedy policy"""
        if not self.active or not actions:
            return random.choice(actions) if actions else None
        
        discrete_state = self.discretize_state(state)
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Exploitation: choose best action
        q_values = {action: self.q_table[(discrete_state, action)] for action in actions}
        return max(q_values, key=q_values.get)
    
    def update(self, state: Dict, action: Any, reward: float, next_state: Dict):
        """Update Q-value based on experience"""
        if not self.active:
            return
        
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Q-learning update
        current_q = self.q_table[(discrete_state, action)]
        max_next_q = max([self.q_table[(discrete_next_state, a)] for a in ['idle', 'active']], default=0)
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
        self.q_table[(discrete_state, action)] = new_q
        
        self.state_history.append({
            'state': discrete_state,
            'action': action,
            'reward': reward,
            'q_value': new_q
        })
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        return {
            'q_table_size': len(self.q_table),
            'total_updates': len(self.state_history),
            'avg_q_value': sum(self.q_table.values()) / len(self.q_table) if self.q_table else 0
        }
    
    def shutdown(self):
        """Stop learning"""
        self.active = False
