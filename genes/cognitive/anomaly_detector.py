"""
GENE: ANOMALY_DETECTOR
Category: cognitive
Function: Detects anomalies using statistical methods (Z-score)
Dependencies: []
Resource Cost: CPU=0.5%, MEM=5MB
"""

import math
from collections import deque
from typing import Dict, List, Optional

class AnomalyDetectorGene:
    """Detects statistical anomalies in metrics"""
    
    METADATA = {
        'name': 'ANOMALY_DETECTOR',
        'category': 'cognitive',
        'essential': False,
        'priority': 40
    }
    
    def __init__(self, window_size: int = 50, threshold: float = 2.5):
        self.window_size = window_size
        self.threshold = threshold  # Z-score threshold
        self.metric_windows: Dict[str, deque] = {}
        self.anomalies: List[Dict] = []
        self.active = False
    
    def express(self):
        """Gene expression - activate anomaly detection"""
        self.active = True
    
    def add_metric(self, metric_name: str, value: float) -> Optional[Dict]:
        """
        Add a metric value and check for anomaly
        Returns: anomaly dict if detected, None otherwise
        """
        if not self.active:
            return None
        
        if metric_name not in self.metric_windows:
            self.metric_windows[metric_name] = deque(maxlen=self.window_size)
        
        window = self.metric_windows[metric_name]
        window.append(value)
        
        # Need enough data for statistics
        if len(window) < 10:
            return None
        
        # Calculate mean and std
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 0.0001
        
        # Calculate Z-score
        z_score = (value - mean) / std
        
        if abs(z_score) > self.threshold:
            anomaly = {
                'metric': metric_name,
                'value': value,
                'mean': mean,
                'std': std,
                'z_score': z_score,
                'severity': 'high' if abs(z_score) > self.threshold * 1.5 else 'medium'
            }
            self.anomalies.append(anomaly)
            return anomaly
        
        return None
    
    def check_metrics(self, metrics: Dict[str, float]) -> List[Dict]:
        """Check multiple metrics at once"""
        anomalies = []
        for name, value in metrics.items():
            anomaly = self.add_metric(name, value)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies
    
    def get_stats(self) -> Dict:
        """Get anomaly detection statistics"""
        return {
            'total_anomalies': len(self.anomalies),
            'recent_anomalies': self.anomalies[-10:],
            'monitored_metrics': len(self.metric_windows)
        }
    
    def shutdown(self):
        """Stop anomaly detection"""
        self.active = False
