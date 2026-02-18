"""Output filter for Ring 2 logs before storing in memory.

Removes system noise (heartbeat messages, process metadata, empty lines, etc.)
while preserving valuable content (user interactions, errors, results).
"""

from __future__ import annotations

import re
from typing import List, Tuple


class Ring2OutputFilter:
    """Filter Ring 2 output logs to extract valuable content."""
    
    # System noise patterns (low value)
    NOISE_PATTERNS = [
        r'^\[Ring \d+\]',  # Ring process markers
        r'pid=\d+',  # Process IDs
        r'^[=\-]{5,}$',  # Separator lines
        r'^\s*$',  # Empty lines
        r'Heartbeat.*active',  # Heartbeat status
        r'Cycle \d+',  # Cycle counters
        r'^\d{2}:\d{2}:\d{2}$',  # Timestamps alone
        r'Running.*seconds?',  # Runtime status
        r'Thread.*alive',  # Thread status
        r'Started at \d+',  # Start timestamps
        r'Generation \d+ (started|completed)',  # Generation status
        r'Mutation rate: \d+\.\d+',  # Mutation rates
        r'Max runtime: \d+s',  # Runtime limits
        r'\[INFO\]',  # Log level markers
        r'\[DEBUG\]',
        r'\[TRACE\]',
        r'Timestamp:',
        r'Status: (OK|Running|Active)',
        # Filter test output noise
        r'过滤测试周期|Filter Test Cycle',
        r'模拟内存过滤系统|Simulating memory',
        r'测试数据|Test Data',
        r'过滤结果|Filtering Results',
        r'各阶段性能|Stage Performance',
        r'实时指标|Live Metrics',
        r'噪声比例|noise ratio',
        r'压缩率|compression',
        r'信息密度提升',
        r'累计处理行数',
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]
    
    def is_noise(self, line: str) -> bool:
        """Check if a line matches noise patterns."""
        line = line.strip()
        
        # Empty lines are noise
        if not line:
            return True
        
        # Check against all patterns
        for pattern in self.compiled_patterns:
            if pattern.search(line):
                return True
        
        return False
    
    def has_value(self, line: str) -> bool:
        """Check if a line has semantic value worth preserving."""
        line = line.strip()
        
        # Too short to be valuable
        if len(line) < 15:
            return False
        
        # Value indicators
        value_patterns = [
            r'任务|task',
            r'用户|user',
            r'请求|request',
            r'分析|analysis',
            r'结果|result',
            r'错误|error',
            r'异常|exception',
            r'traceback',
            r'failed|failure',
            r'warning|⚠',
            r'数据|data',
            r'优先级|priority',
            r'详情|detail',
            r'反馈|feedback',
            r'[\u4e00-\u9fff]{8,}',  # Longer Chinese text (8+ chars)
        ]
        
        for pattern in value_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def filter_lines(self, lines: List[str]) -> Tuple[List[str], int, int]:
        """Filter output lines, return (kept_lines, original_count, removed_count).
        
        Strategy:
        1. Remove obvious noise (system messages, empty lines)
        2. Keep lines with semantic value (errors, user content, results)
        3. Keep context around valuable lines (±1 line for coherence)
        """
        if not lines:
            return [], 0, 0
        
        original_count = len(lines)
        
        # Phase 1: Mark valuable lines
        valuable_indices = set()
        for i, line in enumerate(lines):
            if not self.is_noise(line) and self.has_value(line):
                valuable_indices.add(i)
                # Include context (previous and next line)
                if i > 0:
                    valuable_indices.add(i - 1)
                if i < len(lines) - 1:
                    valuable_indices.add(i + 1)
        
        # Phase 2: Build filtered output
        kept = []
        prev_index = -2  # Track if we're in a continuous sequence
        
        for i in sorted(valuable_indices):
            # Add separator if there's a gap
            if i > prev_index + 1 and kept:
                kept.append("...")
            kept.append(lines[i])
            prev_index = i
        
        removed_count = original_count - len([l for l in kept if l != "..."])
        
        return kept, original_count, removed_count
    
    def filter_text(self, text: str) -> str:
        """Filter a text block, return filtered text with summary."""
        lines = text.splitlines()
        kept, original, removed = self.filter_lines(lines)
        
        if not kept:
            return "(Output filtered — no valuable content detected)"
        
        filtered_text = "\n".join(kept)
        summary = f"\n[Filtered: kept {len(kept)} lines, removed {removed} lines of noise]"
        
        return filtered_text + summary


def filter_ring2_output(output: str, max_lines: int = 100) -> str:
    """Main entry point: filter Ring 2 output before storing in memory.
    
    Args:
        output: Raw output from Ring 2's .output.log
        max_lines: Maximum lines to keep after filtering
    
    Returns:
        Filtered output with noise removed
    """
    if not output or not output.strip():
        return "(no output)"
    
    filter_engine = Ring2OutputFilter()
    filtered = filter_engine.filter_text(output)
    
    # Enforce max_lines limit
    lines = filtered.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        filtered = "\n".join(lines)
        filtered += f"\n[Truncated to last {max_lines} lines]"
    
    return filtered
