#!/usr/bin/env python3
"""Memory Filter Gene - Filters system noise from memory consolidation

This cognitive gene implements intelligent memory filtering to:
1. Identify and remove system noise patterns
2. Evaluate content quality and information density
3. Detect semantic redundancy
4. Prioritize high-value memories for consolidation
5. Learn and adapt filtering patterns over time

The goal is to ensure only meaningful, non-repetitive information
enters long-term memory storage.
"""

import re
import time
import hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import math


# ============= NOISE PATTERN DATABASE =============

class NoisePatternDB:
    """Database of known noise patterns in memory systems"""
    
    # System operational noise
    SYSTEM_NOISE = [
        r'^\[Ring \d+\]',                    # Ring process markers
        r'pid=\d+',                          # Process IDs
        r'^[=\-]{5,}$',                      # Separator lines
        r'^\s*$',                            # Empty lines
        r'Heartbeat.*active',                # Heartbeat status
        r'Cycle \d+',                        # Cycle counters
        r'^\d{2}:\d{2}:\d{2}$',             # Timestamps alone
        r'Running.*seconds?',                # Runtime status
        r'Thread.*alive',                    # Thread status
        r'Started at \d+',                   # Start timestamps
        r'Generation \d+ (started|completed)', # Generation status
        r'Mutation rate: \d+\.\d+',          # Mutation rates
        r'Max runtime: \d+s',                # Runtime limits
    ]
    
    # Logging noise
    LOG_NOISE = [
        r'\[INFO\]',
        r'\[DEBUG\]',
        r'\[TRACE\]',
        r'Timestamp:',
        r'Status: (OK|Running|Active)',
        r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # Log timestamps
    ]
    
    # Repetitive headers and boilerplate
    BOILERPLATE = [
        r'^分析.*\(.*Analysis.*\)',          # Repeated headers
        r'^报告.*\(.*Report.*\)',
        r'^概览.*\(.*Overview.*\)',
        r'^={3,}',                           # Header underlines
        r'^-{3,}',
        r'^\*{3,}',
    ]
    
    # Generic status messages
    STATUS_NOISE = [
        r'^✅.*complete',
        r'^🔄.*processing',
        r'^❌.*failed',
        r'^⚠️.*warning',
        r'successfully (completed|finished|done)',
    ]
    
    @classmethod
    def get_all_patterns(cls) -> List[str]:
        """Get all noise patterns combined"""
        return (cls.SYSTEM_NOISE + cls.LOG_NOISE + 
                cls.BOILERPLATE + cls.STATUS_NOISE)
    
    @classmethod
    def get_compiled_patterns(cls) -> List[re.Pattern]:
        """Get compiled regex patterns for efficiency"""
        return [re.compile(pattern, re.IGNORECASE) 
                for pattern in cls.get_all_patterns()]


# ============= CONTENT QUALITY EVALUATOR =============

@dataclass
class ContentQuality:
    """Quality metrics for memory content"""
    information_density: float      # Information per character
    semantic_uniqueness: float       # How unique vs. seen before
    pattern_diversity: float         # Variety of content types
    noise_ratio: float              # Proportion of noise
    overall_score: float            # Combined quality score
    
    def is_worth_remembering(self, threshold: float = 0.4) -> bool:
        """Determine if content should be stored in memory"""
        return self.overall_score >= threshold


class ContentEvaluator:
    """Evaluates content quality for memory consolidation"""
    
    def __init__(self):
        self.noise_patterns = NoisePatternDB.get_compiled_patterns()
        self.seen_hashes: Set[str] = set()
        self.semantic_patterns: Counter = Counter()
        
    def evaluate(self, content: str) -> ContentQuality:
        """Evaluate content quality for memory storage"""
        if not content or len(content.strip()) < 10:
            return ContentQuality(0, 0, 0, 1.0, 0)
        
        lines = content.split('\n')
        
        # Calculate noise ratio
        noise_lines = self._count_noise_lines(lines)
        noise_ratio = noise_lines / len(lines) if lines else 1.0
        
        # Calculate information density
        info_density = self._calculate_information_density(content)
        
        # Check semantic uniqueness
        uniqueness = self._calculate_uniqueness(content)
        
        # Analyze pattern diversity
        diversity = self._calculate_diversity(content)
        
        # Compute overall quality score
        overall = self._compute_quality_score(
            info_density, uniqueness, diversity, noise_ratio
        )
        
        return ContentQuality(
            information_density=info_density,
            semantic_uniqueness=uniqueness,
            pattern_diversity=diversity,
            noise_ratio=noise_ratio,
            overall_score=overall
        )
    
    def _count_noise_lines(self, lines: List[str]) -> int:
        """Count how many lines match noise patterns"""
        noise_count = 0
        for line in lines:
            for pattern in self.noise_patterns:
                if pattern.search(line):
                    noise_count += 1
                    break
        return noise_count
    
    def _calculate_information_density(self, content: str) -> float:
        """Calculate information per character using entropy"""
        if not content:
            return 0.0
        
        # Character frequency distribution
        char_counts = Counter(content.lower())
        total_chars = sum(char_counts.values())
        
        # Shannon entropy
        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                prob = count / total_chars
                entropy -= prob * math.log2(prob)
        
        # Normalize to 0-1 range (max entropy for ASCII is ~6.57)
        normalized_entropy = min(entropy / 6.57, 1.0)
        
        # Penalize very short or very long content
        length = len(content)
        length_factor = 1.0
        if length < 50:
            length_factor = length / 50
        elif length > 5000:
            length_factor = max(0.5, 5000 / length)
        
        return normalized_entropy * length_factor
    
    def _calculate_uniqueness(self, content: str) -> float:
        """Calculate how unique this content is vs. what we've seen"""
        # Content hash for exact duplicate detection
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self.seen_hashes:
            return 0.0  # Exact duplicate
        
        self.seen_hashes.add(content_hash)
        
        # Extract semantic patterns (key phrases)
        semantic_sig = self._extract_semantic_signature(content)
        
        # Check similarity to previously seen patterns
        max_overlap = 0
        for seen_sig, count in self.semantic_patterns.items():
            overlap = len(semantic_sig & seen_sig) / len(semantic_sig | seen_sig)
            if overlap > max_overlap:
                max_overlap = overlap
        
        # Update patterns
        self.semantic_patterns[semantic_sig] += 1
        
        # Uniqueness is inverse of overlap
        return 1.0 - max_overlap
    
    def _extract_semantic_signature(self, content: str) -> frozenset:
        """Extract key semantic patterns from content"""
        # Extract meaningful words (3+ chars, not all digits)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Use top N most distinctive words
        word_counts = Counter(words)
        top_words = set(w for w, c in word_counts.most_common(20))
        
        return frozenset(top_words)
    
    def _calculate_diversity(self, content: str) -> float:
        """Calculate variety of content types present"""
        features = {
            'has_code': bool(re.search(r'(def |class |import |function|var |const )', content)),
            'has_numbers': bool(re.search(r'\d+', content)),
            'has_chinese': bool(re.search(r'[\u4e00-\u9fff]', content)),
            'has_questions': bool(re.search(r'\?', content)),
            'has_urls': bool(re.search(r'https?://', content)),
            'has_bullets': bool(re.search(r'^\s*[-*•]', content, re.MULTILINE)),
            'has_quotes': bool(re.search(r'["""\'\'\'`]', content)),
        }
        
        diversity = sum(features.values()) / len(features)
        return diversity
    
    def _compute_quality_score(self, info_density: float, uniqueness: float,
                               diversity: float, noise_ratio: float) -> float:
        """Compute overall quality score with weighted factors"""
        # Weighted combination
        score = (
            0.25 * info_density +      # Information content
            0.30 * uniqueness +         # Novelty (important)
            0.20 * diversity +          # Content variety
            0.25 * (1 - noise_ratio)    # Low noise is critical
        )
        
        # Apply penalty for high noise content
        if noise_ratio > 0.7:
            score *= (1 - noise_ratio * 0.5)  # Reduce score by up to 50%
        
        return max(0.0, min(1.0, score))


# ============= MEMORY FILTER GENE =============

class MemoryFilterGene:
    """Cognitive gene for filtering memory content"""
    
    def __init__(self):
        self.evaluator = ContentEvaluator()
        self.stats = {
            'total_evaluated': 0,
            'accepted': 0,
            'rejected': 0,
            'noise_filtered': 0,
            'duplicates_filtered': 0,
        }
        self.quality_history: List[float] = []
        self.adaptive_threshold = 0.4  # Start with moderate threshold
        
    def filter_content(self, content: str, context: str = "") -> Tuple[bool, ContentQuality]:
        """
        Filter content for memory consolidation
        
        Args:
            content: The content to evaluate
            context: Optional context about the content source
            
        Returns:
            (should_remember, quality_metrics)
        """
        self.stats['total_evaluated'] += 1
        
        quality = self.evaluator.evaluate(content)
        self.quality_history.append(quality.overall_score)
        
        # Adaptive threshold adjustment
        if len(self.quality_history) > 100:
            self._adjust_threshold()
        
        should_remember = quality.is_worth_remembering(self.adaptive_threshold)
        
        if should_remember:
            self.stats['accepted'] += 1
        else:
            self.stats['rejected'] += 1
            if quality.noise_ratio > 0.5:
                self.stats['noise_filtered'] += 1
            if quality.semantic_uniqueness < 0.1:
                self.stats['duplicates_filtered'] += 1
        
        return should_remember, quality
    
    def filter_batch(self, contents: List[str]) -> List[Tuple[str, ContentQuality]]:
        """
        Filter a batch of content items
        
        Returns:
            List of (content, quality) for items worth remembering
        """
        results = []
        for content in contents:
            should_remember, quality = self.filter_content(content)
            if should_remember:
                results.append((content, quality))
        return results
    
    def _adjust_threshold(self):
        """Adaptively adjust threshold based on recent quality distribution"""
        recent = self.quality_history[-100:]
        
        # If most content is low quality, lower threshold
        # If most is high quality, raise threshold
        avg_quality = sum(recent) / len(recent)
        
        if avg_quality < 0.3:
            # Lots of noise, be more selective
            self.adaptive_threshold = min(0.5, self.adaptive_threshold + 0.01)
        elif avg_quality > 0.7:
            # High quality stream, can be more accepting
            self.adaptive_threshold = max(0.3, self.adaptive_threshold - 0.01)
    
    def get_filter_stats(self) -> Dict:
        """Get filtering statistics"""
        stats = dict(self.stats)
        
        if self.stats['total_evaluated'] > 0:
            stats['acceptance_rate'] = self.stats['accepted'] / self.stats['total_evaluated']
            stats['rejection_rate'] = self.stats['rejected'] / self.stats['total_evaluated']
        
        if self.quality_history:
            stats['avg_quality'] = sum(self.quality_history) / len(self.quality_history)
            stats['recent_avg_quality'] = (sum(self.quality_history[-20:]) / 
                                          min(20, len(self.quality_history)))
        
        stats['adaptive_threshold'] = self.adaptive_threshold
        
        return stats
    
    def reset_statistics(self):
        """Reset filtering statistics"""
        self.stats = {
            'total_evaluated': 0,
            'accepted': 0,
            'rejected': 0,
            'noise_filtered': 0,
            'duplicates_filtered': 0,
        }
        self.quality_history = []
    
    def express(self):
        """Express gene (called by Ring 2 DNA interpreter)"""
        # This is a passive gene - it's called by other systems
        # No active thread needed
        print(f"[MEMORY_FILTER] Gene expressed - ready to filter memories")
        print(f"[MEMORY_FILTER] Initial threshold: {self.adaptive_threshold:.2f}")
    
    def get_stats(self) -> Dict:
        """Get gene statistics (called by organism)"""
        return self.get_filter_stats()


# ============= STANDALONE TESTING =============

def test_memory_filter():
    """Test the memory filter with sample content"""
    print("🧪 Testing Memory Filter Gene\n")
    print("="*60)
    
    gene = MemoryFilterGene()
    
    # Test cases
    test_contents = [
        # High quality content
        ("Discovered new pattern: when CPU load exceeds 80%, "
         "adaptive sleep should increase by 0.3s to prevent thrashing. "
         "This insight came from analyzing 500 cycles of system behavior.",
         "High quality insight"),
        
        # System noise
        ("[Ring 2] Heartbeat active\nCycle 1234\nRunning 45 seconds\n"
         "Status: OK\npid=12345",
         "System noise"),
        
        # Duplicate content
        ("The system is running normally. All processes are active.",
         "Generic status"),
        
        # Mixed quality
        ("✅ Analysis complete\n\nKey finding: Memory consolidation "
         "efficiency correlates with sleep cycle duration (R²=0.87). "
         "Longer NREM3 phases improve retention by 23%.",
         "Mixed (noise + insight)"),
        
        # Empty/low content
        ("", "Empty"),
        
        # Code snippet (high diversity)
        ("def optimize_memory():\n    # Filter noise patterns\n    "
         "filtered = [x for x in memories if quality(x) > 0.4]\n"
         "    return prioritize(filtered)",
         "Code snippet"),
    ]
    
    print("\n📋 Evaluating content samples:\n")
    
    for content, label in test_contents:
        should_remember, quality = gene.filter_content(content)
        
        print(f"{'='*60}")
        print(f"Label: {label}")
        print(f"Content preview: {content[:80]}...")
        print(f"\nQuality Metrics:")
        print(f"  • Information Density: {quality.information_density:.3f}")
        print(f"  • Semantic Uniqueness: {quality.semantic_uniqueness:.3f}")
        print(f"  • Pattern Diversity:   {quality.pattern_diversity:.3f}")
        print(f"  • Noise Ratio:         {quality.noise_ratio:.3f}")
        print(f"  • Overall Score:       {quality.overall_score:.3f}")
        print(f"\n{'✅ ACCEPT' if should_remember else '❌ REJECT'} "
              f"(threshold: {gene.adaptive_threshold:.3f})")
        print()
    
    print(f"{'='*60}")
    print("\n📊 Filter Statistics:")
    stats = gene.get_filter_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  • {key}: {value:.3f}")
        else:
            print(f"  • {key}: {value}")


if __name__ == "__main__":
    test_memory_filter()
