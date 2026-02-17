"""Progressive task difficulty system for Protea evolution.

Generates increasingly challenging tasks based on current generation
performance and provides dynamic directives to guide evolution.

Pure stdlib.
"""

from __future__ import annotations

import random
from typing import Any


# ============= TASK LEVEL DEFINITIONS =============

TASK_LEVELS = {
    0: {
        "name": "Basic Survival",
        "description": "Maintain heartbeat only",
        "expected_output": "Minimal activity",
    },
    1: {
        "name": "Simple Output",
        "description": "Generate formatted text output",
        "expected_output": "Structured text, basic formatting",
    },
    2: {
        "name": "Data Processing",
        "description": "Process and transform data structures",
        "expected_output": "Sorted/filtered/transformed data",
    },
    3: {
        "name": "Algorithm Implementation",
        "description": "Implement classic algorithms",
        "expected_output": "Algorithm results with performance metrics",
    },
    4: {
        "name": "System Features",
        "description": "Build system-level functionality",
        "expected_output": "File I/O, HTTP servers, inter-process communication",
    },
    5: {
        "name": "Intelligent Behavior",
        "description": "Demonstrate learning and adaptation",
        "expected_output": "Adaptive responses, pattern recognition",
    },
    6: {
        "name": "Complex Systems",
        "description": "Orchestrate distributed components",
        "expected_output": "Multi-component coordination, system monitoring",
    },
}


# ============= TASK TEMPLATES BY LEVEL =============

TASK_TEMPLATES = {
    1: [
        "Generate a formatted table of the first 20 Fibonacci numbers with their indices.",
        "Create a colorful ASCII art banner that displays the current generation number.",
        "Output a JSON object containing system information (PID, timestamp, generation).",
        "Print a multiplication table (1-10) in aligned columns.",
        "Generate 10 random integers and display them with sum, mean, and median.",
    ],
    2: [
        "Sort a list of 100 random integers using quicksort and display execution time.",
        "Process a simulated CSV dataset: filter rows where value > 50, compute average.",
        "Implement a binary search on a sorted array and report search steps.",
        "Transform a list of strings: reverse each, sort alphabetically, remove duplicates.",
        "Group a list of numbers by even/odd, compute statistics for each group.",
    ],
    3: [
        "Implement Dijkstra's shortest path algorithm on a random graph.",
        "Build a min-heap data structure and perform 20 insert/extract operations.",
        "Create a simple LRU cache (capacity=10) and simulate 50 read/write operations.",
        "Implement the Sieve of Eratosthenes to find all primes up to 10,000.",
        "Build a trie (prefix tree) and perform word lookup and autocomplete.",
    ],
    4: [
        "Create an HTTP server that responds to GET /status with system metrics JSON.",
        "Implement a file-based key-value store with GET/SET/DELETE operations.",
        "Build a simple task scheduler that executes functions at specified intervals.",
        "Create a log analyzer that processes a simulated log file and extracts errors.",
        "Implement a basic pub-sub system with topics and multiple subscribers.",
    ],
    5: [
        "Build a simple reinforcement learning agent that learns optimal paths in a grid.",
        "Implement a naive Bayes classifier and train it on synthetic data.",
        "Create an adaptive system that tunes parameters based on performance feedback.",
        "Build a pattern recognition system that detects trends in time-series data.",
        "Implement a genetic algorithm to optimize a simple fitness function.",
    ],
    6: [
        "Orchestrate multiple worker processes with task distribution and result aggregation.",
        "Build a distributed monitoring system that tracks health of multiple components.",
        "Create a load balancer that distributes requests across simulated backend services.",
        "Implement a distributed cache with consistency guarantees across nodes.",
        "Build a self-healing system that detects and recovers from component failures.",
    ],
}


# ============= TASK GENERATOR =============

class TaskGenerator:
    """Generates progressive tasks based on evolution performance."""
    
    def __init__(
        self,
        base_level: int = 1,
        adjustment_window: int = 10,
        upgrade_threshold: float = 0.90,
        downgrade_threshold: float = 0.65,
    ):
        """
        Args:
            base_level: Starting task level (0-6)
            adjustment_window: Number of recent scores to consider
            upgrade_threshold: Avg score to trigger level increase
            downgrade_threshold: Avg score to trigger level decrease
        """
        self.base_level = base_level
        self.adjustment_window = adjustment_window
        self.upgrade_threshold = upgrade_threshold
        self.downgrade_threshold = downgrade_threshold
        
        self.current_level = base_level
        self.level_history: list[int] = [base_level]
    
    def adjust_level(self, recent_scores: list[float]) -> int:
        """Adjust task level based on recent performance.
        
        Args:
            recent_scores: Fitness scores from recent generations
        
        Returns:
            New task level (0-6)
        """
        if len(recent_scores) < self.adjustment_window:
            return self.current_level
        
        # Consider only the most recent window
        window = recent_scores[-self.adjustment_window:]
        avg_score = sum(window) / len(window)
        
        # Upgrade if consistently high performance
        if avg_score >= self.upgrade_threshold:
            new_level = min(self.current_level + 1, 6)
            if new_level > self.current_level:
                print(f"🎯 Performance excellent (avg={avg_score:.2f}) → Level {new_level}")
        
        # Downgrade if struggling
        elif avg_score < self.downgrade_threshold:
            new_level = max(self.current_level - 1, 1)
            if new_level < self.current_level:
                print(f"⚠️ Performance low (avg={avg_score:.2f}) → Level {new_level}")
        
        else:
            new_level = self.current_level
        
        self.current_level = new_level
        self.level_history.append(new_level)
        return new_level
    
    def get_task_directive(
        self,
        generation: int,
        recent_scores: list[float] | None = None,
    ) -> str:
        """Generate a task directive for the current generation.
        
        Args:
            generation: Current generation number
            recent_scores: Recent fitness scores (for level adjustment)
        
        Returns:
            Task directive string (to be included in evolution prompt)
        """
        # Adjust level if scores provided
        if recent_scores:
            self.adjust_level(recent_scores)
        
        level = self.current_level
        level_info = TASK_LEVELS.get(level, TASK_LEVELS[1])
        
        # Select a specific task from templates
        if level in TASK_TEMPLATES:
            templates = TASK_TEMPLATES[level]
            # Use generation as seed for consistency
            random.seed(generation)
            task_description = random.choice(templates)
        else:
            task_description = level_info["description"]
        
        # Build directive
        directive = f"""
# Task Level {level}: {level_info["name"]}

Your goal for this generation:
{task_description}

Expected output characteristics:
- {level_info["expected_output"]}
- Clear, structured output demonstrating task completion
- Performance metrics where applicable

Remember: The heartbeat must be maintained, but focus on completing this task.
High-quality task completion will significantly boost your fitness score.
""".strip()
        
        return directive
    
    def get_level_info(self) -> dict[str, Any]:
        """Get information about the current task level."""
        level = self.current_level
        return {
            "current_level": level,
            "level_name": TASK_LEVELS[level]["name"],
            "level_description": TASK_LEVELS[level]["description"],
            "history": self.level_history[-20:],  # Last 20 levels
        }
    
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about level progression."""
        if not self.level_history:
            return {
                "current_level": self.current_level,
                "generations": 0,
                "upgrades": 0,
                "downgrades": 0,
            }
        
        upgrades = sum(
            1 for i in range(1, len(self.level_history))
            if self.level_history[i] > self.level_history[i - 1]
        )
        downgrades = sum(
            1 for i in range(1, len(self.level_history))
            if self.level_history[i] < self.level_history[i - 1]
        )
        
        return {
            "current_level": self.current_level,
            "generations": len(self.level_history),
            "upgrades": upgrades,
            "downgrades": downgrades,
            "max_level_reached": max(self.level_history),
            "level_distribution": {
                level: self.level_history.count(level)
                for level in set(self.level_history)
            },
        }


# ============= CONVENIENCE FUNCTIONS =============

def get_task_for_generation(
    generation: int,
    recent_scores: list[float] | None = None,
) -> tuple[int, str]:
    """Get task level and directive for a generation.
    
    This is a stateless convenience function for one-off usage.
    For persistent level tracking, use TaskGenerator class.
    
    Args:
        generation: Generation number
        recent_scores: Recent fitness scores (for level determination)
    
    Returns:
        (task_level, directive_string)
    """
    # Simple generation-based level (increases every 30 gens)
    base_level = min(generation // 30, 3)
    
    # Adjust based on recent performance if provided
    if recent_scores and len(recent_scores) >= 10:
        avg = sum(recent_scores[-10:]) / 10
        if avg >= 0.90:
            level = min(base_level + 2, 6)
        elif avg >= 0.80:
            level = min(base_level + 1, 6)
        elif avg < 0.65:
            level = max(base_level - 1, 1)
        else:
            level = max(base_level, 1)
    else:
        level = max(base_level, 1)
    
    # Generate directive
    gen = TaskGenerator(base_level=level)
    directive = gen.get_task_directive(generation)
    
    return level, directive
