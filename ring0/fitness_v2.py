"""Enhanced multi-objective fitness evaluation for Protea evolution.

Extends the original fitness system with:
- Task performance scoring (API correctness, problem-solving)
- Code innovation metrics (algorithm diversity, architecture changes)
- Progressive difficulty adjustment

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
from typing import Any


# ============= CODE INNOVATION ANALYSIS =============

def analyze_code_innovation(source_code: str, recent_sources: list[str]) -> dict[str, Any]:
    """Evaluate code innovation compared to recent generations.
    
    Returns dict with:
        - algorithm_diversity: 0.0-1.0 (new patterns/algorithms)
        - architecture_change: 0.0-1.0 (structure modifications)
        - module_count: number of defined functions/classes
        - comment_ratio: ratio of comment lines to total lines
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {
            "algorithm_diversity": 0.0,
            "architecture_change": 0.0,
            "module_count": 0,
            "comment_ratio": 0.0,
        }
    
    # Count modules (functions + classes)
    current_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            current_modules.add(node.name)
        elif isinstance(node, ast.ClassDef):
            current_modules.add(node.name)
    
    module_count = len(current_modules)
    
    # Compare with recent generations
    if recent_sources:
        recent_module_sets = []
        for old_source in recent_sources:
            try:
                old_tree = ast.parse(old_source)
                old_modules = set()
                for node in ast.walk(old_tree):
                    if isinstance(node, ast.FunctionDef):
                        old_modules.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        old_modules.add(node.name)
                recent_module_sets.append(old_modules)
            except SyntaxError:
                continue
        
        # Algorithm diversity: how many NEW modules?
        if recent_module_sets:
            all_old_modules = set().union(*recent_module_sets)
            new_modules = current_modules - all_old_modules
            algorithm_diversity = len(new_modules) / max(len(current_modules), 1)
        else:
            algorithm_diversity = 1.0
        
        # Architecture change: Jaccard distance of module sets
        if recent_module_sets:
            distances = []
            for old_mods in recent_module_sets:
                if not old_mods:
                    distances.append(1.0)
                    continue
                intersection = len(current_modules & old_mods)
                union = len(current_modules | old_mods)
                jaccard = intersection / union if union > 0 else 0.0
                distances.append(1.0 - jaccard)
            architecture_change = sum(distances) / len(distances)
        else:
            architecture_change = 1.0
    else:
        algorithm_diversity = 1.0
        architecture_change = 1.0
    
    # Comment ratio
    lines = source_code.split("\n")
    comment_lines = sum(1 for ln in lines if ln.strip().startswith("#"))
    comment_ratio = comment_lines / max(len(lines), 1)
    
    return {
        "algorithm_diversity": min(algorithm_diversity, 1.0),
        "architecture_change": min(architecture_change, 1.0),
        "module_count": module_count,
        "comment_ratio": comment_ratio,
    }


# ============= TASK PERFORMANCE ANALYSIS =============

_API_SUCCESS_PATTERNS = [
    re.compile(r"HTTP/\d\.\d\s+200", re.IGNORECASE),
    re.compile(r"status\s*[:=]\s*200", re.IGNORECASE),
    re.compile(r"✓|✅|SUCCESS|OK", re.IGNORECASE),
]

_API_ERROR_PATTERNS = [
    re.compile(r"HTTP/\d\.\d\s+[45]\d\d", re.IGNORECASE),
    re.compile(r"status\s*[:=]\s*[45]\d\d", re.IGNORECASE),
    re.compile(r"✗|❌|FAILED|ERROR", re.IGNORECASE),
]

_COMPUTATION_PATTERNS = [
    re.compile(r"result\s*[:=]\s*\d+", re.IGNORECASE),
    re.compile(r"calculated|computed|processed", re.IGNORECASE),
    re.compile(r"^\s*\d+(\.\d+)?\s*$"),  # numeric results
]

_DATA_PROCESSING_PATTERNS = [
    re.compile(r"processed \d+ (records|items|rows)", re.IGNORECASE),
    re.compile(r"transformed|filtered|sorted|grouped", re.IGNORECASE),
    re.compile(r"output: \[.*\]", re.IGNORECASE),
]


def analyze_task_performance(output_lines: list[str]) -> dict[str, Any]:
    """Evaluate task-solving performance from output.
    
    Returns dict with:
        - api_success_rate: 0.0-1.0 (HTTP/API correctness)
        - computation_quality: 0.0-1.0 (numeric results present)
        - data_processing: 0.0-1.0 (data transformation indicators)
        - problem_solving: 0.0-1.0 (overall task completion)
    """
    meaningful = [ln for ln in output_lines if ln.strip()]
    total = len(meaningful)
    
    if total == 0:
        return {
            "api_success_rate": 0.0,
            "computation_quality": 0.0,
            "data_processing": 0.0,
            "problem_solving": 0.0,
        }
    
    # API success indicators
    api_success = sum(1 for ln in meaningful if any(p.search(ln) for p in _API_SUCCESS_PATTERNS))
    api_errors = sum(1 for ln in meaningful if any(p.search(ln) for p in _API_ERROR_PATTERNS))
    api_total = api_success + api_errors
    api_success_rate = api_success / api_total if api_total > 0 else 0.5
    
    # Computation quality
    computation_count = sum(1 for ln in meaningful if any(p.search(ln) for p in _COMPUTATION_PATTERNS))
    computation_quality = min(computation_count / max(total, 1) * 5, 1.0)
    
    # Data processing
    processing_count = sum(1 for ln in meaningful if any(p.search(ln) for p in _DATA_PROCESSING_PATTERNS))
    data_processing = min(processing_count / max(total, 1) * 5, 1.0)
    
    # Problem solving:综合指标
    problem_solving = (api_success_rate + computation_quality + data_processing) / 3
    
    return {
        "api_success_rate": api_success_rate,
        "computation_quality": computation_quality,
        "data_processing": data_processing,
        "problem_solving": min(problem_solving, 1.0),
    }


# ============= PROGRESSIVE TASK LEVELS =============

TASK_LEVELS = {
    0: "Basic survival (heartbeat only)",
    1: "Simple output (print, format)",
    2: "Data processing (sort, filter, transform)",
    3: "Algorithm implementation (search, optimize)",
    4: "System features (file I/O, HTTP server)",
    5: "Intelligent behavior (learning, adaptation)",
    6: "Complex systems (distributed, orchestration)",
}


def get_task_level_for_generation(generation: int, recent_scores: list[float]) -> int:
    """Determine task difficulty level based on generation and performance.
    
    Args:
        generation: Current generation number
        recent_scores: List of recent fitness scores (e.g., last 10)
    
    Returns:
        Task level (0-6)
    """
    # Base level from generation (slow progression)
    base_level = min(generation // 30, 3)
    
    # Adjust based on performance
    if len(recent_scores) >= 10:
        avg = sum(recent_scores) / len(recent_scores)
        if avg >= 0.90:
            # Performing excellently → increase challenge
            return min(base_level + 2, 6)
        elif avg >= 0.80:
            # Good performance → moderate challenge
            return min(base_level + 1, 6)
        elif avg < 0.65:
            # Struggling → decrease challenge
            return max(base_level - 1, 1)
    
    return max(base_level, 1)


# ============= ENHANCED FITNESS EVALUATION =============

def evaluate_output_v2(
    output_lines: list[str],
    source_code: str,
    survived: bool,
    elapsed: float,
    max_runtime: float,
    recent_fingerprints: list[set[str]] | None = None,
    recent_sources: list[str] | None = None,
    task_level: int = 1,
) -> tuple[float, dict]:
    """Enhanced multi-objective fitness evaluation.
    
    Scoring breakdown (max 1.0):
        - Base survival: 0.40
        - Output quality: 0.20 (volume + diversity + structure)
        - Output novelty: 0.10
        - Task performance: 0.20
        - Code innovation: 0.10
    
    Args:
        output_lines: Program output (list of lines)
        source_code: Source code for innovation analysis
        survived: Whether the program survived full runtime
        elapsed: Actual runtime in seconds
        max_runtime: Maximum allowed runtime
        recent_fingerprints: Token sets from recent gens (for novelty)
        recent_sources: Source code from recent gens (for innovation)
        task_level: Current task difficulty level (0-6)
    
    Returns:
        (score, detail_dict)
    """
    # Import original patterns from fitness.py
    from ring0.fitness import (
        _STRUCTURED_PATTERNS,
        _FUNCTIONAL_PATTERNS,
        _output_fingerprint,
        compute_novelty,
    )
    
    if not survived:
        ratio = min(elapsed / max_runtime, 0.99) if max_runtime > 0 else 0.0
        score = ratio * 0.39
        return score, {
            "basis": "failure",
            "elapsed_ratio": round(ratio, 4),
            "task_level": task_level,
        }
    
    # --- Component scores ---
    base = 0.40
    
    meaningful = [ln for ln in output_lines if ln.strip()]
    total = len(meaningful)
    
    # 1. Output quality (0.20)
    volume = min(total / 50, 1.0) * 0.08
    
    unique = len(set(meaningful)) if total > 0 else 0
    diversity = (unique / total) * 0.06 if total > 0 else 0.0
    
    structured_count = sum(1 for ln in meaningful if any(p.match(ln) for p in _STRUCTURED_PATTERNS))
    structure = min(structured_count / max(total, 1) * 2, 1.0) * 0.06
    
    output_quality = volume + diversity + structure
    
    # 2. Output novelty (0.10)
    current_fp = _output_fingerprint(meaningful)
    if recent_fingerprints:
        novelty_raw = compute_novelty(current_fp, recent_fingerprints)
    else:
        novelty_raw = 1.0
    novelty = novelty_raw * 0.10
    
    # 3. Task performance (0.20)
    task_perf = analyze_task_performance(output_lines)
    task_score = (
        task_perf["problem_solving"] * 0.10 +
        task_perf["api_success_rate"] * 0.04 +
        task_perf["data_processing"] * 0.03 +
        task_perf["computation_quality"] * 0.03
    )
    
    # 4. Code innovation (0.10)
    code_innov = analyze_code_innovation(source_code, recent_sources or [])
    innovation_score = (
        code_innov["algorithm_diversity"] * 0.04 +
        code_innov["architecture_change"] * 0.03 +
        min(code_innov["module_count"] / 10, 1.0) * 0.02 +
        code_innov["comment_ratio"] * 0.01
    )
    
    # Error penalty (from output)
    error_count = sum(1 for ln in output_lines if "error" in ln.lower() or "exception" in ln.lower())
    error_penalty = min(error_count / max(total, 1), 1.0) * 0.05
    
    # Final score
    score = base + output_quality + novelty + task_score + innovation_score - error_penalty
    score = max(0.40, min(score, 1.0))
    
    detail = {
        "basis": "survived_v2",
        "task_level": task_level,
        "base": base,
        "output_quality": round(output_quality, 4),
        "novelty": round(novelty, 4),
        "task_performance": round(task_score, 4),
        "code_innovation": round(innovation_score, 4),
        "error_penalty": round(error_penalty, 4),
        "meaningful_lines": total,
        "task_perf_detail": task_perf,
        "code_innov_detail": code_innov,
        "fingerprint": sorted(list(current_fp))[:50],
    }
    
    return round(score, 4), detail
