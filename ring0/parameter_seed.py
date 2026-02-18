"""Deterministic parameter generation for evolution generations.

Uses a seed-based RNG so every generation's parameters are reproducible.
Pure stdlib -- no external dependencies (Ring 0 constraint).
"""

from typing import NamedTuple
import random


class EvolutionParams(NamedTuple):
    generation: int
    seed: int
    mutation_rate: float       # 0.01 .. 0.50
    population_size: int       # 2 .. 10
    max_runtime_sec: int       # 600 (fixed)
    crossover_rate: float      # 0.1 .. 0.9


def generate_params(generation: int, seed: int) -> EvolutionParams:
    """Return an immutable, deterministic parameter set for *generation*."""
    rng = random.Random(seed + generation)
    return EvolutionParams(
        generation=generation,
        seed=seed,
        mutation_rate=round(rng.uniform(0.01, 0.50), 4),
        population_size=rng.randint(2, 10),
        max_runtime_sec=600,
        crossover_rate=round(rng.uniform(0.1, 0.9), 4),
    )


def params_to_dict(params: EvolutionParams) -> dict:
    """Serialize an EvolutionParams instance to a plain dict."""
    return params._asdict()
