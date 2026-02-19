"""Tests for _should_evolve task idle decay logic."""

from __future__ import annotations

import threading
import time

import pytest

from ring0.sentinel import _should_evolve


class FakeState:
    """Minimal stand-in for SentinelState."""

    def __init__(self):
        self.p0_active = threading.Event()
        self.p1_active = threading.Event()
        self.last_evolution_time: float = 0.0


class FakeFitness:
    """Configurable fitness stub for plateau testing."""

    def __init__(self, plateaued: bool = False):
        self._plateaued = plateaued

    def is_plateaued(self, window=5, epsilon=0.03):
        return self._plateaued


class TestShouldEvolveIdleDecay:
    """Task idle decay: no tasks → extended cooldown."""

    def test_recent_task_no_decay(self):
        """Task completed just now → no idle decay applied."""
        state = FakeState()
        state.last_evolution_time = time.time() - 2000  # well past cooldown
        last_task = time.time() - 60  # 1 minute ago

        should, plateaued = _should_evolve(
            state, cooldown_sec=1800, last_task_time=last_task,
        )
        assert should is True

    def test_idle_2h_doubles_cooldown(self):
        """2h idle → cooldown multiplied by 2.0x."""
        state = FakeState()
        cooldown = 1800
        # Last evolution was 2700s ago (1.5x cooldown — would pass base check).
        state.last_evolution_time = time.time() - 2700
        # Last task was 2 hours ago → multiplier = 1.0 + 2/2 = 2.0
        # idle_cooldown = 1800 * 2.0 = 3600  — 2700 < 3600, so blocked.
        last_task = time.time() - 2 * 3600

        should, plateaued = _should_evolve(
            state, cooldown_sec=cooldown, last_task_time=last_task,
        )
        assert should is False
        assert plateaued is False  # not plateau — just idle decay

    def test_idle_4h_triples_cooldown(self):
        """4h idle → cooldown multiplied by 3.0x."""
        state = FakeState()
        cooldown = 1800
        # Last evolution was 4000s ago (>2.2x cooldown).
        state.last_evolution_time = time.time() - 4000
        # Last task was 4 hours ago → multiplier = 1.0 + 4/2 = 3.0
        # idle_cooldown = 1800 * 3.0 = 5400 — 4000 < 5400, so blocked.
        last_task = time.time() - 4 * 3600

        should, plateaued = _should_evolve(
            state, cooldown_sec=cooldown, last_task_time=last_task,
        )
        assert should is False

    def test_idle_8h_caps_at_4x(self):
        """8h+ idle → cooldown capped at 4.0x."""
        state = FakeState()
        cooldown = 1800
        # Last evolution: 6000s ago (> 3.3x cooldown).
        state.last_evolution_time = time.time() - 6000
        # Last task was 10 hours ago → multiplier = min(1+10/2, 4) = 4.0
        # idle_cooldown = 1800 * 4.0 = 7200 — 6000 < 7200, so blocked.
        last_task = time.time() - 10 * 3600

        should, plateaued = _should_evolve(
            state, cooldown_sec=cooldown, last_task_time=last_task,
        )
        assert should is False

    def test_idle_8h_past_cap_allows_evolution(self):
        """Even at 4x cap, if enough time has passed, evolution proceeds."""
        state = FakeState()
        cooldown = 1800
        # Last evolution: 8000s ago (> 4x cooldown = 7200).
        state.last_evolution_time = time.time() - 8000
        # Last task was 10 hours ago → multiplier = 4.0, idle_cooldown = 7200
        # 8000 > 7200 → allowed.
        last_task = time.time() - 10 * 3600

        should, plateaued = _should_evolve(
            state, cooldown_sec=cooldown, last_task_time=last_task,
        )
        assert should is True

    def test_no_task_time_no_decay(self):
        """last_task_time=0 (default) → no idle decay applied."""
        state = FakeState()
        state.last_evolution_time = time.time() - 2000

        should, plateaued = _should_evolve(
            state, cooldown_sec=1800, last_task_time=0,
        )
        assert should is True

    def test_idle_under_2h_no_decay(self):
        """< 2h idle → no decay applied (normal cooldown only)."""
        state = FakeState()
        state.last_evolution_time = time.time() - 2000  # past base cooldown
        last_task = time.time() - 3600  # 1 hour ago

        should, plateaued = _should_evolve(
            state, cooldown_sec=1800, last_task_time=last_task,
        )
        assert should is True


class TestShouldEvolvePlateau:
    """Plateau detection still works with idle decay."""

    def test_plateau_returns_false_true(self):
        """Plateaued + no directive → (False, True)."""
        state = FakeState()
        state.last_evolution_time = time.time() - 5000
        fitness = FakeFitness(plateaued=True)

        should, plateaued = _should_evolve(
            state, cooldown_sec=1800, fitness=fitness,
            has_directive=False,
        )
        assert should is False
        assert plateaued is True

    def test_plateau_with_directive_allows_evolution(self):
        """Plateaued + directive → (True, True)."""
        state = FakeState()
        state.last_evolution_time = time.time() - 5000
        fitness = FakeFitness(plateaued=True)

        should, plateaued = _should_evolve(
            state, cooldown_sec=1800, fitness=fitness,
            has_directive=True,
        )
        assert should is True
        assert plateaued is True

    def test_p0_active_blocks(self):
        """P0 active → always skip."""
        state = FakeState()
        state.p0_active.set()
        state.last_evolution_time = 0

        should, plateaued = _should_evolve(state, cooldown_sec=0)
        assert should is False
        assert plateaued is False

    def test_p1_active_blocks(self):
        """P1 active → always skip."""
        state = FakeState()
        state.p1_active.set()
        state.last_evolution_time = 0

        should, plateaued = _should_evolve(state, cooldown_sec=0)
        assert should is False
        assert plateaued is False
