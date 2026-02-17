"""Tests for ring0.parameter_seed -- deterministic evolution parameters."""

import pytest

from ring0.parameter_seed import EvolutionParams, generate_params, params_to_dict


class TestReproducibility:
    """Same seed + generation must always yield identical params."""

    def test_same_inputs_produce_same_params(self):
        a = generate_params(generation=0, seed=42)
        b = generate_params(generation=0, seed=42)
        assert a == b

    def test_reproducible_across_calls(self):
        results = [generate_params(generation=5, seed=99) for _ in range(10)]
        assert all(r == results[0] for r in results)


class TestUniqueness:
    """Different seeds or generations must (almost certainly) diverge."""

    def test_different_seeds_produce_different_params(self):
        a = generate_params(generation=0, seed=1)
        b = generate_params(generation=0, seed=2)
        assert a != b

    def test_different_generations_produce_different_params(self):
        a = generate_params(generation=0, seed=42)
        b = generate_params(generation=1, seed=42)
        assert a != b


class TestImmutability:
    """EvolutionParams (NamedTuple) must reject attribute assignment."""

    def test_cannot_set_mutation_rate(self):
        params = generate_params(generation=0, seed=7)
        with pytest.raises(AttributeError):
            params.mutation_rate = 0.99

    def test_cannot_set_population_size(self):
        params = generate_params(generation=0, seed=7)
        with pytest.raises(AttributeError):
            params.population_size = 100


class TestParamRanges:
    """Every generated field must stay within its documented bounds."""

    @pytest.fixture(params=range(50))
    def params(self, request):
        return generate_params(generation=request.param, seed=request.param * 7)

    def test_mutation_rate_bounds(self, params):
        assert 0.01 <= params.mutation_rate <= 0.50

    def test_population_size_bounds(self, params):
        assert 2 <= params.population_size <= 10

    def test_max_runtime_sec_bounds(self, params):
        assert params.max_runtime_sec == 600

    def test_crossover_rate_bounds(self, params):
        assert 0.1 <= params.crossover_rate <= 0.9


class TestSerialization:
    """params_to_dict must round-trip cleanly."""

    def test_returns_dict(self):
        d = params_to_dict(generate_params(generation=0, seed=1))
        assert isinstance(d, dict)

    def test_dict_keys_match_fields(self):
        d = params_to_dict(generate_params(generation=0, seed=1))
        assert set(d.keys()) == set(EvolutionParams._fields)

    def test_dict_values_match_params(self):
        p = generate_params(generation=3, seed=10)
        d = params_to_dict(p)
        assert d["generation"] == p.generation
        assert d["seed"] == p.seed
        assert d["mutation_rate"] == p.mutation_rate
