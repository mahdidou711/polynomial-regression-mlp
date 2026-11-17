"""Tests of src/utils/seeding.py.

Covers: fixed constants, reproducibility, absence of side effects on the
global random state, seed validation.
Command: python -m pytest tests/test_seeding.py -v
"""

import random
import re
from pathlib import Path

import numpy as np
import pytest

from src.utils.seeding import (
    SEED_GLOBAL,
    SEED_POLYNOMIAL,
    SEEDS_MONTE_CARLO,
    SEEDS_STAT_1D,
    make_rng,
    validate_seed,
)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


class TestContractConstants:
    def test_nominal_seeds(self):
        assert SEED_GLOBAL == 42
        assert SEED_POLYNOMIAL == 42

    def test_stat_1d_seeds_are_50_distinct_and_disjoint_from_42(self):
        assert SEEDS_STAT_1D == tuple(range(1000, 1050))
        assert len(set(SEEDS_STAT_1D)) == 50
        assert SEED_POLYNOMIAL not in SEEDS_STAT_1D

    def test_monte_carlo_seeds_are_the_frozen_sequence(self):
        assert SEEDS_MONTE_CARLO == (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021)
        assert len(set(SEEDS_MONTE_CARLO)) == 10

    def test_all_contract_seeds_are_valid(self):
        for s in (SEED_GLOBAL, SEED_POLYNOMIAL, *SEEDS_STAT_1D, *SEEDS_MONTE_CARLO):
            assert validate_seed(s) == s


class TestMakeRng:
    def test_returns_numpy_generator(self):
        assert isinstance(make_rng(0), np.random.Generator)

    def test_same_seed_gives_identical_stream(self):
        a = make_rng(42).normal(size=100)
        b = make_rng(42).normal(size=100)
        assert np.array_equal(a, b)  # binary equality

    def test_different_seeds_give_different_streams(self):
        assert not np.array_equal(make_rng(42).normal(size=10), make_rng(43).normal(size=10))

    def test_does_not_touch_global_numpy_state(self):
        before = np.random.get_state()
        make_rng(7).normal(size=1000)
        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1])
        assert before[2:] == after[2:]

    def test_does_not_touch_python_random_state(self):
        before = random.getstate()
        make_rng(7).normal(size=1000)
        assert random.getstate() == before

    def test_stream_is_independent_of_global_state(self):
        first = make_rng(5).random(20)
        np.random.random(100)  # perturbs the global state (legacy API)
        random.random()
        assert np.array_equal(make_rng(5).random(20), first)

    def test_numpy_integer_seed_is_accepted(self):
        assert np.array_equal(make_rng(np.int64(42)).random(5), make_rng(42).random(5))


class TestValidateSeed:
    def test_returns_python_int(self):
        s = validate_seed(np.int64(42))
        assert s == 42 and type(s) is int

    def test_accepts_bounds(self):
        assert validate_seed(0) == 0
        assert validate_seed(2**32 - 1) == 2**32 - 1

    @pytest.mark.parametrize("bad", [None, 1.0, "42", True, False, [1]])
    def test_rejects_non_integer(self, bad):
        with pytest.raises(TypeError, match="seed"):
            validate_seed(bad)

    @pytest.mark.parametrize("bad", [-1, 2**32])
    def test_rejects_out_of_range(self, bad):
        # 2**32 - 1 is the maximum accepted by scikit-learn's random_state
        with pytest.raises(ValueError, match="seed"):
            validate_seed(bad)

    def test_make_rng_rejects_none(self):
        # None would give a non-reproducible draw
        with pytest.raises(TypeError, match="seed"):
            make_rng(None)


def test_no_global_seeding_call_in_src():
    """No use of the global NumPy/random seed."""
    pattern = re.compile(r"\b(?:np|numpy)\.random\.seed\s*\(|\brandom\.seed\s*\(")
    offenders = [
        str(p.relative_to(SRC_DIR.parent))
        for p in SRC_DIR.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
