"""Reproducible management of pseudo-random seeds.

Principle: no global random state is read or modified. Every source of
randomness explicitly receives a ``numpy.random.Generator`` (``make_rng``) or a
valid integer ``random_state`` (``validate_seed``, for scikit-learn). The same
seed gives the same stream whatever the global state.
"""

import numpy as np

# Seeds fixed by the protocol.
SEED_GLOBAL = 42
SEED_POLYNOMIAL = 42  # nominal data set and illustrative 1D figures
SEEDS_STAT_1D = tuple(range(1000, 1050))  # 50 noise seeds, disjoint from 42
SEEDS_MONTE_CARLO = (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021)

_SEED_MAX = 2**32 - 1  # upper bound of random_state for scikit-learn


def validate_seed(seed) -> int:
    """Validate a seed and return it as a Python ``int``.

    Accepts Python and NumPy integers in [0, 2**32 - 1]. Rejects ``None``
    (non-reproducible draw), booleans and floats (silent or ambiguous
    conversion).
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"integer seed expected, got {type(seed).__name__} : {seed!r}")
    value = int(seed)
    if not 0 <= value <= _SEED_MAX:
        raise ValueError(f"seed outside [0, 2**32 - 1]: {value}")
    return value


def make_rng(seed) -> np.random.Generator:
    """Independent generator ``np.random.default_rng(seed)`` without global side effects."""
    return np.random.default_rng(validate_seed(seed))
