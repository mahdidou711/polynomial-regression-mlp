"""Generation and sampling of the 1D data.

Target function:
    f(x) = sin(-pi + 2*pi*x) = -sin(2*pi*x)

Reference sampling convention: ``np.arange(xmin, xmax, (xmax - xmin) / n_samples)``,
that is n_samples equally spaced points on the half-open interval [xmin, xmax)
(upper bound EXCLUDED, step = (xmax - xmin) / n_samples).
Here ``np.linspace(xmin, xmax, n_samples, endpoint=False)`` guarantees a number of
points equal to n_samples. Its abscissas are bit-for-bit identical to those of
``np.arange`` on the grids used by the project (n_samples in {10, 15, 20, 50,
100, 200} on [0; 1.2)); elsewhere they only agree to a few ulps (maximal
measured difference 2.6e-15 for n_samples from 1 to 399 and three domains), and
``np.arange`` can return n_samples + 1 points (37 cases out of 1197). These
findings are checked by the tests.

The Gaussian noise uses a local ``numpy.random.Generator`` (``make_rng``):
the global random state is neither read nor modified.
"""

import numbers

import numpy as np

from src.utils.seeding import SEED_GLOBAL, make_rng

# Nominal parameters.
XMIN_NOMINAL = 0.0
XMAX_NOMINAL = 1.2
N_SAMPLES_NOMINAL = 20
SIGMA_NOMINAL = 0.2
N_TEST_NOMINAL = 200


def target_function(x):
    """Target function ``sin(-pi + 2*pi*x)`` (noise-free)."""
    return np.sin(-np.pi + 2 * np.asarray(x, dtype=float) * np.pi)


def _check_real(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} : real expected, got {type(value).__name__} : {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite: {value}")
    return value


def _check_count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} : integer expected, got {type(value).__name__} : {value!r}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1: {value}")
    return value


def generate_example_1d(xmin, xmax, n_samples, sigma, seed=SEED_GLOBAL):
    """Noisy 1D data set with an explicit seed.

    Returns ``(x, y)`` with ``x`` of shape (n_samples, 1) (reference abscissas,
    interval [xmin, xmax)) and ``y`` of shape (n_samples,) = f(x) + N(0, sigma^2).
    ``seed`` (default 42, nominal seed) drives a local generator; it is
    validated even if ``sigma == 0``.
    """
    xmin = _check_real(xmin, "xmin")
    xmax = _check_real(xmax, "xmax")
    n_samples = _check_count(n_samples, "n_samples")
    sigma = _check_real(sigma, "sigma")
    if not xmin < xmax:
        raise ValueError(f"empty or reversed domain: xmin={xmin}, xmax={xmax}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0: {sigma}")
    rng = make_rng(seed)

    x = np.linspace(xmin, xmax, n_samples, endpoint=False)
    assert x.shape == (n_samples,)
    y = target_function(x) + rng.normal(loc=0.0, scale=sigma, size=x.size)
    return x.reshape(-1, 1), y


def generate_training_set(n_samples=N_SAMPLES_NOMINAL, sigma=SIGMA_NOMINAL, seed=SEED_GLOBAL,
                              xmin=XMIN_NOMINAL, xmax=XMAX_NOMINAL):
    """Training set (nominal: 20 points, sigma = 0.2, seed 42)."""
    return generate_example_1d(xmin, xmax, n_samples, sigma, seed=seed)


def generate_test_set(n_test=N_TEST_NOMINAL, xmin=XMIN_NOMINAL, xmax=XMAX_NOMINAL):
    """Noise-free test set (nominal: 200 points on [0; 1.2))."""
    return generate_example_1d(xmin, xmax, n_test, 0.0)
