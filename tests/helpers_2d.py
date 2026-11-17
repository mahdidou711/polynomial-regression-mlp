"""Shared test helpers: synthetic data sets and short budgets."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

FAST = {"max_iter": 2, "max_fun": 20}  # budget override: tests only


def synthetic(n_train=200, n_val=40, n_test=30):
    """Distinct, non-constant (training, validation, test) sets. 200 points:
    avoids the scikit-learn warning 'batch_size > n_samples' with batch_size = 32 / 200."""
    rng = np.random.default_rng(7)
    out = []
    for n in (n_train, n_val, n_test):
        x = rng.random((n, 2))
        out.append((x, np.sin(3 * x[:, 0]) * x[:, 1] + 0.1 * x[:, 0]))
    return tuple(out)


@pytest.fixture
def one_thread():
    with threadpool_limits(limits=1):
        yield
