"""Noise-free 2D data: training, validation and final test.

Each data set is a pair (X, y) of shapes (N, 2) and (N,). Validation is used to
choose the configuration; the test set is reserved for the final estimate, after
selection. The grids run over x2 fastest.
"""

from typing import NamedTuple

import numpy as np

from src.data.data_loader import DEFAULT_TRAINING_PATH, load_training_data_2d


class DataSplits2D(NamedTuple):
    training: tuple[np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]


def cardinal_sine(x) -> np.ndarray:
    """2D target function, vectorised over an (N, 2) array.

    z = A(-pi + 2*pi*x + b), h = ||z||; 1 if h < 0.001,
    sin(h)/h otherwise. Returns a vector of shape (N,), without noise.
    """
    values = np.asarray(x)
    if values.dtype.kind not in "iuf":
        raise TypeError("real numeric coordinates expected")
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("coordinates of shape (N, 2) with N > 0 expected")
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("non-finite coordinates or outside [0, 1]")
    scaled = -np.pi + 2 * values * np.pi
    z = (scaled + np.array([0.2, -0.3])) @ np.array([[1.0, -2.0], [1.0, 1.0]])
    h = np.linalg.norm(z, axis=1)
    y = np.ones_like(h)
    mask = h >= 1e-3
    y[mask] = np.sin(h[mask]) / h[mask]
    return y


def _grid(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1, x2 = np.meshgrid(axis, axis, indexing="ij")
    x = np.column_stack((x1.ravel(), x2.ravel()))
    return x, cardinal_sine(x)


def generate_validation_set_2d() -> tuple[np.ndarray, np.ndarray]:
    """400 cell centres; hyper-parameter selection."""
    return _grid((np.arange(20, dtype=float) + 0.5) / 20)


def generate_test_set_2d() -> tuple[np.ndarray, np.ndarray]:
    """1600 points, bounds included; final evaluation only."""
    return _grid(np.linspace(0.0, 1.0, 40))


def generate_training_set_2d(path=DEFAULT_TRAINING_PATH) -> tuple[np.ndarray, np.ndarray]:
    """200 input observations in file order; MLP fitting."""
    x = load_training_data_2d(path)
    return x, cardinal_sine(x)


def prepare_data_2d(path=DEFAULT_TRAINING_PATH) -> DataSplits2D:
    """Build the three data sets, without any random draw."""
    return DataSplits2D(generate_training_set_2d(path),
                        generate_validation_set_2d(), generate_test_set_2d())
