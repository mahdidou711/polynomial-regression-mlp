"""Data generation."""

from .generator_1d import (
    N_TEST_NOMINAL,
    N_SAMPLES_NOMINAL,
    SIGMA_NOMINAL,
    XMAX_NOMINAL,
    XMIN_NOMINAL,
    target_function,
    generate_training_set,
    generate_test_set,
    generate_example_1d,
)

__all__ = [
    "N_TEST_NOMINAL",
    "N_SAMPLES_NOMINAL",
    "SIGMA_NOMINAL",
    "XMAX_NOMINAL",
    "XMIN_NOMINAL",
    "target_function",
    "generate_training_set",
    "generate_test_set",
    "generate_example_1d",
]
