"""Tests of src/data/generator_1d.py.

Covers: analytical exactness of the target, number of samples, reference sampling
convention (half-open np.arange, upper bound excluded), shapes, noise (distribution,
amplitude, zero case), reproducibility, absence of side effects on the global random
state, parameter validation.
Command: python -m pytest tests/test_data_generation.py -v
"""

import numpy as np
import pytest

from src.data import (
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
from src.utils.seeding import SEED_GLOBAL, SEEDS_STAT_1D


def _reference_arange(xmin, xmax, n_samples):
    """Reference abscissas, literal copy used as an oracle."""
    return np.arange(xmin, xmax, (xmax - xmin) / n_samples)


class TestTargetFunction:
    def test_matches_reference_expression(self):
        x = np.linspace(-0.5, 1.7, 57)
        np.testing.assert_array_equal(target_function(x), np.sin(-np.pi + 2 * x * np.pi))

    def test_equals_minus_sin_2pi_x(self):
        x = np.linspace(0.0, 1.2, 101)
        np.testing.assert_allclose(target_function(x), -np.sin(2 * np.pi * x), atol=1e-12)

    @pytest.mark.parametrize(
        "x, expected",
        [(0.0, 0.0), (0.25, -1.0), (0.5, 0.0), (0.75, 1.0), (1.0, 0.0)],
    )
    def test_analytic_values(self, x, expected):
        assert target_function(x) == pytest.approx(expected, abs=1e-12)

    def test_period_is_one(self):
        x = np.linspace(0.0, 0.2, 31)
        np.testing.assert_allclose(target_function(x + 1.0), target_function(x), atol=1e-12)

    def test_range_within_unit_interval(self):
        y = target_function(np.linspace(0.0, 1.2, 1000))
        assert y.min() >= -1.0 and y.max() <= 1.0


class TestNominalConstants:
    def test_values(self):
        assert (XMIN_NOMINAL, XMAX_NOMINAL) == (0.0, 1.2)
        assert N_SAMPLES_NOMINAL == 20
        assert SIGMA_NOMINAL == 0.2
        assert N_TEST_NOMINAL == 200


class TestSamplingConvention:
    @pytest.mark.parametrize("n_samples", [1, 2, 3, 7, 10, 20, 50, 100, 200, 1000])
    def test_exact_sample_count(self, n_samples):
        x, y = generate_example_1d(0.0, 1.2, n_samples, 0.0)
        assert x.shape[0] == n_samples
        assert y.shape[0] == n_samples

    @pytest.mark.parametrize("n_samples", [20, 200])
    def test_abscissas_match_reference_arange(self, n_samples):
        """Same abscissas as the reference np.arange (upper bound excluded)."""
        x, _ = generate_example_1d(0.0, 1.2, n_samples, 0.0)
        ref = _reference_arange(0.0, 1.2, n_samples)
        assert ref.size == n_samples  # the oracle itself is consistent here
        np.testing.assert_allclose(x[:, 0], ref, rtol=0, atol=1e-15)

    @pytest.mark.parametrize("n_samples", [20, 200])
    def test_first_point_is_xmin_and_xmax_excluded(self, n_samples):
        x, _ = generate_example_1d(0.0, 1.2, n_samples, 0.0)
        assert x[0, 0] == 0.0
        assert x.max() < 1.2
        step = 1.2 / n_samples
        assert x[-1, 0] == pytest.approx(1.2 - step, abs=1e-12)

    @pytest.mark.parametrize("n_samples", [20, 200])
    def test_equidistant_and_strictly_increasing(self, n_samples):
        x, _ = generate_example_1d(0.0, 1.2, n_samples, 0.0)
        d = np.diff(x[:, 0])
        assert np.all(d > 0)
        np.testing.assert_allclose(d, 1.2 / n_samples, atol=1e-12)

    def test_general_bounds(self):
        x, _ = generate_example_1d(-1.0, 2.5, 35, 0.0)
        assert x[0, 0] == -1.0 and x.max() < 2.5 and x.shape[0] == 35

    @pytest.mark.parametrize("n_samples", [10, 15, 20, 50, 100, 200])
    def test_abscissas_are_bitwise_identical_to_reference_arange_on_project_grids(self, n_samples):
        """Exact identity only on the project grids ([0; 1.2))."""
        x, _ = generate_example_1d(0.0, 1.2, n_samples, 0.0)
        assert np.array_equal(x[:, 0], _reference_arange(0.0, 1.2, n_samples))

    def test_abscissas_agree_within_a_few_ulps_elsewhere_and_not_always_bitwise(self):
        """Outside the project grids, agreement to a few ulps (measured: 2.6e-15),
        without a general bit-for-bit identity; np.arange drifts to n_samples + 1 points in 37 cases."""
        same_size = not_bitwise = drift = 0
        worst = 0.0
        for n_samples in range(1, 400):
            for xmin, xmax in ((0.0, 1.2), (0.0, 0.7), (0.1, 0.3)):
                ref = _reference_arange(xmin, xmax, n_samples)
                if ref.size != n_samples:
                    drift += 1
                    continue
                x = generate_example_1d(xmin, xmax, n_samples, 0.0)[0][:, 0]
                same_size += 1
                not_bitwise += not np.array_equal(x, ref)
                worst = max(worst, float(np.max(np.abs(x - ref))))
        assert (same_size, drift) == (1160, 37)
        assert not_bitwise > 0  # the claim "same abscissas" would be false without the restriction
        assert worst <= 4e-15

    def test_count_is_exact_where_reference_arange_would_drift(self):
        """Lock for the documented risk: the count stays n_samples."""
        for n_samples in range(1, 400):
            for xmin, xmax in ((0.0, 1.2), (0.0, 0.7), (0.1, 0.3)):
                x, _ = generate_example_1d(xmin, xmax, n_samples, 0.0)
                assert x.shape[0] == n_samples


class TestShapesAndDtypes:
    def test_shapes(self):
        x, y = generate_example_1d(0.0, 1.2, 20, 0.2)
        assert x.shape == (20, 1)
        assert y.shape == (20,)

    def test_dtypes_are_float64(self):
        x, y = generate_example_1d(0.0, 1.2, 20, 0.2)
        assert x.dtype == np.float64 and y.dtype == np.float64

    def test_outputs_finite(self):
        x, y = generate_example_1d(0.0, 1.2, 20, 0.2)
        assert np.isfinite(x).all() and np.isfinite(y).all()


class TestNoise:
    def test_sigma_zero_returns_exact_target(self):
        x, y = generate_example_1d(0.0, 1.2, 200, 0.0)
        np.testing.assert_array_equal(y, target_function(x[:, 0]))

    def test_sigma_zero_independent_of_seed(self):
        _, y1 = generate_example_1d(0.0, 1.2, 50, 0.0, seed=1)
        _, y2 = generate_example_1d(0.0, 1.2, 50, 0.0, seed=2)
        np.testing.assert_array_equal(y1, y2)

    def test_x_independent_of_sigma_and_seed(self):
        x1, _ = generate_example_1d(0.0, 1.2, 20, 0.0)
        x2, _ = generate_example_1d(0.0, 1.2, 20, 0.5, seed=7)
        np.testing.assert_array_equal(x1, x2)

    def test_noise_is_nonzero_when_sigma_positive(self):
        x, y = generate_example_1d(0.0, 1.2, 20, 0.2)
        assert np.any(y != target_function(x[:, 0]))

    @pytest.mark.parametrize("sigma", [0.05, 0.2, 1.0])
    def test_noise_mean_and_std(self, sigma):
        """Mean and standard deviation of the noise, bounds at 5 estimation standard errors."""
        n = 20000
        for seed in SEEDS_STAT_1D[:3]:
            x, y = generate_example_1d(0.0, 1.2, n, sigma, seed=seed)
            eps = y - target_function(x[:, 0])
            assert abs(eps.mean()) < 5 * sigma / np.sqrt(n)
            assert abs(eps.std(ddof=1) / sigma - 1) < 5 / np.sqrt(2 * (n - 1))

    def test_noise_is_gaussian_shape(self):
        """Excess kurtosis close to 0 and symmetry (skewness close to 0)."""
        n = 50000
        x, y = generate_example_1d(0.0, 1.2, n, 0.3, seed=SEEDS_STAT_1D[0])
        z = (y - target_function(x[:, 0])) / 0.3
        skew = np.mean(z**3)
        exkurt = np.mean(z**4) - 3
        assert abs(skew) < 5 * np.sqrt(6 / n)
        assert abs(exkurt) < 5 * np.sqrt(24 / n)

    def test_noise_amplitude_scales_with_sigma(self):
        _, y_a = generate_example_1d(0.0, 1.2, 5000, 0.1, seed=3)
        _, y_b = generate_example_1d(0.0, 1.2, 5000, 0.4, seed=3)
        x, _ = generate_example_1d(0.0, 1.2, 5000, 0.0)
        ea, eb = y_a - target_function(x[:, 0]), y_b - target_function(x[:, 0])
        # same seed, same standardised stream: the ratio is exactly sigma_b / sigma_a
        np.testing.assert_allclose(eb, 4 * ea, rtol=1e-9, atol=1e-12)

    def test_noise_matches_default_rng_normal_stream(self):
        x, y = generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        eps = np.random.default_rng(42).normal(loc=0.0, scale=0.2, size=20)
        np.testing.assert_allclose(y, target_function(x[:, 0]) + eps, rtol=0, atol=1e-15)


class TestReproducibility:
    def test_same_seed_bitwise_identical(self):
        x1, y1 = generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        x2, y2 = generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_default_seed_is_nominal_42(self):
        _, y_default = generate_example_1d(0.0, 1.2, 20, 0.2)
        _, y_42 = generate_example_1d(0.0, 1.2, 20, 0.2, seed=SEED_GLOBAL)
        np.testing.assert_array_equal(y_default, y_42)

    def test_different_seeds_give_different_noise(self):
        _, y1 = generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        _, y2 = generate_example_1d(0.0, 1.2, 20, 0.2, seed=43)
        assert not np.array_equal(y1, y2)

    def test_stat_seeds_all_accepted_and_distinct(self):
        ys = [generate_example_1d(0.0, 1.2, 20, 0.2, seed=s)[1] for s in SEEDS_STAT_1D]
        assert len({y.tobytes() for y in ys}) == len(SEEDS_STAT_1D)

    def test_reproducible_despite_global_state_perturbation(self):
        _, y1 = generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        np.random.random(500)
        np.random.seed(999)
        _, y2 = generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        np.testing.assert_array_equal(y1, y2)

    def test_global_numpy_state_untouched(self):
        np.random.seed(123)
        before = np.random.get_state()
        generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        generate_training_set(seed=7)
        generate_test_set()
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]

    def test_global_python_random_untouched(self):
        import random

        random.seed(5)
        before = random.getstate()
        generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        assert random.getstate() == before


class TestNominalBases:
    def test_training_base_nominal(self):
        xa, ya = generate_training_set()
        x, y = generate_example_1d(0.0, 1.2, 20, 0.2, seed=42)
        assert xa.shape == (20, 1) and ya.shape == (20,)
        np.testing.assert_array_equal(xa, x)
        np.testing.assert_array_equal(ya, y)

    def test_training_base_parametrable_for_explorations(self):
        xa, ya = generate_training_set(n_samples=50, sigma=0.5, seed=1005)
        x, y = generate_example_1d(0.0, 1.2, 50, 0.5, seed=1005)
        np.testing.assert_array_equal(xa, x)
        np.testing.assert_array_equal(ya, y)

    def test_test_base_nominal_is_noise_free(self):
        xt, yt = generate_test_set()
        assert xt.shape == (200, 1) and yt.shape == (200,)
        np.testing.assert_array_equal(yt, target_function(xt[:, 0]))
        np.testing.assert_allclose(xt[:, 0], _reference_arange(0.0, 1.2, 200), atol=1e-15)

    def test_test_base_has_no_seed_dependence(self):
        np.testing.assert_array_equal(generate_test_set()[1], generate_test_set()[1])

    def test_bases_share_domain(self):
        xa, _ = generate_training_set()
        xt, _ = generate_test_set()
        assert xa.min() >= 0.0 and xt.min() >= 0.0
        assert xa.max() < 1.2 and xt.max() < 1.2


class TestInvalidParameters:
    @pytest.mark.parametrize("n_samples", [0, -1, -20])
    def test_n_samples_not_positive(self, n_samples):
        with pytest.raises(ValueError):
            generate_example_1d(0.0, 1.2, n_samples, 0.2)

    @pytest.mark.parametrize("n_samples", [20.0, 20.5, "20", None, True])
    def test_n_samples_wrong_type(self, n_samples):
        with pytest.raises(TypeError):
            generate_example_1d(0.0, 1.2, n_samples, 0.2)

    def test_numpy_integer_n_samples_accepted(self):
        x, _ = generate_example_1d(0.0, 1.2, np.int64(20), 0.2)
        assert x.shape == (20, 1)

    @pytest.mark.parametrize("xmin, xmax", [(1.2, 0.0), (1.0, 1.0)])
    def test_empty_or_reversed_domain(self, xmin, xmax):
        with pytest.raises(ValueError):
            generate_example_1d(xmin, xmax, 20, 0.2)

    @pytest.mark.parametrize(
        "xmin, xmax",
        [(np.nan, 1.2), (0.0, np.nan), (-np.inf, 1.2), (0.0, np.inf)],
    )
    def test_non_finite_bounds(self, xmin, xmax):
        with pytest.raises(ValueError):
            generate_example_1d(xmin, xmax, 20, 0.2)

    @pytest.mark.parametrize("bound", ["0", None, True])
    def test_bound_wrong_type(self, bound):
        with pytest.raises(TypeError):
            generate_example_1d(bound, 1.2, 20, 0.2)

    @pytest.mark.parametrize("sigma", [-0.1, np.nan, np.inf, -np.inf])
    def test_invalid_sigma_value(self, sigma):
        with pytest.raises(ValueError):
            generate_example_1d(0.0, 1.2, 20, sigma)

    @pytest.mark.parametrize("sigma", ["0.2", None, True])
    def test_sigma_wrong_type(self, sigma):
        with pytest.raises(TypeError):
            generate_example_1d(0.0, 1.2, 20, sigma)

    @pytest.mark.parametrize("seed", [None, 1.5, "42", True])
    def test_seed_wrong_type(self, seed):
        with pytest.raises(TypeError):
            generate_example_1d(0.0, 1.2, 20, 0.2, seed=seed)

    @pytest.mark.parametrize("seed", [-1, 2**32])
    def test_seed_out_of_range(self, seed):
        with pytest.raises(ValueError):
            generate_example_1d(0.0, 1.2, 20, 0.2, seed=seed)

    def test_invalid_seed_rejected_even_when_sigma_zero(self):
        with pytest.raises(TypeError):
            generate_example_1d(0.0, 1.2, 20, 0.0, seed=None)

    def test_integer_bounds_accepted(self):
        x, _ = generate_example_1d(0, 2, 10, 0)
        assert x.shape == (10, 1) and x[0, 0] == 0.0


class TestNoRegressionImplementedYet:
    def test_package_does_not_expose_polynomial_regression(self):
        import src.data as data

        assert not any("poly" in name.lower() for name in dir(data))
