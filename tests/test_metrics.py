"""Tests of src/utils/metrics.py.

Covers: analytical results, agreement with scikit-learn, rejection of mismatching
shapes (no spurious broadcasting), invalid inputs, conditioning.
Command: python -m pytest tests/test_metrics.py -v
"""

import logging

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as sk_r2_score

from src.utils.metrics import (
    KAPPA_HTH_ALERT,
    condition_numbers,
    getMSE,
    max_abs_error,
    r2_score,
    rmse,
)


def vandermonde(x, degree):
    """H[i, j] = x_i**j, j = 0..degree (uncentred matrix)."""
    return np.vander(x, degree + 1, increasing=True)


# --------------------------------------------------------------------------
# getMSE: analytical results
# --------------------------------------------------------------------------
class TestGetMSEAnalytic:
    def test_identical_vectors_give_zero(self):
        assert getMSE([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_constant_unit_offset_gives_one(self):
        assert getMSE(np.zeros(4), np.ones(4)) == 1.0

    def test_known_value(self):
        # (1^2 + 2^2 + 3^2) / 3 = 14/3
        assert getMSE([1, 2, 3], [2, 4, 6]) == pytest.approx(14.0 / 3.0, rel=1e-15)

    def test_symmetric_in_arguments(self):
        a, b = np.array([0.3, -1.2, 5.0]), np.array([0.1, 0.4, 4.0])
        assert getMSE(a, b) == getMSE(b, a)

    def test_integer_inputs_do_not_truncate(self):
        # 0.5 expected: an integer division would give 0
        assert getMSE([0, 1], [1, 1]) == 0.5

    def test_returns_python_float(self):
        assert type(getMSE([1.0], [2.0])) is float

    def test_matches_sklearn_to_1e_12(self):
        rng = np.random.default_rng(0)
        y, p = rng.normal(size=200), rng.normal(size=200)
        assert abs(getMSE(y, p) - mean_squared_error(y, p)) <= 1e-12

    def test_multi_output_equal_shapes_average_all_entries(self):
        y = np.arange(6.0).reshape(3, 2)
        p = y + 2.0
        assert getMSE(y, p) == 4.0

    def test_inputs_are_not_mutated(self):
        y, p = np.array([1.0, 2.0]), np.array([3.0, 5.0])
        y0, p0 = y.copy(), p.copy()
        getMSE(y, p)
        assert np.array_equal(y, y0) and np.array_equal(p, p0)


# --------------------------------------------------------------------------
# getMSE: shapes, types, invalid values
# --------------------------------------------------------------------------
class TestGetMSEValidation:
    def test_column_vector_vs_flat_vector_is_rejected(self):
        # The (N, 1) vs (N,) case. Without a check, numpy broadcasts to
        # (N, N) and returns a wrong value without an error.
        y = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.array([1.5, 2.5, 2.5, 4.5])
        silent = np.mean((y.reshape(-1, 1) - p) ** 2)
        assert silent != pytest.approx(getMSE(y, p))  # the trap is real
        with pytest.raises(AssertionError, match="shape"):
            getMSE(y.reshape(-1, 1), p)
        with pytest.raises(AssertionError, match="shape"):
            getMSE(y, p.reshape(-1, 1))

    def test_different_lengths_are_rejected(self):
        with pytest.raises(AssertionError, match="shape"):
            getMSE(np.zeros(3), np.zeros(4))

    def test_scalar_is_rejected(self):
        with pytest.raises(ValueError, match="ndim"):
            getMSE(1.0, 2.0)

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            getMSE(np.array([]), np.array([]))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_is_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            getMSE([1.0, bad], [1.0, 2.0])
        with pytest.raises(ValueError, match="finite"):
            getMSE([1.0, 2.0], [1.0, bad])

    @pytest.mark.parametrize("bad", [["a", "b"], [True, False], [1 + 2j, 3 + 0j]])
    def test_non_real_numeric_dtype_is_rejected(self, bad):
        with pytest.raises(TypeError, match="dtype"):
            getMSE(bad, [1.0, 2.0])


# --------------------------------------------------------------------------
# rmse, max_abs_error, r2_score
# --------------------------------------------------------------------------
class TestDerivedMetrics:
    def test_rmse_is_sqrt_of_mse(self):
        # MSE([0, 0], [3, 4]) = 12.5
        assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5), rel=1e-15)

    def test_max_abs_error_known_value(self):
        assert max_abs_error([1.0, 2.0, 3.0], [1.0, 5.0, 2.0]) == 3.0

    def test_r2_perfect_prediction_is_one(self):
        y = np.array([1.0, 2.0, 4.0, 8.0])
        assert r2_score(y, y) == 1.0

    def test_r2_mean_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0, 6.0])
        assert r2_score(y, np.full(4, y.mean())) == pytest.approx(0.0, abs=1e-15)

    def test_r2_matches_sklearn(self):
        rng = np.random.default_rng(1)
        y, p = rng.normal(size=50), rng.normal(size=50)
        assert r2_score(y, p) == pytest.approx(sk_r2_score(y, p), abs=1e-12)

    def test_r2_constant_target_is_rejected(self):
        with pytest.raises(ValueError, match="variance"):
            r2_score([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])

    def test_r2_requires_1d_vectors(self):
        a = np.arange(6.0).reshape(3, 2)
        with pytest.raises(ValueError, match="1D"):
            r2_score(a, a)

    @pytest.mark.parametrize("fn", [rmse, max_abs_error, r2_score])
    def test_shape_mismatch_is_rejected(self, fn):
        with pytest.raises(AssertionError, match="shape"):
            fn(np.arange(3.0).reshape(-1, 1), np.arange(3.0))


# --------------------------------------------------------------------------
# condition_numbers
# --------------------------------------------------------------------------
class TestConditionNumbers:
    def test_identity(self):
        c = condition_numbers(np.eye(3))
        assert c.kappa_h == pytest.approx(1.0, rel=1e-14)
        assert c.kappa_hth == pytest.approx(1.0, rel=1e-14)

    def test_diagonal_known_values(self):
        c = condition_numbers(np.diag([1.0, 10.0]))
        assert c.kappa_h == pytest.approx(10.0, rel=1e-14)
        assert c.kappa_hth == pytest.approx(100.0, rel=1e-14)

    def test_matches_numpy_cond_on_vandermonde(self):
        x = np.linspace(0.0, 1.2, 20)
        for d in range(1, 13):
            h = vandermonde(x, d)
            c = condition_numbers(h)
            assert c.kappa_h == pytest.approx(np.linalg.cond(h), rel=1e-9)
            assert c.kappa_hth == pytest.approx(c.kappa_h**2, rel=1e-12)

    def test_vandermonde_conditioning_grows_with_degree(self):
        # Nested columns: kappa cannot decrease (interlacing).
        x = np.linspace(0.0, 1.2, 20)
        kappas = [condition_numbers(vandermonde(x, d)).kappa_hth for d in range(1, 13)]
        assert all(b >= a for a, b in zip(kappas, kappas[1:]))
        assert kappas[0] < KAPPA_HTH_ALERT < kappas[-1]

    def test_zero_column_gives_infinite_kappa(self):
        c = condition_numbers(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
        assert c.kappa_h == np.inf and c.kappa_hth == np.inf

    def test_duplicated_column_is_numerically_singular(self):
        h = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert condition_numbers(h).kappa_h > 1e15

    def test_underdetermined_system_has_singular_normal_matrix(self):
        # N = 3 < p = 5 (case n_samples = 10, d >= 10): H^T H has rank <= 3 < 5.
        h = np.random.default_rng(2).normal(size=(3, 5))
        c = condition_numbers(h)
        assert np.isfinite(c.kappa_h)  # conditioning of H itself
        assert c.kappa_hth == np.inf  # but H^T H is singular

    def test_overflow_gives_inf_not_exception(self):
        h = np.diag([1e200, 1e-200])
        assert condition_numbers(h).kappa_hth == np.inf

    @pytest.mark.parametrize(
        "bad, exc",
        [
            (np.ones(3), ValueError),  # 1D
            (np.ones((2, 2, 2)), ValueError),  # 3D
            (np.empty((0, 3)), ValueError),  # empty
            (np.array([[1.0, np.nan], [1.0, 2.0]]), ValueError),
            (np.array([["a", "b"], ["c", "d"]]), TypeError),
        ],
    )
    def test_invalid_matrix_is_rejected(self, bad, exc):
        with pytest.raises(exc):
            condition_numbers(bad)

    def test_warns_above_threshold_without_raising(self, caplog):
        h = vandermonde(np.linspace(0.0, 1.2, 20), 12)
        with caplog.at_level(logging.INFO, logger="polynomial_mlp.metrics"):
            c = condition_numbers(h)
        assert c.kappa_hth > KAPPA_HTH_ALERT
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].getMessage() == "conditioning_alert"
        assert warnings[0].fields["kappa_hth"] == c.kappa_hth

    def test_context_fields_are_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger="polynomial_mlp.metrics"):
            condition_numbers(np.eye(2), degree=1)
        assert caplog.records[0].fields["degree"] == 1

    def test_no_warning_below_threshold(self, caplog):
        h = vandermonde(np.linspace(0.0, 1.2, 20), 2)
        with caplog.at_level(logging.INFO, logger="polynomial_mlp.metrics"):
            condition_numbers(h)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------
# collisions between context fields and internal logging fields
# --------------------------------------------------------------------------
class TestConditionNumbersContextCollision:
    """A context key identical to an internal field raises an explicit ValueError.

    Before the fix: opaque ``TypeError`` (duplicate named argument) for
    ``kappa_h``, ``kappa_hth``, ``threshold``, ``shape``, ``level``, ``event``
    and ``logger``.
    """

    @pytest.mark.parametrize(
        "key",
        ["kappa_h", "kappa_hth", "threshold", "shape", "level", "event", "logger", "time"],
    )
    def test_reserved_context_key_is_rejected_with_explicit_error(self, key):
        with pytest.raises(ValueError, match=key):
            condition_numbers(np.eye(2), **{key: 1})

    def test_all_colliding_keys_are_reported_together(self):
        with pytest.raises(ValueError) as info:
            condition_numbers(np.eye(2), shape=1, threshold=2)
        assert "shape" in str(info.value) and "threshold" in str(info.value)

    def test_rejection_happens_before_any_event_is_logged(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="polynomial_mlp.metrics"):
            with pytest.raises(ValueError):
                condition_numbers(np.eye(2), kappa_h=1.0)
        assert caplog.records == []

    def test_free_context_keys_still_work(self, caplog):
        with caplog.at_level(logging.INFO, logger="polynomial_mlp.metrics"):
            condition_numbers(np.eye(2), degree=3, method="svd", n_samples=2)
        fields = caplog.records[0].fields
        assert (fields["degree"], fields["method"], fields["n_samples"]) == (3, "svd", 2)
