"""Tests de src/models/polynomial.py (etape 2).

Couvre : construction exacte de Vandermonde, terme constant unique, coefficients de
polynomes connus, predictions et residus, concordance OLS / QR / SVD / scikit-learn selon
le regime de conditionnement (ARB-08, avenant 7.10), mauvais conditionnement (grille
nominale, AE1-07), systemes carres, sous-determines et singuliers, robustesse des erreurs.

Les seuils de regime sont calcules sur les matrices effectivement construites (jamais
supposes a partir du seul degre). Les bornes d'erreur sont de la forme C * kappa * eps
avec kappa = kappa_2(H) (QR, SVD) ou kappa(H^T H) (equations normales).
Commande : python -m pytest tests/test_polynomial_solvers.py -v
"""

import logging
from unittest import mock

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import src.models as models
from src.data import genere_base_apprentissage, genere_base_test
from src.models import polynomial
from src.models.polynomial import (
    EPS_0,
    MIXED_ATOL,
    MIXED_RTOL,
    STRICT_RTOL,
    RankDeficientError,
    compare_solvers,
    fit_polynomial,
    predict,
    relative_error_inf,
    residuals,
    solve_normal_equations,
    solve_qr,
    solve_sklearn_reference,
    solve_svd,
    vandermonde,
)
from src.utils.metrics import KAPPA_HTH_ALERT, condition_numbers, getMSE
from src.utils.seeding import SEEDS_STAT_1D

EPS = np.finfo(np.float64).eps
NB_EX_GRID = (10, 15, 20, 50, 100, 200)  # ARB-01
DEGREES = tuple(range(1, 13))
SEEDS = SEEDS_STAT_1D[:5]
LOGGER_METRICS = "c0_tp2_mlp.metrics"
LOGGER_POLY = "c0_tp2_mlp.polynomial"
ALL_SOLVERS = [solve_normal_equations, solve_qr, solve_svd, solve_sklearn_reference]


def nominal_grid():
    """Abscisses d'apprentissage nominales : 20 points sur [0 ; 1,2[ (borne exclue)."""
    return np.linspace(0.0, 1.2, 20, endpoint=False)


def true_coefficients(degree):
    """Coefficients de reference d'amplitude O(1), de signes alternes."""
    return np.array([(-1.0) ** j * (1.0 + 0.5 * j) for j in range(degree + 1)])


# --------------------------------------------------------------------------
# Construction de Vandermonde
# --------------------------------------------------------------------------
class TestVandermonde:
    def test_entries_are_x_to_the_power_j(self):
        x = np.array([-1.5, 0.0, 0.3, 1.0, 2.0])
        h = vandermonde(x, 4)
        expected = np.array([[xi**j for j in range(5)] for xi in x])
        np.testing.assert_allclose(h, expected, rtol=1e-15, atol=0.0)

    @pytest.mark.parametrize("n, d", [(1, 0), (1, 3), (5, 0), (20, 5), (7, 12), (200, 12)])
    def test_shape_is_n_by_d_plus_1(self, n, d):
        assert vandermonde(np.linspace(0.0, 1.0, n), d).shape == (n, d + 1)

    def test_column_zero_is_ones_even_at_x_zero(self):
        h = vandermonde(np.array([0.0, 0.5, -2.0]), 3)
        assert np.array_equal(h[:, 0], np.ones(3))

    def test_column_one_is_x_and_columns_follow_increasing_powers(self):
        x = np.linspace(0.1, 1.1, 6)
        h = vandermonde(x, 5)
        assert np.array_equal(h[:, 1], x)
        np.testing.assert_allclose(h[:, 2], x**2, rtol=1e-15)

    def test_degree_zero_is_a_single_column_of_ones(self):
        assert np.array_equal(vandermonde(np.array([3.0, -4.0]), 0), np.ones((2, 1)))

    def test_matches_numpy_vander(self):
        x = nominal_grid()
        expected = np.vander(x, 13, increasing=True)
        np.testing.assert_allclose(vandermonde(x, 12), expected, rtol=1e-13)

    def test_square_determinant_is_product_of_differences(self):
        # det V = prod_{i<j} (x_j - x_i) pour la matrice de Vandermonde carree.
        x = np.array([0.0, 0.5, 1.0, 2.0])
        expected = np.prod([x[j] - x[i] for i in range(4) for j in range(i + 1, 4)])
        assert np.linalg.det(vandermonde(x, 3)) == pytest.approx(expected, rel=1e-12)

    def test_column_vector_input_gives_same_matrix_as_flat_input(self):
        x = nominal_grid()
        assert np.array_equal(vandermonde(x.reshape(-1, 1), 5), vandermonde(x, 5))

    def test_accepts_generator_output_shape(self):
        x, _ = genere_base_apprentissage()
        assert x.shape == (20, 1)
        assert vandermonde(x, 5).shape == (20, 6)

    def test_integer_abscissas_are_converted_to_float64(self):
        h = vandermonde(np.array([1, 2, 3]), 2)
        assert h.dtype == np.float64
        assert np.array_equal(h, np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9]], dtype=float))

    def test_input_is_not_mutated(self):
        x = nominal_grid()
        x0 = x.copy()
        vandermonde(x, 6)
        assert np.array_equal(x, x0)

    @pytest.mark.parametrize(
        "bad, exc",
        [
            (np.ones((3, 2)), ValueError),  # deux colonnes
            (np.ones((2, 2, 2)), ValueError),
            (np.array(1.0), ValueError),  # 0-D
            (np.array([]), ValueError),
            (np.array([1.0, np.nan]), ValueError),
            (np.array([1.0, np.inf]), ValueError),
            (np.array([True, False]), TypeError),
            (np.array(["a", "b"]), TypeError),
            (np.array([1 + 1j, 2.0]), TypeError),
        ],
    )
    def test_invalid_abscissas_are_rejected(self, bad, exc):
        with pytest.raises(exc):
            vandermonde(bad, 2)

    @pytest.mark.parametrize("bad, exc", [(-1, ValueError), (2.0, TypeError), (True, TypeError),
                                          (None, TypeError), ("3", TypeError)])
    def test_invalid_degree_is_rejected(self, bad, exc):
        with pytest.raises(exc):
            vandermonde(np.ones(3), bad)

    def test_numpy_integer_degree_is_accepted(self):
        assert vandermonde(np.ones(3), np.int64(2)).shape == (3, 3)

    def test_floating_point_overflow_is_reported(self):
        with pytest.raises(ValueError, match="depassement"):
            vandermonde(np.array([1e200]), 3)


# --------------------------------------------------------------------------
# Terme constant : colonne de biais unique (EXG-PROG-B1-03)
# --------------------------------------------------------------------------
class TestSingleBias:
    def test_reference_is_instantiated_with_fit_intercept_false(self):
        h = vandermonde(nominal_grid(), 3)
        y = h @ true_coefficients(3)
        with mock.patch.object(polynomial, "LinearRegression", wraps=LinearRegression) as spy:
            solve_sklearn_reference(h, y)
        spy.assert_called_once_with(fit_intercept=False)

    def test_reference_returns_p_coefficients_and_no_extra_bias(self):
        h = vandermonde(nominal_grid(), 4)
        y = h @ true_coefficients(4)
        coef = solve_sklearn_reference(h, y)
        assert coef.shape == (5,)
        np.testing.assert_allclose(coef, true_coefficients(4), rtol=1e-8)

    def test_a_fitted_intercept_would_change_beta0(self):
        # Contre-test : avec fit_intercept=True, la colonne de uns de H est centree a zero
        # et beta_0 est perdu (double biais). Le test echoue donc si le drapeau est oublie.
        h = vandermonde(nominal_grid(), 3)
        y = h @ true_coefficients(3)
        wrong = LinearRegression(fit_intercept=True).fit(h, y).coef_
        assert abs(wrong[0]) < 1e-9 and abs(true_coefficients(3)[0]) > 0.5
        assert abs(solve_sklearn_reference(h, y)[0] - true_coefficients(3)[0]) < 1e-8

    def test_constant_data_gives_beta0_equal_to_the_constant(self):
        x = nominal_grid()
        for solver in (solve_normal_equations, solve_qr, solve_svd, solve_sklearn_reference):
            coef = solver(vandermonde(x, 3), np.full(20, 3.5))
            np.testing.assert_allclose(coef, [3.5, 0.0, 0.0, 0.0], atol=1e-9)

    def test_residuals_sum_to_zero_because_the_model_contains_a_constant(self):
        x, y = genere_base_apprentissage()
        fit = fit_polynomial(x, y, 3, "svd")
        assert abs(residuals(fit.coef, x, y).sum()) < 1e-12


# --------------------------------------------------------------------------
# Coefficients de polynomes connus
# --------------------------------------------------------------------------
def solver_error_bound(method, kappa_h, kappa_hth, c=50.0):
    """Borne relative c * kappa * eps (kappa(H^T H) pour les equations normales)."""
    return c * (kappa_hth if method == "normal" else kappa_h) * EPS


# Couples (methode, degre) dont la borne 50 * kappa * eps reste < 1e-2 sur la grille nominale :
# QR et SVD jusqu'a d = 12 ; equations normales jusqu'a d = 8 (kappa(H^T H) <= 1,9e11).
RECOVERY_CASES = [("qr", d) for d in DEGREES] + [("svd", d) for d in DEGREES] + [
    ("normal", d) for d in range(1, 9)
]


class TestKnownPolynomials:
    @pytest.mark.parametrize("method, degree", RECOVERY_CASES)
    def test_exact_polynomial_is_recovered_within_conditioning_bound(self, method, degree):
        x = nominal_grid()
        beta = true_coefficients(degree)
        y = vandermonde(x, degree) @ beta
        cond = condition_numbers(vandermonde(x, degree))
        bound = solver_error_bound(method, cond.kappa_h, cond.kappa_hth)
        assert bound < 1e-2  # le couple est bien dans le domaine exploitable
        fit = fit_polynomial(x, y, degree, method)
        assert relative_error_inf(fit.coef, beta) <= bound

    @pytest.mark.parametrize("degree", [9, 10, 11, 12])
    def test_normal_equations_bound_is_vacuous_beyond_degree_8(self, degree):
        cond = condition_numbers(vandermonde(nominal_grid(), degree))
        assert solver_error_bound("normal", cond.kappa_h, cond.kappa_hth) > 1e-2

    def test_line_is_recovered_with_degree_one(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        coef = fit_polynomial(x, 2.0 * x + 1.0, 1).coef
        np.testing.assert_allclose(coef, [1.0, 2.0], atol=1e-12)

    def test_overfitted_degree_gives_zero_higher_coefficients(self):
        x = nominal_grid()
        y = 1.0 - 2.0 * x + 0.5 * x**2
        coef = fit_polynomial(x, y, 5, "qr").coef
        np.testing.assert_allclose(coef[:3], [1.0, -2.0, 0.5], atol=1e-8)
        np.testing.assert_allclose(coef[3:], 0.0, atol=1e-7)

    def test_square_system_interpolates_exactly(self):
        x = np.array([0.0, 0.5, 1.0, 1.5])  # N = p = 4
        y = np.array([2.0, -1.0, 0.0, 3.0])
        for method in ("normal", "qr", "svd"):
            fit = fit_polynomial(x, y, 3, method)
            np.testing.assert_allclose(predict(fit.coef, x), y, atol=1e-10)

    def test_single_point_degree_zero_returns_the_value(self):
        assert fit_polynomial([0.7], [4.2], 0).coef == pytest.approx([4.2], rel=1e-14)


# --------------------------------------------------------------------------
# Predictions et residus
# --------------------------------------------------------------------------
class TestPredictionsAndResiduals:
    def test_predict_matches_polyval(self):
        coef = np.array([1.0, -2.0, 0.5, 3.0])
        x = np.array([-0.3, 0.0, 0.4, 1.2, 1.5])
        np.testing.assert_allclose(predict(coef, x), np.polyval(coef[::-1], x), rtol=1e-12)

    def test_predict_shape_is_n_for_flat_and_column_inputs(self):
        coef = np.array([1.0, 2.0])
        assert predict(coef, np.zeros(4)).shape == (4,)
        assert predict(coef, np.zeros((4, 1))).shape == (4,)

    def test_predict_on_unseen_points_uses_the_fitted_polynomial(self):
        x, y = genere_base_apprentissage()
        fit = fit_polynomial(x, y, 3)
        x_test, _ = genere_base_test()
        assert np.array_equal(fit.predict(x_test), predict(fit.coef, x_test))

    def test_residuals_are_y_minus_prediction(self):
        x, y = genere_base_apprentissage()
        coef = fit_polynomial(x, y, 4).coef
        assert np.array_equal(residuals(coef, x, y), y - predict(coef, x))

    def test_exact_polynomial_has_negligible_residuals(self):
        x = nominal_grid()
        y = vandermonde(x, 4) @ true_coefficients(4)
        coef = fit_polynomial(x, y, 4, "qr").coef
        assert np.max(np.abs(residuals(coef, x, y))) < 1e-9

    @pytest.mark.parametrize("method", ["qr", "svd"])
    def test_least_squares_residual_is_orthogonal_to_the_columns_of_h(self, method):
        x, y = genere_base_apprentissage()
        h = vandermonde(x, 3)
        r = residuals(fit_polynomial(x, y, 3, method).coef, x, y)
        assert np.max(np.abs(h.T @ r)) <= 1e-12 * np.linalg.norm(h) * np.linalg.norm(y)

    def test_least_squares_solution_minimizes_the_mse(self):
        x, y = genere_base_apprentissage()
        coef = fit_polynomial(x, y, 3).coef
        base = getMSE(y, predict(coef, x))
        for j in range(4):
            for step in (-1e-3, 1e-3):
                shifted = coef.copy()
                shifted[j] += step
                assert getMSE(y, predict(shifted, x)) > base

    def test_residual_mse_equals_getmse(self):
        x, y = genere_base_apprentissage()
        coef = fit_polynomial(x, y, 5).coef
        mse = getMSE(y, predict(coef, x))
        assert np.mean(residuals(coef, x, y) ** 2) == pytest.approx(mse, rel=1e-12)

    def test_column_vector_target_is_rejected_not_broadcast(self):
        x, y = genere_base_apprentissage()
        with pytest.raises(AssertionError, match="divergentes"):
            residuals([1.0, 1.0], x, y.reshape(-1, 1))

    @pytest.mark.parametrize("bad", [[], [np.nan, 1.0], np.ones((2, 2))])
    def test_invalid_coefficients_are_rejected(self, bad):
        with pytest.raises(ValueError):
            predict(bad, np.zeros(3))


# --------------------------------------------------------------------------
# Concordance OLS / QR / SVD / scikit-learn selon le regime (ARB-08)
# --------------------------------------------------------------------------
def comparisons():
    """Toutes les comparaisons de la grille ARB-01 x degres 1..12 x 5 graines de bruit."""
    for n in NB_EX_GRID:
        for seed in SEEDS:
            x, y = genere_base_apprentissage(n, 0.2, seed)
            for d in DEGREES:
                yield n, seed, d, compare_solvers(x, y, d)


@pytest.fixture(scope="module")
def grid_comparisons():
    logging.disable(logging.WARNING)  # evite 360 alertes de conditionnement sur stderr
    try:
        return list(comparisons())
    finally:
        logging.disable(logging.NOTSET)


class TestAgreementAcrossRegimes:
    def test_strict_regime_normal_and_qr_match_svd_to_1e_7(self, grid_comparisons):
        strict = [(n, s, d, c) for n, s, d, c in grid_comparisons if c.regime == "strict"]
        assert len(strict) >= 100
        for n, seed, d, c in strict:
            assert c.kappa_hth <= KAPPA_HTH_ALERT and c.rank == c.n_params
            assert c.rel_error_inf["normal"] <= STRICT_RTOL, (n, seed, d)
            assert c.rel_error_inf["qr"] <= STRICT_RTOL, (n, seed, d)
            assert c.within_strict_tolerance("normal") and c.within_strict_tolerance("qr")

    def test_strict_regime_has_margin_on_the_tolerance(self, grid_comparisons):
        worst = max(
            c.rel_error_inf["normal"] for _, _, _, c in grid_comparisons if c.regime == "strict"
        )
        assert worst < STRICT_RTOL / 100  # mesure : ~5e-10 au pire

    def test_strict_regime_mixed_criterion_also_holds(self, grid_comparisons):
        for n, seed, d, c in grid_comparisons:
            if c.regime == "strict":
                assert c.mixed_tolerance_ok["normal"] and c.mixed_tolerance_ok["qr"], (n, seed, d)

    def test_sklearn_reference_equals_svd_in_every_regime(self, grid_comparisons):
        for n, seed, d, c in grid_comparisons:
            assert c.rel_error_inf["sklearn"] <= 1e-12, (n, seed, d)

    def test_qr_stays_accurate_in_the_diagnostic_regime(self, grid_comparisons):
        for n, seed, d, c in grid_comparisons:
            if c.rank == c.n_params:
                bound = solver_error_bound("qr", c.kappa_h, c.kappa_hth)
                assert c.rel_error_inf["qr"] <= bound, (n, seed, d)

    def test_normal_equations_error_follows_kappa_h_squared(self, grid_comparisons):
        for n, seed, d, c in grid_comparisons:
            err = c.rel_error_inf["normal"]
            if c.rank == c.n_params and err is not None and np.isfinite(err):
                assert err <= solver_error_bound("normal", c.kappa_h, c.kappa_hth), (n, seed, d)

    def test_diagnostic_regime_never_raises_and_reports_an_error_value(self, grid_comparisons):
        diagnostic = [c for _, _, _, c in grid_comparisons if c.regime == "diagnostic"]
        assert diagnostic
        for c in diagnostic:
            assert set(c.rel_error_inf) == {"normal", "qr", "sklearn"}
            assert c.rel_error_inf["sklearn"] is not None  # SVD et reference toujours calculees
            if c.rank == c.n_params:  # rang complet (donc N >= p) : les trois methodes tournent
                assert c.rel_error_inf["normal"] is not None and c.rel_error_inf["qr"] is not None
            else:  # N < p ou rang deficient : equations normales et QR qualifiees non applicables
                assert set(c.errors) == {"normal", "qr"}

    def test_regime_is_determined_by_kappa_not_by_degree(self, grid_comparisons):
        # Le regime suit kappa(H^T H) mesure sur la matrice construite, pas le seul degre.
        for n, seed, d, c in grid_comparisons:
            is_strict = c.rank == c.n_params and c.kappa_hth <= KAPPA_HTH_ALERT
            assert c.regime == ("strict" if is_strict else "diagnostic"), (n, seed, d)
        # Meme degre (5) et meme N (20), intervalle different : le regime change (mesures :
        # kappa(H^T H) = 6,2e6 sur [0 ; 1,2[ et 5,3e8 sur [0 ; 0,6[).
        y = np.ones(20)
        wide = compare_solvers(np.linspace(0.0, 1.2, 20, endpoint=False), y, 5)
        narrow = compare_solvers(np.linspace(0.0, 0.6, 20, endpoint=False), y, 5)
        assert (wide.regime, narrow.regime) == ("strict", "diagnostic")


class TestNominalGridRegimes:
    """Seuils de l'etape 2 verifies sur la grille nominale effective (AE1-07)."""

    def setup_method(self):
        self.x, self.y = genere_base_apprentissage()
        logging.disable(logging.WARNING)

    def teardown_method(self):
        logging.disable(logging.NOTSET)

    def test_nominal_abscissas_are_the_half_open_grid(self):
        assert np.array_equal(self.x[:, 0], nominal_grid())
        assert self.x.max() == pytest.approx(1.14) and 1.2 not in self.x

    @pytest.mark.parametrize(
        "degree, kappa_hth, kappa_h",
        [(5, 6.22e6, 2.494e3), (6, 1.83e8, 1.352e4), (12, 5.3e17, 7.30e8)],
    )
    def test_documented_condition_numbers_are_reproduced(self, degree, kappa_hth, kappa_h):
        c = condition_numbers(vandermonde(self.x, degree))
        assert c.kappa_hth == pytest.approx(kappa_hth, rel=1e-2)
        assert c.kappa_h == pytest.approx(kappa_h, rel=1e-2)

    def test_threshold_1e7_separates_degree_5_from_degree_6(self):
        kappas = {d: condition_numbers(vandermonde(self.x, d)).kappa_hth for d in DEGREES}
        assert all(kappas[d] <= KAPPA_HTH_ALERT for d in range(1, 6))
        assert all(kappas[d] > KAPPA_HTH_ALERT for d in range(6, 13))

    def test_regime_is_strict_up_to_degree_5_then_diagnostic(self):
        regimes = {d: compare_solvers(self.x, self.y, d).regime for d in DEGREES}
        assert [d for d, r in regimes.items() if r == "strict"] == [1, 2, 3, 4, 5]
        assert [d for d, r in regimes.items() if r == "diagnostic"] == list(range(6, 13))

    def test_kappa_hth_is_kappa_h_squared_for_full_column_rank(self):
        for d in DEGREES:
            c = condition_numbers(vandermonde(self.x, d))
            assert c.kappa_hth == pytest.approx(c.kappa_h**2, rel=1e-12)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_explicit_normal_matrix_confirms_the_square_law_at_low_degree(self, degree):
        # Verification a bas degre seulement : H^T H explicite est alors fiable.
        h = vandermonde(self.x, degree)
        assert np.linalg.cond(h.T @ h) == pytest.approx(np.linalg.cond(h) ** 2, rel=1e-6)

    def test_condition_number_depends_on_the_actual_abscissas(self):
        # Grille a borne incluse (1,2 atteint) : kappa different de la grille nominale.
        closed = np.linspace(0.0, 1.2, 20, endpoint=True)
        k_open = condition_numbers(vandermonde(self.x, 5)).kappa_hth
        k_closed = condition_numbers(vandermonde(closed, 5)).kappa_hth
        assert k_closed == pytest.approx(5.5e6, rel=0.05)
        assert abs(k_open - k_closed) / k_open > 0.05


class TestIllConditioningConsequences:
    def setup_method(self):
        self.x, self.y = genere_base_apprentissage()
        logging.disable(logging.WARNING)

    def teardown_method(self):
        logging.disable(logging.NOTSET)

    def test_normal_equations_lose_the_strict_tolerance_at_degree_12(self):
        c = compare_solvers(self.x, self.y, 12)
        assert c.regime == "diagnostic"
        assert c.rel_error_inf["normal"] > STRICT_RTOL
        assert not c.within_strict_tolerance("normal")
        assert not c.mixed_tolerance_ok["normal"]

    def test_qr_beats_normal_equations_by_orders_of_magnitude_at_degree_12(self):
        c = compare_solvers(self.x, self.y, 12)
        assert c.rel_error_inf["qr"] < 1e-9
        assert c.rel_error_inf["normal"] > 1e6 * c.rel_error_inf["qr"]

    def test_normal_equations_error_grows_with_the_degree(self):
        err = {d: compare_solvers(self.x, self.y, d).rel_error_inf["normal"] for d in (6, 9, 12)}
        assert err[6] < err[9] < err[12]
        assert err[12] > 1e3 * err[6]

    def test_svd_and_reference_remain_identical_at_degree_12(self):
        c = compare_solvers(self.x, self.y, 12)
        assert np.array_equal(c.coefficients["svd"], c.coefficients["sklearn"])
        assert np.all(np.isfinite(c.coefficients["svd"]))

    def test_alert_is_logged_without_blocking(self, caplog):
        logging.disable(logging.NOTSET)
        with caplog.at_level(logging.INFO, logger="c0_tp2_mlp"):
            c = compare_solvers(self.x, self.y, 12)
        alerts = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(alerts) == 1 and alerts[0].getMessage() == "conditioning_alert"
        assert alerts[0].fields["degree"] == 12
        assert alerts[0].fields["kappa_hth"] == c.kappa_hth

    def test_comparison_event_records_the_measured_gap(self, caplog):
        logging.disable(logging.NOTSET)
        with caplog.at_level(logging.INFO, logger=LOGGER_POLY):
            c = compare_solvers(self.x, self.y, 12)
        events = [r for r in caplog.records if r.getMessage() == "solver_comparison"]
        assert len(events) == 1 and events[0].levelno == logging.INFO
        fields = events[0].fields
        assert fields["rel_error_normal"] == c.rel_error_inf["normal"]
        assert fields["regime"] == "diagnostic" and fields["degree"] == 12
        assert fields["not_applicable"] == []

    def test_no_alert_in_the_strict_regime(self, caplog):
        logging.disable(logging.NOTSET)
        with caplog.at_level(logging.INFO, logger="c0_tp2_mlp"):
            compare_solvers(self.x, self.y, 3)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_fit_logs_conditioning_with_degree_and_method_context(self, caplog):
        logging.disable(logging.NOTSET)
        with caplog.at_level(logging.INFO, logger=LOGGER_METRICS):
            fit_polynomial(self.x, self.y, 4, "qr")
        fields = caplog.records[0].fields
        assert (fields["degree"], fields["method"], fields["n_samples"]) == (4, "qr", 20)


# --------------------------------------------------------------------------
# Systemes carres, sous-determines et singuliers
# --------------------------------------------------------------------------
class TestUnderdetermined:
    """NbEx = 10 avec d >= 10 : p = d + 1 > N (H^T H singuliere)."""

    def setup_method(self):
        logging.disable(logging.WARNING)
        self.x, self.y = genere_base_apprentissage(10, 0.2, 42)

    def teardown_method(self):
        logging.disable(logging.NOTSET)

    @pytest.mark.parametrize("degree", [10, 11, 12])
    def test_normal_and_qr_are_not_applicable_and_reported_not_raised(self, degree):
        c = compare_solvers(self.x, self.y, degree)
        assert c.n_samples == 10 and c.n_params == degree + 1 and c.rank == 10
        assert c.regime == "diagnostic"
        assert c.coefficients["normal"] is None and c.coefficients["qr"] is None
        assert set(c.errors) == {"normal", "qr"}
        assert all("RankDeficientError" in msg and "N = 10" in msg for msg in c.errors.values())
        assert c.rel_error_inf["normal"] is None and c.mixed_tolerance_ok["qr"] is None
        assert not c.within_strict_tolerance("normal")

    @pytest.mark.parametrize("degree", [10, 11, 12])
    def test_hth_is_singular_while_kappa_h_is_finite(self, degree):
        c = compare_solvers(self.x, self.y, degree)
        assert np.isfinite(c.kappa_h) and c.kappa_hth == np.inf

    @pytest.mark.parametrize("degree", [10, 11, 12])
    def test_svd_returns_the_minimum_norm_interpolating_solution(self, degree):
        h = vandermonde(self.x, degree)
        beta = solve_svd(h, self.y)
        assert beta.shape == (degree + 1,)
        # (1) interpole les 10 points ; (2) norme minimale : beta orthogonal au noyau de H.
        assert np.linalg.norm(h @ beta - self.y) <= 1e-6 * np.linalg.norm(self.y)
        _, s, vt = np.linalg.svd(h)
        null_basis = vt[10:]
        assert null_basis.shape[0] == degree + 1 - 10
        assert np.max(np.abs(null_basis @ beta)) <= 1e-6 * np.linalg.norm(beta)

    @pytest.mark.parametrize("degree", [10, 12])
    def test_sklearn_reference_gives_the_same_minimum_norm_solution(self, degree):
        c = compare_solvers(self.x, self.y, degree)
        assert np.array_equal(c.coefficients["sklearn"], c.coefficients["svd"])

    def test_direct_solvers_raise_rank_deficient_error(self):
        h = vandermonde(self.x, 10)
        for solver in (solve_normal_equations, solve_qr):
            with pytest.raises(RankDeficientError, match="N = 10 < p = 11"):
                solver(h, self.y)

    @pytest.mark.parametrize("method", ["normal", "qr"])
    def test_fit_with_a_direct_method_raises_while_svd_succeeds(self, method):
        with pytest.raises(RankDeficientError):
            fit_polynomial(self.x, self.y, 11, method)
        fit = fit_polynomial(self.x, self.y, 11, "svd")
        assert fit.coef.shape == (12,) and fit.rank == 10

    def test_auto_method_falls_back_to_svd_when_underdetermined(self):
        fit = fit_polynomial(self.x, self.y, 11, "auto")  # N = 10 < p = 12 : pas d'exception
        assert fit.method == "svd" and fit.rank == 10
        assert np.array_equal(fit.coef, fit_polynomial(self.x, self.y, 11, "svd").coef)

    def test_auto_method_uses_qr_when_full_column_rank(self):
        x, y = genere_base_apprentissage()
        fit = fit_polynomial(x, y, 3, "auto")
        assert fit.method == "qr"
        assert np.array_equal(fit.coef, fit_polynomial(x, y, 3, "qr").coef)

    def test_auto_method_falls_back_to_svd_on_repeated_abscissas(self):
        fit = fit_polynomial(np.full(5, 0.3), np.arange(5.0), 2, "auto")
        assert fit.method == "svd" and fit.rank == 1

    def test_rank_deficient_error_is_a_linalg_error(self):
        assert issubclass(RankDeficientError, np.linalg.LinAlgError)

    def test_single_point_gives_the_analytic_minimum_norm_solution(self):
        x0, y0 = 0.5, 2.0
        v = np.array([1.0, x0, x0**2])
        np.testing.assert_allclose(fit_polynomial([x0], [y0], 2).coef, y0 * v / (v @ v), rtol=1e-12)

    def test_boundary_case_n_equals_p_is_still_solved_by_all_methods(self):
        x, y = genere_base_apprentissage(10, 0.2, 42)  # N = 10, d = 9 : p = N
        c = compare_solvers(x, y, 9)
        assert c.errors == {} and c.rank == 10
        np.testing.assert_allclose(predict(c.coefficients["svd"], x), y, atol=1e-6)


class TestRankDeficient:
    def setup_method(self):
        logging.disable(logging.WARNING)

    def teardown_method(self):
        logging.disable(logging.NOTSET)

    def test_identical_abscissas_give_rank_one(self):
        x = np.full(5, 0.3)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = compare_solvers(x, y, 2)
        assert c.rank == 1 and c.regime == "diagnostic"
        assert set(c.errors) == {"normal", "qr"}
        assert all("rang(H) = 1 < p = 3" in msg for msg in c.errors.values())

    def test_identical_abscissas_svd_matches_the_analytic_minimum_norm_solution(self):
        x = np.full(5, 0.3)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.array([1.0, 0.3, 0.09])
        expected = y.mean() * v / (v @ v)
        np.testing.assert_allclose(solve_svd(vandermonde(x, 2), y), expected, rtol=1e-12)
        reference = solve_sklearn_reference(vandermonde(x, 2), y)
        np.testing.assert_allclose(reference, expected, rtol=1e-12)

    @pytest.mark.parametrize("degree", [2, 4, 6])
    def test_rank_equals_min_of_distinct_abscissas_and_p(self, degree):
        x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])  # 3 abscisses distinctes
        assert compare_solvers(x, np.arange(6.0), degree).rank == min(3, degree + 1)

    def test_repeated_abscissas_are_fitted_through_the_group_means(self):
        x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        y = np.array([1.0, 3.0, 4.0, 6.0, 0.0, 2.0])  # moyennes 2, 5, 1
        coef = fit_polynomial(x, y, 4, "svd").coef  # p = 5 > 3 : noyau non trivial
        at_nodes = predict(coef, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(at_nodes, [2.0, 5.0, 1.0], atol=1e-9)

    def test_all_zero_abscissas_give_rank_one_and_mean_target(self):
        x = np.zeros(4)
        y = np.array([1.0, 2.0, 3.0, 6.0])
        coef = fit_polynomial(x, y, 3, "svd").coef
        np.testing.assert_allclose(coef, [3.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_direct_solvers_reject_a_rank_deficient_matrix(self):
        h = vandermonde(np.full(5, 0.3), 2)
        for solver in (solve_normal_equations, solve_qr):
            with pytest.raises(RankDeficientError, match="rang"):
                solver(h, np.ones(5))

    def test_exactly_singular_matrix_does_not_crash_the_comparison(self):
        h = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # colonnes proportionnelles
        y = np.array([1.0, 2.0, 3.0])
        for solver in (solve_normal_equations, solve_qr):
            with pytest.raises(RankDeficientError):
                solver(h, y)
        np.testing.assert_allclose(h @ solve_svd(h, y), y, atol=1e-12)


# --------------------------------------------------------------------------
# Robustesse des traitements d'erreur
# --------------------------------------------------------------------------
class TestErrorHandling:
    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    def test_target_shape_mismatch_is_rejected(self, solver):
        h = vandermonde(np.linspace(0, 1, 6), 2)
        with pytest.raises(AssertionError, match="divergentes"):
            solver(h, np.ones((6, 1)))  # (N, 1) contre (N,) : pas de broadcasting
        with pytest.raises(AssertionError, match="divergentes"):
            solver(h, np.ones(5))

    @pytest.mark.parametrize("solver", ALL_SOLVERS)
    @pytest.mark.parametrize(
        "h, y, exc",
        [
            (np.ones(3), np.ones(3), ValueError),  # H 1D
            (np.empty((0, 2)), np.empty(0), ValueError),
            (np.array([[1.0, np.nan], [1.0, 2.0]]), np.ones(2), ValueError),
            (np.ones((2, 2)), np.array([1.0, np.inf]), ValueError),
            (np.array([["a", "b"], ["c", "d"]]), np.ones(2), TypeError),
            (np.ones((2, 2)), np.array([True, False]), TypeError),
        ],
    )
    def test_invalid_system_is_rejected(self, solver, h, y, exc):
        with pytest.raises(exc):
            solver(h, y)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="methode inconnue"):
            fit_polynomial(np.linspace(0, 1, 5), np.ones(5), 2, "cholesky")

    @pytest.mark.parametrize("bad_degree, exc", [(-1, ValueError), (2.5, TypeError)])
    def test_comparison_and_fit_validate_the_degree(self, bad_degree, exc):
        x, y = np.linspace(0, 1, 5), np.ones(5)
        with pytest.raises(exc):
            compare_solvers(x, y, bad_degree)
        with pytest.raises(exc):
            fit_polynomial(x, y, bad_degree)

    def test_nan_in_the_data_is_rejected_by_fit_and_comparison(self):
        x, y = np.linspace(0, 1, 5), np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            fit_polynomial(x, y, 2)
        with pytest.raises(ValueError):
            compare_solvers(x, y, 2)

    def test_inputs_are_not_mutated_by_any_solver(self):
        x, y = genere_base_apprentissage()
        x0, y0 = x.copy(), y.copy()
        for method in ("normal", "qr", "svd"):
            fit_polynomial(x, y, 3, method)
        compare_solvers(x, y, 3)
        assert np.array_equal(x, x0) and np.array_equal(y, y0)
        h = vandermonde(x, 3)
        h0 = h.copy()
        for solver in (solve_normal_equations, solve_qr, solve_svd, solve_sklearn_reference):
            solver(h, y)
        assert np.array_equal(h, h0)

    def test_results_are_float64_vectors_of_length_p(self):
        x, y = genere_base_apprentissage()
        h = vandermonde(x, 5)
        for solver in (solve_normal_equations, solve_qr, solve_svd, solve_sklearn_reference):
            coef = solver(h, y)
            assert coef.dtype == np.float64 and coef.shape == (6,)

    def test_solvers_are_deterministic_and_leave_global_random_state_untouched(self):
        x, y = genere_base_apprentissage()
        np.random.seed(7)
        state = np.random.get_state()[1].copy()
        first = [fit_polynomial(x, y, 5, m).coef for m in ("normal", "qr", "svd")]
        second = [fit_polynomial(x, y, 5, m).coef for m in ("normal", "qr", "svd")]
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
        assert np.array_equal(np.random.get_state()[1], state)

    def test_non_finite_normal_equations_result_is_qualified_not_raised(self):
        # H proche du debordement : H^T H deborde, le resultat est qualifie non applicable.
        x = np.linspace(1e100, 2e100, 6)
        c = compare_solvers(x, np.ones(6), 1)
        assert c.regime == "diagnostic"
        assert c.coefficients["svd"] is not None

    def test_fit_result_carries_conditioning_and_rank(self):
        x, y = genere_base_apprentissage()
        logging.disable(logging.WARNING)
        try:
            fit = fit_polynomial(x, y, 5, "qr")
        finally:
            logging.disable(logging.NOTSET)
        assert fit.degree == 5 and fit.method == "qr" and fit.rank == 6
        assert fit.kappa_hth == pytest.approx(fit.kappa_h**2, rel=1e-12)
        assert fit.coef.shape == (6,)


class TestRelativeError:
    def test_known_value_uses_the_infinity_norm(self):
        # inf : max(0,5 ; 0,5) / max(1 ; 2) = 0,25 ; la norme 2 donnerait ~0,316.
        assert relative_error_inf([1.5, 2.5], [1.0, 2.0]) == pytest.approx(0.25, rel=1e-15)

    def test_denominator_is_floored_by_eps0(self):
        assert relative_error_inf([1e-13], [0.0]) == pytest.approx(1e-13 / EPS_0, rel=1e-12)

    def test_identical_vectors_give_zero(self):
        assert relative_error_inf([1.0, -3.0], [1.0, -3.0]) == 0.0

    def test_non_finite_estimate_gives_infinity(self):
        assert relative_error_inf([np.nan, 1.0], [1.0, 1.0]) == np.inf
        assert relative_error_inf([np.inf, 1.0], [1.0, 1.0]) == np.inf

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(AssertionError):
            relative_error_inf([1.0, 2.0], [1.0])

    def test_documented_tolerances(self):
        assert (STRICT_RTOL, EPS_0, MIXED_ATOL, MIXED_RTOL) == (1e-7, 1e-12, 1e-10, 1e-7)
        assert KAPPA_HTH_ALERT == 1e7


# --------------------------------------------------------------------------
# Interface publique et coherence avec scikit-learn
# --------------------------------------------------------------------------
class TestPublicInterface:
    def test_all_exports_are_importable(self):
        for name in models.__all__:
            assert hasattr(models, name), name

    @pytest.mark.parametrize("degree", [1, 3, 5, 8])
    def test_matches_the_polynomial_features_pipeline_of_the_assignment(self, degree):
        x, y = genere_base_apprentissage()
        x_test, _ = genere_base_test()
        regressor = LinearRegression(fit_intercept=False)
        pipe = make_pipeline(PolynomialFeatures(degree), regressor).fit(x, y)
        logging.disable(logging.WARNING)
        try:
            fit = fit_polynomial(x, y, degree)
        finally:
            logging.disable(logging.NOTSET)
        np.testing.assert_allclose(fit.predict(x_test), pipe.predict(x_test), rtol=1e-8, atol=1e-8)

    def test_methods_are_the_three_solvers_of_the_specification(self):
        assert models.METHODS == ("normal", "qr", "svd")
        assert set(models.SOLVERS) == set(models.METHODS)
