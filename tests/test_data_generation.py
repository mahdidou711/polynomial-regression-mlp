"""Tests de src/data/generator_1d.py (etape 1).

Couvre : exactitude analytique de la cible, nombre d'echantillons, convention
d'echantillonnage de l'enseignant (np.arange semi-ouvert, borne haute exclue),
formes, bruit (distribution, amplitude, cas nul), reproductibilite, absence
d'effet de bord sur l'etat aleatoire global, validation des parametres.
Commande : python -m pytest tests/test_data_generation.py -v
"""

import numpy as np
import pytest

from src.data import (
    N_TEST_NOMINAL,
    NB_EX_NOMINAL,
    SIGMA_NOMINAL,
    XMAX_NOMINAL,
    XMIN_NOMINAL,
    fonction_cible,
    genere_base_apprentissage,
    genere_base_test,
    genere_exemple_dim1,
)
from src.utils.seeding import SEED_GLOBAL, SEEDS_STAT_1D


def _teacher_arange(xmin, xmax, nb_ex):
    """Abscisses de l'enonce (diapositive 9), copie litterale servant d'oracle."""
    return np.arange(xmin, xmax, (xmax - xmin) / nb_ex)


class TestTargetFunction:
    def test_matches_official_expression(self):
        x = np.linspace(-0.5, 1.7, 57)
        np.testing.assert_array_equal(fonction_cible(x), np.sin(-np.pi + 2 * x * np.pi))

    def test_equals_minus_sin_2pi_x(self):
        x = np.linspace(0.0, 1.2, 101)
        np.testing.assert_allclose(fonction_cible(x), -np.sin(2 * np.pi * x), atol=1e-12)

    @pytest.mark.parametrize(
        "x, expected",
        [(0.0, 0.0), (0.25, -1.0), (0.5, 0.0), (0.75, 1.0), (1.0, 0.0)],
    )
    def test_analytic_values(self, x, expected):
        assert fonction_cible(x) == pytest.approx(expected, abs=1e-12)

    def test_period_is_one(self):
        x = np.linspace(0.0, 0.2, 31)
        np.testing.assert_allclose(fonction_cible(x + 1.0), fonction_cible(x), atol=1e-12)

    def test_range_within_unit_interval(self):
        y = fonction_cible(np.linspace(0.0, 1.2, 1000))
        assert y.min() >= -1.0 and y.max() <= 1.0


class TestNominalConstants:
    def test_values(self):
        assert (XMIN_NOMINAL, XMAX_NOMINAL) == (0.0, 1.2)
        assert NB_EX_NOMINAL == 20
        assert SIGMA_NOMINAL == 0.2
        assert N_TEST_NOMINAL == 200


class TestSamplingConvention:
    @pytest.mark.parametrize("nb_ex", [1, 2, 3, 7, 10, 20, 50, 100, 200, 1000])
    def test_exact_sample_count(self, nb_ex):
        x, y = genere_exemple_dim1(0.0, 1.2, nb_ex, 0.0)
        assert x.shape[0] == nb_ex
        assert y.shape[0] == nb_ex

    @pytest.mark.parametrize("nb_ex", [20, 200])
    def test_abscissas_match_teacher_arange(self, nb_ex):
        """Memes abscisses que np.arange de l'enseignant (borne haute exclue)."""
        x, _ = genere_exemple_dim1(0.0, 1.2, nb_ex, 0.0)
        ref = _teacher_arange(0.0, 1.2, nb_ex)
        assert ref.size == nb_ex  # l'oracle lui-meme est conforme ici
        np.testing.assert_allclose(x[:, 0], ref, rtol=0, atol=1e-15)

    @pytest.mark.parametrize("nb_ex", [20, 200])
    def test_first_point_is_xmin_and_xmax_excluded(self, nb_ex):
        x, _ = genere_exemple_dim1(0.0, 1.2, nb_ex, 0.0)
        assert x[0, 0] == 0.0
        assert x.max() < 1.2
        step = 1.2 / nb_ex
        assert x[-1, 0] == pytest.approx(1.2 - step, abs=1e-12)

    @pytest.mark.parametrize("nb_ex", [20, 200])
    def test_equidistant_and_strictly_increasing(self, nb_ex):
        x, _ = genere_exemple_dim1(0.0, 1.2, nb_ex, 0.0)
        d = np.diff(x[:, 0])
        assert np.all(d > 0)
        np.testing.assert_allclose(d, 1.2 / nb_ex, atol=1e-12)

    def test_general_bounds(self):
        x, _ = genere_exemple_dim1(-1.0, 2.5, 35, 0.0)
        assert x[0, 0] == -1.0 and x.max() < 2.5 and x.shape[0] == 35

    @pytest.mark.parametrize("nb_ex", [10, 15, 20, 50, 100, 200])
    def test_abscissas_are_bitwise_identical_to_teacher_arange_on_project_grids(self, nb_ex):
        """AE1-03 : identite exacte uniquement sur les grilles du projet (ARB-01, [0 ; 1,2[)."""
        x, _ = genere_exemple_dim1(0.0, 1.2, nb_ex, 0.0)
        assert np.array_equal(x[:, 0], _teacher_arange(0.0, 1.2, nb_ex))

    def test_abscissas_agree_within_a_few_ulps_elsewhere_and_not_always_bitwise(self):
        """AE1-03 : hors des grilles du projet, accord a quelques ulps (mesure : 2,6e-15),
        sans identite bit a bit generale ; np.arange derive en NbEx + 1 points dans 37 cas."""
        same_size = not_bitwise = drift = 0
        worst = 0.0
        for nb_ex in range(1, 400):
            for xmin, xmax in ((0.0, 1.2), (0.0, 0.7), (0.1, 0.3)):
                ref = _teacher_arange(xmin, xmax, nb_ex)
                if ref.size != nb_ex:
                    drift += 1
                    continue
                x = genere_exemple_dim1(xmin, xmax, nb_ex, 0.0)[0][:, 0]
                same_size += 1
                not_bitwise += not np.array_equal(x, ref)
                worst = max(worst, float(np.max(np.abs(x - ref))))
        assert (same_size, drift) == (1160, 37)
        assert not_bitwise > 0  # l'affirmation « memes abscisses » serait fausse sans restriction
        assert worst <= 4e-15

    def test_count_is_exact_where_teacher_arange_would_drift(self):
        """Verrou du risque documente (EXG-PROG-B2-02) : le compte reste NbEx."""
        for nb_ex in range(1, 400):
            for xmin, xmax in ((0.0, 1.2), (0.0, 0.7), (0.1, 0.3)):
                x, _ = genere_exemple_dim1(xmin, xmax, nb_ex, 0.0)
                assert x.shape[0] == nb_ex


class TestShapesAndDtypes:
    def test_shapes(self):
        x, y = genere_exemple_dim1(0.0, 1.2, 20, 0.2)
        assert x.shape == (20, 1)
        assert y.shape == (20,)

    def test_dtypes_are_float64(self):
        x, y = genere_exemple_dim1(0.0, 1.2, 20, 0.2)
        assert x.dtype == np.float64 and y.dtype == np.float64

    def test_outputs_finite(self):
        x, y = genere_exemple_dim1(0.0, 1.2, 20, 0.2)
        assert np.isfinite(x).all() and np.isfinite(y).all()


class TestNoise:
    def test_sigma_zero_returns_exact_target(self):
        x, y = genere_exemple_dim1(0.0, 1.2, 200, 0.0)
        np.testing.assert_array_equal(y, fonction_cible(x[:, 0]))

    def test_sigma_zero_independent_of_seed(self):
        _, y1 = genere_exemple_dim1(0.0, 1.2, 50, 0.0, seed=1)
        _, y2 = genere_exemple_dim1(0.0, 1.2, 50, 0.0, seed=2)
        np.testing.assert_array_equal(y1, y2)

    def test_x_independent_of_sigma_and_seed(self):
        x1, _ = genere_exemple_dim1(0.0, 1.2, 20, 0.0)
        x2, _ = genere_exemple_dim1(0.0, 1.2, 20, 0.5, seed=7)
        np.testing.assert_array_equal(x1, x2)

    def test_noise_is_nonzero_when_sigma_positive(self):
        x, y = genere_exemple_dim1(0.0, 1.2, 20, 0.2)
        assert np.any(y != fonction_cible(x[:, 0]))

    @pytest.mark.parametrize("sigma", [0.05, 0.2, 1.0])
    def test_noise_mean_and_std(self, sigma):
        """Moyenne et ecart-type du bruit, bornes a 5 ecarts-types d'estimation."""
        n = 20000
        for seed in SEEDS_STAT_1D[:3]:
            x, y = genere_exemple_dim1(0.0, 1.2, n, sigma, seed=seed)
            eps = y - fonction_cible(x[:, 0])
            assert abs(eps.mean()) < 5 * sigma / np.sqrt(n)
            assert abs(eps.std(ddof=1) / sigma - 1) < 5 / np.sqrt(2 * (n - 1))

    def test_noise_is_gaussian_shape(self):
        """Kurtosis en exces proche de 0 et symetrie (asymetrie proche de 0)."""
        n = 50000
        x, y = genere_exemple_dim1(0.0, 1.2, n, 0.3, seed=SEEDS_STAT_1D[0])
        z = (y - fonction_cible(x[:, 0])) / 0.3
        skew = np.mean(z**3)
        exkurt = np.mean(z**4) - 3
        assert abs(skew) < 5 * np.sqrt(6 / n)
        assert abs(exkurt) < 5 * np.sqrt(24 / n)

    def test_noise_amplitude_scales_with_sigma(self):
        _, y_a = genere_exemple_dim1(0.0, 1.2, 5000, 0.1, seed=3)
        _, y_b = genere_exemple_dim1(0.0, 1.2, 5000, 0.4, seed=3)
        x, _ = genere_exemple_dim1(0.0, 1.2, 5000, 0.0)
        ea, eb = y_a - fonction_cible(x[:, 0]), y_b - fonction_cible(x[:, 0])
        # meme graine, meme flux standardise : le rapport vaut exactement sigma_b / sigma_a
        np.testing.assert_allclose(eb, 4 * ea, rtol=1e-9, atol=1e-12)

    def test_noise_matches_default_rng_normal_stream(self):
        x, y = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        eps = np.random.default_rng(42).normal(loc=0.0, scale=0.2, size=20)
        np.testing.assert_allclose(y, fonction_cible(x[:, 0]) + eps, rtol=0, atol=1e-15)


class TestReproducibility:
    def test_same_seed_bitwise_identical(self):
        x1, y1 = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        x2, y2 = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_default_seed_is_nominal_42(self):
        _, y_default = genere_exemple_dim1(0.0, 1.2, 20, 0.2)
        _, y_42 = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=SEED_GLOBAL)
        np.testing.assert_array_equal(y_default, y_42)

    def test_different_seeds_give_different_noise(self):
        _, y1 = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        _, y2 = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=43)
        assert not np.array_equal(y1, y2)

    def test_stat_seeds_all_accepted_and_distinct(self):
        ys = [genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=s)[1] for s in SEEDS_STAT_1D]
        assert len({y.tobytes() for y in ys}) == len(SEEDS_STAT_1D)

    def test_reproducible_despite_global_state_perturbation(self):
        _, y1 = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        np.random.random(500)
        np.random.seed(999)
        _, y2 = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        np.testing.assert_array_equal(y1, y2)

    def test_global_numpy_state_untouched(self):
        np.random.seed(123)
        before = np.random.get_state()
        genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        genere_base_apprentissage(seed=7)
        genere_base_test()
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]

    def test_global_python_random_untouched(self):
        import random

        random.seed(5)
        before = random.getstate()
        genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        assert random.getstate() == before


class TestNominalBases:
    def test_training_base_nominal(self):
        xa, ya = genere_base_apprentissage()
        x, y = genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=42)
        assert xa.shape == (20, 1) and ya.shape == (20,)
        np.testing.assert_array_equal(xa, x)
        np.testing.assert_array_equal(ya, y)

    def test_training_base_parametrable_for_explorations(self):
        xa, ya = genere_base_apprentissage(nb_ex=50, sigma=0.5, seed=1005)
        x, y = genere_exemple_dim1(0.0, 1.2, 50, 0.5, seed=1005)
        np.testing.assert_array_equal(xa, x)
        np.testing.assert_array_equal(ya, y)

    def test_test_base_nominal_is_noise_free(self):
        xt, yt = genere_base_test()
        assert xt.shape == (200, 1) and yt.shape == (200,)
        np.testing.assert_array_equal(yt, fonction_cible(xt[:, 0]))
        np.testing.assert_allclose(xt[:, 0], _teacher_arange(0.0, 1.2, 200), atol=1e-15)

    def test_test_base_has_no_seed_dependence(self):
        np.testing.assert_array_equal(genere_base_test()[1], genere_base_test()[1])

    def test_bases_share_domain(self):
        xa, _ = genere_base_apprentissage()
        xt, _ = genere_base_test()
        assert xa.min() >= 0.0 and xt.min() >= 0.0
        assert xa.max() < 1.2 and xt.max() < 1.2


class TestInvalidParameters:
    @pytest.mark.parametrize("nb_ex", [0, -1, -20])
    def test_nb_ex_not_positive(self, nb_ex):
        with pytest.raises(ValueError):
            genere_exemple_dim1(0.0, 1.2, nb_ex, 0.2)

    @pytest.mark.parametrize("nb_ex", [20.0, 20.5, "20", None, True])
    def test_nb_ex_wrong_type(self, nb_ex):
        with pytest.raises(TypeError):
            genere_exemple_dim1(0.0, 1.2, nb_ex, 0.2)

    def test_numpy_integer_nb_ex_accepted(self):
        x, _ = genere_exemple_dim1(0.0, 1.2, np.int64(20), 0.2)
        assert x.shape == (20, 1)

    @pytest.mark.parametrize("xmin, xmax", [(1.2, 0.0), (1.0, 1.0)])
    def test_empty_or_reversed_domain(self, xmin, xmax):
        with pytest.raises(ValueError):
            genere_exemple_dim1(xmin, xmax, 20, 0.2)

    @pytest.mark.parametrize(
        "xmin, xmax",
        [(np.nan, 1.2), (0.0, np.nan), (-np.inf, 1.2), (0.0, np.inf)],
    )
    def test_non_finite_bounds(self, xmin, xmax):
        with pytest.raises(ValueError):
            genere_exemple_dim1(xmin, xmax, 20, 0.2)

    @pytest.mark.parametrize("bound", ["0", None, True])
    def test_bound_wrong_type(self, bound):
        with pytest.raises(TypeError):
            genere_exemple_dim1(bound, 1.2, 20, 0.2)

    @pytest.mark.parametrize("sigma", [-0.1, np.nan, np.inf, -np.inf])
    def test_invalid_sigma_value(self, sigma):
        with pytest.raises(ValueError):
            genere_exemple_dim1(0.0, 1.2, 20, sigma)

    @pytest.mark.parametrize("sigma", ["0.2", None, True])
    def test_sigma_wrong_type(self, sigma):
        with pytest.raises(TypeError):
            genere_exemple_dim1(0.0, 1.2, 20, sigma)

    @pytest.mark.parametrize("seed", [None, 1.5, "42", True])
    def test_seed_wrong_type(self, seed):
        with pytest.raises(TypeError):
            genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=seed)

    @pytest.mark.parametrize("seed", [-1, 2**32])
    def test_seed_out_of_range(self, seed):
        with pytest.raises(ValueError):
            genere_exemple_dim1(0.0, 1.2, 20, 0.2, seed=seed)

    def test_invalid_seed_rejected_even_when_sigma_zero(self):
        with pytest.raises(TypeError):
            genere_exemple_dim1(0.0, 1.2, 20, 0.0, seed=None)

    def test_integer_bounds_accepted(self):
        x, _ = genere_exemple_dim1(0, 2, 10, 0)
        assert x.shape == (10, 1) and x[0, 0] == 0.0


class TestNoRegressionImplementedYet:
    def test_package_does_not_expose_polynomial_regression(self):
        import src.data as data

        assert not any("poly" in name.lower() for name in dir(data))
