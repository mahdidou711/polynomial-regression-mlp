"""Verrous independants des bases et de la campagne nominale receptionnee."""
from collections import Counter

import numpy as np
from scipy.linalg import lstsq

from src.data.generator_1d import genere_base_apprentissage, genere_base_test
from src.experiments.runner_1d_nominal import run_degree_sweep, run_nominal, run_nominal_campaign
from src.utils.seeding import SEED_POLYNOMIAL, SEEDS_STAT_1D


def test_bases_parametres_non_nominaux():
    x, y = genere_base_apprentissage(nb_ex=17, sigma=.3, seed=71, xmin=-.4, xmax=1.7)
    expected_x = np.linspace(-.4, 1.7, 17, endpoint=False)
    expected_y = np.sin(-np.pi + 2*np.pi*expected_x) + np.random.default_rng(71).normal(0, .3, 17)
    assert x.shape == (17, 1) and y.shape == (17,)
    np.testing.assert_array_equal(x[:, 0], expected_x)
    np.testing.assert_allclose(y, expected_y, atol=1e-15, rtol=0)
    x2, y2 = genere_base_apprentissage(nb_ex=17, sigma=.3, seed=71, xmin=-.4, xmax=1.7)
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    xt, yt = genere_base_test(n_test=37, xmin=-.3, xmax=1.5)
    assert xt.shape == (37, 1) and yt.shape == (37,)
    np.testing.assert_array_equal(xt[:, 0], np.linspace(-.3, 1.5, 37, endpoint=False))
    np.testing.assert_allclose(yt, np.sin(-np.pi + 2*np.pi*xt[:, 0]), atol=1e-15, rtol=0)
    xt2, yt2 = genere_base_test(n_test=37, xmin=-.3, xmax=1.5)
    np.testing.assert_array_equal(xt, xt2)
    np.testing.assert_array_equal(yt, yt2)


def test_verrou_nominal_et_oracle_svd_independant():
    assert SEED_POLYNOMIAL == 42
    assert SEEDS_STAT_1D == tuple(range(1000, 1050))
    nominal = run_nominal()
    assert nominal.optimal_degree == 6
    np.testing.assert_allclose(nominal.eqm_test[5], 0.0070030, rtol=1e-5)
    campaign = run_nominal_campaign()
    assert Counter(campaign.optimal_degrees) == {4: 2, 5: 14, 6: 24, 7: 6, 8: 3, 10: 1}
    x_test = np.linspace(0, 1.2, 200, endpoint=False)
    y_test = np.sin(-np.pi + 2*np.pi*x_test)
    train_rows, test_rows, noises = [], [], []
    for seed in SEEDS_STAT_1D:
        x = np.linspace(0, 1.2, 20, endpoint=False)
        noise = np.random.default_rng(seed).normal(0, .2, 20)
        noises.append(noise)
        y = np.sin(-np.pi + 2*np.pi*x) + noise
        train, test = [], []
        for degree in range(1, 13):
            h = np.vander(x, degree + 1, increasing=True)
            ht = np.vander(x_test, degree + 1, increasing=True)
            coef = lstsq(h, y, lapack_driver='gelsd')[0]
            train.append(np.mean((y - h @ coef)**2))
            test.append(np.mean((y_test - ht @ coef)**2))
        train_rows.append(train)
        test_rows.append(test)
    assert len({row.tobytes() for row in noises}) == 50
    np.testing.assert_allclose(campaign.eqm_train, train_rows, rtol=2e-7, atol=1e-11)
    np.testing.assert_allclose(campaign.eqm_test, test_rows, rtol=2e-7, atol=1e-10)
    np.testing.assert_allclose(np.median(test_rows, axis=0)[11], 1.29, rtol=.01)
    for name, rows in [('train', train_rows), ('test', test_rows)]:
        summary = campaign.summary(name)
        for key, value in [('mean', np.mean(rows, axis=0)), ('median', np.median(rows, axis=0)),
                           ('q1', np.percentile(rows, 25, axis=0)), ('q3', np.percentile(rows, 75, axis=0))]:
            np.testing.assert_allclose(summary[key], value, rtol=2e-7, atol=1e-10)
    x, y = genere_base_apprentissage(seed=1000)
    xt, yt = genere_base_test()
    fast = run_degree_sweep(x, y, xt, yt, diagnose=False)
    diagnosed = run_degree_sweep(x, y, xt, yt, diagnose=True)
    np.testing.assert_allclose(fast.y_pred_test, diagnosed.y_pred_test, rtol=1e-12, atol=1e-12)
    for degree in (1, 6, 12):
        h = np.vander(x[:, 0], degree + 1, increasing=True)
        ht = np.vander(xt[:, 0], degree + 1, increasing=True)
        expected = ht @ lstsq(h, y, lapack_driver='gelsd')[0]
        np.testing.assert_allclose(fast.y_pred_test[degree-1], expected, rtol=2e-7, atol=1e-8)
