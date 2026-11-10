"""Protocole contractuel et verification des explorations."""
import csv
from collections import Counter

import numpy as np
import pytest
from scipy.linalg import lstsq

from src.experiments.runner_1d_explorations import (Configuration, SIZES, SIGMAS,
    export_campaigns, export_extrapolation, run_campaign, run_extrapolation, size_diagnostics)
from src.experiments.runner_1d_nominal import run_nominal_campaign
from src.utils.seeding import SEEDS_STAT_1D
from src.visualization.plots_1d_explorations import save_exploration_figures


def test_grilles_et_points_nominaux():
    assert SIZES == (10, 15, 20, 50, 100, 200)
    assert SIGMAS == (0, .05, .1, .2, .5, 1)
    ref = run_nominal_campaign()
    for config in (Configuration(20, .2), Configuration(nb_ex=SIZES[2]), Configuration(sigma=SIGMAS[3])):
        c = run_campaign(config)
        assert c.seeds == SEEDS_STAT_1D
        np.testing.assert_array_equal(c.eqm_test, ref.eqm_test)
        assert Counter(c.optimal_degrees) == {4: 2, 5: 14, 6: 24, 7: 6, 8: 3, 10: 1}


def test_reproductibilite_variation_et_sans_bruit():
    a = run_campaign(Configuration(10, .2), seeds=(1000, 1001))
    b = run_campaign(Configuration(10, .2), seeds=(1000, 1001))
    np.testing.assert_array_equal(a.eqm_test, b.eqm_test)
    assert not np.array_equal(a.eqm_test[0], a.eqm_test[1])
    assert not np.array_equal(a.eqm_test, run_campaign(Configuration(15, .2), seeds=(1000,1001)).eqm_test)
    assert not np.array_equal(a.eqm_test, run_campaign(Configuration(10, .5), seeds=(1000,1001)).eqm_test)
    zero = run_campaign(Configuration(20, 0))
    assert zero.seeds == (42,) and zero.eqm_test.shape == (1, 12)
    with pytest.raises(ValueError):
        run_campaign(Configuration(), (1000, 1000))


def test_oracle_extrapolation_et_selection():
    rows = run_extrapolation((1000,))
    x = np.linspace(0, 1.2, 20, endpoint=False)
    y = np.sin(-np.pi + 2*np.pi*x) + np.random.default_rng(1000).normal(0, .2, 20)
    grid = np.linspace(-.3, 1.5, 301)
    truth = np.sin(-np.pi + 2*np.pi*grid)
    inside = (grid >= -1e-12) & (grid <= 1.2 + 1e-12)
    assert inside.sum() == 201
    for row in rows:
        degree = row['degree']
        beta = lstsq(np.vander(x, degree+1, increasing=True), y, lapack_driver='gelsd')[0]
        err = (np.vander(grid, degree+1, increasing=True) @ beta - truth)**2
        np.testing.assert_allclose(row['eqm_interp'], np.mean(err[inside]), rtol=2e-7, atol=1e-10)
        np.testing.assert_allclose(row['eqm_extrap'], np.mean(err[~inside]), rtol=2e-7, atol=1e-8)
    assert rows[2]['degree'] == rows[0]['cstar']


def test_exports_et_figures(tmp_path):
    campaigns = {10: run_campaign(Configuration(10, .2), (1000,1001))}
    extra = run_extrapolation((1000,1001))
    paths = export_campaigns(campaigns, 'nb_ex', tmp_path) + export_extrapolation(extra, tmp_path)
    with paths[0].open() as f:
        lines = list(csv.DictReader(f))
    assert len(lines) == 24 and {r['seed'] for r in lines} == {'1000', '1001'}
    with paths[1].open() as f:
        rows = list(csv.DictReader(f))
    np.testing.assert_allclose(float(rows[0]['eqmt_median']), np.median(campaigns[10].eqm_test[:, 0]))
    figures = save_exploration_figures(campaigns, {0: run_campaign(Configuration(20,0))}, extra, tmp_path)
    assert len(figures) == 8
    assert all(p.stat().st_size > 1000 for p in figures)


def test_diagnostics_variance_et_sous_determination():
    variances, diagnostics = size_diagnostics((10,), (1000, 1001))
    assert len(variances) == len(diagnostics) == 12
    assert all(row['mean_prediction_variance'] >= 0 for row in variances)
    x = np.linspace(0, 1.2, 10, endpoint=False)
    xt = np.linspace(0, 1.2, 200, endpoint=False)
    predictions = []
    for seed in (1000, 1001):
        y = np.sin(-np.pi + 2*np.pi*x) + np.random.default_rng(seed).normal(0, .2, 10)
        beta = lstsq(np.vander(x, 7, increasing=True), y, lapack_driver='gelsd')[0]
        predictions.append(np.vander(xt, 7, increasing=True) @ beta)
    expected = np.var(predictions, axis=0, ddof=1).mean()
    np.testing.assert_allclose(variances[5]['mean_prediction_variance'], expected, rtol=1e-8)
    for row in diagnostics[9:]:
        assert row['rank'] < row['n_parameters']
        assert np.isnan(row['normal_rel_error'])


def test_selection_cstar_sur_test_nominal_independante():
    seed = 1000
    x = np.linspace(0, 1.2, 20, endpoint=False)
    y = np.sin(-np.pi + 2*np.pi*x) + np.random.default_rng(seed).normal(0, .2, 20)
    nominal = np.linspace(0, 1.2, 200, endpoint=False)
    extended = np.linspace(-.3, 1.5, 301)
    outside = (extended < -1e-12) | (extended > 1.2 + 1e-12)
    errors_nominal, errors_extra = [], []
    for degree in range(1, 13):
        beta = lstsq(np.vander(x, degree+1, increasing=True), y, lapack_driver='gelsd')[0]
        for grid, errors in ((nominal, errors_nominal), (extended[outside], errors_extra)):
            prediction = np.vander(grid, degree+1, increasing=True) @ beta
            errors.append(np.mean((prediction - np.sin(-np.pi + 2*np.pi*grid))**2))
    expected = int(np.argmin(errors_nominal)) + 1
    assert expected != int(np.argmin(errors_extra)) + 1
    rows = run_extrapolation((seed,))
    selected = next(row for row in rows if row['role'] == 'C*')
    assert selected['degree'] == expected
    assert all(row['seed'] == seed and row['cstar'] == expected for row in rows)


def test_roles_coincidents_conserves(tmp_path):
    rows = run_extrapolation((1011,))
    assert [r['role'] for r in rows] == ['1', '3', 'C*', '10']
    star, ten = rows[2], rows[3]
    assert star['degree'] == ten['degree'] == 10
    assert star['eqm_interp'] == ten['eqm_interp']
    assert star['eqm_extrap'] == ten['eqm_extrap']
    _, path = export_extrapolation(rows, tmp_path)
    with path.open() as f:
        summary = list(csv.DictReader(f))
    assert {r['role']: int(r['n_runs']) for r in summary} == {'1': 1, '3': 1, 'C*': 1, '10': 1}
