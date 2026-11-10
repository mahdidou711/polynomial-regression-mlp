"""Contrats du moteur MLP, sans campagne de selection scientifique."""
import logging
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.models import mlp
from src.utils.seeding import SEEDS_MONTE_CARLO


@pytest.fixture
def data():
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [.25, .75], [.7, .4]])
    y = x[:, 0] + .3 * x[:, 1] ** 2
    return x, y


def train(config, seed, data):
    x, y = data
    return mlp.train_mlp(config, seed, x, y, x, y)


def test_nominal_params_et_configurations():
    c = mlp.MLPConfig()
    assert (c.activation, c.solver, c.alpha, c.max_iter, c.max_fun, c.tol) == (
        'tanh', 'lbfgs', 1e-4, 10000, 100000, 1e-6)
    assert mlp.MLPConfig((20, 10), activation='relu', solver='adam', alpha=0).hidden_layer_sizes == (20, 10)


@pytest.mark.parametrize('layers', [(3,), (4, 2)])
def test_reel_parametres_prediction_et_sortie_lineaire(data, layers):
    c = mlp.MLPConfig(layers, max_iter=30, max_fun=300, alpha=.002)
    r = train(c, 42, data)
    assert r.parameters['hidden_layer_sizes'] == layers
    assert r.parameters['random_state'] == 42
    assert r.parameters['max_fun'] == 300 and r.parameters['alpha'] == .002
    assert r.model.out_activation_ == 'identity'
    assert len(r.model.coefs_) == len(layers) + 1
    assert len(r.model.intercepts_) == len(layers) + 1
    assert r.predict(data[0]).shape == (len(data[0]),)
    assert r.diagnostics['loss_curve'] is None
    assert r.versions['scikit-learn']


def test_determinisme_sur_dix_graines_et_independance(data):
    c = mlp.MLPConfig((2,), max_iter=1)
    x, y = data
    before = np.random.get_state()
    a = mlp.run_multi_seed(c, x, y, x, y)
    b = mlp.run_multi_seed(c, x, y, x, y)
    after = np.random.get_state()
    assert tuple(r.seed for r in a.runs) == SEEDS_MONTE_CARLO == (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021)
    assert len({id(r.model) for r in a.runs}) == 10
    for ra, rb in zip(a.runs, b.runs):
        np.testing.assert_array_equal(ra.predict(x), rb.predict(x))
        for wa, wb in zip(ra.model.coefs_, rb.model.coefs_):
            np.testing.assert_array_equal(wa, wb)
    assert not np.array_equal(a.runs[0].model.coefs_[0], a.runs[1].model.coefs_[0])
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_statistiques_et_metriques_oracle(data):
    x, y = data
    results = mlp.run_multi_seed(mlp.MLPConfig((3,), max_iter=3), x, y, x, y, seeds=(42, 123, 456))
    expected = []
    for run in results.runs:
        pred = run.predict(x)
        residual = y - pred
        mse = sum(float(v)**2 for v in residual) / len(y)
        r2 = 1 - sum(float(v)**2 for v in residual) / sum(float(v - y.mean())**2 for v in y)
        assert run.validation_metrics == pytest.approx({'mse': mse, 'rmse': mse**.5, 'r2': r2, 'max_abs_error': max(abs(residual))})
        expected.append(mse)
    mean = sum(expected) / 3
    std = (sum((v - mean)**2 for v in expected) / 2)**.5
    assert results.statistics['validation']['mse']['mean'] == pytest.approx(mean)
    assert results.statistics['validation']['mse']['std'] == pytest.approx(std)
    assert results.successful_count == 3 and results.warning_count == 3
    single = mlp.run_multi_seed(mlp.MLPConfig((2,), max_iter=1), x, y, x, y, seeds=(42,))
    assert single.statistics['validation']['mse']['std'] is None


def test_warning_capte_et_journalise(data, caplog):
    with caplog.at_level(logging.WARNING, logger='c0_tp2_mlp.mlp'):
        r = train(mlp.MLPConfig((3,), max_iter=1), 42, data)
    assert r.diagnostics['convergence_warnings']
    assert r.diagnostics['status'] == 'warning'
    record = next(r for r in caplog.records if r.msg == 'mlp_run')
    assert record.fields['seed'] == 42
    assert record.fields['diagnostics']['convergence_warnings']
    assert record.fields['parameters']['max_iter'] == 1


def test_autre_warning_pas_masque(data, monkeypatch):
    original = mlp.MLPRegressor.fit
    def fit(self, *args, **kwargs):
        warnings.warn('autre alerte', UserWarning)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(mlp.MLPRegressor, 'fit', fit)
    with pytest.warns(UserWarning, match='autre alerte'):
        train(mlp.MLPConfig((2,), max_iter=1), 42, data)


def test_adam_courbe_et_parametres(data):
    c = mlp.MLPConfig((4, 2), solver='adam', activation='relu', max_iter=8,
                      tol=1e-8, learning_rate_init=.01, batch_size=3, n_iter_no_change=200)
    r = train(c, 123, data)
    assert r.parameters['learning_rate_init'] == .01 and r.parameters['batch_size'] == 3
    assert r.parameters['n_iter_no_change'] == 200 and not r.parameters['early_stopping']
    assert len(r.diagnostics['loss_curve']) == r.diagnostics['n_iter']
    assert r.diagnostics['tail_loss'] == r.diagnostics['loss_curve'][-10:]
    assert r.diagnostics['iteration_limit_reached']


@pytest.mark.parametrize('kwargs', [dict(hidden_layer_sizes=()), dict(hidden_layer_sizes=(0,)),
    dict(hidden_layer_sizes=(True,)), dict(hidden_layer_sizes=3), dict(activation='logistic'),
    dict(solver='sgd'), dict(alpha=-1), dict(alpha=float('nan')), dict(max_iter=0),
    dict(max_fun=0), dict(tol=-1), dict(batch_size=0), dict(batch_size=True),
    dict(n_iter_no_change=0), dict(learning_rate_init=0)])
def test_config_invalide(kwargs):
    with pytest.raises(ValueError): mlp.MLPConfig(**kwargs)


@pytest.mark.parametrize('x,y', [(np.empty((0, 2)), []), ([[1]], [1]), ([[1, 2]], [[1]]),
    ([[1, 2], [3, 4]], [1]), ([[1, float('nan')], [2, 3]], [1, 2]),
    ([[1, 2], [3, 4]], [1, 1])])
def test_donnees_invalides(x, y, data):
    with pytest.raises(ValueError):
        mlp.train_mlp(mlp.MLPConfig(), 42, x, y, *data)


def test_graines_invalides_et_aucun_acces_test(data, monkeypatch):
    from src.data import generator_2d
    def forbidden(*args, **kwargs):
        raise AssertionError('test final accede')
    monkeypatch.setattr(generator_2d, 'genere_base_test_2d', forbidden)
    x, y = data
    for seeds in ((), (42, 42), (None,)):
        with pytest.raises((ValueError, TypeError)):
            mlp.run_multi_seed(mlp.MLPConfig(), x, y, x, y, seeds=seeds)
    result = mlp.run_multi_seed(mlp.MLPConfig((2,), max_iter=1), x, y, x, y, seeds=(42,))
    assert len(result.runs) == 1
    with pytest.raises(TypeError):
        mlp.run_multi_seed(mlp.MLPConfig(), x, y, x, y, x_test=x, y_test=y)


def test_courbe_incoherente_et_oscillation(data, monkeypatch):
    original = mlp.MLPRegressor.fit
    def fit(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.loss_curve_ = [1., 2., 1., 2.]
        self.n_iter_ = 4
        return self
    monkeypatch.setattr(mlp.MLPRegressor, 'fit', fit)
    r = train(mlp.MLPConfig((2,), solver='adam', max_iter=1), 42, data)
    assert r.diagnostics['tail_oscillation']
    def bad_fit(self, *args, **kwargs):
        fit(self, *args, **kwargs)
        self.loss_curve_.pop()
        return self
    monkeypatch.setattr(mlp.MLPRegressor, 'fit', bad_fit)
    with pytest.raises(ValueError, match='loss_curve_'):
        train(mlp.MLPConfig((2,), solver='adam', max_iter=1), 42, data)


def test_initialisations_effectivement_distinctes(data, monkeypatch):
    initial = []
    original = mlp.MLPRegressor._initialize
    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        initial.append([w.copy() for w in self.coefs_])
    monkeypatch.setattr(mlp.MLPRegressor, '_initialize', capture)
    c = mlp.MLPConfig((3,), max_iter=1)
    for seed in (42, 123, 42):
        train(c, seed, data)
    np.testing.assert_array_equal(initial[0][0], initial[2][0])
    assert not np.array_equal(initial[0][0], initial[1][0])
