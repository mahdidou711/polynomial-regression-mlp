"""Tests des donnees 2D, avec oracle scalaire independant du code vectorise."""

import csv
import hashlib
import math

import numpy as np
import pytest

from src.data import data_loader
from src.data.generator_2d import (genere_base_test_2d, genere_base_validation_2d,
                                   prepare_data_2d, sinus_cardinal)
from src.utils.metrics import getMSE, max_abs_error, r2_score, rmse


def oracle(point):
    a = -math.pi + 2 * point[0] * math.pi + 0.2
    b = -math.pi + 2 * point[1] * math.pi - 0.3
    h = math.hypot(a + b, -2 * a + b)
    return 1.0 if abs(h) < 0.001 else math.sin(h) / h


def test_fichier_officiel_et_ordre():
    path = data_loader.DEFAULT_TRAINING_PATH
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == data_loader.TRAINING_SHA256
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    assert rows[0] == ["x1", "x2"] and len(rows) == 201
    expected = np.array([[float(v) for v in row] for row in rows[1:]])
    x = data_loader.load_training_data_2d()
    assert x.shape == (200, 2) and x.dtype == np.float64
    assert np.array_equal(x, expected)
    assert np.isfinite(x).all() and np.all((x >= 0) & (x <= 1))
    assert len(np.unique(x, axis=0)) == 200


def test_identite_et_absence(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_loader.load_training_data_2d(tmp_path / "absent.csv")
    p = tmp_path / "autre.csv"
    p.write_text("x1;x2\n0;0\n")
    with pytest.raises(ValueError, match="SHA-256"):
        data_loader.load_training_data_2d(p)
    with pytest.raises(TypeError):
        data_loader.load_training_data_2d(None)


@pytest.mark.parametrize("mutation", ["header", "count", "separator", "shape", "nan", "inf", "domain", "duplicate", "missing"])
def test_chargeur_rejette_contenu_invalide(tmp_path, monkeypatch, mutation):
    rows = data_loader.DEFAULT_TRAINING_PATH.read_text().splitlines()
    if mutation == "header": rows[0] = "a;b"
    if mutation == "count": rows.pop()
    if mutation == "separator": rows[1] = rows[1].replace(";", ",")
    if mutation == "shape": rows[1] += ";0"
    if mutation == "nan": rows[1] = "nan;0.5"
    if mutation == "inf": rows[1] = "inf;0.5"
    if mutation == "domain": rows[1] = "1.01;0.5"
    if mutation == "duplicate": rows[1] = rows[2]
    if mutation == "missing": rows[1] = ";0.5"
    raw = ("\n".join(rows) + "\n").encode()
    p = tmp_path / "malforme.csv"
    p.write_bytes(raw)
    # Autorise le passage du controle d'identite pour examiner le controle de contenu.
    monkeypatch.setattr(data_loader, "TRAINING_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        data_loader.load_training_data_2d(p)


def test_cible_oracle_et_seuil():
    center = np.array([(math.pi - 0.2) / (2 * math.pi),
                       (math.pi + 0.3) / (2 * math.pi)])
    # Inverse de A pour produire z=(h,0), puis retour dans [0,1]^2.
    points = [center + np.array([h / 3, 2 * h / 3]) / (2 * math.pi)
              for h in (0, 0.0005, 0.000999, 0.001001, 0.002)]
    points += [[0, 0], [1, 1], [0.1, 0.9], [0.9, 0.1], [0.5, 0.5]]
    actual = sinus_cardinal(points)
    expected = [oracle(p) for p in points]
    # Quelques ulp d'ecart entre hypot scalaire et norme vectorielle.
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    assert np.array_equal(actual[:3], np.ones(3))
    assert actual[3] < 1
    assert actual.shape == (len(points),)


@pytest.mark.parametrize("bad", [[1, 2], [[1, 2, 3]], [], [[float('nan'), 0]],
                                   [[float('inf'), 0]], [[-0.1, 0]], [[1.1, 0]]])
def test_cible_forme_et_domaine_invalides(bad):
    with pytest.raises(ValueError): sinus_cardinal(bad)


@pytest.mark.parametrize("bad", [[[True, False]], [["0", "1"]], [[1j, 0]]])
def test_cible_type_invalide(bad):
    with pytest.raises(TypeError): sinus_cardinal(bad)


def test_grilles_et_splits():
    splits = prepare_data_2d()
    coordinates = []
    for (x, y), n, axis in zip(splits, (200, 400, 1600),
                               (None, [(i + 0.5) / 20 for i in range(20)],
                                [j / 39 for j in range(40)])):
        assert x.shape == (n, 2) and y.shape == (n,)
        assert len(np.unique(x, axis=0)) == n
        np.testing.assert_allclose(y, [oracle(p) for p in x], atol=2e-15, rtol=0)
        if axis is not None:
            expected = np.array([(a, b) for a in axis for b in axis])
            # linspace et division peuvent differer de 1 ulp.
            np.testing.assert_allclose(x, expected, atol=2e-16, rtol=0)
        assert getMSE(y, y) == rmse(y, y) == max_abs_error(y, y) == 0
        assert r2_score(y, y) == 1
        coordinates.append(set(map(tuple, x)))
    assert not coordinates[0] & coordinates[1]
    assert not coordinates[0] & coordinates[2]
    assert not coordinates[1] & coordinates[2]
    val, test = splits.validation[0], splits.test[0]
    assert val.min() == 0.025 and val.max() == 0.975
    assert test.min() == 0 and test.max() == 1
    for first, second in zip(splits, prepare_data_2d()):
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.array_equal(val, genere_base_validation_2d()[0])
    assert np.array_equal(test, genere_base_test_2d()[0])


def test_egalite_seuil_utilise_quotient(monkeypatch):
    # Isole la branche exacte, inaccessible de facon fiable par inversion flottante.
    monkeypatch.setattr(np.linalg, "norm", lambda *args, **kwargs: np.array([0.001]))
    assert sinus_cardinal([[0.5, 0.5]])[0] == math.sin(0.001) / 0.001


def test_grilles_ne_modifient_pas_etat_aleatoire():
    before = np.random.get_state()
    genere_base_validation_2d()
    genere_base_test_2d()
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("bad", [np.empty((0, 2)), np.array([[0.5]])])
def test_cible_rejette_formes_vides_ou_incompletes(bad):
    with pytest.raises(ValueError):
        sinus_cardinal(bad)
