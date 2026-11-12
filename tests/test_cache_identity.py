"""AE7-03 (etape 8) : identite du cache des runs MLP et protections de l'export.

Trois lacunes de l'audit 07 (A11, A14, A18) : les tests de l'etape 7 laissaient survivre
(1) l'oubli des cibles dans l'empreinte des donnees, (2) un hachage du mauvais fichier moteur,
(3) l'indifference de ``verify_store`` a un ``.failed.json`` isole. Les tests ci-dessous les
verrouillent ; ceux de la derniere section verifient que le pipeline d'export refuse un cache
dont l'identite ne correspond plus aux donnees ou au moteur (aucun entrainement n'est lance).
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.experiments import runner_2d_nominal as nom
from src.export import pipeline
from src.models import mlp
from src.models.mlp import MLPConfig
from tests.helpers_2d import FAST, one_thread, synthetic  # noqa: F401

pytestmark = pytest.mark.usefixtures("one_thread")


@pytest.fixture
def data():
    (xt, yt), (xv, yv), _ = synthetic()
    return xt, yt, xv, yv


def _key(data_tuple, config=None, seed=42, env=None):
    config = config or MLPConfig((3,), **FAST)
    env = env or nom.environment_identity(1)
    return nom.run_key(nom.run_identity(config, seed, nom.data_fingerprint(*data_tuple), env))


# ------------------------------------------------------------------ A11 : cibles dans la cle
def test_empreinte_des_donnees_depend_des_cibles_et_des_abscisses_de_chaque_jeu(data):
    xt, yt, xv, yv = data
    base = nom.data_fingerprint(xt, yt, xv, yv)
    eps = 1e-9
    cases = {"x_train": (xt + eps, yt, xv, yv), "y_train": (xt, yt + eps, xv, yv),
             "x_validation": (xt, yt, xv + eps, yv), "y_validation": (xt, yt, xv, yv + eps)}
    for name, changed in cases.items():
        fp = nom.data_fingerprint(*changed)
        assert fp != base, name
        # une modification d'un jeu ne change pas l'empreinte de l'autre
        same = "validation_sha256" if name.endswith("train") else "train_sha256"
        assert fp[same] == base[same], name


def test_cle_depend_des_cibles(data):
    xt, yt, xv, yv = data
    base = _key(data)
    assert base != _key((xt, yt + 1e-9, xv, yv))  # cible d'apprentissage
    assert base != _key((xt, yt, xv, yv + 1e-9))  # cible de validation
    assert base == _key((xt.copy(), yt.copy(), xv.copy(), yv.copy()))  # copie : meme cle


# ------------------------------------------------------------------ A14 : moteur reel
def test_identite_d_environnement_hache_le_fichier_moteur_reellement_utilise():
    engine = Path(mlp.__file__).resolve()
    assert nom.ENGINE_PATH.resolve() == engine  # le fichier hache est bien celui qui entraine
    expected = hashlib.sha256(engine.read_bytes()).hexdigest()
    assert nom.environment_identity(1)["engine_sha256"] == expected


def test_identite_d_environnement_suit_le_contenu_du_fichier_moteur(tmp_path, monkeypatch, data):
    fake = tmp_path / "mlp.py"
    fake.write_text("# moteur modifie\n", encoding="utf-8")
    before = _key(data)
    monkeypatch.setattr(nom, "ENGINE_PATH", fake)
    env = nom.environment_identity(1)
    assert env["engine_sha256"] == hashlib.sha256(fake.read_bytes()).hexdigest()
    assert _key(data, env=env) != before
    fake.write_text("# moteur modifie encore\n", encoding="utf-8")
    assert _key(data, env=nom.environment_identity(1)) != _key(data, env=env)


# ------------------------------------------------------------------ A18 : .failed.json isole
def _store_with_runs(tmp_path, data):
    plan = [nom.PlanEntry("T", "a", MLPConfig(hidden_layer_sizes=(3,), **FAST), (1, 2))]
    fp, env = nom.data_fingerprint(*data), nom.environment_identity(1)
    keys = nom.plan_keys(plan, fp, env)
    nom.ensure_runs(plan, *data, tmp_path)
    return keys


def test_failed_json_isole_rend_verify_store_non_conforme(tmp_path, data):
    keys = _store_with_runs(tmp_path, data)
    assert nom.verify_store(tmp_path, keys, set(keys))["ok"]
    (tmp_path / f"{keys[0]}.failed.json").write_text("{}\n", encoding="utf-8")
    report = nom.verify_store(tmp_path, keys, set(keys))
    assert not report["ok"] and report["missing"] == []  # tous les runs valides sont presents
    assert any(".failed.json" in a for a in report["anomalies"])


def test_failed_json_sans_run_valide_est_signale_et_le_run_reste_manquant(tmp_path, data):
    keys = _store_with_runs(tmp_path, data)
    (tmp_path / f"{keys[0]}.json").unlink()
    (tmp_path / f"{keys[0]}.failed.json").write_text("{}\n", encoding="utf-8")
    report = nom.verify_store(tmp_path, keys, set(keys))
    assert not report["ok"] and report["missing"] == [keys[0]]
    assert any(".failed.json" in a for a in report["anomalies"])


# ------------------------------------------------------------------ protections du pipeline d'export
def test_export_refuse_un_cache_dont_la_cible_a_change(monkeypatch):
    """Les cibles entrent dans la cle : si elles changent, le cache n'est plus reconnu."""
    real = nom.load_official_data

    def shifted():
        xt, yt, xv, yv = real()
        return xt, yt + 1e-9, xv, yv

    monkeypatch.setattr(nom, "load_official_data", shifted)
    with pytest.raises(pipeline.ExportError, match="incomplet"):
        pipeline.require_complete_cache(nom.NOMINAL_STORE)


def test_export_refuse_un_cache_dont_le_moteur_a_change(monkeypatch, tmp_path):
    fake = tmp_path / "mlp.py"
    fake.write_text("# autre moteur\n", encoding="utf-8")
    monkeypatch.setattr(nom, "ENGINE_PATH", fake)
    with pytest.raises(pipeline.ExportError, match="incomplet"):
        pipeline.require_complete_cache(nom.NOMINAL_STORE)


def test_export_accepte_le_cache_versionne():
    report = pipeline.require_complete_cache(nom.NOMINAL_STORE)
    assert report["ok"] and report["n_records"] >= 190
