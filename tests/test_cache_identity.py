"""Identity of the MLP run cache and export safeguards.

Three gaps that the tests used to let survive: (1) forgetting the targets in the data
fingerprint, (2) hashing the wrong engine file, (3) ``verify_store`` ignoring an isolated
``.failed.json``. The tests below lock them; those of the last section check that the
export pipeline refuses a cache whose identity no longer matches the data or the engine
(no training is launched).
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


# ------------------------------------------------------------------ targets in the key
def test_data_fingerprint_depends_on_targets_and_abscissas_of_each_set(data):
    xt, yt, xv, yv = data
    base = nom.data_fingerprint(xt, yt, xv, yv)
    eps = 1e-9
    cases = {"x_train": (xt + eps, yt, xv, yv), "y_train": (xt, yt + eps, xv, yv),
             "x_validation": (xt, yt, xv + eps, yv), "y_validation": (xt, yt, xv, yv + eps)}
    for name, changed in cases.items():
        fp = nom.data_fingerprint(*changed)
        assert fp != base, name
        # a change in one data set does not change the fingerprint of the other
        same = "validation_sha256" if name.endswith("train") else "train_sha256"
        assert fp[same] == base[same], name


def test_key_depends_on_targets(data):
    xt, yt, xv, yv = data
    base = _key(data)
    assert base != _key((xt, yt + 1e-9, xv, yv))  # training target
    assert base != _key((xt, yt, xv, yv + 1e-9))  # validation target
    assert base == _key((xt.copy(), yt.copy(), xv.copy(), yv.copy()))  # copy: same key


# ------------------------------------------------------------------ actual engine
def test_environment_identity_hashes_the_engine_file_actually_used():
    engine = Path(mlp.__file__).resolve()
    assert nom.ENGINE_PATH.resolve() == engine  # the hashed file is indeed the one that trains
    expected = hashlib.sha256(engine.read_bytes()).hexdigest()
    assert nom.environment_identity(1)["engine_sha256"] == expected


def test_environment_identity_follows_the_content_of_the_engine_file(tmp_path, monkeypatch, data):
    fake = tmp_path / "mlp.py"
    fake.write_text("# modified engine\n", encoding="utf-8")
    before = _key(data)
    monkeypatch.setattr(nom, "ENGINE_PATH", fake)
    env = nom.environment_identity(1)
    assert env["engine_sha256"] == hashlib.sha256(fake.read_bytes()).hexdigest()
    assert _key(data, env=env) != before
    fake.write_text("# modified engine again\n", encoding="utf-8")
    assert _key(data, env=nom.environment_identity(1)) != _key(data, env=env)


# ------------------------------------------------------------------ isolated .failed.json
def _store_with_runs(tmp_path, data):
    plan = [nom.PlanEntry("T", "a", MLPConfig(hidden_layer_sizes=(3,), **FAST), (1, 2))]
    fp, env = nom.data_fingerprint(*data), nom.environment_identity(1)
    keys = nom.plan_keys(plan, fp, env)
    nom.ensure_runs(plan, *data, tmp_path)
    return keys


def test_isolated_failed_json_makes_verify_store_non_conforming(tmp_path, data):
    keys = _store_with_runs(tmp_path, data)
    assert nom.verify_store(tmp_path, keys, set(keys))["ok"]
    (tmp_path / f"{keys[0]}.failed.json").write_text("{}\n", encoding="utf-8")
    report = nom.verify_store(tmp_path, keys, set(keys))
    assert not report["ok"] and report["missing"] == []  # all the valid runs are present
    assert any(".failed.json" in a for a in report["anomalies"])


def test_failed_json_without_valid_run_is_reported_and_the_run_stays_missing(tmp_path, data):
    keys = _store_with_runs(tmp_path, data)
    (tmp_path / f"{keys[0]}.json").unlink()
    (tmp_path / f"{keys[0]}.failed.json").write_text("{}\n", encoding="utf-8")
    report = nom.verify_store(tmp_path, keys, set(keys))
    assert not report["ok"] and report["missing"] == [keys[0]]
    assert any(".failed.json" in a for a in report["anomalies"])


# ------------------------------------------------------------------ export pipeline safeguards
def test_export_refuses_a_cache_whose_target_changed(monkeypatch):
    """The targets enter the key: if they change, the cache is no longer recognised."""
    real = nom.load_input_data

    def shifted():
        xt, yt, xv, yv = real()
        return xt, yt + 1e-9, xv, yv

    monkeypatch.setattr(nom, "load_input_data", shifted)
    with pytest.raises(pipeline.ExportError, match="incomplete"):
        pipeline.require_complete_cache(nom.NOMINAL_STORE)


def test_export_refuses_a_cache_whose_engine_changed(monkeypatch, tmp_path):
    fake = tmp_path / "mlp.py"
    fake.write_text("# other engine\n", encoding="utf-8")
    monkeypatch.setattr(nom, "ENGINE_PATH", fake)
    with pytest.raises(pipeline.ExportError, match="incomplete"):
        pipeline.require_complete_cache(nom.NOMINAL_STORE)


def test_export_accepts_the_stored_cache():
    report = pipeline.require_complete_cache(nom.NOMINAL_STORE)
    assert report["ok"] and report["n_records"] >= 190
