"""Stage A: grids, parameters, cache, resume, statistics, exports."""
import ast
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor

from src.experiments import runner_2d_nominal as nom
from src.experiments import runner_2d_parametric as par
from src.models import mlp
from src.models.mlp import MLPConfig
from src.utils.seeding import SEEDS_MONTE_CARLO
from tests.helpers_2d import FAST, one_thread, synthetic  # noqa: F401

pytestmark = pytest.mark.usefixtures("one_thread")


def small_plan(seeds=(1, 2, 3, 4)):
    return [nom.PlanEntry("T", "a", MLPConfig(hidden_layer_sizes=(3,), **FAST), seeds)]


@pytest.fixture
def data():
    (xt, yt), (xv, yv), _ = synthetic()
    return xt, yt, xv, yv


# ---------------------------------------------------------------- grids and counts
def test_fixed_constants():
    assert nom.WIDTHS == (5, 10, 20, 50, 100, 200)
    assert nom.TOPOLOGIES == ((20, 10), (50, 25))
    assert nom.ALPHAS == (1e-5, 1e-3, 1e-1)
    assert nom.REG_REFERENCE_WIDTH == 100
    assert nom.ADAM_GRID == ((1e-3, 32), (1e-3, 200), (1e-2, 32), (1e-2, 200))
    assert SEEDS_MONTE_CARLO == (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021)
    assert all(e.seeds == SEEDS_MONTE_CARLO for e in par.full_plan(20))


@pytest.mark.parametrize("h_star", nom.WIDTHS)
def test_counts_210_and_190_or_160(h_star):
    plan = par.full_plan(h_star)
    count = par.count_plan(plan)
    assert (count["roles"], count["realisations"]) == (21, 210)
    expected = 160 if h_star == 100 else 190
    assert count["distinct_runs"] == expected
    assert count["distinct_configs"] == expected // 10
    ids = {(e.campaign, e.role): nom.config_id(e.config) for e in plan}
    assert ids[("ACT", "tanh")] == ids[("SOLV", "lbfgs")] == ids[("NOM", f"H{h_star}")]
    if h_star == 100:  # roles kept, configurations shared
        for a in nom.ALPHAS:
            assert ids[("REG", f"H_star:alpha={a:g}")] == ids[("REG", f"H100:alpha={a:g}")]
        assert sum(e.campaign == "REG" for e in plan) == 6


def test_distribution_of_roles():
    plan = par.full_plan(20)
    counts = {}
    for e in plan:
        counts[e.campaign] = counts.get(e.campaign, 0) + 1
    assert counts == {"NOM": 6, "TOPO": 2, "REG": 6, "ACT": 2, "SOLV": 5}
    assert [nom.config_id(e.config) for e in plan[:6]] == \
        [nom.config_id(nom.nominal_config((h,))) for h in nom.WIDTHS]


# ---------------------------------------------------------------- parameters
def test_effective_parameters_of_each_configuration(monkeypatch, data):
    """get_params() captured at the entry of fit: all the contractual parameters."""
    captured = []
    original = MLPRegressor.fit

    def spy(self, X, y):
        captured.append(self.get_params())
        self.set_params(max_iter=1, max_fun=10)  # a one-iteration run
        return original(self, X, y)

    monkeypatch.setattr(mlp.MLPRegressor, "fit", spy)
    plan = par.full_plan(20)
    seen = {}
    for e in plan:
        seen.setdefault(nom.config_id(e.config), e)
    for e in seen.values():
        nom.run_strict(e.config, 42, *data)
    assert len(captured) == 19
    for (cid, e), p in zip(seen.items(), captured):
        c = e.config
        assert p["hidden_layer_sizes"] == c.hidden_layer_sizes and p["random_state"] == 42
        assert (p["early_stopping"], p["shuffle"], p["learning_rate"]) == (False, True, "constant")
        assert (p["beta_1"], p["beta_2"]) == (0.9, 0.999)
        if c.solver == "lbfgs":
            assert (p["solver"], p["max_iter"], p["max_fun"], p["tol"]) == ("lbfgs", 10000, 100000, 1e-6)
            assert p["alpha"] in (1e-4, *nom.ALPHAS)
        else:  # complete adam protocol, n_iter_no_change=200 and not 10
            assert (p["solver"], p["max_iter"], p["tol"], p["n_iter_no_change"], p["alpha"],
                    p["activation"]) == ("adam", 5000, 1e-8, 200, 1e-4, "tanh")
            assert (p["learning_rate_init"], p["batch_size"]) in nom.ADAM_GRID
        if e.role == "relu":
            assert p["activation"] == "relu"
        elif c.solver == "lbfgs":
            assert p["activation"] == "tanh"
    nominal = [p for p in captured[:6]]
    assert [p["hidden_layer_sizes"] for p in nominal] == [(h,) for h in nom.WIDTHS]
    assert all(p["alpha"] == 1e-4 for p in nominal)


def test_all_contractual_configurations_are_distinct():
    assert len(nom.contractual_config_ids()) == 56  # 6 x 9 + 2 topologies


# ---------------------------------------------------------------- identity and keys
def test_key_depends_on_each_component(data):
    fp = nom.data_fingerprint(*data)
    env = nom.environment_identity(1)
    cfg = MLPConfig((3,), **FAST)
    base = nom.run_key(nom.run_identity(cfg, 42, fp, env))
    assert base == nom.run_key(nom.run_identity(cfg, 42, fp, env))
    assert base != nom.run_key(nom.run_identity(cfg, 43, fp, env))
    assert base != nom.run_key(nom.run_identity(MLPConfig((4,), **FAST), 42, fp, env))
    other = nom.data_fingerprint(data[0] + 1e-9, *data[1:])
    assert base != nom.run_key(nom.run_identity(cfg, 42, other, env))
    assert base != nom.run_key(nom.run_identity(cfg, 42, fp, nom.environment_identity(2)))
    assert base != nom.run_key(nom.run_identity(cfg, 42, fp, {**env, "engine_sha256": "0" * 64}))


def test_threads_refused_beyond_one(monkeypatch):
    monkeypatch.setattr(nom, "threadpool_info", lambda: [{"num_threads": 4, "internal_api": "openblas"}])
    with pytest.raises(RuntimeError, match="a single BLAS/OpenMP thread"):
        nom.check_threads()
    monkeypatch.setattr(nom, "threadpool_info", lambda: [{"num_threads": 1, "internal_api": "openblas"},
                                                         {"num_threads": 1, "internal_api": "openmp"}])
    assert nom.check_threads() == 1


# ---------------------------------------------------------------- cache, reprise
def test_resume_after_interruption_without_duplicate(tmp_path, monkeypatch, data):
    store = tmp_path / "s"
    reference = nom.ensure_runs(small_plan(), *data, tmp_path / "ref")
    calls = []
    original = nom.train_mlp

    def flaky(*a, **k):
        calls.append(a[1])
        if len(calls) == 3:
            raise RuntimeError("simulated interruption")
        return original(*a, **k)

    monkeypatch.setattr(nom, "train_mlp", flaky)
    with pytest.raises(RuntimeError, match="interruption"):
        nom.ensure_runs(small_plan(), *data, store)
    assert sorted(p.name for p in store.glob("*.json") if not p.name.endswith(".failed.json")).__len__() == 2
    assert len(list(store.glob("*.failed.json"))) == 1
    monkeypatch.setattr(nom, "train_mlp", original)
    calls.clear()
    monkeypatch.setattr(nom, "train_mlp", lambda *a, **k: (calls.append(a[1]), original(*a, **k))[1])
    out = nom.ensure_runs(small_plan(), *data, store)
    assert calls == [3, 4]  # only the missing runs are recomputed
    assert not list(store.glob("*.failed.json"))  # failure resolved by the resume
    assert out.keys() == reference.keys()
    for k in out:
        assert out[k]["weights_sha256"] == reference[k]["weights_sha256"]
        assert out[k]["validation_metrics"] == reference[k]["validation_metrics"]
    # complete rerun: no training
    calls.clear()
    nom.ensure_runs(small_plan(), *data, store)
    assert calls == []


def test_run_shared_between_roles_is_trained_once(tmp_path, monkeypatch, data):
    plan = small_plan((1, 2)) + [nom.PlanEntry("U", "b", MLPConfig((3,), **FAST), (1, 2))]
    calls = []
    original = nom.train_mlp
    monkeypatch.setattr(nom, "train_mlp", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    out = nom.ensure_runs(plan, *data, tmp_path)
    assert len(calls) == 2 and len(out) == 2


def test_leftover_tmp_ignored_and_noted(tmp_path, data):
    plan = small_plan((1,))
    keys = nom.plan_keys(plan, nom.data_fingerprint(*data), nom.environment_identity(1))
    (tmp_path / f"{keys[0]}.json.tmp").write_text("{incomplete", encoding="utf-8")
    nom.ensure_runs(plan, *data, tmp_path)
    assert (tmp_path / f"{keys[0]}.json").exists()
    report = nom.verify_store(tmp_path, keys, set(keys))
    assert report["ok"] and report["n_records"] == 1


def test_invalid_records_refused_without_overwriting(tmp_path, data):
    plan = small_plan((1,))
    nom.ensure_runs(plan, *data, tmp_path)
    path = next(tmp_path.glob("*.json"))
    good = path.read_text(encoding="utf-8")
    rec = json.loads(good)
    cases = {
        "altered metric": {**rec, "validation_metrics": {**rec["validation_metrics"], "mse": 9.9}},
        "test field": {**rec, "test_metrics": {"mse": 0.0}},
        "inconsistent key": {**rec, "run_key": "0" * 64},
    }
    for name, bad in cases.items():
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(nom.StoreError):
            nom.ensure_runs(plan, *data, tmp_path)
        assert json.loads(path.read_text(encoding="utf-8")) == bad, name  # never overwritten
    path.write_text("{not a json", encoding="utf-8")
    with pytest.raises(nom.StoreError):
        nom.ensure_runs(plan, *data, tmp_path)
    renamed = tmp_path / ("1" * 64 + ".json")
    renamed.write_text(good, encoding="utf-8")
    with pytest.raises(nom.StoreError, match="key inconsistent"):
        nom.load_record(renamed)


def test_depot_nominal_refuse_budget_court(tmp_path, data):
    before = sorted(nom.NOMINAL_STORE.iterdir()) if nom.NOMINAL_STORE.exists() else None
    with pytest.raises(ValueError, match="non-contractual"):
        nom.ensure_runs(small_plan(), *data, nom.NOMINAL_STORE)
    with pytest.raises(ValueError, match="non-contractual"):  # non-contractual seeds
        nom.ensure_runs([nom.PlanEntry("T", "a", nom.nominal_config((5,)), (1, 2))], *data,
                        nom.NOMINAL_STORE)
    after = sorted(nom.NOMINAL_STORE.iterdir()) if nom.NOMINAL_STORE.exists() else None
    assert before == after


def test_keyboard_interrupt_swallowed_by_sklearn_is_not_published(tmp_path, monkeypatch, data):
    original = MLPRegressor.fit

    def fit(self, X, y):
        import warnings
        warnings.warn("Training interrupted by user.")
        return original(self, X, y)

    monkeypatch.setattr(mlp.MLPRegressor, "fit", fit)
    with pytest.raises(UserWarning, match="interrupted"):
        nom.ensure_runs(small_plan((1,)), *data, tmp_path)
    assert not [p for p in tmp_path.glob("*.json") if not p.name.endswith(".failed.json")]
    assert len(list(tmp_path.glob("*.failed.json"))) == 1


def test_verify_store_detects_foreign_missing_and_failure(tmp_path, data):
    plan = small_plan((1, 2))
    fp, env = nom.data_fingerprint(*data), nom.environment_identity(1)
    keys = nom.plan_keys(plan, fp, env)
    nom.ensure_runs(plan, *data, tmp_path)
    assert nom.verify_store(tmp_path, keys, set(keys))["ok"]
    assert nom.verify_store(tmp_path, keys, {keys[0]})["anomalies"]  # foreign file
    assert nom.verify_store(tmp_path, [*keys, "x" * 64], set(keys))["missing"] == ["x" * 64]
    (tmp_path / "noise.txt").write_text("x")
    assert not nom.verify_store(tmp_path, keys, set(keys))["ok"]
    assert not nom.verify_store(tmp_path / "absent")["ok"]


# ---------------------------------------------------------------- statistics
def test_statistics_identical_to_run_multi_seed_and_oracle(tmp_path, data):
    seeds = SEEDS_MONTE_CARLO
    cfg = MLPConfig((3,), **FAST)
    out = nom.ensure_runs([nom.PlanEntry("T", "a", cfg, seeds)], *data, tmp_path)
    records = list(out.values())
    stats = nom.describe(records)
    ref = mlp.run_multi_seed(cfg, *data)
    for split in ("training", "validation"):
        for m in nom.METRICS:
            for s, v in ref.statistics[split][m].items():
                assert stats[split][m][s] == v
    values = np.array([r["validation_metrics"]["mse"] for r in records])
    assert stats["validation"]["mse"]["std"] == pytest.approx(np.std(values, ddof=1), rel=1e-15)
    assert stats["n_warnings"] == ref.warning_count  # runs with warnings kept
    assert sum(stats["stop_causes"].values()) == 10


def test_describe_refuses_incomplete_sample(tmp_path, data):
    out = nom.ensure_runs(small_plan(), *data, tmp_path)
    with pytest.raises(ValueError, match="runs instead of"):
        nom.describe(list(out.values()))  # 4 runs instead of 10
    assert nom.describe(list(out.values()), expected_runs=4)["n_runs"] == 4


# ---------------------------------------------------------------- diagnostics
@pytest.mark.parametrize("solver, warnings_, expected", [
    ("lbfgs", [], "none"),
    ("lbfgs", ["lbfgs failed to converge after 5 iteration(s) (status=1):\nSTOP: TOTAL NO. OF ITERATIONS REACHED LIMIT\n"], "max_iter"),
    ("lbfgs", ["lbfgs failed to converge after 5 iteration(s) (status=1):\nSTOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT\n"], "max_fun"),
    ("lbfgs", ["lbfgs failed to converge (status=2):\nABNORMAL: \n"], "other_warning"),
    ("adam", ["Stochastic Optimizer: Maximum iterations (5000) reached and the optimization hasn't converged yet."], "max_iter"),
    ("adam", ["other message"], "other_warning"),
])
def test_stop_cause(solver, warnings_, expected):
    assert nom.stop_cause({"convergence_warnings": warnings_}, solver) == expected


def test_stop_cause_on_actual_scikit_learn_messages(data):
    """A one-iteration lbfgs run, then a run with minimal max_fun: actual engine messages."""
    r1 = nom.run_strict(MLPConfig((3,), max_iter=1, max_fun=1000), 42, *data)
    r2 = nom.run_strict(MLPConfig((3,), max_iter=1000, max_fun=3), 42, *data)
    assert nom.stop_cause(r1.diagnostics, "lbfgs") == "max_iter"
    assert nom.stop_cause(r2.diagnostics, "lbfgs") == "max_fun"
    r3 = nom.run_strict(nom.adam_config(3, 1e-3, 32).__class__(
        (3,), solver="adam", max_iter=2, batch_size=32, n_iter_no_change=200), 42, *data)
    assert nom.stop_cause(r3.diagnostics, "adam") == "max_iter"


# ---------------------------------------------------------------- exports
def test_tables_deterministic_from_the_cache(tmp_path, data):
    store = tmp_path / "store"
    plan = nom.nominal_plan(FAST)
    fp, env = nom.data_fingerprint(*data), nom.environment_identity(1)
    nom.ensure_runs(plan, *data, store)
    outputs = []
    for run in range(2):
        collected = nom.collect_stats(plan, store, fp, env)
        directory = tmp_path / f"t{run}"
        nom.write_run_tables(plan, collected, fp, env, directory)
        outputs.append({p.name: p.read_bytes() for p in directory.iterdir()})
    assert outputs[0] == outputs[1]  # byte for byte
    assert set(outputs[0]) == {"mlp_2d_plan.csv", "mlp_2d_runs.csv", "mlp_2d_config_stats.csv"}
    with open(tmp_path / "t0" / "mlp_2d_plan.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 60 and not any(r["shared_with_earlier_role"] == "True" for r in rows)
    with open(tmp_path / "t0" / "mlp_2d_runs.csv", encoding="utf-8") as fh:
        runs = list(csv.DictReader(fh))
    assert len(runs) == 60
    records = list(nom.load_config_records(plan[0].config, plan[0].seeds, store, fp, env))
    first = [r for r in runs if r["config_id"] == nom.config_id(plan[0].config)]
    assert [float(r["val_mse"]) for r in first] == [x["validation_metrics"]["mse"] for x in records]


def test_json_strict_sans_nan(tmp_path):
    with pytest.raises(ValueError):
        nom.write_json(tmp_path / "x.json", {"a": float("nan")})
    nom.write_json(tmp_path / "ok.json", {"a": 1.5})
    assert json.loads((tmp_path / "ok.json").read_text()) == {"a": 1.5}


# ---------------------------------------------------------------- reproducibility
def test_weights_identical_between_two_processes(tmp_path):
    import subprocess
    import sys
    code = (
        "import os, sys, json; sys.path.insert(0, '.');"
        "from tests.helpers_2d import synthetic, FAST;"
        "from src.experiments import runner_2d_nominal as nom; from src.models.mlp import MLPConfig;"
        "(xt,yt),(xv,yv),_ = synthetic();"
        "r = nom.run_strict(MLPConfig((4,), max_iter=30, max_fun=300), 42, xt,yt,xv,yv);"
        "print(nom.weights_sha256(r.model))")
    env = {**__import__("os").environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MPLCONFIGDIR": str(tmp_path)}
    root = Path(__file__).resolve().parents[1]
    out = [subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True,
                          text=True, check=True).stdout.strip() for _ in range(2)]
    assert out[0] == out[1] and len(out[0]) == 64


# ---------------------------------------------------------------- static watertightness
FORBIDDEN = {"generate_test_set_2d", "prepare_data_2d", "DataSplits2D"}


def _names(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.name.split(".")[-1] for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[-1])
    return names


@pytest.mark.parametrize("module", [nom, par])
def test_runners_do_not_reference_the_test_set(module):
    assert not FORBIDDEN & _names(module.__file__)


def test_hidden_layer_sizes_always_passed_by_keyword():
    for module in (nom, par, mlp):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("MLPRegressor",):
                assert not node.args
