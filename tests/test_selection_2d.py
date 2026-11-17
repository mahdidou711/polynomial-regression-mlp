"""Ranking, tie rules, freeze, watertightness and final evaluation."""
import ast
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src.data import generator_2d
from src.experiments import final_evaluation_2d as fin
from src.experiments import runner_2d_nominal as nom
from src.experiments import runner_2d_parametric as par
from src.models.mlp import MLPConfig
from src.utils.seeding import SEEDS_MONTE_CARLO
from tests.helpers_2d import FAST, one_thread, synthetic  # noqa: F401

pytestmark = pytest.mark.usefixtures("one_thread")


def _stats(mean, n=10):
    return {"n_runs": n, "validation": {"mse": {"mean": mean}}}


def _collected(entries):
    """entries: [(config, mean)] in plan order."""
    return {nom.config_id(c): {"config": c, "roles": ["T"], "records": [], "stats": _stats(m)}
            for c, m in entries}


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Complete campaign (190 runs) with a short budget, without any access to the test set."""
    from threadpoolctl import threadpool_limits
    base = tmp_path_factory.mktemp("campaign")
    train, val, test = synthetic()
    calls = []

    def forbidden(*a, **k):
        calls.append(1)
        raise AssertionError("test set read during the selection")

    mp = pytest.MonkeyPatch()
    mp.setattr(generator_2d, "generate_test_set_2d", forbidden)
    mp.setattr(generator_2d, "prepare_data_2d", forbidden)
    try:
        with threadpool_limits(limits=1):
            store, tables = base / "store", base / "tables"
            data = (*train, *val)
            nom.ensure_runs(nom.nominal_plan(FAST), *data, store)
            fp, env = nom.data_fingerprint(*data), nom.environment_identity(1)
            width = nom.select_width({
                int(c["config"].hidden_layer_sizes[0]): c["stats"]
                for c in nom.collect_stats(nom.nominal_plan(FAST), store, fp, env).values()})
            plan = par.full_plan(width["h_star"], FAST)
            nom.ensure_runs(plan, *data, store)
            selection, collected = par.compute_selection(store, fp, env, FAST)
            gel = tables / par.SELECTION_FILE
            digest = par.freeze_selection(gel, selection)
    finally:
        mp.undo()
    return {"store": store, "tables": tables, "train": train, "val": val, "test": test,
            "selection": selection, "collected": collected, "freeze": gel, "digest": digest,
            "plan": plan, "fp": fp, "env": env, "test_calls": calls}


def _read_csv(path):
    import csv
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def evaluate(c, tables=None, load_test=None):
    return fin.evaluate_final(c["freeze"], c["store"], c["train"], c["val"],
                              load_test or (lambda: c["test"]),
                              tables_dir=tables or c["tables"], overrides=FAST)


# ---------------------------------------------------------------- ranking and ties
def test_h_star_is_the_argmin_of_the_mean_validation(campaign):
    c = campaign
    means = {}
    for e in nom.nominal_plan(FAST):
        recs = nom.load_config_records(e.config, e.seeds, c["store"], c["fp"], c["env"])
        means[e.config.hidden_layer_sizes[0]] = np.mean([r["validation_metrics"]["mse"] for r in recs])
    oracle = min(means, key=lambda h: (means[h], h))
    assert c["selection"]["h_star"]["h_star"] == oracle
    ranking = c["selection"]["h_star"]["ranking"]
    assert [r["mean_validation_mse"] for r in ranking] == sorted(r["mean_validation_mse"] for r in ranking)


def test_tie_hstar_smallest_width():
    stats = {200: _stats(0.5), 5: _stats(0.5), 50: _stats(0.5), 20: _stats(0.9)}
    r = nom.select_width(stats)
    assert r["h_star"] == 5 and r["tie_applied"]  # pas 200
    assert nom.select_width({5: _stats(0.4), 10: _stats(0.3)})["h_star"] == 10  # argmin, not argmax
    assert not nom.select_width({5: _stats(0.4), 10: _stats(0.3)})["tie_applied"]


def test_tie_adam_grid_order():
    cfgs = [nom.adam_config(20, lr, b) for lr, b in nom.ADAM_GRID]
    plan_order = cfgs[::-1]  # reversed plan order: the rule follows the grid, not the plan
    r = par.select_adam(_collected([(c, 0.1) for c in plan_order]))
    assert r["selected"] == nom.config_id(cfgs[0]) and r["tie_applied"]
    r2 = par.select_adam(_collected([(cfgs[0], 0.3), (cfgs[1], 0.3), (cfgs[2], 0.2), (cfgs[3], 0.2)]))
    assert r2["selected"] == nom.config_id(cfgs[2])  # (1e-2, 32) before (1e-2, 200)


def test_tie_final_configuration_fewer_parameters_then_plan_order():
    big, small = nom.nominal_config((200,)), nom.nominal_config((5,))
    assert par.select_final(_collected([(big, 0.1), (small, 0.1)]))["selected"] == nom.config_id(small)
    a, b = nom.nominal_config((10,)), nom.nominal_config((10,), activation="relu")  # same number of parameters
    assert par.select_final(_collected([(b, 0.1), (a, 0.1)]))["selected"] == nom.config_id(b)
    assert par.select_final(_collected([(a, 0.1), (b, 0.1)]))["selected"] == nom.config_id(a)
    assert par.select_final(_collected([(big, 0.05), (small, 0.1)]))["selected"] == nom.config_id(big)


def test_figure_seed_validation_then_seed_order():
    def rec(seed, val, train=0.0):
        return {"identity": {"seed": seed}, "run_key": f"k{seed}", "weights_sha256": "w",
                "validation_metrics": {"mse": val}, "training_metrics": {"mse": train}}
    # validation and training order the seeds differently
    recs = [rec(s, v, t) for s, v, t in zip(SEEDS_MONTE_CARLO, (.5, .1, .3, .9, .4, .6, .7, .8, .2, .35),
                                            (.9, .8, .1, .5, .6, .7, .4, .3, .2, .05))]
    assert par.figure_seed(recs)["seed"] == SEEDS_MONTE_CARLO[1]  # minimal validation, not training
    ties = [rec(s, 0.2) for s in reversed(SEEDS_MONTE_CARLO)]
    assert par.figure_seed(ties)["seed"] == SEEDS_MONTE_CARLO[0]  # order of SEEDS_MONTE_CARLO
    with pytest.raises(ValueError):
        par.figure_seed(recs[:9])


def test_refusal_of_nan_and_incomplete_configuration():
    with pytest.raises(ValueError):
        nom.select_width({5: _stats(float("nan")), 10: _stats(0.2)})
    with pytest.raises(ValueError, match="incomplete"):
        nom.select_width({5: _stats(0.1, n=9), 10: _stats(0.2)})
    with pytest.raises(ValueError):
        nom.select_width({})


def test_shared_configurations_counted_once(campaign):
    sel = campaign["selection"]
    assert sel["counts"]["distinct_runs"] in (160, 190)
    ranking_ids = [r["config_id"] for r in sel["final"]["ranking"]]
    assert len(ranking_ids) == len(set(ranking_ids)) == sel["counts"]["distinct_configs"]


# ---------------------------------------------------------------- freeze
def test_freeze_written_once_and_different_content_refused(campaign, tmp_path):
    c = campaign
    path = tmp_path / "freeze.json"
    d1 = par.freeze_selection(path, c["selection"])
    assert par.freeze_selection(path, c["selection"]) == d1  # idempotent
    other = json.loads(json.dumps(c["selection"]))
    other["adam"]["selected"] = "other"
    with pytest.raises(ValueError, match="different"):
        par.freeze_selection(path, other)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen["schema"] == par.SELECTION_SCHEMA and len(frozen["payload_sha256"]) == 64
    assert "test" not in json.dumps(frozen).replace("tests", "")  # no test value in the freeze
    assert frozen["counts"]["realisations"] == 210
    targets = {t["role"] for t in frozen["retrain_targets"]}
    assert targets == {"final", "lbfgs_h_star", "adam_selected"}


# ---------------------------------------------------------------- watertightness
def test_campaign_and_selection_without_test_access(campaign):
    assert campaign["test_calls"] == []


def test_static_watertightness():
    names = {m: _fin_names(m) for m in (nom, par, fin)}
    forbidden = {"generate_test_set_2d", "prepare_data_2d", "DataSplits2D"}
    assert not forbidden & names[nom] and not forbidden & names[par]
    assert "generate_test_set_2d" in names[fin]  # the only module that reads the test set
    assert "freeze_selection" not in names[fin]  # never writes the selection


def _fin_names(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            out.update(a.name for a in node.names)
    return out


def test_record_with_test_field_rejected(campaign, tmp_path):
    path = next(campaign["store"].glob("*.json"))
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["test_metrics"] = {"mse": 0.0}
    copy = tmp_path / path.name
    copy.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(nom.StoreError, match="closed schema"):
        nom.load_record(copy)


# ---------------------------------------------------------------- final evaluation
def test_evaluation_without_freeze_or_with_altered_freeze_does_not_read_the_test_set(campaign, tmp_path):
    loads = []
    loader = lambda: (loads.append(1), campaign["test"])[1]
    tables = tmp_path / "t"
    with pytest.raises(fin.SelectionError, match="missing"):
        fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"],
                           campaign["val"], loader, tables_dir=tables, overrides=FAST)
    text = campaign["freeze"].read_text(encoding="utf-8")
    altered = tmp_path / "freeze.json"
    altered.write_text(text.replace('"relu"', '"rel0"', 1), encoding="utf-8")
    with pytest.raises(fin.SelectionError, match="altered"):
        fin.evaluate_final(altered, campaign["store"], campaign["train"], campaign["val"],
                           loader, tables_dir=tables, overrides=FAST)
    assert loads == []


def test_freeze_different_from_the_recomputation_refused(campaign, tmp_path):
    frozen = json.loads(campaign["freeze"].read_text(encoding="utf-8"))
    frozen.pop("payload_sha256")
    other = [r["config_id"] for r in frozen["final"]["ranking"]][-1]
    frozen["final"]["selected"] = other  # final configuration modified, internal hash recomputed
    frozen["payload_sha256"] = nom.sha256_text(nom.canonical(frozen))
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    loads = []
    with pytest.raises(fin.SelectionError, match="differs"):
        fin.evaluate_final(path, campaign["store"], campaign["train"], campaign["val"],
                           lambda: loads.append(1), tables_dir=tmp_path, overrides=FAST)
    assert loads == []


def test_record_modified_since_the_freeze_refused(campaign, tmp_path):
    import shutil
    store = tmp_path / "store"
    shutil.copytree(campaign["store"], store)
    victim = next(iter(sorted(store.glob("*.json"))))
    rec = json.loads(victim.read_text(encoding="utf-8"))
    rec["stop_cause"] = "none" if rec["stop_cause"] != "none" else "max_iter"
    rec["payload_sha256"] = nom.sha256_text(nom.canonical({k: v for k, v in rec.items() if k != "payload_sha256"}))
    victim.write_text(json.dumps(rec), encoding="utf-8")  # valid in itself, but different from the freeze
    with pytest.raises(fin.SelectionError):
        fin.load_frozen_selection(campaign["freeze"], store, campaign["train"], campaign["val"], FAST)


def test_altered_weights_detected_at_rebuild(campaign):
    key = campaign["selection"]["retrain_targets"][0]["run_key"]
    record = nom.load_record(campaign["store"] / f"{key}.json")
    record = {**record, "weights_sha256": "0" * 64}
    with pytest.raises(fin.SelectionError, match="fingerprint"):
        fin.rebuild_verified_model(record, campaign["train"], campaign["val"])


def test_final_evaluation_scope_oracle_and_unchanged_selection(campaign, tmp_path, monkeypatch, caplog):
    import shutil
    tables = tmp_path / "tables"
    shutil.copytree(campaign["tables"], tables, dirs_exist_ok=True) if campaign["tables"].exists() else None
    tables.mkdir(exist_ok=True)
    shutil.copy(campaign["freeze"], tables / par.SELECTION_FILE)
    loads, built = [], []
    original = fin.rebuild_verified_model
    monkeypatch.setattr(fin, "rebuild_verified_model",
                        lambda *a, **k: built.append(original(*a, **k)) or built[-1])
    with caplog.at_level(logging.INFO):
        result = fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"],
                                    campaign["val"], lambda: (loads.append(1), campaign["test"])[1],
                                    tables_dir=tables, overrides=FAST)
    sel = campaign["selection"]
    targets = {t["config_id"] for t in sel["retrain_targets"]}
    assert loads == [1]  # a single load of the test set
    assert set(result["configs"]) == targets and len(targets) in (2, 3)  # nothing else
    assert len(built) == len({t["run_key"] for t in sel["retrain_targets"]}) == 10 * len(targets)
    events = [r for r in caplog.records if r.getMessage() == "final_retrain"]
    assert len(events) == len(built) and all(r.fields["match"] for r in events)
    adam_ids = {r["config_id"] for r in sel["adam"]["ranking"]}
    assert len(targets & adam_ids) == 1  # the best adam only, never the other three
    # test metrics = NumPy oracle on the test set, distinct from the validation ones
    xs, ys = campaign["test"]
    xv, yv = campaign["val"]
    rows = _read_csv(tables / "mlp_2d_final_test_runs.csv")
    by_key = {r["run_key"]: r for r in rows}
    for model in built:
        key = next(k for k, r in by_key.items() if int(r["seed"]) == model.seed and
                   nom.weights_sha256(model.model) == nom.load_record(campaign["store"] / f"{k}.json")["weights_sha256"])
        pred = model.predict(xs)
        assert float(by_key[key]["test_mse"]) == pytest.approx(np.mean((ys - pred) ** 2), rel=1e-12)
        assert float(by_key[key]["test_mse"]) != pytest.approx(np.mean((yv - model.predict(xv)) ** 2), rel=1e-6)
    assert result["selection_files_sha256"][par.SELECTION_FILE] == nom.sha256_file(tables / par.SELECTION_FILE)
    assert result["figure_network"]["note"].startswith("one case out of 10")
    stats = _read_csv(tables / "mlp_2d_final_test_stats.csv")
    assert {r["metric"] for r in stats} == set(nom.METRICS) and len(stats) == 4 * len(targets)
    # idempotence: same freeze, same outputs
    first = {p.name: p.read_bytes() for p in tables.iterdir() if p.name.startswith("mlp_2d_final")}
    fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"], campaign["val"],
                       lambda: campaign["test"], tables_dir=tables, overrides=FAST)
    second = {p.name: p.read_bytes() for p in tables.iterdir() if p.name.startswith("mlp_2d_final")}
    first.pop("mlp_2d_final_evaluation.json"), second.pop("mlp_2d_final_evaluation.json")
    assert first == second


def test_rerun_refused_if_the_freeze_changes_after_the_evaluation(campaign, tmp_path):
    import shutil
    tables = tmp_path / "t"
    tables.mkdir()
    shutil.copy(campaign["freeze"], tables / par.SELECTION_FILE)
    (tables / fin.EVALUATION_FILE).write_text(json.dumps({"selection_sha256": "0" * 64}), encoding="utf-8")
    loads = []
    with pytest.raises(fin.SelectionError, match="rerun refused"):
        fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"],
                           campaign["val"], lambda: loads.append(1), tables_dir=tables, overrides=FAST)
    assert loads == []


def test_selection_files_modified_during_the_evaluation_are_detected(campaign, tmp_path):
    import shutil
    tables = tmp_path / "t"
    tables.mkdir()
    shutil.copy(campaign["freeze"], tables / par.SELECTION_FILE)
    (tables / fin.HSTAR_FILE).write_text("{}", encoding="utf-8")

    def tamper():
        (tables / fin.HSTAR_FILE).write_text('{"h_star": 5}', encoding="utf-8")  # the test modifies the selection
        return campaign["test"]

    with pytest.raises(fin.SelectionError, match="changed during"):
        fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"],
                           campaign["val"], tamper, tables_dir=tables, overrides=FAST)


# ------------------------------------------------------------------------------------------
# Final evaluation outputs against oracles independent of the freeze and of the code
# ------------------------------------------------------------------------------------------
# The oracles below only read the cache (raw JSON, without load_record) and do not
# reuse src.utils.metrics, nor the selection functions, nor the freeze: an error
# common to the freeze and to the evaluation (role pointing to a wrong configuration) or a
# wrong metric therefore cannot compensate. The test set used (53 points, another
# function, another seed) differs from the training set (200) and the validation set (40).
_WIDTHS = (5, 10, 20, 50, 100, 200)  # reference literals, not the constants of the code
_ADAM_GRID = {(1e-3, 32), (1e-3, 200), (1e-2, 32), (1e-2, 200)}
_ROLES = {"final", "lbfgs_h_star", "adam_selected"}


def _distinct_test_set():
    rng = np.random.default_rng(2026)
    x = rng.random((53, 2))
    return x, np.cos(5 * x[:, 0]) + x[:, 1] ** 2 + 0.05 * rng.standard_normal(53)


def _oracle_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mse = float(np.mean((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": 1.0 - float(np.sum((y - p) ** 2)) / ss_tot,
            "max_abs_error": float(np.max(np.abs(y - p)))}


def _raw_store(store):
    """Cache records, read as raw JSON: [(config, seed, record)]."""
    out = []
    for path in sorted(Path(store).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        out.append((rec["identity"]["config"], rec["identity"]["seed"], rec))
    return out


def _mean_validation_by_config(raw):
    groups = {}
    for cfg, seed, rec in raw:
        groups.setdefault(json.dumps(cfg, sort_keys=True), []).append(rec)
    assert all(len(v) == 10 for v in groups.values())
    return {k: (json.loads(k), float(np.mean([r["validation_metrics"]["mse"] for r in v])))
            for k, v in groups.items()}


def _expected_role_configs(store):
    """Expected configuration of each of the three roles, recomputed from the raw validation."""
    by_cfg = _mean_validation_by_config(_raw_store(store))
    nominal = {c["hidden_layer_sizes"][0]: m for c, m in by_cfg.values()
               if c["solver"] == "lbfgs" and c["activation"] == "tanh" and c["alpha"] == 1e-4
               and len(c["hidden_layer_sizes"]) == 1 and c["hidden_layer_sizes"][0] in _WIDTHS}
    assert set(nominal) == set(_WIDTHS)
    h_star = min(_WIDTHS, key=lambda h: (nominal[h], h))
    adam = {k: (c, m) for k, (c, m) in by_cfg.items() if c["solver"] == "adam"}
    assert {(c["learning_rate_init"], c["batch_size"]) for c, _ in adam.values()} == _ADAM_GRID
    best_adam = min(adam.values(), key=lambda cm: cm[1])[0]
    best_final = min(by_cfg.values(), key=lambda cm: cm[1])[0]
    lbfgs = next(c for c, _ in by_cfg.values()
                 if c["solver"] == "lbfgs" and c["activation"] == "tanh" and c["alpha"] == 1e-4
                 and c["hidden_layer_sizes"] == [h_star])
    return {"h_star": h_star, "final": best_final, "lbfgs_h_star": lbfgs,
            "adam_selected": best_adam}


@pytest.fixture(scope="module")
def final_outputs(campaign, tmp_path_factory):
    """Complete final evaluation on a distinct test set, plus the retrained models."""
    import shutil
    from threadpoolctl import threadpool_limits
    tables = tmp_path_factory.mktemp("sorties_finales")
    shutil.copy(campaign["freeze"], tables / par.SELECTION_FILE)
    test = _distinct_test_set()
    built = {}
    original = fin.rebuild_verified_model
    mp = pytest.MonkeyPatch()
    mp.setattr(fin, "rebuild_verified_model",
               lambda rec, *a, **k: built.setdefault(rec["run_key"], original(rec, *a, **k)))
    try:
        with threadpool_limits(limits=1):
            result = fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"],
                                        campaign["train"], campaign["val"], lambda: test,
                                        tables_dir=tables, overrides=FAST)
    finally:
        mp.undo()
    return {"tables": tables, "result": result, "built": built, "test": test,
            "runs": _read_csv(tables / "mlp_2d_final_test_runs.csv"),
            "stats": _read_csv(tables / "mlp_2d_final_test_stats.csv"),
            "comparison": _read_csv(tables / "mlp_2d_solver_comparison.csv"),
            "expected": _expected_role_configs(campaign["store"])}


def test_distinct_test_set_and_discriminating_expected_roles(campaign, final_outputs):
    """Power guard: the three roles target three different configurations, and
    the metrics computed on the test set differ from the validation and training ones."""
    exp = final_outputs["expected"]
    keys = {json.dumps(exp[r], sort_keys=True) for r in _ROLES}
    assert len(keys) == 3
    alpha_variant = [c for c, _, _ in _raw_store(campaign["store"])
                     if c["hidden_layer_sizes"] == exp["lbfgs_h_star"]["hidden_layer_sizes"]
                     and c["solver"] == "lbfgs" and c["alpha"] != 1e-4]
    assert alpha_variant  # an alpha variant exists in the cache (possible target of an error)
    assert len(final_outputs["test"][1]) not in (len(campaign["val"][1]), len(campaign["train"][1]))


def test_each_role_points_to_the_expected_configuration(campaign, final_outputs):
    """The identity evaluated under each role is compared with the expected definition
    (recomputed from the cache), never with the freeze produced by the same code."""
    exp = final_outputs["expected"]
    lit = exp["lbfgs_h_star"]
    # literal definition of the lbfgs H* role: lbfgs, tanh, (H*,), alpha = 1e-4
    assert (lit["solver"], lit["activation"], lit["hidden_layer_sizes"], lit["alpha"]) == (
        "lbfgs", "tanh", [exp["h_star"]], 1e-4)
    adam = exp["adam_selected"]
    assert adam["solver"] == "adam" and adam["hidden_layer_sizes"] == [exp["h_star"]]
    assert adam["activation"] == "tanh" and adam["alpha"] == 1e-4
    assert (adam["learning_rate_init"], adam["batch_size"]) in _ADAM_GRID
    by_role = {}
    for row in final_outputs["runs"]:
        record = json.loads((campaign["store"] / f"{row['run_key']}.json").read_text(encoding="utf-8"))
        assert record["identity"]["seed"] == int(row["seed"])
        for role in row["roles"].split(";"):
            by_role.setdefault(role, []).append((int(row["seed"]), record["identity"]["config"]))
    assert set(by_role) == _ROLES
    for role, entries in by_role.items():
        assert sorted(s for s, _ in entries) == sorted(SEEDS_MONTE_CARLO)  # 10 distinct seeds
        assert all(cfg == exp[role] for _, cfg in entries), role


def test_roles_consistent_across_the_four_outputs(final_outputs):
    runs = final_outputs["runs"]
    role_of = {}
    for r in runs:
        role_of.setdefault(r["config_id"], r["roles"])
        assert role_of[r["config_id"]] == r["roles"]
    for table in ("stats", "comparison"):
        assert {r["config_id"]: r["roles"] for r in final_outputs[table]} == role_of
    configs = final_outputs["result"]["configs"]
    assert {cid: ";".join(c["roles"]) for cid, c in configs.items()} == role_of
    for cid, c in configs.items():  # readable label consistent with the identity of the role
        assert c["label"] == next(r["label"] for r in final_outputs["stats"] if r["config_id"] == cid)


def test_test_metrics_of_each_run_equal_the_oracle(final_outputs):
    """MSE, RMSE, R2 and maximal error of each run against a NumPy oracle."""
    x, y = final_outputs["test"]
    assert len(final_outputs["runs"]) == 30
    swapped = 0
    for row in final_outputs["runs"]:
        pred = final_outputs["built"][row["run_key"]].predict(x)
        oracle = _oracle_metrics(y, pred)
        for name, value in oracle.items():
            assert float(row[f"test_{name}"]) == pytest.approx(value, rel=1e-9, abs=1e-12), name
        # R2 with swapped arguments (mutation): distinct from the expected value
        ss = float(np.sum((pred - np.mean(pred)) ** 2))
        if ss > 0:
            reversed_r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss
            swapped += not np.isclose(reversed_r2, oracle["r2"], rtol=1e-3)
    assert swapped >= 25  # power guard: the inversion is visible on almost all the runs


def test_test_metrics_distinct_from_validation_and_training(campaign, final_outputs):
    """A mixed-up data set (validation or training instead of test) is detected."""
    x, y = final_outputs["test"]
    for key in ("val", "train"):
        xo, yo = campaign[key]
        for row in final_outputs["runs"][::3]:
            model = final_outputs["built"][row["run_key"]]
            other = _oracle_metrics(yo, model.predict(xo))
            for name in ("mse", "r2", "max_abs_error"):
                assert float(row[f"test_{name}"]) != pytest.approx(other[name], rel=1e-6), (key, name)


def test_statistics_and_comparison_recomputed_from_the_runs(campaign, final_outputs):
    runs = {}
    for r in final_outputs["runs"]:
        runs.setdefault(r["config_id"], []).append(r)
    assert len(final_outputs["stats"]) == 4 * len(runs)
    for row in final_outputs["stats"]:
        values = np.array([float(r[f"test_{row['metric']}"]) for r in runs[row["config_id"]]])
        assert len(values) == 10
        expected = {"mean": values.mean(), "std": values.std(ddof=1), "median": np.median(values),
                    "min": values.min(), "max": values.max()}
        for name, value in expected.items():
            assert float(row[name]) == pytest.approx(value, rel=1e-9, abs=1e-12), (row["metric"], name)
        stored = final_outputs["result"]["configs"][row["config_id"]]["test"][row["metric"]]
        assert stored["mean"] == pytest.approx(float(row["mean"]), rel=1e-12)
    for row in final_outputs["comparison"]:
        mse = np.array([float(r["test_mse"]) for r in runs[row["config_id"]]])
        val = np.array([json.loads((campaign["store"] / f"{r['run_key']}.json").read_text(
            encoding="utf-8"))["validation_metrics"]["mse"] for r in runs[row["config_id"]]])
        assert int(row["n_runs"]) == 10
        for name, value in {"test_mse_mean": mse.mean(), "test_mse_std": mse.std(ddof=1),
                            "test_mse_min": mse.min(), "test_mse_max": mse.max(),
                            "val_mse_mean": val.mean()}.items():
            assert float(row[name]) == pytest.approx(value, rel=1e-9), name


def test_figure_network_oracle_metrics_and_minimal_validation_seed(campaign, final_outputs):
    """The figure network is the seed with minimal validation MSE of the expected final
    configuration; its test metrics are those of the oracle."""
    exp_final = final_outputs["expected"]["final"]
    final_runs = [(seed, rec) for cfg, seed, rec in _raw_store(campaign["store"]) if cfg == exp_final]
    best_seed = min(final_runs, key=lambda sr: (sr[1]["validation_metrics"]["mse"],
                                                SEEDS_MONTE_CARLO.index(sr[0])))[0]
    flagged = [r for r in final_outputs["runs"] if r["is_figure_network"] == "True"]
    assert len(flagged) == 1 and int(flagged[0]["seed"]) == best_seed
    figure = final_outputs["result"]["figure_network"]
    assert figure["seed"] == best_seed
    x, y = final_outputs["test"]
    model = final_outputs["built"][flagged[0]["run_key"]]
    for name, value in _oracle_metrics(y, model.predict(x)).items():
        assert figure["test_metrics"][name] == pytest.approx(value, rel=1e-9, abs=1e-12), name
