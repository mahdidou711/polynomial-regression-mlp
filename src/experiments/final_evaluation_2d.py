"""Single final evaluation on the test set; the only module that reads the test data.

Mandatory sequence: (1) check the selection freeze and recompute the selection from
the validation records alone; (2) retrain each model of the scope and check the
fingerprint of its weights; (3) only then, load the test set.
Scope: final configuration, lbfgs H* and best adam setting, 10 seeds each,
identical trainings de-duplicated. This module selects nothing and writes no
selection file: the fingerprints of the selection files are checked before and after.

Command: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m src.experiments.final_evaluation_2d
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.data.generator_2d import generate_test_set_2d
from src.experiments.runner_2d_nominal import (
    METRICS, NOMINAL_STORE, STOP_CAUSES, TABLES_DIR, StoreError, canonical, check_threads,
    config_from_dict, config_label, data_fingerprint, environment_identity, load_input_data,
    load_record, run_strict, sha256_file, sha256_text, summary_stats, weights_sha256,
    write_csv, write_json,
)
from src.experiments.runner_2d_parametric import (
    HSTAR_FILE, SELECTION_FILE, SELECTION_SCHEMA, compute_selection, selection_body,
)
from src.utils.logger import environment_info, get_logger, log_event
from src.utils.metrics import getMSE, max_abs_error, r2_score, rmse

EVALUATION_FILE = "mlp_2d_final_evaluation.json"
SELECTION_FILES = (SELECTION_FILE, HSTAR_FILE, "mlp_2d_adam_grid.csv")  # monitored before / after
_logger = logging.getLogger("polynomial_mlp.final_evaluation")


class SelectionError(RuntimeError):
    """Freeze missing, altered or different from the recomputation."""


def selection_fingerprints(tables_dir) -> dict:
    return {name: sha256_file(Path(tables_dir) / name)
            for name in SELECTION_FILES if (Path(tables_dir) / name).exists()}


def load_frozen_selection(selection_path, store, train, validation, overrides=None) -> dict:
    """Check the freeze (internal hash, cited records) and recompute it without the test set."""
    path = Path(selection_path)
    if not path.exists():
        raise SelectionError(f"selection freeze missing: {path}")
    try:
        frozen = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SelectionError(f"unreadable freeze: {exc}") from exc
    if frozen.get("schema") != SELECTION_SCHEMA or "payload_sha256" not in frozen:
        raise SelectionError("freeze: invalid schema")
    body = {k: v for k, v in frozen.items() if k != "payload_sha256"}
    if sha256_text(canonical(body)) != frozen["payload_sha256"]:
        raise SelectionError("freeze altered: invalid payload_sha256")
    xt, yt = train
    xv, yv = validation
    fingerprint = data_fingerprint(xt, yt, xv, yv)
    environment = environment_identity(check_threads())
    if frozen["data"] != fingerprint or frozen["environment"] != environment:
        raise SelectionError("freeze produced with other data or another environment")
    for item in frozen["records"]:
        try:
            record = load_record(Path(store) / f"{item['run_key']}.json")
        except StoreError as exc:
            raise SelectionError(f"invalid cited record: {exc}") from exc
        if record["payload_sha256"] != item["payload_sha256"]:
            raise SelectionError(f"record modified since the freeze: {item['run_key']}")
    recomputed, _ = compute_selection(store, fingerprint, environment, overrides)
    if selection_body(recomputed) != selection_body(frozen):
        raise SelectionError("the selection recomputed from validation differs from the freeze")
    return frozen


def rebuild_verified_model(record, train, validation, logger=None):
    """Retrain the recorded run and require the same weight fingerprint."""
    config = config_from_dict(record["identity"]["config"])
    seed = record["identity"]["seed"]
    result = run_strict(config, seed, *train, *validation, logger=logger)
    got = weights_sha256(result.model)
    log_event(logger or _logger, "final_retrain", seed=seed, run_key=record["run_key"],
              weights_sha256=got, expected=record["weights_sha256"],
              match=got == record["weights_sha256"])
    if got != record["weights_sha256"]:
        raise SelectionError(f"weight fingerprint mismatch: {record['run_key']}")
    return result


def _metrics(y, prediction):
    return {"mse": getMSE(y, prediction), "rmse": rmse(y, prediction),
            "r2": r2_score(y, prediction), "max_abs_error": max_abs_error(y, prediction)}


def _targets_by_config(frozen):
    """{config_id: ([roles], [targets])} for the evaluation scope, duplicate configurations merged."""
    out = {}
    for t in frozen["retrain_targets"]:
        roles, targets = out.setdefault(t["config_id"], ([], []))
        if t["role"] not in roles:
            roles.append(t["role"])
        if t["run_key"] not in {x["run_key"] for x in targets}:
            targets.append(t)
    return out


def evaluate_final(selection_path, store, train, validation, load_test, *, tables_dir,
                   logger=None, overrides=None) -> dict:
    """Check the freeze, retrain, and only then evaluate on the test set (a single load)."""
    tables_dir = Path(tables_dir)
    selection_path = Path(selection_path)
    before = selection_fingerprints(tables_dir)
    frozen = load_frozen_selection(selection_path, store, train, validation, overrides)
    selection_sha = sha256_file(selection_path)
    out_path = tables_dir / EVALUATION_FILE
    if out_path.exists():
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if previous.get("selection_sha256") != selection_sha:
            raise SelectionError("the freeze changed after the final evaluation: rerun refused")
    records = {}
    for item in frozen["records"]:
        records[item["run_key"]] = load_record(Path(store) / f"{item['run_key']}.json")
    groups = _targets_by_config(frozen)
    models = {}
    for cid, (roles, targets) in groups.items():
        for t in targets:
            models[t["run_key"]] = rebuild_verified_model(records[t["run_key"]], train,
                                                          validation, logger)
    x_test, y_test = load_test()  # first and only access to the test set
    test_fp = data_fingerprint(x_test, y_test, x_test, y_test)["train_sha256"]
    figure = frozen["final"]["figure_network"]
    run_rows, stat_rows, comparison, per_config = [], [], [], {}
    for cid, (roles, targets) in groups.items():
        cfg = config_from_dict(records[targets[0]["run_key"]]["identity"]["config"])
        test_metrics = []
        for t in targets:
            m = _metrics(y_test, models[t["run_key"]].predict(x_test))
            test_metrics.append(m)
            run_rows.append((cid, ";".join(roles), t["seed"], t["run_key"],
                             *(m[k] for k in METRICS),
                             "final" in roles and t["run_key"] == figure["run_key"]))
        stats = {k: summary_stats([m[k] for m in test_metrics]) for k in METRICS}
        recs = [records[t["run_key"]] for t in targets]
        val = {k: summary_stats([r["validation_metrics"][k] for r in recs]) for k in METRICS}
        per_config[cid] = {"roles": roles, "label": config_label(cfg), "test": stats,
                           "validation": val}
        for k in METRICS:
            stat_rows.append((cid, ";".join(roles), config_label(cfg), k,
                              *(stats[k][s] for s in ("mean", "std", "median", "min", "max"))))
        comparison.append((cid, ";".join(roles), config_label(cfg), cfg.solver, len(recs),
                           val["mse"]["mean"], stats["mse"]["mean"], stats["mse"]["std"],
                           stats["mse"]["min"], stats["mse"]["max"],
                           float(np.mean([r["diagnostics"]["n_iter"] for r in recs])),
                           float(np.mean([r["diagnostics"]["elapsed_seconds"] for r in recs])),
                           sum(bool(r["diagnostics"]["convergence_warnings"]) for r in recs),
                           *(sum(r["stop_cause"] == c for r in recs) for c in STOP_CAUSES)))
    write_csv(tables_dir / "mlp_2d_final_test_runs.csv",
              ("config_id", "roles", "seed", "run_key", *(f"test_{m}" for m in METRICS),
               "is_figure_network"), run_rows)
    write_csv(tables_dir / "mlp_2d_final_test_stats.csv",
              ("config_id", "roles", "label", "metric", "mean", "std", "median", "min", "max"),
              stat_rows)
    write_csv(tables_dir / "mlp_2d_solver_comparison.csv",
              ("config_id", "roles", "label", "solver", "n_runs", "val_mse_mean", "test_mse_mean",
               "test_mse_std", "test_mse_min", "test_mse_max", "n_iter_mean",
               "elapsed_seconds_mean", "n_warnings", *(f"stop_{c}" for c in STOP_CAUSES)),
              comparison)
    after = selection_fingerprints(tables_dir)
    if before != after:
        raise SelectionError("the selection files changed during the evaluation")
    figure_test = None
    if figure["run_key"] in models:
        figure_test = _metrics(y_test, models[figure["run_key"]].predict(x_test))
    evaluation = {"selection_sha256": selection_sha, "frozen_payload_sha256": frozen["payload_sha256"],
                  "test_sha256": test_fp, "n_test": int(len(y_test)),
                  "selection_files_sha256": after, "configs": per_config,
                  "figure_network": {**figure, "test_metrics": figure_test,
                                     "note": "one case out of 10, not representative of the configuration"},
                  "scope": "final configuration, lbfgs H*, best adam; 10 seeds each",
                  "environment": environment_info()}
    write_json(out_path, evaluation)
    return evaluation


def figure_inputs(selection_path=None, store=NOMINAL_STORE, train=None, validation=None,
                  load_test=generate_test_set_2d, tables_dir=TABLES_DIR, logger=None,
                  overrides=None) -> dict:
    """Verified models and data sets for the figures; requires the final evaluation to be done."""
    tables_dir = Path(tables_dir)
    selection_path = Path(selection_path or tables_dir / SELECTION_FILE)
    if train is None or validation is None:
        xt, yt, xv, yv = load_input_data()
        train, validation = (xt, yt), (xv, yv)
    frozen = load_frozen_selection(selection_path, store, train, validation, overrides)
    evaluation_path = tables_dir / EVALUATION_FILE
    if not evaluation_path.exists() or json.loads(evaluation_path.read_text(encoding="utf-8")
                                                  )["selection_sha256"] != sha256_file(selection_path):
        raise SelectionError("final evaluation missing or associated with another freeze")
    def model(item):
        return rebuild_verified_model(load_record(Path(store) / f"{item['run_key']}.json"),
                                      train, validation, logger)
    figure = frozen["final"]["figure_network"]
    out = {"figure_model": model(figure),
           "tanh_model": model(frozen["activation_figure"]["tanh"]),
           "relu_model": model(frozen["activation_figure"]["relu"]),
           "train": train, "validation": validation, "test": load_test(), "frozen": frozen}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=NOMINAL_STORE)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)
    check_threads()
    xt, yt, xv, yv = load_input_data()
    lg = get_logger("polynomial_mlp", log_file=args.log_file)
    log_event(lg, "final_evaluation_start", **environment_info())
    try:
        result = evaluate_final(args.tables_dir / SELECTION_FILE, args.store, (xt, yt), (xv, yv),
                                generate_test_set_2d, tables_dir=args.tables_dir, logger=lg)
    except Exception:
        log_event(lg, "final_evaluation_failed", level=logging.ERROR, exc_info=True)
        raise
    finally:
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
            handler.close()
    for cid, c in result["configs"].items():
        print(c["label"], c["roles"], "mean test MSE =", c["test"]["mse"]["mean"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
