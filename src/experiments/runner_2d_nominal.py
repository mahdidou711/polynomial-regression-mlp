"""Nominal 2D MLP campaign and run infrastructure.

This module runs the width study (H = 5..200, 10 seeds), serves as the foundation of the
parametric runner (cache, resume, statistics, tables) and selects H* on validation
only. It never references the test set.

Cache: one JSON file per realisation (configuration, seed), named by the SHA-256 of its
complete identity (configuration, fixed engine parameters, seed, data fingerprint,
versions, engine fingerprint, number of BLAS threads). A file exists only if the run is
complete (atomic write); resuming never recomputes a valid run.

Command: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m src.experiments.runner_2d_nominal

Warning: outside ``--verify-only``, this command rewrites in ``--tables-dir`` (by
default ``results/tables``) the tables ``mlp_2d_runs.csv``, ``mlp_2d_plan.csv``,
``mlp_2d_config_stats.csv`` and ``mlp_2d_metadata.json`` with stage B only (61 plan rows
instead of 211); only the re-aggregation of the parametric runner restores the reference tables.
For a check or a reproduction, use ``--verify-only`` or ``python main.py verify`` /
``python main.py reproduce --out DIR`` (which never write into ``results/tables``).
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

from src.data.generator_2d import generate_training_set_2d, generate_validation_set_2d
from src.models.mlp import MLPConfig, train_mlp
from src.utils.logger import environment_info, get_logger, log_event
from src.utils.seeding import SEEDS_MONTE_CARLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOMINAL_STORE = PROJECT_ROOT / "data" / "processed" / "mlp_2d_runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "mlp"
ENGINE_PATH = PROJECT_ROOT / "src" / "models" / "mlp.py"

WIDTHS = (5, 10, 20, 50, 100, 200)
TOPOLOGIES = ((20, 10), (50, 25))
ALPHAS = (1e-5, 1e-3, 1e-1)
REG_REFERENCE_WIDTH = 100  # over-parameterised architecture of the regularisation study
ADAM_GRID = ((1e-3, 32), (1e-3, 200), (1e-2, 32), (1e-2, 200))  # fixed grid order
N_RUNS = len(SEEDS_MONTE_CARLO)
METRICS = ("mse", "rmse", "r2", "max_abs_error")
STOP_CAUSES = ("none", "max_iter", "max_fun", "other_warning")
SCHEMA = "c0-mlp2d-run/1"
# Parameters imposed by the engine (train_mlp) and included in the identity of a run.
ENGINE_FIXED = {"early_stopping": False, "shuffle": True, "learning_rate": "constant",
                "beta_1": 0.9, "beta_2": 0.999}
RECORD_KEYS = frozenset({"schema", "identity", "run_key", "status", "parameters",
                         "training_metrics", "validation_metrics", "diagnostics",
                         "stop_cause", "weights_sha256", "payload_sha256"})

_logger = logging.getLogger("polynomial_mlp.runner_2d")


class StoreError(RuntimeError):
    """Cache record that is invalid, inconsistent or foreign to the plan."""


# --------------------------------------------------------------------------
# Canonical JSON and fingerprints
# --------------------------------------------------------------------------
def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def canonical(obj) -> str:
    """Compact JSON, sorted keys, no NaN: basis of all the fingerprints."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(f"{a.shape};".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def data_fingerprint(x_train, y_train, x_validation, y_validation) -> dict:
    """SHA-256 fingerprints of the bytes (float64) of the training and validation sets."""
    return {"train_sha256": _array_digest(x_train, y_train),
            "validation_sha256": _array_digest(x_validation, y_validation)}


def weights_sha256(model) -> str:
    """Fingerprint of the weights and biases (contiguous float64, shapes included)."""
    h = hashlib.sha256()
    for name, arrays in (("coefs", model.coefs_), ("intercepts", model.intercepts_)):
        for a in arrays:
            a = np.ascontiguousarray(a, dtype=np.float64)
            h.update(f"{name}:{a.shape};".encode())
            h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Configurations and plan
# --------------------------------------------------------------------------
def nominal_config(layers, **overrides) -> MLPConfig:
    """Default configuration (tanh, lbfgs, alpha 1e-4, 10000 / 100000, tol 1e-6) + overrides."""
    return MLPConfig(hidden_layer_sizes=tuple(layers), **overrides)


def adam_config(h: int, learning_rate_init: float, batch_size: int) -> MLPConfig:
    """Adam protocol: the rest (constant, early_stopping, shuffle, betas) comes from the engine."""
    return MLPConfig(hidden_layer_sizes=(h,), activation="tanh", solver="adam", alpha=1e-4,
                     max_iter=5000, tol=1e-8, n_iter_no_change=200,
                     learning_rate_init=learning_rate_init, batch_size=batch_size)


def config_dict(config: MLPConfig) -> dict:
    return _plain(asdict(config))


def config_from_dict(d: dict) -> MLPConfig:
    d = dict(d)
    d["hidden_layer_sizes"] = tuple(d["hidden_layer_sizes"])
    return MLPConfig(**d)


def config_id(config: MLPConfig) -> str:
    return sha256_text(canonical(config_dict(config)))


def config_label(config: MLPConfig) -> str:
    layers = "-".join(str(n) for n in config.hidden_layer_sizes)
    if config.solver == "adam":
        return f"adam-{layers}-lr{config.learning_rate_init:g}-b{config.batch_size}"
    return f"{config.solver}-{config.activation}-{layers}-a{config.alpha:g}"


def n_trainable_parameters(config: MLPConfig, n_in: int = 2, n_out: int = 1) -> int:
    sizes = (n_in, *config.hidden_layer_sizes, n_out)
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


@dataclass(frozen=True)
class PlanEntry:
    campaign: str
    role: str
    config: MLPConfig
    seeds: tuple = SEEDS_MONTE_CARLO


def _apply(config, overrides):
    return replace(config, **overrides) if overrides else config


def nominal_plan(overrides=None) -> list:
    """Width study: 6 widths. ``overrides`` (tests only) replaces fields of MLPConfig."""
    return [PlanEntry("NOM", f"H{h}", _apply(nominal_config((h,)), overrides)) for h in WIDTHS]


def contractual_configs() -> list:
    """All the contractual configurations, for any value of H*."""
    configs = []
    for h in WIDTHS:
        configs.append(nominal_config((h,)))
        configs += [nominal_config((h,), alpha=a) for a in ALPHAS]
        configs.append(nominal_config((h,), activation="relu"))
        configs += [adam_config(h, lr, b) for lr, b in ADAM_GRID]
    configs += [nominal_config(t) for t in TOPOLOGIES]
    return configs


def contractual_config_ids() -> frozenset:
    return frozenset(config_id(c) for c in contractual_configs())


def is_nominal_store(store) -> bool:
    return Path(store).resolve() == NOMINAL_STORE.resolve()


# --------------------------------------------------------------------------
# Environment: BLAS threads, identity
# --------------------------------------------------------------------------
def current_threads() -> int:
    """Largest number of threads among the loaded BLAS / OpenMP libraries."""
    return max((int(p["num_threads"]) for p in threadpool_info()), default=1)


def check_threads() -> int:
    """Refuse to start if a BLAS or OpenMP pool exceeds 1 thread."""
    pools = [p for p in threadpool_info() if int(p["num_threads"]) > 1]
    if pools:
        detail = ", ".join(f"{p.get('internal_api')}={p['num_threads']}" for p in pools)
        raise RuntimeError("a single BLAS/OpenMP thread is required (OMP_NUM_THREADS=1 "
                           f"OPENBLAS_NUM_THREADS=1 before starting Python): {detail}")
    return current_threads()


def environment_identity(threads: int) -> dict:
    env = environment_info()
    return {"versions": {k: env[k] for k in ("python", "numpy", "scipy", "scikit-learn")},
            "engine_sha256": sha256_file(ENGINE_PATH), "blas_threads": int(threads)}


def run_identity(config, seed, fingerprint, environment) -> dict:
    return {"config": config_dict(config), "engine_fixed": dict(ENGINE_FIXED),
            "seed": int(seed), "data": dict(fingerprint), **environment}


def run_key(identity) -> str:
    return sha256_text(canonical(identity))


def plan_keys(plan, fingerprint, environment) -> list:
    """Keys of all the realisations of the plan, in order (role duplicates included)."""
    return [run_key(run_identity(e.config, s, fingerprint, environment))
            for e in plan for s in e.seeds]


def contractual_keys(fingerprint, environment) -> frozenset:
    return frozenset(run_key(run_identity(c, s, fingerprint, environment))
                     for c in contractual_configs() for s in SEEDS_MONTE_CARLO)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
def stop_cause(diagnostics, solver) -> str:
    """Stopping cause deduced from the warning text kept by the engine."""
    warns = diagnostics["convergence_warnings"]
    if not warns:
        return "none"
    text = " ".join(warns).lower()
    if solver == "adam":
        return "max_iter" if "maximum iterations" in text else "other_warning"
    if "evaluations exceeds limit" in text:
        return "max_fun"
    if "iterations reached limit" in text:
        return "max_iter"
    return "other_warning"


def build_record(run, identity, key) -> dict:
    record = {"schema": SCHEMA, "identity": _plain(identity), "run_key": key,
              "status": "completed", "parameters": _plain(run.parameters),
              "training_metrics": _plain(run.training_metrics),
              "validation_metrics": _plain(run.validation_metrics),
              "diagnostics": _plain(run.diagnostics),
              "stop_cause": stop_cause(run.diagnostics, run.parameters["solver"]),
              "weights_sha256": weights_sha256(run.model)}
    record["payload_sha256"] = sha256_text(canonical(record))
    return _plain(record)


def load_record(path) -> dict:
    """Read and validate a record (closed schema, hash, key, name); StoreError otherwise."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StoreError(f"{path.name} : unreadable ({exc})") from exc
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise StoreError(f"{path.name} : closed schema not respected")
    if record["schema"] != SCHEMA or record["status"] != "completed":
        raise StoreError(f"{path.name} : invalid schema or status")
    body = {k: v for k, v in record.items() if k != "payload_sha256"}
    try:
        if sha256_text(canonical(body)) != record["payload_sha256"]:
            raise StoreError(f"{path.name} : invalid payload_sha256")
        if run_key(record["identity"]) != record["run_key"] or path.name != record["run_key"] + ".json":
            raise StoreError(f"{path.name} : key inconsistent with the identity")
    except (TypeError, ValueError) as exc:
        raise StoreError(f"{path.name} : invalid content ({exc})") from exc
    return record


def _atomic_write_text(path, text) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _atomic_write(path, record) -> Path:
    return _atomic_write_text(path, json.dumps(record, sort_keys=True, indent=1,
                                               allow_nan=False) + "\n")


def run_strict(config, seed, xt, yt, xv, yv, logger=None):
    """``train_mlp`` with any warning other than ConvergenceWarning turned into an error.

    Guard against ``Training interrupted by user``: scikit-learn swallows KeyboardInterrupt
    during an adam run and only emits a UserWarning; without this filter, a truncated model
    would pass for a finished one.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return train_mlp(config, seed, xt, yt, xv, yv, logger=logger)


def ensure_runs(plan, x_train, y_train, x_validation, y_validation, store, logger=None) -> dict:
    """Train the missing realisations; return {run_key: record}.

    A realisation shared between roles is trained only once; a valid run present
    in ``store`` is never recomputed. An error writes ``<key>.failed.json`` and is
    propagated (no silent exclusion). The nominal store refuses any non-contractual
    configuration.
    """
    store = Path(store)
    if is_nominal_store(store):
        allowed = contractual_config_ids()
        bad = [config_label(e.config) for e in plan if config_id(e.config) not in allowed]
        bad += [config_label(e.config) for e in plan if tuple(e.seeds) != SEEDS_MONTE_CARLO]
        if bad:
            raise ValueError(f"non-contractual configuration or seeds refused in the "
                             f"nominal store: {sorted(set(bad))}")
    threads = check_threads()
    fingerprint = data_fingerprint(x_train, y_train, x_validation, y_validation)
    environment = environment_identity(threads)
    store.mkdir(parents=True, exist_ok=True)
    out = {}
    for entry in plan:
        for seed in entry.seeds:
            identity = run_identity(entry.config, seed, fingerprint, environment)
            key = run_key(identity)
            if key in out:
                continue
            path = store / f"{key}.json"
            if path.exists():
                out[key] = load_record(path)
                continue
            try:
                result = run_strict(entry.config, seed, x_train, y_train, x_validation,
                                    y_validation, logger)
                record = build_record(result, identity, key)
            except BaseException as exc:
                failed = {"run_key": key, "identity": _plain(identity),
                          "error": repr(exc), "trace": traceback.format_exc()}
                _atomic_write(store / f"{key}.failed.json", failed)
                raise
            _atomic_write(path, record)
            stale = store / f"{key}.failed.json"
            if stale.exists():
                stale.unlink()
            out[key] = record
    return out


def verify_store(store, required=(), allowed=None) -> dict:
    """Read-only check of the store invariants; modifies or deletes nothing."""
    store = Path(store)
    anomalies, notes, keys = [], [], set()
    if not store.is_dir():
        return {"ok": False, "n_records": 0, "missing": list(required),
                "anomalies": [f"store missing: {store}"], "notes": []}
    for p in sorted(store.iterdir()):
        if p.name.endswith(".json.tmp"):
            notes.append(f"leftover temporary file (ignored): {p.name}")
        elif p.name.endswith(".failed.json"):
            anomalies.append(f"recorded failure to analyse: {p.name}")
        elif p.name.endswith(".json"):
            try:
                record = load_record(p)
            except StoreError as exc:
                anomalies.append(str(exc))
                continue
            keys.add(record["run_key"])
            if allowed is not None and record["run_key"] not in allowed:
                anomalies.append(f"{p.name} : foreign to the contractual plan")
        else:
            anomalies.append(f"foreign file: {p.name}")
    missing = [k for k in required if k not in keys]
    return {"ok": not anomalies and not missing, "n_records": len(keys),
            "missing": missing, "anomalies": anomalies, "notes": notes}


def load_config_records(config, seeds, store, fingerprint, environment) -> list:
    """Records of the seeds of ``config``; error if one is missing."""
    records = []
    for seed in seeds:
        key = run_key(run_identity(config, seed, fingerprint, environment))
        path = Path(store) / f"{key}.json"
        if not path.exists():
            raise StoreError(f"missing realisation: {config_label(config)}, seed {seed}")
        records.append(load_record(path))
    return records


# --------------------------------------------------------------------------
# Statistics and selection (validation only)
# --------------------------------------------------------------------------
def summary_stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}


def describe(records, expected_runs=N_RUNS) -> dict:
    """Descriptive statistics; refuses an incomplete sample."""
    if len(records) != expected_runs:
        raise ValueError(f"{len(records)} runs instead of {expected_runs}")
    out = {"n_runs": len(records)}
    for split in ("training", "validation"):
        out[split] = {m: summary_stats([r[f"{split}_metrics"][m] for r in records])
                      for m in METRICS}
    causes = [r["stop_cause"] for r in records]
    out["n_warnings"] = sum(bool(r["diagnostics"]["convergence_warnings"]) for r in records)
    out["stop_causes"] = {c: causes.count(c) for c in STOP_CAUSES}
    out["n_iter"] = summary_stats([r["diagnostics"]["n_iter"] for r in records])
    out["elapsed_seconds"] = summary_stats([r["diagnostics"]["elapsed_seconds"] for r in records])
    return out


def validation_mean(stats, expected_runs=N_RUNS) -> float:
    if stats["n_runs"] != expected_runs:
        raise ValueError("configuration incomplete")
    value = stats["validation"]["mse"]["mean"]
    if not np.isfinite(value):
        raise ValueError("non-finite mean validation MSE")
    return float(value)


def select_width(stats_by_width) -> dict:
    """H* = argmin of the mean validation MSE; exact tie: smallest width."""
    if not stats_by_width:
        raise ValueError("no width")
    ranking = sorted(((validation_mean(s), int(h)) for h, s in stats_by_width.items()))
    best = ranking[0][0]
    return {"h_star": ranking[0][1],
            "ranking": [{"width": h, "mean_validation_mse": m,
                         "n_runs": stats_by_width[h]["n_runs"]} for m, h in ranking],
            "tie_applied": sum(m == best for m, _ in ranking) > 1,
            "rule": "smallest mean validation MSE; tie: smallest width"}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if v is None:
        return ""
    return v


def write_csv(path, header, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_fmt(v) for v in row])
    os.replace(tmp, path)
    return path


def write_json(path, obj) -> Path:
    return _atomic_write_text(path, json.dumps(_plain(obj), sort_keys=True, indent=1,
                                               allow_nan=False) + "\n")


def collect_stats(plan, store, fingerprint, environment) -> dict:
    """{config_id: {config, roles, records, stats}} in plan order (distinct configurations)."""
    out = {}
    for entry in plan:
        cid = config_id(entry.config)
        if cid not in out:
            records = load_config_records(entry.config, entry.seeds, store, fingerprint, environment)
            out[cid] = {"config": entry.config, "roles": [], "records": records,
                        "stats": describe(records, len(entry.seeds))}
        out[cid]["roles"].append(f"{entry.campaign}:{entry.role}")
    return out


def write_run_tables(plan, collected, fingerprint, environment, tables_dir) -> list:
    """mlp_2d_plan.csv, mlp_2d_runs.csv, mlp_2d_config_stats.csv (deterministic plan order)."""
    tables_dir = Path(tables_dir)
    seen, plan_rows = set(), []
    for e in plan:
        cid = config_id(e.config)
        for s in e.seeds:
            key = run_key(run_identity(e.config, s, fingerprint, environment))
            plan_rows.append((e.campaign, e.role, cid, s, key, key in seen))
            seen.add(key)
    p1 = write_csv(tables_dir / "mlp_2d_plan.csv",
                   ("campaign", "role", "config_id", "seed", "run_key", "shared_with_earlier_role"),
                   plan_rows)
    run_rows, config_rows = [], []
    for cid, item in collected.items():
        c, st = item["config"], item["stats"]
        layers = "-".join(str(n) for n in c.hidden_layer_sizes)
        for r in item["records"]:
            d = r["diagnostics"]
            run_rows.append((r["run_key"], cid, layers, c.activation, c.solver, c.alpha,
                             c.learning_rate_init, c.batch_size, r["identity"]["seed"],
                             *(r["training_metrics"][m] for m in METRICS),
                             *(r["validation_metrics"][m] for m in METRICS),
                             d["n_iter"], d["loss"], d["status"], r["stop_cause"],
                             len(d["convergence_warnings"]), d["elapsed_seconds"],
                             r["weights_sha256"]))
        row = [cid, ";".join(item["roles"]), layers, c.activation, c.solver, c.alpha,
               c.learning_rate_init, c.batch_size, st["n_runs"]]
        for split in ("validation", "training"):
            for stat in ("mean", "std", "median", "min", "max"):
                row.append(st[split]["mse"][stat])
        row += [st["validation"]["r2"]["mean"], st["n_warnings"],
                *(st["stop_causes"][k] for k in STOP_CAUSES),
                st["n_iter"]["mean"], st["elapsed_seconds"]["mean"]]
        config_rows.append(tuple(row))
    p2 = write_csv(
        tables_dir / "mlp_2d_runs.csv",
        ("run_key", "config_id", "hidden_layers", "activation", "solver", "alpha",
         "learning_rate_init", "batch_size", "seed",
         *(f"train_{m}" for m in METRICS), *(f"val_{m}" for m in METRICS),
         "n_iter", "loss", "status", "stop_cause", "n_convergence_warnings",
         "elapsed_seconds", "weights_sha256"), run_rows)
    p3 = write_csv(
        tables_dir / "mlp_2d_config_stats.csv",
        ("config_id", "roles", "hidden_layers", "activation", "solver", "alpha",
         "learning_rate_init", "batch_size", "n_runs",
         *(f"val_mse_{s}" for s in ("mean", "std", "median", "min", "max")),
         *(f"train_mse_{s}" for s in ("mean", "std", "median", "min", "max")),
         "val_r2_mean", "n_warnings", *(f"stop_{k}" for k in STOP_CAUSES),
         "n_iter_mean", "elapsed_seconds_mean"), config_rows)
    return [p1, p2, p3]


def write_metadata(path, plan, fingerprint, environment, h_star=None) -> Path:
    meta = {"widths": WIDTHS, "topologies": TOPOLOGIES, "alphas": ALPHAS,
            "adam_grid": ADAM_GRID, "seeds": SEEDS_MONTE_CARLO,
            "lbfgs_budget": {"max_iter": 10000, "max_fun": 100000, "tol": 1e-6},
            "adam_protocol": {"max_iter": 5000, "tol": 1e-8, "n_iter_no_change": 200},
            "engine_fixed": ENGINE_FIXED, "data": fingerprint, "environment": environment,
            "h_star": h_star, "n_plan_entries": len(plan),
            "store": "data/processed/mlp_2d_runs",
            "selection": "mean validation MSE; ties: smallest width, then plan order"}
    return write_json(path, meta)


def load_input_data():
    """Training (reference CSV) and validation; the test set is not loaded here."""
    xt, yt = generate_training_set_2d()
    xv, yv = generate_validation_set_2d()
    return xt, yt, xv, yv


# --------------------------------------------------------------------------
# Entry point: stage B
# --------------------------------------------------------------------------
def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--store", type=Path, default=NOMINAL_STORE)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--log-file", type=Path, default=None, help="JSON Lines log (outside the repository)")
    parser.add_argument("--verify-only", action="store_true",
                        help="check the invariants without training")
    return parser


def main(argv=None) -> int:
    args = build_parser(__doc__.splitlines()[0]).parse_args(argv)
    threads = check_threads()
    xt, yt, xv, yv = load_input_data()
    fingerprint = data_fingerprint(xt, yt, xv, yv)
    environment = environment_identity(threads)
    plan = nominal_plan()
    if args.verify_only:
        report = verify_store(args.store, plan_keys(plan, fingerprint, environment),
                              contractual_keys(fingerprint, environment))
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1
    lg = get_logger("polynomial_mlp", log_file=args.log_file)
    log_event(lg, "nominal_2d_start", n_runs=len(plan) * N_RUNS, **environment_info())
    try:
        ensure_runs(plan, xt, yt, xv, yv, args.store, lg)
        collected = collect_stats(plan, args.store, fingerprint, environment)
        selection = select_width({int(c["config"].hidden_layer_sizes[0]): c["stats"]
                                  for c in collected.values()})
        write_run_tables(plan, collected, fingerprint, environment, args.tables_dir)
        write_csv(args.tables_dir / "mlp_2d_nominal_stats.csv",
                  ("width", "n_runs", "val_mse_mean", "val_mse_std", "val_mse_median",
                   "val_mse_min", "val_mse_max", "train_mse_mean", "val_r2_mean",
                   "n_warnings", "stop_max_iter", "stop_max_fun", "n_iter_mean"),
                  [(c["config"].hidden_layer_sizes[0], c["stats"]["n_runs"],
                    *(c["stats"]["validation"]["mse"][s] for s in
                      ("mean", "std", "median", "min", "max")),
                    c["stats"]["training"]["mse"]["mean"], c["stats"]["validation"]["r2"]["mean"],
                    c["stats"]["n_warnings"], c["stats"]["stop_causes"]["max_iter"],
                    c["stats"]["stop_causes"]["max_fun"], c["stats"]["n_iter"]["mean"])
                   for c in collected.values()])
        write_json(args.tables_dir / "mlp_2d_hstar.json", selection)
        write_metadata(args.tables_dir / "mlp_2d_metadata.json", plan, fingerprint,
                       environment, selection["h_star"])
    except Exception:
        log_event(lg, "nominal_2d_failed", level=logging.ERROR, exc_info=True)
        raise
    else:
        log_event(lg, "nominal_2d_done", h_star=selection["h_star"])
    finally:
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
            handler.close()
    print(f"H* = {selection['h_star']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
