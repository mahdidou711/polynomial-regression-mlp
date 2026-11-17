"""2D MLP sensitivity studies, selection on validation and freeze.

Campaigns: topology, regularisation, activation and solver (one factor at a time around
H*, lbfgs at its default budget except for the studied factor; adam with a given grid). The
selection uses only validation records; this module never references the test set.
It writes the selection freeze and stops: the final evaluation belongs to a separate
module.

Tie-breaking rules, applied only when the mean validation values are exactly equal:
final configuration: fewer trainable parameters, then plan order; figure seed:
order of SEEDS_MONTE_CARLO; adam setting: order of the grid.

Command: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m src.experiments.runner_2d_parametric
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.experiments.runner_2d_nominal import (
    ADAM_GRID, ALPHAS, N_RUNS, NOMINAL_STORE, REG_REFERENCE_WIDTH, TABLES_DIR, TOPOLOGIES,
    WIDTHS, PlanEntry, _apply, _atomic_write_text, adam_config, build_parser, canonical,
    check_threads, collect_stats, config_id, config_label, data_fingerprint,
    environment_identity, load_input_data, n_trainable_parameters, nominal_config,
    nominal_plan, plan_keys, select_width, sha256_text, validation_mean, verify_store,
    write_csv, write_json, write_metadata, write_run_tables, ensure_runs, config_dict,
    run_key, run_identity, load_record,
)
from src.utils.logger import environment_info, get_logger, log_event
from src.utils.seeding import SEEDS_MONTE_CARLO

SELECTION_SCHEMA = "c0-mlp2d-selection/1"
SELECTION_FILE = "mlp_2d_selection.json"
HSTAR_FILE = "mlp_2d_hstar.json"


def sensitivity_plan(h_star, overrides=None) -> list:
    """TOPO (2), REG (6 roles), ACT (2), SOLV (5); shared roles are kept."""
    plan = [PlanEntry("TOPO", "-".join(map(str, t)), _apply(nominal_config(t), overrides))
            for t in TOPOLOGIES]
    for label, h in (("H_star", h_star), (f"H{REG_REFERENCE_WIDTH}", REG_REFERENCE_WIDTH)):
        plan += [PlanEntry("REG", f"{label}:alpha={a:g}",
                           _apply(nominal_config((h,), alpha=a), overrides)) for a in ALPHAS]
    plan += [PlanEntry("ACT", "tanh", _apply(nominal_config((h_star,)), overrides)),
             PlanEntry("ACT", "relu", _apply(nominal_config((h_star,), activation="relu"), overrides))]
    plan.append(PlanEntry("SOLV", "lbfgs", _apply(nominal_config((h_star,)), overrides)))
    plan += [PlanEntry("SOLV", f"adam:lr={lr:g},batch={b}",
                       _apply(adam_config(h_star, lr, b), overrides)) for lr, b in ADAM_GRID]
    return plan


def full_plan(h_star, overrides=None) -> list:
    """NOM (6) + sensitivities (15) = 21 roles, 210 realisations."""
    return nominal_plan(overrides) + sensitivity_plan(h_star, overrides)


def count_plan(plan) -> dict:
    ids = [config_id(e.config) for e in plan]
    return {"roles": len(plan), "realisations": sum(len(e.seeds) for e in plan),
            "distinct_configs": len(set(ids)),
            "distinct_runs": len(set(ids)) * N_RUNS}


# --------------------------------------------------------------------------
# Selection (validation only)
# --------------------------------------------------------------------------
def _rank(items, tie_key):
    """items : (config_id, config, stats, plan_index). Sorted by (mean validation, tie rule)."""
    ranked = sorted(items, key=lambda it: (validation_mean(it[2]), *tie_key(it)))
    best = validation_mean(ranked[0][2])
    tie = sum(validation_mean(it[2]) == best for it in ranked) > 1
    return ranked, tie


def _ranking_rows(ranked, collected):
    return [{"config_id": cid, "label": config_label(cfg), "roles": collected[cid]["roles"],
             "mean_validation_mse": validation_mean(st), "n_runs": st["n_runs"]}
            for cid, cfg, st, _ in ranked]


def select_adam(collected) -> dict:
    """Adam setting with the minimal mean validation MSE; tie: grid order."""
    grid_index = {(lr, b): i for i, (lr, b) in enumerate(ADAM_GRID)}
    items = [(cid, c["config"], c["stats"], i) for i, (cid, c) in enumerate(collected.items())
             if c["config"].solver == "adam"]
    if not items:
        raise ValueError("no adam configuration")
    ranked, tie = _rank(items, lambda it: (grid_index[(it[1].learning_rate_init, it[1].batch_size)],))
    return {"selected": ranked[0][0], "ranking": _ranking_rows(ranked, collected),
            "tie_applied": tie,
            "rule": "smallest mean validation MSE; tie: grid order"}


def select_final(collected) -> dict:
    """Final configuration among all the distinct configurations of phases A and B."""
    items = [(cid, c["config"], c["stats"], i) for i, (cid, c) in enumerate(collected.items())]
    ranked, tie = _rank(items, lambda it: (n_trainable_parameters(it[1]), it[3]))
    return {"selected": ranked[0][0], "ranking": _ranking_rows(ranked, collected),
            "tie_applied": tie,
            "rule": "smallest mean validation MSE; tie: fewer trainable "
                    "parameters, then plan order"}


def figure_seed(records) -> dict:
    """Seed with the minimal validation MSE; tie: order of SEEDS_MONTE_CARLO."""
    by_seed = {r["identity"]["seed"]: r for r in records}
    best_seed, best = None, None
    for seed in SEEDS_MONTE_CARLO:
        if seed not in by_seed:
            raise ValueError(f"seed {seed} missing")
        value = by_seed[seed]["validation_metrics"]["mse"]
        if not np.isfinite(value):
            raise ValueError("non-finite validation MSE")
        if best is None or value < best:  # strict: the first seed wins on ties
            best_seed, best = seed, value
    return {"seed": best_seed, "validation_mse": best, "run_key": by_seed[best_seed]["run_key"],
            "weights_sha256": by_seed[best_seed]["weights_sha256"],
            "rule": "smallest validation MSE; tie: order of SEEDS_MONTE_CARLO"}


def _targets(role, cid, records):
    return [{"role": role, "config_id": cid, "seed": r["identity"]["seed"],
             "run_key": r["run_key"], "weights_sha256": r["weights_sha256"]} for r in records]


def compute_selection(store, fingerprint, environment, overrides=None) -> dict:
    """Complete selection, read-only access to the cache (no test data)."""
    nominal = collect_stats(nominal_plan(overrides), store, fingerprint, environment)
    width = select_width({int(c["config"].hidden_layer_sizes[0]): c["stats"]
                          for c in nominal.values()})
    h_star = width["h_star"]
    plan = full_plan(h_star, overrides)
    collected = collect_stats(plan, store, fingerprint, environment)
    adam = select_adam(collected)
    final = select_final(collected)
    final_fig = figure_seed(collected[final["selected"]]["records"])
    lbfgs_id = config_id(_apply(nominal_config((h_star,)), overrides))
    relu_id = config_id(_apply(nominal_config((h_star,), activation="relu"), overrides))
    act = {"tanh": {"config_id": lbfgs_id, **figure_seed(collected[lbfgs_id]["records"])},
           "relu": {"config_id": relu_id, **figure_seed(collected[relu_id]["records"])}}
    retrain = (_targets("final", final["selected"], collected[final["selected"]]["records"])
               + _targets("lbfgs_h_star", lbfgs_id, collected[lbfgs_id]["records"])
               + _targets("adam_selected", adam["selected"],
                          collected[adam["selected"]]["records"]))
    records = [{"config_id": cid, "run_key": r["run_key"], "seed": r["identity"]["seed"],
                "payload_sha256": r["payload_sha256"]}
               for cid, c in collected.items() for r in c["records"]]
    selection = {
        "schema": SELECTION_SCHEMA, "h_star": width, "adam": adam,
        "final": {**final, "config": config_dict(collected[final["selected"]]["config"]),
                  "figure_network": final_fig},
        "activation_figure": act, "retrain_targets": retrain,
        "plan": [{"campaign": e.campaign, "role": e.role, "config_id": config_id(e.config)}
                 for e in plan],
        "counts": count_plan(plan), "records": records,
        "records_digest": sha256_text(canonical(records)),
        "data": fingerprint, "environment": environment,
    }
    return selection, collected


def freeze_selection(path, selection) -> str:
    """Write the freeze once; refuse any different content. Return the SHA-256 of the file."""
    path = Path(path)
    body = dict(selection)
    body.pop("payload_sha256", None)
    body["payload_sha256"] = sha256_text(canonical(body))
    text = json.dumps(json.loads(canonical(body)), sort_keys=True, indent=1) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != json.loads(canonical(body)):
            raise ValueError(f"freeze already written with different content: {path}")
    else:
        _atomic_write_text(path, text)
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_body(selection) -> dict:
    """Comparable content of a freeze (without its own hash)."""
    return json.loads(canonical({k: v for k, v in selection.items() if k != "payload_sha256"}))


def write_adam_grid(path, collected, adam) -> Path:
    rank = {row["config_id"]: i + 1 for i, row in enumerate(adam["ranking"])}
    rows = []
    for cid, c in collected.items():
        if c["config"].solver != "adam":
            continue
        st, cfg = c["stats"], c["config"]
        rows.append((cfg.learning_rate_init, cfg.batch_size, cid, rank[cid],
                     cid == adam["selected"], st["validation"]["mse"]["mean"],
                     st["validation"]["mse"]["std"], st["validation"]["mse"]["min"],
                     st["validation"]["mse"]["max"], st["n_warnings"],
                     st["stop_causes"]["max_iter"], st["n_iter"]["mean"],
                     st["elapsed_seconds"]["mean"]))
    return write_csv(path, ("learning_rate_init", "batch_size", "config_id", "rank_validation",
                            "selected", "val_mse_mean", "val_mse_std", "val_mse_min",
                            "val_mse_max", "n_warnings", "stop_max_iter", "n_iter_mean",
                            "elapsed_seconds_mean"), rows)


# --------------------------------------------------------------------------
# Entry point: stage C (stops after writing the freeze)
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    args = build_parser(__doc__.splitlines()[0]).parse_args(argv)
    threads = check_threads()
    xt, yt, xv, yv = load_input_data()
    fingerprint = data_fingerprint(xt, yt, xv, yv)
    environment = environment_identity(threads)
    hstar_path = Path(args.tables_dir) / HSTAR_FILE
    if not hstar_path.exists():
        raise SystemExit("mlp_2d_hstar.json missing: run stage B first (runner_2d_nominal)")
    declared = json.loads(hstar_path.read_text(encoding="utf-8"))["h_star"]
    nominal = collect_stats(nominal_plan(), args.store, fingerprint, environment)
    recomputed = select_width({int(c["config"].hidden_layer_sizes[0]): c["stats"]
                               for c in nominal.values()})["h_star"]
    if declared != recomputed:
        raise SystemExit(f"H* declare ({declared}) different from the recomputation ({recomputed})")
    plan = full_plan(recomputed)
    if args.verify_only:
        keys = set(plan_keys(plan, fingerprint, environment))
        report = verify_store(args.store, keys, keys)
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1
    lg = get_logger("polynomial_mlp", log_file=args.log_file)
    log_event(lg, "parametric_2d_start", h_star=recomputed, **environment_info())
    try:
        ensure_runs(plan, xt, yt, xv, yv, args.store, lg)
        selection, collected = compute_selection(args.store, fingerprint, environment)
        write_run_tables(plan, collected, fingerprint, environment, args.tables_dir)
        write_adam_grid(Path(args.tables_dir) / "mlp_2d_adam_grid.csv", collected,
                        selection["adam"])
        write_metadata(Path(args.tables_dir) / "mlp_2d_metadata.json", plan, fingerprint,
                       environment, recomputed)
        digest = freeze_selection(Path(args.tables_dir) / SELECTION_FILE, selection)
    except Exception:
        log_event(lg, "parametric_2d_failed", level=logging.ERROR, exc_info=True)
        raise
    else:
        log_event(lg, "parametric_2d_done", frozen_sha256=digest)
    finally:
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
            handler.close()
    final = selection["final"]
    print(f"H* = {recomputed} ; final configuration = {final['ranking'][0]['label']} ; "
          f"figure seed = {final['figure_network']['seed']} ; freeze sha256 = {digest}")
    print("The selection freeze is written; the final evaluation is a separate step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
