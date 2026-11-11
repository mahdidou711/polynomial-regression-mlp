"""Campagne MLP 2D nominale et infrastructure des runs (etape 7, lot A).

Ce module execute EXP-2D-NOM (H = 5..200, 10 graines, DEC-18), sert de socle au runner
parametrique (cache, reprise, statistiques, tables) et choisit H* sur la validation
uniquement (DEC-20). Il ne reference jamais la base de test (reserve R3).

Cache : un fichier JSON par realisation (configuration, graine), nomme par le SHA-256 de son
identite complete (configuration, parametres fixes du moteur, graine, empreinte des donnees,
versions, empreinte du moteur, nombre de threads BLAS). Un fichier n'existe que si le run est
complet (ecriture atomique) ; la reprise ne recalcule jamais un run valide.

Commande : OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m src.experiments.runner_2d_nominal

Attention (AE7-04) : hors ``--verify-only``, cette commande reecrit dans ``--tables-dir`` (par
defaut ``results/tables``) les tables ``mlp_2d_runs.csv``, ``mlp_2d_plan.csv``,
``mlp_2d_config_stats.csv`` et ``mlp_2d_metadata.json`` avec le seul lot B (61 lignes de plan au
lieu de 211) ; seule la reagregation du runner parametrique restitue les tables de reference.
Pour un controle ou une reproduction, utiliser ``--verify-only`` ou ``python main.py verify`` /
``python main.py reproduce --out DIR`` (qui n'ecrivent jamais dans ``results/tables``).
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

from src.data.generator_2d import genere_base_apprentissage_2d, genere_base_validation_2d
from src.models.mlp import MLPConfig, train_mlp
from src.utils.logger import environment_info, get_logger, log_event
from src.utils.seeding import SEEDS_MONTE_CARLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOMINAL_STORE = PROJECT_ROOT / "data" / "processed" / "mlp_2d_runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "mlp"
ENGINE_PATH = PROJECT_ROOT / "src" / "models" / "mlp.py"

WIDTHS = (5, 10, 20, 50, 100, 200)  # ARB-04
TOPOLOGIES = ((20, 10), (50, 25))  # ARB-04
ALPHAS = (1e-5, 1e-3, 1e-1)  # ARB-06
REG_REFERENCE_WIDTH = 100  # architecture sur-parametree de EXP-2D-REG
ADAM_GRID = ((1e-3, 32), (1e-3, 200), (1e-2, 32), (1e-2, 200))  # DEC-19, ordre contractuel
N_RUNS = len(SEEDS_MONTE_CARLO)
METRICS = ("mse", "rmse", "r2", "max_abs_error")
STOP_CAUSES = ("none", "max_iter", "max_fun", "other_warning")
SCHEMA = "c0-mlp2d-run/1"
# Parametres imposes par le moteur (train_mlp) et inclus dans l'identite d'un run.
ENGINE_FIXED = {"early_stopping": False, "shuffle": True, "learning_rate": "constant",
                "beta_1": 0.9, "beta_2": 0.999}
RECORD_KEYS = frozenset({"schema", "identity", "run_key", "status", "parameters",
                         "training_metrics", "validation_metrics", "diagnostics",
                         "stop_cause", "weights_sha256", "payload_sha256"})

_logger = logging.getLogger("c0_tp2_mlp.runner_2d")


class StoreError(RuntimeError):
    """Enregistrement du cache invalide, incoherent ou etranger au plan."""


# --------------------------------------------------------------------------
# JSON canonique et empreintes
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
    """JSON compact, cles triees, sans NaN : base de toutes les empreintes."""
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
    """Empreintes SHA-256 des octets (float64) de l'apprentissage et de la validation."""
    return {"train_sha256": _array_digest(x_train, y_train),
            "validation_sha256": _array_digest(x_validation, y_validation)}


def weights_sha256(model) -> str:
    """Empreinte des poids et biais (float64 contigus, formes incluses)."""
    h = hashlib.sha256()
    for name, arrays in (("coefs", model.coefs_), ("intercepts", model.intercepts_)):
        for a in arrays:
            a = np.ascontiguousarray(a, dtype=np.float64)
            h.update(f"{name}:{a.shape};".encode())
            h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Configurations et plan
# --------------------------------------------------------------------------
def nominal_config(layers, **overrides) -> MLPConfig:
    """Configuration DEC-18 (tanh, lbfgs, alpha 1e-4, 10000 / 100000, tol 1e-6) + surcharges."""
    return MLPConfig(hidden_layer_sizes=tuple(layers), **overrides)


def adam_config(h: int, learning_rate_init: float, batch_size: int) -> MLPConfig:
    """Protocole DEC-19 : le reste (constant, early_stopping, shuffle, betas) vient du moteur."""
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
    """EXP-2D-NOM : 6 largeurs. ``overrides`` (tests uniquement) remplace des champs de MLPConfig."""
    return [PlanEntry("NOM", f"H{h}", _apply(nominal_config((h,)), overrides)) for h in WIDTHS]


def contractual_configs() -> list:
    """Toutes les configurations contractuelles, pour n'importe quelle valeur de H*."""
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
# Environnement : threads BLAS, identite
# --------------------------------------------------------------------------
def current_threads() -> int:
    """Plus grand nombre de threads parmi les bibliotheques BLAS / OpenMP chargees."""
    return max((int(p["num_threads"]) for p in threadpool_info()), default=1)


def check_threads() -> int:
    """Refuse de demarrer si un pool BLAS ou OpenMP depasse 1 thread (D7-04)."""
    pools = [p for p in threadpool_info() if int(p["num_threads"]) > 1]
    if pools:
        detail = ", ".join(f"{p.get('internal_api')}={p['num_threads']}" for p in pools)
        raise RuntimeError("un seul thread BLAS/OpenMP requis (OMP_NUM_THREADS=1 "
                           f"OPENBLAS_NUM_THREADS=1 avant le demarrage de Python) : {detail}")
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
    """Cles de toutes les realisations du plan, dans l'ordre (doublons de role inclus)."""
    return [run_key(run_identity(e.config, s, fingerprint, environment))
            for e in plan for s in e.seeds]


def contractual_keys(fingerprint, environment) -> frozenset:
    return frozenset(run_key(run_identity(c, s, fingerprint, environment))
                     for c in contractual_configs() for s in SEEDS_MONTE_CARLO)


# --------------------------------------------------------------------------
# Enregistrements
# --------------------------------------------------------------------------
def stop_cause(diagnostics, solver) -> str:
    """Cause d'arret deduite du texte d'avertissement conserve par le moteur."""
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
    """Lit et valide un enregistrement (schema ferme, hash, cle, nom) ; StoreError sinon."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StoreError(f"{path.name} : illisible ({exc})") from exc
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        raise StoreError(f"{path.name} : schema ferme non respecte (I7)")
    if record["schema"] != SCHEMA or record["status"] != "completed":
        raise StoreError(f"{path.name} : schema ou statut invalide")
    body = {k: v for k, v in record.items() if k != "payload_sha256"}
    try:
        if sha256_text(canonical(body)) != record["payload_sha256"]:
            raise StoreError(f"{path.name} : payload_sha256 invalide (I3)")
        if run_key(record["identity"]) != record["run_key"] or path.name != record["run_key"] + ".json":
            raise StoreError(f"{path.name} : cle incoherente avec l'identite (I2)")
    except (TypeError, ValueError) as exc:
        raise StoreError(f"{path.name} : contenu invalide ({exc})") from exc
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
    """``train_mlp`` avec tout avertissement autre que ConvergenceWarning transforme en erreur.

    Garde contre ``Training interrupted by user`` : scikit-learn absorbe KeyboardInterrupt
    pendant un run adam et ne produit qu'un UserWarning ; sans ce filtre, un modele tronque
    passerait pour termine.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return train_mlp(config, seed, xt, yt, xv, yv, logger=logger)


def ensure_runs(plan, x_train, y_train, x_validation, y_validation, store, logger=None) -> dict:
    """Entraine les realisations manquantes ; retourne {run_key: enregistrement}.

    Une realisation partagee entre roles n'est entrainee qu'une fois ; un run valide present
    dans ``store`` n'est jamais recalcule. Une erreur ecrit ``<cle>.failed.json`` et est
    propagee (aucune exclusion silencieuse). Le depot nominal refuse toute configuration non
    contractuelle (I8).
    """
    store = Path(store)
    if is_nominal_store(store):
        allowed = contractual_config_ids()
        bad = [config_label(e.config) for e in plan if config_id(e.config) not in allowed]
        bad += [config_label(e.config) for e in plan if tuple(e.seeds) != SEEDS_MONTE_CARLO]
        if bad:
            raise ValueError(f"configuration ou graines non contractuelles refusees dans le "
                             f"depot nominal (I8) : {sorted(set(bad))}")
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
    """Controle en lecture seule des invariants I1 a I4 ; ne modifie ni ne supprime rien."""
    store = Path(store)
    anomalies, notes, keys = [], [], set()
    if not store.is_dir():
        return {"ok": False, "n_records": 0, "missing": list(required),
                "anomalies": [f"depot absent : {store}"], "notes": []}
    for p in sorted(store.iterdir()):
        if p.name.endswith(".json.tmp"):
            notes.append(f"fichier temporaire residuel (ignore) : {p.name}")
        elif p.name.endswith(".failed.json"):
            anomalies.append(f"echec enregistre a analyser : {p.name}")
        elif p.name.endswith(".json"):
            try:
                record = load_record(p)
            except StoreError as exc:
                anomalies.append(str(exc))
                continue
            keys.add(record["run_key"])
            if allowed is not None and record["run_key"] not in allowed:
                anomalies.append(f"{p.name} : etranger au plan contractuel (I4)")
        else:
            anomalies.append(f"fichier etranger : {p.name}")
    missing = [k for k in required if k not in keys]
    return {"ok": not anomalies and not missing, "n_records": len(keys),
            "missing": missing, "anomalies": anomalies, "notes": notes}


def load_config_records(config, seeds, store, fingerprint, environment) -> list:
    """Enregistrements des graines de ``config`` ; erreur si l'un manque (I6)."""
    records = []
    for seed in seeds:
        key = run_key(run_identity(config, seed, fingerprint, environment))
        path = Path(store) / f"{key}.json"
        if not path.exists():
            raise StoreError(f"realisation manquante : {config_label(config)}, graine {seed}")
        records.append(load_record(path))
    return records


# --------------------------------------------------------------------------
# Statistiques et selection (validation uniquement)
# --------------------------------------------------------------------------
def summary_stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))}


def describe(records, expected_runs=N_RUNS) -> dict:
    """Statistiques descriptives (R2) ; refuse un effectif incomplet (I6)."""
    if len(records) != expected_runs:
        raise ValueError(f"{len(records)} runs au lieu de {expected_runs} (I6)")
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
        raise ValueError("EQM de validation moyenne non finie")
    return float(value)


def select_width(stats_by_width) -> dict:
    """H* = argmin de l'EQM de validation moyenne ; egalite exacte : plus petite largeur (D7-01)."""
    if not stats_by_width:
        raise ValueError("aucune largeur")
    ranking = sorted(((validation_mean(s), int(h)) for h, s in stats_by_width.items()))
    best = ranking[0][0]
    return {"h_star": ranking[0][1],
            "ranking": [{"width": h, "mean_validation_mse": m,
                         "n_runs": stats_by_width[h]["n_runs"]} for m, h in ranking],
            "tie_applied": sum(m == best for m, _ in ranking) > 1,
            "rule": "plus petite moyenne de l'EQM de validation ; egalite : plus petite largeur"}


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
    """{config_id: {config, roles, records, stats}} dans l'ordre du plan (configs distinctes)."""
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
    """mlp_2d_plan.csv, mlp_2d_runs.csv, mlp_2d_config_stats.csv (ordre deterministe du plan)."""
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
            "selection": "EQM de validation moyenne ; egalites : D7-01 (avenant 11)"}
    return write_json(path, meta)


def load_official_data():
    """Apprentissage (CSV officiel) et validation ; le test n'est pas charge ici."""
    xt, yt = genere_base_apprentissage_2d()
    xv, yv = genere_base_validation_2d()
    return xt, yt, xv, yv


# --------------------------------------------------------------------------
# Point d'entree : lot B
# --------------------------------------------------------------------------
def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--store", type=Path, default=NOMINAL_STORE)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--log-file", type=Path, default=None, help="journal JSON Lines (hors depot)")
    parser.add_argument("--verify-only", action="store_true",
                        help="controle des invariants sans entrainement")
    return parser


def main(argv=None) -> int:
    args = build_parser(__doc__.splitlines()[0]).parse_args(argv)
    threads = check_threads()
    xt, yt, xv, yv = load_official_data()
    fingerprint = data_fingerprint(xt, yt, xv, yv)
    environment = environment_identity(threads)
    plan = nominal_plan()
    if args.verify_only:
        report = verify_store(args.store, plan_keys(plan, fingerprint, environment),
                              contractual_keys(fingerprint, environment))
        print(json.dumps(report, indent=1))
        return 0 if report["ok"] else 1
    lg = get_logger("c0_tp2_mlp", log_file=args.log_file)
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
