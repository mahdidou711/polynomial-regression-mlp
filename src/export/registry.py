"""Declarative registry of the exported artefacts.

It fixes the naming: each table has an identifier ``stem`` that names
``results/tables/<stem>.csv``; each figure has a ``stem`` that names ``<stem>.pdf`` /
``<stem>.png`` (2D) or ``<stem>.png`` (3D) in ``results/figures/<polynomial|mlp>/``.

A table is *direct* (``derive is None``: the CSV already exists) or *derived* (a function
computes the rows from source CSV files; the derived CSV is written under the same identifier).
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.export.csv_io import is_missing, read_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Sources:
    """Read access to the source tables of a table directory."""

    def __init__(self, tables_dir, raw_csv=None):
        self.tables_dir = Path(tables_dir)
        self.raw_csv = Path(raw_csv or PROJECT_ROOT / "data" / "raw" / "sinc_dim2_input.csv")

    def rows(self, stem) -> list[dict]:
        return read_csv(self.tables_dir / f"{stem}.csv")

    def json(self, name) -> dict:
        import json
        return json.loads((self.tables_dir / name).read_text(encoding="utf-8"))

    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw_csv.read_bytes()).hexdigest()


@dataclass(frozen=True)
class TableSpec:
    stem: str
    source: str | None = None  # direct CSV (table identifier) or None if derived
    derive: Callable | None = None
    inputs: tuple = ()  # identifiers of the CSV files read by ``derive``
    summary: str = ""


# --------------------------------------------------------------------------
# Derivation functions (rows = dictionaries; nothing is written)
# --------------------------------------------------------------------------
def _f(value):
    return None if is_missing(value) else float(value)


def _config_label(row) -> str:
    label = f"{row['solver']}-{row['activation']}-{row['hidden_layers']}-a{float(row['alpha']):g}"
    if row["solver"] == "adam":
        label += f"-lr{float(row['learning_rate_init']):g}-b{row['batch_size']}"
    return label


def _config_rows(src: Sources, prefix: str | None = None) -> list[dict]:
    """One row per distinct configuration; ``prefix`` filters on the role (TOPO, REG, ...)."""
    out = []
    for row in src.rows("mlp_2d_config_stats"):
        roles = row["roles"].split(";")
        if prefix and not any(r.startswith(prefix + ":") for r in roles):
            continue
        adam = row["solver"] == "adam"
        out.append({
            "label": _config_label(row), "hidden_layers": row["hidden_layers"],
            "activation": row["activation"], "solver": row["solver"],
            "alpha": float(row["alpha"]),
            "learning_rate_init": float(row["learning_rate_init"]) if adam else None,
            "batch_size": row["batch_size"] if adam else None,
            "val_mse_mean": float(row["val_mse_mean"]), "val_mse_std": float(row["val_mse_std"]),
            "val_mse_min": float(row["val_mse_min"]), "val_mse_max": float(row["val_mse_max"]),
            "train_mse_mean": float(row["train_mse_mean"]),
            "val_r2_mean": float(row["val_r2_mean"]), "n_warnings": int(row["n_warnings"]),
            "stop_max_iter": int(row["stop_max_iter"]), "n_iter_mean": float(row["n_iter_mean"])})
    return out


def derive_topology(src):
    return _config_rows(src, "TOPO")


def derive_regularization(src):
    return _config_rows(src, "REG")


def derive_activation(src):
    return _config_rows(src, "ACT")


def derive_solver(src):
    return _config_rows(src, "SOLV")


def derive_configurations(src):
    return _config_rows(src)


def _exploration_summary(src: Sources, axis: str) -> list[dict]:
    """One row per value of ``axis`` (n_samples or sigma): distribution of C*, minimum of the test MSE."""
    stats = src.rows(f"exploration_{axis}_stats")
    cstar = src.rows(f"exploration_{axis}_cstar")
    out = []
    for value in dict.fromkeys(r[axis] for r in stats):
        block = [r for r in stats if r[axis] == value]
        c_block = [r for r in cstar if r[axis] == value]
        best = min(block, key=lambda r: (float(r["test_mse_median"]), int(r["degree"])))
        d12 = next(r for r in block if int(r["degree"]) == 12)
        out.append({axis: value, "n_runs": int(block[0]["n_runs"]),
                    "cstar_mode": int(float(c_block[0]["mode"])),
                    "cstar_median": float(c_block[0]["median"]),
                    "degree_min": int(best["degree"]), "test_mse_median_min": float(best["test_mse_median"]),
                    "test_mse_median_d12": float(d12["test_mse_median"])})
    return out


def derive_n_samples_summary(src):
    return _exploration_summary(src, "n_samples")


def derive_sigma_summary(src):
    return _exploration_summary(src, "sigma")


def derive_nominal_diagnostics(src):
    keep = ("degree", "rank", "kappa_h", "kappa_hth", "regime", "rel_error_normal_vs_svd",
            "rel_error_qr_vs_svd")
    return [{k: r[k] for k in keep} for r in src.rows("nominal_1d_seed42")]


def derive_final_summary(src):
    """One row per evaluated configuration: test MSE (5 statistics) and R2 (mean, minimum)."""
    rows = src.rows("mlp_2d_final_test_stats")
    out = []
    for cid in dict.fromkeys(r["config_id"] for r in rows):
        block = {r["metric"]: r for r in rows if r["config_id"] == cid}
        mse, r2 = block["mse"], block["r2"]
        out.append({"roles": mse["roles"], "label": mse["label"],
                    "mse_mean": float(mse["mean"]), "mse_std": float(mse["std"]),
                    "mse_median": float(mse["median"]), "mse_min": float(mse["min"]),
                    "mse_max": float(mse["max"]), "r2_mean": float(r2["mean"]),
                    "r2_min": float(r2["min"])})
    return out


def derive_provenance(src: Sources):
    """Provenance of the 2D data; no read of the test set."""
    from src.data.generator_2d import generate_training_set_2d, generate_validation_set_2d
    meta = src.json("mlp_2d_metadata.json")["data"]
    evaluation = src.json("mlp_2d_final_evaluation.json")
    return [
        {"data_set": "Input file sinc_dim2_input.csv", "n": len(generate_training_set_2d()[0]),
         "origin": "training abscissas supplied as input, order kept",
         "sha256": src.raw_sha256()[:16]},
        {"data_set": "Training", "n": len(generate_training_set_2d()[0]),
         "origin": "cardinal sine target computed on the abscissas of the file, noise-free",
         "sha256": meta["train_sha256"][:16]},
        {"data_set": "Validation", "n": len(generate_validation_set_2d()[0]),
         "origin": "20 x 20 grid of the cell centres of the unit square, deterministic",
         "sha256": meta["validation_sha256"][:16]},
        {"data_set": "Test", "n": int(evaluation["n_test"]),
         "origin": "40 x 40 grid, bounds included, read once after the freeze",
         "sha256": evaluation["test_sha256"][:16]},
    ]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
TABLES = (
    TableSpec("nominal_1d_seed42", source="nominal_1d_seed42", summary="Nominal error profile (seed 42), degrees 1 to 12"),
    TableSpec("nominal_1d_seed42_diagnostics", derive=derive_nominal_diagnostics, inputs=('nominal_1d_seed42',), summary='Conditioning diagnostics and OLS/QR/SVD gaps (seed 42)'),
    TableSpec("nominal_1d_stats_50_seeds", source="nominal_1d_stats_50_seeds", summary='Descriptive statistics over 50 seeds (median, quartiles)'),
    TableSpec("nominal_1d_cstar_50_seeds", source="nominal_1d_cstar_50_seeds", summary='Distribution of C* over 50 seeds'),
    TableSpec("exploration_n_samples_summary", derive=derive_n_samples_summary, inputs=('exploration_n_samples_stats', 'exploration_n_samples_cstar'), summary="Exploration 1 (sample size N): summary per N"),
    TableSpec("exploration_sigma_summary", derive=derive_sigma_summary, inputs=('exploration_sigma_stats', 'exploration_sigma_cstar'), summary='Exploration 2 (noise sigma): summary per sigma'),
    TableSpec("exploration_extrapolation_stats", source="exploration_extrapolation_stats", summary='Exploration 3 (extrapolation): MSE per domain'),
    TableSpec("exploration_n_samples_stats", source="exploration_n_samples_stats", summary='Exploration 1: full statistics per N and per degree'),
    TableSpec("exploration_sigma_stats", source="exploration_sigma_stats", summary='Exploration 2: full statistics per sigma and per degree'),
    TableSpec("exploration_n_samples_diagnostics", source="exploration_n_samples_diagnostics", summary='Conditioning diagnostics per N and per degree'),
    TableSpec("exploration_n_samples_variance", source="exploration_n_samples_variance", summary='Mean prediction variance per N (1/N scale)'),
    TableSpec("mlp_2d_nominal_stats", source="mlp_2d_nominal_stats", summary='Width study: validation MSE per width H, 10 seeds'),
    TableSpec("mlp_2d_sensitivity_topology", derive=derive_topology, inputs=('mlp_2d_config_stats',), summary='Topology sensitivity: depth against width'),
    TableSpec("mlp_2d_sensitivity_regularization", derive=derive_regularization, inputs=('mlp_2d_config_stats',), summary='Sensitivity to L2 regularisation'),
    TableSpec("mlp_2d_sensitivity_activation", derive=derive_activation, inputs=('mlp_2d_config_stats',), summary='Activation sensitivity: tanh against relu'),
    TableSpec("mlp_2d_sensitivity_solver", derive=derive_solver, inputs=('mlp_2d_config_stats',), summary='Solver sensitivity: lbfgs against adam (given budget and grid)'),
    TableSpec("mlp_2d_adam_grid", source="mlp_2d_adam_grid", summary='Adam grid, grid order as specified'),
    TableSpec("mlp_2d_solver_comparison", source="mlp_2d_solver_comparison", summary='lbfgs / adam comparison on the test set (10 seeds)'),
    TableSpec("mlp_2d_final_summary", derive=derive_final_summary, inputs=('mlp_2d_final_test_stats',), summary='Final evaluation on the test set (N = 1600), 10 seeds per configuration'),
    TableSpec("mlp_2d_final_test_stats", source="mlp_2d_final_test_stats", summary='Final evaluation: all the metrics'),
    TableSpec("mlp_2d_final_test_runs", source="mlp_2d_final_test_runs", summary='Final evaluation: result of each seed'),
    TableSpec("mlp_2d_configurations", derive=derive_configurations, inputs=('mlp_2d_config_stats',), summary='All the distinct configurations of the plan (validation)'),
    TableSpec("data_2d_provenance", derive=derive_provenance, inputs=('mlp_2d_metadata.json', 'mlp_2d_final_evaluation.json'), summary='Provenance and fingerprints of the 2D data'),
)

# Files of results/tables/ outside the table registry (bulky or technical); their role is
# documented here so that ``verify`` reports any undeclared file.
CSV_ONLY = {
    "exploration_extrapolation_runs": "raw results per seed (200 rows)",
    "exploration_n_samples_cstar": "distribution of C* per N (summarised in the summary table)",
    "exploration_n_samples_runs": "raw results per seed (3600 rows)",
    "exploration_sigma_cstar": "distribution of C* per sigma (summarised in the summary table)",
    "exploration_sigma_runs": "raw results per seed (3012 rows)",
    "mlp_2d_plan": "plan of the 210 realisations (traceability)",
    "mlp_2d_runs": "190 distinct trainings, one row per run (traceability)",
    "mlp_2d_config_stats": "source of the sensitivity tables (wide columns)",
}
JSON_FILES = ("exploration_1d_metadata.json", "mlp_2d_final_evaluation.json",
              "mlp_2d_hstar.json", "mlp_2d_metadata.json", "mlp_2d_selection.json")
SUMMARY_FILE = "export_summary.csv"

# Origin of each source file: regenerated by the 1D runners, rebuilt from the MLP
# cache (no training), or carried over unchanged (final evaluation, protected by the freeze).
SOURCES_1D = ("nominal_1d_seed42.csv", "nominal_1d_stats_50_seeds.csv",
              "nominal_1d_cstar_50_seeds.csv", "exploration_extrapolation_runs.csv",
              "exploration_extrapolation_stats.csv", "exploration_n_samples_cstar.csv",
              "exploration_n_samples_diagnostics.csv", "exploration_n_samples_runs.csv",
              "exploration_n_samples_stats.csv", "exploration_n_samples_variance.csv",
              "exploration_sigma_cstar.csv", "exploration_sigma_runs.csv",
              "exploration_sigma_stats.csv", "exploration_1d_metadata.json")
SOURCES_MLP_CACHE = ("mlp_2d_plan.csv", "mlp_2d_runs.csv", "mlp_2d_config_stats.csv",
                     "mlp_2d_nominal_stats.csv", "mlp_2d_hstar.json", "mlp_2d_metadata.json",
                     "mlp_2d_adam_grid.csv", "mlp_2d_selection.json")
SOURCES_CARRIED = ("mlp_2d_final_test_runs.csv", "mlp_2d_final_test_stats.csv",
                   "mlp_2d_solver_comparison.csv", "mlp_2d_final_evaluation.json")


def table_by_stem() -> dict:
    return {t.stem: t for t in TABLES}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FigureSpec:
    stem: str
    folder: str  # "polynomial" or "mlp"
    formats: tuple  # ("pdf", "png") or ("png",)
    tables: tuple = ()
    needs_models: bool = False  # requires verified rebuilds of networks (3 at most)
    description: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def files(self) -> list[str]:
        return [f"{self.stem}.{ext}" for ext in self.formats]


_BOTH, _PNG = ("pdf", "png"), ("png",)
FIGURES = (
    FigureSpec("nominal_1d_error_profile", "polynomial", _BOTH, ("nominal_1d_seed42",), description="MSE and test MSE against the degree, C* marked"),
    FigureSpec("nominal_1d_fits", "polynomial", _BOTH, ("nominal_1d_seed42",), description="panel of the 12 fitted polynomials (clipped panels flagged)"),
    FigureSpec("nominal_1d_parity", "polynomial", _BOTH, ("nominal_1d_seed42",), description="test parity for d = 1, C* and 12 (1:1 frame)"),
    FigureSpec("nominal_1d_conditioning", "polynomial", _BOTH, ("nominal_1d_seed42_diagnostics",), description="conditioning of the Vandermonde matrix"),
    FigureSpec("nominal_1d_campaign_50_seeds", "polynomial", _BOTH,
               ("nominal_1d_stats_50_seeds", "nominal_1d_cstar_50_seeds"),
               description="distribution of C* and median MSE / test MSE over 50 seeds"),
    FigureSpec("exploration_n_samples", "polynomial", _BOTH, ("exploration_n_samples_summary", "exploration_n_samples_stats"),
               description="exploration 1: effect of N"),
    FigureSpec("exploration_n_samples_seed42", "polynomial", _BOTH, description="exploration 1: illustrative realisation, seed 42"),
    FigureSpec("exploration_sigma", "polynomial", _BOTH, ("exploration_sigma_summary", "exploration_sigma_stats"),
               description="exploration 2: effect of the noise sigma"),
    FigureSpec("exploration_sigma_seed42", "polynomial", _BOTH, description="exploration 2: illustrative realisation, seed 42"),
    FigureSpec("exploration_extrapolation", "polynomial", _BOTH, ("exploration_extrapolation_stats",),
               description="exploration 3: extrapolation on [-0.3; 1.5]"),
    FigureSpec("mlp_2d_width", "mlp", _BOTH, ("mlp_2d_nominal_stats",),
               description="validation MSE per width H, H* marked"),
    FigureSpec("mlp_2d_sensitivities", "mlp", _BOTH, ("mlp_2d_configurations",), description="validation MSE of all the configurations"),
    FigureSpec("mlp_2d_convergence", "mlp", _BOTH, description="iterations and stopping causes per configuration"),
    FigureSpec("mlp_2d_adam_curves", "mlp", _BOTH, ("mlp_2d_adam_grid",),
               description="adam loss curves per grid setting"),
    FigureSpec("mlp_2d_parity", "mlp", _BOTH, ("mlp_2d_final_summary",),
               needs_models=True, description="2D parity (training and test), figure network"),
    FigureSpec("mlp_2d_residual_map", "mlp", _BOTH, needs_models=True,
               description="map of the absolute residual |y_pred - y|"),
    FigureSpec("mlp_2d_panel_3d", "mlp", _PNG, needs_models=True,
               description="panel: exact surface, predicted surface, absolute residual"),
    FigureSpec("mlp_2d_surface_exact", "mlp", _PNG, needs_models=True,
               description="exact surface"),
    FigureSpec("mlp_2d_surface_predicted", "mlp", _PNG, needs_models=True,
               description="surface predicted by the figure network"),
    FigureSpec("mlp_2d_surface_residuals", "mlp", _PNG, needs_models=True,
               description="3D absolute residual"),
    FigureSpec("mlp_2d_activation_tanh_relu", "mlp", _PNG, ("mlp_2d_sensitivity_activation",), needs_models=True,
               description="tanh and relu predicted surfaces"),
)


def figure_by_stem() -> dict:
    return {f.stem: f for f in FIGURES}


def expected_table_files() -> set[str]:
    """File names expected in results/tables/ (CSV, JSON, summary)."""
    names = {f"{t.stem}.csv" for t in TABLES}
    names |= {f"{stem}.csv" for stem in CSV_ONLY} | set(JSON_FILES) | {SUMMARY_FILE}
    return names
