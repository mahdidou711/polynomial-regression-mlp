"""Read-only checks of the exported artefacts.

Each check returns a ``Check`` (status ``passed``, ``failed``, ``warning`` or
``not run``); none modifies a file, trains a network or reads the test set.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.export import registry
from src.export.build import artifact_list, sha256_file
from src.export.csv_io import read_csv
from src.export.layout import Layout

PASSED, FAILED, WARNING, NOT_RUN = "passed", "failed", "warning", "not run"
STEM = re.compile(r"^[a-z0-9_]+$")


@dataclass
class Check:
    name: str
    status: str
    details: list = field(default_factory=list)

    @classmethod
    def from_issues(cls, name, issues, warn=False):
        return cls(name, (WARNING if warn else FAILED) if issues else PASSED, list(issues))


# --------------------------------------------------------------------------
# Naming, presence, orphans
# --------------------------------------------------------------------------
def check_files(layout: Layout) -> Check:
    """Every artefact of the registry exists; no undeclared file; names are valid."""
    issues = []
    for spec in registry.TABLES:
        if not STEM.match(spec.stem):
            issues.append(f"invalid table identifier: {spec.stem}")
        if not (layout.tables / f"{spec.stem}.csv").exists():
            issues.append(f"missing: results/tables/{spec.stem}.csv")
    for stem in registry.CSV_ONLY:
        if not (layout.tables / f"{stem}.csv").exists():
            issues.append(f"missing: results/tables/{stem}.csv")
    for name in registry.JSON_FILES:
        if not (layout.tables / name).exists():
            issues.append(f"missing: results/tables/{name}")
    present = {p.name for p in layout.tables.iterdir() if p.is_file() and p.name != ".gitkeep"}
    for name in sorted(present - registry.expected_table_files()):
        issues.append(f"undeclared file in results/tables/: {name}")
    if not (layout.tables / registry.SUMMARY_FILE).exists():
        issues.append(f"missing: results/tables/{registry.SUMMARY_FILE}")
    for fig in registry.FIGURES:
        if not STEM.match(fig.stem):
            issues.append(f"invalid figure identifier: {fig.stem}")
        for name in fig.files():
            if not (layout.figure_dir(fig.folder) / name).exists():
                issues.append(f"missing: results/figures/{fig.folder}/{name}")
    for folder in ("polynomial", "mlp"):
        expected = {name for f in registry.FIGURES if f.folder == folder for name in f.files()}
        directory = layout.figure_dir(folder)
        present = {p.name for p in directory.iterdir() if p.is_file() and p.name != ".gitkeep"}
        for name in sorted(present - expected):
            issues.append(f"undeclared figure: results/figures/{folder}/{name}")
    if not layout.figures_manifest.exists():
        issues.append("missing: results/figures/figures_manifest.json")
    return Check.from_issues("naming: presence and undeclared files", issues)


def check_figure_formats(layout: Layout) -> Check:
    """2D: PDF and PNG; 3D: PNG only; 300 DPI PNG; valid file headers."""
    issues = []
    for fig in registry.FIGURES:
        directory = layout.figure_dir(fig.folder)
        for name in fig.files():
            path = directory / name
            if not path.exists():
                continue
            head = path.read_bytes()[:8]
            if name.endswith(".pdf") and not head.startswith(b"%PDF-"):
                issues.append(f"{name} : invalid PDF header")
            if name.endswith(".png"):
                if head != b"\x89PNG\r\n\x1a\n":
                    issues.append(f"{name} : invalid PNG header")
                else:
                    data = path.read_bytes()
                    i = data.find(b"pHYs")
                    ppm = int.from_bytes(data[i + 4:i + 8], "big") if i > 0 else 0
                    if abs(ppm * 0.0254 - 300) > 1:
                        issues.append(f"{name} : resolution {ppm * 0.0254:.0f} DPI instead of 300")
        if fig.formats == ("png",) and (directory / f"{fig.stem}.pdf").exists():
            issues.append(f"{fig.stem} : a PDF exists for a 3D figure (PNG only expected)")
    return Check.from_issues("figures: formats and resolution", issues)


# --------------------------------------------------------------------------
# Manifest, final evaluation
# --------------------------------------------------------------------------
def check_manifest(layout: Layout) -> Check:
    if not layout.manifest.exists():
        return Check("manifest: fingerprints", FAILED, [f"missing: {layout.manifest.name}"])
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    issues = []
    listed = {a["path"]: a for a in manifest["artifacts"]}
    current = {a["path"]: a for a in artifact_list(layout)}
    for path in sorted(set(current) - set(listed)):
        issues.append(f"artefact missing from the manifest: {path}")
    for path in sorted(set(listed) - set(current)):
        issues.append(f"artefact of the manifest not found: {path}")
    for path in sorted(set(listed) & set(current)):
        if listed[path]["sha256"] != current[path]["sha256"]:
            issues.append(f"different fingerprint: {path}")
    gel = layout.tables / "mlp_2d_selection.json"
    if manifest["protocol"]["freeze_selection_sha256"] != sha256_file(gel):
        issues.append("freeze fingerprint differs from the one in the manifest")
    return Check.from_issues("manifest: fingerprints of all the artefacts", issues)


def _close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)


def check_final_evaluation(layout: Layout) -> Check:
    """Consistency of the final evaluation without replaying it: freeze, fingerprints, statistics."""
    from src.experiments.runner_2d_nominal import canonical, sha256_text
    issues = []
    tables = layout.tables
    selection = json.loads((tables / "mlp_2d_selection.json").read_text(encoding="utf-8"))
    evaluation = json.loads((tables / "mlp_2d_final_evaluation.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in selection.items() if k != "payload_sha256"}
    if sha256_text(canonical(body)) != selection["payload_sha256"]:
        issues.append("freeze: invalid internal payload_sha256")
    if evaluation["selection_sha256"] != sha256_file(tables / "mlp_2d_selection.json"):
        issues.append("final evaluation associated with another freeze")
    if evaluation["frozen_payload_sha256"] != selection["payload_sha256"]:
        issues.append("final evaluation: internal freeze fingerprint differs")
    runs = read_csv(tables / "mlp_2d_final_test_runs.csv")
    stats = read_csv(tables / "mlp_2d_final_test_stats.csv")
    comparison = {r["config_id"]: r for r in read_csv(tables / "mlp_2d_solver_comparison.csv")}
    config_stats = {r["config_id"]: r for r in read_csv(tables / "mlp_2d_config_stats.csv")}
    column = {"mse": "test_mse", "rmse": "test_rmse", "r2": "test_r2",
              "max_abs_error": "test_max_abs_error"}
    for row in stats:
        values = np.array([float(r[column[row["metric"]]]) for r in runs
                           if r["config_id"] == row["config_id"]])
        if len(values) != 10:
            issues.append(f"{row['label']} : {len(values)} seeds instead of 10")
            continue
        recomputed = {"mean": values.mean(), "std": values.std(ddof=1), "median": np.median(values),
                      "min": values.min(), "max": values.max()}
        for key, value in recomputed.items():
            if not _close(row[key], value):
                issues.append(f"{row['label']} {row['metric']} {key} : table {row[key]} "
                              f"against recomputed {value!r}")
        if row["metric"] == "mse":
            cid = row["config_id"]
            if not _close(comparison[cid]["test_mse_mean"], row["mean"]):
                issues.append(f"{row['label']} : mean test MSE differs from mlp_2d_solver_comparison")
            if not _close(comparison[cid]["val_mse_mean"], config_stats[cid]["val_mse_mean"]):
                issues.append(f"{row['label']} : validation MSE differs from mlp_2d_config_stats")
            if not _close(evaluation["configs"][cid]["test"]["mse"]["mean"], row["mean"]):
                issues.append(f"{row['label']} : mean test MSE differs from the evaluation JSON")
    if evaluation["n_test"] != 1600:
        issues.append(f"n_test = {evaluation['n_test']} instead of 1600")
    return Check.from_issues("final evaluation: internal consistency and freeze (test set not read again)",
                             issues)


STATIC_CHECKS = (check_files, check_figure_formats, check_manifest,
                 check_final_evaluation)


def run_static_checks(layout: Layout) -> list[Check]:
    return [check(layout) for check in STATIC_CHECKS]
