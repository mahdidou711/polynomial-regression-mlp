"""Production and comparison of the exported artefacts (CSV/JSON tables, figures, manifests).

``produce`` writes all the artefacts into a given layout (``Layout``), never elsewhere:

* 1D tables: recomputed by the 1D runners (6 s + 9 s, deterministic);
* 2D MLP tables of stages B and C: rebuilt from the cache of 190 runs, **without any
  training** (a prior check of the cache refuses to continue if it is incomplete);
* MLP figures coming from the campaigns: from the same cache;
* figures that depend on a network (``needs_models``): only with ``with_models=True``,
  through the verified rebuild (weight fingerprint) of the three figure networks; otherwise
  the existing files are carried over unchanged;
* final evaluation tables: carried over unchanged from the reference (protected by the
  freeze); when the reference has none, they are computed from the cache.

``compare`` compares two layouts byte for byte. ``export`` and ``verify`` (see ``main.py``)
are compositions of ``produce`` in a temporary directory.
"""

import contextlib
import io
import json
import logging
import shutil
from pathlib import Path

from src.export import build, registry
from src.export.layout import Layout, REPO


class ExportError(RuntimeError):
    """Unmet precondition or drift detected: nothing is written to the repository."""


@contextlib.contextmanager
def quiet():
    """Silence the logs and prints of the runners called in-process."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def carry_over(out: Layout, reference: Layout, names, folder=None) -> list[str]:
    """Copy from the reference the files that are not regenerated; return those actually copied."""
    copied = []
    for name in names:
        source = (reference.tables if folder is None else folder(reference)) / name
        target = (out.tables if folder is None else folder(out)) / name
        if source.exists():
            _copy(source, target)
            copied.append(name)
    return copied


def regenerate_1d(out: Layout) -> None:
    from src.experiments import runner_1d_explorations, runner_1d_nominal
    with quiet():
        runner_1d_nominal.main(["--tables-dir", str(out.tables), "--figures-dir", str(out.fig_poly)])
        runner_1d_explorations.main(["--tables-dir", str(out.tables),
                                     "--figures-dir", str(out.fig_poly)])


def _mlp_context():
    from src.experiments import runner_2d_nominal as nom
    threads = nom.check_threads()  # explicit RuntimeError if more than one BLAS thread
    xt, yt, xv, yv = nom.load_input_data()
    fingerprint = nom.data_fingerprint(xt, yt, xv, yv)
    return nom, fingerprint, nom.environment_identity(threads)


def require_complete_cache(store, h_star=None) -> dict:
    """Read-only check: the cache covers the whole plan; otherwise ``ExportError``."""
    from src.experiments import runner_2d_parametric as par
    nom, fingerprint, environment = _mlp_context()
    plan = nom.nominal_plan() if h_star is None else par.full_plan(h_star)
    keys = set(nom.plan_keys(plan, fingerprint, environment))
    allowed = nom.contractual_keys(fingerprint, environment) if h_star is None else keys
    report = nom.verify_store(store, keys, allowed)
    if not report["ok"]:
        raise ExportError("MLP run cache incomplete or non-conforming (no training "
                          f"will be launched): {report['anomalies'][:3]} ; missing: "
                          f"{len(report['missing'])}")
    return report


def regenerate_mlp_tables(out: Layout, store=None, train: bool = False) -> int:
    """Tables of stages B and C and campaign figures.

    ``train=False`` (default): from the cache, without any training (an incomplete cache
    raises ``ExportError``). ``train=True``: full reproduction, reserved for ``produce_full``;
    ``store`` must then be an isolated store (never the nominal versioned store).
    """
    from src.experiments import runner_2d_nominal as nom
    from src.experiments import runner_2d_parametric as par
    from src.visualization import plots_2d
    store = Path(store or nom.NOMINAL_STORE)
    if train:
        if nom.is_nominal_store(store):
            raise ExportError("full reproduction refused in the nominal versioned store")
    else:
        require_complete_cache(store)
    argv = ["--store", str(store), "--tables-dir", str(out.tables)]
    with quiet():
        if nom.main(argv) != 0:
            raise ExportError("stage B: failure")
    h_star = build_hstar(out)
    if not train:
        require_complete_cache(store, h_star)
    with quiet():
        if par.main(argv) != 0:
            raise ExportError("stage C: failure")
        _, fingerprint, environment = _mlp_context()
        selection, collected = par.compute_selection(store, fingerprint, environment)
        plots_2d.save_campaign_figures(collected, selection["h_star"]["h_star"], out.fig_mlp)
    return h_star


def build_hstar(out: Layout) -> int:
    import json
    return int(json.loads((out.tables / "mlp_2d_hstar.json").read_text(encoding="utf-8"))["h_star"])


def render_model_figures(out: Layout, store=None) -> list:
    """Figures that depend on networks: verified rebuild (weight fingerprint) of 3 networks."""
    from src.experiments import final_evaluation_2d
    from src.experiments.runner_2d_nominal import NOMINAL_STORE
    from src.visualization import plots_2d
    with quiet():
        inputs = final_evaluation_2d.figure_inputs(tables_dir=out.tables, store=store or NOMINAL_STORE)
        return plots_2d.save_mlp_figures(inputs, out.fig_mlp)


def compute_final_evaluation(out: Layout, store=None) -> None:
    """Final evaluation from the cache, when no frozen reference is available (bootstrap).

    Same sequence as ``produce_full``: the recomputed selection freeze is verified, the models
    are retrained and verified by the fingerprint of their weights, then only the test set is read.
    """
    from src.experiments import final_evaluation_2d
    from src.experiments.runner_2d_nominal import NOMINAL_STORE
    with quiet():
        status = final_evaluation_2d.main(["--store", str(store or NOMINAL_STORE),
                                           "--tables-dir", str(out.tables)])
    if status != 0:
        raise ExportError("final evaluation: failure")


def model_figure_names() -> list[tuple[str, str]]:
    return [(f.folder, name) for f in registry.FIGURES if f.needs_models for name in f.files()]


def produce(out: Layout, reference: Layout = REPO, with_models: bool = False, store=None) -> dict:
    """Write all the artefacts into ``out``; ``reference`` provides what is not regenerated."""
    info = {"carried_tables": carry_over(out, reference, registry.SOURCES_CARRIED)}
    if info["carried_tables"] and len(info["carried_tables"]) < len(registry.SOURCES_CARRIED):
        raise ExportError("incomplete final evaluation tables in the reference: "
                          f"{sorted(set(registry.SOURCES_CARRIED) - set(info['carried_tables']))}")
    regenerate_1d(out)
    info["h_star"] = regenerate_mlp_tables(out, store)
    if not info["carried_tables"]:  # bootstrap: no reference, final evaluation computed
        compute_final_evaluation(out, store)
    if with_models:
        render_model_figures(out, store)
        info["model_figures"] = "regenerated (3 networks rebuilt, fingerprints verified)"
    else:
        for folder, name in model_figure_names():
            source = reference.figure_dir(folder) / name
            if source.exists():
                _copy(source, out.figure_dir(folder) / name)
        info["model_figures"] = "carried over from the reference (not regenerated)"
    src = registry.Sources(out.tables)
    build.build_tables(out, src)
    build.build_figures_manifest(out, src)
    build.build_manifest(out, src)
    return info


# --------------------------------------------------------------------------
# Full reproduction (explicit, isolated)
# --------------------------------------------------------------------------
# Records, tables and freezes contain run durations: they are not
# reproducible. Only the scientific fields are compared.
TIMING_PREFIX = "elapsed"
VOLATILE_CSV = ("mlp_2d_runs.csv", "mlp_2d_config_stats.csv", "mlp_2d_adam_grid.csv",
                "mlp_2d_solver_comparison.csv")
VOLATILE_JSON = ("mlp_2d_selection.json", "mlp_2d_final_evaluation.json")
HASH_DERIVED = ("data/processed/export_manifest.json", "results/tables/export_summary.csv")


def record_digest(record: dict) -> dict:
    """Scientific content of a record: weights, metrics, convergence; without the duration."""
    diag = {k: v for k, v in record["diagnostics"].items() if not k.startswith(TIMING_PREFIX)}
    return {"run_key": record["run_key"], "weights_sha256": record["weights_sha256"],
            "training_metrics": record["training_metrics"],
            "validation_metrics": record["validation_metrics"], "diagnostics": diag,
            "stop_cause": record["stop_cause"], "status": record["status"]}


def compare_stores(store_a, store_b) -> list[str]:
    """Scientific differences between two run stores (read-only)."""
    from src.experiments.runner_2d_nominal import load_record
    files_a = {p.name for p in Path(store_a).glob("*.json")}
    files_b = {p.name for p in Path(store_b).glob("*.json")}
    issues = [f"missing from the first store: {n}" for n in sorted(files_b - files_a)]
    issues += [f"missing from the second store: {n}" for n in sorted(files_a - files_b)]
    for name in sorted(files_a & files_b):
        if record_digest(load_record(Path(store_a) / name)) != record_digest(load_record(Path(store_b) / name)):
            issues.append(f"different scientific content: {name}")
    return issues


def scientific_selection(selection: dict) -> dict:
    """Freeze without its record fingerprints (which include the durations)."""
    body = {k: v for k, v in selection.items() if k not in ("payload_sha256", "records_digest")}
    body["records"] = [{"run_key": r["run_key"]} for r in selection["records"]]
    return json.loads(json.dumps(body, sort_keys=True))


def scientific_evaluation(evaluation: dict) -> dict:
    drop = ("selection_sha256", "frozen_payload_sha256", "selection_files_sha256", "environment")
    return {k: v for k, v in evaluation.items() if k not in drop}


def _csv_content(path: Path) -> list[dict]:
    from src.export.csv_io import read_csv
    return [{k: v for k, v in row.items() if not k.startswith(TIMING_PREFIX)}
            for row in read_csv(path)]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_scientific(a: Layout, b: Layout) -> dict:
    """Like ``compare`` but insensitive to durations and to the fingerprints that depend on them.

    Byte for byte: everything except the files listed above; scientific content for these
    files; manifest and summary: same paths and same fingerprints for the
    non-volatile artefacts.
    """
    fa, fb = tracked_files(a), tracked_files(b)
    keys = set(fa) | set(fb)
    volatile = {f"results/tables/{n}" for n in VOLATILE_CSV + VOLATILE_JSON} | set(HASH_DERIVED)
    different = []
    for k in sorted(keys & set(fa) & set(fb)):
        pa, pb = fa[k], fb[k]
        name = Path(k).name
        if k in HASH_DERIVED:
            continue
        if name in VOLATILE_CSV:
            same = _csv_content(pa) == _csv_content(pb)
        elif name == "mlp_2d_selection.json":
            same = scientific_selection(_json(pa)) == scientific_selection(_json(pb))
        elif name == "mlp_2d_final_evaluation.json":
            same = scientific_evaluation(_json(pa)) == scientific_evaluation(_json(pb))
        else:
            same = pa.read_bytes() == pb.read_bytes()
        if not same:
            different.append(k)
    for k in HASH_DERIVED:  # fingerprints of the non-volatile artefacts
        if k in fa and k in fb:
            ra = {x["path"]: x["sha256"] for x in _artifacts(fa[k]) if x["path"] not in volatile}
            rb = {x["path"]: x["sha256"] for x in _artifacts(fb[k]) if x["path"] not in volatile}
            if ra != rb:
                different.append(k)
    return {"missing_in_first": sorted(k for k in keys if k not in fa),
            "missing_in_second": sorted(k for k in keys if k not in fb),
            "different": sorted(different)}


def _artifacts(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return _json(path)["artifacts"]
    from src.export.csv_io import read_csv
    return read_csv(path)


def produce_full(out: Layout, reference: Layout = REPO) -> dict:
    """Rebuild the whole scientific pipeline in ``out``: 1D runners, stages B and C with
    training in an isolated store ``out/store``, final evaluation, figures, artefacts.

    Safeguard: the test set is read only after checking that the 190 retrained runs
    and the recomputed selection freeze are scientifically identical to those of the
    reference. Otherwise ``ExportError``: the test set is not read.
    """
    from src.experiments import final_evaluation_2d
    from src.experiments.runner_2d_nominal import NOMINAL_STORE
    store = out.root / "store"
    regenerate_1d(out)
    info = {"h_star": regenerate_mlp_tables(out, store, train=True)}
    issues = compare_stores(store, NOMINAL_STORE)
    ref_freeze = _json(reference.tables / "mlp_2d_selection.json")
    if scientific_selection(_json(out.tables / "mlp_2d_selection.json")) != scientific_selection(ref_freeze):
        issues.append("rebuilt selection freeze differs from the reference freeze")
    if issues:
        raise ExportError("diverging reproduction before any read of the test set: "
                          + "; ".join(issues[:3]) + (f" (+{len(issues) - 3})" if len(issues) > 3 else ""))
    info["runs_compared"] = len(list(store.glob("*.json")))
    with quiet():
        if final_evaluation_2d.main(["--store", str(store), "--tables-dir", str(out.tables)]) != 0:
            raise ExportError("final evaluation: failure")
    render_model_figures(out, store)
    info["model_figures"] = "regenerated (networks rebuilt from the isolated store)"
    src = registry.Sources(out.tables)
    build.build_tables(out, src)
    build.build_figures_manifest(out, src)
    build.build_manifest(out, src)
    return info


# --------------------------------------------------------------------------
# Comparison and synchronisation
# --------------------------------------------------------------------------
def tracked_files(layout: Layout) -> dict[str, Path]:
    """Exported files of a layout, by relative path (excluding .gitkeep)."""
    files = {}
    for directory in (layout.tables, layout.fig_poly, layout.fig_mlp):
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file() and path.name != ".gitkeep":
                    files[path.relative_to(layout.root).as_posix()] = path
    for path in (layout.figures_manifest, layout.manifest):
        if path.exists():
            files[path.relative_to(layout.root).as_posix()] = path
    return files


def source_paths() -> set[str]:
    names = registry.SOURCES_1D + registry.SOURCES_MLP_CACHE + registry.SOURCES_CARRIED
    return {f"results/tables/{n}" for n in names}


def compare(a: Layout, b: Layout, only=None, exclude=()) -> dict:
    """Differences between two layouts: missing from ``a``, missing from ``b``, different contents."""
    fa, fb = tracked_files(a), tracked_files(b)
    keys = set(fa) | set(fb)
    if only is not None:
        keys &= set(only)
    keys -= set(exclude)
    return {"missing_in_first": sorted(k for k in keys if k not in fa),
            "missing_in_second": sorted(k for k in keys if k not in fb),
            "different": sorted(k for k in keys if k in fa and k in fb
                                and fa[k].read_bytes() != fb[k].read_bytes())}


def is_clean(diff: dict) -> bool:
    return not any(diff.values())


def sync(src: Layout, dst: Layout, exclude=()) -> list[str]:
    """Copy to ``dst`` the files of ``src`` that are missing or different; return the paths."""
    changed = []
    for rel, path in sorted(tracked_files(src).items()):
        if rel in exclude:
            continue
        target = dst.root / rel
        if not target.exists() or target.read_bytes() != path.read_bytes():
            _copy(path, target)
            changed.append(rel)
    return changed
