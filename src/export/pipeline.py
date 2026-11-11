"""Production et comparaison des artefacts exportes (etape 8).

``produce`` ecrit tous les artefacts dans une disposition (``Layout``) donnee, jamais ailleurs :

* tables 1D : recalcul par les runners 1D (6 s + 9 s, deterministes) ;
* tables MLP 2D des lots B et C : reconstruction depuis le cache de 190 runs, **sans aucun
  entrainement** (un controle prealable du cache refuse de continuer s'il est incomplet) ;
* figures MLP issues des campagnes : depuis ce meme cache ;
* figures dependant d'un reseau (``needs_models``) : uniquement avec ``with_models=True``,
  par la reconstruction verifiee (empreinte des poids) des trois reseaux des figures ; sinon
  les fichiers existants sont reportes tels quels ;
* tables de l'evaluation finale : reportees telles quelles (protegees par le gel, DEC-20).

``compare`` compare deux dispositions octet a octet. ``export`` et ``verify`` (voir ``main.py``)
sont des compositions de ``produce`` dans un repertoire temporaire.
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
    """Pre-condition non satisfaite ou derive detectee : rien n'est ecrit dans le depot."""


@contextlib.contextmanager
def quiet():
    """Silence les journaux et les impressions des runners appeles en processus."""
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
    """Copie depuis la reference les fichiers non regeneres ; renvoie ceux effectivement copies."""
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
    threads = nom.check_threads()  # RuntimeError explicite si plus d'un thread BLAS (D7-04)
    xt, yt, xv, yv = nom.load_official_data()
    fingerprint = nom.data_fingerprint(xt, yt, xv, yv)
    return nom, fingerprint, nom.environment_identity(threads)


def require_complete_cache(store, h_star=None) -> dict:
    """Controle en lecture seule : le cache couvre tout le plan ; sinon ``ExportError``."""
    from src.experiments import runner_2d_parametric as par
    nom, fingerprint, environment = _mlp_context()
    plan = nom.nominal_plan() if h_star is None else par.full_plan(h_star)
    keys = set(nom.plan_keys(plan, fingerprint, environment))
    allowed = nom.contractual_keys(fingerprint, environment) if h_star is None else keys
    report = nom.verify_store(store, keys, allowed)
    if not report["ok"]:
        raise ExportError("cache des runs MLP incomplet ou non conforme (aucun entrainement ne "
                          f"sera lance) : {report['anomalies'][:3]} ; manquants : "
                          f"{len(report['missing'])}")
    return report


def regenerate_mlp_tables(out: Layout, store=None, train: bool = False) -> int:
    """Tables des lots B et C et figures de campagne.

    ``train=False`` (defaut) : depuis le cache, sans aucun entrainement (un cache incomplet
    leve ``ExportError``). ``train=True`` : reproduction integrale, reservee a ``produce_full`` ;
    ``store`` doit alors etre un depot isole (jamais le depot nominal versionne).
    """
    from src.experiments import runner_2d_nominal as nom
    from src.experiments import runner_2d_parametric as par
    from src.visualization import plots_2d
    store = Path(store or nom.NOMINAL_STORE)
    if train:
        if nom.is_nominal_store(store):
            raise ExportError("reproduction integrale refusee dans le depot nominal versionne")
    else:
        require_complete_cache(store)
    argv = ["--store", str(store), "--tables-dir", str(out.tables)]
    with quiet():
        if nom.main(argv) != 0:
            raise ExportError("lot B : echec")
    h_star = build_hstar(out)
    if not train:
        require_complete_cache(store, h_star)
    with quiet():
        if par.main(argv) != 0:
            raise ExportError("lot C : echec")
        _, fingerprint, environment = _mlp_context()
        selection, collected = par.compute_selection(store, fingerprint, environment)
        plots_2d.save_campaign_figures(collected, selection["h_star"]["h_star"], out.fig_mlp)
    return h_star


def build_hstar(out: Layout) -> int:
    import json
    return int(json.loads((out.tables / "mlp_2d_hstar.json").read_text(encoding="utf-8"))["h_star"])


def render_model_figures(out: Layout, store=None) -> list:
    """Figures dependant des reseaux : reconstruction verifiee (empreinte des poids) de 3 reseaux."""
    from src.experiments import final_evaluation_2d
    from src.experiments.runner_2d_nominal import NOMINAL_STORE
    from src.visualization import plots_2d
    with quiet():
        inputs = final_evaluation_2d.figure_inputs(tables_dir=out.tables, store=store or NOMINAL_STORE)
        return plots_2d.save_mlp_figures(inputs, out.fig_mlp)


def model_figure_names() -> list[tuple[str, str]]:
    return [(f.folder, name) for f in registry.FIGURES if f.needs_models for name in f.files()]


def produce(out: Layout, reference: Layout = REPO, with_models: bool = False, store=None) -> dict:
    """Ecrit tous les artefacts dans ``out`` ; ``reference`` fournit ce qui n'est pas regenere."""
    info = {"carried_tables": carry_over(out, reference, registry.SOURCES_CARRIED)}
    regenerate_1d(out)
    info["h_star"] = regenerate_mlp_tables(out, store)
    if with_models:
        render_model_figures(out, store)
        info["model_figures"] = "regenerees (3 reseaux reconstruits, empreintes verifiees)"
    else:
        for folder, name in model_figure_names():
            source = reference.figure_dir(folder) / name
            if source.exists():
                _copy(source, out.figure_dir(folder) / name)
        info["model_figures"] = "reportees depuis la reference (non regenerees)"
    src = registry.Sources(out.tables)
    build.build_tables(out, src)
    build.build_figures_manifest(out, src)
    build.build_preview(out)
    build.build_manifest(out, src)
    return info


# --------------------------------------------------------------------------
# Reproduction integrale (explicite, isolee)
# --------------------------------------------------------------------------
# Les enregistrements, tables et gels contiennent des durees d'execution : elles ne se
# reproduisent pas. Seuls les champs scientifiques sont compares.
TIMING_PREFIX = "elapsed"
VOLATILE_CSV = ("mlp_2d_runs.csv", "mlp_2d_config_stats.csv", "mlp_2d_adam_grid.csv",
                "mlp_2d_solver_comparison.csv")
VOLATILE_JSON = ("mlp_2d_selection.json", "mlp_2d_final_evaluation.json")
HASH_DERIVED = ("data/processed/export_manifest.json", "results/tables/export_summary.csv")


def record_digest(record: dict) -> dict:
    """Contenu scientifique d'un enregistrement : poids, metriques, convergence ; sans la duree."""
    diag = {k: v for k, v in record["diagnostics"].items() if not k.startswith(TIMING_PREFIX)}
    return {"run_key": record["run_key"], "weights_sha256": record["weights_sha256"],
            "training_metrics": record["training_metrics"],
            "validation_metrics": record["validation_metrics"], "diagnostics": diag,
            "stop_cause": record["stop_cause"], "status": record["status"]}


def compare_stores(store_a, store_b) -> list[str]:
    """Differences scientifiques entre deux depots de runs (lecture seule)."""
    from src.experiments.runner_2d_nominal import load_record
    files_a = {p.name for p in Path(store_a).glob("*.json")}
    files_b = {p.name for p in Path(store_b).glob("*.json")}
    issues = [f"absent du premier depot : {n}" for n in sorted(files_b - files_a)]
    issues += [f"absent du second depot : {n}" for n in sorted(files_a - files_b)]
    for name in sorted(files_a & files_b):
        if record_digest(load_record(Path(store_a) / name)) != record_digest(load_record(Path(store_b) / name)):
            issues.append(f"contenu scientifique different : {name}")
    return issues


def scientific_selection(selection: dict) -> dict:
    """Gel sans ses empreintes d'enregistrements (qui incluent les durees)."""
    body = {k: v for k, v in selection.items() if k not in ("payload_sha256", "records_digest")}
    body["records"] = [{"run_key": r["run_key"]} for r in selection["records"]]
    return json.loads(json.dumps(body, sort_keys=True))


def scientific_evaluation(evaluation: dict) -> dict:
    drop = ("selection_sha256", "frozen_payload_sha256", "selection_files_sha256", "environment")
    return {k: v for k, v in evaluation.items() if k not in drop}


def _csv_content(path: Path) -> list[dict]:
    from src.export.latex_tables import read_csv
    return [{k: v for k, v in row.items() if not k.startswith(TIMING_PREFIX)}
            for row in read_csv(path)]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_scientific(a: Layout, b: Layout) -> dict:
    """Comme ``compare`` mais insensible aux durees et aux empreintes qui en dependent.

    Octet a octet : tout sauf les fichiers listes ci-dessus ; contenu scientifique pour ces
    fichiers ; manifeste et recapitulatif : memes chemins et memes empreintes pour les
    artefacts non volatils.
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
    for k in HASH_DERIVED:  # empreintes des artefacts non volatils
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
        return _json(path)["artefacts"]
    from src.export.latex_tables import read_csv
    return read_csv(path)


def produce_full(out: Layout, reference: Layout = REPO) -> dict:
    """Reconstruit tout le pipeline scientifique dans ``out`` : runners 1D, lots B et C avec
    entrainement dans un depot isole ``out/store``, evaluation finale, figures, artefacts.

    Garde-fou DEC-20 / R3 : la base de test n'est lue qu'apres avoir constate que les 190 runs
    reentraines et le gel de selection recalcule sont scientifiquement identiques a ceux de la
    reference validee (point de controle P2). Sinon ``ExportError`` : le test n'est pas lu.
    """
    from src.experiments import final_evaluation_2d
    from src.experiments.runner_2d_nominal import NOMINAL_STORE
    store = out.root / "store"
    regenerate_1d(out)
    info = {"h_star": regenerate_mlp_tables(out, store, train=True)}
    issues = compare_stores(store, NOMINAL_STORE)
    ref_freeze = _json(reference.tables / "mlp_2d_selection.json")
    if scientific_selection(_json(out.tables / "mlp_2d_selection.json")) != scientific_selection(ref_freeze):
        issues.append("gel de selection reconstruit different du gel valide")
    if issues:
        raise ExportError("reproduction divergente avant toute lecture du test : "
                          + "; ".join(issues[:3]) + (f" (+{len(issues) - 3})" if len(issues) > 3 else ""))
    info["runs_compared"] = len(list(store.glob("*.json")))
    with quiet():
        if final_evaluation_2d.main(["--store", str(store), "--tables-dir", str(out.tables)]) != 0:
            raise ExportError("evaluation finale : echec")
    render_model_figures(out, store)
    info["model_figures"] = "regenerees (reseaux reconstruits du depot isole)"
    src = registry.Sources(out.tables)
    build.build_tables(out, src)
    build.build_figures_manifest(out, src)
    build.build_preview(out)
    build.build_manifest(out, src)
    return info


# --------------------------------------------------------------------------
# Comparaison et synchronisation
# --------------------------------------------------------------------------
def tracked_files(layout: Layout) -> dict[str, Path]:
    """Fichiers exportes d'une disposition, par chemin relatif (hors .gitkeep)."""
    files = {}
    for directory in (layout.tables, layout.fig_poly, layout.fig_mlp):
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file() and path.name != ".gitkeep":
                    files[path.relative_to(layout.root).as_posix()] = path
    for path in (layout.figures_manifest, layout.preview, layout.manifest):
        if path.exists():
            files[path.relative_to(layout.root).as_posix()] = path
    return files


def source_paths() -> set[str]:
    names = registry.SOURCES_1D + registry.SOURCES_MLP_CACHE + registry.SOURCES_CARRIED
    return {f"results/tables/{n}" for n in names}


def compare(a: Layout, b: Layout, only=None, exclude=()) -> dict:
    """Differences entre deux dispositions : absents de ``a``, absents de ``b``, contenus differents."""
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
    """Copie vers ``dst`` les fichiers de ``src`` absents ou differents ; renvoie les chemins."""
    changed = []
    for rel, path in sorted(tracked_files(src).items()):
        if rel in exclude:
            continue
        target = dst.root / rel
        if not target.exists() or target.read_bytes() != path.read_bytes():
            _copy(path, target)
            changed.append(rel)
    return changed
