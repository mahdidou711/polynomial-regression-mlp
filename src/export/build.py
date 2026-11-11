"""Construction des artefacts derives : tables CSV/TeX, metadonnees de figures, apercu, manifeste.

Aucune fonction ne lance un entrainement ni ne lit la base de test : les entrees sont les
tables sources et les JSON de selection / d'evaluation deja versionnes (etapes 3 a 7).
Toutes les sorties sont deterministes (aucun horodatage, aucun SHA Git, ordre fixe).
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

from src.export import registry
from src.export.latex_tables import REQUIRED_PACKAGES, escape_tex, read_csv, render_table, write_csv
from src.export.layout import Layout

MANIFEST_SCHEMA = "c0-export/1"
KNOWN_LIMITS = (
    "Reproductibilit\u00e9 \u00e9tablie sur un seul h\u00f4te et un seul thread BLAS (audit 07, \u00a712), "
    "non entre plateformes.",
    "Les journaux d'ex\u00e9cution ne sont pas versionn\u00e9s : l'absence de lecture du test avant le gel "
    "(AE7-05) repose sur la d\u00e9claration, neutralis\u00e9e par la reconstruction de la s\u00e9lection depuis "
    "la seule validation.",
    "Les bandes et statistiques sur 10 ou 50 graines sont descriptives, non des intervalles de "
    "confiance (R2).",
    "Les tables de l'\u00e9valuation finale (mlp_2d_final_*, mlp_2d_solver_comparison, "
    "mlp_2d_final_evaluation.json) ne sont pas r\u00e9g\u00e9n\u00e9r\u00e9es par l'export ; leur coh\u00e9rence interne et "
    "avec le gel est contr\u00f4l\u00e9e par `main.py verify`.",
)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_text(path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def write_json(path, obj) -> Path:
    return write_text(path, json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False,
                                       allow_nan=False) + "\n")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def table_rows(spec: registry.TableSpec, src: registry.Sources) -> list[dict]:
    return src.rows(spec.source) if spec.derive is None else spec.derive(src)


def build_tables(layout: Layout, src: registry.Sources | None = None) -> list[Path]:
    """Ecrit les CSV derives et tous les ``.tex`` du registre dans ``layout.tables``."""
    src = src or registry.Sources(layout.tables)
    written = []
    for spec in registry.TABLES:
        rows = table_rows(spec, src)
        if not rows:
            raise ValueError(f"table vide : {spec.stem}")
        if spec.derive is not None:
            header = list(rows[0])
            written.append(write_csv(layout.tables / f"{spec.stem}.csv", header, rows))
            origin = "derive de " + ", ".join(spec.inputs)
        else:
            origin = f"{spec.source}.csv"
        tex = render_table(spec.columns, rows, origin)
        written.append(write_text(layout.tables / f"{spec.stem}.tex", tex))
    return written


# --------------------------------------------------------------------------
# Metadonnees de figures (AE4-05, AE7-08)
# --------------------------------------------------------------------------
def figure_parameters(src: registry.Sources) -> dict:
    """Parametres a citer dans les legendes de figure : lus dans les fichiers sources."""
    from src.data.generator_1d import NB_EX_NOMINAL, SIGMA_NOMINAL
    meta1d = src.json("exploration_1d_metadata.json")
    meta2d = src.json("mlp_2d_metadata.json")
    selection = src.json("mlp_2d_selection.json")
    evaluation = src.json("mlp_2d_final_evaluation.json")
    stat_seeds = meta1d["statistical_seeds"]
    n_runs_sigma = {r["sigma"]: int(r["n_runs"]) for r in src.rows("exploration_sigma_stats")
                    if int(r["degree"]) == 1}
    n_runs_nb_ex = {r["nb_ex"]: int(r["n_runs"]) for r in src.rows("exploration_nb_ex_stats")
                    if int(r["degree"]) == 1}
    nominal_1d = {"graine_illustrative": meta1d["illustration_seed"], "nb_ex": NB_EX_NOMINAL,
                  "sigma_apprentissage": SIGMA_NOMINAL, "sigma_test": 0.0,
                  "domaine_apprentissage": meta1d["train_domain"],
                  "borne_haute_apprentissage": meta1d["train_endpoint"],
                  "n_test": meta1d["test_points"], "degres": [min(meta1d["degrees"]), max(meta1d["degrees"])]}
    campaign = {"graines": f"{min(stat_seeds)} \u00e0 {max(stat_seeds)}", "n_graines": len(stat_seeds)}
    adam_rows = src.rows("mlp_2d_adam_grid")
    grid_order = [f"lr={float(a):g}, lot={b}" for a, b in meta2d["adam_grid"]]
    by_setting = {(float(r["learning_rate_init"]), int(r["batch_size"])): int(r["n_warnings"])
                  for r in adam_rows}
    adam_warnings = [by_setting[(float(a), int(b))] for a, b in meta2d["adam_grid"]]
    final = selection["final"]["ranking"][0]
    mlp_common = {"graines_monte_carlo": meta2d["seeds"], "n_realisations_par_configuration": len(meta2d["seeds"]),
                  "h_star": meta2d["h_star"], "configuration_finale": final["label"],
                  "graine_reseau_des_figures": selection["final"]["figure_network"]["seed"],
                  "n_test": evaluation["n_test"]}
    render = {"grille_de_rendu": "200 x 200, pas 0,005, borne 1 exclue (np.arange(0, 1, 0.005))",
              "vue_3d": {"elev": 30, "azim": -60}}
    return {
        "nominal_1d": {**nominal_1d, "fenetre_planche": [-2.0, 2.0]},
        "campagne_1d": {**nominal_1d, **campaign,
                        "repetitions_bruitees": len(stat_seeds)},
        "exploration_nb_ex": {"sigma_fixe": SIGMA_NOMINAL, "valeurs_N": meta1d["sizes"],
                              **campaign, "repetitions_par_N": n_runs_nb_ex},
        "exploration_sigma": {"nb_ex_fixe": NB_EX_NOMINAL, "valeurs_sigma": meta1d["sigmas"], **campaign,
                              "repetitions_par_sigma": n_runs_sigma,
                              "note": "sigma = 0 : une seule r\u00e9alisation (les tirages seraient identiques)"},
        "exploration_extrapolation": {"domaine_etendu": meta1d["extended_domain"],
                                      "points_de_rendu": meta1d["extended_points"], **campaign,
                                      "selection": meta1d["selection"]},
        "mlp_campagnes": {**mlp_common,
                          "ordre_grille_adam": grid_order,
                          "avertissements_adam_dans_l_ordre_de_la_grille": adam_warnings,
                          "budget_adam": meta2d["adam_protocol"], "budget_lbfgs": meta2d["lbfgs_budget"]},
        "mlp_modele": {**mlp_common, **render},
    }


_PARAMS_OF = {
    "nominal_1d_error_profile": "nominal_1d", "nominal_1d_fits": "nominal_1d",
    "nominal_1d_parity": "nominal_1d", "nominal_1d_conditioning": "nominal_1d",
    "nominal_1d_campaign_50_graines": "campagne_1d",
    "exploration_nb_ex": "exploration_nb_ex", "exploration_nb_ex_seed42": "nominal_1d",
    "exploration_sigma": "exploration_sigma", "exploration_sigma_seed42": "nominal_1d",
    "exploration_extrapolation": "exploration_extrapolation",
    "mlp_2d_largeur": "mlp_campagnes", "mlp_2d_sensibilites": "mlp_campagnes",
    "mlp_2d_convergence": "mlp_campagnes", "mlp_2d_adam_courbes": "mlp_campagnes",
}
_NOTES = {
    "mlp_2d_adam_courbes": "Un r\u00e9glage par couleur ; ordre de la grille adam : (1e-3, 32), (1e-3, 200), "
                           "(1e-2, 32), (1e-2, 200) (AE7-08).",
    "mlp_2d_planche_3d": "Grille de rendu : la borne 1 est exclue (AE7-08). Le r\u00e9seau trac\u00e9 est un cas parmi 10.",
    "mlp_2d_residus_carte": "Grille de rendu : la borne 1 est exclue (AE7-08). Le r\u00e9seau trac\u00e9 est un cas parmi 10.",
    "mlp_2d_surface_predite": "Le r\u00e9seau trac\u00e9 est un cas parmi 10, non repr\u00e9sentatif de la configuration.",
    "mlp_2d_parite": "Le r\u00e9seau trac\u00e9 est un cas parmi 10 ; la parit\u00e9 d'apprentissage et la parit\u00e9 de test "
                     "sont trac\u00e9es pour ce r\u00e9seau.",
    "nominal_1d_fits": "Panneaux dont le polyn\u00f4me sort de la fen\u00eatre verticale : signal\u00e9s \u00ab coup\u00e9 \u00bb (AE3-08).",
    "nominal_1d_parity": "Fen\u00eatre commune fix\u00e9e par la cible ; les points hors cadre sont compt\u00e9s sur le panneau (AE3-08).",
}


def build_figures_manifest(layout: Layout, src: registry.Sources | None = None) -> Path:
    src = src or registry.Sources(layout.tables)
    params = figure_parameters(src)
    figures = []
    for spec in registry.FIGURES:
        key = _PARAMS_OF.get(spec.stem, "mlp_modele" if spec.folder == "mlp" else "nominal_1d")
        figures.append({
            "stem": spec.stem, "dossier": f"results/figures/{spec.folder}",
            "fichiers": spec.files(), "exigences": list(spec.requirements),
            "tables": list(spec.tables), "description": spec.description,
            "parametres": params[key], "note": _NOTES.get(spec.stem, "")})
    return write_json(layout.figures_manifest, {"schema": "c0-figures/1", "figures": figures})


# --------------------------------------------------------------------------
# Apercu autonome pour Overleaf
# --------------------------------------------------------------------------
PREVIEW_HEAD = r"""% Apercu autonome des tables exportees (etape 8). Genere par src/export, ne pas editer.
% Usage : importer le depot a la racine dans Overleaf, choisir ce fichier comme document
% principal, compiler UNE fois et consulter le journal (aucune compilation locale).
% Si le projet Overleaf n'a pas le depot a sa racine, modifier uniquement \tablesroot.
\documentclass[a4paper,10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{adjustbox}
\usepackage{caption}
\usepackage{siunitx}
\sisetup{output-decimal-marker={,}}% presentation francaise ; les donnees gardent le point
\newcommand{\tablesroot}{results/tables}
\begin{document}
\section*{Aper\c{c}u des tables export\'ees}
"""


def render_preview(longtables=frozenset()) -> str:
    """Source de l'apercu ; les ``longtable`` ne sont pas places dans une boite (``adjustbox``)."""
    parts = [PREVIEW_HEAD]
    for spec in registry.TABLES:
        title = escape_tex(spec.stem)
        parts.append(f"\\subsection*{{\\texttt{{{title}}}}}\n{escape_tex(spec.summary)}.\n")
        path = f"\\input{{\\tablesroot/{spec.stem}.tex}}"
        if spec.stem in longtables:
            body = f"\\begingroup\\footnotesize\n{path}\n\\endgroup\n"
        else:
            body = (f"\\begingroup\\footnotesize\n\\begin{{adjustbox}}{{max width=\\linewidth}}\n"
                    f"{path}\n\\end{{adjustbox}}\n\\endgroup\n")
        parts.append(f"\\captionof{{table}}{{{title}}}\n{body}\\medskip\n")
    parts.append("\\end{document}\n")
    return "\n".join(parts)


def build_preview(layout: Layout) -> Path:
    long = {spec.stem for spec in registry.TABLES
            if r"\begin{longtable}" in (layout.tables / f"{spec.stem}.tex").read_text(encoding="utf-8")}
    return write_text(layout.preview, render_preview(long))


# --------------------------------------------------------------------------
# Manifeste et recapitulatif (traçabilite)
# --------------------------------------------------------------------------
def _versions() -> dict:
    return {name: metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "matplotlib")}


def artifact_list(layout: Layout) -> list[dict]:
    """Artefacts exportes (hors manifeste et recapitulatif eux-memes, pour eviter la circularite)."""
    arts = []

    def add(kind, path, source="", requirements=(), ident=None):
        path = Path(path)
        arts.append({"id": ident or path.stem, "kind": kind,
                     "path": path.relative_to(layout.root).as_posix(),
                     "sha256": sha256_file(path), "bytes": path.stat().st_size,
                     "source": source, "requirements": list(requirements)})

    for fig in registry.FIGURES:
        for name in fig.files():
            add("figure", layout.figure_dir(fig.folder) / name, "src/visualization",
                fig.requirements, ident=name)
    add("figures_manifest", layout.figures_manifest, "src/export/build.py")
    for spec in registry.TABLES:
        origin = spec.source or ("derive:" + "+".join(spec.inputs))
        add("table_csv", layout.tables / f"{spec.stem}.csv", origin, spec.requirements,
            ident=f"{spec.stem}.csv")
        add("table_tex", layout.tables / f"{spec.stem}.tex", f"{spec.stem}.csv",
            spec.requirements, ident=f"{spec.stem}.tex")
    for stem in registry.CSV_ONLY:
        add("table_csv", layout.tables / f"{stem}.csv", "runner", ident=f"{stem}.csv")
    for name in registry.JSON_FILES:
        add("table_json", layout.tables / name, "runner", ident=name)
    add("preview", layout.preview, "src/export/build.py")
    return sorted(arts, key=lambda a: a["path"])


def build_manifest(layout: Layout, src: registry.Sources | None = None) -> tuple[Path, Path]:
    """Ecrit ``data/processed/export_manifest.json`` et ``results/tables/export_summary.csv``."""
    src = src or registry.Sources(layout.tables)
    selection_file = layout.tables / "mlp_2d_selection.json"
    evaluation = src.json("mlp_2d_final_evaluation.json")
    meta2d = src.json("mlp_2d_metadata.json")
    arts = artifact_list(layout)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "environment": _versions(),
        "latex": {"paquets_requis": list(REQUIRED_PACKAGES),
                  "entree_numerique": "point decimal ; presentation francaise : "
                                      "\\sisetup{output-decimal-marker={,}}",
                  "tables": "fragments \\input{} : tabular ou longtable, sans caption ni label",
                  "figures": "PDF + PNG 300 DPI (2D) ; PNG 300 DPI (3D)"},
        "protocole": {"gel_selection_sha256": sha256_file(selection_file),
                      "gel_selection_payload_sha256": evaluation["frozen_payload_sha256"],
                      "evaluation_finale_selection_sha256": evaluation["selection_sha256"],
                      "donnees": meta2d["data"], "fichier_officiel_sha256": src.raw_sha256(),
                      "test_sha256": evaluation["test_sha256"], "n_test": evaluation["n_test"]},
        "limites": list(KNOWN_LIMITS),
        "artefacts": arts,
    }
    path = write_json(layout.manifest, manifest)
    header = ["id", "kind", "path", "sha256", "bytes", "requirements"]
    rows = [{**a, "requirements": ";".join(a["requirements"])} for a in arts]
    summary = write_csv(layout.tables / registry.SUMMARY_FILE, header, rows)
    return path, summary
