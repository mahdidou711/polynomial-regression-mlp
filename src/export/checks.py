"""Controles en lecture seule des artefacts exportes (etape 8).

Chaque controle renvoie un ``Check`` (statut ``passed``, ``failed``, ``warning`` ou
``not run``) ; aucun ne modifie un fichier, n'entraine un reseau ni ne lit la base de test.
Les controles LaTeX sont statiques : ils analysent les sources sans les compiler.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.export import registry
from src.export.build import artifact_list, sha256_file
from src.export.latex_tables import read_csv
from src.export.layout import Layout

PASSED, FAILED, WARNING, NOT_RUN = "passed", "failed", "warning", "not run"
RULES = (r"\toprule", r"\midrule", r"\bottomrule")
SKIPPED_LINES = (r"\toprule", r"\midrule", r"\bottomrule", r"\endfirsthead", r"\endhead",
                 r"\endlastfoot")
NUMBER = re.compile(r"^-?\d+(\.\d+)?(e[+-]\d+)?$")
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
# Analyse statique d'un fragment .tex
# --------------------------------------------------------------------------
def count_columns(spec: str) -> int:
    """Nombre de colonnes d'une specification (l r c S[...] p{...}, separateurs ignores)."""
    count, i = 0, 0
    while i < len(spec):
        ch = spec[i]
        if ch in "lrcSX":
            count += 1
            i += 1
            if ch == "S" and i < len(spec) and spec[i] == "[":
                i = spec.index("]", i) + 1
        elif ch in "pmb":
            count += 1
            i = _skip_braces(spec, i + 1)
        elif ch == "@" or ch == "!" or ch == ">" or ch == "<":
            i = _skip_braces(spec, i + 1)
        else:
            i += 1
    return count


def _skip_braces(text, i) -> int:
    if i >= len(text) or text[i] != "{":
        return i
    depth = 0
    while i < len(text):
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
        if depth == 0:
            return i
    raise ValueError("accolades non fermees")


def _balanced(text: str) -> bool:
    depth = 0
    for match in re.finditer(r"(?<!\\)[{}]", text):
        depth += 1 if match.group() == "{" else -1
        if depth < 0:
            return False
    return depth == 0


def _split_cells(row: str) -> list[str]:
    cells, depth, current = [], 0, []
    i = 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):
            current.append(row[i:i + 2])
            i += 2
            continue
        depth += (ch == "{") - (ch == "}")
        if ch == "&" and depth == 0:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    cells.append("".join(current).strip())
    return cells


def check_tex_text(text: str) -> list[str]:
    """Anomalies d'un fragment ``\\input{}`` de table ; liste vide si conforme."""
    issues = []
    if not text.isascii():
        issues.append("caractere non ASCII")
    if "$" in text.replace(r"\$", ""):
        issues.append("signe $ (mathematiques en \\( \\) uniquement)")
    for banned in (r"\hline", r"\input", r"\include", r"\caption", r"\label", r"\sisetup",
                   r"\usepackage", "/home/", "C:\\"):
        if banned in text:
            issues.append(f"commande ou chemin interdit : {banned}")
    if not _balanced(text):
        issues.append("accolades non equilibrees")
    if text.count(r"\(") != text.count(r"\)"):
        issues.append("delimiteurs \\( \\) desequilibres")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("%")]
    if not lines:
        return issues + ["fragment vide"]
    envs = re.findall(r"\\begin\{(tabular|longtable)\}", text)
    ends = re.findall(r"\\end\{(tabular|longtable)\}", text)
    if len(envs) != 1 or envs != ends:
        return issues + [f"un seul environnement tabular/longtable attendu : {envs} / {ends}"]
    for rule in RULES:
        if rule not in text:
            issues.append(f"regle booktabs absente : {rule}")
    first = lines[0]
    start = first.index("{" + envs[0] + "}") + len(envs[0]) + 2
    try:
        spec_end = _skip_braces(first, start)
    except ValueError:
        return issues + ["specification de colonnes illisible"]
    spec = first[start + 1:spec_end - 1]
    n_cols = count_columns(spec)
    s_cols = [j for j, cell in enumerate(re.findall(r"S\[[^\]]*\]|[lrcpmbX](?:\{[^}]*\})?", spec))
              if cell.startswith("S[")]
    n_rows = 0
    for ln in lines[1:]:
        if ln.startswith(SKIPPED_LINES) or ln.startswith("\\end{"):
            continue
        if not ln.endswith(r"\\"):
            issues.append(f"ligne sans fin de ligne \\\\ : {ln[:60]}")
            continue
        cells = _split_cells(ln[:-2])
        n_rows += 1
        if len(cells) != n_cols:
            issues.append(f"{len(cells)} cellules pour {n_cols} colonnes : {ln[:60]}")
            continue
        if any(c == "" for c in cells):
            issues.append(f"cellule vide : {ln[:60]}")
        if n_rows > 1 or not ln.lstrip().startswith("{"):  # lignes de donnees
            for j in s_cols:
                cell = cells[j]
                if not (NUMBER.match(cell) or (cell.startswith("{") and cell.endswith("}"))):
                    issues.append(f"cellule S non numerique et non protegee : {cell[:30]}")
    if n_rows < 2:
        issues.append("aucune ligne de donnees")
    return issues


def check_tex_files(layout: Layout) -> Check:
    issues = []
    for spec in registry.TABLES:
        path = layout.tables / f"{spec.stem}.tex"
        if not path.exists():
            continue  # signale par check_files
        issues += [f"{path.name} : {m}" for m in check_tex_text(path.read_text(encoding="utf-8"))]
        rows = read_csv(layout.tables / f"{spec.stem}.csv") if (layout.tables / f"{spec.stem}.csv").exists() else []
        n_data = sum(1 for ln in path.read_text(encoding="utf-8").splitlines()
                     if ln.rstrip().endswith(r"\\") and not ln.startswith("{") and ln.count("&") > 0)
        if rows and n_data != len(rows):
            issues.append(f"{path.name} : {n_data} lignes de donnees pour {len(rows)} dans le CSV")
    return Check.from_issues("tables .tex : analyse statique (aucune compilation)", issues)


def check_preview(layout: Layout) -> Check:
    issues = []
    path = layout.preview
    if not path.exists():
        return Check("apercu Overleaf", FAILED, [f"absent : {path.name}"])
    text = path.read_text(encoding="utf-8")
    if not text.isascii():
        issues.append("caractere non ASCII")
    if "$" in text:
        issues.append("signe $")
    if not _balanced(text):
        issues.append("accolades non equilibrees")
    inputs = re.findall(r"\\input\{\\tablesroot/([a-z0-9_]+)\.tex\}", text)
    if inputs != [t.stem for t in registry.TABLES]:
        issues.append("les \\input de l'apercu ne suivent pas le registre")
    if re.search(r"\\input\{(?!\\tablesroot/)", text) or "/home/" in text:
        issues.append("chemin non relatif a \\tablesroot")
    for package in ("booktabs", "siunitx", "longtable"):
        if f"\\usepackage{{{package}}}" not in text:
            issues.append(f"paquet manquant : {package}")
    blocks = dict(zip(inputs, re.split(r"\\subsection\*", text)[1:]))
    for stem, block in blocks.items():  # un longtable ne peut pas etre place dans une boite
        tex = layout.tables / f"{stem}.tex"
        if tex.exists() and r"\begin{longtable}" in tex.read_text(encoding="utf-8") \
                and "adjustbox" in block:
            issues.append(f"{stem} : longtable dans une boite adjustbox (erreur LaTeX)")
    if text.count(r"\begin{document}") != 1 or text.count(r"\end{document}") != 1:
        issues.append("document mal forme")
    return Check.from_issues("apercu Overleaf : source statique", issues)


# --------------------------------------------------------------------------
# Nomenclature, presence, orphelins
# --------------------------------------------------------------------------
def check_files(layout: Layout) -> Check:
    """Chaque artefact du registre existe ; aucun fichier non declare ; noms conformes."""
    issues = []
    for spec in registry.TABLES:
        if not STEM.match(spec.stem):
            issues.append(f"identifiant de table invalide : {spec.stem}")
        for ext in ("csv", "tex"):
            if not (layout.tables / f"{spec.stem}.{ext}").exists():
                issues.append(f"manquant : results/tables/{spec.stem}.{ext}")
    for stem in registry.CSV_ONLY:
        if not (layout.tables / f"{stem}.csv").exists():
            issues.append(f"manquant : results/tables/{stem}.csv")
    for name in registry.JSON_FILES:
        if not (layout.tables / name).exists():
            issues.append(f"manquant : results/tables/{name}")
    present = {p.name for p in layout.tables.iterdir() if p.is_file() and p.name != ".gitkeep"}
    for name in sorted(present - registry.expected_table_files()):
        issues.append(f"fichier non declare dans results/tables/ : {name}")
    if not (layout.tables / registry.SUMMARY_FILE).exists():
        issues.append(f"manquant : results/tables/{registry.SUMMARY_FILE}")
    for fig in registry.FIGURES:
        if not STEM.match(fig.stem):
            issues.append(f"identifiant de figure invalide : {fig.stem}")
        for name in fig.files():
            if not (layout.figure_dir(fig.folder) / name).exists():
                issues.append(f"manquant : results/figures/{fig.folder}/{name}")
    for folder in ("polynomial", "mlp"):
        expected = {name for f in registry.FIGURES if f.folder == folder for name in f.files()}
        directory = layout.figure_dir(folder)
        present = {p.name for p in directory.iterdir() if p.is_file() and p.name != ".gitkeep"}
        for name in sorted(present - expected):
            issues.append(f"figure non declaree : results/figures/{folder}/{name}")
    if not layout.figures_manifest.exists():
        issues.append("manquant : results/figures/figures_manifest.json")
    return Check.from_issues("nomenclature : presence et fichiers non declares", issues)


def check_figure_formats(layout: Layout) -> Check:
    """2D : PDF et PNG ; 3D : PNG seul ; PNG de 300 DPI ; en-tetes de fichier valides."""
    issues = []
    for fig in registry.FIGURES:
        directory = layout.figure_dir(fig.folder)
        for name in fig.files():
            path = directory / name
            if not path.exists():
                continue
            head = path.read_bytes()[:8]
            if name.endswith(".pdf") and not head.startswith(b"%PDF-"):
                issues.append(f"{name} : en-tete PDF invalide")
            if name.endswith(".png"):
                if head != b"\x89PNG\r\n\x1a\n":
                    issues.append(f"{name} : en-tete PNG invalide")
                else:
                    data = path.read_bytes()
                    i = data.find(b"pHYs")
                    ppm = int.from_bytes(data[i + 4:i + 8], "big") if i > 0 else 0
                    if abs(ppm * 0.0254 - 300) > 1:
                        issues.append(f"{name} : resolution {ppm * 0.0254:.0f} DPI au lieu de 300")
        if fig.formats == ("png",) and (directory / f"{fig.stem}.pdf").exists():
            issues.append(f"{fig.stem} : un PDF existe pour une figure 3D (PNG seul attendu)")
    return Check.from_issues("figures : formats et resolution", issues)


# --------------------------------------------------------------------------
# Manifeste, hygiene, evaluation finale
# --------------------------------------------------------------------------
def check_manifest(layout: Layout) -> Check:
    if not layout.manifest.exists():
        return Check("manifeste : empreintes", FAILED, [f"absent : {layout.manifest.name}"])
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    issues = []
    listed = {a["path"]: a for a in manifest["artefacts"]}
    current = {a["path"]: a for a in artifact_list(layout)}
    for path in sorted(set(current) - set(listed)):
        issues.append(f"artefact absent du manifeste : {path}")
    for path in sorted(set(listed) - set(current)):
        issues.append(f"artefact du manifeste introuvable : {path}")
    for path in sorted(set(listed) & set(current)):
        if listed[path]["sha256"] != current[path]["sha256"]:
            issues.append(f"empreinte differente : {path}")
    gel = layout.tables / "mlp_2d_selection.json"
    if manifest["protocole"]["gel_selection_sha256"] != sha256_file(gel):
        issues.append("empreinte du gel differente de celle du manifeste")
    return Check.from_issues("manifeste : empreintes de tous les artefacts", issues)


def check_hygiene(layout: Layout) -> Check:
    """AE7-06 : fichiers transitoires du bac a sable (signales, jamais supprimes)."""
    found = []
    for top in ("results", "data", "src", "tests"):
        base = layout.root / top
        if base.exists():
            found += [p for p in base.rglob(".mcp.json")] + [p for p in base.rglob(".claude")]
    issues = [f"{p.relative_to(layout.root)} (ignore par git ; suppression sur autorisation)"
              for p in sorted(found)]
    return Check.from_issues("hygiene : fichiers transitoires (AE7-06)", issues, warn=True)


def _close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)


def check_final_evaluation(layout: Layout) -> Check:
    """Coherence de l'evaluation finale sans la rejouer : gel, empreintes, statistiques."""
    from src.experiments.runner_2d_nominal import canonical, sha256_text
    issues = []
    tables = layout.tables
    selection = json.loads((tables / "mlp_2d_selection.json").read_text(encoding="utf-8"))
    evaluation = json.loads((tables / "mlp_2d_final_evaluation.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in selection.items() if k != "payload_sha256"}
    if sha256_text(canonical(body)) != selection["payload_sha256"]:
        issues.append("gel : payload_sha256 interne invalide")
    if evaluation["selection_sha256"] != sha256_file(tables / "mlp_2d_selection.json"):
        issues.append("evaluation finale associee a un autre gel")
    if evaluation["frozen_payload_sha256"] != selection["payload_sha256"]:
        issues.append("evaluation finale : empreinte interne du gel differente")
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
            issues.append(f"{row['label']} : {len(values)} graines au lieu de 10")
            continue
        recomputed = {"mean": values.mean(), "std": values.std(ddof=1), "median": np.median(values),
                      "min": values.min(), "max": values.max()}
        for key, value in recomputed.items():
            if not _close(row[key], value):
                issues.append(f"{row['label']} {row['metric']} {key} : table {row[key]} "
                              f"contre recalcul {value!r}")
        if row["metric"] == "mse":
            cid = row["config_id"]
            if not _close(comparison[cid]["test_mse_mean"], row["mean"]):
                issues.append(f"{row['label']} : EQMT moyen different de mlp_2d_solver_comparison")
            if not _close(comparison[cid]["val_mse_mean"], config_stats[cid]["val_mse_mean"]):
                issues.append(f"{row['label']} : EQM de validation different de mlp_2d_config_stats")
            if not _close(evaluation["configs"][cid]["test"]["mse"]["mean"], row["mean"]):
                issues.append(f"{row['label']} : EQMT moyen different de l'evaluation JSON")
    if evaluation["n_test"] != 1600:
        issues.append(f"n_test = {evaluation['n_test']} au lieu de 1600")
    return Check.from_issues("evaluation finale : coherence interne et gel (sans relecture du test)",
                             issues)


STATIC_CHECKS = (check_files, check_figure_formats, check_tex_files, check_preview,
                 check_manifest, check_hygiene, check_final_evaluation)


def run_static_checks(layout: Layout) -> list[Check]:
    return [check(layout) for check in STATIC_CHECKS]
