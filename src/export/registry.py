"""Registre declaratif des artefacts exportes pour le rapport (etape 8).

Il fixe la nomenclature (contrat entre les etapes 8 et 9) : chaque table a un identifiant
``stem`` qui nomme ``results/tables/<stem>.csv`` et ``results/tables/<stem>.tex`` ; chaque
figure a un ``stem`` qui nomme ``<stem>.pdf`` / ``<stem>.png`` (2D) ou ``<stem>.png`` (3D)
dans ``results/figures/<polynomial|mlp>/``. Les noms des figures et des tables deja produites
par les etapes 3 a 7 sont conserves (les audits et les passations y renvoient).

Une table est *directe* (``derive is None`` : le CSV existe deja, le ``.tex`` en montre un
sous-ensemble de colonnes) ou *derivee* (une fonction calcule les lignes a partir de CSV
sources ; le CSV derive est ecrit a cote du ``.tex``, sous le meme identifiant).
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.export.latex_tables import Col, is_missing, read_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Sources:
    """Acces en lecture aux tables sources d'un repertoire de tables."""

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
    columns: tuple
    requirements: tuple = ()
    annex: bool = False
    source: str | None = None  # CSV direct (identifiant de table) ou None si derive
    derive: Callable | None = None
    inputs: tuple = ()  # identifiants des CSV lus par ``derive``
    summary: str = ""


# --------------------------------------------------------------------------
# Fonctions de derivation (lignes = dictionnaires ; aucune ecriture)
# --------------------------------------------------------------------------
def _f(value):
    return None if is_missing(value) else float(value)


def _config_label(row) -> str:
    label = f"{row['solver']}-{row['activation']}-{row['hidden_layers']}-a{float(row['alpha']):g}"
    if row["solver"] == "adam":
        label += f"-lr{float(row['learning_rate_init']):g}-b{row['batch_size']}"
    return label


def _config_rows(src: Sources, prefix: str | None = None) -> list[dict]:
    """Une ligne par configuration distincte ; ``prefix`` filtre sur le role (TOPO, REG, ...)."""
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


def derive_topologie(src):
    return _config_rows(src, "TOPO")


def derive_regularisation(src):
    return _config_rows(src, "REG")


def derive_activation(src):
    return _config_rows(src, "ACT")


def derive_solveur(src):
    return _config_rows(src, "SOLV")


def derive_configurations(src):
    return _config_rows(src)


def _exploration_synthese(src: Sources, axis: str) -> list[dict]:
    """Une ligne par valeur de ``axis`` (nb_ex ou sigma) : distribution de C*, minimum de l'EQMT."""
    stats = src.rows(f"exploration_{axis}_stats")
    cstar = src.rows(f"exploration_{axis}_cstar")
    out = []
    for value in dict.fromkeys(r[axis] for r in stats):
        block = [r for r in stats if r[axis] == value]
        c_block = [r for r in cstar if r[axis] == value]
        best = min(block, key=lambda r: (float(r["eqmt_median"]), int(r["degree"])))
        d12 = next(r for r in block if int(r["degree"]) == 12)
        out.append({axis: value, "n_runs": int(block[0]["n_runs"]),
                    "cstar_mode": int(float(c_block[0]["mode"])),
                    "cstar_median": float(c_block[0]["median"]),
                    "degree_min": int(best["degree"]), "eqmt_median_min": float(best["eqmt_median"]),
                    "eqmt_median_d12": float(d12["eqmt_median"])})
    return out


def derive_nb_ex_synthese(src):
    return _exploration_synthese(src, "nb_ex")


def derive_sigma_synthese(src):
    return _exploration_synthese(src, "sigma")


def derive_nominal_diagnostics(src):
    keep = ("degree", "rank", "kappa_h", "kappa_hth", "regime", "rel_error_normal_vs_svd",
            "rel_error_qr_vs_svd")
    return [{k: r[k] for k in keep} for r in src.rows("nominal_1d_seed42")]


def derive_final_synthese(src):
    """Une ligne par configuration evaluee : EQMT (5 statistiques) et R2 (moyenne, minimum)."""
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
    """Provenance des donnees 2D (``EXG-RAP-B3-05``) ; aucune lecture de la base de test."""
    from src.data.generator_2d import genere_base_apprentissage_2d, genere_base_validation_2d
    meta = src.json("mlp_2d_metadata.json")["data"]
    evaluation = src.json("mlp_2d_final_evaluation.json")
    return [
        {"jeu": "Fichier officiel sinc_dim2_input.csv", "n": len(genere_base_apprentissage_2d()[0]),
         "origine": "abscisses d'apprentissage fournies avec l'énoncé, ordre conservé",
         "sha256": src.raw_sha256()[:16]},
        {"jeu": "Apprentissage", "n": len(genere_base_apprentissage_2d()[0]),
         "origine": "cible sinus cardinal calculée sur les abscisses du fichier, sans bruit",
         "sha256": meta["train_sha256"][:16]},
        {"jeu": "Validation", "n": len(genere_base_validation_2d()[0]),
         "origine": "grille 20 x 20 des centres de mailles du carré unité, déterministe",
         "sha256": meta["validation_sha256"][:16]},
        {"jeu": "Test", "n": int(evaluation["n_test"]),
         "origine": "grille 40 x 40, bornes incluses, lue une seule fois après le gel (DEC-20)",
         "sha256": evaluation["test_sha256"][:16]},
    ]


# --------------------------------------------------------------------------
# Colonnes reutilisees
# --------------------------------------------------------------------------
def _val_cols(prefix="val_mse"):
    return (Col(f"{prefix}_mean", r"\(\overline{\mathrm{EQM}}\)"),
            Col(f"{prefix}_std", r"\(\sigma\)"),
            Col(f"{prefix}_min", "min."), Col(f"{prefix}_max", "max."))


_DEGREE = Col("degree", r"\(d\)", "int")
_Q = {"mediane": "m\\'ediane", "q1": r"\(Q_1\)", "q3": r"\(Q_3\)"}
_KAPPA = r"\(\kappa(H^{\top}H)\)"
_SENS_TAIL = (Col("val_r2_mean", r"\(\overline{R^2}\)", "fixed", 4),
              Col("n_warnings", "avert.", "int"), Col("stop_max_iter", "plafond", "int"),
              Col("n_iter_mean", "it\\'er.", "fixed", 0))

TABLES = (
    # --- 1D nominal -------------------------------------------------------
    TableSpec("nominal_1d_seed42", (
        _DEGREE, Col("eqm_train", "EQM"), Col("eqm_test", "EQMT"),
        Col("kappa_hth", _KAPPA), Col("regime", "r\\'egime", "text"),
        Col("is_optimal", r"\(C^{*}\)", "bool")),
        requirements=("EXG-RAP-B2-02",), source="nominal_1d_seed42",
        summary="Profil d'erreur nominal (graine 42), degrés 1 à 12"),
    TableSpec("nominal_1d_seed42_diagnostics", (
        _DEGREE, Col("rank", "rang", "int"), Col("kappa_h", r"\(\kappa(H)\)"),
        Col("kappa_hth", _KAPPA), Col("regime", "r\\'egime", "text"),
        Col("rel_error_normal_vs_svd", "\\(\\varepsilon_{\\mathrm{norm}}\\)"),
        Col("rel_error_qr_vs_svd", r"\(\varepsilon_{\mathrm{QR}}\)")),
        requirements=("EXG-RAP-B1-02",), annex=True, derive=derive_nominal_diagnostics,
        inputs=("nominal_1d_seed42",),
        summary="Diagnostics de conditionnement et écarts OLS/QR/SVD (graine 42)"),
    TableSpec("nominal_1d_stats_50_graines", (
        _DEGREE, Col("eqm_median", "EQM " + _Q["mediane"]), Col("eqm_q1", _Q["q1"]),
        Col("eqm_q3", _Q["q3"]), Col("eqmt_median", "EQMT " + _Q["mediane"]),
        Col("eqmt_q1", _Q["q1"]), Col("eqmt_q3", _Q["q3"])),
        requirements=("EXG-RAP-B2-02", "EXG-RAP-B2-04"), source="nominal_1d_stats_50_graines",
        summary="Statistiques descriptives sur 50 graines (médiane, quartiles)"),
    TableSpec("nominal_1d_cstar_50_graines", (
        _DEGREE, Col("count_cstar", "effectif de \\(C^{*}\\)", "int")),
        requirements=("EXG-RAP-B2-02",), source="nominal_1d_cstar_50_graines",
        summary="Distribution de C* sur 50 graines"),
    # --- explorations 1D --------------------------------------------------
    TableSpec("exploration_nb_ex_synthese", (
        Col("nb_ex", r"\(N\)", "int"), Col("n_runs", "r\\'ep.", "int"),
        Col("cstar_mode", r"mode de \(C^{*}\)", "int"),
        Col("cstar_median", "m\\'ed. de \\(C^{*}\\)", "fixed", 1),
        Col("degree_min", r"\(d\) du min.", "int"),
        Col("eqmt_median_min", "EQMT m\\'ed. min."),
        Col("eqmt_median_d12", "EQMT m\\'ed. \\(d=12\\)")),
        requirements=("EXG-RAP-B2-04",), derive=derive_nb_ex_synthese,
        inputs=("exploration_nb_ex_stats", "exploration_nb_ex_cstar"),
        summary="Exploration 1 (taille de l'échantillon N) : synthèse par N"),
    TableSpec("exploration_sigma_synthese", (
        Col("sigma", r"\(\sigma\)", "fixed", 2), Col("n_runs", "r\\'ep.", "int"),
        Col("cstar_mode", r"mode de \(C^{*}\)", "int"),
        Col("cstar_median", "m\\'ed. de \\(C^{*}\\)", "fixed", 1),
        Col("degree_min", r"\(d\) du min.", "int"),
        Col("eqmt_median_min", "EQMT m\\'ed. min."),
        Col("eqmt_median_d12", "EQMT m\\'ed. \\(d=12\\)")),
        requirements=("EXG-RAP-B2-04",), derive=derive_sigma_synthese,
        inputs=("exploration_sigma_stats", "exploration_sigma_cstar"),
        summary="Exploration 2 (bruit sigma) : synthèse par sigma"),
    TableSpec("exploration_extrapolation_stats", (
        Col("role", "mod\\`ele", "text"), Col("n_runs", "r\\'ep.", "int"),
        Col("eqm_interp_median", "interp. " + _Q["mediane"]), Col("eqm_interp_q1", _Q["q1"]),
        Col("eqm_interp_q3", _Q["q3"]), Col("eqm_extrap_median", "extrap. " + _Q["mediane"]),
        Col("eqm_extrap_q1", _Q["q1"]), Col("eqm_extrap_q3", _Q["q3"])),
        requirements=("EXG-RAP-B2-04",), source="exploration_extrapolation_stats",
        summary="Exploration 3 (extrapolation) : EQM par domaine"),
    TableSpec("exploration_nb_ex_stats", (
        Col("nb_ex", r"\(N\)", "int"), _DEGREE, Col("eqm_median", "EQM m\\'ed."),
        Col("eqmt_median", "EQMT m\\'ed."), Col("eqmt_q1", _Q["q1"]), Col("eqmt_q3", _Q["q3"])),
        requirements=("EXG-RAP-B2-04",), annex=True, source="exploration_nb_ex_stats",
        summary="Exploration 1 : statistiques complètes par N et par degré"),
    TableSpec("exploration_sigma_stats", (
        Col("sigma", r"\(\sigma\)", "fixed", 2), _DEGREE, Col("eqm_median", "EQM m\\'ed."),
        Col("eqmt_median", "EQMT m\\'ed."), Col("eqmt_q1", _Q["q1"]), Col("eqmt_q3", _Q["q3"])),
        requirements=("EXG-RAP-B2-04",), annex=True, source="exploration_sigma_stats",
        summary="Exploration 2 : statistiques complètes par sigma et par degré"),
    TableSpec("exploration_nb_ex_diagnostics", (
        Col("nb_ex", r"\(N\)", "int"), _DEGREE, Col("rank", "rang", "int"),
        Col("kappa_hth", _KAPPA), Col("regime", "r\\'egime", "text"),
        Col("normal_rel_error", r"\(\varepsilon_{\mathrm{norm}}\)"),
        Col("qr_rel_error", r"\(\varepsilon_{\mathrm{QR}}\)")),
        requirements=("EXG-RAP-B1-02",), annex=True, source="exploration_nb_ex_diagnostics",
        summary="Diagnostics de conditionnement par N et par degré"),
    TableSpec("exploration_nb_ex_variance", (
        Col("nb_ex", r"\(N\)", "int"), _DEGREE,
        Col("mean_prediction_variance", "variance moy."),
        Col("n_times_variance", r"\(N \times\) variance")),
        requirements=("EXG-RAP-B2-04",), annex=True, source="exploration_nb_ex_variance",
        summary="Variance de prédiction moyenne par N (échelle en 1/N)"),
    # --- MLP 2D -----------------------------------------------------------
    TableSpec("mlp_2d_nominal_stats", (
        Col("width", r"\(H\)", "int"),
        Col("val_mse_mean", r"\(\overline{\mathrm{EQM}}\)"), Col("val_mse_std", r"\(\sigma\)"),
        Col("val_mse_median", "m\\'ed."), Col("val_mse_min", "min."), Col("val_mse_max", "max."),
        Col("val_r2_mean", r"\(\overline{R^2}\)", "fixed", 4),
        Col("n_warnings", "avert.", "int")),
        requirements=("EXG-RAP-B3-04",), source="mlp_2d_nominal_stats",
        summary="EXP-2D-NOM : EQM de validation par largeur H, 10 graines"),
    TableSpec("mlp_2d_sensibilite_topologie", (
        Col("hidden_layers", "couches", "text"), *_val_cols(), *_SENS_TAIL),
        requirements=("EXG-RAP-B3-04",), derive=derive_topologie, inputs=("mlp_2d_config_stats",),
        summary="EXP-2D-TOPO : profondeur contre largeur"),
    TableSpec("mlp_2d_sensibilite_regularisation", (
        Col("hidden_layers", "couches", "text"), Col("alpha", r"\(\alpha\)", "sci", 1),
        *_val_cols(), *_SENS_TAIL),
        requirements=("EXG-RAP-B3-04",), derive=derive_regularisation,
        inputs=("mlp_2d_config_stats",), summary="EXP-2D-REG : régularisation L2"),
    TableSpec("mlp_2d_sensibilite_activation", (
        Col("activation", "activation", "text"), Col("hidden_layers", "couches", "text"),
        *_val_cols(), *_SENS_TAIL),
        requirements=("EXG-RAP-B3-03",), derive=derive_activation,
        inputs=("mlp_2d_config_stats",), summary="EXP-2D-ACT : tanh contre relu"),
    TableSpec("mlp_2d_sensibilite_solveur", (
        Col("solver", "solveur", "text"), Col("learning_rate_init", r"\(\eta\)", "sci", 1),
        Col("batch_size", "lot", "text"), *_val_cols(), *_SENS_TAIL),
        requirements=("EXG-RAP-B3-04",), derive=derive_solveur, inputs=("mlp_2d_config_stats",),
        summary="EXP-2D-SOLV : lbfgs contre adam (budget et grille donnés)"),
    TableSpec("mlp_2d_adam_grid", (
        Col("learning_rate_init", r"\(\eta\)", "sci", 1), Col("batch_size", "lot", "int"),
        Col("rank_validation", "rang", "int"), Col("selected", "retenu", "bool"),
        *_val_cols(), Col("n_warnings", "avert.", "int"), Col("stop_max_iter", "plafond", "int"),
        Col("n_iter_mean", "it\\'er.", "fixed", 0)),
        requirements=("EXG-RAP-B3-04",), source="mlp_2d_adam_grid",
        summary="Grille adam (DEC-19), ordre contractuel de la grille"),
    TableSpec("mlp_2d_solver_comparison", (
        Col("label", "configuration", "code"), Col("val_mse_mean", "EQM val."),
        Col("test_mse_mean", "EQMT moy."), Col("test_mse_std", r"\(\sigma\)"),
        Col("test_mse_min", "min."), Col("test_mse_max", "max."),
        Col("n_iter_mean", "it\\'er.", "fixed", 0), Col("n_warnings", "avert.", "int"),
        Col("stop_max_iter", "plafond", "int")),
        requirements=("EXG-RAP-B3-04",), source="mlp_2d_solver_comparison",
        summary="Comparaison lbfgs / adam sur le test (10 graines)"),
    TableSpec("mlp_2d_final_synthese", (
        Col("roles", "r\\^ole", "text"), Col("label", "configuration", "code"),
        Col("mse_mean", "EQMT moy."), Col("mse_std", r"\(\sigma\)"),
        Col("mse_median", "m\\'ed."), Col("mse_min", "min."), Col("mse_max", "max."),
        Col("r2_mean", r"\(\overline{R^2}\)", "fixed", 5), Col("r2_min", r"\(R^2\) min.", "fixed", 4)),
        requirements=("EXG-RAP-B3-02", "EXG-RAP-B3-04"), derive=derive_final_synthese,
        inputs=("mlp_2d_final_test_stats",),
        summary="Évaluation finale sur le test (N = 1600), 10 graines par configuration"),
    TableSpec("mlp_2d_final_test_stats", (
        Col("roles", "r\\^ole", "text"), Col("metric", "m\\'etrique", "text"),
        Col("mean", "moyenne"), Col("std", r"\(\sigma\)"), Col("median", "m\\'ediane"),
        Col("min", "min."), Col("max", "max.")),
        requirements=("EXG-RAP-B3-02",), annex=True, source="mlp_2d_final_test_stats",
        summary="Évaluation finale : toutes les métriques"),
    TableSpec("mlp_2d_final_test_runs", (
        Col("roles", "r\\^ole", "text"), Col("seed", "graine", "int"),
        Col("test_mse", "EQMT"), Col("test_rmse", "REQMT"),
        Col("test_r2", r"\(R^2\)", "fixed", 5), Col("test_max_abs_error", r"\(\max|e|\)"),
        Col("is_figure_network", "figures", "bool")),
        requirements=("EXG-RAP-B3-04",), annex=True, source="mlp_2d_final_test_runs",
        summary="Évaluation finale : résultat de chaque graine"),
    TableSpec("mlp_2d_configurations", (
        Col("label", "configuration", "code"), Col("val_mse_mean", "EQM val. moy."),
        Col("val_mse_std", r"\(\sigma\)"), Col("train_mse_mean", "EQM appr. moy."),
        Col("val_r2_mean", r"\(\overline{R^2}\)", "fixed", 4),
        Col("n_warnings", "avert.", "int"), Col("n_iter_mean", "it\\'er.", "fixed", 0)),
        requirements=("EXG-RAP-B3-04",), annex=True, derive=derive_configurations,
        inputs=("mlp_2d_config_stats",),
        summary="Toutes les configurations distinctes du plan (validation)"),
    TableSpec("donnees_2d_provenance", (
        Col("jeu", "jeu de donn\\'ees", "text"), Col("n", r"\(N\)", "int"),
        Col("origine", "origine", "text"), Col("sha256", "empreinte (16 car.)", "code")),
        requirements=("EXG-RAP-B3-05",), derive=derive_provenance,
        inputs=("mlp_2d_metadata.json", "mlp_2d_final_evaluation.json"),
        summary="Provenance et empreintes des données 2D"),
)

# Fichiers de results/tables/ sans table LaTeX (volumineux ou techniques) ; leur role est
# documente ici pour que ``verify`` signale tout fichier non declare.
CSV_ONLY = {
    "exploration_extrapolation_runs": "résultats bruts par graine (200 lignes)",
    "exploration_nb_ex_cstar": "distribution de C* par N (résumée dans la synthèse)",
    "exploration_nb_ex_runs": "résultats bruts par graine (3600 lignes)",
    "exploration_sigma_cstar": "distribution de C* par sigma (résumée dans la synthèse)",
    "exploration_sigma_runs": "résultats bruts par graine (3012 lignes)",
    "mlp_2d_plan": "plan des 210 réalisations (traçabilité)",
    "mlp_2d_runs": "190 entraînements distincts, une ligne par run (traçabilité)",
    "mlp_2d_config_stats": "source des tables de sensibilité (colonnes larges)",
}
JSON_FILES = ("exploration_1d_metadata.json", "mlp_2d_final_evaluation.json",
              "mlp_2d_hstar.json", "mlp_2d_metadata.json", "mlp_2d_selection.json")
SUMMARY_FILE = "export_summary.csv"

# Origine de chaque fichier source : regenere par les runners 1D, reconstruit depuis le cache
# MLP (aucun entrainement), ou conserve tel quel (evaluation finale, protegee par le gel).
SOURCES_1D = ("nominal_1d_seed42.csv", "nominal_1d_stats_50_graines.csv",
              "nominal_1d_cstar_50_graines.csv", "exploration_extrapolation_runs.csv",
              "exploration_extrapolation_stats.csv", "exploration_nb_ex_cstar.csv",
              "exploration_nb_ex_diagnostics.csv", "exploration_nb_ex_runs.csv",
              "exploration_nb_ex_stats.csv", "exploration_nb_ex_variance.csv",
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
    folder: str  # "polynomial" ou "mlp"
    formats: tuple  # ("pdf", "png") ou ("png",)
    requirements: tuple
    tables: tuple = ()
    needs_models: bool = False  # exige la reconstruction de reseaux verifies (3 au plus)
    description: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def files(self) -> list[str]:
        return [f"{self.stem}.{ext}" for ext in self.formats]


_BOTH, _PNG = ("pdf", "png"), ("png",)
FIGURES = (
    FigureSpec("nominal_1d_error_profile", "polynomial", _BOTH, ("EXG-RAP-B2-02",),
               ("nominal_1d_seed42",), description="EQM et EQMT en fonction du degré, C* repéré"),
    FigureSpec("nominal_1d_fits", "polynomial", _BOTH, ("EXG-RAP-B2-01",),
               ("nominal_1d_seed42",), description="planche des 12 polynômes ajustés (panneaux coupés signalés)"),
    FigureSpec("nominal_1d_parity", "polynomial", _BOTH, ("EXG-RAP-B2-03",),
               ("nominal_1d_seed42",), description="parité sur le test pour d = 1, C* et 12 (repère 1:1)"),
    FigureSpec("nominal_1d_conditioning", "polynomial", _BOTH, ("EXG-RAP-B1-02",),
               ("nominal_1d_seed42_diagnostics",), description="conditionnement de la matrice de Vandermonde"),
    FigureSpec("nominal_1d_campaign_50_graines", "polynomial", _BOTH,
               ("EXG-RAP-B2-02", "EXG-RAP-B2-04"),
               ("nominal_1d_stats_50_graines", "nominal_1d_cstar_50_graines"),
               description="distribution de C* et EQM/EQMT médianes sur 50 graines"),
    FigureSpec("exploration_nb_ex", "polynomial", _BOTH, ("EXG-RAP-B2-04",),
               ("exploration_nb_ex_synthese", "exploration_nb_ex_stats"),
               description="exploration 1 : effet de N"),
    FigureSpec("exploration_nb_ex_seed42", "polynomial", _BOTH, ("EXG-RAP-B2-04",),
               description="exploration 1 : réalisation illustrative, graine 42"),
    FigureSpec("exploration_sigma", "polynomial", _BOTH, ("EXG-RAP-B2-04",),
               ("exploration_sigma_synthese", "exploration_sigma_stats"),
               description="exploration 2 : effet du bruit sigma"),
    FigureSpec("exploration_sigma_seed42", "polynomial", _BOTH, ("EXG-RAP-B2-04",),
               description="exploration 2 : réalisation illustrative, graine 42 (AE4-05)"),
    FigureSpec("exploration_extrapolation", "polynomial", _BOTH, ("EXG-RAP-B2-04",),
               ("exploration_extrapolation_stats",),
               description="exploration 3 : extrapolation sur [-0,3 ; 1,5]"),
    FigureSpec("mlp_2d_largeur", "mlp", _BOTH, ("EXG-RAP-B3-04",), ("mlp_2d_nominal_stats",),
               description="EQM de validation par largeur H, H* repéré"),
    FigureSpec("mlp_2d_sensibilites", "mlp", _BOTH, ("EXG-RAP-B3-04",),
               ("mlp_2d_configurations",), description="EQM de validation de toutes les configurations"),
    FigureSpec("mlp_2d_convergence", "mlp", _BOTH, ("EXG-RAP-B3-04",),
               description="itérations et causes d'arrêt par configuration"),
    FigureSpec("mlp_2d_adam_courbes", "mlp", _BOTH, ("EXG-RAP-B3-04",), ("mlp_2d_adam_grid",),
               description="courbes de perte adam par réglage de la grille"),
    FigureSpec("mlp_2d_parite", "mlp", _BOTH, ("EXG-RAP-B3-02",), ("mlp_2d_final_synthese",),
               needs_models=True, description="parité 2D (apprentissage et test), réseau des figures"),
    FigureSpec("mlp_2d_residus_carte", "mlp", _BOTH, ("EXG-RAP-B3-01",), needs_models=True,
               description="carte du résidu absolu |y_pred - y|"),
    FigureSpec("mlp_2d_planche_3d", "mlp", _PNG, ("EXG-RAP-B3-01",), needs_models=True,
               description="planche : surface exacte, prédite, résidu absolu"),
    FigureSpec("mlp_2d_surface_exacte", "mlp", _PNG, ("EXG-RAP-B3-01",), needs_models=True,
               description="surface exacte"),
    FigureSpec("mlp_2d_surface_predite", "mlp", _PNG, ("EXG-RAP-B3-01",), needs_models=True,
               description="surface prédite par le réseau des figures"),
    FigureSpec("mlp_2d_surface_residus", "mlp", _PNG, ("EXG-RAP-B3-01",), needs_models=True,
               description="résidu absolu en 3D"),
    FigureSpec("mlp_2d_activation_tanh_relu", "mlp", _PNG, ("EXG-RAP-B3-03",),
               ("mlp_2d_sensibilite_activation",), needs_models=True,
               description="surfaces prédites tanh et relu"),
)


def figure_by_stem() -> dict:
    return {f.stem: f for f in FIGURES}


def expected_table_files() -> set[str]:
    """Noms de fichiers attendus dans results/tables/ (CSV, TeX, JSON, aperçu, recapitulatif)."""
    names = {f"{t.stem}.csv" for t in TABLES} | {f"{t.stem}.tex" for t in TABLES}
    names |= {f"{stem}.csv" for stem in CSV_ONLY} | set(JSON_FILES) | {SUMMARY_FILE}
    return names
