"""Experience 1D nominale : degres 1 a 12, EQM / EQMT, degre optimal (etape 3).

Protocole (enonce 2026-2027, diapositives 9 a 13 ; CDC 5.1, avenant 7.1) :
- apprentissage : NbEx = 20 points sur [0 ; 1,2[, bruit gaussien sigma = 0,2, graine 42 ;
- test : 200 points sur [0 ; 1,2[, sans bruit ;
- pour chaque degre d = 1..12, ajustement par ``LinearRegression(fit_intercept=False)`` sur
  la matrice de Vandermonde non centree (solveur SVD, ARB-08), EQM(d) sur l'apprentissage,
  EQMT(d) sur le test ;
- degre optimal C* = argmin_d EQMT(d), a egalite le plus petit degre. Le degre optimal est
  une mesure, jamais une attente.

Limite statistique (EXG-RAP-AV1-02) : C* est choisi sur la base de test (protocole de
l'enseignant) ; EQMT(C*) est donc une estimation optimiste de la generalisation, pas une
estimation independante. Un resultat a graine unique (42) est une realisation ; la
distribution de C* sur les 50 graines ``SEEDS_STAT_1D`` (campagne du profil nominal) est
rapportee separement par ``run_nominal_campaign``.

``run_degree_sweep`` est independant des parametres nominaux (jeux de donnees passes en
argument) pour servir aux explorations de l'etape 4. Commande :
``python -m src.experiments.runner_1d_nominal`` (tables et figures dans ``results/``).
"""

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import genere_base_apprentissage, genere_base_test
from src.models import compare_solvers, predict, solve_sklearn_reference, vandermonde
from src.utils.logger import environment_info, get_logger, log_event
from src.utils.metrics import getMSE
from src.utils.seeding import SEED_POLYNOMIAL, SEEDS_STAT_1D

DEGREES = tuple(range(1, 13))  # degres de l'enonce : 1 a 12
N_TEST_DENSE = 1000  # points de la grille dense de trace des courbes (plots_1d)

_logger = logging.getLogger("c0_tp2_mlp.runner_1d_nominal")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "polynomial"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def optimal_degree(degrees, eqmt) -> int:
    """C* = argmin EQMT ; le plus petit degre en cas d'egalite exacte (``np.argmin``)."""
    degrees = np.asarray(degrees)
    eqmt = np.asarray(eqmt, dtype=float)
    if degrees.shape != eqmt.shape or degrees.ndim != 1 or degrees.size == 0:
        raise ValueError(
            f"degres et EQMT : vecteurs 1D de meme longueur attendus, "
            f"{degrees.shape} / {eqmt.shape}"
        )
    if not np.isfinite(eqmt).all():
        raise ValueError("EQMT : valeurs non finies")
    return int(degrees[int(np.argmin(eqmt))])


@dataclass(frozen=True)
class SweepResult:
    """Resultat d'un balayage de degres sur un couple (apprentissage, test).

    Les diagnostics (``rank`` ... ``rel_error_*``) sont des tableaux de longueur
    ``len(degrees)`` si le balayage les a calcules (``diagnose=True``), sinon ``None``.
    ``rel_error_normal`` / ``rel_error_qr`` : ecart relatif (norme infinie, avenant 7.10) a
    la SVD, ``nan`` si la methode est non applicable ; ``regime`` : 'strict' ou
    'diagnostic' (ARB-08).
    """

    degrees: np.ndarray
    eqm_train: np.ndarray
    eqm_test: np.ndarray
    coefs: tuple
    y_pred_test: np.ndarray  # (n_degrees, n_test)
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    seed: int | None
    rank: np.ndarray | None = None
    kappa_h: np.ndarray | None = None
    kappa_hth: np.ndarray | None = None
    regime: tuple | None = None
    rel_error_normal: np.ndarray | None = None
    rel_error_qr: np.ndarray | None = None

    @property
    def optimal_degree(self) -> int:
        return optimal_degree(self.degrees, self.eqm_test)

    def index_of(self, degree: int) -> int:
        return int(np.flatnonzero(self.degrees == degree)[0])


def run_degree_sweep(x_train, y_train, x_test, y_test, degrees=DEGREES, diagnose=True,
                     seed=None) -> SweepResult:
    """Ajuste chaque degre par ``LinearRegression(fit_intercept=False)`` et mesure EQM / EQMT.

    ``diagnose=True`` : appelle ``compare_solvers`` (rang, kappa, regime, ecarts des
    equations normales et de QR a la SVD ; evenement journalise) et prend les
    coefficients de la solution scikit-learn qu'il renvoie. ``diagnose=False`` : ajuste
    directement par ``solve_sklearn_reference`` (campagnes, sans diagnostic repete).
    """
    degrees = np.asarray(list(degrees), dtype=int)
    x_tr = np.asarray(x_train, dtype=float)
    x_te = np.asarray(x_test, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    y_te = np.asarray(y_test, dtype=float)

    coefs, eqm, eqmt, preds = [], [], [], []
    diag = {k: [] for k in ("rank", "kappa_h", "kappa_hth", "regime", "normal", "qr")}
    for d in degrees:
        d = int(d)
        if diagnose:
            cmp = compare_solvers(x_tr, y_tr, d)
            coef = cmp.coefficients["sklearn"]
            diag["rank"].append(cmp.rank)
            diag["kappa_h"].append(cmp.kappa_h)
            diag["kappa_hth"].append(cmp.kappa_hth)
            diag["regime"].append(cmp.regime)
            for m in ("normal", "qr"):
                err = cmp.rel_error_inf[m]
                diag[m].append(np.nan if err is None else err)
        else:
            coef = solve_sklearn_reference(vandermonde(x_tr, d), y_tr)
        y_hat_train = predict(coef, x_tr)
        y_hat_test = predict(coef, x_te)
        coefs.append(coef)
        eqm.append(getMSE(y_tr, y_hat_train))
        eqmt.append(getMSE(y_te, y_hat_test))
        preds.append(y_hat_test)

    extra = {}
    if diagnose:
        extra = dict(
            rank=np.array(diag["rank"]),
            kappa_h=np.array(diag["kappa_h"]),
            kappa_hth=np.array(diag["kappa_hth"]),
            regime=tuple(diag["regime"]),
            rel_error_normal=np.array(diag["normal"]),
            rel_error_qr=np.array(diag["qr"]),
        )
    return SweepResult(
        degrees=degrees,
        eqm_train=np.array(eqm),
        eqm_test=np.array(eqmt),
        coefs=tuple(coefs),
        y_pred_test=np.array(preds),
        x_train=x_tr,
        y_train=y_tr,
        x_test=x_te,
        y_test=y_te,
        seed=seed,
        **extra,
    )


def run_nominal(seed: int = SEED_POLYNOMIAL, degrees=DEGREES) -> SweepResult:
    """Experience nominale (apprentissage 20 points, sigma = 0,2 ; test 200 points)."""
    x_train, y_train = genere_base_apprentissage(seed=seed)
    x_test, y_test = genere_base_test()
    result = run_degree_sweep(x_train, y_train, x_test, y_test, degrees, diagnose=True, seed=seed)
    log_event(
        _logger,
        "nominal_done",
        seed=seed,
        optimal_degree=result.optimal_degree,
        eqmt_min=float(result.eqm_test.min()),
    )
    return result


@dataclass(frozen=True)
class CampaignResult:
    """Campagne sur plusieurs graines de bruit (meme configuration nominale).

    ``eqm_train`` et ``eqm_test`` : (n_seeds, n_degrees). ``optimal_degrees`` : C* par
    graine. Les statistiques sont descriptives, calculees sur ces realisations.
    """

    seeds: tuple
    degrees: np.ndarray
    eqm_train: np.ndarray
    eqm_test: np.ndarray
    optimal_degrees: np.ndarray

    def cstar_counts(self) -> dict:
        """Comptage de C* par degre (tous les degres, y compris ceux jamais optimaux)."""
        return {int(d): int(np.sum(self.optimal_degrees == d)) for d in self.degrees}

    @property
    def cstar_mode(self) -> int:
        """Mode de C* ; a egalite le plus petit degre."""
        counts = self.cstar_counts()
        return min(counts, key=lambda d: (-counts[d], d))

    @property
    def cstar_median(self) -> float:
        return float(np.median(self.optimal_degrees))

    def summary(self, which: str) -> dict:
        """Moyenne, mediane, quartiles par degre de l'EQM ('train') ou de l'EQMT ('test')."""
        data = {"train": self.eqm_train, "test": self.eqm_test}[which]
        q1, med, q3 = np.percentile(data, [25, 50, 75], axis=0)
        return {"mean": data.mean(axis=0), "median": med, "q1": q1, "q3": q3,
                "iqr": q3 - q1}


def run_nominal_campaign(seeds=SEEDS_STAT_1D, degrees=DEGREES) -> CampaignResult:
    """Repete l'experience nominale pour chaque graine de bruit de ``seeds``.

    Seule la graine de bruit change : la grille d'apprentissage et la base de test sont
    identiques d'une graine a l'autre (diagnostics numeriques inchangeables, donc non
    recalcules ici).
    """
    seeds = tuple(int(s) for s in seeds)
    x_test, y_test = genere_base_test()
    eqm, eqmt, cstar = [], [], []
    for s in seeds:
        x_train, y_train = genere_base_apprentissage(seed=s)
        r = run_degree_sweep(x_train, y_train, x_test, y_test, degrees, diagnose=False, seed=s)
        eqm.append(r.eqm_train)
        eqmt.append(r.eqm_test)
        cstar.append(r.optimal_degree)
    result = CampaignResult(seeds, np.asarray(list(degrees), dtype=int), np.array(eqm),
                            np.array(eqmt), np.array(cstar))
    log_event(
        _logger,
        "campaign_done",
        n_seeds=len(seeds),
        cstar_counts=result.cstar_counts(),
        cstar_mode=result.cstar_mode,
        cstar_median=result.cstar_median,
    )
    return result


# --------------------------------------------------------------------------
# Tables CSV (results/tables/)
# --------------------------------------------------------------------------
def _write_csv(path, header, rows) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v
                             for v in row])
    return path


def write_nominal_table(result: SweepResult, path) -> Path:
    """Table par degre : EQM, EQMT, rang, kappa, regime, ecarts des solveurs, degre optimal.

    Les valeurs sont ecrites telles que calculees (aucun plancher d'affichage).
    """
    star = result.optimal_degree
    rows = [
        (int(d), result.eqm_train[i], result.eqm_test[i], int(result.rank[i]),
         result.kappa_h[i], result.kappa_hth[i], result.regime[i],
         result.rel_error_normal[i], result.rel_error_qr[i], int(d) == star)
        for i, d in enumerate(result.degrees)
    ]
    header = ("degree", "eqm_train", "eqm_test", "rank", "kappa_h", "kappa_hth", "regime",
              "rel_error_normal_vs_svd", "rel_error_qr_vs_svd", "is_optimal")
    return _write_csv(path, header, rows)


def write_campaign_tables(result: CampaignResult, path_stats, path_counts) -> tuple[Path, Path]:
    """Deux tables : statistiques par degre et comptage de C*."""
    train, test = result.summary("train"), result.summary("test")
    rows = [
        (int(d), train["mean"][i], train["median"][i], train["q1"][i], train["q3"][i],
         test["mean"][i], test["median"][i], test["q1"][i], test["q3"][i])
        for i, d in enumerate(result.degrees)
    ]
    header = ("degree", "eqm_mean", "eqm_median", "eqm_q1", "eqm_q3",
              "eqmt_mean", "eqmt_median", "eqmt_q1", "eqmt_q3")
    a = _write_csv(path_stats, header, rows)
    b = _write_csv(path_counts, ("degree", "count_cstar"), sorted(result.cstar_counts().items()))
    return a, b


# --------------------------------------------------------------------------
# Point d'entree
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--log-file", type=Path, default=None, help="journal JSON Lines")
    args = parser.parse_args(argv)

    from src.visualization import plots_1d  # import tardif : le calcul ne depend pas du trace

    lg = get_logger("c0_tp2_mlp", log_file=args.log_file)
    log_event(lg, "run_start", seed=SEED_POLYNOMIAL, n_stat_seeds=len(SEEDS_STAT_1D),
              **environment_info())
    try:
        nominal = run_nominal()
        campaign = run_nominal_campaign()
        write_nominal_table(nominal, args.tables_dir / "nominal_1d_seed42.csv")
        write_campaign_tables(campaign, args.tables_dir / "nominal_1d_stats_50_graines.csv",
                              args.tables_dir / "nominal_1d_cstar_50_graines.csv")
        paths = plots_1d.save_nominal_figures(nominal, campaign, args.figures_dir)
    except Exception:
        log_event(lg, "run_failed", level=logging.ERROR, exc_info=True)
        raise
    else:
        log_event(lg, "run_done", optimal_degree=nominal.optimal_degree,
                  figures=[str(p) for p in paths])
    finally:  # ferme le fichier de journal (aucun descripteur laisse ouvert)
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
            handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
