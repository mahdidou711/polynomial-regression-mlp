"""Nominal 1D experiment: degrees 1 to 12, MSE / test MSE, optimal degree.

Protocol:
- training: n_samples = 20 points on [0; 1.2), Gaussian noise sigma = 0.2, seed 42;
- test: 200 points on [0; 1.2), noise-free;
- for each degree d = 1..12, fit with ``LinearRegression(fit_intercept=False)`` on
  the uncentred Vandermonde matrix (SVD solver), MSE(d) on the training set,
  test MSE(d) on the test set;
- optimal degree C* = argmin_d test MSE(d), the smallest degree on ties. The optimal
  degree is a measurement, never an expectation.

Statistical limit: C* is chosen on the test set; test MSE(C*) is therefore an
optimistic estimate of generalisation, not an independent estimate. A single-seed
result (42) is one realisation; the distribution of C* over the 50 seeds
``SEEDS_STAT_1D`` (nominal-profile campaign) is reported separately by
``run_nominal_campaign``.

``run_degree_sweep`` is independent of the nominal parameters (data sets passed as
arguments) so that it can serve the explorations. Command:
``python -m src.experiments.runner_1d_nominal`` (tables and figures in ``results/``).
"""

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import generate_training_set, generate_test_set
from src.models import compare_solvers, predict, solve_sklearn_reference, vandermonde
from src.utils.logger import environment_info, get_logger, log_event
from src.utils.metrics import getMSE
from src.utils.seeding import SEED_POLYNOMIAL, SEEDS_STAT_1D

DEGREES = tuple(range(1, 13))  # degrees of the study: 1 to 12
N_TEST_DENSE = 1000  # points of the dense grid used to draw the curves (plots_1d)

_logger = logging.getLogger("polynomial_mlp.runner_1d_nominal")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "polynomial"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"


def optimal_degree(degrees, test_mse) -> int:
    """C* = argmin test MSE; the smallest degree on an exact tie (``np.argmin``)."""
    degrees = np.asarray(degrees)
    test_mse = np.asarray(test_mse, dtype=float)
    if degrees.shape != test_mse.shape or degrees.ndim != 1 or degrees.size == 0:
        raise ValueError(
            f"degrees and test MSE: 1D vectors of the same length expected, "
            f"{degrees.shape} / {test_mse.shape}"
        )
    if not np.isfinite(test_mse).all():
        raise ValueError("test MSE: non-finite values")
    return int(degrees[int(np.argmin(test_mse))])


@dataclass(frozen=True)
class SweepResult:
    """Result of a degree sweep on a (training, test) pair.

    The diagnostics (``rank`` ... ``rel_error_*``) are arrays of length
    ``len(degrees)`` if the sweep computed them (``diagnose=True``), otherwise ``None``.
    ``rel_error_normal`` / ``rel_error_qr``: relative gap (infinity norm) to
    the SVD, ``nan`` if the method is not applicable; ``regime``: 'strict' or
    'diagnostic'.
    """

    degrees: np.ndarray
    mse_train: np.ndarray
    mse_test: np.ndarray
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
        return optimal_degree(self.degrees, self.mse_test)

    def index_of(self, degree: int) -> int:
        return int(np.flatnonzero(self.degrees == degree)[0])


def run_degree_sweep(x_train, y_train, x_test, y_test, degrees=DEGREES, diagnose=True,
                     seed=None) -> SweepResult:
    """Fit each degree with ``LinearRegression(fit_intercept=False)`` and measure MSE / test MSE.

    ``diagnose=True``: calls ``compare_solvers`` (rank, kappa, regime, gaps of the
    normal equations and of QR to the SVD; logged event) and takes the
    coefficients of the scikit-learn solution it returns. ``diagnose=False``: fits
    directly with ``solve_sklearn_reference`` (campaigns, no repeated diagnostics).
    """
    degrees = np.asarray(list(degrees), dtype=int)
    x_tr = np.asarray(x_train, dtype=float)
    x_te = np.asarray(x_test, dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    y_te = np.asarray(y_test, dtype=float)

    coefs, mse, test_mse, preds = [], [], [], []
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
        mse.append(getMSE(y_tr, y_hat_train))
        test_mse.append(getMSE(y_te, y_hat_test))
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
        mse_train=np.array(mse),
        mse_test=np.array(test_mse),
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
    """Nominal experiment (training 20 points, sigma = 0.2; test 200 points)."""
    x_train, y_train = generate_training_set(seed=seed)
    x_test, y_test = generate_test_set()
    result = run_degree_sweep(x_train, y_train, x_test, y_test, degrees, diagnose=True, seed=seed)
    log_event(
        _logger,
        "nominal_done",
        seed=seed,
        optimal_degree=result.optimal_degree,
        test_mse_min=float(result.mse_test.min()),
    )
    return result


@dataclass(frozen=True)
class CampaignResult:
    """Campaign over several noise seeds (same nominal configuration).

    ``mse_train`` and ``mse_test``: (n_seeds, n_degrees). ``optimal_degrees``: C* per
    seed. The statistics are descriptive, computed over these realisations.
    """

    seeds: tuple
    degrees: np.ndarray
    mse_train: np.ndarray
    mse_test: np.ndarray
    optimal_degrees: np.ndarray

    def cstar_counts(self) -> dict:
        """Count of C* per degree (all degrees, including those never optimal)."""
        return {int(d): int(np.sum(self.optimal_degrees == d)) for d in self.degrees}

    @property
    def cstar_mode(self) -> int:
        """Mode of C*; the smallest degree on ties."""
        counts = self.cstar_counts()
        return min(counts, key=lambda d: (-counts[d], d))

    @property
    def cstar_median(self) -> float:
        return float(np.median(self.optimal_degrees))

    def summary(self, which: str) -> dict:
        """Mean, median, quartiles per degree of the MSE ('train') or of the test MSE ('test')."""
        data = {"train": self.mse_train, "test": self.mse_test}[which]
        q1, med, q3 = np.percentile(data, [25, 50, 75], axis=0)
        return {"mean": data.mean(axis=0), "median": med, "q1": q1, "q3": q3,
                "iqr": q3 - q1}


def run_nominal_campaign(seeds=SEEDS_STAT_1D, degrees=DEGREES) -> CampaignResult:
    """Repeat the nominal experiment for each noise seed of ``seeds``.

    Only the noise seed changes: the training grid and the test set are
    identical from one seed to the next (numerical diagnostics unchanged, hence not
    recomputed here).
    """
    seeds = tuple(int(s) for s in seeds)
    x_test, y_test = generate_test_set()
    mse, test_mse, cstar = [], [], []
    for s in seeds:
        x_train, y_train = generate_training_set(seed=s)
        r = run_degree_sweep(x_train, y_train, x_test, y_test, degrees, diagnose=False, seed=s)
        mse.append(r.mse_train)
        test_mse.append(r.mse_test)
        cstar.append(r.optimal_degree)
    result = CampaignResult(seeds, np.asarray(list(degrees), dtype=int), np.array(mse),
                            np.array(test_mse), np.array(cstar))
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
    """Per-degree table: MSE, test MSE, rank, kappa, regime, solver gaps, optimal degree.

    Values are written as computed (no display floor).
    """
    star = result.optimal_degree
    rows = [
        (int(d), result.mse_train[i], result.mse_test[i], int(result.rank[i]),
         result.kappa_h[i], result.kappa_hth[i], result.regime[i],
         result.rel_error_normal[i], result.rel_error_qr[i], int(d) == star)
        for i, d in enumerate(result.degrees)
    ]
    header = ("degree", "mse_train", "mse_test", "rank", "kappa_h", "kappa_hth", "regime",
              "rel_error_normal_vs_svd", "rel_error_qr_vs_svd", "is_optimal")
    return _write_csv(path, header, rows)


def write_campaign_tables(result: CampaignResult, path_stats, path_counts) -> tuple[Path, Path]:
    """Two tables: statistics per degree and count of C*."""
    train, test = result.summary("train"), result.summary("test")
    rows = [
        (int(d), train["mean"][i], train["median"][i], train["q1"][i], train["q3"][i],
         test["mean"][i], test["median"][i], test["q1"][i], test["q3"][i])
        for i, d in enumerate(result.degrees)
    ]
    header = ("degree", "mse_mean", "mse_median", "mse_q1", "mse_q3",
              "test_mse_mean", "test_mse_median", "test_mse_q1", "test_mse_q3")
    a = _write_csv(path_stats, header, rows)
    b = _write_csv(path_counts, ("degree", "count_cstar"), sorted(result.cstar_counts().items()))
    return a, b


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--log-file", type=Path, default=None, help="journal JSON Lines")
    args = parser.parse_args(argv)

    from src.visualization import plots_1d  # late import: the computation does not depend on plotting

    lg = get_logger("polynomial_mlp", log_file=args.log_file)
    log_event(lg, "run_start", seed=SEED_POLYNOMIAL, n_stat_seeds=len(SEEDS_STAT_1D),
              **environment_info())
    try:
        nominal = run_nominal()
        campaign = run_nominal_campaign()
        write_nominal_table(nominal, args.tables_dir / "nominal_1d_seed42.csv")
        write_campaign_tables(campaign, args.tables_dir / "nominal_1d_stats_50_seeds.csv",
                              args.tables_dir / "nominal_1d_cstar_50_seeds.csv")
        paths = plots_1d.save_nominal_figures(nominal, campaign, args.figures_dir)
    except Exception:
        log_event(lg, "run_failed", level=logging.ERROR, exc_info=True)
        raise
    else:
        log_event(lg, "run_done", optimal_degree=nominal.optimal_degree,
                  figures=[str(p) for p in paths])
    finally:  # closes the log file (no descriptor left open)
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
            handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
