"""1D figures of the nominal experiment.

Each ``plot_*`` function returns a ``Figure`` (no pyplot); ``save_nominal_figures``
exports them as vector PDF + 300 DPI PNG through ``save_figure``. Logarithmic
scales use a display floor (``FLOOR_LOG``) applied to a copy: the stored values
are never modified.
"""

from pathlib import Path

import numpy as np

from src.data import XMAX_NOMINAL, XMIN_NOMINAL, target_function
from src.experiments.runner_1d_nominal import CampaignResult, SweepResult
from src.models import predict
from src.utils.metrics import KAPPA_HTH_ALERT, r2_score
from src.visualization.style import (
    COLOR_FIT,
    COLOR_TARGET,
    COLOR_TEST,
    COLOR_TRAIN,
    FLOOR_LOG,
    floor_for_log,
    new_figure,
    save_figure,
    style_context,
)

N_DENSE = 1000  # points of the dense grid used to draw the polynomials
FIT_YLIM = (-2.0, 2.0)  # vertical window of the panels (high degrees can diverge)
PARITY_PAD = 0.25  # margin of the parity window, as a fraction of the target range


def key_degrees(result: SweepResult) -> list[int]:
    """Degrees kept for the parity plot: the lowest, C* (measured) and the highest."""
    return sorted({int(result.degrees[0]), result.optimal_degree, int(result.degrees[-1])})


def fit_overflow(values, ylim=FIT_YLIM) -> float | None:
    """Maximal amplitude ``max |y|`` if ``values`` leaves the ``ylim`` window, otherwise ``None``."""
    values = np.asarray(values, dtype=float)
    if np.all((values >= ylim[0]) & (values <= ylim[1])):
        return None
    return float(np.max(np.abs(values)))


def parity_window(y_true, pad: float = PARITY_PAD) -> tuple[float, float]:
    """Common window (abscissa and ordinate) of the parity plot: target range, widened by ``pad``.

    It does not depend on the predictions: a diverging model (d = 12) no longer squashes the
    target into a vertical column; excluded points are counted and reported.
    """
    y_true = np.asarray(y_true, dtype=float)
    lo, hi = float(y_true.min()), float(y_true.max())
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def plot_error_profile(result: SweepResult):
    """MSE(d) and test MSE(d) on a logarithmic scale; C* (argmin of the test MSE) marked."""
    with style_context():
        fig = new_figure()
        ax = fig.add_subplot()
        d = result.degrees
        ax.semilogy(d, floor_for_log(result.mse_train), "o-", color=COLOR_TRAIN,
                    label="MSE (training)")
        ax.semilogy(d, floor_for_log(result.mse_test), "s-", color=COLOR_TEST,
                    label="test MSE")
        star = result.optimal_degree
        ax.axvline(star, color="0.4", ls="--", lw=1, label=f"$C^*$ = {star} (min. test MSE)")
        ax.set_xlabel("degree $d$")
        ax.set_ylabel("mean squared error")
        ax.set_xticks(d)
        ax.set_title(f"Error profile, seed {result.seed} (display floor "
                     f"{FLOOR_LOG:g})")
        ax.legend()
    return fig


def plot_fits(result: SweepResult):
    """Panel of the fitted polynomials (d = 1..12) over the training data and the target."""
    n = len(result.degrees)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    xs = np.linspace(XMIN_NOMINAL, XMAX_NOMINAL, N_DENSE, endpoint=False)
    star = result.optimal_degree
    with style_context():
        fig = new_figure(2.4 * ncols, 2.0 * nrows)
        axes = fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        for ax, d, coef in zip(axes.ravel(), result.degrees, result.coefs):
            ax.plot(xs, target_function(xs), color=COLOR_TARGET, lw=1, label="target")
            pred = predict(coef, xs)
            ax.plot(xs, pred, color=COLOR_FIT, lw=1.4, label="polynomial")
            ax.plot(result.x_train[:, 0] if result.x_train.ndim == 2 else result.x_train,
                    result.y_train, "o", color=COLOR_TRAIN, ms=3, label="training")
            ax.set_ylim(*FIT_YLIM)
            overflow = fit_overflow(pred)
            if overflow is not None:  # clipped panel: reported on the panel
                ax.text(0.03, 0.04, f"out of frame: |y| max = {overflow:.3g}",
                        transform=ax.transAxes, fontsize=6, color=COLOR_TEST)
                for spine in ax.spines.values():
                    spine.set_edgecolor(COLOR_TEST)
            ax.set_title(f"$d$ = {int(d)}" + (" ($C^*$)" if int(d) == star else "")
                         + (" (clipped)" if overflow is not None else ""))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=3)
        fig.supxlabel("x")
        fig.supylabel(f"y (window [{FIT_YLIM[0]:g} ; {FIT_YLIM[1]:g}])")
    return fig


def plot_parity(result: SweepResult, degrees=None):
    """Parity y_test (target) against y_pred on the test set, orthonormal 1:1 frame.

    The window is set by the target (``parity_window``); predictions outside it
    are still drawn but out of frame, and their number is shown on the panel.
    """
    degrees = key_degrees(result) if degrees is None else list(degrees)
    star = result.optimal_degree
    y_te = result.y_test
    lo, hi = parity_window(y_te)
    with style_context():
        fig = new_figure(3.0 * len(degrees), 3.4)
        axes = fig.subplots(1, len(degrees), squeeze=False)[0]
        for ax, d in zip(axes, degrees):
            y_hat = result.y_pred_test[result.index_of(d)]
            ax.plot([lo, hi], [lo, hi], color="0.4", lw=1, label="y = x")
            ax.plot(y_te, y_hat, ".", color=COLOR_TEST, ms=4)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("$y_{test}$ (target)")
            ax.set_ylabel("$\\hat{y}$")
            n_out = int(np.sum((y_hat < lo) | (y_hat > hi)))
            if n_out:
                ax.text(0.03, 0.97, f"{n_out} points out of frame\n(|$\\hat{{y}}$| max = "
                        f"{float(np.max(np.abs(y_hat))):.3g})", transform=ax.transAxes,
                        fontsize=6, va="top", color=COLOR_TEST)
            tag = " ($C^*$)" if d == star else ""
            ax.set_title(f"$d$ = {d}{tag}, $R^2$ = {r2_score(y_te, y_hat):.4f}")
            ax.legend(loc="lower right")
    return fig


def plot_conditioning(result: SweepResult):
    """kappa(H^T H) against the degree, diagnostic threshold 1e7 (strict / diagnostic regimes)."""
    with style_context():
        fig = new_figure()
        ax = fig.add_subplot()
        ax.semilogy(result.degrees, floor_for_log(result.kappa_hth, 1.0), "o-",
                    color=COLOR_TRAIN, label="$\\kappa(H^TH)$")
        ax.axhline(KAPPA_HTH_ALERT, color="0.4", ls="--", lw=1,
                   label=f"threshold {KAPPA_HTH_ALERT:.0e}")
        ax.set_xlabel("degree $d$")
        ax.set_ylabel("$\\kappa(H^TH)$")
        ax.set_xticks(result.degrees)
        ax.set_title("Conditioning of the Vandermonde matrix (nominal grid)")
        ax.legend()
    return fig


def plot_campaign(campaign: CampaignResult):
    """Distribution of C* and median MSE / test MSE (interquartile band) over the seeds."""
    d = campaign.degrees
    test, train = campaign.summary("test"), campaign.summary("train")
    counts = campaign.cstar_counts()
    with style_context():
        fig = new_figure(9.0, 3.8)
        ax_c, ax_e = fig.subplots(1, 2)
        ax_c.bar(d, [counts[int(k)] for k in d], color=COLOR_TEST)
        ax_c.set_xticks(d)
        ax_c.set_xlabel("optimal degree $C^*$")
        ax_c.set_ylabel("number of seeds")
        ax_c.set_title(f"$C^*$ over {len(campaign.seeds)} seeds "
                       f"(mode {campaign.cstar_mode}, median {campaign.cstar_median:g})")
        for stats, color, label in ((train, COLOR_TRAIN, "MSE"), (test, COLOR_TEST, "test MSE")):
            ax_e.semilogy(d, floor_for_log(stats["median"]), "o-", color=color,
                          label=f"{label} median")
            ax_e.fill_between(d, floor_for_log(stats["q1"]), floor_for_log(stats["q3"]),
                              color=color, alpha=0.2, label=f"{label} Q1-Q3")
        ax_e.set_xticks(d)
        ax_e.set_xlabel("degree $d$")
        ax_e.set_ylabel("mean squared error")
        ax_e.set_title(f"Errors over {len(campaign.seeds)} seeds (floor {FLOOR_LOG:g})")
        ax_e.legend()
    return fig


def save_nominal_figures(result: SweepResult, campaign: CampaignResult, out_dir) -> list[Path]:
    """Generate and export (PDF + 300 DPI PNG) the five nominal 1D figures; return the paths."""
    out_dir = Path(out_dir)
    figures = {
        "nominal_1d_error_profile": plot_error_profile(result),
        "nominal_1d_fits": plot_fits(result),
        "nominal_1d_parity": plot_parity(result),
        "nominal_1d_conditioning": plot_conditioning(result),
        "nominal_1d_campaign_50_seeds": plot_campaign(campaign),
    }
    paths = []
    for stem, fig in figures.items():
        paths.extend(save_figure(fig, out_dir / stem))
    return paths
