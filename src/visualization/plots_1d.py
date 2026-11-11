"""Figures 1D de l'experience nominale (etape 3).

Chaque fonction ``plot_*`` renvoie une ``Figure`` (sans pyplot) ; ``save_nominal_figures``
les exporte en PDF vectoriel + PNG 300 DPI via ``save_figure``. Les echelles
logarithmiques utilisent un plancher d'affichage (``FLOOR_LOG``, avenant 7.9) applique a
une copie : les valeurs stockees ne sont jamais modifiees.
"""

from pathlib import Path

import numpy as np

from src.data import XMAX_NOMINAL, XMIN_NOMINAL, fonction_cible
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

N_DENSE = 1000  # points de la grille dense de trace des polynomes
FIT_YLIM = (-2.0, 2.0)  # fenetre verticale des planches (les degres eleves peuvent diverger)
PARITY_PAD = 0.25  # marge de la fenetre de parite, en fraction de l'etendue de la cible (AE3-08)


def key_degrees(result: SweepResult) -> list[int]:
    """Degres retenus pour la parite : le plus bas, C* (mesure) et le plus eleve."""
    return sorted({int(result.degrees[0]), result.optimal_degree, int(result.degrees[-1])})


def fit_overflow(values, ylim=FIT_YLIM) -> float | None:
    """Amplitude maximale ``max |y|`` si ``values`` sort de la fenetre ``ylim``, sinon ``None``."""
    values = np.asarray(values, dtype=float)
    if np.all((values >= ylim[0]) & (values <= ylim[1])):
        return None
    return float(np.max(np.abs(values)))


def parity_window(y_true, pad: float = PARITY_PAD) -> tuple[float, float]:
    """Fenetre commune (abscisse et ordonnee) de la parite : etendue de la cible, elargie de ``pad``.

    Elle ne depend pas des predictions : un modele divergent (d = 12) ne ecrase plus la
    cible dans une colonne verticale ; les points exclus sont comptes et signales (AE3-08).
    """
    y_true = np.asarray(y_true, dtype=float)
    lo, hi = float(y_true.min()), float(y_true.max())
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def plot_error_profile(result: SweepResult):
    """EQM(d) et EQMT(d) en echelle logarithmique ; C* (argmin EQMT) repere."""
    with style_context():
        fig = new_figure()
        ax = fig.add_subplot()
        d = result.degrees
        ax.semilogy(d, floor_for_log(result.eqm_train), "o-", color=COLOR_TRAIN,
                    label="EQM (apprentissage)")
        ax.semilogy(d, floor_for_log(result.eqm_test), "s-", color=COLOR_TEST,
                    label="EQMT (test)")
        star = result.optimal_degree
        ax.axvline(star, color="0.4", ls="--", lw=1, label=f"$C^*$ = {star} (min. EQMT)")
        ax.set_xlabel("degré $d$")
        ax.set_ylabel("erreur quadratique moyenne")
        ax.set_xticks(d)
        ax.set_title(f"Profil d'erreur, graine {result.seed} (plancher d'affichage "
                     f"{FLOOR_LOG:g})")
        ax.legend()
    return fig


def plot_fits(result: SweepResult):
    """Planche des polynomes ajustes (d = 1..12) sur les donnees d'apprentissage et la cible."""
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
            ax.plot(xs, fonction_cible(xs), color=COLOR_TARGET, lw=1, label="cible")
            pred = predict(coef, xs)
            ax.plot(xs, pred, color=COLOR_FIT, lw=1.4, label="polynôme")
            ax.plot(result.x_train[:, 0] if result.x_train.ndim == 2 else result.x_train,
                    result.y_train, "o", color=COLOR_TRAIN, ms=3, label="apprentissage")
            ax.set_ylim(*FIT_YLIM)
            overflow = fit_overflow(pred)
            if overflow is not None:  # panneau coupe : signale sur le panneau (AE3-08)
                ax.text(0.03, 0.04, f"hors cadre : |y| max = {overflow:.3g}",
                        transform=ax.transAxes, fontsize=6, color=COLOR_TEST)
                for spine in ax.spines.values():
                    spine.set_edgecolor(COLOR_TEST)
            ax.set_title(f"$d$ = {int(d)}" + (" ($C^*$)" if int(d) == star else "")
                         + (" (coupé)" if overflow is not None else ""))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=3)
        fig.supxlabel("x")
        fig.supylabel(f"y (fenêtre [{FIT_YLIM[0]:g} ; {FIT_YLIM[1]:g}])")
    return fig


def plot_parity(result: SweepResult, degrees=None):
    """Parite y_test (cible) contre y_pred sur la base de test, repere orthonormé 1:1.

    La fenetre est fixee par la cible (``parity_window``) ; les predictions qui en sortent
    restent tracees mais hors cadre, et leur nombre est indique sur le panneau (AE3-08).
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
            ax.set_xlabel("$y_{test}$ (cible)")
            ax.set_ylabel("$\\hat{y}$")
            n_out = int(np.sum((y_hat < lo) | (y_hat > hi)))
            if n_out:
                ax.text(0.03, 0.97, f"{n_out} points hors cadre\n(|$\\hat{{y}}$| max = "
                        f"{float(np.max(np.abs(y_hat))):.3g})", transform=ax.transAxes,
                        fontsize=6, va="top", color=COLOR_TEST)
            tag = " ($C^*$)" if d == star else ""
            ax.set_title(f"$d$ = {d}{tag}, $R^2$ = {r2_score(y_te, y_hat):.4f}")
            ax.legend(loc="lower right")
    return fig


def plot_conditioning(result: SweepResult):
    """kappa(H^T H) en fonction du degre, seuil diagnostique 1e7 (regimes strict / diagnostic)."""
    with style_context():
        fig = new_figure()
        ax = fig.add_subplot()
        ax.semilogy(result.degrees, floor_for_log(result.kappa_hth, 1.0), "o-",
                    color=COLOR_TRAIN, label="$\\kappa(H^TH)$")
        ax.axhline(KAPPA_HTH_ALERT, color="0.4", ls="--", lw=1,
                   label=f"seuil {KAPPA_HTH_ALERT:.0e}")
        ax.set_xlabel("degré $d$")
        ax.set_ylabel("$\\kappa(H^TH)$")
        ax.set_xticks(result.degrees)
        ax.set_title("Conditionnement de la matrice de Vandermonde (grille nominale)")
        ax.legend()
    return fig


def plot_campaign(campaign: CampaignResult):
    """Distribution de C* et EQM / EQMT medianes (bande interquartile) sur les graines."""
    d = campaign.degrees
    test, train = campaign.summary("test"), campaign.summary("train")
    counts = campaign.cstar_counts()
    with style_context():
        fig = new_figure(9.0, 3.8)
        ax_c, ax_e = fig.subplots(1, 2)
        ax_c.bar(d, [counts[int(k)] for k in d], color=COLOR_TEST)
        ax_c.set_xticks(d)
        ax_c.set_xlabel("degré optimal $C^*$")
        ax_c.set_ylabel("nombre de graines")
        ax_c.set_title(f"$C^*$ sur {len(campaign.seeds)} graines "
                       f"(mode {campaign.cstar_mode}, médiane {campaign.cstar_median:g})")
        for stats, color, label in ((train, COLOR_TRAIN, "EQM"), (test, COLOR_TEST, "EQMT")):
            ax_e.semilogy(d, floor_for_log(stats["median"]), "o-", color=color,
                          label=f"{label} médiane")
            ax_e.fill_between(d, floor_for_log(stats["q1"]), floor_for_log(stats["q3"]),
                              color=color, alpha=0.2, label=f"{label} Q1-Q3")
        ax_e.set_xticks(d)
        ax_e.set_xlabel("degré $d$")
        ax_e.set_ylabel("erreur quadratique moyenne")
        ax_e.set_title(f"Erreurs sur {len(campaign.seeds)} graines (plancher {FLOOR_LOG:g})")
        ax_e.legend()
    return fig


def save_nominal_figures(result: SweepResult, campaign: CampaignResult, out_dir) -> list[Path]:
    """Genere et exporte (PDF + PNG 300 DPI) les cinq figures 1D nominales ; renvoie les chemins."""
    out_dir = Path(out_dir)
    figures = {
        "nominal_1d_error_profile": plot_error_profile(result),
        "nominal_1d_fits": plot_fits(result),
        "nominal_1d_parity": plot_parity(result),
        "nominal_1d_conditioning": plot_conditioning(result),
        "nominal_1d_campaign_50_graines": plot_campaign(campaign),
    }
    paths = []
    for stem, fig in figures.items():
        paths.extend(save_figure(fig, out_dir / stem))
    return paths
