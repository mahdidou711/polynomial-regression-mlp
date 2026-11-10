"""Metriques d'erreur et conditionnement matriciel (etape 0).

Toutes les metriques exigent deux tableaux reels finis, non vides, de formes
strictement egales : aucun broadcasting n'est tolere (un vecteur colonne (N, 1)
contre un vecteur plat (N,) produirait silencieusement une matrice (N, N) et
une valeur fausse). Une forme divergente leve ``AssertionError`` (plan etape 0) ;
les autres entrees invalides levent ``TypeError`` ou ``ValueError``.
"""

import logging
from typing import NamedTuple

import numpy as np

from src.utils.logger import log_event

KAPPA_HTH_ALERT = 1e7  # seuil diagnostique sur kappa(H^T H) (CDC 9.1, avenant 7.10)

_logger = logging.getLogger("c0_tp2_mlp.metrics")


def _as_float_array(x, name: str) -> np.ndarray:
    """Copie de ``x`` en float64 ; ``TypeError`` si le dtype n'est pas reel (int ou float)."""
    a = np.asarray(x)
    if a.dtype.kind not in "iuf":  # exclut bool, complexe, chaines, objets
        raise TypeError(f"dtype reel (entier ou flottant) attendu pour {name}, recu {a.dtype}")
    if a.ndim == 0:
        raise ValueError(f"{name} : ndim = 0 non supporte (tableau attendu)")
    return a.astype(np.float64)


def _check_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = _as_float_array(y_true, "y_true")
    p = _as_float_array(y_pred, "y_pred")
    if y.shape != p.shape:  # explicite : survit a ``python -O``, contrairement a ``assert``
        raise AssertionError(
            f"formes divergentes : y_true.shape={y.shape} != y_pred.shape={p.shape} "
            "(broadcasting interdit)"
        )
    if y.size == 0:
        raise ValueError("tableaux vides")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("valeurs non finies (NaN ou inf) dans y_true ou y_pred")
    return y, p


def getMSE(y_true, y_pred) -> float:
    """EQM = (1/N) sum_i (y_i - y_hat_i)^2 (nom impose par EXG-OFF-B2-01).

    Pour des tableaux de forme (N, k), la moyenne porte sur toutes les entrees.
    """
    y, p = _check_pair(y_true, y_pred)
    return float(np.mean((y - p) ** 2))


def rmse(y_true, y_pred) -> float:
    """Racine de l'EQM."""
    return float(np.sqrt(getMSE(y_true, y_pred)))


def max_abs_error(y_true, y_pred) -> float:
    """Erreur absolue maximale MAE_max = max_k |y_k - y_hat_k| (CDC section 7)."""
    y, p = _check_pair(y_true, y_pred)
    return float(np.max(np.abs(y - p)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient de determination 1 - SS_res / SS_tot, pour des vecteurs 1D.

    ``ValueError`` si ``y_true`` est constant (SS_tot nul : R^2 indefini).
    """
    y, p = _check_pair(y_true, y_pred)
    if y.ndim != 1:
        raise ValueError(f"r2_score : vecteurs 1D attendus, recu ndim = {y.ndim}")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        raise ValueError("variance nulle de y_true : R^2 indefini")
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot


class Conditioning(NamedTuple):
    """Nombres de conditionnement de H (2-norme) et de H^T H."""

    kappa_h: float
    kappa_hth: float


def condition_numbers(h, **context) -> Conditioning:
    """kappa_2(H) = s_max / s_min et kappa(H^T H) pour une matrice H (N x p).

    - kappa(H^T H) = kappa(H)^2 lorsque N >= p. Pour N < p (par exemple NbEx = 10
      avec d >= 10), H^T H est de rang <= N < p donc singuliere : kappa(H^T H) = inf,
      alors que kappa(H) reste fini sur les N valeurs singulieres non nulles.
    - Une valeur singuliere minimale nulle donne kappa = inf ; un depassement
      de capacite flottante donne aussi inf (aucune exception).
    - Journalise le resultat ; au-dessus de ``KAPPA_HTH_ALERT`` l'evenement est
      emis au niveau WARNING (alerte diagnostique, sans blocage). Les arguments
      nommes ``context`` (par exemple ``degree=5``) sont ajoutes a l'evenement.
    """
    a = _as_float_array(h, "H")
    if a.ndim != 2:
        raise ValueError(f"H : matrice 2D attendue, recu ndim = {a.ndim}")
    if a.size == 0:
        raise ValueError("H : matrice vide")
    if not np.isfinite(a).all():
        raise ValueError("H : valeurs non finies (NaN ou inf)")

    n_rows, n_cols = a.shape
    s = np.linalg.svd(a, compute_uv=False)  # decroissantes, min(N, p) valeurs
    with np.errstate(over="ignore", divide="ignore"):
        kappa_h = np.inf if s[-1] == 0.0 else float(s[0] / s[-1])
        kappa_hth = np.inf if n_rows < n_cols else float(np.float64(kappa_h) ** 2)

    alert = kappa_hth > KAPPA_HTH_ALERT
    log_event(
        _logger,
        "conditioning_alert" if alert else "conditioning",
        level=logging.WARNING if alert else logging.INFO,
        kappa_h=kappa_h,
        kappa_hth=kappa_hth,
        threshold=KAPPA_HTH_ALERT,
        shape=[n_rows, n_cols],
        **context,
    )
    return Conditioning(kappa_h, kappa_hth)
