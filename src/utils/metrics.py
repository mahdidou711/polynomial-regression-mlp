"""Error metrics and matrix conditioning.

All metrics require two finite, non-empty real arrays of strictly equal shapes:
no broadcasting is tolerated (a column vector (N, 1) against a flat vector (N,)
would silently produce an (N, N) matrix and a wrong value). A mismatching shape
raises ``AssertionError``; other invalid inputs raise ``TypeError`` or ``ValueError``.
"""

import logging
from typing import NamedTuple

import numpy as np

from src.utils.logger import log_event

KAPPA_HTH_ALERT = 1e7  # diagnostic threshold on kappa(H^T H)

# Names that ``condition_numbers`` itself sets in the event (kappa_h ... shape), plus
# the parameters of ``log_event`` (logger, event, level, exc_info) and the ``time`` field of the
# JSON Lines format. Accepting them as context would cause an opaque ``TypeError``.
_CONTEXT_FORBIDDEN = frozenset(
    {"kappa_h", "kappa_hth", "threshold", "shape", "level", "event", "logger", "time", "exc_info"}
)

_logger = logging.getLogger("polynomial_mlp.metrics")


def _as_float_array(x, name: str) -> np.ndarray:
    """Copy of ``x`` as float64; ``TypeError`` if the dtype is not real (int or float)."""
    a = np.asarray(x)
    if a.dtype.kind not in "iuf":  # excludes bool, complex, strings, objects
        raise TypeError(f"real dtype (int or float) expected for {name}, got {a.dtype}")
    if a.ndim == 0:
        raise ValueError(f"{name} : ndim = 0 not supported (array expected)")
    return a.astype(np.float64)


def _check_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = _as_float_array(y_true, "y_true")
    p = _as_float_array(y_pred, "y_pred")
    if y.shape != p.shape:  # explicit: survives ``python -O``, unlike ``assert``
        raise AssertionError(
            f"mismatching shapes: y_true.shape={y.shape} != y_pred.shape={p.shape} "
            "(broadcasting forbidden)"
        )
    if y.size == 0:
        raise ValueError("empty arrays")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("non-finite values (NaN or inf) in y_true or y_pred")
    return y, p


def getMSE(y_true, y_pred) -> float:
    """MSE = (1/N) sum_i (y_i - y_hat_i)^2.

    For arrays of shape (N, k), the mean runs over all entries.
    """
    y, p = _check_pair(y_true, y_pred)
    return float(np.mean((y - p) ** 2))


def rmse(y_true, y_pred) -> float:
    """Root of the MSE."""
    return float(np.sqrt(getMSE(y_true, y_pred)))


def max_abs_error(y_true, y_pred) -> float:
    """Maximal absolute error max_k |y_k - y_hat_k|."""
    y, p = _check_pair(y_true, y_pred)
    return float(np.max(np.abs(y - p)))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination 1 - SS_res / SS_tot, for 1D vectors.

    ``ValueError`` if ``y_true`` is constant (SS_tot zero: R^2 undefined).
    """
    y, p = _check_pair(y_true, y_pred)
    if y.ndim != 1:
        raise ValueError(f"r2_score: 1D vectors expected, got ndim = {y.ndim}")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        raise ValueError("zero variance of y_true: R^2 undefined")
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot


class Conditioning(NamedTuple):
    """Condition numbers of H (2-norm) and of H^T H."""

    kappa_h: float
    kappa_hth: float


def condition_numbers(h, **context) -> Conditioning:
    """kappa_2(H) = s_max / s_min and kappa(H^T H) for a matrix H (N x p).

    - kappa(H^T H) = kappa(H)^2 when N >= p. For N < p (for example n_samples = 10
      with d >= 10), H^T H has rank <= N < p hence is singular: kappa(H^T H) = inf,
      whereas kappa(H) stays finite on the N non-zero singular values.
    - A zero minimal singular value gives kappa = inf; a floating-point overflow
      also gives inf (no exception).
    - Logs the result; above ``KAPPA_HTH_ALERT`` the event is emitted at WARNING
      level (diagnostic alert, non-blocking). The keyword arguments ``context``
      (for example ``degree=5``) are added to the event; a reserved context name
      (``kappa_h``, ``kappa_hth``, ``threshold``, ``shape``, ``level``, ``event``,
      ``logger``, ``time``, ``exc_info``) raises ``ValueError`` before any
      computation.
    """
    clash = _CONTEXT_FORBIDDEN.intersection(context)
    if clash:
        raise ValueError(f"reserved context names (internal fields): {sorted(clash)}")
    a = _as_float_array(h, "H")
    if a.ndim != 2:
        raise ValueError(f"H: 2D matrix expected, got ndim = {a.ndim}")
    if a.size == 0:
        raise ValueError("H: empty matrix")
    if not np.isfinite(a).all():
        raise ValueError("H: non-finite values (NaN or inf)")

    n_rows, n_cols = a.shape
    s = np.linalg.svd(a, compute_uv=False)  # decreasing, min(N, p) values
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
