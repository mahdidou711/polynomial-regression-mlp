"""1D polynomial regression: Vandermonde matrix and OLS / QR / SVD solvers.

Model: y ~ H @ beta, with H[i, j] = x_i ** j for j = 0..d (uncentred matrix).
Column 0 of H (ones) already carries the constant term: no additional bias is
added, and the scikit-learn reference is built with
``LinearRegression(fit_intercept=False)``.

Three solvers:
- ``solve_normal_equations``: (H^T H) beta = H^T y through ``np.linalg.solve``;
- ``solve_qr``              : H = QR then R beta = Q^T y (``scipy.linalg.solve_triangular``);
- ``solve_svd``             : ``scipy.linalg.lstsq`` (driver ``gelsd``, SVD), the solver of
  ``LinearRegression``; minimum-norm solution if H has rank lower than p.

The normal equations and QR require a full column rank system (N >= p and
rank(H) = p); otherwise they raise ``RankDeficientError`` (subclass of
``numpy.linalg.LinAlgError``). Only the SVD solver handles N < p and rank-deficient
matrices. ``compare_solvers`` never raises for this reason: it qualifies each method
(computed or not applicable) and measures the gap to the SVD solver.

The two condition numbers are distinct: kappa_2(H) governs QR and SVD;
kappa(H^T H) = kappa_2(H)**2 (full column rank) governs the normal equations.
Both are obtained by ``condition_numbers`` (SVD of H, without forming H^T H).
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression

from src.utils.logger import log_event
from src.utils.metrics import KAPPA_HTH_ALERT, _as_float_array, condition_numbers

METHODS = ("normal", "qr", "svd")  # presentation order; "svd" is the reference

EPS_0 = 1e-12  # floor of the denominator of the agreement criterion
STRICT_RTOL = 1e-7  # strict relative tolerance (regime kappa(H^T H) <= KAPPA_HTH_ALERT)
MIXED_ATOL = 1e-10  # mixed per-coefficient criterion: diagnostic quantity
MIXED_RTOL = 1e-7

_logger = logging.getLogger("polynomial_mlp.polynomial")


class RankDeficientError(np.linalg.LinAlgError):
    """H has incomplete column rank (N < p or rank(H) < p): H^T H is singular."""


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def _check_degree(degree) -> int:
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise TypeError(f"degree: integer expected, got {type(degree).__name__} : {degree!r}")
    if degree < 0:
        raise ValueError(f"degree must be >= 0: {degree}")
    return int(degree)


def _check_abscissas(x) -> np.ndarray:
    """Abscissas as an (N,) vector; an (N, 1) column vector is accepted
    (native shape of ``generate_example_1d``), any other shape is rejected."""
    a = _as_float_array(x, "x")
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"x: shape (N,) or (N, 1) expected, got {a.shape}")
    if a.size == 0:
        raise ValueError("x: empty vector")
    if not np.isfinite(a).all():
        raise ValueError("x: non-finite values (NaN or inf)")
    return a


def _check_system(h, y) -> tuple[np.ndarray, np.ndarray]:
    """Validate (H, y): H (N, p) real finite, y (N,) real finite (no broadcasting)."""
    a = _as_float_array(h, "H")
    if a.ndim != 2:
        raise ValueError(f"H: 2D matrix expected, got ndim = {a.ndim}")
    if a.size == 0:
        raise ValueError("H: empty matrix")
    b = _as_float_array(y, "y")
    if b.shape != (a.shape[0],):  # explicit: survives ``python -O``
        raise AssertionError(
            f"mismatching shapes: y.shape={b.shape} != ({a.shape[0]},) "
            "(1D vector of length N expected, broadcasting forbidden)"
        )
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite values (NaN or inf) in H or y")
    return a, b


def _require_full_column_rank(h: np.ndarray, method: str) -> None:
    n_rows, n_cols = h.shape
    if n_rows < n_cols:
        raise RankDeficientError(
            f"{method} : N = {n_rows} < p = {n_cols}, H^T H singular (use the SVD solver)"
        )
    rank = int(np.linalg.matrix_rank(h))
    if rank < n_cols:
        raise RankDeficientError(
            f"{method} : rank(H) = {rank} < p = {n_cols}, H^T H singular "
            "(use the SVD solver)"
        )


# --------------------------------------------------------------------------
# Vandermonde matrix, prediction, residuals
# --------------------------------------------------------------------------
def vandermonde(x, degree) -> np.ndarray:
    """Uncentred Vandermonde matrix H (N, d+1), H[i, j] = x_i ** j, j = 0..d.

    ``x``: vector (N,) or column (N, 1) of finite reals. Column 0 equals 1 (constant
    term, including for x_i = 0). ``ValueError`` on floating-point overflow.
    """
    d = _check_degree(degree)
    a = _check_abscissas(x)
    with np.errstate(over="ignore", invalid="ignore"):
        h = a[:, None] ** np.arange(d + 1)
    if not np.isfinite(h).all():
        raise ValueError(f"overflow in x ** j for degree = {d}")
    return h


def _check_coef(coef) -> np.ndarray:
    c = _as_float_array(coef, "coef")
    if c.ndim != 1 or c.size == 0:
        raise ValueError(f"coef: non-empty 1D vector expected, got {c.shape}")
    if not np.isfinite(c).all():
        raise ValueError("coef: non-finite values (NaN or inf)")
    return c


def predict(coef, x) -> np.ndarray:
    """Prediction y_hat = H(x) @ coef, of shape (N,); the degree is ``len(coef) - 1``."""
    c = _check_coef(coef)
    return vandermonde(x, c.size - 1) @ c


def residuals(coef, x, y) -> np.ndarray:
    """Residuals y - y_hat, of shape (N,); ``y`` must be a 1D vector of length N."""
    y_hat = predict(coef, x)
    b = _as_float_array(y, "y")
    if b.shape != y_hat.shape:
        raise AssertionError(
            f"mismatching shapes: y.shape={b.shape} != {y_hat.shape} (broadcasting forbidden)"
        )
    return b - y_hat


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------
def solve_normal_equations(h, y) -> np.ndarray:
    """Normal equations (H^T H) beta = H^T y (``np.linalg.solve``).

    Accuracy O(kappa(H^T H) * eps): unstable as soon as kappa(H)^2 approaches 1/eps.
    ``RankDeficientError`` if N < p or rank(H) < p; ``LinAlgError`` if the LU
    factorisation meets an exactly zero pivot.
    """
    a, b = _check_system(h, y)
    _require_full_column_rank(a, "normal equations")
    return np.linalg.solve(a.T @ a, a.T @ b)


def solve_qr(h, y) -> np.ndarray:
    """Economy QR factorisation H = QR then R beta = Q^T y (back substitution).

    Accuracy O(kappa(H) * eps). ``RankDeficientError`` if N < p or rank(H) < p.
    """
    a, b = _check_system(h, y)
    _require_full_column_rank(a, "QR")
    q, r = linalg.qr(a, mode="economic")
    return linalg.solve_triangular(r, q.T @ b, lower=False)


def solve_svd(h, y) -> np.ndarray:
    """Least squares by SVD (``scipy.linalg.lstsq``, driver ``gelsd``).

    Same call as ``LinearRegression``: singular values below
    ``max(N, p) * eps * sigma_max`` are ignored. Minimum-norm solution if H has rank
    lower than p (N < p, repeated abscissas).
    """
    a, b = _check_system(h, y)
    cond = max(a.shape) * np.finfo(np.float64).eps
    return linalg.lstsq(a, b, cond=cond, lapack_driver="gelsd")[0]


def solve_sklearn_reference(h, y) -> np.ndarray:
    """scikit-learn reference: ``LinearRegression(fit_intercept=False)`` on H.

    The column of ones of H is the constant term; ``fit_intercept=False`` avoids a
    second bias. Same algorithm as ``solve_svd``.
    """
    a, b = _check_system(h, y)
    return LinearRegression(fit_intercept=False).fit(a, b).coef_


SOLVERS = {"normal": solve_normal_equations, "qr": solve_qr, "svd": solve_svd}


def relative_error_inf(beta, beta_ref, eps0: float = EPS_0) -> float:
    """Relative gap in infinity norm ||beta - ref||_inf / max(||ref||_inf, eps0).

    Returns ``inf`` if ``beta`` contains a non-finite value.
    """
    b = _as_float_array(beta, "beta")
    r = _as_float_array(beta_ref, "beta_ref")
    if b.shape != r.shape or b.ndim != 1:
        raise AssertionError(f"mismatching or non-1D shapes: {b.shape} against {r.shape}")
    if not np.isfinite(b).all():
        return float("inf")
    return float(np.max(np.abs(b - r)) / max(float(np.max(np.abs(r))), eps0))


# --------------------------------------------------------------------------
# Fitting and comparison
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PolynomialFit:
    """Result of ``fit_polynomial``: coefficients (beta_0..beta_d) and diagnostics."""

    coef: np.ndarray
    degree: int
    method: str
    rank: int
    kappa_h: float
    kappa_hth: float

    def predict(self, x) -> np.ndarray:
        return predict(self.coef, x)


AUTO = "auto"  # automatic selection of fit_polynomial: QR if full rank, otherwise SVD


def _check_method(method) -> str:
    if method != AUTO and method not in SOLVERS:
        raise ValueError(f"unknown method: {method!r} (expected among {METHODS + (AUTO,)})")
    return method


def fit_polynomial(x, y, degree, method: str = "svd") -> PolynomialFit:
    """Fit a polynomial of degree ``degree`` with ``method``.

    ``method``: 'normal', 'qr', 'svd' (default, reference solver) or 'auto'. 'auto'
    switches to the SVD, the only method defined for N < p or rank(H) < p, and uses QR
    otherwise; ``PolynomialFit.method`` gives the method actually used.
    The conditioning is computed and logged (``condition_numbers``: WARNING alert
    if kappa(H^T H) > KAPPA_HTH_ALERT, non-blocking). ``RankDeficientError`` if
    'normal' or 'qr' is explicitly requested on a system of incomplete column rank.
    """
    method = _check_method(method)
    h = vandermonde(x, degree)
    a, b = _check_system(h, y)
    rank = int(np.linalg.matrix_rank(a))
    if method == AUTO:
        method = "qr" if rank == a.shape[1] else "svd"
    cond = condition_numbers(a, degree=int(degree), method=method, n_samples=a.shape[0])
    coef = SOLVERS[method](a, b)
    return PolynomialFit(coef, int(degree), method, rank, cond.kappa_h, cond.kappa_hth)


@dataclass(frozen=True)
class SolverComparison:
    """Comparison of the solvers on a system (x, y, degree).

    - ``regime``: ``'strict'`` if rank(H) = p and kappa(H^T H) <= KAPPA_HTH_ALERT (agreement
      at ``STRICT_RTOL`` is then expected), otherwise ``'diagnostic'`` (gap
      measured and reported, never rejected).
    - ``coefficients``: per method ('normal', 'qr', 'svd', 'sklearn'); ``None`` if the
      method is not applicable (see ``errors``).
    - ``rel_error_inf``: agreement criterion against the SVD, for
      'normal', 'qr' and 'sklearn' (``None`` if not computed).
    - ``mixed_tolerance_ok``: mixed per-coefficient criterion (atol 1e-10, rtol 1e-7),
      diagnostic quantity.
    """

    degree: int
    n_samples: int
    n_params: int
    rank: int
    kappa_h: float
    kappa_hth: float
    regime: str
    coefficients: dict
    errors: dict
    rel_error_inf: dict
    mixed_tolerance_ok: dict

    def within_strict_tolerance(self, method: str) -> bool:
        """True if ``method`` was computed and agrees with the SVD at ``STRICT_RTOL``."""
        err = self.rel_error_inf.get(method)
        return err is not None and err <= STRICT_RTOL


def compare_solvers(x, y, degree) -> SolverComparison:
    """Compare normal equations, QR, SVD and ``LinearRegression`` on (x, y, degree).

    Does not raise when 'normal' or 'qr' do not apply (N < p, rank deficiency,
    zero pivot, non-finite result): the method is qualified in ``errors``. Logs
    the ``solver_comparison`` event (INFO level); the conditioning alert is
    emitted by ``condition_numbers``.
    """
    d = _check_degree(degree)
    a, b = _check_system(vandermonde(x, d), y)
    n_rows, n_cols = a.shape
    cond = condition_numbers(a, degree=d, n_samples=n_rows)
    rank = int(np.linalg.matrix_rank(a))

    coefficients = {"svd": solve_svd(a, b), "sklearn": solve_sklearn_reference(a, b)}
    errors = {}
    for method in ("normal", "qr"):
        try:
            coef = SOLVERS[method](a, b)
        except np.linalg.LinAlgError as exc:  # includes RankDeficientError
            coefficients[method], errors[method] = None, f"{type(exc).__name__}: {exc}"
            continue
        if not np.isfinite(coef).all():
            coefficients[method], errors[method] = None, "non-finite result"
        else:
            coefficients[method] = coef

    ref = coefficients["svd"]
    rel_error_inf, mixed_ok = {}, {}
    for method in ("normal", "qr", "sklearn"):
        coef = coefficients.get(method)
        if coef is None:
            rel_error_inf[method] = mixed_ok[method] = None
            continue
        rel_error_inf[method] = relative_error_inf(coef, ref)
        mixed_ok[method] = bool(np.all(np.abs(coef - ref) <= MIXED_ATOL + MIXED_RTOL * np.abs(ref)))

    regime = "strict" if rank == n_cols and cond.kappa_hth <= KAPPA_HTH_ALERT else "diagnostic"
    log_event(
        _logger,
        "solver_comparison",
        degree=d,
        n_samples=n_rows,
        n_params=n_cols,
        rank=rank,
        kappa_h=cond.kappa_h,
        kappa_hth=cond.kappa_hth,
        regime=regime,
        rel_error_normal=rel_error_inf["normal"],
        rel_error_qr=rel_error_inf["qr"],
        rel_error_sklearn=rel_error_inf["sklearn"],
        not_applicable=sorted(errors),
    )
    return SolverComparison(
        degree=d,
        n_samples=n_rows,
        n_params=n_cols,
        rank=rank,
        kappa_h=cond.kappa_h,
        kappa_hth=cond.kappa_hth,
        regime=regime,
        coefficients=coefficients,
        errors=errors,
        rel_error_inf=rel_error_inf,
        mixed_tolerance_ok=mixed_ok,
    )
