"""Regression polynomiale 1D : Vandermonde et solveurs OLS / QR / SVD (etape 2).

Modele : y ~ H @ beta, avec H[i, j] = x_i ** j pour j = 0..d (matrice non centree).
La colonne 0 de H (des uns) porte deja le terme constant : aucun biais
supplementaire n'est ajoute, et la reference scikit-learn est construite avec
``LinearRegression(fit_intercept=False)`` (EXG-PROG-B1-03).

Trois solveurs, au sens du CDC 7.1 :
- ``solve_normal_equations`` : (H^T H) beta = H^T y par ``np.linalg.solve`` ;
- ``solve_qr``               : H = QR puis R beta = Q^T y (``scipy.linalg.solve_triangular``) ;
- ``solve_svd``              : ``scipy.linalg.lstsq`` (pilote ``gelsd``, SVD), le solveur de
  ``LinearRegression`` ; solution de norme minimale si H est de rang inferieur a p.

Les equations normales et QR exigent un systeme de rang colonne complet (N >= p et
rang(H) = p) ; sinon elles levent ``RankDeficientError`` (sous-classe de
``numpy.linalg.LinAlgError``). Seul le solveur SVD traite N < p et les matrices
deficientes en rang. ``compare_solvers`` ne leve jamais pour cette raison : il qualifie
chaque methode (calculee ou non applicable) et mesure l'ecart au solveur SVD.

Distinction des conditionnements (avenant 10.2) : kappa_2(H) conditionne QR et SVD ;
kappa(H^T H) = kappa_2(H)**2 (rang colonne complet) conditionne les equations normales.
Les deux sont obtenus par ``condition_numbers`` (SVD de H, sans former H^T H).
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from sklearn.linear_model import LinearRegression

from src.utils.logger import log_event
from src.utils.metrics import KAPPA_HTH_ALERT, _as_float_array, condition_numbers

METHODS = ("normal", "qr", "svd")  # ordre de presentation ; "svd" est la reference

EPS_0 = 1e-12  # plancher du denominateur du critere de concordance (avenant 7.10)
STRICT_RTOL = 1e-7  # tolerance relative stricte (regime kappa(H^T H) <= KAPPA_HTH_ALERT)
MIXED_ATOL = 1e-10  # critere mixte par coefficient : grandeur diagnostique (CDC 7.1 point 3)
MIXED_RTOL = 1e-7

_logger = logging.getLogger("c0_tp2_mlp.polynomial")


class RankDeficientError(np.linalg.LinAlgError):
    """H est de rang colonne incomplet (N < p ou rang(H) < p) : H^T H est singuliere."""


# --------------------------------------------------------------------------
# Validation des entrees
# --------------------------------------------------------------------------
def _check_degree(degree) -> int:
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise TypeError(f"degree : entier attendu, recu {type(degree).__name__} : {degree!r}")
    if degree < 0:
        raise ValueError(f"degree doit etre >= 0 : {degree}")
    return int(degree)


def _check_abscissas(x) -> np.ndarray:
    """Abscisses sous forme de vecteur (N,) ; un vecteur colonne (N, 1) est accepte
    (forme native de ``genere_exemple_dim1``), toute autre forme est rejetee."""
    a = _as_float_array(x, "x")
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"x : forme (N,) ou (N, 1) attendue, recu {a.shape}")
    if a.size == 0:
        raise ValueError("x : vecteur vide")
    if not np.isfinite(a).all():
        raise ValueError("x : valeurs non finies (NaN ou inf)")
    return a


def _check_system(h, y) -> tuple[np.ndarray, np.ndarray]:
    """Valide (H, y) : H (N, p) reelle finie, y (N,) reelle finie (pas de broadcasting)."""
    a = _as_float_array(h, "H")
    if a.ndim != 2:
        raise ValueError(f"H : matrice 2D attendue, recu ndim = {a.ndim}")
    if a.size == 0:
        raise ValueError("H : matrice vide")
    b = _as_float_array(y, "y")
    if b.shape != (a.shape[0],):  # explicite : survit a ``python -O``
        raise AssertionError(
            f"formes divergentes : y.shape={b.shape} != ({a.shape[0]},) "
            "(vecteur 1D de longueur N attendu, broadcasting interdit)"
        )
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("valeurs non finies (NaN ou inf) dans H ou y")
    return a, b


def _require_full_column_rank(h: np.ndarray, method: str) -> None:
    n_rows, n_cols = h.shape
    if n_rows < n_cols:
        raise RankDeficientError(
            f"{method} : N = {n_rows} < p = {n_cols}, H^T H singuliere (utiliser le solveur SVD)"
        )
    rank = int(np.linalg.matrix_rank(h))
    if rank < n_cols:
        raise RankDeficientError(
            f"{method} : rang(H) = {rank} < p = {n_cols}, H^T H singuliere "
            "(utiliser le solveur SVD)"
        )


# --------------------------------------------------------------------------
# Matrice de Vandermonde, prediction, residus
# --------------------------------------------------------------------------
def vandermonde(x, degree) -> np.ndarray:
    """Matrice de Vandermonde non centree H (N, d+1), H[i, j] = x_i ** j, j = 0..d.

    ``x`` : vecteur (N,) ou colonne (N, 1) de reels finis. La colonne 0 vaut 1 (terme
    constant, y compris pour x_i = 0). ``ValueError`` en cas de depassement flottant.
    """
    d = _check_degree(degree)
    a = _check_abscissas(x)
    with np.errstate(over="ignore", invalid="ignore"):
        h = a[:, None] ** np.arange(d + 1)
    if not np.isfinite(h).all():
        raise ValueError(f"depassement de capacite dans x ** j pour degree = {d}")
    return h


def _check_coef(coef) -> np.ndarray:
    c = _as_float_array(coef, "coef")
    if c.ndim != 1 or c.size == 0:
        raise ValueError(f"coef : vecteur 1D non vide attendu, recu {c.shape}")
    if not np.isfinite(c).all():
        raise ValueError("coef : valeurs non finies (NaN ou inf)")
    return c


def predict(coef, x) -> np.ndarray:
    """Prediction y_hat = H(x) @ coef, de forme (N,) ; le degre est ``len(coef) - 1``."""
    c = _check_coef(coef)
    return vandermonde(x, c.size - 1) @ c


def residuals(coef, x, y) -> np.ndarray:
    """Residus y - y_hat, de forme (N,) ; ``y`` doit etre un vecteur 1D de longueur N."""
    y_hat = predict(coef, x)
    b = _as_float_array(y, "y")
    if b.shape != y_hat.shape:
        raise AssertionError(
            f"formes divergentes : y.shape={b.shape} != {y_hat.shape} (broadcasting interdit)"
        )
    return b - y_hat


# --------------------------------------------------------------------------
# Solveurs
# --------------------------------------------------------------------------
def solve_normal_equations(h, y) -> np.ndarray:
    """Equations normales (H^T H) beta = H^T y (``np.linalg.solve``).

    Precision en O(kappa(H^T H) * eps) : instable des que kappa(H)^2 approche 1/eps.
    ``RankDeficientError`` si N < p ou rang(H) < p ; ``LinAlgError`` si la factorisation
    LU rencontre un pivot exactement nul.
    """
    a, b = _check_system(h, y)
    _require_full_column_rank(a, "equations normales")
    return np.linalg.solve(a.T @ a, a.T @ b)


def solve_qr(h, y) -> np.ndarray:
    """Factorisation QR economique H = QR puis R beta = Q^T y (substitution arriere).

    Precision en O(kappa(H) * eps). ``RankDeficientError`` si N < p ou rang(H) < p.
    """
    a, b = _check_system(h, y)
    _require_full_column_rank(a, "QR")
    q, r = linalg.qr(a, mode="economic")
    return linalg.solve_triangular(r, q.T @ b, lower=False)


def solve_svd(h, y) -> np.ndarray:
    """Moindres carres par SVD (``scipy.linalg.lstsq``, pilote ``gelsd``).

    Meme appel que ``LinearRegression`` : valeurs singulieres inferieures a
    ``max(N, p) * eps * sigma_max`` ignorees. Solution de norme minimale si H est de rang
    inferieur a p (N < p, abscisses repetees).
    """
    a, b = _check_system(h, y)
    cond = max(a.shape) * np.finfo(np.float64).eps
    return linalg.lstsq(a, b, cond=cond, lapack_driver="gelsd")[0]


def solve_sklearn_reference(h, y) -> np.ndarray:
    """Reference scikit-learn : ``LinearRegression(fit_intercept=False)`` sur H.

    La colonne de uns de H est le terme constant ; ``fit_intercept=False`` evite un
    second biais (EXG-PROG-B1-03). Meme algorithme que ``solve_svd``.
    """
    a, b = _check_system(h, y)
    return LinearRegression(fit_intercept=False).fit(a, b).coef_


SOLVERS = {"normal": solve_normal_equations, "qr": solve_qr, "svd": solve_svd}


def relative_error_inf(beta, beta_ref, eps0: float = EPS_0) -> float:
    """Ecart relatif en norme infinie ||beta - ref||_inf / max(||ref||_inf, eps0) (CDC 7.1).

    Renvoie ``inf`` si ``beta`` contient une valeur non finie.
    """
    b = _as_float_array(beta, "beta")
    r = _as_float_array(beta_ref, "beta_ref")
    if b.shape != r.shape or b.ndim != 1:
        raise AssertionError(f"formes divergentes ou non 1D : {b.shape} contre {r.shape}")
    if not np.isfinite(b).all():
        return float("inf")
    return float(np.max(np.abs(b - r)) / max(float(np.max(np.abs(r))), eps0))


# --------------------------------------------------------------------------
# Ajustement et comparaison
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PolynomialFit:
    """Resultat de ``fit_polynomial`` : coefficients (beta_0..beta_d) et diagnostics."""

    coef: np.ndarray
    degree: int
    method: str
    rank: int
    kappa_h: float
    kappa_hth: float

    def predict(self, x) -> np.ndarray:
        return predict(self.coef, x)


AUTO = "auto"  # selection automatique de fit_polynomial : QR si rang complet, sinon SVD


def _check_method(method) -> str:
    if method != AUTO and method not in SOLVERS:
        raise ValueError(f"methode inconnue : {method!r} (attendu parmi {METHODS + (AUTO,)})")
    return method


def fit_polynomial(x, y, degree, method: str = "svd") -> PolynomialFit:
    """Ajuste un polynome de degre ``degree`` par ``method``.

    ``method`` : 'normal', 'qr', 'svd' (defaut, solveur de reference) ou 'auto'. 'auto'
    bascule sur la SVD, seule methode definie pour N < p ou rang(H) < p, et utilise QR
    sinon ; ``PolynomialFit.method`` indique la methode effectivement employee.
    Le conditionnement est calcule et journalise (``condition_numbers`` : alerte
    WARNING si kappa(H^T H) > KAPPA_HTH_ALERT, sans blocage). ``RankDeficientError`` si
    'normal' ou 'qr' est demande explicitement sur un systeme de rang colonne incomplet.
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
    """Comparaison des solveurs sur un systeme (x, y, degre), protocole ARB-08.

    - ``regime`` : ``'strict'`` si rang(H) = p et kappa(H^T H) <= KAPPA_HTH_ALERT (la
      concordance a ``STRICT_RTOL`` est alors attendue), sinon ``'diagnostic'`` (ecart
      mesure et rapporte, jamais rejete).
    - ``coefficients`` : par methode ('normal', 'qr', 'svd', 'sklearn') ; ``None`` si la
      methode est non applicable (voir ``errors``).
    - ``rel_error_inf`` : critere de concordance de l'avenant 7.10 face a la SVD, pour
      'normal', 'qr' et 'sklearn' (``None`` si non calcule).
    - ``mixed_tolerance_ok`` : critere mixte par coefficient (atol 1e-10, rtol 1e-7),
      grandeur diagnostique.
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
        """Vrai si ``method`` a ete calculee et concorde avec la SVD a ``STRICT_RTOL``."""
        err = self.rel_error_inf.get(method)
        return err is not None and err <= STRICT_RTOL


def compare_solvers(x, y, degree) -> SolverComparison:
    """Compare equations normales, QR, SVD et ``LinearRegression`` sur (x, y, degre).

    Ne leve pas lorsque 'normal' ou 'qr' ne s'appliquent pas (N < p, rang deficient,
    pivot nul, resultat non fini) : la methode est qualifiee dans ``errors``. Journalise
    l'evenement ``solver_comparison`` (niveau INFO) ; l'alerte de conditionnement est
    emise par ``condition_numbers``.
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
        except np.linalg.LinAlgError as exc:  # inclut RankDeficientError
            coefficients[method], errors[method] = None, f"{type(exc).__name__}: {exc}"
            continue
        if not np.isfinite(coef).all():
            coefficients[method], errors[method] = None, "resultat non fini"
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
