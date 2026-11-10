"""Entrainement MLP 2D et repetitions descriptives sur la validation.

Ce module ne charge aucune donnee et ne choisit aucun modele. Le test final est
reserve a une evaluation distincte apres selection (DEC-20).
"""

from dataclasses import asdict, dataclass
import logging
from time import perf_counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from src.utils.logger import environment_info, log_event
from src.utils.metrics import getMSE, max_abs_error, r2_score, rmse
from src.utils.seeding import SEEDS_MONTE_CARLO, validate_seed


@dataclass(frozen=True)
class MLPConfig:
    """Parametres explicites ; budgets lbfgs de DEC-18 par defaut.

    Pour la comparaison adam de DEC-19, fournir max_iter=5000, tol=1e-8,
    n_iter_no_change=200, et le couple learning_rate_init/batch_size choisi.
    """
    hidden_layer_sizes: tuple[int, ...] = (20,)
    activation: str = "tanh"
    solver: str = "lbfgs"
    alpha: float = 1e-4
    max_iter: int = 10000
    max_fun: int = 100000
    tol: float = 1e-6
    learning_rate_init: float = 1e-3
    batch_size: int | str = "auto"
    n_iter_no_change: int = 10

    def __post_init__(self):
        layers = self.hidden_layer_sizes
        if not isinstance(layers, tuple) or not layers or any(
            isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0
            for n in layers
        ):
            raise ValueError("hidden_layer_sizes doit etre un tuple non vide d'entiers positifs")
        if self.activation not in ("tanh", "relu") or self.solver not in ("lbfgs", "adam"):
            raise ValueError("activation ou solveur non pris en charge")
        for name in ("alpha", "tol", "learning_rate_init"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
                raise ValueError(f"{name} doit etre reel et fini")
            if value < 0 or (name == "learning_rate_init" and value == 0):
                raise ValueError(f"{name} hors domaine")
        for name in ("max_iter", "max_fun", "n_iter_no_change"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} doit etre un entier positif")
        b = self.batch_size
        if not (b == "auto" if isinstance(b, str) else isinstance(b, (int, np.integer)) and not isinstance(b, (bool, np.bool_)) and b > 0):
            raise ValueError("batch_size doit etre 'auto' ou un entier positif")


@dataclass
class RunResult:
    seed: int
    model: MLPRegressor
    parameters: dict
    versions: dict
    training_metrics: dict
    validation_metrics: dict
    diagnostics: dict

    def predict(self, x):
        """Prediction sur des coordonnees 2D ; tout usage final est distinct."""
        values = _x(x)
        prediction = np.asarray(self.model.predict(values))
        if prediction.shape != (len(values),) or not np.isfinite(prediction).all():
            raise ValueError("predictions invalides")
        return prediction


@dataclass
class MultiSeedResult:
    runs: list[RunResult]
    statistics: dict
    successful_count: int
    warning_count: int


def _x(x):
    a = np.asarray(x)
    if a.dtype.kind not in "iuf":
        raise TypeError("X doit etre reel")
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        raise ValueError("X doit avoir la forme (N, 2), N > 0")
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all():
        raise ValueError("X doit etre fini")
    return a


def _pair(x, y):
    x = _x(x)
    y = np.asarray(y)
    if y.dtype.kind not in "iuf":
        raise TypeError("y doit etre reel")
    if y.shape != (len(x),):
        raise ValueError("y doit avoir la forme (N,) correspondant a X")
    y = np.array(y, dtype=float, copy=True)
    if not np.isfinite(y).all():
        raise ValueError("y doit etre fini")
    if np.all(y == y[0]):
        raise ValueError("R² indefini pour une cible constante")
    return x, y


def _metrics(y, prediction):
    return {"mse": getMSE(y, prediction), "rmse": rmse(y, prediction),
            "r2": r2_score(y, prediction), "max_abs_error": max_abs_error(y, prediction)}


def train_mlp(config, seed, x_train, y_train, x_validation, y_validation, *, logger=None):
    """Ajuste un modele neuf et mesure apprentissage et validation uniquement.

    Une erreur d'execution est propagee ; aucun run partiel n'est presente comme
    une reussite. Les ConvergenceWarning sont conserves, les autres reemis.
    """
    if not isinstance(config, MLPConfig):
        raise TypeError("config doit etre MLPConfig")
    seed = validate_seed(seed)
    xt, yt = _pair(x_train, y_train)
    xv, yv = _pair(x_validation, y_validation)
    model = MLPRegressor(**asdict(config), random_state=seed, early_stopping=False,
                         shuffle=True, learning_rate="constant", beta_1=0.9, beta_2=0.999)
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(xt, yt)
    elapsed = perf_counter() - start
    convergence_warnings = []
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            convergence_warnings.append(str(warning.message))
        else:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    curve = getattr(model, "loss_curve_", None)
    if curve is not None:
        curve = np.asarray(curve, dtype=float)
        if curve.ndim != 1 or len(curve) != model.n_iter_ or not np.isfinite(curve).all():
            raise ValueError("loss_curve_ incoherente avec n_iter_")
        curve = curve.tolist()
    # Signal descriptif, pas un diagnostic de convergence : au moins deux
    # inversions de pente dans les dix dernieres valeurs.
    tail = curve[-10:] if curve is not None else None
    diffs = np.diff(tail) if tail is not None else np.array([])
    signs = np.sign(diffs[diffs != 0])
    oscillation = bool(np.count_nonzero(signs[1:] != signs[:-1]) >= 2) if len(signs) else False
    limit = model.n_iter_ >= config.max_iter
    status = "warning" if convergence_warnings else ("iteration_limit" if limit else "stopped_unverified")
    diagnostics = {"status": status, "n_iter": int(model.n_iter_), "loss": float(model.loss_),
                   "loss_curve": curve, "loss_curve_available": curve is not None,
                   "tail_loss": tail, "tail_oscillation": oscillation,
                   "iteration_limit_reached": limit, "convergence_warnings": convergence_warnings,
                   "elapsed_seconds": elapsed}
    result = RunResult(seed, model, model.get_params(deep=False), environment_info(), {}, {}, diagnostics)
    result.training_metrics = _metrics(yt, result.predict(xt))
    result.validation_metrics = _metrics(yv, result.predict(xv))
    log_event(logger or logging.getLogger("c0_tp2_mlp.mlp"), "mlp_run",
              level=logging.WARNING if convergence_warnings or limit else logging.INFO,
              seed=seed, parameters=result.parameters, versions=result.versions,
              diagnostics=diagnostics, training_metrics=result.training_metrics,
              validation_metrics=result.validation_metrics)
    return result


def run_multi_seed(config, x_train, y_train, x_validation, y_validation, *,
                   seeds=SEEDS_MONTE_CARLO, logger=None):
    """Runs independants ; ecart-type corrige descriptif (None si un seul run).

    Les runs avec avertissement restent dans les statistiques. Toute exception
    interrompt le protocole et est propagee ; aucun echec n'est ignore.
    """
    seeds = tuple(validate_seed(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("graines non vides et distinctes requises")
    # Copie unique pour preserver les memes donnees meme si l'appelant les modifie.
    xt, yt = _pair(x_train, y_train)
    xv, yv = _pair(x_validation, y_validation)
    runs = [train_mlp(config, seed, xt, yt, xv, yv, logger=logger) for seed in seeds]
    stats = {}
    for split in ("training", "validation"):
        stats[split] = {}
        for metric in ("mse", "rmse", "r2", "max_abs_error"):
            values = np.array([getattr(run, split + "_metrics")[metric] for run in runs])
            stats[split][metric] = {"mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "min": float(np.min(values)), "max": float(np.max(values)),
                "median": float(np.median(values))}
    return MultiSeedResult(runs, stats, len(runs), sum(bool(r.diagnostics["convergence_warnings"]) for r in runs))
