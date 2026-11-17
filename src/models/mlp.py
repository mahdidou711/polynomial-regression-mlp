"""2D MLP training and descriptive repetitions on the validation set.

This module loads no data and selects no model. The final test is reserved for a
separate evaluation after selection.
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
    """Explicit parameters; default lbfgs budgets.

    For the adam comparison, provide max_iter=5000, tol=1e-8,
    n_iter_no_change=200, and the chosen learning_rate_init/batch_size pair.
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
            raise ValueError("hidden_layer_sizes must be a non-empty tuple of positive integers")
        if self.activation not in ("tanh", "relu") or self.solver not in ("lbfgs", "adam"):
            raise ValueError("unsupported activation or solver")
        for name in ("alpha", "tol", "learning_rate_init"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
                raise ValueError(f"{name} must be real and finite")
            if value < 0 or (name == "learning_rate_init" and value == 0):
                raise ValueError(f"{name} out of range")
        for name in ("max_iter", "max_fun", "n_iter_no_change"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        b = self.batch_size
        if not (b == "auto" if isinstance(b, str) else isinstance(b, (int, np.integer)) and not isinstance(b, (bool, np.bool_)) and b > 0):
            raise ValueError("batch_size must be 'auto' or a positive integer")


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
        """Prediction on 2D coordinates; any final use is separate."""
        values = _x(x)
        prediction = np.asarray(self.model.predict(values))
        if prediction.shape != (len(values),) or not np.isfinite(prediction).all():
            raise ValueError("invalid predictions")
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
        raise TypeError("X must be real")
    if a.ndim != 2 or a.shape[1] != 2 or a.shape[0] == 0:
        raise ValueError("X must have shape (N, 2), N > 0")
    a = np.array(a, dtype=float, copy=True)
    if not np.isfinite(a).all():
        raise ValueError("X must be finite")
    return a


def _pair(x, y):
    x = _x(x)
    y = np.asarray(y)
    if y.dtype.kind not in "iuf":
        raise TypeError("y must be real")
    if y.shape != (len(x),):
        raise ValueError("y must have shape (N,) matching X")
    y = np.array(y, dtype=float, copy=True)
    if not np.isfinite(y).all():
        raise ValueError("y must be finite")
    if np.all(y == y[0]):
        raise ValueError("R² undefined for a constant target")
    return x, y


def _metrics(y, prediction):
    return {"mse": getMSE(y, prediction), "rmse": rmse(y, prediction),
            "r2": r2_score(y, prediction), "max_abs_error": max_abs_error(y, prediction)}


def train_mlp(config, seed, x_train, y_train, x_validation, y_validation, *, logger=None):
    """Fit a fresh model and measure training and validation only.

    A runtime error is propagated; no partial run is reported as a success.
    ConvergenceWarnings are kept, others are re-emitted.
    """
    if not isinstance(config, MLPConfig):
        raise TypeError("config must be an MLPConfig")
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
            raise ValueError("loss_curve_ inconsistent with n_iter_")
        curve = curve.tolist()
    # Descriptive signal, not a convergence diagnostic: at least two
    # slope reversals in the last ten values.
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
    log_event(logger or logging.getLogger("polynomial_mlp.mlp"), "mlp_run",
              level=logging.WARNING if convergence_warnings or limit else logging.INFO,
              seed=seed, parameters=result.parameters, versions=result.versions,
              diagnostics=diagnostics, training_metrics=result.training_metrics,
              validation_metrics=result.validation_metrics)
    return result


def run_multi_seed(config, x_train, y_train, x_validation, y_validation, *,
                   seeds=SEEDS_MONTE_CARLO, logger=None):
    """Independent runs; descriptive corrected standard deviation (None for a single run).

    Runs with a warning remain in the statistics. Any exception interrupts the
    protocol and is propagated; no failure is ignored.
    """
    seeds = tuple(validate_seed(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("non-empty and distinct seeds required")
    # Single copy so that the data stay the same even if the caller modifies them.
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
