"""Structured logging of runs and warnings.

Two complementary outputs, for experiments that are rerun and compared:
- console (stderr): readable line ``time LEVEL logger | event key=value``;
- optional file in strict JSON Lines format (one JSON object per line, appended
  at the end of the file), readable with ``json.loads`` or pandas.

Usage:
    lg = get_logger("polynomial_mlp", log_file="results/run.jsonl")
    log_event(lg, "run_start", seed=42, **environment_info())
    log_event(lg, "fit_done", degree=5, mse=0.25)
"""

import json
import logging
import math
import os
import platform
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

DEFAULT_LOGGER_NAME = "polynomial_mlp"
_OWN_HANDLER_FLAG = "_polynomial_mlp_handler"  # marks the handlers managed by get_logger
_RESERVED_FIELDS = frozenset({"time", "level", "logger", "event"})


def _jsonable(value):
    """Convert ``value`` to a strict JSON structure (no raw NaN/Infinity)."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)  # 'inf', '-inf', 'nan'
    if value is None or isinstance(value, (str, int)):  # bool is an int
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return repr(value)


class _JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(_jsonable(getattr(record, "fields", {})))
        if record.exc_info:  # exception trace kept in the JSON Lines output
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)


class _ConsoleFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        fields = _jsonable(getattr(record, "fields", {}))
        if fields:
            line += " " + " ".join(f"{k}={v}" for k, v in fields.items())
        return line


def get_logger(name=DEFAULT_LOGGER_NAME, log_file=None, level=logging.INFO) -> logging.Logger:
    """Configure and return the logger ``name`` (console + optional JSON Lines file).

    Idempotent: a new call replaces the handlers previously installed by this
    function, without touching foreign handlers. The file is opened in append
    mode; its parent directories are created.
    """
    logger = logging.getLogger(name)
    for handler in [h for h in logger.handlers if getattr(h, _OWN_HANDLER_FLAG, False)]:
        logger.removeHandler(handler)
        handler.close()

    handlers = []
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_ConsoleFormatter())
    handlers.append(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(_JsonLinesFormatter())
        handlers.append(file_handler)

    for handler in handlers:
        setattr(handler, _OWN_HANDLER_FLAG, True)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, exc_info=False, **fields
) -> None:
    """Log ``event`` with named fields (``key=value`` / JSON keys).

    ``exc_info`` (as for ``Logger.log``) attaches the trace of an exception: ``True``
    inside an ``except`` block, or an exception instance. The JSON Lines file
    stores it under the ``exc_info`` key (string of the full trace); the console
    prints it after the event line.
    """
    clash = _RESERVED_FIELDS.intersection(fields)
    if clash:
        raise ValueError(f"reserved field names: {sorted(clash)}")
    logger.log(level, event, exc_info=exc_info, extra={"fields": fields})


def environment_info() -> dict:
    """Versions of Python and of the scientific libraries, to trace a run."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "scikit-learn", "matplotlib"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = "absent"
    return info
