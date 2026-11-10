"""Journalisation structuree des runs et alertes (etape 0).

Deux sorties complementaires, pour des experiences relancees et comparees :
- console (stderr) : ligne lisible ``heure NIVEAU logger | evenement cle=valeur`` ;
- fichier optionnel au format JSON Lines strict (un objet JSON par ligne, ajout
  en fin de fichier), exploitable par ``json.loads`` ou pandas.

Utilisation :
    lg = get_logger("c0_tp2_mlp", log_file="results/run.jsonl")
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

DEFAULT_LOGGER_NAME = "c0_tp2_mlp"
_OWN_HANDLER_FLAG = "_c0_tp2_mlp_handler"  # marque les handlers geres par get_logger
_RESERVED_FIELDS = frozenset({"time", "level", "logger", "event"})


def _jsonable(value):
    """Convertit ``value`` en structure JSON stricte (pas de NaN/Infinity bruts)."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)  # 'inf', '-inf', 'nan'
    if value is None or isinstance(value, (str, int)):  # bool est un int
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
        if record.exc_info:  # AE0-03 : trace d'exception conservee dans le JSON Lines
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
    """Configure et renvoie le logger ``name`` (console + fichier JSON Lines optionnel).

    Idempotent : un nouvel appel remplace les handlers precedemment poses par
    cette fonction, sans toucher aux handlers etrangers. Le fichier est ouvert
    en ajout ; ses repertoires parents sont crees.
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
    """Journalise ``event`` avec des champs nommes (``cle=valeur`` / cles JSON).

    ``exc_info`` (comme pour ``Logger.log``) joint la trace d'une exception : ``True``
    dans un bloc ``except``, ou une instance d'exception. Le fichier JSON Lines la
    consigne sous la cle ``exc_info`` (chaine de la trace complete, AE0-03) ; la console
    l'affiche apres la ligne de l'evenement.
    """
    clash = _RESERVED_FIELDS.intersection(fields)
    if clash:
        raise ValueError(f"noms de champs reserves : {sorted(clash)}")
    logger.log(level, event, exc_info=exc_info, extra={"fields": fields})


def environment_info() -> dict:
    """Versions de Python et des bibliotheques scientifiques, pour tracer un run."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "scikit-learn", "matplotlib"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = "absent"
    return info
