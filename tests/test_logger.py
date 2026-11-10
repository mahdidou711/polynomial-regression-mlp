"""Tests de src/utils/logger.py (etape 0).

Couvre : configuration idempotente, sortie console lisible, fichier JSON Lines
strict, conversion des types numpy et des valeurs non finies, niveaux,
metadonnees d'environnement.
Commande : python -m pytest tests/test_logger.py -v
"""

import json
import logging

import numpy as np
import pytest
import scipy
import sklearn

from src.utils.logger import environment_info, get_logger, log_event

NAME = "c0_tp2_mlp.test_logger"


@pytest.fixture
def logger_name():
    """Nom dedie ; handlers fermes et retires apres le test."""
    yield NAME
    lg = logging.getLogger(NAME)
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()


def read_jsonl(path):
    def refuse_constant(token):  # NaN / Infinity ne sont pas du JSON strict
        raise ValueError(f"constante non standard : {token}")

    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line, parse_constant=refuse_constant) for line in lines]


class TestGetLogger:
    def test_returns_named_logger(self, logger_name):
        lg = get_logger(logger_name)
        assert isinstance(lg, logging.Logger) and lg.name == NAME

    def test_is_idempotent_no_duplicate_handlers(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        n1 = len(get_logger(logger_name, log_file=f).handlers)
        n2 = len(get_logger(logger_name, log_file=f).handlers)
        assert n1 == n2 == 2  # console + fichier

    def test_no_file_handler_without_log_file(self, logger_name):
        assert len(get_logger(logger_name).handlers) == 1

    def test_reconfiguration_switches_file(self, logger_name, tmp_path):
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        log_event(get_logger(logger_name, log_file=a), "first")
        log_event(get_logger(logger_name, log_file=b), "second")
        assert [r["event"] for r in read_jsonl(a)] == ["first"]
        assert [r["event"] for r in read_jsonl(b)] == ["second"]

    def test_creates_parent_directories(self, logger_name, tmp_path):
        f = tmp_path / "deep" / "er" / "run.jsonl"
        log_event(get_logger(logger_name, log_file=f), "x")
        assert f.exists()

    def test_does_not_remove_foreign_handlers(self, logger_name):
        lg = logging.getLogger(NAME)
        foreign = logging.NullHandler()
        lg.addHandler(foreign)
        get_logger(logger_name)
        assert foreign in lg.handlers

    def test_level_filters_records(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f, level=logging.WARNING)
        log_event(lg, "quiet", level=logging.INFO)
        log_event(lg, "loud", level=logging.WARNING)
        assert [r["event"] for r in read_jsonl(f)] == ["loud"]


class TestLogEvent:
    def test_json_record_structure(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        log_event(lg, "fit_done", degree=5, mse=0.25, solver="svd")
        (rec,) = read_jsonl(f)
        assert rec["event"] == "fit_done"
        assert rec["level"] == "INFO"
        assert rec["logger"] == NAME
        assert rec["degree"] == 5 and rec["mse"] == 0.25 and rec["solver"] == "svd"
        assert rec["time"].endswith("+00:00")  # horodatage UTC

    def test_one_line_per_event(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        for i in range(3):
            log_event(lg, "tick", i=i)
        assert [r["i"] for r in read_jsonl(f)] == [0, 1, 2]

    def test_numpy_values_are_serialised(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        log_event(lg, "np", a=np.float64(1.5), b=np.int64(3), c=np.array([1.0, 2.0]))
        (rec,) = read_jsonl(f)
        assert rec["a"] == 1.5 and rec["b"] == 3 and rec["c"] == [1.0, 2.0]

    def test_non_finite_floats_become_strings_for_strict_json(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        log_event(lg, "kappa", k=float("inf"), m=float("nan"), arr=np.array([np.inf, 1.0]))
        (rec,) = read_jsonl(f)
        assert rec["k"] == "inf" and rec["m"] == "nan" and rec["arr"] == ["inf", 1.0]

    def test_unserialisable_value_falls_back_to_repr(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        log_event(lg, "obj", thing=object)
        (rec,) = read_jsonl(f)
        assert "object" in rec["thing"]

    def test_level_is_recorded(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        log_event(lg, "alert", level=logging.WARNING, kappa=1e12)
        assert read_jsonl(f)[0]["level"] == "WARNING"

    def test_reserved_field_name_time_is_rejected(self, logger_name):
        # 'logger', 'event' et 'level' sont deja des parametres de log_event
        # (TypeError natif) ; 'time' est la seule cle JSON qu'un champ pourrait ecraser.
        lg = get_logger(logger_name)
        with pytest.raises(ValueError, match="reserv"):
            log_event(lg, "bad", time=1)

    def test_console_output_is_human_readable(self, logger_name, capsys):
        lg = get_logger(logger_name)
        log_event(lg, "fit_done", degree=5)
        err = capsys.readouterr().err
        assert "fit_done" in err and "degree=5" in err and "INFO" in err

    def test_standard_logging_calls_still_work(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        lg.warning("plain message")
        (rec,) = read_jsonl(f)
        assert rec["event"] == "plain message" and rec["level"] == "WARNING"


class TestEnvironmentInfo:
    def test_reports_library_versions(self):
        info = environment_info()
        assert info["numpy"] == np.__version__
        assert info["scipy"] == scipy.__version__
        assert info["scikit-learn"] == sklearn.__version__
        assert info["python"].startswith("3.")

    def test_is_json_serialisable(self):
        json.dumps(environment_info())


class TestExceptionLogging:
    """AE0-03 : la trace d'une exception figure dans le JSON Lines."""

    def test_exc_info_true_records_traceback(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        try:
            raise ZeroDivisionError("boom 42")
        except ZeroDivisionError:
            log_event(lg, "run_failed", level=logging.ERROR, exc_info=True, seed=42)
        (rec,) = read_jsonl(f)
        assert rec["event"] == "run_failed" and rec["level"] == "ERROR" and rec["seed"] == 42
        assert rec["exc_info"].startswith("Traceback (most recent call last)")
        assert "ZeroDivisionError: boom 42" in rec["exc_info"]

    def test_exception_instance_is_accepted(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        err = ValueError("valeur invalide")
        log_event(lg, "bad", level=logging.ERROR, exc_info=err)
        assert "ValueError: valeur invalide" in read_jsonl(f)[0]["exc_info"]

    def test_no_exc_info_key_for_ordinary_events(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        log_event(get_logger(logger_name, log_file=f), "ok", x=1)
        assert "exc_info" not in read_jsonl(f)[0]

    def test_each_line_stays_one_valid_json_object(self, logger_name, tmp_path):
        f = tmp_path / "run.jsonl"
        lg = get_logger(logger_name, log_file=f)
        try:
            raise RuntimeError("multi\nligne")
        except RuntimeError:
            log_event(lg, "failed", exc_info=True)
        log_event(lg, "after")
        assert len(f.read_text(encoding="utf-8").splitlines()) == 2  # trace multi-ligne echappee
        assert [r["event"] for r in read_jsonl(f)] == ["failed", "after"]

    def test_console_shows_traceback(self, logger_name, capsys):
        lg = get_logger(logger_name)
        try:
            raise KeyError("cle")
        except KeyError:
            log_event(lg, "failed", exc_info=True)
        assert "KeyError" in capsys.readouterr().err

    def test_exc_info_is_not_usable_as_a_condition_number_context(self):
        from src.utils.metrics import condition_numbers

        with pytest.raises(ValueError, match="exc_info"):
            condition_numbers(np.eye(3), exc_info=1)
