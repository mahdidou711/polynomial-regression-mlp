"""Deterministic reading and writing of the CSV files in results/tables/."""

import csv
import math
from pathlib import Path

MISSING = {"", "nan", "NaN", "None", "none"}


def read_csv(path) -> list[dict]:
    """Rows of a CSV file (values as strings; typing is left to the caller)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, header, rows) -> Path:
    """Deterministic CSV: ``repr`` of floats (exact round trip), ``\\n`` line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(row.get(k)) for k in header])
    return path


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return value


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return str(value).strip() in MISSING
