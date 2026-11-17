"""Loading of the reference 2D input file, without synthetic substitution."""

import hashlib
import io
from pathlib import Path

import numpy as np

TRAINING_SHA256 = "997facd8edf592eb0f2d31eeffb84d88923dce04ace906f3a93f12067af1ad9a"
DEFAULT_TRAINING_PATH = Path(__file__).resolve().parents[2] / "data/raw/sinc_dim2_input.csv"


def load_training_data_2d(path=DEFAULT_TRAINING_PATH) -> np.ndarray:
    """Read the 200 input pairs in file order; check the bytes and the content.

    ``path`` can point to an identical copy of the reference file. A missing
    file raises FileNotFoundError; an invalid identity or content raises
    ValueError. The hash and the parsing use the same bytes.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAINING_SHA256:
        raise ValueError("incorrect SHA-256 of the input file")
    try:
        content = raw.decode("utf-8")
        lines = content.splitlines()
        if not lines or lines[0] != "x1;x2" or len(lines) != 201:
            raise ValueError("incorrect header or number of rows")
        x = np.loadtxt(io.StringIO(content), delimiter=";", skiprows=1, ndmin=2)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"malformed training CSV: {exc}") from exc
    if x.shape != (200, 2):
        raise ValueError(f"expected shape (200, 2), got {x.shape}")
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("non-finite coordinates or outside [0, 1]")
    if np.unique(x, axis=0).shape[0] != 200:
        raise ValueError("duplicated training points")
    return x
