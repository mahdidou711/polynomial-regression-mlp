"""Chargement du fichier 2D de reference, sans substitution synthetique."""

import hashlib
import io
from pathlib import Path

import numpy as np

TRAINING_SHA256 = "997facd8edf592eb0f2d31eeffb84d88923dce04ace906f3a93f12067af1ad9a"
DEFAULT_TRAINING_PATH = Path(__file__).resolve().parents[2] / "data/raw/sinc_dim2_input.csv"


def load_training_data_2d(path=DEFAULT_TRAINING_PATH) -> np.ndarray:
    """Lit les 200 couples officiels dans leur ordre ; verifie les octets et le contenu.

    ``path`` permet de localiser une copie identique du fichier de reference.
    Une absence leve FileNotFoundError ; une identite ou un contenu invalide,
    ValueError. Le calcul du hash et le parsing portent sur les memes octets.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAINING_SHA256:
        raise ValueError("SHA-256 du fichier d'apprentissage incorrect")
    try:
        content = raw.decode("utf-8")
        lines = content.splitlines()
        if not lines or lines[0] != "x1;x2" or len(lines) != 201:
            raise ValueError("en-tete ou nombre de lignes incorrect")
        x = np.loadtxt(io.StringIO(content), delimiter=";", skiprows=1, ndmin=2)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"CSV d'apprentissage mal forme : {exc}") from exc
    if x.shape != (200, 2):
        raise ValueError(f"forme attendue (200, 2), recue {x.shape}")
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("coordonnees non finies ou hors de [0, 1]")
    if np.unique(x, axis=0).shape[0] != 200:
        raise ValueError("points d'apprentissage dupliques")
    return x
