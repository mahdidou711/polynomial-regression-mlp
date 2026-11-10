"""Donnees 2D sans bruit : apprentissage, validation et test final.

Chaque base est un couple (X, y), de formes respectives (N, 2) et (N,).
La validation sert a choisir la configuration ; le test est reserve a son
estimation finale, apres selection. Les grilles parcourent x2 le plus vite.
"""

from typing import NamedTuple

import numpy as np

from src.data.data_loader import DEFAULT_TRAINING_PATH, load_training_data_2d


class DataSplits2D(NamedTuple):
    training: tuple[np.ndarray, np.ndarray]
    validation: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]


def sinus_cardinal(x) -> np.ndarray:
    """Cible de la diapositive 17, vectorisee sur un tableau (N, 2).

    z = A(-pi + 2*pi*x + b), h = ||z|| ; 1 si h < 0.001,
    sin(h)/h sinon. Retourne un vecteur de forme (N,), sans bruit.
    """
    values = np.asarray(x)
    if values.dtype.kind not in "iuf":
        raise TypeError("coordonnees reelles numeriques attendues")
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("coordonnees de forme (N, 2), N > 0, attendues")
    values = values.astype(np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("coordonnees non finies ou hors de [0, 1]")
    scaled = -np.pi + 2 * values * np.pi
    z = (scaled + np.array([0.2, -0.3])) @ np.array([[1.0, -2.0], [1.0, 1.0]])
    h = np.linalg.norm(z, axis=1)
    y = np.ones_like(h)
    mask = h >= 1e-3
    y[mask] = np.sin(h[mask]) / h[mask]
    return y


def _grid(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1, x2 = np.meshgrid(axis, axis, indexing="ij")
    x = np.column_stack((x1.ravel(), x2.ravel()))
    return x, sinus_cardinal(x)


def genere_base_validation_2d() -> tuple[np.ndarray, np.ndarray]:
    """400 centres de cellules ; choix d'hyperparametres."""
    return _grid((np.arange(20, dtype=float) + 0.5) / 20)


def genere_base_test_2d() -> tuple[np.ndarray, np.ndarray]:
    """1600 points, bornes incluses ; evaluation finale uniquement."""
    return _grid(np.linspace(0.0, 1.0, 40))


def genere_base_apprentissage_2d(path=DEFAULT_TRAINING_PATH) -> tuple[np.ndarray, np.ndarray]:
    """200 observations officielles dans l'ordre du fichier ; ajustement du MLP."""
    x = load_training_data_2d(path)
    return x, sinus_cardinal(x)


def prepare_data_2d(path=DEFAULT_TRAINING_PATH) -> DataSplits2D:
    """Construit les trois bases identifiees, sans tirage aleatoire."""
    return DataSplits2D(genere_base_apprentissage_2d(path),
                        genere_base_validation_2d(), genere_base_test_2d())
