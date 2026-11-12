"""Gestion reproductible des graines pseudo-aleatoires (etape 0, CDC section 10).

Principe : aucun etat aleatoire global n'est lu ni modifie. Chaque source
d'aleatoire recoit explicitement un generateur ``numpy.random.Generator``
(``make_rng``) ou un ``random_state`` entier valide (``validate_seed``, pour
scikit-learn). Une meme graine donne le meme flux, quel que soit l'etat global.
"""

import numpy as np

# Graines contractuelles (CDC section 10 ; avenant 7.1 pour SEEDS_STAT_1D).
SEED_GLOBAL = 42
SEED_POLYNOMIAL = 42  # base nominale et figures illustratives 1D
SEEDS_STAT_1D = tuple(range(1000, 1050))  # 50 graines de bruit, disjointes de 42
SEEDS_MONTE_CARLO = (42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021)  # ARB-05

_SEED_MAX = 2**32 - 1  # borne de random_state pour scikit-learn


def validate_seed(seed) -> int:
    """Valide une graine et la renvoie comme ``int`` Python.

    Accepte les entiers Python et NumPy dans [0, 2**32 - 1]. Rejette ``None``
    (tirage non reproductible), les booleens et les flottants (conversion
    silencieuse ou ambigue).
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"graine entiere attendue, recu {type(seed).__name__} : {seed!r}")
    value = int(seed)
    if not 0 <= value <= _SEED_MAX:
        raise ValueError(f"graine hors de [0, 2**32 - 1] : {value}")
    return value


def make_rng(seed) -> np.random.Generator:
    """Generateur independant ``np.random.default_rng(seed)``, sans effet de bord global."""
    return np.random.default_rng(validate_seed(seed))
