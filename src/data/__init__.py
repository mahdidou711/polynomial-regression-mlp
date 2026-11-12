"""Generation des donnees (etape 1 : 1D)."""

from .generator_1d import (
    N_TEST_NOMINAL,
    NB_EX_NOMINAL,
    SIGMA_NOMINAL,
    XMAX_NOMINAL,
    XMIN_NOMINAL,
    fonction_cible,
    genere_base_apprentissage,
    genere_base_test,
    genere_exemple_dim1,
)

__all__ = [
    "N_TEST_NOMINAL",
    "NB_EX_NOMINAL",
    "SIGMA_NOMINAL",
    "XMAX_NOMINAL",
    "XMIN_NOMINAL",
    "fonction_cible",
    "genere_base_apprentissage",
    "genere_base_test",
    "genere_exemple_dim1",
]
