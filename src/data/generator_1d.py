"""Generation et echantillonnage des donnees 1D (etape 1, plan 04 etape 1).

Fonction cible officielle (enonce 2026-2027, diapositive 9) :
    f(x) = sin(-pi + 2*pi*x) = -sin(2*pi*x)

Convention d'echantillonnage de l'enseignant : ``np.arange(xmin, xmax,
(xmax - xmin) / NbEx)``, soit NbEx points equidistants sur l'intervalle
semi-ouvert [xmin, xmax) (borne haute EXCLUE, pas = (xmax - xmin) / NbEx).
Ici ``np.linspace(xmin, xmax, NbEx, endpoint=False)`` produit les memes
abscisses (meme formule xmin + k * pas) avec un nombre de points garanti egal
a NbEx (EXG-PROG-B2-02) ; l'equivalence est verifiee par les tests.

Le bruit gaussien utilise un ``numpy.random.Generator`` local (``make_rng``) :
l'etat aleatoire global n'est ni lu ni modifie.
"""

import numbers

import numpy as np

from src.utils.seeding import SEED_GLOBAL, make_rng

# Parametres nominaux (CDC, plan 04 etape 1).
XMIN_NOMINAL = 0.0
XMAX_NOMINAL = 1.2
NB_EX_NOMINAL = 20
SIGMA_NOMINAL = 0.2
N_TEST_NOMINAL = 200


def fonction_cible(x):
    """Fonction cible officielle ``sin(-pi + 2*pi*x)`` (sans bruit)."""
    return np.sin(-np.pi + 2 * np.asarray(x, dtype=float) * np.pi)


def _check_real(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} : reel attendu, recu {type(value).__name__} : {value!r}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} doit etre fini : {value}")
    return value


def _check_count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} : entier attendu, recu {type(value).__name__} : {value!r}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} doit etre >= 1 : {value}")
    return value


def genere_exemple_dim1(xmin, xmax, NbEx, sigma, seed=SEED_GLOBAL):
    """Base 1D bruitee, signature pedagogique + graine explicite.

    Renvoie ``(x, y)`` avec ``x`` de forme (NbEx, 1) (abscisses de l'enseignant,
    intervalle [xmin, xmax)) et ``y`` de forme (NbEx,) = f(x) + N(0, sigma^2).
    ``seed`` (defaut 42, graine nominale) pilote un generateur local ; il est
    valide meme si ``sigma == 0``.
    """
    xmin = _check_real(xmin, "xmin")
    xmax = _check_real(xmax, "xmax")
    nb_ex = _check_count(NbEx, "NbEx")
    sigma = _check_real(sigma, "sigma")
    if not xmin < xmax:
        raise ValueError(f"domaine vide ou inverse : xmin={xmin}, xmax={xmax}")
    if sigma < 0:
        raise ValueError(f"sigma doit etre >= 0 : {sigma}")
    rng = make_rng(seed)

    x = np.linspace(xmin, xmax, nb_ex, endpoint=False)
    assert x.shape == (nb_ex,)  # EXG-PROG-B2-02
    y = fonction_cible(x) + rng.normal(loc=0.0, scale=sigma, size=x.size)
    return x.reshape(-1, 1), y


def genere_base_apprentissage(nb_ex=NB_EX_NOMINAL, sigma=SIGMA_NOMINAL, seed=SEED_GLOBAL,
                              xmin=XMIN_NOMINAL, xmax=XMAX_NOMINAL):
    """Base d'apprentissage (nominal : 20 points, sigma = 0,2, graine 42)."""
    return genere_exemple_dim1(xmin, xmax, nb_ex, sigma, seed=seed)


def genere_base_test(n_test=N_TEST_NOMINAL, xmin=XMIN_NOMINAL, xmax=XMAX_NOMINAL):
    """Base de test sans bruit (nominal : 200 points sur [0 ; 1,2))."""
    return genere_exemple_dim1(xmin, xmax, n_test, 0.0)
