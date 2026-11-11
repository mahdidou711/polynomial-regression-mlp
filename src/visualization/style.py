"""Charte graphique minimale et export unique des figures (etape 3, avenant 7.11).

``save_figure(fig, path_stem, vector=True)`` ecrit ``<path_stem>.pdf`` (vectoriel) et
``<path_stem>.png`` (300 DPI) ; PNG seul si ``vector=False`` (surfaces 3D). Les figures
sont construites avec ``matplotlib.figure.Figure`` (sans ``pyplot``) : aucun etat global,
aucun backend interactif, aucune figure conservee en memoire. L'etape 8 consolide ce
module sans redefinir cette interface : les exports sont deterministes (PDF sans date de
creation, AE7-07), deux exports successifs d'une meme figure sont identiques octet a octet.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

DPI = 300
CMAP = "coolwarm"  # colormap harmonisee du projet (CDC section 11)
FLOOR_LOG = 1e-16  # plancher d'affichage des echelles logarithmiques (avenant 7.9)

COLOR_TRAIN = colormaps[CMAP](0.05)  # bleu : apprentissage
COLOR_TEST = colormaps[CMAP](0.95)  # rouge : test
COLOR_TARGET = "0.15"  # gris fonce : fonction cible
COLOR_FIT = colormaps[CMAP](0.80)

# Metadonnees PDF : la date de creation est supprimee pour que deux exports successifs de la
# meme figure soient identiques octet a octet (AE3-04 / AE7-07).
PDF_METADATA = {"CreationDate": None}

RC = {
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,  # polices TrueType embarquees (inclusion LaTeX)
}


def style_context():
    """Contexte ``rc_context`` de la charte, a ouvrir autour de la construction d'une figure."""
    return mpl.rc_context(RC)


def new_figure(width: float = 6.4, height: float = 4.2, **kwargs) -> Figure:
    """Figure vierge a la charte du projet (``Figure`` sans pyplot)."""
    return Figure(figsize=(width, height), layout="constrained", **kwargs)


def floor_for_log(values, floor: float = FLOOR_LOG) -> np.ndarray:
    """Copie de ``values`` bornee inferieurement par ``floor`` pour un affichage log.

    Les valeurs stockees ne sont jamais modifiees : seule la copie tracee l'est.
    """
    return np.maximum(np.asarray(values, dtype=float), floor)


def save_figure(fig: Figure, path_stem, vector: bool = True) -> list[Path]:
    """Exporte ``fig`` en PDF vectoriel (si ``vector``) et PNG 300 DPI ; renvoie les chemins.

    ``path_stem`` : chemin sans extension (les repertoires parents sont crees). Un
    suffixe ``.pdf`` ou ``.png`` dans ``path_stem`` est refuse pour eviter un nom ambigu.
    """
    stem = Path(path_stem)
    if stem.suffix.lower() in (".pdf", ".png"):
        raise ValueError(f"path_stem doit etre sans extension : {stem}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with style_context():
        if vector:
            paths.append(stem.with_name(stem.name + ".pdf"))
            fig.savefig(paths[-1], format="pdf", metadata=PDF_METADATA)
        paths.append(stem.with_name(stem.name + ".png"))
        fig.savefig(paths[-1], format="png", dpi=DPI)
    return paths
