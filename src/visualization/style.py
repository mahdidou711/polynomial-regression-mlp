"""Minimal graphical style and single export path for the figures.

``save_figure(fig, path_stem, vector=True)`` writes ``<path_stem>.pdf`` (vector) and
``<path_stem>.png`` (300 DPI); PNG only if ``vector=False`` (3D surfaces). Figures
are built with ``matplotlib.figure.Figure`` (no ``pyplot``): no global state,
no interactive backend, no figure kept in memory. Exports are deterministic (PDF without
creation date), two successive exports of the same figure are byte-identical.

The palette, fonts and sizes are applied here, in ``save_figure``: the figures written in
``results/`` are the final ones.
"""

import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

DPI = 300
FLOOR_LOG = 1e-16  # display floor of the logarithmic scales

# Project palette.
PLUM = "#63003C"
ANTHRACITE = "#313E48"
LIGHT_GRAY = "#D6DBDF"
RED = "#C60B46"
BLUE = "#0E87C9"
DARK_BLUE = "#004E7D"
TEAL = "#00AFAA"
ORANGE = "#F37320"
PURPLE = "#7C2A90"
LILAC = "#B9B1D7"
SEQUENTIAL = [LIGHT_GRAY, LILAC, PURPLE, PLUM]  # monotone luminance scale
CYCLE = [BLUE, PLUM, ORANGE, TEAL, PURPLE, ANTHRACITE, RED, DARK_BLUE]

CMAP = "house_seq"  # project colormap
colormaps.register(LinearSegmentedColormap.from_list(CMAP, SEQUENTIAL, N=256), name=CMAP, force=True)

COLOR_TRAIN = BLUE  # training
COLOR_TEST = PLUM  # test
COLOR_TARGET = ANTHRACITE  # target function
COLOR_FIT = PLUM

# PDF metadata: the creation date is removed so that two successive exports of the
# same figure are byte-identical.
PDF_METADATA = {"CreationDate": None}

RC = {
    "font.family": ["DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": LIGHT_GRAY,
    "text.color": ANTHRACITE,
    "axes.labelcolor": ANTHRACITE,
    "axes.edgecolor": ANTHRACITE,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": ANTHRACITE,
    "ytick.color": ANTHRACITE,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.color": LIGHT_GRAY,
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler(color=CYCLE),
    "axes.formatter.use_mathtext": True,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,  # embedded TrueType fonts
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes3d.grid": True,
    "axes3d.xaxis.panecolor": (0.97, 0.97, 0.98, 1.0),
    "axes3d.yaxis.panecolor": (0.97, 0.97, 0.98, 1.0),
    "axes3d.zaxis.panecolor": (0.97, 0.97, 0.98, 1.0),
}

# Final sizes (width, height) in inches, so that the figures can be included at scale 1:1.
# in a document with a 6.1 inch text width.
_W = 6.1
FIGURE_SIZES = {
    "nominal_1d_conditioning": (_W, 2.55), "nominal_1d_error_profile": (_W, 2.8),
    "nominal_1d_fits": (_W, 5.1), "nominal_1d_parity": (_W, 2.6),
    "nominal_1d_campaign_50_seeds": (_W, 2.6), "exploration_n_samples": (_W, 2.6),
    "exploration_sigma": (_W, 2.6), "exploration_extrapolation": (_W, 2.6),
    "exploration_n_samples_seed42": (5.0, 3.0), "exploration_sigma_seed42": (5.0, 3.0),
    "mlp_2d_width": (5.0, 3.1), "mlp_2d_activation_tanh_relu": (_W, 2.7),
    "mlp_2d_panel_3d": (_W, 2.5), "mlp_2d_parity": (_W, 3.0), "mlp_2d_residual_map": (4.9, 4.2),
    "mlp_2d_convergence": (_W, 3.6), "mlp_2d_adam_curves": (5.2, 3.3),
    "mlp_2d_sensitivities": (_W, 3.8),
}
# 3D panels: manual layout (3D ticks are not handled by the constrained layout).
BOARDS = {"mlp_2d_panel_3d": ([0.0, 0.333, 0.667], 0.25, [0.262, 0.595, 0.929]),
          "mlp_2d_activation_tanh_relu": ([0.0, 0.42], 0.40, [0.88])}
MIN_FONT = 6.0


def style_context():
    """``rc_context`` of the style, to open around the construction of a figure."""
    return mpl.rc_context(RC)


def new_figure(width: float = 6.4, height: float = 4.2, **kwargs) -> Figure:
    """Blank figure with the project style (``Figure`` without pyplot)."""
    return Figure(figsize=(width, height), layout="constrained", **kwargs)


def floor_for_log(values, floor: float = FLOOR_LOG) -> np.ndarray:
    """Copy of ``values`` bounded below by ``floor`` for a log display.

    The stored values are never modified: only the plotted copy is.
    """
    return np.maximum(np.asarray(values, dtype=float), floor)


def _math_labels(text: str) -> str:
    """Typeset coordinate names, residual symbols and R2 as mathematics in a figure text."""
    if text in ("x1", "x2"):
        return f"$x_{text[1]}$"
    if re.match(r"^(adam|lbfgs)-", text):  # configuration identifiers are kept as they are
        return text
    text = re.sub(r"(?<![\d.])1e-16(?![\d])", "$10^{-16}$", text)
    text = re.sub(r"(?<![\d.])1e\+07(?![\d])", "$10^{7}$", text)
    for old, new in (("|y_pred - y|", "$|\\hat{y}-y|$"), ("(R2 =", "($R^2$ =")):
        text = text.replace(old, new)
    return text


def _texts(fig):
    """Texts of titles, axes, annotations and legends (not the tick labels)."""
    yield from fig.texts
    for ax in fig.axes:
        yield from (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label)
        if hasattr(ax, "zaxis"):
            yield ax.zaxis.label
        yield from ax.texts
        legend = ax.get_legend()
        if legend is not None:
            yield from legend.get_texts()
    for legend in fig.legends:
        yield from legend.get_texts()


def _layout_board(fig, spec) -> None:
    lefts, width, cbar_x = spec
    axes3d = [ax for ax in fig.axes if hasattr(ax, "zaxis")]
    cbars = [ax for ax in fig.axes if not hasattr(ax, "zaxis")]
    fig.set_layout_engine("none")
    for ax, left in zip(axes3d, lefts):
        ax.set_position([left, 0.1, width, 0.78])
        ax.set_box_aspect((1, 1, 0.8), zoom=0.95)
    for cb, x in zip(cbars, cbar_x):
        cb.set_position([x, 0.2, 0.012, 0.52])


def apply_house_style(fig: Figure, stem: str) -> None:
    """Apply to ``fig`` the math labels, alert colour, minimal font size and final dimensions.

    The plotted data are never modified; the operation is idempotent.
    """
    for t in _texts(fig):
        text = t.get_text()
        if text:
            t.set_text(_math_labels(text))
    for ax in fig.axes:
        flagged = any("out of frame" in t.get_text() for t in ax.texts)
        if flagged:
            for t in ax.texts:
                t.set_color(RED)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(RED)
    axes3d = [ax for ax in fig.axes if hasattr(ax, "zaxis")]
    for ax in axes3d:
        ax.set_box_aspect((1, 1, 0.8), zoom=1.3)  # then adjusted by _layout_board for the panels
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(pad=-2)
        ax.xaxis.labelpad = -6
        ax.yaxis.labelpad = -6
    if len(axes3d) > 1:  # panels: the colour bar carries the value scale
        for ax in axes3d:
            ax.set_zticklabels([])
            ax.set_zlabel("")
    for t in fig.findobj(mpl.text.Text):
        if t.get_fontsize() < MIN_FONT:
            t.set_fontsize(MIN_FONT)
    if stem in FIGURE_SIZES:
        fig.set_size_inches(*FIGURE_SIZES[stem])
    if stem in BOARDS:
        _layout_board(fig, BOARDS[stem])


def save_figure(fig: Figure, path_stem, vector: bool = True) -> list[Path]:
    """Export ``fig`` as vector PDF (if ``vector``) and 300 DPI PNG; return the paths.

    ``path_stem``: path without extension (parent directories are created); the style is
    applied to ``fig`` before writing (``apply_house_style``). A ``.pdf`` or ``.png``
    suffix in ``path_stem`` is refused to avoid an ambiguous name.
    """
    stem = Path(path_stem)
    if stem.suffix.lower() in (".pdf", ".png"):
        raise ValueError(f"path_stem must be without extension: {stem}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    apply_house_style(fig, stem.name)
    paths = []
    with style_context():
        if vector:
            paths.append(stem.with_name(stem.name + ".pdf"))
            fig.savefig(paths[-1], format="pdf", metadata=PDF_METADATA)
        paths.append(stem.with_name(stem.name + ".png"))
        fig.savefig(paths[-1], format="png", dpi=DPI)
    return paths
