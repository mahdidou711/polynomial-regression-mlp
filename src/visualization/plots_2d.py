"""2D MLP figures: 3D surfaces, residual map, parity, campaigns, convergence.

Orientation: a typical ``meshgrid(x1v, x2v)`` (indexing 'xy', x1 along columns) filled
as ``R[i, j]`` with x1 along rows gives a transposed surface. Here,
``X1, X2 = meshgrid(axis, axis, indexing='ij')`` and ``Z = f(points).reshape`` give
``Z[i, j] = f(x1_i, x2_j)`` in a single vectorised prediction; the triplet
(X1, X2, Z) passed to ``plot_surface`` is consistent. No figure reads the test set: the
data arrive as arguments from ``final_evaluation_2d`` (after freeze and verification).

3D surfaces: 300 DPI PNG only; 2D figures: PDF + PNG (``save_figure``).
Command: python -m src.visualization.plots_2d (after the final evaluation).
"""

import argparse
from pathlib import Path

import numpy as np

from src.data.generator_2d import cardinal_sine
from src.experiments.runner_2d_nominal import FIGURES_DIR, STOP_CAUSES, TABLES_DIR
from src.utils.metrics import r2_score
from src.visualization.style import (CMAP, COLOR_FIT, COLOR_TEST, COLOR_TRAIN, FLOOR_LOG,
                                     floor_for_log, new_figure, save_figure, style_context)

RENDER_STEP = 0.005  # 200 x 200 = 40,000 points
ELEV, AZIM = 30, -60  # default 3D view angles


def render_axis(step: float = RENDER_STEP) -> np.ndarray:
    """Rendering axis: bit-for-bit identical to ``np.arange(0, 1, 0.005)`` (200 points, bound 1 excluded)."""
    return np.linspace(0.0, 1.0, int(round(1.0 / step)), endpoint=False)


def surface_grid(func, step: float = RENDER_STEP):
    """(X1, X2, Z) with ``Z[i, j] = f(x1_i, x2_j)``; ``func`` receives an (N, 2) array."""
    axis = render_axis(step)
    x1, x2 = np.meshgrid(axis, axis, indexing="ij")
    z = np.asarray(func(np.column_stack((x1.ravel(), x2.ravel())))).reshape(x1.shape)
    return x1, x2, z


def plot_surface_3d(ax, x1, x2, z, title, vmin=None, vmax=None):
    """Surface (project colormap), antialiasing on, view (30, -60), axes x1 then x2 labelled."""
    surf = ax.plot_surface(x1, x2, z, cmap=CMAP, linewidth=0, antialiased=True,
                           vmin=vmin, vmax=vmax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.view_init(elev=ELEV, azim=AZIM)
    return surf


def _abs_residual(predict, step):
    x1, x2, exact = surface_grid(cardinal_sine, step)
    _, _, pred = surface_grid(predict, step)
    return x1, x2, exact, pred, np.abs(pred - exact)


def surface_figure(x1, x2, z, title):
    with style_context():
        fig = new_figure(5.2, 4.4)
        ax = fig.add_subplot(projection="3d")
        surf = plot_surface_3d(ax, x1, x2, z, title)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    return fig


def surfaces_board(predict, step: float = RENDER_STEP):
    """Panel: exact surface, predicted surface, |residual|."""
    x1, x2, exact, pred, res = _abs_residual(predict, step)
    with style_context():
        fig = new_figure(14.0, 4.6)
        for k, (z, title) in enumerate([(exact, "Exact surface"), (pred, "Predicted surface"),
                                         (res, "Absolute residual |y_pred - y|")]):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            surf = plot_surface_3d(ax, x1, x2, z, title)
            fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    return fig


def residual_map_2d(predict, step: float = RENDER_STEP):
    """2D map of |residual| (pcolormesh, rasterised) with contours of the exact surface."""
    x1, x2, exact, _, res = _abs_residual(predict, step)
    with style_context():
        fig = new_figure(5.4, 4.6)
        ax = fig.add_subplot()
        mesh = ax.pcolormesh(x1, x2, res, cmap=CMAP, shading="nearest", rasterized=True)
        ax.contour(x1, x2, exact, levels=8, colors="0.15", linewidths=0.6)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
        ax.set_title("Absolute residual |y_pred - y| (contours: exact surface)")
        fig.colorbar(mesh, ax=ax, label="|y_pred - y|")
    return fig


def activation_figure(predict_tanh, predict_relu, step: float = RENDER_STEP):
    x1, x2, _, pred_t, _ = _abs_residual(predict_tanh, step)
    _, _, _, pred_r, _ = _abs_residual(predict_relu, step)
    vmin = min(pred_t.min(), pred_r.min())
    vmax = max(pred_t.max(), pred_r.max())
    with style_context():
        fig = new_figure(10.0, 4.6)
        for k, (z, title) in enumerate([(pred_t, "tanh"), (pred_r, "relu")]):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            surf = plot_surface_3d(ax, x1, x2, z, f"Activation {title}", vmin, vmax)
        fig.colorbar(surf, ax=fig.axes, shrink=0.55, pad=0.05)
    return fig


def parity_figure(y_train, pred_train, y_test, pred_test):
    """1:1 parity (orthonormal frame, line y = x) for training and test."""
    with style_context():
        fig = new_figure(9.0, 4.4)
        for k, (y, p, label, color) in enumerate([(y_train, pred_train, "training", COLOR_TRAIN),
                                                   (y_test, pred_test, "test", COLOR_TEST)]):
            y, p = np.asarray(y), np.asarray(p)
            ax = fig.add_subplot(1, 2, k + 1)
            ax.scatter(y, p, s=6, color=color, alpha=0.6)
            lim = (min(y.min(), p.min()), max(y.max(), p.max()))
            ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("y exact")
            ax.set_ylabel("y predicted")
            ax.set_title(f"Parity, {label} (R2 = {r2_score(y, p):.5f}, N = {len(y)})")
            ax.legend()
    return fig


def width_figure(val_mse_by_width, h_star):
    """Validation MSE per width: points per seed, mean, log scale, H* marked."""
    with style_context():
        fig = new_figure(6.4, 4.2)
        ax = fig.add_subplot()
        widths = sorted(val_mse_by_width)
        for i, h in enumerate(widths):
            v = floor_for_log(val_mse_by_width[h])
            ax.semilogy([i] * len(v), v, "o", color=COLOR_FIT, alpha=0.5, ms=4)
        means = [floor_for_log(np.mean(val_mse_by_width[h])) for h in widths]
        ax.semilogy(range(len(widths)), means, "k-s", label="mean (10 seeds)")
        ax.axvline(widths.index(h_star), color=COLOR_TEST, ls=":", label=f"H* = {h_star}")
        ax.set_xticks(range(len(widths)), [str(h) for h in widths])
        ax.set_xlabel("width H")
        ax.set_ylabel("validation MSE")
        ax.legend()
    return fig


def sensitivity_figure(groups):
    """groups: {label: [validation MSE per seed]}; dictionary order kept."""
    with style_context():
        fig = new_figure(9.0, 4.6)
        ax = fig.add_subplot()
        for i, (label, values) in enumerate(groups.items()):
            v = floor_for_log(values)
            ax.semilogy([i] * len(v), v, "o", color=COLOR_FIT, alpha=0.5, ms=4)
            ax.semilogy([i], [np.mean(v)], "k_", ms=14)
        ax.set_xticks(range(len(groups)), list(groups), rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("validation MSE")
    return fig


def convergence_figure(rows):
    """rows: {label: {'n_iter': [...], 'causes': {cause: n}}}; n_iter per run and stopping causes."""
    with style_context():
        fig = new_figure(10.0, 4.6)
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
        for i, (label, row) in enumerate(rows.items()):
            ax1.semilogy([i] * len(row["n_iter"]), floor_for_log(row["n_iter"], 1.0), "o",
                         color=COLOR_FIT, alpha=0.5, ms=4)
        ax1.set_xticks(range(len(rows)), list(rows), rotation=60, ha="right", fontsize=5)
        ax1.set_ylabel("iterations / epochs")
        bottom = np.zeros(len(rows))
        for cause in STOP_CAUSES:
            counts = np.array([row["causes"].get(cause, 0) for row in rows.values()])
            ax2.bar(range(len(rows)), counts, bottom=bottom, label=cause)
            bottom += counts
        ax2.set_xticks(range(len(rows)), list(rows), rotation=60, ha="right", fontsize=5)
        ax2.set_ylabel("runs")
        ax2.legend(fontsize=6)
    return fig


def adam_curves_figure(curves):
    """curves: {label: [[loss per epoch] per seed]} (log scale)."""
    with style_context():
        fig = new_figure(7.0, 4.4)
        ax = fig.add_subplot()
        for label, runs in curves.items():
            for k, c in enumerate(runs):
                ax.semilogy(floor_for_log(c), lw=0.6, alpha=0.5, label=label if k == 0 else None)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=6)
    return fig


def save_mlp_figures(inputs, directory=FIGURES_DIR, step: float = RENDER_STEP) -> list:
    """Export the figures that depend on the figure network; ``inputs`` comes from
    ``final_evaluation_2d.figure_inputs`` (verified models, data, freeze)."""
    directory = Path(directory)
    model = inputs["figure_model"]
    paths = []
    x1, x2, exact = surface_grid(cardinal_sine, step)
    _, _, pred = surface_grid(model.predict, step)
    for stem, z, title in [("mlp_2d_surface_exact", exact, "Exact surface"),
                           ("mlp_2d_surface_predicted", pred, "Predicted surface"),
                           ("mlp_2d_surface_residuals", np.abs(pred - exact), "Absolute residual")]:
        paths += save_figure(surface_figure(x1, x2, z, title), directory / stem, vector=False)
    paths += save_figure(surfaces_board(model.predict, step), directory / "mlp_2d_panel_3d",
                         vector=False)
    paths += save_figure(residual_map_2d(model.predict, step), directory / "mlp_2d_residual_map")
    (xt, yt), (xs, ys) = inputs["train"], inputs["test"]
    paths += save_figure(parity_figure(yt, model.predict(xt), ys, model.predict(xs)),
                         directory / "mlp_2d_parity")
    paths += save_figure(activation_figure(inputs["tanh_model"].predict,
                                           inputs["relu_model"].predict, step),
                         directory / "mlp_2d_activation_tanh_relu", vector=False)
    return paths


def save_campaign_figures(collected, h_star, directory=FIGURES_DIR) -> list:
    """Campaign figures, computed only from the validation records.

    ``collected``: output of ``collect_stats`` ({config_id: {config, roles, records, stats}}).
    """
    from src.experiments.runner_2d_nominal import config_label
    directory = Path(directory)
    by_width, groups, rows, curves = {}, {}, {}, {}
    for item in collected.values():
        cfg, recs = item["config"], item["records"]
        label = config_label(cfg)
        vals = [r["validation_metrics"]["mse"] for r in recs]
        groups[label] = vals
        rows[label] = {"n_iter": [r["diagnostics"]["n_iter"] for r in recs],
                       "causes": item["stats"]["stop_causes"]}
        if any(role.startswith("NOM:") for role in item["roles"]):
            by_width[cfg.hidden_layer_sizes[0]] = vals
        if cfg.solver == "adam":
            curves[label] = [r["diagnostics"]["loss_curve"] for r in recs
                             if r["diagnostics"]["loss_curve"]]
    paths = save_figure(width_figure(by_width, h_star), directory / "mlp_2d_width")
    paths += save_figure(sensitivity_figure(groups), directory / "mlp_2d_sensitivities")
    paths += save_figure(convergence_figure(rows), directory / "mlp_2d_convergence")
    if curves:
        paths += save_figure(adam_curves_figure(curves), directory / "mlp_2d_adam_curves")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args(argv)
    from src.experiments import final_evaluation_2d  # checks the freeze and provides models and data
    from src.experiments.runner_2d_nominal import check_threads
    check_threads()
    from src.experiments.runner_2d_nominal import (NOMINAL_STORE, data_fingerprint,
                                                   environment_identity, load_input_data)
    from src.experiments.runner_2d_parametric import compute_selection
    inputs = final_evaluation_2d.figure_inputs(tables_dir=args.tables_dir)
    xt, yt, xv, yv = load_input_data()
    selection, collected = compute_selection(NOMINAL_STORE, data_fingerprint(xt, yt, xv, yv),
                                             environment_identity(check_threads()))
    paths = save_mlp_figures(inputs, args.figures_dir)
    paths += save_campaign_figures(collected, selection["h_star"]["h_star"], args.figures_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
