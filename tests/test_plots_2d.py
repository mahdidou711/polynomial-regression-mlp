"""Numerical orientation of the 3D surfaces and exports."""
import numpy as np
import pytest
from matplotlib import image as mpimg
from mpl_toolkits.mplot3d import Axes3D

from src.data.generator_2d import cardinal_sine
from src.visualization import plots_2d as p

COARSE = 0.05  # coarse rendering for the file tests; the orientation is tested at 0.005
SUMMIT = (0.468169, 0.547746)  # ((pi - 0.2) / 2pi ; (pi + 0.3) / 2pi), exact top of cardinal_sine


def asym(points):  # asymmetric function: swapping x1 and x2 changes the value
    return 3.0 * points[:, 0] + points[:, 1] ** 2


def naive_surface(func, step=0.005):
    """Reproduction of a typical naive code: meshgrid 'xy' + R[i, j] = f(x1_i, x2_j)."""
    x1v = np.arange(0, 1, step)
    x2v = np.arange(0, 1, step)
    xv, yv = np.meshgrid(x1v, x2v)
    r = np.zeros((len(x1v), len(x2v)))
    for i, x1 in enumerate(x1v):
        for j, x2 in enumerate(x2v):
            r[i, j] = func(np.array([[x1, x2]]))[0]
    return xv, yv, r


def test_render_axis_identical_to_arange():
    axis = p.render_axis()
    assert axis.shape == (200,) and p.RENDER_STEP == 0.005
    np.testing.assert_array_equal(axis, np.arange(0, 1, 0.005))
    assert axis[0] == 0.0 and axis[-1] == 0.995  # bound 1 excluded


def test_asymmetric_z_ij_correspondence():
    x1, x2, z = p.surface_grid(asym)
    np.testing.assert_array_equal(z, 3.0 * x1 + x2 ** 2)  # Z[i, j] = f(x1_i, x2_j), exactly
    assert not np.allclose(z, z.T)  # a transposition would be detected
    axis = p.render_axis()
    assert z[10, 150] == 3.0 * axis[10] + axis[150] ** 2
    np.testing.assert_array_equal(x1[:, 0], axis)  # x1 varies along axis 0
    np.testing.assert_array_equal(x2[0, :], axis)  # x2 varies along axis 1


def test_top_of_the_exact_surface_in_the_plotted_frame():
    x1, x2, z = p.surface_grid(cardinal_sine)
    i, j = np.unravel_index(np.argmax(z), z.shape)
    assert abs(x1[i, j] - SUMMIT[0]) < p.RENDER_STEP and abs(x2[i, j] - SUMMIT[1]) < p.RENDER_STEP
    assert z[i, j] > 0.999
    assert z[j, i] < 0.7  # at the transposed position, the surface is far from the top


def test_the_naive_code_is_transposed_and_ours_is_not():
    """Documents the defect: the naive surface is the transpose of ours."""
    step = 0.05
    x1, x2, z = p.surface_grid(asym, step)
    xv, yv, r = naive_surface(asym, step)
    # naive (Xv, Yv, R): the point plotted at (Xv[i,j], Yv[i,j]) has value R[i,j] = f(x1_i, x2_j),
    # whereas Xv[i,j] = x1_j and Yv[i,j] = x2_i: the value is not f(plotted x).
    assert not np.allclose(r, 3.0 * xv + yv ** 2)
    np.testing.assert_allclose(r, z)  # same values, but associated with the wrong coordinates
    np.testing.assert_array_equal(z, 3.0 * x1 + x2 ** 2)  # ours: consistent triplet


def test_triplet_passed_to_plot_surface(monkeypatch):
    calls = []
    original = Axes3D.plot_surface

    def spy(self, X, Y, Z, *a, **k):
        calls.append((X.copy(), Y.copy(), Z.copy(), k))
        return original(self, X, Y, Z, *a, **k)

    monkeypatch.setattr(Axes3D, "plot_surface", spy)
    p.surfaces_board(cardinal_sine, COARSE)
    assert len(calls) == 3
    for X, Y, Z, k in calls[:1]:  # exact surface: Z = f(X, Y)
        np.testing.assert_array_equal(Z, cardinal_sine(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape))
    assert calls[1][2].shape == calls[0][2].shape
    assert all(k["antialiased"] is True and k["cmap"] == "house_seq" for *_, k in calls)
    np.testing.assert_allclose(calls[2][2], 0.0, atol=1e-15)  # perfect model: zero residual


def test_view_labels_and_antialiasing():
    x1, x2, z = p.surface_grid(asym, COARSE)
    fig = p.surface_figure(x1, x2, z, "t")
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("x1", "x2")
    assert (ax.elev, ax.azim) == (30, -60)
    assert ax.name == "3d"


def test_2d_residual_map_uses_the_same_grid(monkeypatch):
    seen = []
    from matplotlib.axes import Axes
    original = Axes.pcolormesh
    monkeypatch.setattr(Axes, "pcolormesh", lambda self, X, Y, C, *a, **k: (seen.append((X, Y, C)), original(self, X, Y, C, *a, **k))[1])
    fig = p.residual_map_2d(lambda pts: cardinal_sine(pts) + 0.1 * pts[:, 0], COARSE)
    X, Y, C = seen[0]
    np.testing.assert_allclose(C, 0.1 * X)  # |residual| = 0.1 x1: x1 is indeed the horizontal axis
    assert fig.axes[0].get_xlabel() == "x1"
    assert fig.axes[0].get_aspect() == 1.0
    seen.clear()
    p.residual_map_2d(cardinal_sine, COARSE)
    np.testing.assert_allclose(seen[0][2], 0.0, atol=1e-15)


def test_parity_orthonormal_frame_and_identity_line():
    rng = np.random.default_rng(0)
    y = rng.random(50)
    fig = p.parity_figure(y, y + 0.01, y, y - 0.02)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert ax.get_aspect() == 1.0
        line = ax.lines[0]
        np.testing.assert_array_equal(line.get_xdata(), line.get_ydata())  # y = x
        assert ax.get_xlim() == ax.get_ylim()
    assert "R2" in fig.axes[0].get_title()


def _data(seed, n):
    x = np.random.default_rng(seed).random((n, 2))
    return x, cardinal_sine(x)


class _Model:
    def __init__(self, f):
        self.predict = f


def test_export_formats_3d_png_only_2d_pdf_and_png(tmp_path):
    inputs = {"figure_model": _Model(lambda x: cardinal_sine(x) * 0.98),
              "tanh_model": _Model(cardinal_sine), "relu_model": _Model(lambda x: cardinal_sine(x) + 0.05),
              "train": _data(1, 20), "test": _data(2, 30)}
    paths = {q.name for q in p.save_mlp_figures(inputs, tmp_path, COARSE)}
    three_d = {"mlp_2d_surface_exact", "mlp_2d_surface_predicted", "mlp_2d_surface_residuals",
               "mlp_2d_panel_3d", "mlp_2d_activation_tanh_relu"}
    two_d = {"mlp_2d_residual_map", "mlp_2d_parity"}
    assert paths == {f"{s}.png" for s in three_d} | {f"{s}.png" for s in two_d} | {f"{s}.pdf" for s in two_d}
    for s in three_d:
        assert not (tmp_path / f"{s}.pdf").exists()
    img = mpimg.imread(tmp_path / "mlp_2d_surface_exact.png")
    assert img.shape[:2] == (round(4.4 * 300), round(5.2 * 300))  # 300 DPI
    assert (tmp_path / "mlp_2d_parity.pdf").read_bytes().startswith(b"%PDF")


def test_campaign_figures(tmp_path):
    from src.experiments import runner_2d_nominal as nom
    from src.experiments import runner_2d_parametric as par
    from threadpoolctl import threadpool_limits
    from tests.helpers_2d import FAST, synthetic
    (xt, yt), (xv, yv), _ = synthetic()
    with threadpool_limits(limits=1):
        plan = par.full_plan(10, FAST)
        nom.ensure_runs(plan, xt, yt, xv, yv, tmp_path / "s")
        collected = nom.collect_stats(plan, tmp_path / "s", nom.data_fingerprint(xt, yt, xv, yv),
                                      nom.environment_identity(1))
    names = {q.name for q in p.save_campaign_figures(collected, 10, tmp_path / "f")}
    for stem in ("mlp_2d_width", "mlp_2d_sensitivities", "mlp_2d_convergence", "mlp_2d_adam_curves"):
        assert {f"{stem}.pdf", f"{stem}.png"} <= names


def test_plots_do_not_reference_the_test_set():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(p.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert not {"generate_test_set_2d", "prepare_data_2d", "DataSplits2D"} & names


def test_plotted_residual_is_the_absolute_value_of_pred_minus_exact(monkeypatch):
    """The residual array Z is |pred - exact|, including for a negative gap."""
    calls = []
    original = Axes3D.plot_surface

    def spy(self, X, Y, Z, *a, **k):
        calls.append((X.copy(), Y.copy(), Z.copy()))
        return original(self, X, Y, Z, *a, **k)

    monkeypatch.setattr(Axes3D, "plot_surface", spy)

    def below(pts):  # below the target by a variable gap: pred - exact < 0
        return cardinal_sine(pts) - 0.05 - 0.1 * pts[:, 0]

    p.surfaces_board(below, COARSE)
    X, Y, Z = calls[2]
    np.testing.assert_allclose(Z, 0.05 + 0.1 * X, atol=1e-12)  # positive, = |pred - exact|
    assert Z.min() > 0  # forgetting np.abs would give negative values
    np.testing.assert_allclose(Z, np.abs(calls[1][2] - calls[0][2]))


def test_saved_residual_figures_use_the_absolute_value(monkeypatch, tmp_path):
    """Same check for mlp_2d_surface_residuals and the 2D map, through save_mlp_figures."""
    from matplotlib.axes import Axes
    surf, mesh = [], []
    original_surf, original_mesh = Axes3D.plot_surface, Axes.pcolormesh
    monkeypatch.setattr(Axes3D, "plot_surface", lambda self, X, Y, Z, *a, **k: (
        surf.append(Z.copy()), original_surf(self, X, Y, Z, *a, **k))[1])
    monkeypatch.setattr(Axes, "pcolormesh", lambda self, X, Y, C, *a, **k: (
        mesh.append(np.asarray(C).copy()), original_mesh(self, X, Y, C, *a, **k))[1])
    inputs = {"figure_model": _Model(lambda x: cardinal_sine(x) - 0.2),
              "tanh_model": _Model(cardinal_sine), "relu_model": _Model(cardinal_sine),
              "train": _data(1, 20), "test": _data(2, 30)}
    p.save_mlp_figures(inputs, tmp_path, COARSE)
    # surface order: exact, predicted, residuals, then panel (3), then activation (2)
    np.testing.assert_allclose(surf[2], 0.2, atol=1e-12)
    np.testing.assert_allclose(surf[5], 0.2, atol=1e-12)  # third panel of the board
    grid = [c for c in mesh if c.shape == (20, 20)]  # excludes the colour bar gradient
    assert len(grid) == 1
    np.testing.assert_allclose(grid[0], 0.2, atol=1e-12)
