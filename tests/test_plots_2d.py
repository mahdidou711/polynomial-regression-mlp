"""Etape 7, lot A : orientation numerique des surfaces 3D (EXG-PROG-B3-02) et exports."""
import numpy as np
import pytest
from matplotlib import image as mpimg
from mpl_toolkits.mplot3d import Axes3D

from src.data.generator_2d import sinus_cardinal
from src.visualization import plots_2d as p

COARSE = 0.05  # rendu grossier pour les tests de fichiers ; l'orientation est testee a 0,005
SUMMIT = (0.468169, 0.547746)  # ((pi - 0,2) / 2pi ; (pi + 0,3) / 2pi), sommet exact de sinus_cardinal


def asym(points):  # fonction asymetrique : permuter x1 et x2 change la valeur
    return 3.0 * points[:, 0] + points[:, 1] ** 2


def teacher_surface(func, step=0.005):
    """Reproduction du code de l'enseignant : meshgrid 'xy' + R[i, j] = f(x1_i, x2_j)."""
    x1v = np.arange(0, 1, step)
    x2v = np.arange(0, 1, step)
    xv, yv = np.meshgrid(x1v, x2v)
    r = np.zeros((len(x1v), len(x2v)))
    for i, x1 in enumerate(x1v):
        for j, x2 in enumerate(x2v):
            r[i, j] = func(np.array([[x1, x2]]))[0]
    return xv, yv, r


def test_axe_de_rendu_identique_a_arange():
    axis = p.render_axis()
    assert axis.shape == (200,) and p.RENDER_STEP == 0.005
    np.testing.assert_array_equal(axis, np.arange(0, 1, 0.005))
    assert axis[0] == 0.0 and axis[-1] == 0.995  # borne 1 exclue


def test_correspondance_z_ij_asymetrique():
    x1, x2, z = p.surface_grid(asym)
    np.testing.assert_array_equal(z, 3.0 * x1 + x2 ** 2)  # Z[i, j] = f(x1_i, x2_j), exactement
    assert not np.allclose(z, z.T)  # une transposition serait detectee
    axis = p.render_axis()
    assert z[10, 150] == 3.0 * axis[10] + axis[150] ** 2
    np.testing.assert_array_equal(x1[:, 0], axis)  # x1 varie selon l'axe 0
    np.testing.assert_array_equal(x2[0, :], axis)  # x2 varie selon l'axe 1


def test_sommet_de_la_surface_exacte_dans_le_repere_trace():
    x1, x2, z = p.surface_grid(sinus_cardinal)
    i, j = np.unravel_index(np.argmax(z), z.shape)
    assert abs(x1[i, j] - SUMMIT[0]) < p.RENDER_STEP and abs(x2[i, j] - SUMMIT[1]) < p.RENDER_STEP
    assert z[i, j] > 0.999
    assert z[j, i] < 0.7  # a la position transposee, la surface est loin du sommet


def test_le_code_de_l_enseignant_est_bien_transpose_et_pas_le_notre():
    """Documente le defaut (diapositive 18) : sa surface est la transposee de la notre."""
    step = 0.05
    x1, x2, z = p.surface_grid(asym, step)
    xv, yv, r = teacher_surface(asym, step)
    # (Xv, Yv, R) de l'enseignant : le point trace en (Xv[i,j], Yv[i,j]) vaut R[i,j] = f(x1_i, x2_j),
    # alors que Xv[i,j] = x1_j et Yv[i,j] = x2_i : la valeur n'est pas f(x trace).
    assert not np.allclose(r, 3.0 * xv + yv ** 2)
    np.testing.assert_allclose(r, z)  # memes valeurs, mais associees aux mauvaises coordonnees
    np.testing.assert_array_equal(z, 3.0 * x1 + x2 ** 2)  # le notre : triplet coherent


def test_triplet_transmis_a_plot_surface(monkeypatch):
    calls = []
    original = Axes3D.plot_surface

    def spy(self, X, Y, Z, *a, **k):
        calls.append((X.copy(), Y.copy(), Z.copy(), k))
        return original(self, X, Y, Z, *a, **k)

    monkeypatch.setattr(Axes3D, "plot_surface", spy)
    p.surfaces_board(sinus_cardinal, COARSE)
    assert len(calls) == 3
    for X, Y, Z, k in calls[:1]:  # surface exacte : Z = f(X, Y)
        np.testing.assert_array_equal(Z, sinus_cardinal(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape))
    assert calls[1][2].shape == calls[0][2].shape
    assert all(k["antialiased"] is True and k["cmap"] == "coolwarm" for *_, k in calls)
    np.testing.assert_allclose(calls[2][2], 0.0, atol=1e-15)  # modele parfait : residu nul


def test_vue_etiquettes_et_antialiasing():
    x1, x2, z = p.surface_grid(asym, COARSE)
    fig = p.surface_figure(x1, x2, z, "t")
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("x1", "x2")
    assert (ax.elev, ax.azim) == (30, -60)
    assert ax.name == "3d"


def test_carte_residus_2d_utilise_la_meme_grille(monkeypatch):
    seen = []
    from matplotlib.axes import Axes
    original = Axes.pcolormesh
    monkeypatch.setattr(Axes, "pcolormesh", lambda self, X, Y, C, *a, **k: (seen.append((X, Y, C)), original(self, X, Y, C, *a, **k))[1])
    fig = p.residual_map_2d(lambda pts: sinus_cardinal(pts) + 0.1 * pts[:, 0], COARSE)
    X, Y, C = seen[0]
    np.testing.assert_allclose(C, 0.1 * X)  # |residu| = 0,1 x1 : x1 est bien l'axe horizontal
    assert fig.axes[0].get_xlabel() == "x1"
    assert fig.axes[0].get_aspect() == 1.0
    seen.clear()
    p.residual_map_2d(sinus_cardinal, COARSE)
    np.testing.assert_allclose(seen[0][2], 0.0, atol=1e-15)


def test_parite_repere_orthonorme_et_droite_identite():
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
    return x, sinus_cardinal(x)


class _Model:
    def __init__(self, f):
        self.predict = f


def test_exports_formats_3d_png_seul_2d_pdf_et_png(tmp_path):
    inputs = {"figure_model": _Model(lambda x: sinus_cardinal(x) * 0.98),
              "tanh_model": _Model(sinus_cardinal), "relu_model": _Model(lambda x: sinus_cardinal(x) + 0.05),
              "train": _data(1, 20), "test": _data(2, 30)}
    paths = {q.name for q in p.save_mlp_figures(inputs, tmp_path, COARSE)}
    three_d = {"mlp_2d_surface_exacte", "mlp_2d_surface_predite", "mlp_2d_surface_residus",
               "mlp_2d_planche_3d", "mlp_2d_activation_tanh_relu"}
    two_d = {"mlp_2d_residus_carte", "mlp_2d_parite"}
    assert paths == {f"{s}.png" for s in three_d} | {f"{s}.png" for s in two_d} | {f"{s}.pdf" for s in two_d}
    for s in three_d:
        assert not (tmp_path / f"{s}.pdf").exists()
    img = mpimg.imread(tmp_path / "mlp_2d_surface_exacte.png")
    assert img.shape[:2] == (round(4.4 * 300), round(5.2 * 300))  # 300 DPI
    assert (tmp_path / "mlp_2d_parite.pdf").read_bytes().startswith(b"%PDF")


def test_figures_de_campagne(tmp_path):
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
    for stem in ("mlp_2d_largeur", "mlp_2d_sensibilites", "mlp_2d_convergence", "mlp_2d_adam_courbes"):
        assert {f"{stem}.pdf", f"{stem}.png"} <= names


def test_plots_ne_referencent_pas_le_test():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(p.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert not {"genere_base_test_2d", "prepare_data_2d", "DataSplits2D"} & names


def test_residu_trace_est_la_valeur_absolue_de_pred_moins_exact(monkeypatch):
    """AE7-02 (A10) : le tableau Z des residus est |pred - exact|, y compris pour un ecart negatif."""
    calls = []
    original = Axes3D.plot_surface

    def spy(self, X, Y, Z, *a, **k):
        calls.append((X.copy(), Y.copy(), Z.copy()))
        return original(self, X, Y, Z, *a, **k)

    monkeypatch.setattr(Axes3D, "plot_surface", spy)

    def below(pts):  # sous la cible d'un ecart variable : pred - exact < 0
        return sinus_cardinal(pts) - 0.05 - 0.1 * pts[:, 0]

    p.surfaces_board(below, COARSE)
    X, Y, Z = calls[2]
    np.testing.assert_allclose(Z, 0.05 + 0.1 * X, atol=1e-12)  # positif, = |pred - exact|
    assert Z.min() > 0  # un oubli de np.abs donnerait des valeurs negatives
    np.testing.assert_allclose(Z, np.abs(calls[1][2] - calls[0][2]))


def test_figures_de_residus_enregistrees_utilisent_la_valeur_absolue(monkeypatch, tmp_path):
    """AE7-02 : meme controle pour mlp_2d_surface_residus et la carte 2D, via save_mlp_figures."""
    from matplotlib.axes import Axes
    surf, mesh = [], []
    original_surf, original_mesh = Axes3D.plot_surface, Axes.pcolormesh
    monkeypatch.setattr(Axes3D, "plot_surface", lambda self, X, Y, Z, *a, **k: (
        surf.append(Z.copy()), original_surf(self, X, Y, Z, *a, **k))[1])
    monkeypatch.setattr(Axes, "pcolormesh", lambda self, X, Y, C, *a, **k: (
        mesh.append(np.asarray(C).copy()), original_mesh(self, X, Y, C, *a, **k))[1])
    inputs = {"figure_model": _Model(lambda x: sinus_cardinal(x) - 0.2),
              "tanh_model": _Model(sinus_cardinal), "relu_model": _Model(sinus_cardinal),
              "train": _data(1, 20), "test": _data(2, 30)}
    p.save_mlp_figures(inputs, tmp_path, COARSE)
    # ordre des surfaces : exacte, predite, residus, puis planche (3), puis activation (2)
    np.testing.assert_allclose(surf[2], 0.2, atol=1e-12)
    np.testing.assert_allclose(surf[5], 0.2, atol=1e-12)  # troisieme panneau de la planche
    grid = [c for c in mesh if c.shape == (20, 20)]  # exclut le gradient de la barre de couleur
    assert len(grid) == 1
    np.testing.assert_allclose(grid[0], 0.2, atol=1e-12)
