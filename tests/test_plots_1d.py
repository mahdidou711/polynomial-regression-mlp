"""Tests de src/visualization (style.py et plots_1d.py, etape 3).

Couvre : ``save_figure`` (PDF vectoriel + PNG 300 DPI, PNG seul), plancher des echelles
logarithmiques, contenu des figures (donnees tracees, repere 1:1), export des cinq figures.
Commande : python -m pytest tests/test_plots_1d.py -v
"""

import logging
import struct

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.experiments.runner_1d_nominal import run_nominal, run_nominal_campaign
from src.models import predict
from src.visualization import plots_1d
from src.visualization.style import DPI, FLOOR_LOG, floor_for_log, new_figure, save_figure


@pytest.fixture(scope="module")
def nominal():
    logging.disable(logging.WARNING)
    try:
        yield run_nominal()
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def campaign():
    logging.disable(logging.WARNING)
    try:
        yield run_nominal_campaign(seeds=range(1000, 1010))
    finally:
        logging.disable(logging.NOTSET)


def png_dpi(path):
    """DPI lu dans le chunk pHYs du PNG (pixels par metre -> pouces)."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    i = data.index(b"pHYs")
    ppm_x, ppm_y, unit = struct.unpack(">IIB", data[i + 4:i + 13])
    assert unit == 1 and ppm_x == ppm_y
    return ppm_x * 0.0254


class TestSaveFigure:
    def test_writes_vector_pdf_and_300_dpi_png(self, tmp_path):
        fig = new_figure(4, 3)
        fig.add_subplot().plot([0, 1], [0, 1])
        paths = save_figure(fig, tmp_path / "sub" / "demo")
        assert [p.name for p in paths] == ["demo.pdf", "demo.png"]
        pdf, png = paths
        assert pdf.read_bytes()[:5] == b"%PDF-" and pdf.stat().st_size > 0
        assert png_dpi(png) == pytest.approx(300, abs=1) and DPI == 300
        # 4 x 3 pouces a 300 DPI : 1200 x 900 pixels
        w, h = struct.unpack(">II", png.read_bytes()[16:24])
        assert (w, h) == (1200, 900)

    def test_pdf_is_vector_not_embedded_raster(self, tmp_path):
        fig = new_figure()
        fig.add_subplot().plot(np.arange(10), np.arange(10) ** 2)
        pdf = save_figure(fig, tmp_path / "v")[0].read_bytes()
        assert b"/Subtype /Image" not in pdf

    def test_vector_false_writes_png_only(self, tmp_path):
        fig = new_figure()
        fig.add_subplot()
        paths = save_figure(fig, tmp_path / "surf", vector=False)
        assert [p.suffix for p in paths] == [".png"] and not (tmp_path / "surf.pdf").exists()

    @pytest.mark.parametrize("name", ["a.pdf", "a.PNG"])
    def test_extension_in_stem_is_rejected(self, tmp_path, name):
        with pytest.raises(ValueError, match="sans extension"):
            save_figure(new_figure(), tmp_path / name)

    def test_dotted_stem_keeps_its_name(self, tmp_path):
        paths = save_figure(new_figure(), tmp_path / "nominal_v1.2")
        assert [p.name for p in paths] == ["nominal_v1.2.pdf", "nominal_v1.2.png"]

    def test_two_exports_of_the_same_figure_are_identical_byte_for_byte(self, tmp_path):
        """AE7-07 : le PDF ne porte plus de date de creation (reproductibilite octet a octet)."""
        def make():
            fig = new_figure(4, 3)
            fig.add_subplot().plot([0, 1, 2], [0, 1, 4])
            return fig

        first = save_figure(make(), tmp_path / "a" / "f")
        second = save_figure(make(), tmp_path / "b" / "f")
        for p1, p2 in zip(first, second):
            assert p1.read_bytes() == p2.read_bytes(), p1.suffix
        assert b"/CreationDate" not in first[0].read_bytes()

    def test_no_pyplot_state_is_created(self, tmp_path):
        import sys

        save_figure(new_figure(), tmp_path / "x")
        if "matplotlib.pyplot" in sys.modules:
            assert sys.modules["matplotlib.pyplot"].get_fignums() == []


class TestLogFloor:
    def test_floor_applies_to_a_copy_only(self):
        v = np.array([0.0, 1e-20, 0.5])
        out = floor_for_log(v)
        assert np.array_equal(out, [FLOOR_LOG, FLOOR_LOG, 0.5])
        assert np.array_equal(v, [0.0, 1e-20, 0.5])

    def test_zero_training_error_gives_finite_log_axis(self, nominal):
        from dataclasses import replace

        zeroed = replace(nominal, eqm_train=np.zeros_like(nominal.eqm_train))
        fig = plots_1d.plot_error_profile(zeroed)
        ydata = fig.axes[0].lines[0].get_ydata()
        assert np.all(np.isfinite(np.log10(ydata))) and np.all(ydata == FLOOR_LOG)
        assert np.all(zeroed.eqm_train == 0.0)  # valeurs stockees inchangees


class TestFigures:
    def test_error_profile_plots_the_stored_errors_on_a_log_axis(self, nominal):
        ax = plots_1d.plot_error_profile(nominal).axes[0]
        assert ax.get_yscale() == "log"
        assert np.array_equal(ax.lines[0].get_ydata(), nominal.eqm_train)
        assert np.array_equal(ax.lines[1].get_ydata(), nominal.eqm_test)
        assert np.array_equal(ax.lines[0].get_xdata(), nominal.degrees)
        star_line = ax.lines[2].get_xdata()[0]
        assert star_line == nominal.optimal_degree

    def test_fits_has_one_panel_per_degree_with_data_and_target(self, nominal):
        fig = plots_1d.plot_fits(nominal)
        panels = [a for a in fig.axes if a.get_visible() and a.lines]
        assert len(panels) == 12
        titles = [a.get_title() for a in panels]
        assert sum("C^*" in t for t in titles) == 1
        for a in panels:
            assert len(a.lines) == 3  # cible, polynome, apprentissage
            assert len(a.lines[2].get_xdata()) == 20

    def test_parity_key_degrees_and_equal_aspect(self, nominal):
        assert plots_1d.key_degrees(nominal) == sorted({1, nominal.optimal_degree, 12})
        fig = plots_1d.plot_parity(nominal)
        assert len(fig.axes) == len(plots_1d.key_degrees(nominal))
        for ax in fig.axes:
            assert ax.get_aspect() == 1.0 and ax.get_xlim() == ax.get_ylim()
            assert len(ax.lines[1].get_xdata()) == 200  # points de test
            assert ax.lines[0].get_xdata()[0] == ax.lines[0].get_ydata()[0]  # droite y = x

    def test_parity_has_a_y_equals_x_legend_entry(self, nominal):
        """AE3-08 : la droite y = x est legendee sur chaque panneau."""
        for ax in plots_1d.plot_parity(nominal).axes:
            texts = [t.get_text() for t in ax.get_legend().get_texts()]
            assert texts == ["y = x"]

    def test_parity_window_depends_on_the_target_only(self, nominal):
        """AE3-08 : un modele divergent (d = 12) ne modifie pas la fenetre ; ses points exclus sont comptes."""
        lo, hi = plots_1d.parity_window(nominal.y_test)
        span = float(nominal.y_test.max() - nominal.y_test.min())
        assert lo == pytest.approx(float(nominal.y_test.min()) - plots_1d.PARITY_PAD * span)
        fig = plots_1d.plot_parity(nominal, degrees=[1, 12])
        for ax in fig.axes:
            assert (ax.get_xlim(), ax.get_ylim()) == ((lo, hi), (lo, hi))
        y12 = nominal.y_pred_test[nominal.index_of(12)]
        n_out = int(np.sum((y12 < lo) | (y12 > hi)))
        assert n_out > 0  # le panneau d = 12 contient des points hors cadre
        notes = [t.get_text() for t in fig.axes[1].texts]
        assert len(notes) == 1 and notes[0].startswith(f"{n_out} points hors cadre")

    def test_parity_panel_without_overflow_has_no_note(self, nominal):
        fig = plots_1d.plot_parity(nominal, degrees=[nominal.optimal_degree])
        assert len(fig.axes[0].texts) == 0

    def test_fit_overflow_helper(self):
        assert plots_1d.fit_overflow(np.array([-1.0, 0.0, 2.0])) is None  # bornes incluses
        assert plots_1d.fit_overflow(np.array([-1.0, 3.5])) == 3.5
        assert plots_1d.fit_overflow(np.array([-7.0, 1.0])) == 7.0

    def test_fits_mark_every_clipped_panel_and_only_those(self, nominal):
        """AE3-08 : un panneau dont le polynome sort de la fenetre est signale sur le panneau."""
        xs = np.linspace(plots_1d.XMIN_NOMINAL, plots_1d.XMAX_NOMINAL, plots_1d.N_DENSE,
                         endpoint=False)
        expected = {int(d) for d, coef in zip(nominal.degrees, nominal.coefs)
                    if plots_1d.fit_overflow(predict(coef, xs)) is not None}
        assert expected, "le jeu nominal doit comporter des panneaux coupes (degres 3, 9, 11, 12)"
        fig = plots_1d.plot_fits(nominal)
        marked = {int(a.get_title().split("=")[1].split()[0]) for a in fig.axes
                  if a.get_visible() and a.texts}
        assert marked == expected
        for a in fig.axes:
            if a.get_visible():
                assert ("coupé" in a.get_title()) == bool(a.texts)

    def test_parity_uses_requested_degrees(self, nominal):
        fig = plots_1d.plot_parity(nominal, degrees=[3, 4])
        assert len(fig.axes) == 2
        assert np.array_equal(fig.axes[0].lines[1].get_ydata(),
                              nominal.y_pred_test[nominal.index_of(3)])

    def test_conditioning_has_the_diagnostic_threshold(self, nominal):
        ax = plots_1d.plot_conditioning(nominal).axes[0]
        assert ax.get_yscale() == "log"
        assert ax.lines[1].get_ydata()[0] == 1e7

    def test_campaign_figure_shows_counts_and_medians(self, campaign):
        fig = plots_1d.plot_campaign(campaign)
        bars = [p.get_height() for p in fig.axes[0].patches]
        assert bars == [campaign.cstar_counts()[d] for d in range(1, 13)]
        assert np.array_equal(fig.axes[1].lines[1].get_ydata(), campaign.summary("test")["median"])

    def test_figures_are_plain_figure_objects(self, nominal):
        assert isinstance(plots_1d.plot_error_profile(nominal), Figure)


class TestSaveNominalFigures:
    def test_exports_five_figures_as_pdf_and_png(self, nominal, campaign, tmp_path):
        paths = plots_1d.save_nominal_figures(nominal, campaign, tmp_path)
        assert len(paths) == 10
        stems = {p.stem for p in paths}
        assert stems == {"nominal_1d_error_profile", "nominal_1d_fits", "nominal_1d_parity",
                         "nominal_1d_conditioning", "nominal_1d_campaign_50_graines"}
        for p in paths:
            assert p.exists() and p.stat().st_size > 1000
            if p.suffix == ".png":
                assert png_dpi(p) == pytest.approx(300, abs=1)
            else:
                assert p.read_bytes()[:5] == b"%PDF-"
