"""Tests de src/experiments/runner_1d_nominal.py (etape 3).

Couvre : protocole nominal (donnees, degres, metriques recalculees independamment),
degre optimal mesure (jamais impose), diagnostics, campagne a 50 graines distincte de la
graine 42, tables CSV, journalisation des exceptions du point d'entree.
Commande : python -m pytest tests/test_experiment_1d_nominal.py -v
"""

import csv
import json
import logging

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.data import genere_base_apprentissage, genere_base_test
from src.experiments import runner_1d_nominal as r
from src.utils.seeding import SEED_POLYNOMIAL, SEEDS_STAT_1D


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def nominal():
    logging.disable(logging.WARNING)
    try:
        yield r.run_nominal()
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def campaign():
    logging.disable(logging.WARNING)
    try:
        yield r.run_nominal_campaign()
    finally:
        logging.disable(logging.NOTSET)


class TestNominalProtocol:
    def test_degrees_are_1_to_12(self, nominal):
        assert list(nominal.degrees) == list(range(1, 13)) == list(r.DEGREES)

    def test_data_are_the_nominal_ones(self, nominal):
        assert nominal.seed == SEED_POLYNOMIAL == 42
        x, y = genere_base_apprentissage(seed=42)
        xt, yt = genere_base_test()
        assert nominal.x_train.shape == (20, 1) and nominal.x_test.shape == (200, 1)
        assert np.array_equal(nominal.x_train, x) and np.array_equal(nominal.y_train, y)
        assert np.array_equal(nominal.y_test, yt) and np.array_equal(nominal.x_test, xt)
        assert x.min() == 0.0 and x.max() < 1.2 and xt.max() < 1.2

    def test_metrics_match_an_independent_sklearn_computation(self, nominal):
        x, y = nominal.x_train[:, 0], nominal.y_train
        xt, yt = nominal.x_test[:, 0], nominal.y_test
        for i, d in enumerate(nominal.degrees):
            h = np.vander(x, d + 1, increasing=True)
            ht = np.vander(xt, d + 1, increasing=True)
            m = LinearRegression(fit_intercept=False).fit(h, y)
            assert nominal.eqm_train[i] == pytest.approx(np.mean((y - m.predict(h)) ** 2),
                                                          rel=1e-9, abs=1e-14)
            assert nominal.eqm_test[i] == pytest.approx(np.mean((yt - m.predict(ht)) ** 2),
                                                         rel=1e-9, abs=1e-14)

    def test_train_and_test_errors_are_distinct_quantities(self, nominal):
        assert not np.allclose(nominal.eqm_train, nominal.eqm_test)
        assert np.all(nominal.eqm_train > 0) and np.all(np.isfinite(nominal.eqm_test))

    def test_training_error_is_non_increasing_in_degree(self, nominal):
        # Modeles emboites ajustes par moindres carres : propriete mathematique, pas un
        # resultat empirique (tolerance de l'arrondi).
        assert np.all(np.diff(nominal.eqm_train) <= 1e-10)

    def test_stored_values_are_never_floored(self, nominal):
        assert np.all(nominal.eqm_train > 0) and nominal.eqm_train.min() > 1e-3

    def test_predictions_are_consistent_with_coefficients(self, nominal):
        for i, coef in enumerate(nominal.coefs):
            h = np.vander(nominal.x_test[:, 0], len(coef), increasing=True)
            # coefficients ~1e4 : annulations, d'ou la tolerance absolue
            assert np.allclose(nominal.y_pred_test[i], h @ coef, rtol=1e-9, atol=1e-8)

    def test_run_is_reproducible_and_leaves_global_state_untouched(self):
        state = np.random.get_state()[1].copy()
        logging.disable(logging.WARNING)
        try:
            a, b = r.run_nominal(), r.run_nominal()
        finally:
            logging.disable(logging.NOTSET)
        assert np.array_equal(a.eqm_test, b.eqm_test)
        assert np.array_equal(np.random.get_state()[1], state)


class TestOptimalDegree:
    def test_is_the_measured_argmin_of_eqmt(self, nominal):
        assert nominal.optimal_degree == int(nominal.degrees[np.argmin(nominal.eqm_test)])
        assert nominal.eqm_test[nominal.index_of(nominal.optimal_degree)] == nominal.eqm_test.min()

    def test_synthetic_profile_is_not_biased_toward_any_degree(self):
        for star in (1, 4, 9, 12):
            eqmt = np.full(12, 1.0)
            eqmt[star - 1] = 0.1
            assert r.optimal_degree(range(1, 13), eqmt) == star

    def test_tie_goes_to_the_smallest_degree(self):
        assert r.optimal_degree([1, 2, 3], [0.5, 0.2, 0.2]) == 2

    @pytest.mark.parametrize("degrees, eqmt", [([1, 2], [1.0]), ([], []), ([1, 2], [1.0, np.nan]),
                                               ([[1, 2]], [[1.0, 2.0]])])
    def test_invalid_input_is_rejected(self, degrees, eqmt):
        with pytest.raises(ValueError):
            r.optimal_degree(degrees, eqmt)

    def test_a_different_seed_may_change_it_and_is_reported_as_measured(self):
        logging.disable(logging.WARNING)
        try:
            stars = {s: r.run_nominal(seed=s).optimal_degree for s in (42, 1000, 1001, 1002, 1003)}
        finally:
            logging.disable(logging.NOTSET)
        assert all(1 <= v <= 12 for v in stars.values())


class TestDiagnostics:
    def test_rank_and_regimes_on_the_nominal_grid(self, nominal):
        assert list(nominal.rank) == list(range(2, 14))  # rang complet : p = d + 1
        strict = [int(d) for d, g in zip(nominal.degrees, nominal.regime) if g == "strict"]
        assert strict == [1, 2, 3, 4, 5]
        assert nominal.kappa_hth[4] == pytest.approx(6.22e6, rel=1e-2)
        assert nominal.kappa_hth[5] == pytest.approx(1.83e8, rel=1e-2)
        assert nominal.kappa_hth[11] == pytest.approx(5.3e17, rel=1e-2)

    def test_strict_regime_solvers_agree_with_the_loop_solver(self, nominal):
        for i, g in enumerate(nominal.regime):
            if g == "strict":
                assert nominal.rel_error_normal[i] <= 1e-7 and nominal.rel_error_qr[i] <= 1e-7

    def test_normal_equations_degrade_while_qr_does_not(self, nominal):
        assert nominal.rel_error_normal[-1] > 1e-2  # d = 12 : equations normales degradees
        assert np.all(nominal.rel_error_qr < 1e-9)

    def test_diagnostics_can_be_skipped(self):
        x, y = genere_base_apprentissage()
        xt, yt = genere_base_test()
        res = r.run_degree_sweep(x, y, xt, yt, [1, 2, 3], diagnose=False)
        assert res.rank is None and res.kappa_hth is None and res.regime is None
        assert list(res.degrees) == [1, 2, 3]

    def test_sweep_is_reusable_with_other_data(self):
        # Reutilisabilite (etape 4) : autre NbEx, autre sigma, autre degre.
        x, y = genere_base_apprentissage(nb_ex=50, sigma=0.1, seed=7)
        xt, yt = genere_base_test()
        logging.disable(logging.WARNING)
        try:
            res = r.run_degree_sweep(x, y, xt, yt, range(2, 6), seed=7)
        finally:
            logging.disable(logging.NOTSET)
        assert res.eqm_test.shape == (4,) and res.seed == 7 and res.x_train.shape == (50, 1)


class TestCampaign:
    def test_uses_the_50_defined_seeds_and_not_42(self, campaign):
        assert campaign.seeds == SEEDS_STAT_1D and len(campaign.seeds) == 50
        assert 42 not in campaign.seeds

    def test_shapes_and_counts(self, campaign):
        assert campaign.eqm_train.shape == campaign.eqm_test.shape == (50, 12)
        counts = campaign.cstar_counts()
        assert sum(counts.values()) == 50 and set(counts) == set(range(1, 13))
        assert counts[campaign.cstar_mode] == max(counts.values())

    def test_cstar_per_seed_matches_the_argmin_of_that_seed(self, campaign):
        for k in (0, 17, 49):
            assert campaign.optimal_degrees[k] == campaign.degrees[np.argmin(campaign.eqm_test[k])]

    def test_one_seed_matches_a_single_run(self, campaign):
        logging.disable(logging.WARNING)
        try:
            single = r.run_nominal_campaign(seeds=(1007,))
        finally:
            logging.disable(logging.NOTSET)
        assert np.array_equal(single.eqm_test[0], campaign.eqm_test[7])

    def test_campaign_differs_from_the_single_realisation(self, nominal, campaign):
        # Une realisation (graine 42) n'est pas une statistique sur 50 realisations.
        assert not np.allclose(nominal.eqm_test, campaign.summary("test")["median"])
        assert campaign.eqm_test.std(axis=0).min() > 0

    def test_summary_statistics(self, campaign):
        s = campaign.summary("test")
        assert np.allclose(s["median"], np.median(campaign.eqm_test, axis=0))
        assert np.allclose(s["iqr"], s["q3"] - s["q1"]) and np.all(s["iqr"] >= 0)
        assert np.all(s["q1"] <= s["median"]) and np.all(s["median"] <= s["q3"])
        with pytest.raises(KeyError):
            campaign.summary("val")

    def test_mode_tie_goes_to_the_smallest_degree(self):
        c = r.CampaignResult((1, 2), np.arange(1, 4), np.ones((2, 3)), np.ones((2, 3)),
                             np.array([3, 2]))
        assert c.cstar_mode == 2 and c.cstar_median == 2.5

    def test_fixed_grid_so_diagnostics_do_not_depend_on_the_seed(self):
        xa, _ = genere_base_apprentissage(seed=1000)
        xb, _ = genere_base_apprentissage(seed=42)
        assert np.array_equal(xa, xb)


class TestTables:
    def test_nominal_table_roundtrip(self, nominal, tmp_path):
        p = r.write_nominal_table(nominal, tmp_path / "t" / "nominal.csv")
        rows = read_csv(p)
        assert [int(x["degree"]) for x in rows] == list(range(1, 13))
        assert [float(x["eqm_test"]) for x in rows] == list(nominal.eqm_test)  # repr exact
        flags = [x["is_optimal"] == "True" for x in rows]
        assert sum(flags) == 1 and rows[flags.index(True)]["degree"] == str(nominal.optimal_degree)
        assert rows[0]["regime"] == "strict" and rows[-1]["regime"] == "diagnostic"

    def test_campaign_tables(self, campaign, tmp_path):
        a, b = r.write_campaign_tables(campaign, tmp_path / "s.csv", tmp_path / "c.csv")
        stats, counts = read_csv(a), read_csv(b)
        assert len(stats) == 12 and sum(int(c["count_cstar"]) for c in counts) == 50
        assert float(stats[5]["eqmt_median"]) == campaign.summary("test")["median"][5]


class TestMain:
    def test_main_writes_tables_figures_and_journal(self, tmp_path):
        log = tmp_path / "run.jsonl"
        code = r.main(["--figures-dir", str(tmp_path / "fig"), "--tables-dir",
                       str(tmp_path / "tab"), "--log-file", str(log)])
        assert code == 0
        assert len(list((tmp_path / "fig").glob("*.pdf"))) == 5
        assert len(list((tmp_path / "fig").glob("*.png"))) == 5
        assert len(list((tmp_path / "tab").glob("*.csv"))) == 3
        events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
        assert events[0] == "run_start" and events[-1] == "run_done"

    def test_failure_is_logged_with_traceback_then_reraised(self, tmp_path, monkeypatch):
        # AE0-03 en situation : une exception du run laisse sa trace dans le JSON Lines.
        def boom(*a, **k):
            raise RuntimeError("echec simule du calcul")

        monkeypatch.setattr(r, "run_nominal", boom)
        log = tmp_path / "run.jsonl"
        with pytest.raises(RuntimeError, match="echec simule"):
            r.main(["--figures-dir", str(tmp_path / "f"), "--tables-dir", str(tmp_path / "t"),
                    "--log-file", str(log)])
        last = json.loads(log.read_text().splitlines()[-1])
        assert last["event"] == "run_failed" and last["level"] == "ERROR"
        assert "RuntimeError: echec simule du calcul" in last["exc_info"]
