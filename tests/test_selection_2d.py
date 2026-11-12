"""Etape 7, lot A : classement, regles d'egalite, gel, etancheite et evaluation finale."""
import ast
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src.data import generator_2d
from src.experiments import final_evaluation_2d as fin
from src.experiments import runner_2d_nominal as nom
from src.experiments import runner_2d_parametric as par
from src.models.mlp import MLPConfig
from src.utils.seeding import SEEDS_MONTE_CARLO
from tests.helpers_2d import FAST, one_thread, synthetic  # noqa: F401

pytestmark = pytest.mark.usefixtures("one_thread")


def _stats(mean, n=10):
    return {"n_runs": n, "validation": {"mse": {"mean": mean}}}


def _collected(entries):
    """entries : [(config, mean)] dans l'ordre du plan."""
    return {nom.config_id(c): {"config": c, "roles": ["T"], "records": [], "stats": _stats(m)}
            for c, m in entries}


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Campagne complete (190 runs) a budget court, sans aucun acces au test."""
    from threadpoolctl import threadpool_limits
    base = tmp_path_factory.mktemp("campagne")
    train, val, test = synthetic()
    calls = []

    def forbidden(*a, **k):
        calls.append(1)
        raise AssertionError("base de test lue pendant la selection")

    mp = pytest.MonkeyPatch()
    mp.setattr(generator_2d, "genere_base_test_2d", forbidden)
    mp.setattr(generator_2d, "prepare_data_2d", forbidden)
    try:
        with threadpool_limits(limits=1):
            store, tables = base / "store", base / "tables"
            data = (*train, *val)
            nom.ensure_runs(nom.nominal_plan(FAST), *data, store)
            fp, env = nom.data_fingerprint(*data), nom.environment_identity(1)
            width = nom.select_width({
                int(c["config"].hidden_layer_sizes[0]): c["stats"]
                for c in nom.collect_stats(nom.nominal_plan(FAST), store, fp, env).values()})
            plan = par.full_plan(width["h_star"], FAST)
            nom.ensure_runs(plan, *data, store)
            selection, collected = par.compute_selection(store, fp, env, FAST)
            gel = tables / par.SELECTION_FILE
            digest = par.freeze_selection(gel, selection)
    finally:
        mp.undo()
    return {"store": store, "tables": tables, "train": train, "val": val, "test": test,
            "selection": selection, "collected": collected, "gel": gel, "digest": digest,
            "plan": plan, "fp": fp, "env": env, "test_calls": calls}


def _read_csv(path):
    import csv
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def evaluate(c, tables=None, load_test=None):
    return fin.evaluate_final(c["gel"], c["store"], c["train"], c["val"],
                              load_test or (lambda: c["test"]),
                              tables_dir=tables or c["tables"], overrides=FAST)


# ---------------------------------------------------------------- classement et egalites
def test_h_star_est_l_argmin_de_la_moyenne_de_validation(campaign):
    c = campaign
    means = {}
    for e in nom.nominal_plan(FAST):
        recs = nom.load_config_records(e.config, e.seeds, c["store"], c["fp"], c["env"])
        means[e.config.hidden_layer_sizes[0]] = np.mean([r["validation_metrics"]["mse"] for r in recs])
    oracle = min(means, key=lambda h: (means[h], h))
    assert c["selection"]["h_star"]["h_star"] == oracle
    ranking = c["selection"]["h_star"]["ranking"]
    assert [r["mean_validation_mse"] for r in ranking] == sorted(r["mean_validation_mse"] for r in ranking)


def test_egalite_hstar_plus_petite_largeur():
    stats = {200: _stats(0.5), 5: _stats(0.5), 50: _stats(0.5), 20: _stats(0.9)}
    r = nom.select_width(stats)
    assert r["h_star"] == 5 and r["tie_applied"]  # pas 200
    assert nom.select_width({5: _stats(0.4), 10: _stats(0.3)})["h_star"] == 10  # argmin, non argmax
    assert not nom.select_width({5: _stats(0.4), 10: _stats(0.3)})["tie_applied"]


def test_egalite_adam_ordre_de_la_grille():
    cfgs = [nom.adam_config(20, lr, b) for lr, b in nom.ADAM_GRID]
    plan_order = cfgs[::-1]  # ordre du plan inverse : la regle suit la grille, pas le plan
    r = par.select_adam(_collected([(c, 0.1) for c in plan_order]))
    assert r["selected"] == nom.config_id(cfgs[0]) and r["tie_applied"]
    r2 = par.select_adam(_collected([(cfgs[0], 0.3), (cfgs[1], 0.3), (cfgs[2], 0.2), (cfgs[3], 0.2)]))
    assert r2["selected"] == nom.config_id(cfgs[2])  # (1e-2, 32) avant (1e-2, 200)


def test_egalite_configuration_finale_moins_de_parametres_puis_ordre_du_plan():
    big, small = nom.nominal_config((200,)), nom.nominal_config((5,))
    assert par.select_final(_collected([(big, 0.1), (small, 0.1)]))["selected"] == nom.config_id(small)
    a, b = nom.nominal_config((10,)), nom.nominal_config((10,), activation="relu")  # meme nb de parametres
    assert par.select_final(_collected([(b, 0.1), (a, 0.1)]))["selected"] == nom.config_id(b)
    assert par.select_final(_collected([(a, 0.1), (b, 0.1)]))["selected"] == nom.config_id(a)
    assert par.select_final(_collected([(big, 0.05), (small, 0.1)]))["selected"] == nom.config_id(big)


def test_graine_des_figures_validation_puis_ordre_des_graines():
    def rec(seed, val, train=0.0):
        return {"identity": {"seed": seed}, "run_key": f"k{seed}", "weights_sha256": "w",
                "validation_metrics": {"mse": val}, "training_metrics": {"mse": train}}
    # la validation et l'apprentissage ordonnent les graines differemment
    recs = [rec(s, v, t) for s, v, t in zip(SEEDS_MONTE_CARLO, (.5, .1, .3, .9, .4, .6, .7, .8, .2, .35),
                                            (.9, .8, .1, .5, .6, .7, .4, .3, .2, .05))]
    assert par.figure_seed(recs)["seed"] == SEEDS_MONTE_CARLO[1]  # validation minimale, pas apprentissage
    ties = [rec(s, 0.2) for s in reversed(SEEDS_MONTE_CARLO)]
    assert par.figure_seed(ties)["seed"] == SEEDS_MONTE_CARLO[0]  # ordre de SEEDS_MONTE_CARLO
    with pytest.raises(ValueError):
        par.figure_seed(recs[:9])


def test_refus_nan_et_configuration_incomplete():
    with pytest.raises(ValueError):
        nom.select_width({5: _stats(float("nan")), 10: _stats(0.2)})
    with pytest.raises(ValueError, match="incomplete"):
        nom.select_width({5: _stats(0.1, n=9), 10: _stats(0.2)})
    with pytest.raises(ValueError):
        nom.select_width({})


def test_configurations_partagees_comptees_une_fois(campaign):
    sel = campaign["selection"]
    assert sel["counts"]["distinct_runs"] in (160, 190)
    ranking_ids = [r["config_id"] for r in sel["final"]["ranking"]]
    assert len(ranking_ids) == len(set(ranking_ids)) == sel["counts"]["distinct_configs"]


# ---------------------------------------------------------------- gel
def test_gel_ecrit_une_fois_et_refuse_un_contenu_different(campaign, tmp_path):
    c = campaign
    path = tmp_path / "gel.json"
    d1 = par.freeze_selection(path, c["selection"])
    assert par.freeze_selection(path, c["selection"]) == d1  # idempotent
    other = json.loads(json.dumps(c["selection"]))
    other["adam"]["selected"] = "autre"
    with pytest.raises(ValueError, match="different"):
        par.freeze_selection(path, other)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    assert frozen["schema"] == par.SELECTION_SCHEMA and len(frozen["payload_sha256"]) == 64
    assert "test" not in json.dumps(frozen).replace("tests", "")  # aucune valeur de test dans le gel
    assert frozen["counts"]["realisations"] == 210
    targets = {t["role"] for t in frozen["retrain_targets"]}
    assert targets == {"final", "lbfgs_h_star", "adam_selected"}


# ---------------------------------------------------------------- etancheite
def test_campagne_et_selection_sans_acces_au_test(campaign):
    assert campaign["test_calls"] == []


def test_etancheite_statique():
    names = {m: _fin_names(m) for m in (nom, par, fin)}
    forbidden = {"genere_base_test_2d", "prepare_data_2d", "DataSplits2D"}
    assert not forbidden & names[nom] and not forbidden & names[par]
    assert "genere_base_test_2d" in names[fin]  # le seul module qui lit le test
    assert "freeze_selection" not in names[fin]  # n'ecrit jamais la selection


def _fin_names(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            out.update(a.name for a in node.names)
    return out


def test_enregistrement_avec_champ_test_rejete(campaign, tmp_path):
    path = next(campaign["store"].glob("*.json"))
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["test_metrics"] = {"mse": 0.0}
    copy = tmp_path / path.name
    copy.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(nom.StoreError, match="I7"):
        nom.load_record(copy)


# ---------------------------------------------------------------- evaluation finale
def test_evaluation_sans_gel_ou_avec_gel_altere_ne_lit_pas_le_test(campaign, tmp_path):
    loads = []
    loader = lambda: (loads.append(1), campaign["test"])[1]
    tables = tmp_path / "t"
    with pytest.raises(fin.SelectionError, match="absent"):
        fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"],
                           campaign["val"], loader, tables_dir=tables, overrides=FAST)
    text = campaign["gel"].read_text(encoding="utf-8")
    altered = tmp_path / "gel.json"
    altered.write_text(text.replace('"relu"', '"rel0"', 1), encoding="utf-8")
    with pytest.raises(fin.SelectionError, match="altere"):
        fin.evaluate_final(altered, campaign["store"], campaign["train"], campaign["val"],
                           loader, tables_dir=tables, overrides=FAST)
    assert loads == []


def test_gel_different_du_recalcul_refuse(campaign, tmp_path):
    frozen = json.loads(campaign["gel"].read_text(encoding="utf-8"))
    frozen.pop("payload_sha256")
    other = [r["config_id"] for r in frozen["final"]["ranking"]][-1]
    frozen["final"]["selected"] = other  # configuration finale modifiee, hash interne recalcule
    frozen["payload_sha256"] = nom.sha256_text(nom.canonical(frozen))
    path = tmp_path / "gel.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    loads = []
    with pytest.raises(fin.SelectionError, match="differe"):
        fin.evaluate_final(path, campaign["store"], campaign["train"], campaign["val"],
                           lambda: loads.append(1), tables_dir=tmp_path, overrides=FAST)
    assert loads == []


def test_enregistrement_modifie_depuis_le_gel_refuse(campaign, tmp_path):
    import shutil
    store = tmp_path / "store"
    shutil.copytree(campaign["store"], store)
    victim = next(iter(sorted(store.glob("*.json"))))
    rec = json.loads(victim.read_text(encoding="utf-8"))
    rec["stop_cause"] = "none" if rec["stop_cause"] != "none" else "max_iter"
    rec["payload_sha256"] = nom.sha256_text(nom.canonical({k: v for k, v in rec.items() if k != "payload_sha256"}))
    victim.write_text(json.dumps(rec), encoding="utf-8")  # valide en soi, mais different du gel
    with pytest.raises(fin.SelectionError):
        fin.load_frozen_selection(campaign["gel"], store, campaign["train"], campaign["val"], FAST)


def test_poids_alteres_detectes_a_la_reconstruction(campaign):
    key = campaign["selection"]["retrain_targets"][0]["run_key"]
    record = nom.load_record(campaign["store"] / f"{key}.json")
    record = {**record, "weights_sha256": "0" * 64}
    with pytest.raises(fin.SelectionError, match="empreinte"):
        fin.rebuild_verified_model(record, campaign["train"], campaign["val"])


def test_evaluation_finale_perimetre_oracle_et_selection_inchangee(campaign, tmp_path, monkeypatch, caplog):
    import shutil
    tables = tmp_path / "tables"
    shutil.copytree(campaign["tables"], tables, dirs_exist_ok=True) if campaign["tables"].exists() else None
    tables.mkdir(exist_ok=True)
    shutil.copy(campaign["gel"], tables / par.SELECTION_FILE)
    loads, built = [], []
    original = fin.rebuild_verified_model
    monkeypatch.setattr(fin, "rebuild_verified_model",
                        lambda *a, **k: built.append(original(*a, **k)) or built[-1])
    with caplog.at_level(logging.INFO):
        result = fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"],
                                    campaign["val"], lambda: (loads.append(1), campaign["test"])[1],
                                    tables_dir=tables, overrides=FAST)
    sel = campaign["selection"]
    targets = {t["config_id"] for t in sel["retrain_targets"]}
    assert loads == [1]  # un seul chargement du test
    assert set(result["configs"]) == targets and len(targets) in (2, 3)  # D7-05 : rien d'autre
    assert len(built) == len({t["run_key"] for t in sel["retrain_targets"]}) == 10 * len(targets)
    events = [r for r in caplog.records if r.getMessage() == "final_retrain"]
    assert len(events) == len(built) and all(r.fields["match"] for r in events)
    adam_ids = {r["config_id"] for r in sel["adam"]["ranking"]}
    assert len(targets & adam_ids) == 1  # le meilleur adam seul, jamais les trois autres
    # metriques de test = oracle NumPy sur le jeu de test, distinctes de celles de validation
    xs, ys = campaign["test"]
    xv, yv = campaign["val"]
    rows = _read_csv(tables / "mlp_2d_final_test_runs.csv")
    by_key = {r["run_key"]: r for r in rows}
    for model in built:
        key = next(k for k, r in by_key.items() if int(r["seed"]) == model.seed and
                   nom.weights_sha256(model.model) == nom.load_record(campaign["store"] / f"{k}.json")["weights_sha256"])
        pred = model.predict(xs)
        assert float(by_key[key]["test_mse"]) == pytest.approx(np.mean((ys - pred) ** 2), rel=1e-12)
        assert float(by_key[key]["test_mse"]) != pytest.approx(np.mean((yv - model.predict(xv)) ** 2), rel=1e-6)
    assert result["selection_files_sha256"][par.SELECTION_FILE] == nom.sha256_file(tables / par.SELECTION_FILE)
    assert result["figure_network"]["note"].startswith("un cas parmi 10")
    stats = _read_csv(tables / "mlp_2d_final_test_stats.csv")
    assert {r["metric"] for r in stats} == set(nom.METRICS) and len(stats) == 4 * len(targets)
    # idempotence : meme gel, memes sorties
    first = {p.name: p.read_bytes() for p in tables.iterdir() if p.name.startswith("mlp_2d_final")}
    fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"], campaign["val"],
                       lambda: campaign["test"], tables_dir=tables, overrides=FAST)
    second = {p.name: p.read_bytes() for p in tables.iterdir() if p.name.startswith("mlp_2d_final")}
    first.pop("mlp_2d_final_evaluation.json"), second.pop("mlp_2d_final_evaluation.json")
    assert first == second


def test_relance_refusee_si_le_gel_change_apres_l_evaluation(campaign, tmp_path):
    import shutil
    tables = tmp_path / "t"
    tables.mkdir()
    shutil.copy(campaign["gel"], tables / par.SELECTION_FILE)
    (tables / fin.EVALUATION_FILE).write_text(json.dumps({"selection_sha256": "0" * 64}), encoding="utf-8")
    loads = []
    with pytest.raises(fin.SelectionError, match="relance refusee"):
        fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"],
                           campaign["val"], lambda: loads.append(1), tables_dir=tables, overrides=FAST)
    assert loads == []


def test_fichiers_de_selection_modifies_pendant_l_evaluation_detectes(campaign, tmp_path):
    import shutil
    tables = tmp_path / "t"
    tables.mkdir()
    shutil.copy(campaign["gel"], tables / par.SELECTION_FILE)
    (tables / fin.HSTAR_FILE).write_text("{}", encoding="utf-8")

    def tamper():
        (tables / fin.HSTAR_FILE).write_text('{"h_star": 5}', encoding="utf-8")  # le test modifie la selection
        return campaign["test"]

    with pytest.raises(fin.SelectionError, match="ont change"):
        fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"], campaign["train"],
                           campaign["val"], tamper, tables_dir=tables, overrides=FAST)


# ------------------------------------------------------------------------------------------
# AE7-01 : sorties de l'evaluation finale contre des oracles independants du gel et du code
# ------------------------------------------------------------------------------------------
# Les oracles ci-dessous ne lisent que le cache (JSON bruts, sans load_record) et ne
# reutilisent ni src.utils.metrics, ni les fonctions de selection, ni le gel : une erreur
# commune au gel et a l'evaluation (role pointant vers une mauvaise configuration) ou une
# metrique fausse ne peut donc pas se compenser. Le jeu de test employe (53 points, autre
# fonction, autre graine) differe de l'apprentissage (200) et de la validation (40).
_WIDTHS = (5, 10, 20, 50, 100, 200)  # litteraux de l'enonce, pas les constantes du code
_ADAM_GRID = {(1e-3, 32), (1e-3, 200), (1e-2, 32), (1e-2, 200)}
_ROLES = {"final", "lbfgs_h_star", "adam_selected"}


def _distinct_test_set():
    rng = np.random.default_rng(2026)
    x = rng.random((53, 2))
    return x, np.cos(5 * x[:, 0]) + x[:, 1] ** 2 + 0.05 * rng.standard_normal(53)


def _oracle_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mse = float(np.mean((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "r2": 1.0 - float(np.sum((y - p) ** 2)) / ss_tot,
            "max_abs_error": float(np.max(np.abs(y - p)))}


def _raw_store(store):
    """Enregistrements du cache, lus en JSON brut : [(config, graine, record)]."""
    out = []
    for path in sorted(Path(store).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        out.append((rec["identity"]["config"], rec["identity"]["seed"], rec))
    return out


def _mean_validation_by_config(raw):
    groups = {}
    for cfg, seed, rec in raw:
        groups.setdefault(json.dumps(cfg, sort_keys=True), []).append(rec)
    assert all(len(v) == 10 for v in groups.values())
    return {k: (json.loads(k), float(np.mean([r["validation_metrics"]["mse"] for r in v])))
            for k, v in groups.items()}


def _expected_role_configs(store):
    """Configuration attendue de chacun des trois roles, recalculee depuis la validation brute."""
    by_cfg = _mean_validation_by_config(_raw_store(store))
    nominal = {c["hidden_layer_sizes"][0]: m for c, m in by_cfg.values()
               if c["solver"] == "lbfgs" and c["activation"] == "tanh" and c["alpha"] == 1e-4
               and len(c["hidden_layer_sizes"]) == 1 and c["hidden_layer_sizes"][0] in _WIDTHS}
    assert set(nominal) == set(_WIDTHS)
    h_star = min(_WIDTHS, key=lambda h: (nominal[h], h))
    adam = {k: (c, m) for k, (c, m) in by_cfg.items() if c["solver"] == "adam"}
    assert {(c["learning_rate_init"], c["batch_size"]) for c, _ in adam.values()} == _ADAM_GRID
    best_adam = min(adam.values(), key=lambda cm: cm[1])[0]
    best_final = min(by_cfg.values(), key=lambda cm: cm[1])[0]
    lbfgs = next(c for c, _ in by_cfg.values()
                 if c["solver"] == "lbfgs" and c["activation"] == "tanh" and c["alpha"] == 1e-4
                 and c["hidden_layer_sizes"] == [h_star])
    return {"h_star": h_star, "final": best_final, "lbfgs_h_star": lbfgs,
            "adam_selected": best_adam}


@pytest.fixture(scope="module")
def final_outputs(campaign, tmp_path_factory):
    """Evaluation finale complete sur un jeu de test distinct, plus les modeles reentraines."""
    import shutil
    from threadpoolctl import threadpool_limits
    tables = tmp_path_factory.mktemp("sorties_finales")
    shutil.copy(campaign["gel"], tables / par.SELECTION_FILE)
    test = _distinct_test_set()
    built = {}
    original = fin.rebuild_verified_model
    mp = pytest.MonkeyPatch()
    mp.setattr(fin, "rebuild_verified_model",
               lambda rec, *a, **k: built.setdefault(rec["run_key"], original(rec, *a, **k)))
    try:
        with threadpool_limits(limits=1):
            result = fin.evaluate_final(tables / par.SELECTION_FILE, campaign["store"],
                                        campaign["train"], campaign["val"], lambda: test,
                                        tables_dir=tables, overrides=FAST)
    finally:
        mp.undo()
    return {"tables": tables, "result": result, "built": built, "test": test,
            "runs": _read_csv(tables / "mlp_2d_final_test_runs.csv"),
            "stats": _read_csv(tables / "mlp_2d_final_test_stats.csv"),
            "comparison": _read_csv(tables / "mlp_2d_solver_comparison.csv"),
            "expected": _expected_role_configs(campaign["store"])}


def test_jeu_de_test_distinct_et_roles_attendus_discriminants(campaign, final_outputs):
    """Garde de puissance : les trois roles visent trois configurations differentes, et
    les metriques calculees sur le test different de celles de validation et d'apprentissage."""
    exp = final_outputs["expected"]
    keys = {json.dumps(exp[r], sort_keys=True) for r in _ROLES}
    assert len(keys) == 3
    alpha_variant = [c for c, _, _ in _raw_store(campaign["store"])
                     if c["hidden_layer_sizes"] == exp["lbfgs_h_star"]["hidden_layer_sizes"]
                     and c["solver"] == "lbfgs" and c["alpha"] != 1e-4]
    assert alpha_variant  # une variante d'alpha existe dans le cache (cible possible d'une erreur)
    assert len(final_outputs["test"][1]) not in (len(campaign["val"][1]), len(campaign["train"][1]))


def test_chaque_role_pointe_vers_la_configuration_attendue(campaign, final_outputs):
    """A03 : l'identite evaluee sous chaque role est comparee a la definition attendue
    (recalculee depuis le cache), jamais au gel produit par le meme code."""
    exp = final_outputs["expected"]
    lit = exp["lbfgs_h_star"]
    # definition litterale du role lbfgs H* : lbfgs, tanh, (H*,), alpha = 1e-4
    assert (lit["solver"], lit["activation"], lit["hidden_layer_sizes"], lit["alpha"]) == (
        "lbfgs", "tanh", [exp["h_star"]], 1e-4)
    adam = exp["adam_selected"]
    assert adam["solver"] == "adam" and adam["hidden_layer_sizes"] == [exp["h_star"]]
    assert adam["activation"] == "tanh" and adam["alpha"] == 1e-4
    assert (adam["learning_rate_init"], adam["batch_size"]) in _ADAM_GRID
    by_role = {}
    for row in final_outputs["runs"]:
        record = json.loads((campaign["store"] / f"{row['run_key']}.json").read_text(encoding="utf-8"))
        assert record["identity"]["seed"] == int(row["seed"])
        for role in row["roles"].split(";"):
            by_role.setdefault(role, []).append((int(row["seed"]), record["identity"]["config"]))
    assert set(by_role) == _ROLES
    for role, entries in by_role.items():
        assert sorted(s for s, _ in entries) == sorted(SEEDS_MONTE_CARLO)  # 10 graines distinctes
        assert all(cfg == exp[role] for _, cfg in entries), role


def test_roles_coherents_entre_les_quatre_sorties(final_outputs):
    runs = final_outputs["runs"]
    role_of = {}
    for r in runs:
        role_of.setdefault(r["config_id"], r["roles"])
        assert role_of[r["config_id"]] == r["roles"]
    for table in ("stats", "comparison"):
        assert {r["config_id"]: r["roles"] for r in final_outputs[table]} == role_of
    configs = final_outputs["result"]["configs"]
    assert {cid: ";".join(c["roles"]) for cid, c in configs.items()} == role_of
    for cid, c in configs.items():  # etiquette lisible coherente avec l'identite du role
        assert c["label"] == next(r["label"] for r in final_outputs["stats"] if r["config_id"] == cid)


def test_metriques_de_test_de_chaque_run_egales_a_l_oracle(final_outputs):
    """A04 : EQM, RMSE, R2 et erreur maximale de chaque run contre un oracle NumPy."""
    x, y = final_outputs["test"]
    assert len(final_outputs["runs"]) == 30
    swapped = 0
    for row in final_outputs["runs"]:
        pred = final_outputs["built"][row["run_key"]].predict(x)
        oracle = _oracle_metrics(y, pred)
        for name, value in oracle.items():
            assert float(row[f"test_{name}"]) == pytest.approx(value, rel=1e-9, abs=1e-12), name
        # R2 a arguments inverses (mutation A04) : distinct de la valeur attendue
        ss = float(np.sum((pred - np.mean(pred)) ** 2))
        if ss > 0:
            reversed_r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss
            swapped += not np.isclose(reversed_r2, oracle["r2"], rtol=1e-3)
    assert swapped >= 25  # la garde de puissance : l'inversion est visible sur presque tous les runs


def test_metriques_de_test_distinctes_de_validation_et_apprentissage(campaign, final_outputs):
    """Un jeu de donnees confondu (validation ou apprentissage au lieu du test) est detecte."""
    x, y = final_outputs["test"]
    for key in ("val", "train"):
        xo, yo = campaign[key]
        for row in final_outputs["runs"][::3]:
            model = final_outputs["built"][row["run_key"]]
            other = _oracle_metrics(yo, model.predict(xo))
            for name in ("mse", "r2", "max_abs_error"):
                assert float(row[f"test_{name}"]) != pytest.approx(other[name], rel=1e-6), (key, name)


def test_statistiques_et_comparaison_recalculees_depuis_les_runs(campaign, final_outputs):
    runs = {}
    for r in final_outputs["runs"]:
        runs.setdefault(r["config_id"], []).append(r)
    assert len(final_outputs["stats"]) == 4 * len(runs)
    for row in final_outputs["stats"]:
        values = np.array([float(r[f"test_{row['metric']}"]) for r in runs[row["config_id"]]])
        assert len(values) == 10
        expected = {"mean": values.mean(), "std": values.std(ddof=1), "median": np.median(values),
                    "min": values.min(), "max": values.max()}
        for name, value in expected.items():
            assert float(row[name]) == pytest.approx(value, rel=1e-9, abs=1e-12), (row["metric"], name)
        stored = final_outputs["result"]["configs"][row["config_id"]]["test"][row["metric"]]
        assert stored["mean"] == pytest.approx(float(row["mean"]), rel=1e-12)
    for row in final_outputs["comparison"]:
        mse = np.array([float(r["test_mse"]) for r in runs[row["config_id"]]])
        val = np.array([json.loads((campaign["store"] / f"{r['run_key']}.json").read_text(
            encoding="utf-8"))["validation_metrics"]["mse"] for r in runs[row["config_id"]]])
        assert int(row["n_runs"]) == 10
        for name, value in {"test_mse_mean": mse.mean(), "test_mse_std": mse.std(ddof=1),
                            "test_mse_min": mse.min(), "test_mse_max": mse.max(),
                            "val_mse_mean": val.mean()}.items():
            assert float(row[name]) == pytest.approx(value, rel=1e-9), name


def test_reseau_des_figures_metriques_oracle_et_graine_de_validation_minimale(campaign, final_outputs):
    """Le reseau des figures est la graine d'EQM de validation minimale de la configuration
    finale attendue ; ses metriques de test sont celles de l'oracle."""
    exp_final = final_outputs["expected"]["final"]
    final_runs = [(seed, rec) for cfg, seed, rec in _raw_store(campaign["store"]) if cfg == exp_final]
    best_seed = min(final_runs, key=lambda sr: (sr[1]["validation_metrics"]["mse"],
                                                SEEDS_MONTE_CARLO.index(sr[0])))[0]
    flagged = [r for r in final_outputs["runs"] if r["is_figure_network"] == "True"]
    assert len(flagged) == 1 and int(flagged[0]["seed"]) == best_seed
    figure = final_outputs["result"]["figure_network"]
    assert figure["seed"] == best_seed
    x, y = final_outputs["test"]
    model = final_outputs["built"][flagged[0]["run_key"]]
    for name, value in _oracle_metrics(y, model.predict(x)).items():
        assert figure["test_metrics"][name] == pytest.approx(value, rel=1e-9, abs=1e-12), name
