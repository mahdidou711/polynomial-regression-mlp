"""Etape 8 : formateur LaTeX, registre, controles statiques, comparaison et pipeline d'export.

Les tables sont validees par analyse statique ; la compilabilite reelle est constatee par
compilation de l'apercu autonome.
"""
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

import main as main_cli
from src.export import build, checks, pipeline, registry
from src.export import latex_tables as lt
from src.export.layout import REPO, Layout
from tests.helpers_2d import one_thread  # noqa: F401

FROZEN_FILE_SHA256 = "173153edb9d903e3d68b1c43c9a6e88f424c66b2105ed28319a4e6460753395f"


# ------------------------------------------------------------------ formateur
class TestFormatting:
    def test_sci_a_trois_chiffres_significatifs_et_exposant_signe(self):
        assert lt.format_sci(1.2345e-5) == "1.23e-05"
        assert lt.format_sci(-4.5678e3, 3) == "-4.57e+03"
        assert lt.format_sci(5e-5, 1) == "5e-05"
        with pytest.raises(ValueError):
            lt.format_sci(1.0, 0)

    def test_fixed_et_int(self):
        col = lt.Col("x", "x", "fixed", 4)
        assert lt.format_cell("0.996689603", col) == "0.9967"
        assert lt.format_cell("12.0", lt.Col("n", "n", "int")) == "12"

    def test_bool_texte_et_valeurs_manquantes(self):
        col = lt.Col("b", "b", "bool")
        assert lt.format_cell("True", col) == "oui" and lt.format_cell(False, col) == "non"
        with pytest.raises(ValueError):
            lt.format_cell("peut-etre", col)
        for missing in ("", "nan", None, float("nan")):
            assert lt.format_cell(missing, lt.Col("x", "x")) == "---"

    def test_infini_ecrit_en_mathematiques_sans_dollar(self):
        out = lt.format_cell(float("inf"), lt.Col("k", "k"))
        assert out == r"\(\infty\)" and "$" not in out

    def test_echappement_et_accents_ascii(self):
        assert lt.escape_tex("a_b & 50% #1") == r"a\_b \& 50\% \#1"
        assert lt.escape_tex("régime ç") == r"r\'egime \c{c}"
        assert lt.escape_tex("x").isascii()
        with pytest.raises(ValueError, match="non ASCII"):
            lt.escape_tex("日本")

    def test_table_format_calcule_sur_les_chaines_ecrites(self):
        sci = lt.Col("v", "v", "sci", 3)
        assert lt.table_format(["1.23e-04", "9.99e+17"], sci) == "1.2e-2"
        assert lt.table_format(["-1.23e-04"], sci) == "-1.2e-2"
        assert lt.table_format(["1.23e-104"], sci) == "1.2e-3"
        fixed = lt.Col("v", "v", "fixed", 4)
        assert lt.table_format(["0.9967", "12.1000"], fixed) == "2.4"
        assert lt.table_format(["---", r"\(\infty\)"], sci) == "1.0"  # rien de numerique

    def test_rendu_d_un_tabular_complet(self):
        cols = [lt.Col("d", r"\(d\)", "int"), lt.Col("e", "EQM"), lt.Col("r", "regime", "text")]
        rows = [{"d": 1, "e": 0.0123, "r": "strict"}, {"d": 2, "e": None, "r": "a_b"}]
        tex = lt.render_table(cols, rows, "src.csv")
        lines = tex.splitlines()
        assert lines[2] == r"\begin{tabular}{r S[table-format=1.2e-2] l}"
        assert lines[4] == r"{\(d\)} & {EQM} & {regime} \\"
        assert lines[6] == r"1 & 1.23e-02 & strict \\"
        assert lines[7] == r"2 & {---} & a\_b \\"  # valeur manquante protegee dans une colonne S
        assert tex.rstrip().endswith(r"\end{tabular}") and checks.check_tex_text(tex) == []

    def test_longtable_au_dela_du_seuil(self):
        cols = [lt.Col("d", "d", "int"), lt.Col("e", "e")]
        rows = [{"d": i, "e": 1e-3 * i} for i in range(1, lt.LONGTABLE_ROWS + 2)]
        tex = lt.render_table(cols, rows, "s")
        assert lt.environment_of(tex) == "longtable" and r"\endhead" in tex
        assert checks.check_tex_text(tex) == []
        assert lt.environment_of(lt.render_table(cols, rows[:3], "s")) == "tabular"

    def test_csv_aller_retour_exact(self, tmp_path):
        value = 0.1 + 0.2
        path = lt.write_csv(tmp_path / "t.csv", ["a", "b"], [{"a": value, "b": None}])
        row = lt.read_csv(path)[0]
        assert float(row["a"]) == value and row["b"] == ""
        assert path.read_bytes().endswith(b"\n") and b"\r" not in path.read_bytes()


# ------------------------------------------------------------------ analyse statique
GOOD = lt.render_table([lt.Col("d", "d", "int"), lt.Col("e", "e")],
                       [{"d": 1, "e": 0.5}, {"d": 2, "e": 0.25}], "s")


class TestStaticAnalysis:
    @pytest.mark.parametrize("mutation, expected", [
        (lambda t: t.replace("1.", "$1.", 1), "signe $"),
        (lambda t: t.replace(r"\midrule", r"\hline"), "interdit"),
        (lambda t: t.replace(r"\bottomrule", ""), "regle booktabs absente"),
        (lambda t: t.replace("5.00e-01", "5.00e-01 & 7"), "cellules pour"),
        (lambda t: t.replace("\\begin{tabular}{r", "\\begin{tabular}{r r"), "cellules pour"),
        (lambda t: t.replace("{e}", "{e"), "accolades"),
        (lambda t: t.replace("{e}", "{é}"), "non ASCII"),
        (lambda t: t.replace("5.00e-01", "cinq"), "cellule S non numerique"),
        (lambda t: t.replace("5.00e-01", ""), "cellule vide"),
        (lambda t: t + r"\caption{x}", "interdit"),
        (lambda t: t.replace(r"\end{tabular}", r"\end{longtable}"), "un seul environnement"),
        (lambda t: t.replace(r"\(d\)", r"\(d"), "delimiteurs"),
        (lambda t: t.replace("2 & 2.50e-01 \\\\", "2 & 2.50e-01"), "fin de ligne"),
        (lambda t: "% rien\n", "fragment vide"),
    ])
    def test_chaque_defaut_est_detecte(self, mutation, expected):
        text = GOOD.replace(r"{d}", r"{\(d\)}", 1)
        bad = mutation(text)
        assert bad != text
        issues = checks.check_tex_text(bad)
        assert any(expected in i for i in issues), issues

    def test_fragment_conforme(self):
        assert checks.check_tex_text(GOOD) == []

    def test_comptage_des_colonnes(self):
        assert checks.count_columns("r S[table-format=1.2e-2] l c p{3cm} @{} S[table-format=4]") == 6
        assert checks.count_columns("lll") == 3


# ------------------------------------------------------------------ registre
class TestRegistry:
    def test_identifiants_uniques_conformes_et_figures_sans_collision(self):
        stems = [t.stem for t in registry.TABLES]
        assert len(set(stems)) == len(stems) and all(checks.STEM.match(s) for s in stems)
        figs = [f.stem for f in registry.FIGURES]
        assert len(set(figs)) == len(figs) and all(checks.STEM.match(s) for s in figs)
        assert not set(stems) & set(registry.CSV_ONLY)
        for spec in registry.TABLES:  # une table derivee n'ecrase jamais un CSV source
            if spec.derive is not None:
                assert spec.stem not in registry.CSV_ONLY and spec.stem + ".csv" not in (
                    registry.SOURCES_1D + registry.SOURCES_MLP_CACHE + registry.SOURCES_CARRIED)
                assert spec.inputs
            else:
                assert spec.source == spec.stem


    def test_figures_3d_png_seul_et_2d_double_format(self):
        for f in registry.FIGURES:
            three_d = f.stem.startswith("mlp_2d_") and f.stem in {
                "mlp_2d_planche_3d", "mlp_2d_surface_exacte", "mlp_2d_surface_predite",
                "mlp_2d_surface_residus", "mlp_2d_activation_tanh_relu"}
            assert f.formats == (("png",) if three_d else ("pdf", "png")), f.stem
        assert {f.stem for f in registry.FIGURES if f.needs_models} == {
            "mlp_2d_parite", "mlp_2d_residus_carte", "mlp_2d_planche_3d", "mlp_2d_surface_exacte",
            "mlp_2d_surface_predite", "mlp_2d_surface_residus", "mlp_2d_activation_tanh_relu"}

    def test_les_noms_des_figures_des_etapes_3_a_7_sont_conserves(self):
        """Les audits et passations renvoient a ces noms : ils ne doivent pas changer."""
        names = {n for f in registry.FIGURES for n in f.files()}
        assert {"nominal_1d_fits.pdf", "nominal_1d_campaign_50_graines.png", "exploration_nb_ex.pdf",
                "exploration_nb_ex_seed42.png", "mlp_2d_planche_3d.png", "mlp_2d_parite.pdf",
                "mlp_2d_activation_tanh_relu.png"} <= names
        assert sum(1 for f in registry.FIGURES for _ in f.files()) == 37  # 16 figures en PDF + PNG et 5 en PNG seul

    def test_toutes_les_colonnes_existent_dans_les_sources(self):
        src = registry.Sources(REPO.tables)
        for spec in registry.TABLES:
            rows = build.table_rows(spec, src)
            assert rows, spec.stem
            for col in spec.columns:
                assert col.key in rows[0], (spec.stem, col.key)


# ------------------------------------------------------------------ derivations (oracles independants)
class TestDerivations:
    src = registry.Sources(REPO.tables)

    def test_synthese_nb_ex_recalculee_a_la_main(self):
        rows = {r["nb_ex"]: r for r in registry.derive_nb_ex_synthese(self.src)}
        assert list(rows) == ["10", "15", "20", "50", "100", "200"]
        stats = self.src.rows("exploration_nb_ex_stats")
        block = [r for r in stats if r["nb_ex"] == "20"]
        medians = {int(r["degree"]): float(r["eqmt_median"]) for r in block}
        best = min(medians, key=lambda d: (medians[d], d))
        assert rows["20"]["degree_min"] == best and rows["20"]["eqmt_median_min"] == medians[best]
        assert rows["20"]["eqmt_median_d12"] == medians[12] and rows["20"]["n_runs"] == 50
        assert rows["20"]["cstar_mode"] == 6  # distribution nominale (mode de C*, 50 graines)

    def test_synthese_sigma_zero_est_une_realisation_unique(self):
        rows = {r["sigma"]: r for r in registry.derive_sigma_synthese(self.src)}
        assert rows["0.0"]["n_runs"] == 1 and rows["0.2"]["n_runs"] == 50

    def test_tables_de_sensibilite_partitionnent_le_plan(self):
        counts = {name: len(fn(self.src)) for name, fn in (
            ("TOPO", registry.derive_topologie), ("REG", registry.derive_regularisation),
            ("ACT", registry.derive_activation), ("SOLV", registry.derive_solveur))}
        assert counts == {"TOPO": 2, "REG": 6, "ACT": 2, "SOLV": 5}
        assert len(registry.derive_configurations(self.src)) == len(self.src.rows("mlp_2d_config_stats"))
        labels = [r["label"] for r in registry.derive_configurations(self.src)]
        assert len(set(labels)) == len(labels)  # adam : lr et lot distinguent les configurations
        solveur = registry.derive_solveur(self.src)
        assert solveur[0]["learning_rate_init"] is None and solveur[1]["learning_rate_init"] == 1e-3

    def test_synthese_finale_egale_aux_statistiques_de_l_evaluation(self):
        stats = self.src.rows("mlp_2d_final_test_stats")
        for row in registry.derive_final_synthese(self.src):
            mse = next(r for r in stats if r["label"] == row["label"] and r["metric"] == "mse")
            assert row["mse_mean"] == float(mse["mean"]) and row["mse_max"] == float(mse["max"])
        assert [r["roles"] for r in registry.derive_final_synthese(self.src)] == [
            "final", "lbfgs_h_star", "adam_selected"]

    def test_provenance_des_donnees_2d(self):
        rows = registry.derive_provenance(self.src)
        assert [r["n"] for r in rows] == [200, 200, 400, 1600]
        assert rows[0]["sha256"] == hashlib.sha256(self.src.raw_csv.read_bytes()).hexdigest()[:16]
        assert rows[0]["sha256"] == "997facd8edf592eb"  # empreinte du CSV officiel (etat d'avancement §4)

    def test_aucune_derivation_ne_lit_la_base_de_test(self):
        import inspect
        text = inspect.getsource(registry) + inspect.getsource(build)
        assert "genere_base_test" not in text and "prepare_data_2d" not in text


# ------------------------------------------------------------------ metadonnees de figures (AE4-05, AE7-08)
class TestFigureMetadata:
    params = build.figure_parameters(registry.Sources(REPO.tables))

    def test_ae4_05_graines_repetitions_et_realisation_unique(self):
        sigma = self.params["exploration_sigma"]
        assert sigma["graines"] == "1000 \u00e0 1049" and sigma["n_graines"] == 50
        assert sigma["repetitions_par_sigma"]["0.0"] == 1
        assert all(sigma["repetitions_par_sigma"][s] == 50 for s in ("0.05", "0.1", "0.2", "0.5", "1.0"))
        assert sigma["nb_ex_fixe"] == 20
        assert self.params["exploration_nb_ex"]["sigma_fixe"] == 0.2

    def test_ae7_08_ordre_de_la_grille_adam_et_avertissements(self):
        mlp = self.params["mlp_campagnes"]
        assert mlp["ordre_grille_adam"] == ["lr=0.001, lot=32", "lr=0.001, lot=200",
                                            "lr=0.01, lot=32", "lr=0.01, lot=200"]
        assert mlp["avertissements_adam_dans_l_ordre_de_la_grille"] == [5, 10, 6, 10]
        assert "borne 1 exclue" in self.params["mlp_modele"]["grille_de_rendu"]

    def test_le_manifeste_de_figures_couvre_le_registre(self):
        data = json.loads(REPO.figures_manifest.read_text(encoding="utf-8"))
        assert [f["stem"] for f in data["figures"]] == [f.stem for f in registry.FIGURES]
        by = {f["stem"]: f for f in data["figures"]}
        assert "borne 1 est exclue" in by["mlp_2d_planche_3d"]["note"]
        assert by["mlp_2d_parite"]["parametres"]["graine_reseau_des_figures"] == 1819


# ------------------------------------------------------------------ depot reel : controles statiques
class TestRepositoryArtifacts:
    @pytest.mark.parametrize("check", [checks.check_files, checks.check_figure_formats,
                                       checks.check_tex_files, checks.check_preview,
                                       checks.check_manifest, checks.check_final_evaluation],
                             ids=lambda c: c.__name__)
    def test_controle_passe_sur_le_depot(self, check):
        result = check(REPO)
        assert result.status == checks.PASSED, result.details

    def test_le_gel_de_selection_est_inchange(self):
        digest = hashlib.sha256((REPO.tables / "mlp_2d_selection.json").read_bytes()).hexdigest()
        assert digest == FROZEN_FILE_SHA256
        manifest = json.loads(REPO.manifest.read_text(encoding="utf-8"))
        assert manifest["protocole"]["gel_selection_sha256"] == FROZEN_FILE_SHA256

    def test_chaque_tex_a_un_csv_du_meme_nom_et_toutes_les_lignes(self):
        for spec in registry.TABLES:
            csv_rows = lt.read_csv(REPO.tables / f"{spec.stem}.csv")
            tex = (REPO.tables / f"{spec.stem}.tex").read_text(encoding="utf-8")
            data_lines = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and "&" in ln
                          and not ln.startswith("{")]
            assert len(data_lines) == len(csv_rows), spec.stem

    def test_resume_csv_est_la_liste_plate_du_manifeste(self):
        manifest = json.loads(REPO.manifest.read_text(encoding="utf-8"))
        rows = lt.read_csv(REPO.tables / registry.SUMMARY_FILE)
        assert [r["path"] for r in rows] == [a["path"] for a in manifest["artefacts"]]
        assert [r["sha256"] for r in rows] == [a["sha256"] for a in manifest["artefacts"]]

    def test_manifeste_sans_horodatage_ni_sha_git(self):
        text = REPO.manifest.read_text(encoding="utf-8")
        assert "git" not in text.lower() and "202" not in text.split('"artefacts"')[0].replace("2026", "")
        assert checks.check_hygiene(REPO).status in (checks.PASSED, checks.WARNING)


# ------------------------------------------------------------------ depot reel copie : defauts injectes
@pytest.fixture
def layout_copy(tmp_path):
    out = Layout(tmp_path / "copy")
    for rel, path in pipeline.tracked_files(REPO).items():
        target = out.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return out


class TestInjectedDefects:
    def test_copie_conforme(self, layout_copy):
        assert all(c.status in (checks.PASSED, checks.WARNING) for c in checks.run_static_checks(layout_copy))
        assert pipeline.is_clean(pipeline.compare(layout_copy, REPO))

    def test_manifeste_detecte_un_octet_modifie(self, layout_copy):
        target = layout_copy.tables / "nominal_1d_seed42.csv"
        target.write_bytes(target.read_bytes().replace(b"0.47234770197568643", b"0.47234770197568644"))
        result = checks.check_manifest(layout_copy)
        assert result.status == checks.FAILED and any("nominal_1d_seed42.csv" in d for d in result.details)

    def test_fichier_non_declare_est_signale(self, layout_copy):
        (layout_copy.tables / "table_orpheline.tex").write_text("x", encoding="utf-8")
        (layout_copy.fig_mlp / "mlp_2d_orpheline.png").write_bytes(b"x")
        details = checks.check_files(layout_copy).details
        assert any("table_orpheline.tex" in d for d in details)
        assert any("mlp_2d_orpheline.png" in d for d in details)

    def test_figure_manquante_et_pdf_de_figure_3d(self, layout_copy):
        (layout_copy.fig_poly / "nominal_1d_fits.pdf").unlink()
        assert any("nominal_1d_fits.pdf" in d for d in checks.check_files(layout_copy).details)
        (layout_copy.fig_mlp / "mlp_2d_planche_3d.pdf").write_bytes(b"%PDF-1.4")
        assert any("figure 3D" in d for d in checks.check_figure_formats(layout_copy).details)

    def test_png_de_mauvaise_resolution_ou_entete(self, layout_copy):
        path = layout_copy.fig_poly / "nominal_1d_fits.png"
        data = path.read_bytes()
        i = data.index(b"pHYs")
        path.write_bytes(data[:i + 4] + (2000).to_bytes(4, "big") + data[i + 8:])
        assert any("DPI" in d for d in checks.check_figure_formats(layout_copy).details)
        path.write_bytes(b"GIF89a" + b"\0" * 10)
        assert any("en-tete PNG" in d for d in checks.check_figure_formats(layout_copy).details)

    def test_tex_corrompu_dans_le_depot(self, layout_copy):
        path = layout_copy.tables / "nominal_1d_cstar_50_graines.tex"
        path.write_text(path.read_text(encoding="utf-8").replace(r"\bottomrule", r"\hline"), encoding="utf-8")
        assert checks.check_tex_files(layout_copy).status == checks.FAILED

    def test_apercu_incoherent(self, layout_copy):
        text = layout_copy.preview.read_text(encoding="utf-8")
        layout_copy.preview.write_text(text.replace("\\usepackage{siunitx}\n", ""), encoding="utf-8")
        assert any("siunitx" in d for d in checks.check_preview(layout_copy).details)
        layout_copy.preview.write_text(text.replace("\\input{\\tablesroot/nominal_1d_seed42.tex}",
                                                    "\\input{/home/x/nominal_1d_seed42.tex}"), encoding="utf-8")
        assert checks.check_preview(layout_copy).status == checks.FAILED

    def test_longtable_dans_une_boite_est_refuse(self, layout_copy):
        text = layout_copy.preview.read_text(encoding="utf-8")
        block = "\\input{\\tablesroot/exploration_nb_ex_stats.tex}"
        assert block in text and "adjustbox" not in text.split(block)[0].split("\\subsection*")[-1]
        bad = text.replace(block, "\\begin{adjustbox}{max width=\\linewidth}\n" + block + "\n\\end{adjustbox}")
        layout_copy.preview.write_text(bad, encoding="utf-8")
        assert any("longtable dans une boite" in d for d in checks.check_preview(layout_copy).details)

    def test_apercu_garde_les_legendes_et_corrige_le_compteur_longtable(self):
        long_name = "exploration_nb_ex_stats"
        text = build.render_preview({long_name})
        sections = text.split(r"\subsection*")[1:]
        assert len(sections) == len(registry.TABLES)
        for spec, section in zip(registry.TABLES, sections):
            caption = section.index(r"\captionof{table}")
            input_pos = section.index(r"\input{\tablesroot/")
            assert caption < input_pos
            if spec.stem == long_name:
                assert section.index(r"\addtocounter{table}{-1}") < input_pos
                assert r"\end{minipage}" in section[caption:input_pos]
            else:
                assert r"\end{minipage}" not in section[caption:input_pos]

    def test_evaluation_finale_incoherente(self, layout_copy):
        path = layout_copy.tables / "mlp_2d_final_test_stats.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        parts = lines[1].split(",")
        parts[4] = repr(float(parts[4]) * 1.01)  # moyenne de l'EQMT truquee
        lines[1] = ",".join(parts)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = checks.check_final_evaluation(layout_copy)
        assert result.status == checks.FAILED and any("recalcul" in d for d in result.details)

    def test_evaluation_associee_a_un_autre_gel(self, layout_copy):
        path = layout_copy.tables / "mlp_2d_selection.json"
        path.write_text(path.read_text(encoding="utf-8").replace('"schema"', '"schema" ', 1), encoding="utf-8")
        details = checks.check_final_evaluation(layout_copy).details
        assert any("autre gel" in d for d in details)

    def test_hygiene_signale_sans_supprimer(self, layout_copy):
        stray = layout_copy.root / "results" / ".mcp.json"
        stray.write_text("{}", encoding="utf-8")
        result = checks.check_hygiene(layout_copy)
        assert result.status == checks.WARNING and stray.exists()

    def test_comparaison_et_synchronisation(self, layout_copy, tmp_path):
        other = Layout(tmp_path / "other")
        shutil.copytree(layout_copy.root, other.root)
        (other.tables / "nominal_1d_seed42.tex").write_text("x", encoding="utf-8")
        (other.tables / "nouveau.csv").write_text("a\n", encoding="utf-8")
        (other.fig_poly / "nominal_1d_fits.png").unlink()
        diff = pipeline.compare(other, layout_copy)
        assert diff["different"] == ["results/tables/nominal_1d_seed42.tex"]
        assert diff["missing_in_second"] == ["results/tables/nouveau.csv"]
        assert diff["missing_in_first"] == ["results/figures/polynomial/nominal_1d_fits.png"]
        assert not pipeline.is_clean(diff)
        changed = pipeline.sync(other, layout_copy, exclude={"results/tables/nominal_1d_seed42.tex"})
        assert changed == ["results/tables/nouveau.csv"]  # le fichier exclu n'est jamais copie
        only = pipeline.compare(other, layout_copy, only={"results/tables/nominal_1d_seed42.tex"})
        assert only["different"] == ["results/tables/nominal_1d_seed42.tex"]


# ------------------------------------------------------------------ garde-fous de main.py
class TestCommandLine:
    def test_reproduce_refuse_une_sortie_dans_le_depot(self, capsys):
        assert main_cli.main(["reproduce", "--out", str(REPO.root / "results" / "x")]) == 2
        assert "hors du depot" in capsys.readouterr().out

    def test_reproduce_refuse_un_repertoire_non_vide(self, tmp_path, capsys):
        (tmp_path / "x").write_text("1")
        assert main_cli.main(["reproduce", "--out", str(tmp_path)]) == 2
        assert "n'est pas vide" in capsys.readouterr().out

    def test_une_commande_est_obligatoire(self):
        with pytest.raises(SystemExit):
            main_cli.main([])

    def test_ae7_04_les_runners_ne_sont_jamais_appeles_sans_repertoire_isole(self):
        """Aucun appel de ``main`` d'un runner n'omet ``--tables-dir`` : la reecriture destructive
        des tables versionnees (AE7-04) est impossible depuis le pipeline."""
        import re
        text = Path(pipeline.__file__).read_text(encoding="utf-8")
        calls = []
        for match in re.finditer(r"\w+\.main\(", text):
            depth, i = 1, match.end()
            while depth:  # parentheses equilibrees : l'argument peut contenir des appels
                depth += (text[i] == "(") - (text[i] == ")")
                i += 1
            calls.append(text[match.end():i - 1])
        assert len(calls) >= 5  # 2 runners 1D, lots B et C, evaluation finale
        for args in calls:
            assert "--tables-dir" in args or args == "argv", args
        assert text.count('"--tables-dir"') >= 4  # 2 runners 1D, liste commune des lots B et C, evaluation finale

    def test_ae7_06_le_motif_mcp_json_n_est_plus_ancre(self):
        lines = (REPO.root / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".mcp.json" in lines and "/.mcp.json" not in lines

    def test_main_fixe_un_seul_thread_avant_numpy(self):
        text = Path(main_cli.__file__).read_text(encoding="utf-8")
        assert text.index("OMP_NUM_THREADS") < text.index("import argparse") < text.index("from src.export")


# ------------------------------------------------------------------ pipeline complet (AE4-04)
@pytest.mark.usefixtures("one_thread")
class TestFullPipeline:
    def test_regeneration_complete_identique_au_depot(self, tmp_path):
        """Derive nulle : tables 1D recalculees, tables MLP depuis le cache, figures (reseaux inclus),
        tables .tex, manifestes : tout est identique octet a octet aux fichiers versionnes."""
        before = {p: p.read_bytes() for p in REPO.tables.iterdir() if p.is_file()}
        out = Layout(tmp_path / "out")
        info = pipeline.produce(out, REPO, with_models=True)
        # le depot n'est jamais ecrit par produce : toutes les tables y sont inchangees
        assert {p: p.read_bytes() for p in REPO.tables.iterdir() if p.is_file()} == before
        assert info["h_star"] == 50 and "regenerees" in info["model_figures"]
        assert pipeline.compare(out, REPO) == {"missing_in_first": [], "missing_in_second": [], "different": []}
        assert hashlib.sha256((out.tables / "mlp_2d_selection.json").read_bytes()).hexdigest() == FROZEN_FILE_SHA256

    def test_sans_modeles_les_figures_dependantes_sont_reportees(self, tmp_path):
        out = Layout(tmp_path / "out")
        info = pipeline.produce(out, REPO, with_models=False)
        assert "reportees" in info["model_figures"]
        for folder, name in pipeline.model_figure_names():
            assert (out.figure_dir(folder) / name).read_bytes() == (REPO.figure_dir(folder) / name).read_bytes()

    def test_un_cache_incomplet_arrete_avant_tout_entrainement(self, tmp_path, monkeypatch):
        from src.experiments import runner_2d_nominal as nom
        trained = []
        monkeypatch.setattr(nom, "run_strict", lambda *a, **k: trained.append(1))
        empty = tmp_path / "store"
        empty.mkdir()
        with pytest.raises(pipeline.ExportError, match="aucun entrainement"):
            pipeline.regenerate_mlp_tables(Layout(tmp_path / "o"), store=empty)
        assert trained == [] and not list(empty.iterdir())


# ------------------------------------------------------------------ determinisme de deux exports successifs
class TestSuccessiveExports:
    def test_les_constructions_derivees_sont_identiques_a_chaque_appel(self, layout_copy):
        """Tables .tex, CSV derives, metadonnees, apercu, manifeste et recapitulatif : memes octets
        a chaque construction, et memes octets que les fichiers versionnes."""
        def build_all():
            src = registry.Sources(layout_copy.tables)
            build.build_tables(layout_copy, src)
            build.build_figures_manifest(layout_copy, src)
            build.build_preview(layout_copy)
            build.build_manifest(layout_copy, src)
            return {rel: p.read_bytes() for rel, p in pipeline.tracked_files(layout_copy).items()}

        first, second = build_all(), build_all()
        assert first == second
        assert first == {rel: p.read_bytes() for rel, p in pipeline.tracked_files(REPO).items()}

    def test_manifeste_et_recapitulatif_ne_dependent_pas_de_l_ordre_du_systeme_de_fichiers(self, layout_copy):
        manifest = json.loads(layout_copy.manifest.read_text(encoding="utf-8"))
        paths = [a["path"] for a in manifest["artefacts"]]
        assert paths == sorted(paths) and len(set(paths)) == len(paths)

    def test_sync_n_ecrit_rien_quand_rien_ne_change(self, layout_copy, tmp_path):
        other = Layout(tmp_path / "other")
        shutil.copytree(layout_copy.root, other.root)
        assert pipeline.sync(other, layout_copy) == []


# ------------------------------------------------------------------ reproduction integrale (comparaisons)
class TestFullReproductionComparisons:
    def _record(self, **diag):
        base = {"n_iter": 10, "loss": 0.1, "elapsed_seconds": 1.0, "convergence_warnings": []}
        return {"run_key": "k", "weights_sha256": "w", "training_metrics": {"mse": 1.0},
                "validation_metrics": {"mse": 2.0}, "stop_cause": "none", "status": "completed",
                "diagnostics": {**base, **diag}}

    def test_le_condense_d_un_enregistrement_ignore_la_duree_seulement(self):
        a = pipeline.record_digest(self._record())
        assert a == pipeline.record_digest(self._record(elapsed_seconds=99.0))
        assert a != pipeline.record_digest(self._record(n_iter=11))
        assert a != pipeline.record_digest(self._record(loss=0.2))
        changed = self._record()
        changed["weights_sha256"] = "x"
        assert a != pipeline.record_digest(changed)

    def test_comparaison_de_depots_de_runs(self, tmp_path):
        from src.experiments import runner_2d_nominal as nom
        (a, b) = (tmp_path / "a", tmp_path / "b")
        for d in (a, b):
            d.mkdir()
        source = sorted(nom.NOMINAL_STORE.glob("*.json"))[:3]
        for p in source:
            shutil.copyfile(p, a / p.name)
            shutil.copyfile(p, b / p.name)
        assert pipeline.compare_stores(a, b) == []
        (b / source[1].name).unlink()
        assert pipeline.compare_stores(a, b) == [f"absent du second depot : {source[1].name}"]

    def test_depots_au_contenu_scientifique_different_sont_signales(self, tmp_path, monkeypatch):
        from src.experiments import runner_2d_nominal as nom
        a, b = tmp_path / "a", tmp_path / "b"
        for d in (a, b):
            d.mkdir()
        first = sorted(nom.NOMINAL_STORE.glob("*.json"))[0]
        for d in (a, b):
            shutil.copyfile(first, d / first.name)
        original = nom.load_record

        def load(path):
            record = original(path)
            if Path(path).parent == b:  # meme duree, poids differents : divergence scientifique
                record["weights_sha256"] = "0" * 64
            return record

        monkeypatch.setattr(nom, "load_record", load)
        assert pipeline.compare_stores(a, b) == [f"contenu scientifique different : {first.name}"]

    def test_gel_compare_hors_empreintes_d_enregistrements(self):
        freeze = json.loads((REPO.tables / "mlp_2d_selection.json").read_text(encoding="utf-8"))
        sci = pipeline.scientific_selection(freeze)
        other = json.loads(json.dumps(freeze))
        other["payload_sha256"] = "0" * 64
        other["records_digest"] = "1" * 64
        for record in other["records"]:
            record["payload_sha256"] = "2" * 64
        assert pipeline.scientific_selection(other) == sci
        other["h_star"] = {**other["h_star"], "h_star": 49}
        assert pipeline.scientific_selection(other) != sci
        other = json.loads(json.dumps(freeze))
        other["final"]["figure_network"]["seed"] = 42
        assert pipeline.scientific_selection(other) != sci

    def test_evaluation_comparee_hors_empreintes_du_gel(self):
        ev = json.loads((REPO.tables / "mlp_2d_final_evaluation.json").read_text(encoding="utf-8"))
        sci = pipeline.scientific_evaluation(ev)
        other = json.loads(json.dumps(ev))
        other["selection_sha256"] = "0" * 64
        other["environment"] = {"platform": "autre"}
        assert pipeline.scientific_evaluation(other) == sci
        label = next(iter(other["configs"]))
        other["configs"][label]["test"]["mse"]["mean"] *= 1.0000001
        assert pipeline.scientific_evaluation(other) != sci

    def test_comparaison_scientifique_tolere_les_durees_et_refuse_le_reste(self, layout_copy, tmp_path):
        other = Layout(tmp_path / "other")
        shutil.copytree(layout_copy.root, other.root)
        path = other.tables / "mlp_2d_runs.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",")
        col = header.index("elapsed_seconds")
        rows = [ln.split(",") for ln in lines[1:]]
        for r in rows:
            r[col] = "123.456"
        path.write_text("\n".join([lines[0]] + [",".join(r) for r in rows]) + "\n", encoding="utf-8")
        assert pipeline.compare(other, layout_copy)["different"] == ["results/tables/mlp_2d_runs.csv"]
        assert pipeline.compare_scientific(other, layout_copy)["different"] == []
        rows[0][header.index("val_mse")] = "9.9"
        path.write_text("\n".join([lines[0]] + [",".join(r) for r in rows]) + "\n", encoding="utf-8")
        assert pipeline.compare_scientific(other, layout_copy)["different"] == ["results/tables/mlp_2d_runs.csv"]
        # un tex different reste une difference (aucune tolerance hors fichiers volatils)
        shutil.copyfile(layout_copy.tables / "mlp_2d_runs.csv", path)
        (other.tables / "nominal_1d_seed42.tex").write_text("x", encoding="utf-8")
        assert pipeline.compare_scientific(other, layout_copy)["different"] == ["results/tables/nominal_1d_seed42.tex"]

    def test_reproduction_integrale_refusee_dans_le_depot_nominal(self, tmp_path):
        from src.experiments import runner_2d_nominal as nom
        with pytest.raises(pipeline.ExportError, match="depot nominal"):
            pipeline.regenerate_mlp_tables(Layout(tmp_path), nom.NOMINAL_STORE, train=True)

    def test_full_est_incompatible_avec_no_models(self, tmp_path, capsys):
        assert main_cli.main(["reproduce", "--full", "--no-models", "--out", str(tmp_path / "o")]) == 2
        assert "incompatibles" in capsys.readouterr().out

    def test_le_test_n_est_lu_qu_apres_la_garde_de_divergence(self):
        """Dans produce_full, l'evaluation finale (seule lecture du test) suit la comparaison."""
        import inspect
        text = inspect.getsource(pipeline.produce_full)
        assert text.index("compare_stores") < text.index("raise ExportError") < text.index("final_evaluation_2d.main")
