"""Registry, static checks, comparison and export pipeline of the tables and figures."""
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

import main as main_cli
from src.export import build, checks, pipeline, registry
from src.export.csv_io import read_csv, write_csv
from src.export.layout import REPO, Layout
from tests.helpers_2d import one_thread  # noqa: F401



# ------------------------------------------------------------------ CSV
class TestCsv:
    def test_csv_exact_round_trip(self, tmp_path):
        value = 0.1 + 0.2
        path = write_csv(tmp_path / "t.csv", ["a", "b"], [{"a": value, "b": None}])
        row = read_csv(path)[0]
        assert float(row["a"]) == value and row["b"] == ""
        assert path.read_bytes().endswith(b"\n") and b"\r" not in path.read_bytes()


# ------------------------------------------------------------------ registry
class TestRegistry:
    def test_identifiers_unique_valid_and_figures_without_collision(self):
        stems = [t.stem for t in registry.TABLES]
        assert len(set(stems)) == len(stems) and all(checks.STEM.match(s) for s in stems)
        figs = [f.stem for f in registry.FIGURES]
        assert len(set(figs)) == len(figs) and all(checks.STEM.match(s) for s in figs)
        assert not set(stems) & set(registry.CSV_ONLY)
        for spec in registry.TABLES:  # a derived table never overwrites a source CSV
            if spec.derive is not None:
                assert spec.stem not in registry.CSV_ONLY and spec.stem + ".csv" not in (
                    registry.SOURCES_1D + registry.SOURCES_MLP_CACHE + registry.SOURCES_CARRIED)
                assert spec.inputs
            else:
                assert spec.source == spec.stem

    def test_3d_figures_png_only_and_2d_double_format(self):
        for f in registry.FIGURES:
            three_d = f.stem.startswith("mlp_2d_") and f.stem in {
                "mlp_2d_panel_3d", "mlp_2d_surface_exact", "mlp_2d_surface_predicted",
                "mlp_2d_surface_residuals", "mlp_2d_activation_tanh_relu"}
            assert f.formats == (("png",) if three_d else ("pdf", "png")), f.stem
        assert {f.stem for f in registry.FIGURES if f.needs_models} == {
            "mlp_2d_parity", "mlp_2d_residual_map", "mlp_2d_panel_3d", "mlp_2d_surface_exact",
            "mlp_2d_surface_predicted", "mlp_2d_surface_residuals", "mlp_2d_activation_tanh_relu"}

    def test_figure_file_names_are_stable(self):
        """The file names are referenced from the documentation: they must not change."""
        names = {n for f in registry.FIGURES for n in f.files()}
        assert {"nominal_1d_fits.pdf", "nominal_1d_campaign_50_seeds.png", "exploration_n_samples.pdf",
                "exploration_n_samples_seed42.png", "mlp_2d_panel_3d.png", "mlp_2d_parity.pdf",
                "mlp_2d_activation_tanh_relu.png"} <= names
        assert sum(1 for f in registry.FIGURES for _ in f.files()) == 37  # 16 figures as PDF + PNG and 5 as PNG only

    def test_every_table_of_the_registry_is_non_empty(self):
        src = registry.Sources(REPO.tables)
        for spec in registry.TABLES:
            assert build.table_rows(spec, src), spec.stem


# ------------------------------------------------------------------ derivations (independent oracles)
class TestDerivations:
    src = registry.Sources(REPO.tables)

    def test_n_samples_summary_recomputed_by_hand(self):
        rows = {r["n_samples"]: r for r in registry.derive_n_samples_summary(self.src)}
        assert list(rows) == ["10", "15", "20", "50", "100", "200"]
        stats = self.src.rows("exploration_n_samples_stats")
        block = [r for r in stats if r["n_samples"] == "20"]
        medians = {int(r["degree"]): float(r["test_mse_median"]) for r in block}
        best = min(medians, key=lambda d: (medians[d], d))
        assert rows["20"]["degree_min"] == best and rows["20"]["test_mse_median_min"] == medians[best]
        assert rows["20"]["test_mse_median_d12"] == medians[12] and rows["20"]["n_runs"] == 50
        assert rows["20"]["cstar_mode"] == 6  # nominal distribution (mode of C*, 50 seeds)

    def test_sigma_zero_summary_is_a_single_realisation(self):
        rows = {r["sigma"]: r for r in registry.derive_sigma_summary(self.src)}
        assert rows["0.0"]["n_runs"] == 1 and rows["0.2"]["n_runs"] == 50

    def test_sensitivity_tables_partition_the_plan(self):
        counts = {name: len(fn(self.src)) for name, fn in (
            ("TOPO", registry.derive_topology), ("REG", registry.derive_regularization),
            ("ACT", registry.derive_activation), ("SOLV", registry.derive_solver))}
        assert counts == {"TOPO": 2, "REG": 6, "ACT": 2, "SOLV": 5}
        assert len(registry.derive_configurations(self.src)) == len(self.src.rows("mlp_2d_config_stats"))
        labels = [r["label"] for r in registry.derive_configurations(self.src)]
        assert len(set(labels)) == len(labels)  # adam: lr and batch distinguish the configurations
        solveur = registry.derive_solver(self.src)
        assert solveur[0]["learning_rate_init"] is None and solveur[1]["learning_rate_init"] == 1e-3

    def test_final_summary_equals_the_evaluation_statistics(self):
        stats = self.src.rows("mlp_2d_final_test_stats")
        for row in registry.derive_final_summary(self.src):
            mse = next(r for r in stats if r["label"] == row["label"] and r["metric"] == "mse")
            assert row["mse_mean"] == float(mse["mean"]) and row["mse_max"] == float(mse["max"])
        assert [r["roles"] for r in registry.derive_final_summary(self.src)] == [
            "final", "lbfgs_h_star", "adam_selected"]

    def test_provenance_of_the_2d_data(self):
        rows = registry.derive_provenance(self.src)
        assert [r["n"] for r in rows] == [200, 200, 400, 1600]
        assert rows[0]["sha256"] == hashlib.sha256(self.src.raw_csv.read_bytes()).hexdigest()[:16]
        assert rows[0]["sha256"] == "997facd8edf592eb"  # fingerprint of the input CSV

    def test_no_derivation_reads_the_test_set(self):
        import inspect
        text = inspect.getsource(registry) + inspect.getsource(build)
        assert "generate_test_set" not in text and "prepare_data_2d" not in text


# ------------------------------------------------------------------ figure metadata
class TestFigureMetadata:
    params = build.figure_parameters(registry.Sources(REPO.tables))

    def test_seeds_repetitions_and_single_realisation(self):
        sigma = self.params["exploration_sigma"]
        assert sigma["seeds"] == "1000 to 1049" and sigma["n_seeds"] == 50
        assert sigma["repetitions_per_sigma"]["0.0"] == 1
        assert all(sigma["repetitions_per_sigma"][s] == 50 for s in ("0.05", "0.1", "0.2", "0.5", "1.0"))
        assert sigma["n_samples_fixed"] == 20
        assert self.params["exploration_n_samples"]["sigma_fixed"] == 0.2

    def test_adam_grid_order_and_warnings(self):
        mlp = self.params["mlp_campaigns"]
        assert mlp["adam_grid_order"] == ["lr=0.001, batch=32", "lr=0.001, batch=200",
                                            "lr=0.01, batch=32", "lr=0.01, batch=200"]
        assert mlp["adam_warnings_in_grid_order"] == [5, 10, 6, 10]
        assert "bound 1 excluded" in self.params["mlp_model"]["render_grid"]

    def test_the_figure_manifest_covers_the_registry(self):
        data = json.loads(REPO.figures_manifest.read_text(encoding="utf-8"))
        assert [f["stem"] for f in data["figures"]] == [f.stem for f in registry.FIGURES]
        by = {f["stem"]: f for f in data["figures"]}
        assert "bound 1 is excluded" in by["mlp_2d_panel_3d"]["note"]
        assert by["mlp_2d_parity"]["parameters"]["figure_network_seed"] == 1819


# ------------------------------------------------------------------ real repository: static checks
class TestRepositoryArtifacts:
    @pytest.mark.parametrize("check", [checks.check_files, checks.check_figure_formats,
                                       checks.check_manifest, checks.check_final_evaluation],
                             ids=lambda c: c.__name__)
    def test_check_passes_on_the_repository(self, check):
        result = check(REPO)
        assert result.status == checks.PASSED, result.details

    def test_selection_freeze_hash_matches_the_manifest(self):
        digest = hashlib.sha256((REPO.tables / "mlp_2d_selection.json").read_bytes()).hexdigest()
        manifest = json.loads(REPO.manifest.read_text(encoding="utf-8"))
        assert manifest["protocol"]["freeze_selection_sha256"] == digest
    def test_summary_csv_is_the_flat_list_of_the_manifest(self):
        manifest = json.loads(REPO.manifest.read_text(encoding="utf-8"))
        rows = read_csv(REPO.tables / registry.SUMMARY_FILE)
        assert [r["path"] for r in rows] == [a["path"] for a in manifest["artifacts"]]
        assert [r["sha256"] for r in rows] == [a["sha256"] for a in manifest["artifacts"]]

    def test_manifest_without_timestamp_or_git_sha(self):
        text = REPO.manifest.read_text(encoding="utf-8")
        assert "git" not in text.lower() and "202" not in text.split('"artifacts"')[0].replace("2026", "")


# ------------------------------------------------------------------ copied repository: injected defects
@pytest.fixture
def layout_copy(tmp_path):
    out = Layout(tmp_path / "copy")
    for rel, path in pipeline.tracked_files(REPO).items():
        target = out.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return out


class TestInjectedDefects:
    def test_conforming_copy(self, layout_copy):
        assert all(c.status in (checks.PASSED, checks.WARNING) for c in checks.run_static_checks(layout_copy))
        assert pipeline.is_clean(pipeline.compare(layout_copy, REPO))

    def test_manifest_detects_a_modified_byte(self, layout_copy):
        target = layout_copy.tables / "nominal_1d_seed42.csv"
        target.write_bytes(target.read_bytes().replace(b"0.47234770197568643", b"0.47234770197568644"))
        result = checks.check_manifest(layout_copy)
        assert result.status == checks.FAILED and any("nominal_1d_seed42.csv" in d for d in result.details)

    def test_undeclared_file_is_reported(self, layout_copy):
        (layout_copy.tables / "table_orphan.csv").write_text("x", encoding="utf-8")
        (layout_copy.fig_mlp / "mlp_2d_orphan.png").write_bytes(b"x")
        details = checks.check_files(layout_copy).details
        assert any("table_orphan.csv" in d for d in details)
        assert any("mlp_2d_orphan.png" in d for d in details)

    def test_missing_figure_and_pdf_of_a_3d_figure(self, layout_copy):
        (layout_copy.fig_poly / "nominal_1d_fits.pdf").unlink()
        assert any("nominal_1d_fits.pdf" in d for d in checks.check_files(layout_copy).details)
        (layout_copy.fig_mlp / "mlp_2d_panel_3d.pdf").write_bytes(b"%PDF-1.4")
        assert any("3D figure" in d for d in checks.check_figure_formats(layout_copy).details)

    def test_png_with_wrong_resolution_or_header(self, layout_copy):
        path = layout_copy.fig_poly / "nominal_1d_fits.png"
        data = path.read_bytes()
        i = data.index(b"pHYs")
        path.write_bytes(data[:i + 4] + (2000).to_bytes(4, "big") + data[i + 8:])
        assert any("DPI" in d for d in checks.check_figure_formats(layout_copy).details)
        path.write_bytes(b"GIF89a" + b"\0" * 10)
        assert any("PNG header" in d for d in checks.check_figure_formats(layout_copy).details)

    def test_inconsistent_final_evaluation(self, layout_copy):
        path = layout_copy.tables / "mlp_2d_final_test_stats.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        parts = lines[1].split(",")
        parts[4] = repr(float(parts[4]) * 1.01)  # tampered mean test MSE
        lines[1] = ",".join(parts)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = checks.check_final_evaluation(layout_copy)
        assert result.status == checks.FAILED and any("recomputed" in d for d in result.details)

    def test_evaluation_associated_with_another_freeze(self, layout_copy):
        path = layout_copy.tables / "mlp_2d_selection.json"
        path.write_text(path.read_text(encoding="utf-8").replace('"schema"', '"schema" ', 1), encoding="utf-8")
        details = checks.check_final_evaluation(layout_copy).details
        assert any("another freeze" in d for d in details)

    def test_comparison_and_synchronisation(self, layout_copy, tmp_path):
        other = Layout(tmp_path / "other")
        shutil.copytree(layout_copy.root, other.root)
        (other.tables / "nominal_1d_seed42.csv").write_text("x", encoding="utf-8")
        (other.tables / "new.csv").write_text("a\n", encoding="utf-8")
        (other.fig_poly / "nominal_1d_fits.png").unlink()
        diff = pipeline.compare(other, layout_copy)
        assert diff["different"] == ["results/tables/nominal_1d_seed42.csv"]
        assert diff["missing_in_second"] == ["results/tables/new.csv"]
        assert diff["missing_in_first"] == ["results/figures/polynomial/nominal_1d_fits.png"]
        assert not pipeline.is_clean(diff)
        changed = pipeline.sync(other, layout_copy, exclude={"results/tables/nominal_1d_seed42.csv"})
        assert changed == ["results/tables/new.csv"]  # the excluded file is never copied
        only = pipeline.compare(other, layout_copy, only={"results/tables/nominal_1d_seed42.csv"})
        assert only["different"] == ["results/tables/nominal_1d_seed42.csv"]


# ------------------------------------------------------------------ main.py safeguards
class TestCommandLine:
    def test_reproduce_refuses_an_output_inside_the_repository(self, capsys):
        assert main_cli.main(["reproduce", "--out", str(REPO.root / "results" / "x")]) == 2
        assert "outside the repository" in capsys.readouterr().out

    def test_reproduce_refuses_a_non_empty_directory(self, tmp_path, capsys):
        (tmp_path / "x").write_text("1")
        assert main_cli.main(["reproduce", "--out", str(tmp_path)]) == 2
        assert "is not empty" in capsys.readouterr().out

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            main_cli.main([])

    def test_runners_are_never_called_without_an_isolated_directory(self):
        """No call to a runner's ``main`` omits ``--tables-dir``: the destructive rewrite
        of the tables is impossible from the pipeline."""
        import re
        text = Path(pipeline.__file__).read_text(encoding="utf-8")
        calls = []
        for match in re.finditer(r"\w+\.main\(", text):
            depth, i = 1, match.end()
            while depth:  # balanced parentheses: the argument can contain calls
                depth += (text[i] == "(") - (text[i] == ")")
                i += 1
            calls.append(text[match.end():i - 1])
        assert len(calls) >= 5  # 2 1D runners, stages B and C, final evaluation
        for args in calls:
            assert "--tables-dir" in args or args == "argv", args
        assert text.count('"--tables-dir"') >= 4  # 2 1D runners, common list of stages B and C, final evaluation

    def test_main_sets_a_single_thread_before_numpy(self):
        text = Path(main_cli.__file__).read_text(encoding="utf-8")
        assert text.index("OMP_NUM_THREADS") < text.index("import argparse") < text.index("from src.export")


# ------------------------------------------------------------------ complete pipeline
@pytest.mark.usefixtures("one_thread")
class TestFullPipeline:
    def test_complete_regeneration_identical_to_the_repository(self, tmp_path):
        """No drift: 1D tables recomputed, MLP tables from the cache, figures (networks included),
        tables, manifests: everything is byte-identical to the existing files."""
        before = {p: p.read_bytes() for p in REPO.tables.iterdir() if p.is_file()}
        out = Layout(tmp_path / "out")
        info = pipeline.produce(out, REPO, with_models=True)
        # produce never writes into the repository: all the tables there are unchanged
        assert {p: p.read_bytes() for p in REPO.tables.iterdir() if p.is_file()} == before
        assert info["h_star"] == 50 and "regenerated" in info["model_figures"]
        assert pipeline.compare(out, REPO) == {"missing_in_first": [], "missing_in_second": [], "different": []}
        assert (out.tables / "mlp_2d_selection.json").read_bytes() == (REPO.tables / "mlp_2d_selection.json").read_bytes()

    def test_without_models_the_dependent_figures_are_carried_over(self, tmp_path):
        out = Layout(tmp_path / "out")
        info = pipeline.produce(out, REPO, with_models=False)
        assert "carried over" in info["model_figures"]
        for folder, name in pipeline.model_figure_names():
            assert (out.figure_dir(folder) / name).read_bytes() == (REPO.figure_dir(folder) / name).read_bytes()

    def test_an_incomplete_cache_stops_before_any_training(self, tmp_path, monkeypatch):
        from src.experiments import runner_2d_nominal as nom
        trained = []
        monkeypatch.setattr(nom, "run_strict", lambda *a, **k: trained.append(1))
        empty = tmp_path / "store"
        empty.mkdir()
        with pytest.raises(pipeline.ExportError, match="no training"):
            pipeline.regenerate_mlp_tables(Layout(tmp_path / "o"), store=empty)
        assert trained == [] and not list(empty.iterdir())


# ------------------------------------------------------------------ determinism of two successive exports
class TestSuccessiveExports:
    def test_derived_builds_are_identical_at_each_call(self, layout_copy):
        """Derived CSV, metadata, manifest and summary: same bytes
        at each construction, and same bytes as the existing files."""
        def build_all():
            src = registry.Sources(layout_copy.tables)
            build.build_tables(layout_copy, src)
            build.build_figures_manifest(layout_copy, src)
            build.build_manifest(layout_copy, src)
            return {rel: p.read_bytes() for rel, p in pipeline.tracked_files(layout_copy).items()}

        first, second = build_all(), build_all()
        assert first == second
        assert first == {rel: p.read_bytes() for rel, p in pipeline.tracked_files(REPO).items()}

    def test_manifest_and_summary_do_not_depend_on_file_system_order(self, layout_copy):
        manifest = json.loads(layout_copy.manifest.read_text(encoding="utf-8"))
        paths = [a["path"] for a in manifest["artifacts"]]
        assert paths == sorted(paths) and len(set(paths)) == len(paths)

    def test_sync_writes_nothing_when_nothing_changes(self, layout_copy, tmp_path):
        other = Layout(tmp_path / "other")
        shutil.copytree(layout_copy.root, other.root)
        assert pipeline.sync(other, layout_copy) == []


# ------------------------------------------------------------------ full reproduction (comparisons)
class TestFullReproductionComparisons:
    def _record(self, **diag):
        base = {"n_iter": 10, "loss": 0.1, "elapsed_seconds": 1.0, "convergence_warnings": []}
        return {"run_key": "k", "weights_sha256": "w", "training_metrics": {"mse": 1.0},
                "validation_metrics": {"mse": 2.0}, "stop_cause": "none", "status": "completed",
                "diagnostics": {**base, **diag}}

    def test_the_digest_of_a_record_ignores_only_the_duration(self):
        a = pipeline.record_digest(self._record())
        assert a == pipeline.record_digest(self._record(elapsed_seconds=99.0))
        assert a != pipeline.record_digest(self._record(n_iter=11))
        assert a != pipeline.record_digest(self._record(loss=0.2))
        changed = self._record()
        changed["weights_sha256"] = "x"
        assert a != pipeline.record_digest(changed)

    def test_comparison_of_run_stores(self, tmp_path):
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
        assert pipeline.compare_stores(a, b) == [f"missing from the second store: {source[1].name}"]

    def test_stores_with_different_scientific_content_are_reported(self, tmp_path, monkeypatch):
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
            if Path(path).parent == b:  # same duration, different weights: scientific divergence
                record["weights_sha256"] = "0" * 64
            return record

        monkeypatch.setattr(nom, "load_record", load)
        assert pipeline.compare_stores(a, b) == [f"different scientific content: {first.name}"]

    def test_freeze_compared_without_record_fingerprints(self):
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

    def test_evaluation_compared_without_freeze_fingerprints(self):
        ev = json.loads((REPO.tables / "mlp_2d_final_evaluation.json").read_text(encoding="utf-8"))
        sci = pipeline.scientific_evaluation(ev)
        other = json.loads(json.dumps(ev))
        other["selection_sha256"] = "0" * 64
        other["environment"] = {"platform": "other"}
        assert pipeline.scientific_evaluation(other) == sci
        label = next(iter(other["configs"]))
        other["configs"][label]["test"]["mse"]["mean"] *= 1.0000001
        assert pipeline.scientific_evaluation(other) != sci

    def test_scientific_comparison_tolerates_durations_and_refuses_the_rest(self, layout_copy, tmp_path):
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
        # a different file remains a difference (no tolerance outside volatile files)
        shutil.copyfile(layout_copy.tables / "mlp_2d_runs.csv", path)
        (other.tables / "nominal_1d_seed42.csv").write_text("x", encoding="utf-8")
        assert pipeline.compare_scientific(other, layout_copy)["different"] == ["results/tables/nominal_1d_seed42.csv"]

    def test_full_reproduction_refused_in_the_nominal_store(self, tmp_path):
        from src.experiments import runner_2d_nominal as nom
        with pytest.raises(pipeline.ExportError, match="nominal versioned store"):
            pipeline.regenerate_mlp_tables(Layout(tmp_path), nom.NOMINAL_STORE, train=True)

    def test_full_is_incompatible_with_no_models(self, tmp_path, capsys):
        assert main_cli.main(["reproduce", "--full", "--no-models", "--out", str(tmp_path / "o")]) == 2
        assert "incompatible" in capsys.readouterr().out

    def test_the_test_set_is_read_only_after_the_divergence_guard(self):
        """In produce_full, the final evaluation (the only read of the test set) follows the comparison."""
        import inspect
        text = inspect.getsource(pipeline.produce_full)
        assert text.index("compare_stores") < text.index("raise ExportError") < text.index("final_evaluation_2d.main")
