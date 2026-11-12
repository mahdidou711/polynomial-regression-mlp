"""Optional test mode that prevents actual MLP optimization."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--no-mlp-training", action="store_true",
                     help="Skip tests when they reach scikit-learn MLP optimization")


@pytest.fixture(autouse=True)
def prevent_mlp_training(request, monkeypatch):
    if request.config.getoption("--no-mlp-training"):
        from sklearn.neural_network import MLPRegressor

        def skip_optimization(*args, **kwargs):
            pytest.skip("MLP optimization disabled by --no-mlp-training")

        monkeypatch.setattr(MLPRegressor, "_fit", skip_optimization)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--no-mlp-training"):
        return
    training_modules = {"test_mlp_pipeline.py", "test_experiment_2d.py", "test_selection_2d.py"}
    training_tests = {
        "test_figures_de_campagne",
        "test_regeneration_complete_identique_au_depot",
        "test_failed_json_isole_rend_verify_store_non_conforme",
        "test_failed_json_sans_run_valide_est_signale_et_le_run_reste_manquant",
    }
    for item in items:
        if item.path.name in training_modules or item.originalname in training_tests:
            item.add_marker(pytest.mark.skip(reason="MLP training disabled"))
