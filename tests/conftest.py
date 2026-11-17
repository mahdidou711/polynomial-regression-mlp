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
        "test_campaign_figures",
        "test_complete_regeneration_identical_to_the_repository",
        "test_isolated_failed_json_makes_verify_store_non_conforming",
        "test_failed_json_without_valid_run_is_reported_and_the_run_stays_missing",
    }
    for item in items:
        if item.path.name in training_modules or item.originalname in training_tests:
            item.add_marker(pytest.mark.skip(reason="MLP training disabled"))
