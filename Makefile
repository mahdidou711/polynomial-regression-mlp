# Targets: all, test, figures, repro, clean.
# The interpreter can be overridden: make PYTHON=/path/to/python
PYTHON ?= python3

# A single BLAS thread (reproducibility) and a local matplotlib cache.
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export MPLCONFIGDIR ?= $(CURDIR)/.mplconfig

RESULTS := results/tables/mlp_2d_final_evaluation.json
SHOWCASE_POLY := nominal_1d_fits nominal_1d_error_profile nominal_1d_campaign_50_seeds exploration_extrapolation
SHOWCASE_MLP := mlp_2d_panel_3d mlp_2d_parity mlp_2d_residual_map

.PHONY: all test figures repro clean

all: figures test repro

# Builds results/ from the cache of trained networks (no training) and refreshes assets/.
# The final evaluation is computed on the first run, after verifying the selection freeze.
figures:
	$(PYTHON) main.py export --with-models
	mkdir -p assets
	for f in $(SHOWCASE_POLY); do cp results/figures/polynomial/$$f.png assets/; done
	for f in $(SHOWCASE_MLP); do cp results/figures/mlp/$$f.png assets/; done

$(RESULTS):
	$(MAKE) figures

# Full test suite, including the tests that train small MLPs.
test: $(RESULTS)
	$(PYTHON) -m pytest -q

# Byte-for-byte check: regenerates everything in a temporary directory and compares with results/.
repro: $(RESULTS)
	$(PYTHON) main.py verify --with-models

clean:
	rm -rf results .mplconfig .pytest_cache data/processed/export_manifest.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
