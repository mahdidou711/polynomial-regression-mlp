# Methods

This document describes the mathematical and numerical methods implemented in `src/`. Experimental settings are in [experiments](experiments.md) and measured values in [results](results.md).

## 1. Polynomial least squares

For $N$ observations $(x_i, y_i)$ and degree $d$, the model is $y \approx H\beta$ with the uncentred Vandermonde matrix

$$H_{ij} = x_i^{\,j}, \qquad j = 0, \dots, d, \qquad p = d + 1.$$

Column 0 (all ones) carries the constant term, so no additional intercept is fitted. The scikit-learn reference is `LinearRegression(fit_intercept=False)` for the same reason (`src/models/polynomial.py`).

Three solvers are implemented:

| Solver | Computation | Requirement |
| --- | --- | --- |
| Normal equations | solve $(H^\top H)\beta = H^\top y$ with `numpy.linalg.solve` | full column rank, $N \ge p$ |
| QR | factor $H = QR$, then solve $R\beta = Q^\top y$ with `scipy.linalg.solve_triangular` | full column rank, $N \ge p$ |
| SVD | `scipy.linalg.lstsq` with the LAPACK driver `gelsd`; minimum-norm solution when $H$ is rank-deficient or $N < p$ | none |

The SVD solver is also the one used by scikit-learn's `LinearRegression`, and it is the reference against which the other two are compared. Normal equations and QR raise `RankDeficientError` (a subclass of `numpy.linalg.LinAlgError`) when $N < p$ or when the rank of $H$ is below $p$. `compare_solvers` never raises for that reason: it labels each method as computed or not applicable and measures its gap to the SVD solution.

## 2. Conditioning

$\kappa_2(H)$ governs the QR and SVD solvers. Forming $H^\top H$ squares the condition number: for a full-column-rank matrix, $\kappa(H^\top H) = \kappa_2(H)^2$, and this quantity governs the normal equations. Both numbers are obtained from the singular values of $H$, without forming $H^\top H$.

The implementation distinguishes two regimes using a threshold of $10^7$ on $\kappa(H^\top H)$. Below it (strict regime) the three solvers are expected to agree within a relative tolerance of $10^{-7}$. Above it (diagnostic regime) the gap between methods is reported as a measured quantity. In the nominal 1D experiment with seed 42 the strict regime covers degrees 1 to 5 and the diagnostic regime degrees 6 to 12 (`results/tables/nominal_1d_seed42_diagnostics.csv`).

## 3. Degree selection in one dimension

For each degree $d \in \{1, \dots, 12\}$ the training error EQM($d$) and the test error EQMT($d$) are mean squared errors on the training set and on 200 noise-free test points. The optimal degree is

$$C^\* = \arg\min_d \; \mathrm{EQMT}(d),$$

with the smallest degree chosen on ties. Because $C^\*$ is selected on the test set, EQMT($C^\*$) is an optimistically biased estimate of generalisation error. A single noisy sample is only one realisation, so the project also reports the distribution of $C^\*$ over 50 noise seeds.

## 4. Multilayer perceptron regression

Models are `sklearn.neural_network.MLPRegressor` instances, fully connected, trained to minimise the squared error plus an L2 penalty of strength $\alpha$. Parameters explored (`src/models/mlp.py`, `results/tables/mlp_2d_metadata.json`):

| Parameter | Values |
| --- | --- |
| Hidden layers | single layers of 5, 10, 20, 50, 100, 200 units; two layers (20, 10) and (50, 25) |
| Activation | tanh (default), ReLU |
| L2 penalty $\alpha$ | $10^{-4}$ (default); $10^{-5}$, $10^{-3}$, $10^{-1}$ in the regularisation study |
| L-BFGS budget | `max_iter` 10000, `max_fun` 100000, `tol` $10^{-6}$ |
| Adam budget | `max_iter` 5000, `tol` $10^{-8}$, `n_iter_no_change` 200, $\beta_1 = 0.9$, $\beta_2 = 0.999$, constant learning rate, shuffling, no early stopping |
| Adam grid | learning rate in {0.001, 0.01}, batch size in {32, 200} |
| Seeds | 10 Monte Carlo seeds: 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021 |

Metrics are mean squared error, root mean squared error, the coefficient of determination $R^2$ and the maximum absolute error (`src/utils/metrics.py`). Random numbers come from explicit `numpy.random.Generator` objects or integer `random_state` values, never from global state (`src/utils/seeding.py`).

## 5. Selection protocol

1. Networks are trained on the 200 official observations.
2. All configurations are compared by mean validation MSE over 10 seeds on a separate validation grid. Ties are resolved by a fixed rule (smallest width, or grid order).
3. The selected configurations are written to `results/tables/mlp_2d_selection.json` and `results/tables/mlp_2d_hstar.json`, and their SHA-256 hashes are recorded.
4. Only then does the final evaluation module recompute the selection from validation records, check it against the frozen files, retrain the selected networks, verify their weight fingerprints and read the test set.

The test data are therefore not used to choose any configuration.
