# Experimental protocols

Measured values for every experiment below are in [results](results.md). Methods are in [methods](methods.md).

## 1. One-dimensional polynomial regression

**Target.** $f(x) = \sin(-\pi + 2\pi x) = -\sin(2\pi x)$.

**Sampling.** $N$ equidistant points on the half-open interval $[0, 1.2)$, so the upper bound is excluded. Training targets are $f(x) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ drawn from a local generator. Test targets are noise-free.

| Setting | Nominal value |
| --- | --- |
| Training points $N$ | 20 |
| Noise $\sigma$ | 0.2 |
| Test points | 200, no noise |
| Degrees | 1 to 12 |
| Illustration seed | 42 |
| Statistical seeds | 50 seeds, 1000 to 1049 |

### Nominal experiment

For each degree the polynomial is fitted on the training set and the training error (EQM) and test error (EQMT) are recorded. $C^\*$ is the degree minimising EQMT. The seed-42 realisation provides the figures and the solver diagnostics. The same experiment is repeated over the 50 statistical seeds, which gives the distribution of $C^\*$ and the median and mean errors per degree.

### Explorations

All explorations use the 50 statistical seeds and vary one factor at a time.

| Exploration | Factor | Values |
| --- | --- | --- |
| Sample size | $N$ (with $\sigma = 0.2$) | 10, 15, 20, 50, 100, 200 |
| Noise | $\sigma$ (with $N = 20$) | 0, 0.05, 0.1, 0.2, 0.5, 1.0 |
| Extrapolation | extended domain $[-0.3, 1.5]$ rendered on 301 points | degrees 1, 3, $C^\*$ and 10 |

For $\sigma = 0$ the samples would be identical across seeds, so a single realisation is used. In the extrapolation study the error is split into points inside the training domain $[0, 1.2]$ (both bounds included) and points outside it.

## 2. Two-dimensional MLP regression

**Target.** A cardinal-sine surface on $[0, 1]^2$. Write $u = -\pi + 2\pi x + b$ with $b = (0.2, -0.3)$, treated as a row vector, and $z = uM$ with $M = \begin{pmatrix} 1 & -2 \\ 1 & 1 \end{pmatrix}$, as in `src/data/generator_2d.py`. The value is $y = \sin(h)/h$ with $h = \lVert z \rVert$, and $y = 1$ when $h < 10^{-3}$. All 2D data are noise-free.

| Dataset | Content | Use |
| --- | --- | --- |
| Training | 200 observations read from `data/raw/sinc_dim2_input.csv` (semicolon-separated, columns `x1;x2`) | fitting the networks |
| Validation | 400 cell centres of a 20 x 20 grid | choosing configurations |
| Test | 1600 points of a 40 x 40 grid, bounds included | final evaluation only |

### Configuration plan

The plan has 21 entries covering 19 distinct configurations. With 10 seeds each, there are 210 planned realisations and 190 distinct trained runs, because some entries share a run.

| Study | Configurations |
| --- | --- |
| Width | tanh, L-BFGS, single layer of 5, 10, 20, 50, 100, 200 units |
| Depth | two layers (20, 10) and (50, 25) |
| Activation | ReLU against tanh at width 50 |
| Regularisation | $\alpha \in \{10^{-5}, 10^{-3}, 10^{-1}\}$ at widths 50 and 100 |
| Optimiser | Adam at width 50 with learning rate in {0.001, 0.01} and batch size in {32, 200}, against L-BFGS |

### Selection and final evaluation

1. $H^\*$ is the width with the lowest mean validation MSE among single-layer L-BFGS networks (tie: smallest width).
2. The final configuration is the one with the lowest mean validation MSE among all 19 configurations.
3. The Adam setting is the one with the lowest mean validation MSE on the Adam grid.
4. The selection is stored in `results/tables/mlp_2d_selection.json` and `results/tables/mlp_2d_hstar.json` with hashes.
5. The final evaluation trains the three selected configurations (final, L-BFGS at $H^\*$, selected Adam) with 10 seeds each, checks the weight fingerprints against the cache, and reads the 1600 test points once.

Each trained run is stored as one JSON file in `data/processed/mlp_2d_runs/`, named by the SHA-256 identity of the run. The identity includes the configuration, the seed, the data fingerprints, the software versions, the model source and the number of BLAS threads.
