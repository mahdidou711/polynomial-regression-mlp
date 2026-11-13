# Development history

This repository was developed between 2026-11-10 and 2026-11-12. `main` contains that development history, merged with the public release commit.

## Milestones

Dates are author dates (UTC+01:00). The commit subjects are in French, the language in which the work was done; the Milestone column is a short English description.

| Date | Commit | Milestone | Subject |
| --- | --- | --- | --- |
| 2026-11-10 15:06 | [`27baba1`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/27baba1e5ea2f73164ff72c2496520b8e584bb2d) | Project skeleton, pinned dependencies and the raw 2D observations | chore: initialisation de la structure du projet |
| 2026-11-10 15:15 | [`e91a9e6`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/e91a9e6354d918dc055ffe3a00a68ebe47d0e624) | Error metrics, deterministic seeding and structured logging | feat(utils): socle metriques, graines et journalisation |
| 2026-11-10 16:00 | [`3490de8`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/3490de840a94e1e15427377662901f329d566ac1) | 1D data generation and sampling | feat(data): generation et echantillonnage des donnees 1D |
| 2026-11-10 16:27 | [`7455862`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/745586236a2b41992fb33d339ddf5ff5acb5e915) | Polynomial regression with normal-equation, QR and SVD solvers | feat(polynomial): regression polynomiale 1D et solveurs OLS / QR / SVD |
| 2026-11-10 18:00 | [`92e2c6d`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/92e2c6da473b9fdae47e4d5408015e29c24e3a82) | Nominal 1D experiment (degrees 1 to 12) and base figures | feat(experiments): experience 1D nominale et visualisations de base |
| 2026-11-10 18:42 | [`1659d5f`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/1659d5f08d4197be84aad4b831e96388a2e998ce) | 1D explorations: sample size, noise level and extrapolation | feat(experiments): explorations polynomiales 1D |
| 2026-11-10 19:12 | [`1107aed`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/1107aed9be4c7a74bf8fe1ced196d1856522602c) | Tests locking the selection rule and the extrapolation cases | test(selection): verrouiller la selection et les roles extrapoles |
| 2026-11-10 19:27 | [`c58580b`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/c58580bc630c426e67c380fcc91b5e70f2313b5b) | 2D data preparation: training, validation and test grids | feat(data): preparer les donnees sinc 2D |
| 2026-11-10 20:00 | [`42e9b24`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/42e9b246485b5e4b7c3d4d3bd01bd470b91314f7) | MLP engine and multi-seed repetitions | feat(mlp): moteur MLP 2D et repetitions multi-graines |
| 2026-11-10 20:24 | [`e216758`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/e216758eaa3265fafffcb092655170b0e60551d6) | Tests protecting the separation of training and validation data | test(mlp): proteger la separation apprentissage validation |
| 2026-11-10 21:49 | [`7fa1473`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/7fa1473ec1c4a159a6192cf69bda456527407f3b) | MLP campaigns, 190-run cache, validation tables and the selection frozen before any test evaluation | feat(selection): gel de la selection avant evaluation sur le test |
| 2026-11-10 21:52 | [`810de52`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/810de5299a9762816467fb71eb668ae3060e3b6c) | Final evaluation on the test grid and 2D visualisations | feat(evaluation): evaluation finale sur le test et visualisations MLP 2D |
| 2026-11-11 15:33 | [`f50d0ee`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/f50d0ee359657022eea39918d450a0f731e12a21) | Independent-oracle tests for the final evaluation | test(evaluation): oracles independants pour l'evaluation finale |
| 2026-11-11 17:32 | [`0660338`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/0660338666366db0a0429b83eddec6c02d216bf9) | Deterministic figure rendering and readability fixes | feat(figures): rendu deterministe des figures et corrections de lisibilite |
| 2026-11-11 17:32 | [`2aa5c2f`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/2aa5c2fbc202acfe4181442d4b1aeb3a4e79b4ba) | Export pipeline, command-line entry point and isolated full reproduction | feat(export): pipeline d'export LaTeX, main.py et reproduction integrale isolee |
| 2026-11-11 17:32 | [`f0bf71b`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/f0bf71ba5a62a9a54b736b4bfd3119215b7ab777) | Regenerated figures, LaTeX tables and manifest | feat(export): figures regenerees et tables LaTeX exportees |
| 2026-11-11 18:48 | [`a41b9c7`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/a41b9c73e88690ec6a8105e3658957af4b636a44) | Fix of the preview presentation | fix(export): corriger la presentation de l'apercu |
| 2026-11-12 14:44 | [`b4c714f`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/b4c714fc43888a873d0829b42a9aad1506db18fa) | Docstring clarification on the static LaTeX checks | docs(export): preciser le statut des controles LaTeX statiques |
| 2026-11-12 17:46 | [`317b61e`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/317b61eedf5ce2f74471caa2852ca578e1748815) | Public release: licence, README, ignore rules, test configuration and the published version of the code and results | feat: initial public release of polynomial regression and MLP project |
| 2026-11-13 14:55 | [`a1b9248`](https://github.com/mahdidou711/polynomial-regression-mlp/commit/a1b924852bfd33cc09f09400fb5092866de21ce9) | Merge of the development history into `main` | Merge development history (2026-11-10 to 2026-11-12) |

## How the history is organised

- The first parent of the merge commit `a1b9248` is the public release commit `317b61e`. The second parent is the last commit of the development line, `b4c714f`.
- `git log --first-parent --oneline` shows the public line only. `git log --graph --oneline` shows the full development.
- The tree of the merge commit is identical to the tree of the public release commit. The merge records the development history without changing the published files.

## What was kept and what was changed

- **Kept:** the commits that changed code, tests, data, results and the entry point, with their original author and commit dates and with the file contents they had at the time. For the kept paths, the listing of files and contents at each commit was compared with the original development repository.
- **Not published:** course documents, review documents and other working notes, together with the commits that only touched them. Their absence is why this history has 18 commits.
- **Edited:** commit messages, to remove references to the unpublished documents; and the author identity, which uses the GitHub no-reply address. Because messages and metadata changed, the hashes above differ from those of the original repository.
- **Not verified:** the intermediate commits were not individually re-tested. The licence, the README, the ignore rules and the test configuration file are part of the public release commit, not of the earlier commits.
