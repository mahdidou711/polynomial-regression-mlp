"""1D explorations: size, noise and extrapolation.

C* minimises the nominal test MSE: its selection on the test set is optimistic and
does not provide an independent estimate of generalisation. Repetitions are descriptive.
Command: python -m src.experiments.runner_1d_explorations
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.generator_1d import generate_training_set, generate_test_set
from src.experiments.runner_1d_nominal import (CampaignResult, DEGREES, FIGURES_DIR,
                                                TABLES_DIR, run_degree_sweep)
from src.models import predict
from src.utils.seeding import SEED_POLYNOMIAL, SEEDS_STAT_1D

SIZES = (10, 15, 20, 50, 100, 200)
SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
EXTENDED_MIN, EXTENDED_MAX, EXTENDED_POINTS = -0.3, 1.5, 301


@dataclass(frozen=True)
class Configuration:
    n_samples: int = 20
    sigma: float = 0.2


def run_campaign(config, seeds=SEEDS_STAT_1D):
    """A single noise-free realisation; otherwise each seed is distinct."""
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('non-empty and distinct seeds required')
    if config.sigma == 0:
        seeds = (SEED_POLYNOMIAL,)
    xt, yt = generate_test_set()
    runs = []
    for seed in seeds:
        x, y = generate_training_set(n_samples=config.n_samples, sigma=config.sigma, seed=seed)
        runs.append(run_degree_sweep(x, y, xt, yt, diagnose=False, seed=seed))
    return CampaignResult(seeds, np.asarray(DEGREES), np.array([r.mse_train for r in runs]),
                          np.array([r.mse_test for r in runs]),
                          np.array([r.optimal_degree for r in runs]))


def run_extrapolation(seeds=SEEDS_STAT_1D):
    """C* is chosen on the nominal test set; errors on an inclusive extended grid.

    Returns one row per seed and role (1, 3, C*, 10), even if two roles
    correspond to the same degree. Both boundaries belong to the interpolation.
    """
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('non-empty and distinct seeds required')
    xt, yt = generate_test_set()
    # The generator is half-open: 300 points, then the bound 1.5 is added.
    xe, ye = generate_test_set(n_test=EXTENDED_POINTS - 1, xmin=EXTENDED_MIN,
                              xmax=EXTENDED_MAX)
    # inclusive grid, step .006: the right bound is explicitly added
    xe = np.vstack((xe, [[EXTENDED_MAX]]))
    from src.data.generator_1d import target_function
    ye = np.append(ye, target_function(EXTENDED_MAX))
    inside = (xe[:, 0] >= -1e-12) & (xe[:, 0] <= 1.2 + 1e-12)
    rows = []
    for seed in seeds:
        x, y = generate_training_set(seed=seed)
        result = run_degree_sweep(x, y, xt, yt, diagnose=False, seed=seed)
        for role, degree in [('1', 1), ('3', 3), ('C*', result.optimal_degree), ('10', 10)]:
            pred = predict(result.coefs[result.index_of(degree)], xe)
            err = (pred - ye)**2
            rows.append(dict(seed=seed, role=role, degree=degree, cstar=result.optimal_degree,
                             mse_interp=float(np.mean(err[inside])),
                             mse_extrap=float(np.mean(err[~inside]))))
    return rows


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_campaigns(campaigns, axis, directory):
    raw, stats, counts = [], [], []
    for value, campaign in campaigns.items():
        tr, te = campaign.summary('train'), campaign.summary('test')
        for i, degree in enumerate(campaign.degrees):
            for j, seed in enumerate(campaign.seeds):
                raw.append({axis: value, 'seed': seed, 'degree': int(degree),
                            'mse_train': campaign.mse_train[j, i],
                            'mse_test': campaign.mse_test[j, i],
                            'cstar': campaign.optimal_degrees[j]})
            row = {axis: value, 'n_runs': len(campaign.seeds), 'degree': int(degree)}
            for label, summary in [('mse', tr), ('test_mse', te)]:
                row.update({f'{label}_{k}': summary[k][i] for k in ('mean', 'median', 'q1', 'q3', 'iqr')})
            stats.append(row)
            counts.append({axis: value, 'degree': int(degree),
                           'count': campaign.cstar_counts()[int(degree)],
                           'mode': campaign.cstar_mode, 'median': campaign.cstar_median})
    return [_write(Path(directory) / f'exploration_{axis}_{suffix}.csv', data)
            for suffix, data in [('runs', raw), ('stats', stats), ('cstar', counts)]]


def export_extrapolation(rows, directory):
    summary = []
    for role in ('1', '3', 'C*', '10'):
        selected = [r for r in rows if r['role'] == role]
        row = {'role': role, 'n_runs': len(selected)}
        for metric in ('mse_interp', 'mse_extrap'):
            values = [r[metric] for r in selected]
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            row.update({metric + '_mean': np.mean(values), metric + '_median': median,
                        metric + '_q1': q1, metric + '_q3': q3, metric + '_iqr': q3-q1})
        summary.append(row)
    return [_write(Path(directory) / 'exploration_extrapolation_runs.csv', rows),
            _write(Path(directory) / 'exploration_extrapolation_stats.csv', summary)]


def size_diagnostics(sizes=SIZES, seeds=SEEDS_STAT_1D):
    """Variance of the predictions at fixed x and diagnostics of the geometry, seed 42."""
    xt, yt = generate_test_set()
    variance_rows, diagnostic_rows = [], []
    for n in sizes:
        predictions = []
        for seed in seeds:
            x, y = generate_training_set(n_samples=n, seed=seed)
            predictions.append(run_degree_sweep(x, y, xt, yt, diagnose=False).y_pred_test)
        variances = np.var(predictions, axis=0, ddof=1).mean(axis=1)
        x, y = generate_training_set(n_samples=n, seed=SEED_POLYNOMIAL)
        diagnostic = run_degree_sweep(x, y, xt, yt, diagnose=True)
        for i, degree in enumerate(DEGREES):
            variance_rows.append({'n_samples': n, 'degree': degree, 'n_seeds': len(seeds),
                                  'mean_prediction_variance': variances[i],
                                  'n_times_variance': n * variances[i]})
            diagnostic_rows.append({'n_samples': n, 'degree': degree, 'rank': diagnostic.rank[i],
                                    'n_parameters': degree + 1, 'kappa_h': diagnostic.kappa_h[i],
                                    'kappa_hth': diagnostic.kappa_hth[i],
                                    'regime': diagnostic.regime[i],
                                    'normal_rel_error': diagnostic.rel_error_normal[i],
                                    'qr_rel_error': diagnostic.rel_error_qr[i]})
    return variance_rows, diagnostic_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tables-dir', type=Path, default=TABLES_DIR)
    parser.add_argument('--figures-dir', type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    sizes = {n: run_campaign(Configuration(n, .2)) for n in SIZES}
    sigmas = {s: run_campaign(Configuration(20, s)) for s in SIGMAS}
    extra = run_extrapolation()
    export_campaigns(sizes, 'n_samples', args.tables_dir)
    export_campaigns(sigmas, 'sigma', args.tables_dir)
    export_extrapolation(extra, args.tables_dir)
    variances, diagnostics = size_diagnostics()
    _write(args.tables_dir / 'exploration_n_samples_variance.csv', variances)
    _write(args.tables_dir / 'exploration_n_samples_diagnostics.csv', diagnostics)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'sizes': SIZES, 'sigmas': SIGMAS, 'degrees': DEGREES,
                'statistical_seeds': SEEDS_STAT_1D, 'illustration_seed': SEED_POLYNOMIAL,
                'train_domain': [0, 1.2], 'train_endpoint': 'excluded', 'test_points': 200,
                'extended_domain': [EXTENDED_MIN, EXTENDED_MAX], 'extended_points': EXTENDED_POINTS,
                'extended_endpoints': 'included', 'method': 'sklearn LinearRegression(fit_intercept=False), SVD',
                'selection': 'minimum nominal test MSE, smallest degree on ties; optimistically biased'}
    (args.tables_dir / 'exploration_1d_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    from src.visualization.plots_1d_explorations import save_exploration_figures
    save_exploration_figures(sizes, sigmas, extra, args.figures_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
