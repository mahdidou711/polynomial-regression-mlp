"""Visualisations des explorations 1D, separees des calculs et exports CSV."""
from pathlib import Path

import numpy as np

from src.data.generator_1d import fonction_cible, genere_base_apprentissage, genere_base_test
from src.experiments.runner_1d_nominal import run_degree_sweep
from src.models import predict
from src.visualization.style import floor_for_log, new_figure, save_figure, style_context


def plot_campaign(campaigns, label):
    with style_context():
        fig = new_figure(10, 4.2)
        ax, hist = fig.subplots(1, 2)
        for value, c in campaigns.items():
            s = c.summary('test')
            line, = ax.semilogy(c.degrees, floor_for_log(s['median']), label=f'{label} = {value}')
            ax.fill_between(c.degrees, floor_for_log(s['q1']), floor_for_log(s['q3']),
                            color=line.get_color(), alpha=.12)
            hist.plot(c.degrees, [c.cstar_counts()[int(d)] / len(c.seeds) for d in c.degrees],
                      'o-', ms=3, color=line.get_color(), label=f'{label} = {value}')
        ax.set(xlabel='degré', ylabel='EQMT médiane (Q1–Q3)', title='Erreur de test')
        hist.set(xlabel='degré $C^*$', ylabel='proportion des réalisations', title='Distribution de $C^*$')
        ax.legend(fontsize=7)
        hist.legend(fontsize=7)
    return fig


def plot_size_illustration(sizes):
    xt, yt = genere_base_test()
    with style_context():
        fig = new_figure()
        ax = fig.add_subplot()
        for n in sizes:
            x, y = genere_base_apprentissage(nb_ex=n, seed=42)
            result = run_degree_sweep(x, y, xt, yt, diagnose=False, seed=42)
            ax.semilogy(result.degrees, floor_for_log(result.eqm_test), 'o-', ms=3, label=f'N = {n}')
        ax.set(xlabel='degré', ylabel='EQMT', title='Taille : réalisation illustrative, graine 42, σ = 0,2')
        ax.legend(ncols=2)
    return fig


def plot_extrapolation(rows):
    x, y = genere_base_apprentissage(seed=42)
    xt, yt = genere_base_test()
    result = run_degree_sweep(x, y, xt, yt, diagnose=False, seed=42)
    xe = np.linspace(-.3, 1.5, 1001)
    with style_context():
        fig = new_figure(10, 4.2)
        ax, stats = fig.subplots(1, 2)
        ax.plot(xe, fonction_cible(xe), color='black', label='cible')
        for role, d in [('1', 1), ('3', 3), ('$C^*$', result.optimal_degree), ('10', 10)]:
            pred = predict(result.coefs[result.index_of(d)], xe)
            clipped = np.any((pred < -3) | (pred > 3))
            ax.plot(xe, pred, label=f'{role} (d={d})' + (' : hors cadre' if clipped else ''))
        ax.scatter(x[:, 0], y, s=10, color='black', marker='.', label='apprentissage')
        for bound in (0, 1.2):
            ax.axvline(bound, color='gray', ls='--')
        ax.set(xlabel='x', ylabel='y', ylim=(-3, 3), title='Modèles, graine 42')
        ax.legend(fontsize=7)
        roles = ('1', '3', 'C*', '10')
        for key, label in [('eqm_interp', 'interpolation'), ('eqm_extrap', 'extrapolation')]:
            medians = [np.median([r[key] for r in rows if r['role'] == role]) for role in roles]
            stats.semilogy(roles, floor_for_log(medians), 'o-', label=label)
        n_seeds = len({row['seed'] for row in rows})
        stats.set(xlabel='modèle (C* par graine)', ylabel='EQM médiane',
                  title=f'{n_seeds} graines : erreurs par domaine')
        stats.legend()
    return fig


def save_exploration_figures(sizes, sigmas, extra, directory):
    figures = [('exploration_nb_ex', plot_campaign(sizes, 'N')),
               ('exploration_sigma', plot_campaign(sigmas, 'σ')),
               ('exploration_nb_ex_seed42', plot_size_illustration(sizes)),
               ('exploration_extrapolation', plot_extrapolation(extra))]
    paths = []
    for name, fig in figures:
        paths.extend(save_figure(fig, Path(directory) / name))
    return paths
