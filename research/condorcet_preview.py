"""Exploratory Condorcet graphic from the existing NBA winner-only panel curve.

Reproduce from the repository root:
    .venv/bin/python -B research/condorcet_preview.py

This is a descriptive comparison to a homogeneous iid binomial model, not an
estimate of dependence or a new empirical study. No uncertainty is inferred.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.stats import binom

try:
    from .panel_power import first_equivalent_size
except ImportError:
    from panel_power import first_equivalent_size


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'benchmarks/results/sports-survey-equivalence/panel_curves.csv'
MANIFEST = SOURCE.with_name('results.json')
OUT = ROOT / 'benchmarks/results/condorcet-plan-preview'
REPRODUCE = '.venv/bin/python -B research/condorcet_preview.py'


def majority_accuracy(k, p):
    """Exact binomial majority accuracy, awarding half credit to even ties."""
    result = math.fsum(
        math.comb(k, correct) * p**correct * (1 - p)**(k - correct)
        for correct in range(k // 2 + 1, k + 1)
    )
    if k % 2 == 0:
        result += .5 * math.comb(k, k // 2) * p**(k // 2) * (1 - p)**(k // 2)
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with SOURCE.open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['method'] == 'winner_only']
    rows.sort(key=lambda row: int(row['k']))
    ks = np.array([int(row['k']) for row in rows])
    observed = np.array([float(row['accuracy']) for row in rows])
    np.testing.assert_array_equal(ks, np.arange(1, 11))
    assert np.all(np.isfinite(observed)) and np.all((observed >= 0) & (observed <= 1))
    meta = json.loads(MANIFEST.read_text())

    p = float(observed[0])
    iid = np.array([majority_accuracy(int(k), p) for k in ks])
    scipy_check = binom.sf(ks // 2, ks, p) + np.where(
        ks % 2 == 0, .5 * binom.pmf(ks // 2, ks, p), 0
    )
    np.testing.assert_allclose(iid, scipy_check, rtol=0, atol=1e-14)
    np.testing.assert_allclose(iid[0::2], iid[1::2], rtol=0, atol=1e-14)
    np.testing.assert_allclose(observed[0::2], observed[1::2], rtol=0, atol=1e-14)
    # Also verify the formula on chance, below-chance and perfect-accuracy inputs.
    for check_p in [0, .2, .5, .7, 1]:
        for k in range(1, 11):
            expected = binom.sf(k // 2, k, check_p)
            if k % 2 == 0:
                expected += .5 * binom.pmf(k // 2, k, check_p)
            assert abs(majority_accuracy(k, check_p) - expected) < 1e-14

    target = float(observed[-1])
    equivalent = first_equivalent_size(ks, iid, target)
    whole_equivalent = int(ks[np.flatnonzero(iid >= target)[0]])
    observed_first_match = first_equivalent_size(ks, observed, target)
    assert 2 < equivalent < 3 and whole_equivalent == 3 and observed_first_match == 9

    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/condorcet-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12,
        'axes.labelsize': 10, 'svg.fonttype': 'none', 'axes.titleweight': 'bold',
    })
    teal, orange, ink = '#187c83', '#b96828', '#283744'
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 7.1), gridspec_kw={'width_ratios': [1.45, 1]}
    )
    fig.subplots_adjust(left=.075, right=.96, top=.76, bottom=.29, wspace=.28)
    fig.suptitle('Condorcet benchmark: an NBA preview', x=.075, y=.965,
                 ha='left', fontsize=19, fontweight='bold', color=ink)
    fig.text(.075, .905, 'Observed finite-pool accuracy compared with a homogeneous iid majority model',
             color=ink, fontsize=12)
    fig.text(.075, .858, f'Model input: p = observed one-person accuracy = {p:.2%}  |  Half credit for ties',
             color='#515a63', fontsize=10)

    left, right = axes
    left.plot(ks, iid, color=orange, marker='o', markersize=4, linewidth=2,
              label='Homogeneous iid model')
    left.plot(ks, observed, color=teal, marker='o', markersize=4, linewidth=2,
              label='Observed NBA pools')
    left.set(title='Panel accuracy', xlabel='People per panel (k)',
             ylabel='Expected winner accuracy', xlim=(.7, 10.4), ylim=(.61, .825),
             xticks=ks, yticks=np.arange(.62, .83, .04))
    left.legend(loc='center left', bbox_to_anchor=(0, .52), frameon=False, fontsize=10)
    left.annotate(f'k = 9 or 10: {iid[-1]:.2%}', xy=(9.5, iid[-1]),
                  xytext=(6, .809), color=orange, fontsize=10)
    left.annotate(f'k = 9 or 10: {target:.2%}', xy=(9.5, target),
                  xytext=(6, .638), color=teal, fontsize=10)

    right.plot(ks[:3], iid[:3], color=orange, marker='o', markersize=5, linewidth=2)
    right.axhline(target, color=teal, linestyle='--', linewidth=1.4)
    right.vlines(equivalent, .62, target, color=ink, linestyle=':', linewidth=1.4)
    right.scatter([equivalent], [target], s=75, facecolor='white',
                   edgecolor=orange, linewidth=2, zorder=5)
    right.text(1.06, target + .0035, f'10-person NBA target: {target:.2%}', color=teal, fontsize=10)
    right.text(1.06, .698, f'Interpolated model size: {equivalent:.2f}\nSmallest whole model panel: {whole_equivalent}',
               color=ink, fontsize=11, linespacing=1.6, va='top')
    right.text(equivalent, .622, f'{equivalent:.2f}', color=ink, fontsize=10,
               ha='center', va='bottom', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    right.set(title='Equivalent size in the iid model', xlabel='Model jurors (k; zoomed view)',
              xlim=(.9, 3.15), ylim=(.62, .705), xticks=[1, 2, 3])

    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis='y', color='#d8dde2', linewidth=.7, alpha=.65)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#a7b0b9')
        ax.tick_params(colors='#48525b')

    fig.text(.075, .193, 'EXPLORATORY PREVIEW  ·  A descriptive model comparison; no new uncertainty analysis.',
             color=ink, fontsize=10, fontweight='bold')
    fig.text(.075, .145,
             f'Existing data: {meta["games"]} games, {meta["participants"]} people, {meta["complete_responses"]:,} complete responses; '
             'uniform finite-pool panels without replacement.\n'
             'The gap does not identify social dependence: item difficulty, voter heterogeneity and finite sampling also differ.\n'
             'Equivalent size uses first crossing on the full integer grid, with linear interpolation; odd/even plateaus are retained.',
             color='#515a63', fontsize=9.2, linespacing=1.65, va='top')
    fig.text(.075, .034, 'Source: Rutchick et al. (2020), NBA Study 3 release; existing repository panel_curves.csv.',
             color='#69727a', fontsize=8.5)
    for extension in ['png', 'svg']:
        fig.savefig(OUT / f'condorcet_nba_preview.{extension}', dpi=180, facecolor='white')
    plt.close(fig)

    result = {
        'status': 'Exploratory descriptive preview; not a new empirical study.',
        'reproduction_command': REPRODUCE,
        'input': {
            'curve_path': str(SOURCE.relative_to(ROOT)), 'curve_sha256': sha256(SOURCE),
            'manifest_path': str(MANIFEST.relative_to(ROOT)), 'manifest_sha256': sha256(MANIFEST),
            'dataset': meta['dataset'], 'dataset_url': meta['source_url'],
            'dataset_sha256': meta['source_sha256'], 'method': 'winner_only',
            'games': meta['games'], 'participants': meta['participants'],
            'complete_responses': meta['complete_responses'],
        },
        'script': {'path': str(Path(__file__).resolve().relative_to(ROOT)), 'sha256': sha256(Path(__file__))},
        'model': {
            'description': 'Independent Bernoulli correctness with constant p across jurors and items.',
            'p': p, 'p_source': 'Observed game-averaged k=1 accuracy, a descriptive plug-in value.',
            'formula': 'P[Binomial(k,p) > k/2] + 0.5 * P[Binomial(k,p) = k/2]',
            'ties': 'Half credit; ties occur only at even k.',
        },
        'curves': [
            {'k': int(k), 'observed_accuracy': float(o), 'iid_model_accuracy': float(i),
             'model_minus_observed_pp': float(100 * (i - o))}
            for k, o, i in zip(ks, observed, iid)
        ],
        'reference': {
            'k': 10, 'observed_accuracy': target, 'iid_model_accuracy_at_same_k': float(iid[-1]),
            'model_minus_observed_pp': float(100 * (iid[-1] - target)),
            'observed_first_matching_k': observed_first_match,
        },
        'equivalence': {
            'direction': 'Number of homogeneous iid model jurors matching observed 10-person NBA accuracy.',
            'interpolated_size': equivalent, 'smallest_integer_size': whole_equivalent,
            'method': 'research.panel_power.first_equivalent_size on full integer k=1..10; no smoothing.',
            'interpretation': 'Descriptive model projection; not a measured number of independent real people.',
            'uncertainty': 'None estimated. This preview does not reuse pointwise curve bands as equivalence intervals.',
        },
        'caveat': 'The gap does not identify social dependence: item difficulty, voter heterogeneity and finite sampling also differ.',
        'verification': {
            'binomial_formula': 'Direct binomial sum agrees with scipy.stats.binom to absolute tolerance 1e-14.',
            'additional_p_values': [0, .2, .5, .7, 1],
            'odd_even_plateaus': 'Both curves verified at all adjacent odd/even k to absolute tolerance 1e-14.',
        },
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
    }
    (OUT / 'preview_numerics.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'README.md').write_text(
        '# Condorcet / rater-equivalence preview\n\n'
        'A descriptive preview based on the existing NBA winner-only panel curve. '
        'No source data or existing results are modified.\n\n'
        'Reproduce from the repository root:\n\n'
        f'```sh\n{REPRODUCE}\n```\n\n'
        '- `condorcet_nba_preview.png`: review image.\n'
        '- `condorcet_nba_preview.svg`: scalable graphic.\n'
        '- `preview_numerics.json`: exact plotted values, crossing, source hashes and verification.\n\n'
        f'Observed accuracy at 9 and 10 people is {target:.6%}; the homogeneous iid '
        f'model with p={p:.6f} reaches {iid[-1]:.6%} at those sizes. '
        f'The model reaches the observed target at {equivalent:.6f} interpolated '
        f'jurors, or {whole_equivalent} whole jurors. The interpolated number is '
        'a model projection, not a measured count of independent real people.\n\n'
        'The gap does not identify social dependence: item difficulty, voter '
        'heterogeneity and finite sampling also differ. No new uncertainty '
        'analysis is performed. Odd/even plateaus and the full integer grid are retained.\n'
    )
    print(json.dumps({'output': str(OUT), 'p': p, 'observed_k10': target,
                      'iid_k10': float(iid[-1]), 'equivalent_iid_size': equivalent,
                      'smallest_integer_iid_size': whole_equivalent}, indent=2))


if __name__ == '__main__':
    main()
