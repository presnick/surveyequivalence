"""Exact NBA panel accuracy and equivalence, conditional on released respondents.

Run sports_survey_data.py first. No odds, betting returns, or model fitting.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .panel_power import equivalence_label, first_equivalent_size
except ImportError:
    from panel_power import equivalence_label, first_equivalent_size

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.cache/sports-survey-data/nba_complete_responses.jsonl'
OUT = ROOT / 'benchmarks/results/sports-survey-equivalence'
METHODS = ['winner_only', 'confidence_weighted', 'surprisingly_popular']
LABELS = ['Winner only', 'Winner + confidence', 'Winner + peer agreement']
KMAX = 10
SEED = 20260924
BOOTSTRAPS = 10000


def exact_curve(weights, kmax=KMAX):
    """Uniform subsets without replacement; positive signed sum is correct.

    Integer subset counts are exact; only the final divisions use floats.
    Zero sums receive half credit. Weights are oriented to the actual winner.
    """
    values = np.asarray(weights)
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise ValueError('Weights must be finite integers')
    values = values.astype(np.int64)
    n = len(values)
    if not 1 <= kmax <= n:
        raise ValueError('Panel range exceeds available people')
    if max(math.comb(n, k) for k in range(kmax + 1)) > np.iinfo(np.int64).max:
        raise ValueError('Subset counts would overflow int64')
    bound = kmax * max(1, int(np.max(np.abs(values))))
    counts = np.zeros((kmax + 1, 2 * bound + 1), dtype=np.int64)
    counts[0, bound] = 1
    for i, w in enumerate(values):
        for k in range(min(i + 1, kmax), 0, -1):
            if w > 0:
                counts[k, w:] += counts[k - 1, :-w]
            elif w < 0:
                counts[k, :w] += counts[k - 1, -w:]
            else:
                counts[k] += counts[k - 1]
    denominators = np.array([math.comb(n, k) for k in range(1, kmax + 1)])
    np.testing.assert_array_equal(counts[1:].sum(axis=1), denominators)
    return (counts[1:, bound + 1:].sum(axis=1) + .5 * counts[1:, bound]) / denominators


def first_crossing(curves, target):
    """Smallest integer k reaching the target; 11 encodes >10, not k=11."""
    eligible = curves >= np.asarray(target)[..., None] - 1e-12
    return np.where(eligible.any(axis=-1), eligible.argmax(axis=-1) + 1, curves.shape[-1] + 1)


def panel_label(value):
    return '>10' if value > KMAX else str(int(value))


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    from sports_survey_data import EXPECTED
    original = DATA.parent / 'nba_data.xlsx'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == EXPECTED
    groups = defaultdict(list)
    records = [json.loads(line) for line in DATA.read_text().splitlines()]
    assert len(records) == 9788
    assert len({(r['participant_id'], r['game_id']) for r in records}) == len(records)
    for record in records:
        groups[record['game_id']].append(record)
    games = sorted(groups)
    assert len(games) == 465
    curves = np.empty((len(games), len(METHODS), KMAX))
    game_waves = []
    for i, game in enumerate(games):
        rows = groups[game]
        assert 10 <= len(rows) <= 61
        assert len({r['winner_code'] for r in rows}) == 1
        assert len({r['study_wave'] for r in rows}) == 1
        game_waves.append(rows[0]['study_wave'])
        sign = np.array([1 if r['pick_code'] == r['winner_code'] else -1 for r in rows])
        confidence = np.array([r['confidence_ordinal'] for r in rows])
        agreement = np.array([r['predicted_peer_agreement_percent'] for r in rows])
        # Excel represents one 57% value as 56.9999999999999; no substantive rounding.
        assert np.max(np.abs(agreement - np.rint(agreement))) < 1e-9
        weights = [sign, sign * confidence, sign * (100 - np.rint(agreement))]
        for m, values in enumerate(weights):
            curves[i, m] = exact_curve(values)
        # Independent hypergeometric majority calculation on every real game.
        good, n = int((sign > 0).sum()), len(sign)
        expected = []
        for k in range(1, KMAX + 1):
            numerator = sum(math.comb(good, j) * math.comb(n-good, k-j) *
                            (1 if 2*j > k else .5 if 2*j == k else 0)
                            for j in range(max(0, k-(n-good)), min(k, good)+1))
            expected.append(numerator / math.comb(n, k))
        np.testing.assert_allclose(curves[i, 0], expected, atol=1e-14, rtol=0)
    mean = curves.mean(axis=0)
    reference = mean[0, -1]
    equivalence = first_crossing(mean, reference)

    # Cluster bootstrap: sample 11 study waves with replacement; retain all games.
    waves = sorted(set(game_waves))
    membership = np.array([[w == wave for w in game_waves] for wave in waves])
    block_sums = np.einsum('wg,gmk->wmk', membership, curves)
    block_sizes = membership.sum(axis=1)
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(len(waves), np.full(len(waves), 1/len(waves)), size=BOOTSTRAPS)
    boot = np.einsum('bw,wmk->bmk', draws, block_sums) / (draws @ block_sizes)[:, None, None]
    boot_reference = boot[:, 0, -1]
    boot_eq = first_crossing(boot, boot_reference[:, None])
    interval = np.quantile(boot_eq, [.025, .975], axis=0, method='inverted_cdf').astype(int)
    accuracy_interval = np.quantile(boot, [.025, .975], axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    for m, method in enumerate(METHODS):
        k = int(equivalence[m])
        table.append(dict(method=method, people_to_match_10_winner_only=panel_label(k),
                          bootstrap_95_lower=panel_label(interval[0,m]),
                          bootstrap_95_upper=panel_label(interval[1,m]),
                          bootstrap_no_crossing_fraction=float(np.mean(boot_eq[:,m] > KMAX)),
                          accuracy_at_equivalent_k=float(mean[m,k-1]) if k <= KMAX else None,
                          accuracy_at_10=float(mean[m,-1]),
                          accuracy_at_10_minus_reference_pp=float(100*(mean[m,-1]-reference)),
                          difference_at_10_95_lower_pp=float(np.quantile(100*(boot[:,m,-1]-boot_reference),.025)),
                          difference_at_10_95_upper_pp=float(np.quantile(100*(boot[:,m,-1]-boot_reference),.975))))
    manifest = dict(dataset='Rutchick et al. 2020, NBA Study 3 released workbook',
                    source_url='https://osf.io/download/zwsgu/', source_sha256=EXPECTED,
                    prepared_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                    games=len(games), participants=len({r['participant_id'] for r in records}),
                    complete_responses=len(records), waves=waves, wave_game_counts=dict(zip(waves, map(int,block_sizes))),
                    panel_range=[1,KMAX], score='Game-averaged actual-winner accuracy; ties receive half credit',
                    calculation='Exact integer subset-sum dynamic programming, uniform panels without replacement',
                    reference_accuracy=float(reference), reference_people=10, reference_method=METHODS[0],
                    reference_accuracy_95_interval=np.quantile(boot_reference,[.025,.975]).tolist(),
                    equivalence='Smallest integer k whose accuracy reaches the 10-person winner-only reference; >10 denotes no crossing in 1..10',
                    bootstrap=dict(seed=SEED, replicates=BOOTSTRAPS, unit='Study wave',
                                   method='Resample 11 whole waves with replacement; game-weighted accuracy; reference recomputed each draw',
                                   equivalence_interval='2.5% and 97.5% inverse empirical CDF; >10 censoring retained',
                                   conditional_on='Observed respondents and their recorded responses'),
                    limits=['Only 11 survey waves; bootstrap intervals are exploratory.',
                            'This compares response subsets and aggregation rules, not randomized questionnaire assignment.',
                            'Respondent pools and membership vary by game; uncertainty does not cover new respondent recruitment.',
                            'Release differs from final published participant counts and aggregate results.',
                            'No odds, wagers, or betting returns are analyzed.'],
                    results=table)
    (OUT/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save_csv(OUT/'equivalent_panels.csv', table)
    save_csv(OUT/'panel_curves.csv', [dict(method=method,k=k+1,accuracy=float(mean[m,k]),
             bootstrap_95_lower=float(accuracy_interval[0,m,k]),bootstrap_95_upper=float(accuracy_interval[1,m,k]))
             for m,method in enumerate(METHODS) for k in range(KMAX)])
    save_csv(OUT/'game_panel_curves.csv', [dict(game_id=game,study_wave=game_waves[i],method=method,k=k+1,
             accuracy=float(curves[i,m,k])) for i,game in enumerate(games)
             for m,method in enumerate(METHODS) for k in range(KMAX)])
    np.savez_compressed(OUT/'bootstrap_draws.npz', wave_multiplicities=draws,
                        equivalence=boot_eq, reference_accuracy=boot_reference)
    plot(mean, boot, reference)
    print(json.dumps(manifest, indent=2))


def plot_saved():
    """Reconstruct the original paired bootstrap without recalculating panels."""
    meta = json.loads((OUT / 'results.json').read_text())
    with (OUT / 'game_panel_curves.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    games = sorted({row['game_id'] for row in rows})
    game_positions = {game: i for i, game in enumerate(games)}
    method_positions = {method: i for i, method in enumerate(METHODS)}
    curves = np.full((len(games), len(METHODS), KMAX), np.nan)
    game_waves = {}
    for row in rows:
        game, wave = row['game_id'], row['study_wave']
        if game in game_waves and game_waves[game] != wave:
            raise ValueError(f'Inconsistent survey wave for game {game}')
        game_waves[game] = wave
        index = (game_positions[game], method_positions[row['method']], int(row['k']) - 1)
        if np.isfinite(curves[index]):
            raise ValueError(f'Duplicate saved curve point: {row}')
        curves[index] = float(row['accuracy'])
    if not np.isfinite(curves).all() or len(games) != meta['games']:
        raise ValueError('Saved game curves are incomplete')
    waves = meta['waves']
    if set(game_waves.values()) != set(waves):
        raise ValueError('Saved game waves differ from the bootstrap wave ordering')
    membership = np.array([[game_waves[game] == wave for game in games] for wave in waves])
    block_sums = np.einsum('wg,gmk->wmk', membership, curves)
    block_sizes = membership.sum(axis=1)
    np.testing.assert_array_equal(block_sizes, [meta['wave_game_counts'][wave] for wave in waves])
    mean = curves.mean(axis=0)
    with np.load(OUT / 'bootstrap_draws.npz') as saved:
        draws = saved['wave_multiplicities']
        if draws.shape != (meta['bootstrap']['replicates'], len(waves)):
            raise ValueError('Saved bootstrap dimensions differ from the manifest')
        boot = np.einsum('bw,wmk->bmk', draws, block_sums) / (draws @ block_sizes)[:, None, None]
        np.testing.assert_allclose(boot[:, 0, -1], saved['reference_accuracy'], rtol=0, atol=1e-12)
    intervals = np.quantile(boot, [.025, .975], axis=0)
    with (OUT / 'panel_curves.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            m, k = method_positions[row['method']], int(row['k']) - 1
            np.testing.assert_allclose(
                [mean[m, k], intervals[0, m, k], intervals[1, m, k]],
                [float(row['accuracy']), float(row['bootstrap_95_lower']), float(row['bootstrap_95_upper'])],
                rtol=0, atol=1e-12)
    np.testing.assert_allclose(mean[0, -1], meta['reference_accuracy'], rtol=0, atol=1e-12)
    plot(mean, boot, meta['reference_accuracy'])
    print('Rebuilt sports figures; saved means and bootstrap intervals verified to 1e-12.')


def plot(mean, boot, reference):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#526478', '#187c83', '#ba6334']
    fig, axes = plt.subplots(1,2,figsize=(13,6.4),gridspec_kw={'width_ratios':[1.12,1]})
    x = np.arange(1,KMAX+1)
    floor = 100 * mean.min() - .7
    axes[0].set_ylim(floor, 100 * max(mean.max(), reference) + 1)
    for m,label in enumerate(LABELS):
        equivalent = first_equivalent_size(x, mean[m], reference)
        axes[0].plot(x,100*mean[m],marker='o',markersize=4,color=colors[m],
                     label=f'{label}  |  REQ {equivalence_label(equivalent, x)}')
        if np.isfinite(equivalent):
            axes[0].vlines(equivalent,floor,100*reference,color=colors[m],ls=':',lw=1.4,alpha=.9)
            axes[0].scatter([equivalent],[100*reference],s=66,facecolor='white',
                            edgecolor=colors[m],linewidth=1.8,zorder=5)
        diff = 100*(boot[:,m] - boot[:,0,-1,None])
        low,high = np.quantile(diff,[.025,.975],axis=0)
        axes[1].fill_between(x,low,high,color=colors[m],alpha=.10)
        axes[1].plot(x,100*(mean[m]-reference),marker='o',markersize=4,color=colors[m])
    axes[0].axhline(100*reference,color='#343434',linestyle='--',linewidth=1)
    axes[0].set(ylabel='Expected winner accuracy (%) — higher is better',title='Power curves and first matching panel sizes')
    axes[0].text(.02,.96,f'10 winner-only reference: {100*reference:.3f}%',
                 transform=axes[0].transAxes,va='top',fontsize=9,color='#343434')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.055,.91),
               ncol=3,frameon=False,fontsize=9)
    axes[1].axhline(0,color='#343434',linestyle='--',linewidth=1)
    axes[1].set(ylabel='Accuracy difference (percentage points)',title='Relative to 10 winner-only respondents')
    for ax in axes:
        ax.set(xlabel='Number of benchmark raters per game (k)',xticks=x,xlim=(1,10))
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('NBA questionnaire rater equivalence',fontsize=17,fontweight='bold',x=.06,ha='left')
    fig.text(.06,.025,'REQ: first crossing of the reference, linearly interpolated between measured panel sizes; dotted guides project it to k.\n'
             '465 games | 113 people | 9,788 responses. Shading: paired pointwise 95% bootstrap intervals from 11 survey waves.\n'
             'Scored against actual winners; empirical plateaus and reversals retained. Tables report whole-person thresholds.',
             fontsize=9,color='#444444',linespacing=1.5)
    fig.tight_layout(rect=(.02,.15,1,.91))
    for extension in ['png','svg']:
        fig.savefig(OUT/f'panel_accuracy.{extension}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true', help='Verify saved curves and rebuild only PNG/SVG figures')
    args = parser.parse_args()
    if args.plot_only:
        plot_saved()
    else:
        main()
