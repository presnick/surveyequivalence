"""Real candy-jar and other estimates before/after social feedback.

Exact finite-pool curves condition on intact observed groups. A sampled report
after feedback can contain information from people outside the sampled panel.
Run with the original Becker, Brackbill & Centola (2017) CSV; --plot-only uses
saved results and does not need the source data. No simulated human responses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .panel_power import first_equivalent_size, equivalence_label
except ImportError:
    from panel_power import first_equivalent_size, equivalence_label

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.cache/social-influence-data/becker2017/becker_brackbill_centola_supporting_data.csv'
SOURCE_SHA = '2c12eac9df56e80bd8d4145b574116a54910e2efa888cf48a190ec1746c16fa1'
SOURCE_URL = 'https://ndg.asc.upenn.edu/wp-content/uploads/2017/09/NDG_becker_brackbill_centola_collective_intelligence_dataset.zip'
PAPER = 'https://doi.org/10.1073/pnas.1615978114'
OUT = ROOT / 'benchmarks/results/social-influence-equivalence'
PDF = ROOT / 'output/pdf/social_influence_equivalence.pdf'
KS = np.arange(1, 32)
REFERENCE_K = 10
SEED = 28092026
BOOTSTRAPS = 5000
NETWORKS = ['Solo', 'Decentralized', 'Centralized']
TITLES = ['Private throughout', 'Feedback from several peers', 'Feedback dominated by one person']
COLORS = ['#657386', '#138575', '#C06337']
INK, MUTED, GRID, BEFORE = '#202C3B', '#586879', '#DEE5EB', '#34495E'


def panel_mse(errors, ks):
    """Exact squared error of a uniformly sampled mean of k distinct reports."""
    errors, ks = np.asarray(errors, float), np.asarray(ks, float)
    if (errors.ndim != 1 or len(errors) < 2 or not np.isfinite(errors).all()
            or ks.ndim != 1 or not np.isfinite(ks).all()
            or np.any(ks != np.floor(ks)) or np.any(ks < 1) or np.any(ks > len(errors))):
        raise ValueError('Need finite errors and supported integer panel sizes')
    return errors.mean() ** 2 + errors.var(ddof=1) * (1 / ks - 1 / len(errors))


def first_integer(curves, targets):
    curves = np.atleast_2d(curves)
    targets = np.asarray(targets).reshape(-1, 1)
    tolerance = 1e-12 * np.maximum(1., np.abs(targets))
    reached = curves <= targets + tolerance
    return np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, np.inf)


def quantile_with_censoring(values, q):
    """Inverse empirical CDF; +inf draws stay in the distribution."""
    values = np.sort(values)
    return float(values[max(0, min(len(values) - 1, int(np.ceil(q * len(values))) - 1))])


def bound(value):
    return float(value) if np.isfinite(value) else '>31'


def load_data(path):
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != SOURCE_SHA:
        raise ValueError('Input differs from the verified original author release')
    raw = pd.read_csv(path)
    assert not raw.duplicated(['group_number', 'task', 'subject_id']).any()
    assert raw.groupby('subject_id').group_number.nunique().max() == 1
    assert raw.groupby('group_number').network.nunique().max() == 1
    assert raw.groupby('task').truth.nunique().max() == 1
    assert raw.groupby('task').sd_pool.nunique().max() == 1
    assert (raw.sd_pool > 0).all()
    np.testing.assert_allclose(raw.groupby('task').response_1.std(),
                               raw.groupby('task').sd_pool.first(), rtol=1e-12)
    paired = raw.dropna(subset=['response_1', 'response_3']).copy()
    assert np.isfinite(paired[['response_1', 'response_3', 'truth', 'sd_pool']]).all().all()
    assert paired.groupby(['group_number', 'task']).size().min() == KS[-1]
    assert len(raw) == 5909 and len(paired) == 5626
    return raw, paired


def analyze(path=SOURCE):
    raw, paired = load_data(path)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, check_errors = [], []
    for (network, group, task), data in paired.groupby(['network', 'group_number', 'task']):
        truth, scale = float(data.truth.iloc[0]), float(data.sd_pool.iloc[0])
        for r in [1, 3]:
            response = data[f'response_{r}'].to_numpy()
            errors = (response - truth) / scale
            curve = panel_mse(errors, KS)
            ii, jj = np.triu_indices(len(errors), k=1)
            direct = [np.mean(errors ** 2), np.mean(((errors[ii] + errors[jj]) / 2) ** 2), errors.mean() ** 2]
            formula = panel_mse(errors, [1, 2, len(errors)])
            np.testing.assert_allclose(formula, direct, rtol=1e-12, atol=1e-12)
            assert np.all(np.diff(curve) <= 1e-12)
            check_errors.extend(np.abs(formula - direct).tolist())
            rows.append(dict(network=network, group=int(group), task=task, round=r,
                             n=len(data), truth=truth, scale=scale,
                             mean=float(response.mean()), sd=float(response.std(ddof=1)),
                             squared_group_error=float(errors.mean() ** 2),
                             variance=float(errors.var(ddof=1)),
                             standardized_sd=float(errors.std(ddof=1))))
    moments = pd.DataFrame(rows)
    moments.to_csv(OUT / 'group_task_moments.csv', index=False)
    paired[['group_number', 'network', 'task', 'subject_id', 'is_central',
            'response_1', 'response_3', 'truth', 'sd_pool']].to_csv(OUT / 'paired_responses.csv', index=False)
    rng = np.random.default_rng(SEED)
    curve_rows, eq_rows, boot_save = [], [], {}
    for subset in ['candy', 'all']:
        selected = moments[moments.task.str.startswith('3')] if subset == 'candy' else moments
        for network in NETWORKS:
            cells = selected[selected.network == network]
            groups = sorted(cells.group.unique())
            group_curves, group_diagnostics = [], []
            for group in groups:
                g = cells[cells.group == group]
                round_curves, round_diagnostics = [], []
                for r in [1, 3]:
                    gr = g[g['round'] == r]
                    curve = gr.squared_group_error.to_numpy()[:, None] + gr.variance.to_numpy()[:, None] * (
                        1 / KS[None, :] - 1 / gr.n.to_numpy()[:, None])
                    round_curves.append(curve.mean(axis=0))
                    round_diagnostics.append([gr.squared_group_error.mean(), gr.standardized_sd.mean()])
                group_curves.append(round_curves)
                group_diagnostics.append(round_diagnostics)
            group_curves, group_diagnostics = np.asarray(group_curves), np.asarray(group_diagnostics)
            mean = group_curves.mean(axis=0)
            draws = rng.integers(0, len(groups), size=(BOOTSTRAPS, len(groups)))
            boot = group_curves[draws].mean(axis=1)
            target = mean[0, REFERENCE_K - 1]
            boot_target = boot[:, 0, REFERENCE_K - 1]
            eq = first_equivalent_size(KS, -mean[1], -target)
            eq_int = first_integer(mean[1], target)[0]
            boot_int = first_integer(boot[:, 1], boot_target)
            boot_frac = np.array([first_equivalent_size(KS, -b[1], -t) for b, t in zip(boot, boot_target)])
            np.testing.assert_array_equal(first_integer(boot[:, 0], boot_target), REFERENCE_K)
            assert first_equivalent_size(KS, -mean[0], -target) == REFERENCE_K
            lower, upper = np.quantile(boot, [.025, .975], axis=0)
            delta = boot[:, 1, REFERENCE_K - 1] - boot_target
            eq_rows.append(dict(subset=subset, network=network, groups=len(groups),
                                tasks=int(cells.task.nunique()), paired_responses=int(cells[cells['round'] == 1].n.sum()),
                                reference_k=REFERENCE_K, reference_mse=float(target),
                                revised_mse_k10=float(mean[1, REFERENCE_K - 1]),
                                revised_minus_initial_mse_k10=float(mean[1, REFERENCE_K - 1] - target),
                                difference_95_low=float(np.quantile(delta, .025)),
                                difference_95_high=float(np.quantile(delta, .975)),
                                equivalent_k=bound(eq), equivalent_integer_k=bound(eq_int),
                                equivalent_integer_95_low=bound(quantile_with_censoring(boot_int, .025)),
                                equivalent_integer_95_high=bound(quantile_with_censoring(boot_int, .975)),
                                bootstrap_fraction_above_31=float(np.isinf(boot_int).mean()),
                                initial_full_group_mse=float(group_diagnostics[:, 0, 0].mean()),
                                revised_full_group_mse=float(group_diagnostics[:, 1, 0].mean())))
            for ri, r in enumerate([1, 3]):
                for j, k in enumerate(KS):
                    curve_rows.append(dict(subset=subset, network=network, round=r, k=int(k),
                                           expected_mse=float(mean[ri, j]), score=float(-mean[ri, j]),
                                           mse_95_low=float(lower[ri, j]), mse_95_high=float(upper[ri, j])))
            key = f'{subset}_{network}'
            boot_save[key + '_sampled_groups'] = np.asarray(groups)[draws]
            boot_save[key + '_equivalent_integer'] = boot_int
            boot_save[key + '_equivalent_fractional'] = boot_frac
            boot_save[key + '_reference_mse'] = boot_target
            boot_save[key + '_mse_difference_k10'] = delta
    pd.DataFrame(curve_rows).to_csv(OUT / 'panel_curves.csv', index=False)
    pd.DataFrame(eq_rows).to_csv(OUT / 'equivalent_panels.csv', index=False)
    np.savez_compressed(OUT / 'bootstrap_draws.npz', **boot_save)
    group_diagnostics = moments.groupby(['network', 'group', 'round'], as_index=False).agg(
        squared_group_error=('squared_group_error', 'mean'), standardized_sd=('standardized_sd', 'mean'))
    group_diagnostics.to_csv(OUT / 'group_diagnostics.csv', index=False)
    metadata = dict(data_kind='Observed human estimates only; no simulated responses',
                    study='Becker, Brackbill & Centola (2017)', paper=PAPER, source=SOURCE_URL,
                    source_sha256=SOURCE_SHA, raw_rows=len(raw), raw_people=int(raw.subject_id.nunique()),
                    paired_rows=len(paired), paired_people=int(paired.subject_id.nunique()),
                    excluded_rows=len(raw) - len(paired),
                    missingness='Paired complete response_1/response_3 within each group-task; round 2 not required. Disconnected subject-task records can be absent entirely in the source.',
                    groups={str(k): int(v) for k, v in raw.groupby('network').group_number.nunique().items()},
                    score='Negative expected squared error after division by the released question-specific sd_pool; zero is perfect.',
                    normalization='Fixed released pooled SD of all initial responses to each question, shared across rounds and conditions. Not recomputed on paired responders.',
                    formula='MSE(k) = mean(z)^2 + sample_variance(z)*(1/k - 1/n), z=(response-truth)/sd_pool',
                    aggregation='Mean of questions within each group, then equal weight per group within condition; no between-condition effect is estimated.',
                    sampling='Uniform k distinct paired respondents within each original group-task; same observed pool before and after; k=1..31 supported everywhere.',
                    reference='10 initial private reports from the SAME condition and SAME task subset. Each facet has its own reference.',
                    uncertainty=f'{BOOTSTRAPS} percentile bootstrap draws of whole groups within each condition, carrying all tasks and both rounds together. Resamples the observed group/task clusters while retaining their paired respondents and released normalization scales. Does not establish generalization to new jars/questions. Exploratory small-cluster intervals; not a formal equivalence test.',
                    seed=SEED, bootstrap_replicates=BOOTSTRAPS,
                    checks=dict(group_task_rounds=len(moments), largest_absolute_enumeration_error=max(check_errors),
                                verified='Direct k=1, exhaustive all k=2 pairs, full-pool mean, monotonicity, self-equivalence in all bootstrap draws'),
                    equivalences=eq_rows,
                    limitations=['Social k counts sampled reports from groups originally assigned 40 people, not k total communicators or independent information sources.',
                                 'Initial versus final is a paired before/after comparison; the no-feedback arm shows revision alone can change accuracy.',
                                 'All-task arms have different task frequencies; within-arm references avoid treating their absolute scores as causal arm comparisons.',
                                 'Candy-only results describe six jars, one group per jar per condition; observations are not Skittles or shouted answers.',
                                 'Squaring makes this reanalysis sensitive to extreme responses. No outliers are trimmed.',
                                 'Nonincreasing curves follow mechanically from averaging a fixed observed pool; they do not prove a causal benefit from adding communicators.',
                                 'The original study reports other metrics and weighting; these panel MSE curves are a new reanalysis, not a reproduction of published effect sizes.'])
    (OUT / 'results.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata


def plot_all():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': GRID,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    meta = json.loads((OUT / 'results.json').read_text())
    curves = pd.read_csv(OUT / 'panel_curves.csv')
    moments = pd.read_csv(OUT / 'group_task_moments.csv')
    diagnostics = pd.read_csv(OUT / 'group_diagnostics.csv')
    PDF.parent.mkdir(parents=True, exist_ok=True)

    def page(title, subtitle, number):
        fig = plt.figure(figsize=(14, 9))
        fig.text(.065, .945, title, fontsize=24, weight='bold', va='top')
        fig.text(.065, .878, subtitle, fontsize=11, color=MUTED, va='top', linespacing=1.55)
        fig.text(.94, .978, f'OBSERVED HUMAN ESTIMATES  /  {number:02d}', fontsize=9, color=MUTED, ha='right')
        fig.text(.065, .028, 'Source: Becker, Brackbill & Centola (2017)  |  doi.org/10.1073/pnas.1615978114  |  Reanalysis: 28 Sep 2026',
                 fontsize=8.5, color=MUTED, url=PAPER)
        return fig

    def save(fig, stem, pdf):
        fig.savefig(OUT / f'{stem}.png', dpi=180)
        fig.savefig(OUT / f'{stem}.svg')
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(PDF, metadata={'Title': 'Independent and socially influenced survey estimates',
                                'Author': 'SurveyEquivalence research'}) as pdf:
        for number, subset in enumerate(['candy', 'all'], 1):
            candy = subset == 'candy'
            fig = page('Candy jars: private estimates and social feedback' if candy else 'The same comparison across all estimation tasks',
                       ('Six actual jars, with 259-291 candies each. Each curve uses recorded answers from the same people before and after revision.\n'
                        if candy else '34 distinct questions in 34 groups. Questions are averaged within groups, then groups receive equal weight within each condition.\n') +
                       'How many final reports match the accuracy of 10 initial private reports from that same condition?', number)
            subset_curves = curves[curves.subset == subset]
            # Display every supported panel size, including the single report.
            floor = min(subset_curves.score) * 1.07
            ceiling = max(subset_curves.score) + .07 * abs(floor)
            for ni, network in enumerate(NETWORKS):
                ax = fig.add_axes([.075 + .30 * ni, .32, .255, .415])
                c = subset_curves[subset_curves.network == network]
                entry = next(e for e in meta['equivalences'] if e['subset'] == subset and e['network'] == network)
                initial = c[c['round'] == 1].sort_values('k')
                final = c[c['round'] == 3].sort_values('k')
                ax.plot(initial.k, initial.score, color=BEFORE, lw=2, ls='--', label='Initial private estimates')
                ax.plot(final.k, final.score, color=COLORS[ni], lw=2.8,
                        label='After private revision' if ni == 0 else 'After two feedback rounds')
                target = -entry['reference_mse']
                ax.axhline(target, color='#9A865A', lw=1.1, ls=':')
                ax.scatter([REFERENCE_K], [target], marker='s', s=36, color=BEFORE, zorder=5)
                eq = entry['equivalent_k']
                if isinstance(eq, (int, float)):
                    ax.vlines(eq, floor, target, color=COLORS[ni], lw=1.2, ls=':')
                    ax.scatter([eq], [target], s=68, color=COLORS[ni], edgecolors='white', zorder=6)
                ax.set(xlim=(1, 31), ylim=(floor, ceiling), xticks=[1, 5, 10, 20, 31],
                       xlabel='Reports included in the panel (k)')
                ax.set_title(TITLES[ni], fontsize=11, weight='bold', loc='left', pad=18)
                ax.grid(axis='y', color=GRID, lw=.7)
                if ni == 0:
                    ax.set_ylabel('Expected score (-standardized MSE)\nHigher is more accurate; zero is perfect', labelpad=11)
                else:
                    ax.tick_params(labelleft=False)
                ax.legend(loc='lower right', fontsize=8.1, frameon=False)
                label = equivalence_label(float(eq) if not isinstance(eq, str) else np.inf, KS)
                x = .075 + .30 * ni
                fig.text(x, .251, f'{label} final reports match 10 initial', fontsize=12, color=COLORS[ni], weight='bold')
                lo, hi = entry['equivalent_integer_95_low'], entry['equivalent_integer_95_high']
                lo_s = f'{lo:g}' if isinstance(lo, (int, float)) else lo
                hi_s = f'{hi:g}' if isinstance(hi, (int, float)) else hi
                integer = entry['equivalent_integer_k']
                integer_s = f'{integer:g}' if isinstance(integer, (int, float)) else integer
                fig.text(x, .208, f'Whole reports: {integer_s}  |  95% interval: {lo_s} to {hi_s}\n'
                         f'{entry["groups"]} groups; {entry["paired_responses"]:,} paired person-task records',
                         fontsize=9, color=MUTED, linespacing=1.65)
            fig.text(.065, .119, 'The dotted horizontal reference is specific to each condition. Fractional crossings interpolate expected score.\n'
                     'Intervals resample whole groups, preserving both rounds and every task; >31 retains draws without a supported crossing.',
                     fontsize=9.2, color=MUTED, linespacing=1.65)
            fig.text(.065, .069, 'k counts sampled reports from groups originally assigned 40 people. It does not measure the number of people needed to run the discussion.',
                     fontsize=9.2, weight='bold')
            save(fig, 'candy_jar_panel_accuracy' if candy else 'all_tasks_panel_accuracy', pdf)

        fig = page('Where did each candy-jar estimate move?',
                   'Each arrow follows the average estimate of one observed group, using exactly the same respondents before and after revision.\n'
                   'The vertical line marks the true count. Movement toward zero improves the full-group estimate; tighter agreement alone is insufficient.', 3)
        candy = moments[moments.task.str.startswith('3')]
        tasks = sorted(candy.task.unique())
        low, high = -65, 100
        for ni, network in enumerate(NETWORKS):
            ax = fig.add_axes([.09 + .30 * ni, .25, .25, .46])
            cells = candy[candy.network == network]
            closer = 0
            for qi, task in enumerate(tasks):
                pair = cells[cells.task == task].sort_values('round')
                initial, revised = (100 * (pair['mean'].to_numpy() / pair.truth.to_numpy() - 1))
                closer += abs(revised) < abs(initial)
                ax.annotate('', xy=(revised, qi), xytext=(initial, qi),
                            arrowprops=dict(arrowstyle='->', color=COLORS[ni], lw=2, shrinkA=0, shrinkB=4))
                ax.scatter(initial, qi, facecolor='white', edgecolor=BEFORE, s=60, linewidth=1.5, zorder=4)
                ax.scatter(revised, qi, color=COLORS[ni], s=50, zorder=5)
            ax.axvline(0, color=INK, lw=1.2, ls=':')
            ax.set(xlim=(low, high), ylim=(5.6, -.6), yticks=range(6), xticks=[-50, 0, 50, 100],
                   xlabel='Group mean error (% of true count)')
            ax.set_yticklabels([f'Jar {i + 1}\n{int(candy[candy.task == t].truth.iloc[0])} candies' for i, t in enumerate(tasks)] if ni == 0 else [])
            ax.set_title(TITLES[ni], fontsize=11, weight='bold', loc='left', pad=18)
            ax.grid(axis='x', color=GRID, lw=.7)
            ax.tick_params(axis='y', length=0, pad=9)
            fig.text(.09 + .30 * ni, .183, f'{closer} of 6 group means moved closer', fontsize=11.5, color=COLORS[ni], weight='bold')
        fig.legend(handles=[Line2D([], [], marker='o', markerfacecolor='white', markeredgecolor=BEFORE, ls='', label='Initial private mean'),
                            Line2D([], [], marker='o', color=MUTED, ls='', label='Final mean')],
                   loc='center', bbox_to_anchor=(.50, .775), ncol=2, frameon=False, fontsize=10)
        fig.text(.065, .11, 'Each row is a different jar; these are 18 separate group-by-jar observations, not a single group changing conditions.\n'
                 'No outlier trimming. One private-control initial mean is elevated by extreme responses; the revision control is therefore essential.',
                 fontsize=9.2, color=MUTED, linespacing=1.65)
        fig.text(.065, .065, 'The study used on-screen numerical feedback. It did not test shouting, and the candy brand is not identified here.', fontsize=9.2, color=MUTED)
        save(fig, 'candy_jar_mean_movement', pdf)

        fig = page('Agreement and accuracy are different outcomes',
                   'One point per observed group, averaging its tasks. Both axes compare the same respondents before and after revision.\n'
                   'Left means estimates became more similar. Down means the full-group mean became more accurate.', 4)
        diag = diagnostics.pivot(index=['network', 'group'], columns='round', values=['squared_group_error', 'standardized_sd'])
        dx = diag['standardized_sd'][3] - diag['standardized_sd'][1]
        dy = diag['squared_group_error'][3] - diag['squared_group_error'][1]
        limit_x = max(.12, max(abs(dx)) * 1.17)
        limit_y = max(.12, max(abs(dy)) * 1.17)
        for ni, network in enumerate(NETWORKS):
            ax = fig.add_axes([.075 + .30 * ni, .285, .255, .435])
            x, y = dx.loc[network], dy.loc[network]
            ax.axhspan(0, limit_y, color='#FCF3EF', zorder=0)
            ax.axhspan(-limit_y, 0, color='#EDF6F3', zorder=0)
            ax.axhline(0, color=INK, lw=1)
            ax.axvline(0, color=INK, lw=1)
            ax.scatter(x, y, color=COLORS[ni], s=64, edgecolors='white', alpha=.9, zorder=3)
            ax.set(xlim=(-limit_x, limit_x), ylim=(-limit_y, limit_y),
                   xlabel='Change in response spread\n(standardized SD; final - initial)')
            ax.set_title(TITLES[ni], fontsize=11, weight='bold', loc='left', pad=18)
            if ni == 0:
                ax.set_ylabel('Change in full-group squared error\n(final - initial; below zero is better)')
            else:
                ax.tick_params(labelleft=False)
            n_tighter = int((x < 0).sum())
            n_tighter_worse = int(((x < 0) & (y > 0)).sum())
            fig.text(.075 + .30 * ni, .192, f'{n_tighter} of {len(x)} groups converged', fontsize=11.5, weight='bold', color=COLORS[ni])
            fig.text(.075 + .30 * ni, .15, f'{n_tighter_worse} converged while error increased', fontsize=10, color=MUTED)
        fig.text(.065, .087, 'All 34 observed groups are shown. Group membership, task mix and the released initial-response scale stay fixed between rounds.\n'
                 'Descriptive paired changes; the score is squared error, so this does not reproduce the paper\'s reported absolute-error effect sizes.',
                 fontsize=9.2, color=MUTED, linespacing=1.65)
        save(fig, 'agreement_vs_accuracy', pdf)
    print(PDF)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SOURCE)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if not args.plot_only:
        result = analyze(args.input)
        print(json.dumps(result['equivalences'], indent=2))
    plot_all()
