"""Additional real-response plots: Lorenz (2011) and Kao (2018).

Uses the verified respondent releases, not the papers' simulated trajectories.
--plot-only rebuilds four PNG/SVG figures and the PDF from retained results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .social_influence_equivalence import panel_mse, first_integer, quantile_with_censoring
    from .panel_power import first_equivalent_size, equivalence_label
except ImportError:
    from social_influence_equivalence import panel_mse, first_integer, quantile_with_censoring
    from panel_power import first_equivalent_size, equivalence_label

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/social-influence-data'
OUT = ROOT / 'benchmarks/results/social-influence-additional'
PDF = ROOT / 'output/pdf/social_influence_additional_datasets.pdf'
BOOTSTRAPS, SEED = 5000, 28092027
LFILE, KFILE = 'task-lorenz2011.csv', 'kao-rsif20180130_si_005.csv'
SOURCES = {
    LFILE: dict(sha256='76158c03002be8a8f2fcc3ff9a40632c6ee6fef04e1bde1666cdfa5b917acd13',
                url='https://raw.githubusercontent.com/amaatouq/task-dependence/master/data/prior_research_data/lorenz2011.csv',
                paper='https://doi.org/10.1073/pnas.1008636108'),
    KFILE: dict(sha256='8a4cf280ba235939f4dd7fd61420486ba35532b05b64292011a3b0182909dd8d',
                url='https://ndownloader.figshare.com/files/11094635',
                paper='https://doi.org/10.1098/rsif.2018.0130')}
CONDITIONS = ['Solo', 'Decentralized_ aggregated', 'Decentralized_ full_info']
TITLES = ['Later: no feedback', 'Later: see the group average', "Later: see everyone's estimates"]
COLORS = ['#657386', '#138575', '#B66735']
INK, MUTED, GRID, BEFORE = '#202C3B', '#586879', '#DEE5EB', '#34495E'


def checked_csv(name):
    path = CACHE / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCES[name]['sha256']:
        raise ValueError(f'{name} differs from the verified released file')
    return pd.read_csv(path)


def log_errors(estimates, truth):
    estimates = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if (not np.isfinite(estimates).all() or not np.isfinite(truth).all()
            or np.any(estimates <= 0) or np.any(truth <= 0)):
        raise ValueError('Log scoring requires positive finite estimates and truths')
    return np.log(estimates / truth)


def serial_bound(value, kmax):
    return float(value) if np.isfinite(value) else f'>{kmax}'


def summarize(study, condition, ks, mean, boot):
    """Paired reference recalculated per bootstrap; preserve unresolved bounds."""
    target = mean[0, 9]
    boot_target = boot[:, 0, 9]
    eq = first_equivalent_size(ks, -mean[1], -target)
    eq_integer = first_integer(mean[1], target)[0]
    boot_integer = first_integer(boot[:, 1], boot_target)
    boot_fractional = np.array([first_equivalent_size(ks, -b[1], -t) for b, t in zip(boot, boot_target)])
    np.testing.assert_array_equal(first_integer(boot[:, 0], boot_target), 10)
    assert first_equivalent_size(ks, -mean[0], -target) == 10
    difference = boot[:, 1, 9] - boot_target
    result = dict(study=study, condition=condition, max_k=int(ks[-1]), reference_k=10,
                  reference_mse=float(target), final_mse_k10=float(mean[1, 9]),
                  final_minus_initial_mse_k10=float(mean[1, 9] - target),
                  difference_95_low=float(np.quantile(difference, .025)),
                  difference_95_high=float(np.quantile(difference, .975)),
                  equivalent_k=serial_bound(eq, ks[-1]),
                  equivalent_at_lower_boundary=bool(eq == ks[0]),
                  equivalent_integer_k=serial_bound(eq_integer, ks[-1]),
                  equivalent_integer_95_low=serial_bound(quantile_with_censoring(boot_integer, .025), ks[-1]),
                  equivalent_integer_95_high=serial_bound(quantile_with_censoring(boot_integer, .975), ks[-1]),
                  bootstrap_fraction_above_range=float(np.isinf(boot_integer).mean()))
    lower, upper = np.quantile(boot, [.025, .975], axis=0)
    rows = [dict(study=study, condition=condition, phase=phase, k=int(k),
                 expected_mse=float(mean[r, j]), score=float(-mean[r, j]),
                 mse_95_low=float(lower[r, j]), mse_95_high=float(upper[r, j]))
            for r, phase in enumerate(['initial', 'final']) for j, k in enumerate(ks)]
    saved = dict(reference_mse=boot_target, equivalent_integer=boot_integer,
                 equivalent_fractional=boot_fractional, mse_difference_k10=difference)
    return result, rows, saved


def check_pool(errors):
    """Independent enumeration of one/two reports and the full observed pool."""
    ii, jj = np.triu_indices(len(errors), 1)
    direct = [np.mean(errors ** 2), np.mean(((errors[ii] + errors[jj]) / 2) ** 2), errors.mean() ** 2]
    computed = panel_mse(errors, [1, 2, len(errors)])
    np.testing.assert_allclose(computed, direct, rtol=1e-12, atol=1e-12)
    return float(np.max(np.abs(computed - direct)))


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    lorenz, kao = checked_csv(LFILE), checked_csv(KFILE)
    assert lorenz.shape == (852, 12) and kao.shape == (602, 9)
    assert not lorenz.duplicated(['session', 'task', 'subject_id']).any()
    assert lorenz.groupby(['session', 'task']).network.nunique().max() == 1
    assert (lorenz.groupby(['session', 'task']).size() == 12).all()
    assert lorenz.groupby('task').truth.nunique().max() == 1
    assert not lorenz[['response_1', 'response_5']].isna().any().any()
    sessions = sorted(lorenz.session.unique())
    assert len(sessions) == 12
    lorenz.to_csv(OUT / 'lorenz_released_responses.csv', index=False)
    # This row ID refers only to the source-file order, not a supplied identity.
    kao_pairs = kao[['GUESS 1', 'GUESS 2', 'N', 'S']].copy()
    kao_pairs.insert(0, 'source_row', np.arange(1, len(kao_pairs) + 1))
    kao_pairs.to_csv(OUT / 'kao_paired_responses.csv', index=False)
    rng = np.random.default_rng(SEED)
    session_draws = rng.integers(0, len(sessions), size=(BOOTSTRAPS, len(sessions)))
    summary, curve_rows, moments, diagnostics, qa = [], [], [], [], []
    saved_draws = {'lorenz_sampled_sessions': np.asarray(sessions)[session_draws]}
    ks = np.arange(1, 13)
    for ci, condition in enumerate(CONDITIONS):
        session_curves = []
        for session in sessions:
            group = lorenz[(lorenz.session == session) & (lorenz.network == condition)]
            task_curves, task_diagnostics = [], []
            for task, cell in group.groupby('task'):
                truth = float(cell.truth.iloc[0])
                errors = log_errors(cell[['response_1', 'response_5']].to_numpy(), truth)
                task_curves.append(np.stack([panel_mse(errors[:, r], ks) for r in range(2)]))
                task_diagnostics.append([errors.mean(axis=0) ** 2, errors.std(axis=0, ddof=1)])
                for r, phase in enumerate(['initial', 'final']):
                    qa.append(check_pool(errors[:, r]))
                    moments.append(dict(session=session, condition=condition, task=int(task), phase=phase,
                                        n=len(cell), truth=truth, mean_log_error=float(errors[:, r].mean()),
                                        variance_log_error=float(errors[:, r].var(ddof=1))))
            session_curves.append(np.mean(task_curves, axis=0))
            diag = np.mean(task_diagnostics, axis=0)
            diagnostics.append(dict(session=session, condition=condition, tasks=len(task_curves),
                                    initial_group_loss=float(diag[0, 0]), final_group_loss=float(diag[0, 1]),
                                    initial_log_sd=float(diag[1, 0]), final_log_sd=float(diag[1, 1])))
        session_curves = np.asarray(session_curves)
        mean = session_curves.mean(axis=0)
        # All three conditions share session draws: the people recur across
        # conditions, but answer different questions under each condition.
        boot = session_curves[session_draws].mean(axis=1)
        result, rows, draws = summarize('Lorenz2011', condition, ks, mean, boot)
        summary.append(result)
        curve_rows.extend(rows)
        saved_draws.update({f'lorenz_{ci}_{key}': value for key, value in draws.items()})
    pd.DataFrame(moments).to_csv(OUT / 'lorenz_group_task_moments.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / 'lorenz_session_diagnostics.csv', index=False)
    ks = np.arange(1, 101)
    errors = kao[['GUESS 1', 'GUESS 2']].to_numpy(dtype=float) - 659
    assert np.isfinite(errors).all() and (kao[['GUESS 1', 'GUESS 2', 'S', 'N']] > 0).all().all()
    mean = np.stack([panel_mse(errors[:, r], ks) for r in range(2)])
    qa.extend(check_pool(errors[:, r]) for r in range(2))
    boot = np.empty((BOOTSTRAPS, 2, len(ks)))
    cue_groups = [np.flatnonzero(kao.N.to_numpy() == n) for n in sorted(kao.N.unique())]
    # Paired participant resampling preserves the observed mixture of claimed
    # cue sizes. N is not the number of actual observed peers or panel size k.
    for start in range(0, BOOTSTRAPS, 100):
        count = min(100, BOOTSTRAPS - start)
        indices = np.concatenate([rng.choice(group, size=(count, len(group))) for group in cue_groups], axis=1)
        sampled = errors[indices]
        boot[start:start + count] = sampled.mean(axis=1)[:, :, None] ** 2 + sampled.var(axis=1, ddof=1)[:, :, None] * (
            1 / ks[None, None, :] - 1 / len(kao))
    result, rows, draws = summarize('Kao2018', 'Displayed artificial social cue', ks, mean, boot)
    summary.append(result)
    curve_rows.extend(rows)
    saved_draws.update({f'kao_{key}': value for key, value in draws.items()})
    initial_error, final_error = np.abs(errors).T
    kao_outcomes = dict(closer=int((final_error < initial_error).sum()),
                        farther=int((final_error > initial_error).sum()),
                        same_error=int((final_error == initial_error).sum()),
                        unchanged_guess=int((errors[:, 0] == errors[:, 1]).sum()),
                        initial_mean=float(kao['GUESS 1'].mean()), final_mean=float(kao['GUESS 2'].mean()),
                        initial_median=float(kao['GUESS 1'].median()), final_median=float(kao['GUESS 2'].median()),
                        truth=659)
    assert kao_outcomes['unchanged_guess'] == 231
    pd.DataFrame(curve_rows).to_csv(OUT / 'panel_curves.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'equivalent_panels.csv', index=False)
    np.savez_compressed(OUT / 'bootstrap_draws.npz', **saved_draws)
    metadata = dict(sources=SOURCES, data_kind='Observed human responses only; no simulated estimates or model-generated social rounds',
                    seed=SEED, bootstrap_replicates=BOOTSTRAPS, reference='10 initial, pre-feedback reports in the SAME condition',
                    lorenz=dict(people=144, sessions=12, rows=852, group_task_cells=71,
                                comparison='Round 1 before feedback versus round 5 after four opportunities to revise; same 12 people per session-task.',
                                metric='Negative expected squared natural-log ratio of panel geometric mean to truth.',
                                weighting='Equal tasks within each session-condition, then equal sessions; control session 081121_0739 has only one released question.',
                                bootstrap='12 entire sessions resampled together across all conditions, retaining participants, tasks and both scored rounds.',
                                exclusions='No additional response exclusions. Only rounds 1 and 5 scored; both entirely positive. A zero in round 3 is not changed or used.',
                                release_caveat='2022 authors\' respondent file, not a reproduction of the 2011 paper\'s cleaned analysis. One 12-person control task is absent; original extreme entries remain.'),
                    kao=dict(people=602, jars=1, truth=659,
                             comparison='Actual private guess versus actual revision after a displayed computer-generated social cue; not mutual peer interaction.',
                             metric='Negative expected squared error of the arithmetic panel mean, in objects squared.',
                             bootstrap='Paired participant rows resampled within each claimed cue-size N stratum; observed stratum counts fixed. Conditional on this one jar and recruitment sample.',
                             outcomes=kao_outcomes,
                             cue_counts={str(k): int(v) for k, v in kao.N.value_counts().sort_index().items()}),
                    equivalences=summary,
                    checks=dict(one_two_full_pool_checks=len(qa), max_absolute_enumeration_error=max(qa),
                                bootstrap_self_equivalence='Initial k=10 verified in every draw in all four comparisons'),
                    limitations=['Panel size counts recorded reports; Lorenz social reports can include information from other original session members.',
                                 'Lorenz and Kao use different aggregators and score units. Do not compare their vertical scales or equivalences as a study ranking.',
                                 'Within-condition before/after curves are not causal between-condition contrasts.',
                                 'Bootstrap intervals are exploratory, not formal equivalence tests; unachieved crossings remain censored.',
                                 'The 2022 paper reanalyzes earlier experiments; the Lorenz file is obtained through its repository, not new 2022 participants.',
                                 'The Kao paper\'s fitted correction methods and simulated repeated interactions are not plotted.'])
    (OUT / 'results.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata


def plot_all():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
                         'axes.labelcolor': INK, 'axes.edgecolor': GRID, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    curves = pd.read_csv(OUT / 'panel_curves.csv')
    meta = json.loads((OUT / 'results.json').read_text())
    PDF.parent.mkdir(parents=True, exist_ok=True)

    def page(title, subtitle, number, source):
        fig = plt.figure(figsize=(14, 9))
        fig.text(.065, .945, title, fontsize=23, weight='bold', va='top')
        fig.text(.065, .875, subtitle, fontsize=11, color=MUTED, va='top', linespacing=1.55)
        fig.text(.94, .978, f'OBSERVED RESPONSES ONLY  /  {number:02d}', fontsize=9, color=MUTED, ha='right')
        fig.text(.065, .026, source, fontsize=8.5, color=MUTED)
        return fig

    def save(fig, stem, pdf):
        fig.savefig(OUT / f'{stem}.png', dpi=180)
        fig.savefig(OUT / f'{stem}.svg')
        pdf.savefig(fig)
        plt.close(fig)

    def label_bound(v):
        return f'{v:g}' if isinstance(v, (int, float)) else v

    lsource = 'Lorenz et al. (2011)  |  doi.org/10.1073/pnas.1008636108  |  Released data: Almaatouq et al. (2022) repository'
    ksource = 'Kao et al. (2018)  |  doi.org/10.1098/rsif.2018.0130  |  Data: figshare file 11094635  |  602 measured response pairs'
    with PdfPages(PDF, metadata={'Title': 'Additional independent and socially influenced estimate datasets',
                                'Author': 'SurveyEquivalence research'}) as pdf:
        fig = page('Lorenz: factual estimates before and after feedback',
                   'All dashed curves are private answers BEFORE seeing anyone else\'s answer to that question. Titles describe the LATER condition.\n'
                   'The same 144 people answered different questions under each condition; each session contained 12 people.', 1, lsource)
        c = curves[curves.study == 'Lorenz2011']
        floor, ceiling = float(c.score.min()) * 1.06, 0
        for i, condition in enumerate(CONDITIONS):
            ax = fig.add_axes([.075 + .30 * i, .33, .255, .405])
            entry = next(s for s in meta['equivalences'] if s['condition'] == condition)
            for phase, color, style in [('initial', BEFORE, '--'), ('final', COLORS[i], '-')]:
                line = c[(c.condition == condition) & (c.phase == phase)].sort_values('k')
                ax.plot(line.k, line.score, color=color, lw=2.4, ls=style,
                        label='Before feedback (round 1)' if phase == 'initial' else 'Final answers (round 5)')
            target = -entry['reference_mse']
            eq = entry['equivalent_k']
            eq_float = eq if isinstance(eq, (int, float)) else np.inf
            ax.axhline(target, color='#9A865A', lw=1.2, ls=':')
            ax.scatter(10, target, color=BEFORE, marker='s', s=36, zorder=5)
            if np.isfinite(eq_float):
                achieved = float(c[(c.condition == condition) & (c.phase == 'final') & (c.k == 1)].score.iloc[0]) if eq_float == 1 else target
                ax.vlines(eq_float, floor, achieved, color=COLORS[i], ls=':', lw=1.2)
                ax.scatter(eq_float, achieved, color=COLORS[i], edgecolors='white', s=64, zorder=6)
            ax.set(xlim=(.85, 12.15), ylim=(floor, ceiling), xticks=[1, 3, 6, 10, 12], xlabel='Recorded reports in panel (k)')
            ax.set_title(TITLES[i], loc='left', fontsize=11, weight='bold', pad=16)
            ax.grid(axis='y', color=GRID, lw=.7)
            if i == 0:
                ax.set_ylabel('Expected score (-squared log error)\nHigher is more accurate; zero is perfect')
            else:
                ax.tick_params(labelleft=False)
            ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='white', framealpha=1, fontsize=8)
            x = .075 + .30 * i
            eq_label = equivalence_label(eq_float, np.arange(1, 13))
            verb = 'reach the reference' if eq_float == 1 else 'match 10 initial'
            fig.text(x, .265, f'{eq_label} final reports {verb}', fontsize=11.8, color=COLORS[i], weight='bold')
            fig.text(x, .219, f'Whole reports: {label_bound(entry["equivalent_integer_k"])}  |  95% interval: '
                     f'{label_bound(entry["equivalent_integer_95_low"])} to {label_bound(entry["equivalent_integer_95_high"])}\n'
                     f'12 sessions; {23 if i == 0 else 24} recorded session-question groups', fontsize=9, color=MUTED, linespacing=1.65)
        fig.text(.065, .134, 'Panel estimate = geometric mean; loss = squared natural-log ratio to truth. This limits domination by extremely large numeric guesses.\n'
                 'Each condition has its own initial 10-report target. Intervals resample whole sessions; a boundary of <=1 does not identify a sub-person panel.',
                 fontsize=9.2, color=MUTED, linespacing=1.6)
        fig.text(.065, .077, 'The release retains extreme entries and lacks one control question-group. These are new calculations, not the paper\'s cleaned effect sizes.\n'
                 'Social k counts reports from existing 12-person sessions, not the number of participants needed to run an experiment.',
                 fontsize=9.2, color=MUTED, linespacing=1.6)
        save(fig, 'lorenz_panel_accuracy', pdf)

        fig = page('Lorenz: convergence does not ensure better accuracy',
                   'Each dot is one session, averaging its questions under the named condition. The same sessions appear in all three panels.\n'
                   'Left means less disagreement after revision. Down means a more accurate full-session geometric mean.', 2, lsource)
        diagnostics = pd.read_csv(OUT / 'lorenz_session_diagnostics.csv')
        diagnostics['dx'] = diagnostics.final_log_sd - diagnostics.initial_log_sd
        diagnostics['dy'] = diagnostics.final_group_loss - diagnostics.initial_group_loss
        limit_x = diagnostics.dx.abs().max() * 1.14
        limit_y = diagnostics.dy.abs().max() * 1.14
        for i, condition in enumerate(CONDITIONS):
            ax = fig.add_axes([.075 + .30 * i, .285, .255, .435])
            d = diagnostics[diagnostics.condition == condition]
            ax.axhspan(0, limit_y, color='#FCF3EF')
            ax.axhspan(-limit_y, 0, color='#EDF6F3')
            ax.axhline(0, color=INK, lw=1)
            ax.axvline(0, color=INK, lw=1)
            ax.scatter(d.dx, d.dy, color=COLORS[i], s=65, edgecolors='white', zorder=3)
            ax.set(xlim=(-limit_x, limit_x), ylim=(-limit_y, limit_y), xlabel='Change in log-response spread\n(final - initial)')
            ax.set_title(TITLES[i], loc='left', fontsize=11, weight='bold', pad=16)
            if i == 0:
                ax.set_ylabel('Change in squared log error\n(final - initial; below zero is better)')
            else:
                ax.tick_params(labelleft=False)
            n_converged = int((d.dx < 0).sum())
            n_worse = int(((d.dx < 0) & (d.dy > 0)).sum())
            fig.text(.075 + .30 * i, .193, f'{n_converged} of 12 sessions converged', fontsize=11.5, color=COLORS[i], weight='bold')
            fig.text(.075 + .30 * i, .15, f'{n_worse} converged while error increased', fontsize=10, color=MUTED)
        fig.text(.065, .09, 'Before = round 1, before feedback about that question. After = round 5. All 12 recorded answers enter each group estimate.\n'
                 'Observed session changes on the released rows; no simulated trajectories, fitted response model, or additional outlier exclusions.',
                 fontsize=9.2, color=MUTED, linespacing=1.65)
        save(fig, 'lorenz_agreement_vs_accuracy', pdf)

        fig = page('Kao: a jar estimate before and after a social cue',
                   '602 people estimated a jar containing 659 objects, then could revise after seeing a number presented as other people\'s average.\n'
                   'Their guesses are real. The displayed social number was computer-generated; participants were not interacting with each other.', 3, ksource)
        c = curves[curves.study == 'Kao2018']
        entry = next(s for s in meta['equivalences'] if s['study'] == 'Kao2018')
        target, eq = -entry['reference_mse'] / 1000, entry['equivalent_k']
        for i in range(2):
            ax = fig.add_axes([.09 + .45 * i, .315, .37, .42])
            for phase, color, style in [('initial', BEFORE, '--'), ('final', '#8A609C', '-')]:
                d = c[c.phase == phase].sort_values('k')
                ax.plot(d.k, d.score / 1000, color=color, ls=style, lw=2.5,
                        label='Private estimate, before cue' if phase == 'initial' else 'Revised estimate, after cue')
            ax.set(xlim=(1, 100) if i == 0 else (5, 30), ylim=(-625, 0) if i == 0 else (-140, 0),
                   xticks=[1, 20, 40, 60, 80, 100] if i == 0 else [5, 10, 15, 20, 25, 30],
                   xlabel='Recorded reports included in panel (k)',
                   ylabel='Expected score (-MSE; thousands of objects²)')
            ax.axhline(target, color='#9A865A', lw=1.2, ls=':')
            ax.scatter(10, target, color=BEFORE, marker='s', s=40, zorder=6)
            if isinstance(eq, (int, float)):
                ax.vlines(eq, ax.get_ylim()[0], target, color='#8A609C', ls=':', lw=1.2)
                ax.scatter(eq, target, color='#8A609C', edgecolors='white', s=68, zorder=6)
            ax.set_title('Full panel range' if i == 0 else 'Detail around the 10-report reference', fontsize=11, weight='bold', loc='left', pad=16)
            ax.grid(axis='y', color=GRID, lw=.7)
            ax.legend(loc='lower right', frameon=False, fontsize=9)
        fig.text(.09, .245, f'{label_bound(round(eq, 2))} post-cue reports match 10 private reports', fontsize=14, color='#8A609C', weight='bold')
        fig.text(.09, .20, f'Whole reports: {label_bound(entry["equivalent_integer_k"])}  |  Exploratory 95% interval: '
                 f'{label_bound(entry["equivalent_integer_95_low"])} to {label_bound(entry["equivalent_integer_95_high"])}', fontsize=10.5, color=MUTED)
        fig.text(.065, .127, 'Panel estimate = arithmetic mean; score = negative expected squared error. Higher is better. No correction model is fitted.\n'
                 'Paired bootstrap preserves the observed mix of claimed cue sizes; the interval is conditional on this jar and this respondent sample.',
                 fontsize=9.2, color=MUTED, linespacing=1.65)
        fig.text(.065, .073, 'The experiment\'s claimed social group size N is separate from plotted panel size k. These curves average actual reports, not fictitious peers.',
                 fontsize=9.2, color=MUTED)
        save(fig, 'kao_jar_panel_accuracy', pdf)

        fig = page('Kao: who moved closer to the true count?',
                   'Each point pairs one person\'s initial private guess with their actual revised guess. The true count is 659 objects.\n'
                   'Revision can help, hurt, or leave the answer unchanged; no simulated people or rounds are included.', 4, ksource)
        pairs = pd.read_csv(OUT / 'kao_paired_responses.csv')
        outcomes = meta['kao']['outcomes']
        ax = fig.add_axes([.095, .255, .405, .47])
        ax.plot([60, 12000], [60, 12000], color=MUTED, lw=1, ls=':', label='Unchanged answer')
        ax.axvline(659, color='#9A865A', lw=1.1, ls='--')
        ax.axhline(659, color='#9A865A', lw=1.1, ls='--')
        ax.scatter(pairs['GUESS 1'], pairs['GUESS 2'], color='#8A609C', s=19, alpha=.4, edgecolors='none')
        ax.set(xscale='log', yscale='log', xlim=(60, 12000), ylim=(60, 12000),
               xlabel='Initial private guess (logarithmic axis)', ylabel='Revised guess after cue (logarithmic axis)',
               xticks=[100, 300, 659, 2000, 10000], yticks=[100, 300, 659, 2000, 10000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        ax.text(.96, .025, 'Dashed lines: truth = 659', transform=ax.transAxes, ha='right', color='#927947', fontsize=9)
        ax = fig.add_axes([.63, .34, .28, .33])
        values = [outcomes['closer'], outcomes['farther'], outcomes['unchanged_guess']]
        ax.barh(range(3), values, color=['#138575', '#B66735', '#657386'], height=.58)
        for i, value in enumerate(values):
            ax.text(value + 5, i, f'{value}  ({100 * value / 602:.1f}%)', va='center', fontsize=10, weight='bold')
        ax.set(yticks=range(3), yticklabels=['Closer to truth', 'Farther from truth', 'Unchanged guess'],
               ylim=(2.65, -.65), xlim=(0, 310), xlabel='People (out of 602)')
        ax.set_title('Change in absolute count error', fontsize=12, weight='bold', loc='left', pad=18)
        ax.grid(axis='x', color=GRID, lw=.7)
        ax.set_axisbelow(True)
        fig.text(.63, .23, f'Mean guess: {outcomes["initial_mean"]:.0f} to {outcomes["final_mean"]:.0f}\n'
                 f'Median guess: {outcomes["initial_median"]:.0f} to {outcomes["final_median"]:.0f}',
                 fontsize=12, color=INK, linespacing=1.7)
        fig.text(.095, .152, 'All 602 pairs are plotted, including extreme guesses; overlapping dots can represent multiple people.\n'
                 'Counts use absolute error in objects. The log axes only make the full range of guesses visible.',
                 fontsize=9.5, color=MUTED, linespacing=1.65)
        fig.text(.065, .078, 'The cue was experimentally constructed relative to each person\'s first guess. This is an observed cue-response experiment, not a live group discussion.',
                 fontsize=9.2, color=MUTED)
        save(fig, 'kao_individual_revisions', pdf)
    print(PDF)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if not args.plot_only:
        results = analyze()
        print(json.dumps(results['equivalences'], indent=2))
    plot_all()
