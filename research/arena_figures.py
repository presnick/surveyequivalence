"""Paper-style figure using saved real MT-Bench estimates, without new inference."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(source=None, output=None):
    source = Path(source) if source is not None else ROOT / 'benchmarks/results/arena-validation'
    output = Path(output) if output is not None else ROOT / 'benchmarks/results/paper-audit'
    output.mkdir(parents=True, exist_ok=True)
    filenames = ['mt_bayesian_power.csv', 'mt_classifier_scores.csv',
                 'mt_equivalence_summary.json', 'mt_data_audit.json']
    curve = pd.read_csv(source / filenames[0], index_col=0, float_precision='round_trip')
    curve.columns = curve.columns.astype(int)
    scores = pd.read_csv(source / filenames[1], index_col=0, float_precision='round_trip').iloc[:, 0]
    summary = json.loads((source / filenames[2]).read_text())['Bayesian']
    audit = json.loads((source / filenames[3]).read_text())
    assert curve.index.equals(scores.index) and len(curve) > 2
    assert summary['actual']['status'] == 'crossing'
    eq = summary['actual']['estimate']
    eq_interval = summary['interval']
    eq_low, eq_high = eq_interval if eq_interval is not None else (np.nan, np.nan)
    interval_text = (f'95% interval: {eq_low:.2f}–{eq_high:.2f}' if eq_interval is not None
                     else '95% interval extends beyond supported crossings')
    censored = summary['bootstrap_status_counts'].get('above_range', 0)
    lo, hi = np.quantile(curve.iloc[1:], [.025, .975], axis=0)
    ai_lo, ai_hi = np.quantile(scores.iloc[1:], [.025, .975])
    delta = scores.iloc[1:] - curve[1].iloc[1:]
    delta_interval = np.quantile(delta, [.025, .975])
    blue, orange, gray = '#176782', '#b45724', '#626a73'
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6), gridspec_kw={'width_ratios':[1.45, 1]})
    fig.subplots_adjust(left=.075, right=.96, bottom=.29, top=.78, wspace=.27)
    fig.suptitle('Human-equivalent judging: show the crossing and its uncertainty',
                 x=.06, y=.97, ha='left', fontsize=18, weight='bold')
    fig.text(.06, .905, 'Observed MT-Bench results · GPT-4 as a preference judge · Agreement score',
             fontsize=12, color=gray)
    ax = axes[0]
    x = curve.columns.to_numpy()
    ax.fill_between(x, lo*100, hi*100, alpha=.14, color=blue)
    ax.plot(x, curve.iloc[0]*100, 'o-', lw=2.2, color=blue, label='Human benchmark (ABC)')
    ax.axhspan(ai_lo*100, ai_hi*100, color=orange, alpha=.10)
    ax.axhline(scores.iloc[0]*100, linestyle='--', color=orange, lw=1.8, label='GPT-4 judge')
    ax.plot([eq, eq], [32, scores.iloc[0]*100], linestyle=':', color=orange)
    ax.scatter([eq], [scores.iloc[0]*100], color=orange, zorder=4, s=55)
    ax.text(eq+.045, 36, f'Crossing: {eq:.2f}', color=orange, fontsize=11)
    ax.set(xlabel='Human judgments used by the benchmark predictor',
           ylabel='Agreement with an individual held-out human (%)',
           xlim=(-.05, 2.05), ylim=(32, 79), xticks=[0, 1, 2])
    ax.set_title('A. The power curve', loc='left', weight='bold', pad=13)
    ax.grid(alpha=.16)
    ax.legend(loc='upper left', fontsize=10, frameon=False)
    ax = axes[1]
    ax.axvspan(2, 2.35, color='#eff1f3')
    ax.axvline(1, color=gray, linestyle=':', lw=1.2)
    ax.axvline(2, color=gray, linestyle='--', lw=1)
    ax.plot([eq_low, eq_high], [.58, .58], color=orange, lw=3)
    ax.plot([eq_low, eq_high], [.58, .58], '|', color=orange, markersize=12)
    ax.scatter([eq], [.58], color=orange, s=65, zorder=4)
    ax.text(eq, .76, f'GPT-4: {eq:.2f} humans', ha='center', color=orange, weight='bold', fontsize=13)
    ax.text(eq, .40, interval_text, ha='center', fontsize=11)
    ax.text(1, .07, 'One-human benchmark', ha='center', fontsize=9, color=gray)
    ax.text(2.18, .84, 'Beyond\nobserved\ncoverage', ha='center', fontsize=9, color=gray)
    ax.set(xlabel='Rater equivalence (higher means a stronger judge)',
           xlim=(0, 2.35), ylim=(0, 1.1), xticks=[0, 1, 2], yticks=[])
    ax.spines['left'].set_visible(False)
    ax.set_title('B. Leaderboard entry', loc='left', weight='bold', pad=13)
    interpretation = (f'GPT-4’s point estimate matches a {eq:.2f}-human benchmark. '
                      + ('The interval spans one human.' if eq_interval is not None and eq_low <= 1 <= eq_high
                         else 'Uncertainty is shown above.'))
    fig.text(.06, .19, interpretation,
             fontsize=11, weight='bold')
    fig.text(.06, .14, f'{audit["selected_items"]} exact items; {audit["unique_question_clusters"]} question clusters; '
             f'{audit["selected_humans_per_item"]} selected external experts per item. {len(curve)-1} paired cluster bootstraps.\n'
             'Shading in A shows pointwise score intervals; B uses paired equivalence replicates. '
             f'{censored} replicate(s) exceed the two-human range and are retained.', fontsize=9.5, linespacing=1.5)
    fig.text(.06, .045, 'Exploratory: independence of released human votes from GPT-4 feedback is unverified. '
             'Intervals condition on selected votes and the fitted combiner.\n'
             'Only one AI judge has this repeated-human comparison. The 16-judge Arena release has one human vote per battle; '
             'positive human-panel equivalence is not identified.', fontsize=9.2, color=gray, linespacing=1.5)
    for extension in ['png', 'svg']:
        fig.savefig(output / f'rater_equivalence_paper_view.{extension}', dpi=180)
    plt.close(fig)
    record = dict(input_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in filenames},
                  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  actual_rater_equivalence=eq, percentile_interval=eq_interval,
                  bootstrap_count=len(curve)-1, censored_count=censored,
                  actual_gpt4_minus_one_human=float(scores.iloc[0]-curve[1].iloc[0]),
                  paired_difference_percentile_interval=delta_interval.tolist(),
                  note='Replots frozen scores; no runtime fixes, new labels, calibration, or inference. Bootstrap-only quantiles.')
    (output/'figure_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    plot_arena(source, output)


def plot_arena(source, output):
    """Show the paper's information-gain scale and the identified zero boundary."""
    audit_path = output/'arena_zero_boundary.json'
    audit = json.loads(audit_path.read_text())
    hard = {r['judge']:r for r in json.loads((source/'arena_results.json').read_text())}
    rows = sorted(audit['judges'], key=lambda r:r['information_gain_vs_loo_prior_bits'])
    blue, orange = '#176782', '#b45724'
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True, gridspec_kw={'width_ratios':[1.35,1]})
    fig.subplots_adjust(left=.33, right=.96, top=.82, bottom=.22, wspace=.22)
    for y, row in enumerate(rows):
        low, high = row['paired_gain_interval']
        axes[0].plot([low,high],[y,y],color=blue,lw=2)
        axes[0].scatter(row['information_gain_vs_loo_prior_bits'],y,color=blue,s=28)
        score = hard[row['judge']]
        axes[1].plot([score['agreement_low']*100,score['agreement_high']*100],[y,y],color=orange,lw=2)
        axes[1].scatter(score['agreement']*100,y,color=orange,s=28)
    axes[0].set_yticks(range(len(rows)),[r['judge'] for r in rows],fontsize=10)
    axes[0].axvline(0,color='#626a73',linestyle='--',lw=1)
    axes[1].axvline(audit['zero_human_agreement']*100,color='#626a73',linestyle='--',lw=1)
    axes[0].set_title('Probability forecasts: below the prior',loc='left',weight='bold',pad=15)
    axes[1].set_title('Hard decisions: above the prior',loc='left',weight='bold',pad=15)
    axes[0].set_xlabel('Log-score gain over leave-one-out prior (bits)')
    axes[1].set_xlabel('Three-way human agreement (%)')
    axes[0].set_xlim(right=.1)
    for ax in axes:
        ax.grid(axis='x',alpha=.16)
    fig.suptitle('Arena judges: the scoring rule changes the conclusion',x=.035,y=.975,ha='left',fontsize=18,weight='bold')
    fig.text(.035,.92,f'Under the paper’s definition, {sum(r["empirical_equivalence"] == 0 for r in rows)} of {len(rows)} released forecasts have empirical log-score equivalence 0.',fontsize=12)
    fig.text(.035,.13,f'{audit["items"]:,} exact battles · 95% intervals from {rows[0]["bootstrap_total"]} paired prompt-cluster bootstraps · Dashed lines: zero-human prior',fontsize=11,weight='bold')
    fig.text(.035,.075,'Left: uncalibrated softmax probabilities, clipped at .02 and renormalized as in the existing experiment. '
             'Paired intervals compare each forecast with the same prior.\n'
             'Right: released strict-winner/tie decisions. Positive human-panel equivalence cannot be measured with one human vote per battle.',fontsize=10,linespacing=1.5)
    fig.text(.035,.025,'This ranks preference judges, not the answer-generating chatbots. Zero log-score equivalence does not mean zero agreement or no useful information.',fontsize=10,color='#626a73')
    for extension in ['png','svg']:
        fig.savefig(output/f'arena_paper_scorecard.{extension}',dpi=180)
    plt.close(fig)
    (output/'arena_figure_provenance.json').write_text(json.dumps({
        'input_sha256':{str(audit_path):hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                        'arena_results.json':hashlib.sha256((source/'arena_results.json').read_bytes()).hexdigest()},
        'note':'Replots saved estimates with paired gain intervals; no runtime changes.'},indent=2)+'\n')


if __name__ == '__main__':
    main()
