"""Offline, ordinal-score demonstration on the released 200-item human subset.

Run: .venv/bin/python research/judges_verdict.py
This is a custom-score analysis, not the restricted Agreement/log-score paper_mode.
AI scores are two-prompt judge-system outputs, never interpreted as probabilities.
"""
from collections import Counter
from itertools import combinations, permutations
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
from surveyequivalence import AnalysisPipeline, AnonymousBayesianCombiner
from surveyequivalence.combiners import NumericPrediction
from surveyequivalence.scoring_functions import Scorer
from surveyequivalence._equivalence import supported_equivalence
from research.arena import cluster_samples, digest, equivalence_summary, write_json

CACHE = ROOT / '.cache/dataset-search/september21'
OUTPUT = ROOT / 'benchmarks/results/judges-verdict'
LABELS = ['0', '0.5', '1']
JUDGES = {'GPT-4o': ('verdict-gpt4o.json', 'gpt-4o'),
          'Claude Sonnet 4': ('verdict-claude.json', 'claude-sonnet-4'),
          'Gemini 2.0 Flash': ('verdict-gemini.json', 'gemini_gemini-2.0-flash')}
SEED, BOOTSTRAPS = 1729, 500


def scalar(prediction):
    if prediction is None:
        raise ValueError('Unsupported Bayesian prediction: do not drop this item')
    if hasattr(prediction, 'probabilities'):
        return sum(float(y) * p for y, p in zip(prediction.label_names, prediction.probabilities))
    return float(prediction.value)


class OrdinalScore(Scorer):
    """Expected 1 - squared error against a uniformly selected disjoint human."""
    def __init__(self):
        super().__init__(num_ref_raters_per_virtual_rater=1)

    @staticmethod
    def score(classifier_predictions, rater_labels):
        return sum(1 - (scalar(p) - float(y)) ** 2
                   for p, y in zip(classifier_predictions, rater_labels)) / len(rater_labels)

    def expected_score_anonymous_raters(self, classifier_predictions, W, **kwargs):
        values = W.to_numpy(dtype=float)
        if not np.isfinite(values).all() or len(classifier_predictions) != len(values):
            raise ValueError('This demonstration requires complete aligned ratings')
        terms = [sum(1 - (scalar(p) - float(y)) ** 2 for y in row) / len(row)
                 for p, row in zip(classifier_predictions, values)]
        return sum(terms) / len(terms)


def load():
    humans = json.loads((CACHE / 'verdict-train.json').read_text())
    key = lambda row: (row['question'], row['gen_answer'], row['gt_answer'])
    keys = [key(row) for row in humans]
    assert len(keys) == len(set(keys)) == 200
    assert all(len(row['annotations']) == 3 and
               len({a['annotator'] for a in row['annotations']}) == 3 for row in humans)
    ids = [hashlib.sha256(json.dumps(k).encode()).hexdigest() for k in keys]
    W = pd.DataFrame([[a['score'] for a in row['annotations']] for row in humans],
                     index=ids, columns=['rating_slot_1', 'rating_slot_2', 'rating_slot_3'])
    assert W.isin(LABELS).to_numpy().all()
    scores = {}
    for name, (filename, _) in JUDGES.items():
        rows = json.loads((CACHE / filename).read_text())
        indexed = {}
        for row in rows:
            k = (row['user_input'], row['response'], row['reference'])
            if k in indexed:
                raise ValueError('Duplicate AI item in ' + filename)
            indexed[k] = row['nv_accuracy']
        scores[name] = [indexed[k] for k in keys]
    scores = pd.DataFrame(scores, index=ids, dtype=float)
    coverage = {name: dict(matched_items=len(scores),
                 finite_scores=int(np.isfinite(scores[name]).sum()),
                 missing_item_ids=scores.index[~np.isfinite(scores[name])].tolist()) for name in scores}
    scores = scores.loc[:, np.isfinite(scores).all(axis=0)]
    assert np.isfinite(scores.to_numpy()).all()
    assert scores.isin([0, .25, .5, .75, 1]).to_numpy().all()
    groups = [hashlib.sha256(row['question'].encode()).hexdigest() for row in humans]
    metadata = pd.DataFrame({'item_id': ids, 'question_sha256': groups,
                             'source_dataset': [r['dataset_name'] for r in humans]})
    return W, scores, groups, metadata, coverage


def independent_check(W, scores, samples, run):
    """Enumerate ordered draws as integers, independently of falling factorial code."""
    rows = W.to_numpy()
    totals, contributions = {}, {}
    for length in (1, 2, 3):
        contributions[length] = [Counter(permutations(row, length)) for row in rows]
        totals[length] = sum(contributions[length], Counter())
    abc = AnonymousBayesianCombiner(W=W, extreme_cutoff=0)
    terms = np.empty((len(W), 3))
    largest_error, checked = 0., 0
    for i, row in enumerate(rows):
        for k in range(3):
            values = []
            for subset in combinations(range(3), k):
                observed = tuple(row[s] for s in subset)
                counts = [totals[k + 1][observed + (label,)] -
                          contributions[k + 1][i][observed + (label,)] for label in LABELS]
                assert sum(counts) > 0, 'Unsupported item/panel'
                posterior = np.array(counts, dtype=float) / sum(counts)
                prediction = abc.combine(LABELS, [(s, row[s]) for s in subset], item_id=i)
                assert prediction is not None
                error = float(np.max(np.abs(posterior - prediction.probabilities)))
                largest_error = max(largest_error, error)
                np.testing.assert_allclose(prediction.probabilities, posterior, atol=2e-13, rtol=0)
                mean = sum(float(label) * p for label, p in zip(LABELS, posterior))
                refs = [float(row[r]) for r in range(3) if r not in subset]
                values.append(sum(1 - (mean - y) ** 2 for y in refs) / len(refs))
                checked += 1
            terms[i, k] = sum(values) / len(values)
    expected_curves = np.array([terms[s].mean(axis=0) for s in samples])
    actual_curves = run.expert_power_curve.df.to_numpy()
    np.testing.assert_allclose(actual_curves, expected_curves, atol=2e-13, rtol=0)
    judge_terms = 1 - ((scores.to_numpy()[:, :, None] - rows.astype(float)[:, None, :]) ** 2).mean(axis=2)
    expected_scores = np.array([judge_terms[s].mean(axis=0) for s in samples])
    np.testing.assert_allclose(run.classifier_scores.df.to_numpy(), expected_scores, atol=2e-13, rtol=0)
    # Verify the orientation and treatment of partial credit on a hand-calculated case.
    assert OrdinalScore.score([NumericPrediction(.25)], ['0.5']) == .9375
    assert OrdinalScore.score([NumericPrediction(1)], ['0']) == 0
    assert OrdinalScore.score([NumericPrediction(.5)], ['0.5']) == 1
    return dict(checked_item_panels=checked, unsupported_predictions=0,
                maximum_probability_error=largest_error,
                maximum_curve_error=float(np.max(np.abs(actual_curves - expected_curves))),
                all_501_samples_checked=True, comparison_absolute_tolerance=2e-13,
                independent_method='integer enumeration of all ordered draws; subtract query item')


def figures(curve, scores, summaries, output):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'GPT-4o': '#0072B2', 'Claude Sonnet 4': '#D55E00', 'Gemini 2.0 Flash': '#009E73'}
    footer = ('200 TechQA items; 3 human ratings/item; released trial 1 (two-prompt AI systems).\n'
              '95% question-bootstrap intervals, conditional on fixed ratings and fitted combiner. Exploratory train subset.\n'
              'Custom ordinal-score equivalence; not log-score / information equivalence.')
    fig, ax = plt.subplots(figsize=(10, 6.8))
    lo, hi = np.quantile(curve.iloc[1:], [.025, .975], axis=0)
    ax.fill_between([0, 1, 2], lo, hi, color='#596273', alpha=.18)
    ax.plot([0, 1, 2], curve.iloc[0], '-o', color='#343C49', lw=2.4, label='Bayesian human panel')
    for name in scores:
        value = float(scores.loc[0, name]); eq = summaries[name]['equivalence']['actual']
        ax.axhline(value, color=colors[name], lw=1.8, linestyle='--' if name.startswith('Gemini') else '-', label=name)
        if eq['status'] == 'crossing':
            ax.plot(eq['estimate'], value, '*', color=colors[name], ms=13)
    ax.set(xlabel='Number of input human ratings (one or more others held out)',
           ylabel='Ordinal score: 1 − mean squared error (higher is better)',
           xticks=[0, 1, 2], xlim=(-.06, 2.06),
           title='How do AI judge systems compare with human panels?')
    ax.legend(loc='best', fontsize=10); ax.grid(alpha=.18)
    fig.text(.5, .025, footer, ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .125, 1, 1))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / ('human_panel_comparison.' + ext), dpi=180)
    plt.close(fig)

    order = sorted(scores, key=lambda n: scores.loc[0, n], reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={'width_ratios': [1, 1.1]})
    for i, name in enumerate(order):
        result = summaries[name]; eq = result['equivalence']; actual = eq['actual']
        lo, hi = result['score_interval']; point = result['score']
        axes[0].errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt='o', color=colors[name], capsize=4)
        axes[0].annotate(f'{point:.3f}', (point, i), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)
        if actual['status'] == 'crossing':
            value = actual['estimate']; ci = eq.get('interval')
            axes[1].plot(value, i, 'o', color=colors[name])
            if ci is not None:
                axes[1].plot(ci, [i, i], color=colors[name], lw=2)
            text = f'{value:.2f} humans' + ('' if ci else ' (CI unresolved)')
        else:
            axes[1].annotate('', xy=(2.2, i), xytext=(2, i), arrowprops={'arrowstyle': '->', 'color': colors[name], 'lw': 2})
            text = '>2; crossing not observed'
        axes[1].text(.02, i + .23, text, fontsize=10, color=colors[name])
    for ax in axes:
        ax.set(yticks=range(len(order)), ylim=(len(order)-.4, -.55))
        ax.grid(axis='x', alpha=.2)
    axes[0].set(yticklabels=order, xlabel='Ordinal score (1 − mean squared error)', title='Observed judge-system performance')
    axes[1].set(yticklabels=[], xlabel='Human-panel equivalent on this score', title='Supported equivalence', xticks=[0, 1, 2], xlim=(-.08, 2.3))
    axes[1].axvspan(2, 2.3, color='#eeeeee'); axes[1].axvline(2, color='#999999', linestyle=':', lw=1)
    fig.suptitle('Judge’s Verdict: judge-system pilot', fontsize=16, y=.99)
    fig.text(.5, .025, footer + '\nOrdering by point score does not establish a statistically reliable ranking.', ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .16, 1, .94))
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / ('judge_system_equivalence.' + ext), dpi=180)
    plt.close(fig)


def main():
    W, raw_scores, groups, metadata, coverage = load()
    samples = cluster_samples(groups, BOOTSTRAPS, SEED)
    assert len(samples) == BOOTSTRAPS + 1 and np.array_equal(samples[0], np.arange(len(W)))
    predictions = raw_scores.applymap(NumericPrediction)
    run = AnalysisPipeline(W, classifier_predictions=predictions,
        combiner=AnonymousBayesianCombiner(extreme_cutoff=0), scorer=OrdinalScore(),
        allowable_labels=LABELS, anonymous_raters=True, max_K=3, max_rater_subsets=200,
        item_samples=[W.index[s] for s in samples], random_state=SEED, procs=1, verbosity=0)
    audit = independent_check(W, raw_scores, samples, run)
    curve, scores = run.expert_power_curve.df, run.classifier_scores.df
    summaries = {}
    for name in scores:
        entries = [supported_equivalence(float(s), row.to_dict())
                   for s, (_, row) in zip(scores[name], curve.iterrows())]
        summaries[name] = dict(score=float(scores.loc[0, name]),
            score_interval=np.quantile(scores[name].iloc[1:], [.025, .975]).tolist(),
            equivalence=equivalence_summary(entries),
            paired_gain_vs_one_human_interval=np.quantile(scores[name].iloc[1:] - curve[1].iloc[1:], [.025, .975]).tolist(),
            paired_gain_vs_two_humans_interval=np.quantile(scores[name].iloc[1:] - curve[2].iloc[1:], [.025, .975]).tolist())
    differences = {f'{a} minus {b}': dict(
        difference=float(scores.loc[0, a] - scores.loc[0, b]),
        interval=np.quantile(scores[a].iloc[1:] - scores[b].iloc[1:], [.025, .975]).tolist())
        for a, b in combinations(scores.columns, 2)}
    provenance = []
    revision = '92764ec7e3f742e022b122b32a6739c85da0ed20'
    for name, (file, remote) in JUDGES.items():
        provenance.append(dict(judge=name, filename=file, sha256=digest(CACHE / file),
            url=f'https://huggingface.co/spaces/nvidia/judges-verdict/resolve/{revision}/benchmark/judge_results/{remote}/trial1.json'))
    provenance.append(dict(filename='verdict-train.json', sha256=digest(CACHE / 'verdict-train.json'),
        url='https://huggingface.co/datasets/nvidia/judges-verdict/resolve/9936c2fc75ac7a093d1a7c42f7fe187226701aff/annotations_train.json'))
    report = dict(items=len(W), question_clusters=len(set(groups)), human_ratings_per_item=3,
        datasets=dict(Counter(metadata.source_dataset)), seed=SEED, bootstraps=BOOTSTRAPS,
        human_curve=curve.iloc[0].to_dict(), judges=summaries, paired_judge_differences=differences,
        coverage=coverage, checks=audit, sources=provenance,
        script_sha256=digest(__file__), scoring='1 - squared error against individual human ordinal rating',
        human_predictor='Unclipped anonymous Bayesian posterior mean; leave query item out',
        protocol='Enumerate all human input subsets; average score against disjoint remaining humans',
        limitations=['Custom ordinal score; not the restricted Agreement/log-score paper_mode',
            'Public human-annotated train subset; exploratory, not an untouched test set',
            'Two complete AI systems plotted; Claude Sonnet 4 excluded for six missing scores',
            'Released trial 1 only; two-prompt average per system, no trial averaging',
            'Input-panel range 0..2; do not extrapolate beyond observed crossings',
            'Bootstrap conditional on fixed annotators, AI outputs, and fitted combiner',
            'Anonymous slots are not assumed to identify the same person across items',
            'Squared error treats ordinal steps as equally spaced; not full-distribution agreement',
            'Annotation independence and benchmark exposure cannot be established from files alone'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    W.to_csv(OUTPUT / 'human_ratings.csv', index_label='item_id')
    raw_scores.to_csv(OUTPUT / 'judge_ratings.csv', index_label='item_id')
    metadata.to_csv(OUTPUT / 'item_manifest.csv', index=False)
    curve.to_csv(OUTPUT / 'human_curve.csv', index_label='sample_id')
    scores.to_csv(OUTPUT / 'judge_scores.csv', index_label='sample_id')
    np.savez_compressed(OUTPUT / 'sample_schedule.npz', indices=np.concatenate(samples),
                        offsets=np.cumsum([0] + [len(s) for s in samples]))
    write_json(OUTPUT / 'results.json', report)
    figures(curve, scores, summaries, OUTPUT)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
