"""Report independent scorer counterexamples as JSON without changing runtime code.

Run from the repository root with:
    .venv/bin/python benchmarks/scorer_math_audit.py
Confirmed implementation defects produce a nonzero exit status; use
--report-only to request a successful reporting command instead.
The default also evaluates the original scoring source at revision a4bbe2e.
"""

import argparse
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
import json
from math import isclose, log2
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from surveyequivalence import scoring_functions as current
from surveyequivalence.combiners import (
    DiscretePrediction, DiscreteDistributionPrediction, NumericPrediction,
)


def hard(labels):
    return [DiscretePrediction(label) for label in labels]


def soft(vectors, labels=('a', 'b')):
    return [DiscreteDistributionPrediction(list(labels), vector) for vector in vectors]


def panel_probability(labels, predicted, size, replacement):
    panels = (list(product(range(len(labels)), repeat=size)) if replacement
              else list(combinations(range(len(labels)), size)))
    result = Fraction(0)
    for panel in panels:
        counts = Counter(labels[i] for i in panel)
        winners = [label for label, count in counts.items() if count == max(counts.values())]
        if predicted in winners:
            result += Fraction(1, len(panels) * len(winners))
    return float(result)


def exact_execution_of_multiclass_sampler(module):
    # Repeat every ordered draw six times to exhaust uniform 1/2/3-way ties.
    labels = ['a', 'a', 'b', 'c']
    scheduled = []
    for indices in product(range(4), repeat=3):
        draw = [labels[index] for index in indices]
        counts = Counter(draw)
        winners = [label for label, count in counts.items() if count == max(counts.values())]
        scheduled.extend((draw, winners[i % len(winners)]) for i in range(6))
    schedule = iter(scheduled)
    pending_winner = None
    replacement_flags = []

    def choice(values, size=None, replace=True):
        nonlocal pending_winner
        if size == 3:
            draw, pending_winner = next(schedule)
            replacement_flags.append(bool(replace))
            return np.array(draw)
        if size == 1:
            if pending_winner not in values:
                raise AssertionError('Sampler mode disagrees with independently counted winners')
            return np.array([pending_winner])
        raise AssertionError(f'Unexpected choice size: {size}')

    with patch.object(module.np.random, 'choice', side_effect=choice):
        actual = module.AgreementScore(
            num_virtual_raters=len(scheduled), num_ref_raters_per_virtual_rater=3
        ).expected_score_anonymous_raters(hard(['a']), pd.DataFrame([labels]))
    return actual, sorted(set(replacement_flags))


def capture(function):
    try:
        return function()
    except Exception as exc:
        return {'exception': type(exc).__name__, 'message': str(exc)}


def audit(module):
    cases = []

    def record(name, expected, actual, explanation, lines, category='implementation_bug', severity='high'):
        matches = isinstance(actual, (int, float, np.number)) and isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        cases.append(dict(name=name, category=category, severity=severity,
                          expected=expected, actual=actual, matches_reference=matches,
                          explanation=explanation, current_source_lines=lines))

    predictions = soft([[.1, .9], [.2, .8], [.8, .2], [.9, .1]], ['neg', 'pos'])
    record('auc_perfect_binary_ranking', 1.0,
           capture(lambda: module.AUCScore.score(predictions, ['pos', 'pos', 'neg', 'neg'])),
           'All four positive-negative pairs rank correctly using P(pos). Maximum-class confidence loses direction.',
           [950])

    same = soft([[.8, .2], [.2, .8]])
    reordered = [same[0], soft([[.8, .2]], ['b', 'a'])[0]]
    for name, function in [
            ('soft_dmi_static_reordered_labels', lambda: module.DMIScore_for_Soft_Classifier.score(reordered, ['a', 'b'])),
            ('soft_dmi_anonymous_reordered_labels', lambda: module.DMIScore_for_Soft_Classifier().expected_score_anonymous_raters(
                reordered, pd.DataFrame([['a'], ['b']]))),
    ]:
        record(name, .15, capture(function),
               'The label-aligned joint matrix is [[.4,.1],[.1,.4]], with determinant .16-.01=.15; per-item vocabulary ordering changes no probabilities.',
               [1186, 1230])

    matrix = pd.DataFrame([['a'] * 14 + ['b'] * 7])
    module.frac_cache.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        agreement = capture(lambda: module.AgreementScore().expected_score_anonymous_raters(hard(['a']), matrix))
        after_agreement = capture(lambda: module.CrossEntropyScore().expected_score_anonymous_raters(soft([[.8, .2]]), matrix))
        module.frac_cache.clear()
        fresh_entropy = capture(lambda: module.CrossEntropyScore().expected_score_anonymous_raters(soft([[.8, .2]]), matrix))
    expected_entropy = 2 / 3 * log2(.8) + 1 / 3 * log2(.2)
    record('agreement_int64_factorial_overflow_21_raters', 2 / 3, agreement,
           'A single reference draw matches 14 of 21 raters. Pandas NumPy-integer counts enter the factorial cache and overflow at 21!.',
           [47, 55, 65, 468, 500], severity='critical')
    record('cross_entropy_after_agreement_cache_contamination', expected_entropy, after_agreement,
           'The same valid cross-entropy call changes with previous Agreement calls because the shared factorial cache contains overflowed integers.',
           [45, 55, 65, 636], severity='critical')
    record('cross_entropy_fresh_cache_control', expected_entropy, fresh_entropy,
           'A control: Python-integer factorials avoid the NumPy-integer overflow in this small example.',
           [636], category='control', severity='none')

    matrix = pd.DataFrame([['a', 'b'], ['a', 'b']])
    for scorer, predictions, expected in [
            (module.DMIScore_for_Hard_Classifier, hard(['a', 'b']), .125),
            (module.DMIScore_for_Soft_Classifier, same, .075),
    ]:
        outcomes = [scorer.score(predictions, labels) for labels in product(['a', 'b'], repeat=2)]
        record(scorer.__name__ + '_expected_score_contract', expected,
               capture(lambda: scorer().expected_score_anonymous_raters(predictions, matrix)),
               f'The four equiprobable complete reference vectors give scores {outcomes}; their arithmetic mean is {sum(outcomes) / 4}. Current code instead returns abs(det(E[joint])).',
               [1042, 1043, 1189, 1191], category='estimator_contract')

    predictions, labels = hard(['pos', 'neg']), ['pos', 'pos']
    record('precision_anonymous_uses_different_averaging', .5,
           capture(lambda: module.PrecisionScore().expected_score_anonymous_raters(predictions, pd.DataFrame({0: labels}))),
           'With one deterministic reference column, anonymous expected_score should equal score. Direct/default micro precision is 1/2; anonymous precision conditions only on the positive prediction and returns 1.',
           [754, 762, 770, 806], category='estimator_contract')

    module.frac_cache.clear()
    sampled = capture(lambda: exact_execution_of_multiclass_sampler(module))
    actual = sampled[0] if isinstance(sampled, tuple) else sampled
    flags = sampled[1] if isinstance(sampled, tuple) else None
    record('multiclass_panel_replacement_policy', panel_probability(['a', 'a', 'b', 'c'], 'a', 3, False), actual,
           f'Uniform three-rater subsets give P(a)=2/3. Exhausting the implemented with-replacement draws and ties gives 9/16. Actual sampler replacement flags: {flags}. Binary closed forms instead use without-replacement subsets.',
           [161, 478, 479, 500, 614, 615], category='panel_policy_contract')

    record('correlation_anonymous_mean_unimplemented', 1.0,
           capture(lambda: module.Correlation().expected_score_anonymous_raters(
               [NumericPrediction(1), NumericPrediction(2)], pd.DataFrame([[1], [2]]))),
           'The sole possible numeric reference vector equals the predictions, so Pearson correlation is 1. The default numeric mean combiner is not implemented by the anonymous sampling path.',
           [150, 165, 342, 367], category='unsupported_declared_capability', severity='medium')
    return cases


def load_revision(revision):
    source = subprocess.check_output(
        ['git', 'show', f'{revision}:surveyequivalence/scoring_functions.py'], cwd=ROOT, text=True)
    module = types.ModuleType('surveyequivalence._math_audit_historical_scoring')
    module.__package__ = 'surveyequivalence'
    exec(compile(source, f'{revision}:surveyequivalence/scoring_functions.py', 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='a4bbe2e', help='Git revision of the original scoring module')
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--output', type=Path, help='Also write the JSON report to this path')
    parser.add_argument('--report-only', action='store_true', help='Report known failures without a nonzero exit status')
    args = parser.parse_args()
    report = {
        'current_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'numpy_version': np.__version__, 'pandas_version': pd.__version__,
        'current_cases': audit(current),
    }
    if not args.skip_baseline:
        report['baseline_revision'] = args.baseline
        report['baseline_scope'] = 'Historical scoring module loaded from git, using current compatible Prediction value classes.'
        report['baseline_cases'] = audit(load_revision(args.baseline))
    serialized = json.dumps(report, indent=2, allow_nan=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    unresolved = [case for case in report['current_cases']
                  if case['category'] != 'control' and not case['matches_reference']]
    return 1 if unresolved and not args.report_only else 0


if __name__ == '__main__':
    raise SystemExit(main())
