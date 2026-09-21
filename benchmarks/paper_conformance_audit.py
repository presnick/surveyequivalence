"""Read-only counterexamples against rater.pdf (arXiv:2106.01254v3).

This diagnostic deliberately exits 1 while strict paper conformity fails.
It does not regenerate parity fixtures or alter production behavior.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
from surveyequivalence import AnalysisPipeline, AnonymousBayesianCombiner, FrequencyCombiner, AgreementScore
from surveyequivalence.combiners import DiscreteDistributionPrediction, DiscretePrediction
from surveyequivalence.equivalence import PowerCurve
from surveyequivalence._equivalence import supported_equivalence


def pipeline(W, **kwargs):
    options = dict(combiner=FrequencyCombiner(), scorer=AgreementScore(),
                   allowable_labels=['a', 'b'], num_bootstrap_item_samples=0,
                   procs=1, random_state=0, verbosity=0, anonymous_raters=True)
    options.update(kwargs)
    return AnalysisPipeline(W, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/results/paper-audit/conformance.json')
    args = parser.parse_args()
    cases = []

    def record(name, expected, actual, paper, location, category='deviation', note=''):
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            matches = math.isclose(actual, expected, rel_tol=0, abs_tol=2e-15)
        elif isinstance(expected, list) and isinstance(actual, list):
            matches = bool(np.allclose(actual, expected, rtol=0, atol=2e-15))
        else:
            matches = actual == expected
        cases.append(dict(name=name, expected=expected, actual=actual, conforms=matches,
                          paper=paper, code=location, category=category, note=note))

    abc = AnonymousBayesianCombiner()
    actual, eligible = abc.probabilityOneItem(np.array([2, 3]), np.array(['D']*2 + ['C']*7), ['D', 'C'])
    record('paper_worked_ordered_sequence', float(Fraction(1, 36)), actual,
           'p45 worked example; Algorithm 1 p46', 'combiners.py:501', 'control',
           '420 matching ordered draws / 15120 distinct-position draws; eligible=%s' % eligible)

    W = np.array([['a', 'a', 'b'], ['a', 'b', 'b'], ['a', 'a', 'b']], dtype=object)
    prediction = abc.combine(['a', 'b'], [(0, 'a')], W=W, item_id=0)
    record('leave_one_out_interior_posterior', [1/3, 2/3], prediction.probabilities,
           'Algorithm 2 p46', 'combiners.py:405-423,466-479', 'control',
           'Remaining rows have prefix mass 1/2 and aa mass 1/6; ratio=1/3.')

    W = np.array([['a']*99 + ['b']]*3, dtype=object)
    actual = abc.combine(['a', 'b'], [], W=W, item_id=0).probabilities
    record('abc_posterior_clipped_even_with_full_support', [.99, .01], actual,
           'Algorithm 2 p46; Claims 24-25 p24', 'combiners.py:64-82,423',
           note='Repeating this row distribution indefinitely does not remove the fixed .02 distortion.')
    record('prediction_object_changes_valid_distribution', [.99, .01],
           DiscreteDistributionPrediction(['a', 'b'], [.99, .01]).probabilities,
           'Classifier definition p12', 'combiners.py:64-82',
           note='Explicit regularization is permissible as a different predictor, but is not the raw input.')
    record('frequency_combiner_is_regularized', [1., 0.],
           FrequencyCombiner().combine(['a', 'b'], [(0, 'a')]).probabilities,
           'Frequency combiner §5.1 p17', 'combiners.py:343')

    # A separate training set has no item to redact. The current pipeline passes
    # evaluation row positions as training row positions regardless of identity.
    train = pd.DataFrame([['a', 'a'], ['b', 'b']], index=['train-a', 'train-b'])
    bound = AnonymousBayesianCombiner(W=train)
    record('disjoint_training_matrix_wrong_holdout', [.5, .5],
           bound.combine(['a', 'b'], [], W=np.array([['a', 'b']]), item_id=0).probabilities,
           'Algorithm 2 redacts the CURRENT item, p46', 'equivalence.py:928-935,1009',
           note='No training item matches this evaluation item; positional holdout removes train-a anyway. This input mode extends the paper.')
    try:
        evaluation = pd.DataFrame([['a', 'b']]*3, index=['eval-x', 'eval-y', 'eval-z'])
        pipeline(evaluation, combiner=AnonymousBayesianCombiner(W=train), max_K=1)
        external_result = 'completed'
    except ValueError as exc:
        external_result = str(exc)
    record('pipeline_disjoint_training_position_out_of_bounds', 'completed', external_result,
           'Algorithm 2 p46: redact only the current training item', 'equivalence.py:928-935,1009',
           note='Three evaluation items, two separate training items: the third evaluation position is not a training identity.')

    curve = PowerCurve(df=pd.DataFrame([{0:.2, 1:.5, 2:.5, 3:.8}]))
    record('earliest_equal_crossing', 1., curve.compute_one_equivalence(.5),
           'Definitions 9/17 pp18/21', 'equivalence.py:328-338')
    record('empirical_range_is_a_bound', 'at least 3', curve.compute_one_equivalence(.9),
           'Remark 18 p21', 'equivalence.py:331',
           note='The numerical endpoint alone discards censoring; a bound needs a status in addition to its value.')
    result = supported_equivalence(.1, {0:.2, 1:.5})
    record('arena_helper_below_zero_baseline', dict(status='crossing', estimate=0., bound=None), result,
           'Definition 9: minimum x is zero, p18', 'benchmarks/arena_validation.py:146',
           note='The Arena reporting helper incorrectly treats this identifiable zero as lower censoring. Existing MT result is not in this case.')

    # Freeze one legitimate benchmark subset. Identical classifiers should have
    # identical scores when the evaluation labels are actually shared.
    W = pd.DataFrame([['a', 'b', 'b']]*2, columns=['r0', 'r1', 'r2'])
    predictions = pd.DataFrame({'AI': [DiscretePrediction('a')]*2})
    run = pipeline(W, classifier_predictions=predictions, min_k=1, max_K=2,
                   ratersets_memo={tuple(W.columns): {1:[(0,)]}})
    record('shared_evaluation_partition', float(run.expert_power_curve.values[1]),
           float(run.classifier_scores.values['AI']),
           'Definition 16 p21; shared-panel requirement p29', 'equivalence.py:793,803,1021-1027',
           note='Human predictor and AI both always predict a. Benchmark is scored on r1/r2 (0); AI also sees r0 (1/3).')

    # Paper Definition 13 samples unique set-valued partitions. Check the actual
    # generated benchmark sets, without making claims about population bias.
    for n, k, cap, name in [(4, 2, 4, 'subset_budget_lost_after_replacement'),
                            (6, 3, 3, 'ordered_tuples_duplicate_unordered_subsets')]:
        found = None
        for seed in range(100):
            W = pd.DataFrame([['a']*n, ['b']*n])
            run = pipeline(W, min_k=k, max_K=k+1, max_rater_subsets=cap, random_state=seed)
            subsets = next(iter(run.ratersets_memo.values()))[k]
            unique = len(set(frozenset(s) for s in subsets))
            observation = dict(seed=seed, subsets=[list(map(int, s)) for s in subsets], unique=unique)
            if unique < cap:
                found = observation
                break
        if found is None:
            found = dict(observation, tested_seeds_without_counterexample=100)
        record(name, cap, found['unique'], 'Definition 13 p20', 'equivalence.py:827-846',
               note=json.dumps(found))

    # Equal item weights and equal annotation weights differ with sparse data.
    W = pd.DataFrame([['a', 'a', 'a'], ['b', None, None]])
    p = [DiscretePrediction('a')]*2
    scorer = AgreementScore()
    record('sparse_item_vs_column_weighting', .5,
           scorer.expected_score(p, list(W.columns), W, anonymous=False),
           'IID uniform items pp12,47; paper assumes fixed rating depth', 'scoring_functions.py:197-229',
           'extension', 'Anonymous item-weighted score is .5; default averages column scores (.5+1+1)/3. Neither sparse policy is specified in the paper.')

    # Paper errata are independently evaluated, not counted as runtime failures.
    errata = dict(
        claim12='p19 inequality is reversed relative to Def9, Claim11, and its own proof.',
        calibration=dict(note='E[Y|prediction]=prediction is weaker than E[Y|all inputs].',
                         constant_calibrated_log_score=-1.,
                         posterior_log_score=.9*math.log2(.9)+.1*math.log2(.1)),
        squared_error=dict(note='p28 claim of linearity in evaluation label is false.',
                           prediction=.5, labels=[0,1], loss_against_mean=0., mean_loss=.25))
    report = dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  paper=dict(arxiv='2106.01254v3', date='2026-04-22', pages=60,
                             sha256='9e8ef1828a914882a497521e0768ad61f169dc817478166fe1d57526935568f3'),
                  cases=cases, passed=sum(c['conforms'] for c in cases), failed=sum(not c['conforms'] for c in cases),
                  paper_errata=errata,
                  audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  numerical_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted((ROOT/'surveyequivalence').glob('*.py'))},
                  comparison_note='2e-15 only for independent arithmetic; historical exact-parity fixtures unchanged.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('%s controls conform; %s paper-contract deviations/extensions reproduced -> %s' % (report['passed'], report['failed'], args.output))
    return int(report['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
