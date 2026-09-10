"""Generate exact pipeline fixtures from an explicitly selected reference tree.

Never regenerate fixtures implicitly in a test. The recorded source tree is the
correctness stage, before count caching, prepared scores, or execution changes.
"""

import argparse
import json
from pathlib import Path
import sys


def make_case(name):
    import numpy as np
    import pandas as pd
    from surveyequivalence import (
        AnonymousBayesianCombiner, FrequencyCombiner, PluralityVote, MeanCombiner,
        AgreementScore, CrossEntropyScore, Correlation, AUCScore, F1Score,
        DMIScore_for_Hard_Classifier, DMIScore_for_Soft_Classifier,
        DiscreteDistributionPrediction, NumericPrediction,
    )
    rng = np.random.RandomState(923)
    labels = ['a', 'b', 'c'] if 'multiclass' in name else ['a', 'b']
    W = pd.DataFrame(rng.choice(labels, (9, 5)), index=['item_%d' % i for i in range(9)],
                     columns=['r%d' % i for i in range(5)])
    if 'sparse' in name:
        W = W.astype(object)
        W.iloc[0, 1:] = None
        W.iloc[3, [0, 2]] = np.nan
        W.iloc[5, :] = None
        W.iloc[7, 2] = ''
    probabilities = rng.uniform(.1, 1, (len(W), len(labels)))
    predictions = pd.DataFrame({'classifier': [DiscreteDistributionPrediction(labels, list(row))
                                               for row in probabilities]}, index=W.index)
    combiner = (PluralityVote() if 'plurality' in name else FrequencyCombiner()
                if 'frequency' in name or 'multiclass' in name else AnonymousBayesianCombiner())
    scorer = AgreementScore() if 'agreement' in name or 'plurality' in name else CrossEntropyScore()
    if 'panel' in name or 'multiclass' in name:
        scorer.num_ref_raters_per_virtual_rater = 3
        scorer.num_virtual_raters = 7
    if name == 'auc':
        scorer = AUCScore()
        scorer.num_virtual_raters = 7
    if name == 'f1':
        scorer = F1Score()
        scorer.num_virtual_raters = 7
    if name == 'hard_dmi':
        scorer = DMIScore_for_Hard_Classifier()
    if name == 'soft_dmi':
        scorer = DMIScore_for_Soft_Classifier()
    min_k = 0
    if name == 'numeric':
        W = pd.DataFrame(rng.normal(size=(9, 5)), index=W.index, columns=W.columns)
        W.iloc[2, 0] = np.nan
        predictions = pd.DataFrame({'classifier': [NumericPrediction(float(x)) for x in rng.normal(size=9)]}, index=W.index)
        combiner, scorer, labels, min_k = MeanCombiner(), Correlation(), None, 1
    options = dict(W=W, classifier_predictions=predictions, combiner=combiner, scorer=scorer,
                   allowable_labels=labels, anonymous_raters='nonanonymous' not in name and name != 'numeric',
                   item_samples=[W.index, W.index[[8, 0, 8, 1, 4, 2, 3, 5, 6]], W.index[::-1]],
                   max_K=4, min_k=min_k, max_rater_subsets=4, verbosity=0, procs=1, random_state=1729)
    if name == 'amateurs':
        options.update(expert_cols=['r1', 'r3', 'r4'], amateur_cols=['r0', 'r2'], max_K=3)
    if min_k == 0:
        options['performance_ratio_k'] = 1
    return options


CASES = ('abc_cross_entropy', 'abc_agreement', 'frequency_cross_entropy',
         'frequency_agreement', 'abc_nonanonymous', 'frequency_nonanonymous_agreement',
         'abc_sparse_panel', 'frequency_sparse_panel_agreement',
         'plurality', 'multiclass_cross_entropy', 'multiclass_agreement',
         'amateurs', 'auc', 'f1', 'hard_dmi', 'soft_dmi', 'numeric')


def capture(pipeline):
    from benchmarks.run_benchmarks import pipeline_result, frame_result
    result = pipeline_result(pipeline)
    for name in ('classifier_scores', 'expert_power_curve', 'expert_survey_equivalences',
                 'amateur_power_curve', 'amateur_survey_equivalences'):
        value = getattr(pipeline, name, None)
        if value is not None:
            for statistic in ('std_lower_bounds', 'std_upper_bounds',
                              'empirical_lower_bounds', 'empirical_upper_bounds'):
                result[name][statistic] = frame_result(getattr(value, statistic).to_frame(name=statistic))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical', action='store_true', help='Capture valid deterministic cases from the original tree')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(args.repo.resolve()))
    import surveyequivalence
    from surveyequivalence import AnalysisPipeline
    result = {'source': str(Path(surveyequivalence.__file__).resolve()), 'cases': {}}
    cases = CASES[:4] if args.historical else CASES
    for name in cases:
        options = make_historical_case(name) if args.historical else make_case(name)
        result['cases'][name] = capture(AnalysisPipeline(**options))
        print(name, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


def make_historical_case(name):
    from itertools import combinations
    options = make_case(name)
    frame = options['W']
    samples = [frame.index.get_indexer(sample) for sample in options['item_samples']]
    options['W'] = frame.reset_index(drop=True)
    options['combiner'] = type(options['combiner'])(allowable_labels=options['allowable_labels'], W=options['W'])
    options['classifier_predictions'] = options['classifier_predictions'].reset_index(drop=True)
    options['item_samples'] = samples
    options['verbosity'] = 1  # Original power-curve scoring accidentally used this as anonymous=True.
    options.pop('random_state')
    options['max_rater_subsets'] = 200
    options['ratersets_memo'] = {tuple(frame.columns): {
        k: list(combinations(range(len(frame.columns)), k)) for k in range(options['max_K'])}}
    return options


if __name__ == '__main__':
    main()
