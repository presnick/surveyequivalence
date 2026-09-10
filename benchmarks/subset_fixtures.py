"""Capture subset schedules from the separately frozen correctness reference."""

import argparse
import json
from pathlib import Path
import sys


def capture():
    import numpy as np
    import pandas as pd
    from surveyequivalence import AnalysisPipeline, FrequencyCombiner, AgreementScore

    ratings = pd.DataFrame([['a', 'b'] * 3, ['b', 'a'] * 3],
                           columns=['r%d' % i for i in range(6)])
    cases = {}
    for seed in (0, 1, 2, 73):
        for cap in (1, 4, 20):
            np.random.seed(seed)
            pipeline = AnalysisPipeline(
                ratings, combiner=FrequencyCombiner(), scorer=AgreementScore(),
                allowable_labels=['a', 'b'], anonymous_raters=True,
                item_samples=[ratings.index], num_bootstrap_item_samples=0,
                max_K=3, max_rater_subsets=cap, procs=1, verbosity=0)
            subsets = next(iter(pipeline.ratersets_memo.values()))
            cases['%d:%d' % (seed, cap)] = {
                str(k): [list(map(int, subset)) for subset in values]
                for k, values in subsets.items()}
    return cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    args.output.write_text(json.dumps({'reference': '093c94c', 'cases': capture()}, indent=2) + '\n')
