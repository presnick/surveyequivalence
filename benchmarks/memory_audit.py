"""Measure prepared buffers, cache size, and retained temporary blocks.

This measures working objects and files, not process RSS. The private combiner
is retained by an instrumentation wrapper so its live caches can be inspected.
"""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from surveyequivalence import AnalysisPipeline, AnonymousBayesianCombiner, CrossEntropyScore
from surveyequivalence._data import PreparedRatings
from surveyequivalence._execution import PredictionBlock


def graph_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(graph_bytes(key, seen) + graph_bytes(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(graph_bytes(item, seen) for item in value)
    elif hasattr(value, '__dict__'):
        size += graph_bytes(vars(value), seen)
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/correctness-audit/memory.json'))
    args = parser.parse_args()
    labels = ['a', 'b']
    W = pd.DataFrame(np.random.RandomState(91).choice(labels, size=(500, 8)))
    prepared_bytes = PreparedRatings(W, labels).nbytes
    retained, files_at_close = [], []
    original_prepare, original_close = AnonymousBayesianCombiner._prepare, PredictionBlock.close

    def prepare(combiner, *args, **kwargs):
        result = original_prepare(combiner, *args, **kwargs)
        retained.append(combiner)
        return result

    def close(block):
        result = original_close(block)
        directory = Path(block.directory).parent
        paths = list(directory.glob('*/*.npy'))
        files_at_close.append(dict(blocks=len(list(directory.iterdir())), bytes=sum(p.stat().st_size for p in paths)))
        return result

    with patch.object(AnonymousBayesianCombiner, '_prepare', prepare), patch.object(PredictionBlock, 'close', close):
        pipeline = AnalysisPipeline(W, combiner=AnonymousBayesianCombiner(), scorer=CrossEntropyScore(),
                                    allowable_labels=labels, anonymous_raters=True, max_K=4,
                                    max_rater_subsets=30, num_bootstrap_item_samples=0,
                                    random_state=1729, working_memory_mb=.01, procs=1, verbosity=0)
    combiner = retained[0]
    report = dict(shape=list(W.shape), max_K=4, max_rater_subsets=30, seed=1729,
                  working_memory_mb=.01, budget_bytes=int(.01 * 1024 * 1024),
                  prepared_numeric_bytes=prepared_bytes,
                  prediction_cache_entries=len(combiner.combined),
                  cache_graph_bytes=graph_bytes((combiner.combined, combiner.memo, combiner._count_probabilities)),
                  first_block_close=files_at_close[0], last_block_close=files_at_close[-1],
                  execution=pipeline.execution_stats,
                  note='Unique Python object sizes and temporary array files; neither is RSS. Cache retention is instrumented for observation. Caches are already retained throughout normal curve computation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
