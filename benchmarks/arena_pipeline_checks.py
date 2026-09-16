"""Exact optimized/generic parity on real Arena subsets; no downloads."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.arena_validation import (ROOT, LABELS, CrossEntropyScore, FrequencyCombiner,
                                        DiscreteDistributionPrediction, pipeline, exact_frame,
                                        read_judge, audit_arena, write_json)
from surveyequivalence.equivalence import load_saved_pipeline
import numpy as np
import pandas as pd


class GenericCrossEntropy(CrossEntropyScore):
    """Inherited scalar implementation deliberately bypasses prepared dispatch."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstraps', type=int, default=500)
    args = parser.parse_args()
    cache = ROOT / '.cache/arena-validation'
    output = ROOT / 'benchmarks/results/arena-validation'
    battles, labels, _, _ = audit_arena(cache)
    ids = [r['id'] for r in battles]
    _, _, probabilities, _ = read_judge(cache / 'Meta-Llama-3-70B-Instruct.forward.jsonl', ids, labels)
    rng = np.random.RandomState(1729)
    ordering = rng.permutation(len(ids))
    results = []
    for size in [50, 1000, 2000]:
        chosen = ordering[:size]
        index = pd.Index([f'battle-{ids[i]}' for i in chosen])
        W = pd.DataFrame({'human': [labels[i] for i in chosen]}, index=index)
        predictions = pd.DataFrame({'judge': [DiscreteDistributionPrediction(LABELS, [float(p) for p in probabilities[i]])
                                               for i in chosen]}, index=index)
        local_rng = np.random.RandomState(1729 + size)
        samples = [np.arange(size)] + [local_rng.randint(size, size=size) for _ in range(args.bootstraps)]
        times = {}
        outputs = {}
        for name, scorer in [('optimized', CrossEntropyScore()), ('generic', GenericCrossEntropy())]:
            started = time.perf_counter()
            outputs[name] = pipeline(W, predictions.iloc[::-1], FrequencyCombiner(), scorer, samples)
            times[name] = time.perf_counter() - started
        optimized, generic = outputs['optimized'], outputs['generic']
        for attr in ['classifier_scores', 'expert_power_curve', 'expert_survey_equivalences']:
            left, right = getattr(optimized, attr), getattr(generic, attr)
            exact_frame(left.df, right.df)
            for summary in ['means', 'stds', 'lower_bounds', 'upper_bounds']:
                pd.testing.assert_series_equal(getattr(left, summary), getattr(right, summary), check_exact=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            optimized.save(directory, save_results=False)
            restored = load_saved_pipeline(directory)
            exact_frame(optimized.classifier_scores.df, restored.classifier_scores.df)
            exact_frame(optimized.expert_power_curve.df, restored.expert_power_curve.df)
        records = dict(n=size, bootstrap_replicates=args.bootstraps, exact_parity=True,
                       save_load_exact=True, classifier_cells=int(optimized.classifier_scores.df.size),
                       power_cells=int(optimized.expert_power_curve.df.size), seconds=times,
                       note='Single-human k=0 curve only; derived equivalences are parity checks, not human-equivalence estimates.')
        results.append(records)
        write_json(output / 'real_data_pipeline_parity.json', results)
        print(records, flush=True)


if __name__ == '__main__':
    main()
