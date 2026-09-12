"""Independent execution checks; reports evidence without adding failing tests.

Run with the project's interpreter:
    .venv/bin/python benchmarks/execution_audit.py

Every reported ``matches`` field should be true once the corresponding
compatibility or snapshot behavior has been addressed. This is an audit tool,
not a generator for the frozen historical/corrected fixtures.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='Also save the complete JSON evidence to this path')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from benchmarks.reference_fixtures import make_case
    from benchmarks.run_benchmarks import frame_result
    from surveyequivalence import AnalysisPipeline, CrossEntropyScore, load_saved_pipeline

    class PublicCrossEntropy(CrossEntropyScore):
        """Inherited public scorer implementation without prepared dispatch."""

    python_state, numpy_state = random.getstate(), np.random.get_state()
    evidence = {}
    try:
        for name in ('plurality', 'f1'):
            runs = []
            for memory in (512, .002):
                random.seed(7123)
                np.random.seed(7123)
                options = make_case(name)
                options.update(random_state=None, working_memory_mb=memory, procs=1)
                pipeline = AnalysisPipeline(**options)
                runs.append({'memory_mb': memory,
                             'blocks': pipeline.execution_stats[0]['blocks'],
                             'curve': frame_result(pipeline.expert_power_curve.df)})
            evidence['global_rng_blocks_' + name] = {
                'matches': runs[0]['curve'] == runs[1]['curve'], 'runs': runs}

        runs = []
        for public in (False, True):
            random.seed(7123)
            np.random.seed(7123)
            options = make_case('multiclass_cross_entropy')
            predictions = options['classifier_predictions']
            predictions['second'] = predictions['classifier'].iloc[::-1].to_list()
            options.update(random_state=None, working_memory_mb=512, procs=1)
            if public:
                options['scorer'] = PublicCrossEntropy(
                    num_ref_raters_per_virtual_rater=3, num_virtual_raters=7)
            pipeline = AnalysisPipeline(**options)
            runs.append({'public_dispatch': public,
                         'scores': frame_result(pipeline.classifier_scores.df)})
        evidence['global_rng_classifier_call_order'] = {
            'matches': runs[0]['scores'] == runs[1]['scores'], 'runs': runs}

        pipeline = AnalysisPipeline(**make_case('abc_cross_entropy'))
        pipeline.W = pipeline.W.iloc[::-1].copy()
        pipeline.run()
        options = make_case('abc_cross_entropy')
        options['W'] = options['W'].iloc[::-1].copy()
        fresh = AnalysisPipeline(**options)
        actual = frame_result(pipeline.classifier_scores.df)
        expected = frame_result(fresh.classifier_scores.df)
        evidence['rerun_realigns_classifier_rows'] = {
            'matches': actual == expected, 'rerun': actual, 'fresh': expected}

        options = make_case('abc_cross_entropy')
        options['run_on_creation'] = False
        pipeline = AnalysisPipeline(**options)
        external = options['classifier_predictions'].iloc[0, 0]
        before = list(pipeline.classifier_predictions.iloc[0, 0].probabilities)
        external.probabilities[:] = [.99, .01]
        after = list(pipeline.classifier_predictions.iloc[0, 0].probabilities)
        evidence['prediction_object_snapshot'] = {
            'matches': before == after, 'before': before, 'after': after,
            'same_object': external is pipeline.classifier_predictions.iloc[0, 0]}

        options = make_case('abc_cross_entropy')
        options['classifier_predictions'].columns = [17]
        pipeline = AnalysisPipeline(**options)
        with tempfile.TemporaryDirectory(prefix='surveyequivalence-audit-') as directory:
            pipeline.save(directory, save_results=False)
            loaded = load_saved_pipeline(directory)
        original = pipeline.classifier_scores.df.columns.tolist()
        restored = loaded.classifier_scores.df.columns.tolist()
        evidence['typed_classifier_result_roundtrip'] = {
            'matches': original == restored, 'original': original, 'loaded': restored}
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    serialized = json.dumps(evidence, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')
    return 0 if all(case['matches'] for case in evidence.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
