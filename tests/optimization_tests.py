"""Exact differential checks for optimized deterministic computations."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from surveyequivalence import (
    AnalysisPipeline, AnonymousBayesianCombiner, FrequencyCombiner,
    AgreementScore, CrossEntropyScore, DiscreteDistributionPrediction, DiscretePrediction,
)
from surveyequivalence._scoring import ordered_mean, prepare_scores, score_prepared
from surveyequivalence._execution import PredictionBlock


class FailingAgreement(AgreementScore):
    def expected_score(self, *args, **kwargs):
        raise RuntimeError('intentional worker failure')


class RecordingFrequency(FrequencyCombiner):
    def __init__(self):
        super().__init__()
        self.calls = []

    def combine(self, allowable_labels, labels, W=None, item_id=None, **kwargs):
        self.calls.append((item_id, tuple(rater for rater, _ in labels)))
        return super().combine(allowable_labels, labels, W, item_id, **kwargs)


class ScalarBayesian(AnonymousBayesianCombiner):
    """Force the public, unprepared combiner path for parity comparisons."""


class StatefulAgreement(AgreementScore):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def expected_score(self, *args, **kwargs):
        self.calls += 1
        return self.calls / 100


class PreparedScoreTests(unittest.TestCase):
    def test_ordered_reduction_keeps_rounding_and_signed_zero(self):
        for values in ([1e16, 1., -1e16, 3.], [-0.], [-1., 1.], [1e-300] * 101,
                       list(np.random.RandomState(8).normal(size=301))):
            expected = sum(values) / len(values)
            self.assertEqual(float(ordered_mean(values)).hex(), float(expected).hex())
        self.assertIsNone(ordered_mean([]))

    def test_prepared_scores_equal_public_scorers_exactly(self):
        rng = np.random.RandomState(983)
        for n_items, n_raters in ((7, 3), (3, 9), (6, 6)):
            W = pd.DataFrame(rng.choice(['a', 'b', None], (n_items, n_raters)))
            probabilities = rng.uniform(.021, .979, n_items)
            predictions = [DiscreteDistributionPrediction(['b', 'a'], [float(p), float(1 - p)]) for p in probabilities]
            predictions[1] = None
            samples = [np.arange(n_items), rng.choice(n_items, n_items + 2), np.arange(n_items)[::-1]]
            for scorer in (AgreementScore(), CrossEntropyScore(), AgreementScore(num_ref_raters_per_virtual_rater=4),
                           CrossEntropyScore(num_ref_raters_per_virtual_rater=4)):
                for anonymous in (False, True):
                    terms = prepare_scores(scorer, predictions, W, anonymous)
                    self.assertIsNotNone(terms)
                    values, valid, mode = terms
                    for indices in samples:
                        expected = scorer.expected_score([predictions[i] for i in indices], list(W.columns),
                                                         W.iloc[indices].reset_index(drop=True), anonymous=anonymous)
                        actual = score_prepared(values, valid, indices, mode)
                        self.assertEqual(actual, expected, (n_items, n_raters, scorer, anonymous, indices))

    def test_custom_override_and_multiclass_sampling_use_fallback(self):
        W = pd.DataFrame([['a', 'b', 'c']])
        pred = [DiscretePrediction('a')]
        self.assertIsNone(prepare_scores(AgreementScore(num_ref_raters_per_virtual_rater=2), pred, W, True))
        scorer = AgreementScore()
        scorer.expected_score = lambda *args, **kwargs: .125
        self.assertIsNone(prepare_scores(scorer, pred, W, True))

    def test_prepared_bayesian_counts_match_scalar_arithmetic(self):
        rng = np.random.RandomState(47)
        for shape in ((9, 3), (3, 9), (7, 7)):
            W = rng.choice(['a', 'b', None], size=shape)
            reference = AnonymousBayesianCombiner()
            prepared = AnonymousBayesianCombiner()
            prepared._prepare(W, ['a', 'b'])
            for counts in ((0, 0), (1, 0), (0, 1), (2, 1), (2, 2)):
                labels = [(i, label) for i, label in enumerate(['a'] * counts[0] + ['b'] * counts[1])]
                for item in [None] + list(range(len(W))):
                    left = reference.combine(['a', 'b'], labels, W, item)
                    right = prepared.combine(['a', 'b'], labels, W, item)
                    if left is None:
                        self.assertIsNone(right)
                    else:
                        self.assertEqual(left.probabilities, right.probabilities)

    def test_explicit_frequency_dataset_overrides_bound_prior(self):
        combiner = FrequencyCombiner(W=np.array([['a', 'a']]))
        result = combiner.combine(['a', 'b'], [], W=np.array([['b', 'b']]))
        self.assertEqual(result.probabilities, [.02, .98])

    def test_prediction_buffers_do_not_renormalize(self):
        prediction = DiscreteDistributionPrediction(['a', 'b', 'c'], [.2, .3, .7])
        with tempfile.TemporaryDirectory() as directory:
            block = PredictionBlock(directory, [(0, 0, ())], 2, prediction.label_names,
                                    FrequencyCombiner(), 3, disk=True)
            block.put(0, 0, prediction)
            block.seal()
            loaded = PredictionBlock.load(block.descriptor())
            try:
                values = loaded.predictions(0, [0, 1, 0])
                self.assertEqual(values[0].probabilities, prediction.probabilities)
                self.assertIsNone(values[1])
                self.assertFalse(loaded.values.flags.writeable)
            finally:
                loaded.close()
                block.close()


class ExactPipelineFixtureTests(unittest.TestCase):
    def test_subset_sampling_matches_reference_across_seeds_and_branches(self):
        from benchmarks.subset_fixtures import capture
        state = np.random.get_state()
        try:
            expected = json.loads((Path(__file__).parent / 'fixtures/subset_samples.json').read_text())
            self.assertEqual(capture(), expected['cases'])
        finally:
            np.random.set_state(state)

    def test_plurality_ties_retain_independent_uniform_draws(self):
        from collections import Counter
        from surveyequivalence import PluralityVote
        from surveyequivalence._random import random_stream
        combiner = PluralityVote()
        counts = Counter()
        different = 0
        for seed in range(1000):
            draws = []
            for item in (0, 1):
                with random_stream(seed, 3, item):
                    draws.append(combiner.combine(['a', 'b'], [(0, 'a'), (1, 'b')]).value)
            counts[draws[0]] += 1
            different += draws[0] != draws[1]
        # Two equally frequent winners each have probability 1/2; independent
        # pairs differ with probability 1/2. Fixed seeds avoid flaky sampling.
        self.assertEqual(set(counts), {'a', 'b'})
        self.assertTrue(435 <= counts['a'] <= 565, counts)
        self.assertTrue(435 <= different <= 565, different)

    def test_valid_historical_outputs_are_unchanged(self):
        import contextlib
        import io
        from benchmarks.reference_fixtures import make_historical_case, capture
        expected = json.loads((Path(__file__).parent / 'fixtures/historical_pipeline.json').read_text())['cases']
        for name in expected:
            with self.subTest(case=name), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(capture(AnalysisPipeline(**make_historical_case(name))), expected[name])

    def test_corrected_reference_outputs(self):
        from benchmarks.reference_fixtures import make_case, capture
        filename = Path(__file__).parent / 'fixtures/corrected_pipeline.json'
        expected = json.loads(filename.read_text())['cases']
        for name in expected:
            with self.subTest(case=name):
                pipeline = AnalysisPipeline(**make_case(name))
                self.assertEqual(capture(pipeline), expected[name])

    def test_blocks_and_workers_preserve_seeded_outputs(self):
        from benchmarks.reference_fixtures import make_case, capture
        for name in ('abc_sparse_panel', 'plurality', 'multiclass_cross_entropy', 'frequency_nonanonymous_agreement'):
            reference = capture(AnalysisPipeline(**make_case(name)))
            for workers in (1, 2):
                options = make_case(name)
                options.update(procs=workers, working_memory_mb=.002)
                with self.subTest(case=name, workers=workers):
                    pipeline = AnalysisPipeline(**options)
                    self.assertGreater(pipeline.execution_stats[0]['blocks'], 1)
                    self.assertEqual(capture(pipeline), reference)

    def test_serial_path_never_creates_pool(self):
        from benchmarks.reference_fixtures import make_case
        with mock.patch('surveyequivalence.equivalence.multiprocess.get_context', side_effect=AssertionError('unexpected pool')):
            AnalysisPipeline(**make_case('abc_cross_entropy'))

    def test_worker_exceptions_clean_up_pool_and_files(self):
        from benchmarks.reference_fixtures import make_case
        import multiprocess
        created = []
        temporary_directory = tempfile.TemporaryDirectory

        def recording_directory(*args, **kwargs):
            directory = temporary_directory(*args, **kwargs)
            created.append(Path(directory.name))
            return directory

        before = {process.pid for process in multiprocess.active_children()}
        options = make_case('abc_cross_entropy')
        options.update(scorer=FailingAgreement(), classifier_predictions=None, performance_ratio_k=None, procs=2)
        with mock.patch('surveyequivalence.equivalence.tempfile.TemporaryDirectory', side_effect=recording_directory):
            with self.assertRaisesRegex(RuntimeError, 'intentional worker failure'):
                AnalysisPipeline(**options)
        self.assertTrue(created)
        self.assertTrue(all(not directory.exists() for directory in created))
        self.assertEqual(before, {process.pid for process in multiprocess.active_children()})

    def test_custom_combiner_retains_row_major_order(self):
        from benchmarks.reference_fixtures import make_case, capture
        options = make_case('frequency_agreement')
        combiner = RecordingFrequency()
        options.update(combiner=combiner, working_memory_mb=.002, procs=2)
        pipeline = AnalysisPipeline(**options)
        subsets = next(iter(pipeline.ratersets_memo.values()))
        expected_calls = [(item, tuple(subset)) for item in range(len(pipeline.W))
                          for k in subsets for subset in subsets[k]]
        self.assertEqual(combiner.calls, expected_calls)
        reference = AnalysisPipeline(**make_case('frequency_agreement'))
        self.assertEqual(capture(pipeline), capture(reference))

    def test_instance_combiner_override_uses_object_fallback(self):
        from benchmarks.reference_fixtures import make_case, capture
        def combine(*args, **kwargs):
            return DiscretePrediction('a')
        options = make_case('frequency_agreement')
        reference_combiner = RecordingFrequency()
        reference_combiner.combine = combine
        options['combiner'] = reference_combiner
        reference = capture(AnalysisPipeline(**options))
        for factory in (FrequencyCombiner, AnonymousBayesianCombiner):
            for workers in (1, 2):
                combiner = factory()
                combiner.combine = combine
                options.update(combiner=combiner, procs=workers, working_memory_mb=.002)
                actual = AnalysisPipeline(**options)
                self.assertEqual(actual.execution_stats[0]['prepared_subsets'], 0)
                self.assertEqual(capture(actual), reference)

    def test_rerun_rebuilds_dataset_caches(self):
        from benchmarks.reference_fixtures import make_case, capture
        options = make_case('abc_cross_entropy')
        pipeline = AnalysisPipeline(**options)
        pipeline.W.iloc[0, :] = 'b'
        pipeline.run()
        options = make_case('abc_cross_entropy')
        options['W'].iloc[0, :] = 'b'
        self.assertEqual(capture(pipeline), capture(AnalysisPipeline(**options)))

    def test_explicit_bayesian_training_data_remains_authoritative(self):
        from benchmarks.reference_fixtures import make_case, capture
        options = make_case('abc_cross_entropy')
        training = options['W'].replace({'a': 'b', 'b': 'a'})
        options['combiner'] = AnonymousBayesianCombiner(W=training)
        actual = capture(AnalysisPipeline(**options))
        options['combiner'] = ScalarBayesian(W=training)
        self.assertEqual(actual, capture(AnalysisPipeline(**options)))

    def test_custom_scorer_state_is_not_reset_at_block_boundaries(self):
        from benchmarks.reference_fixtures import make_case, capture
        options = make_case('abc_cross_entropy')
        options.update(scorer=StatefulAgreement(), classifier_predictions=None, performance_ratio_k=None)
        reference = AnalysisPipeline(**options)
        self.assertGreater(reference.expert_power_curve.values[1], .01)
        options.update(scorer=StatefulAgreement(), working_memory_mb=.002)
        actual = AnalysisPipeline(**options)
        self.assertEqual(actual.execution_stats[0]['blocks'], 1)
        self.assertEqual(capture(actual), capture(reference))

    def test_instance_scorer_override_receives_fresh_run_state(self):
        from types import MethodType
        from benchmarks.reference_fixtures import make_case, capture
        options = make_case('abc_cross_entropy')
        options.update(scorer=StatefulAgreement(), classifier_predictions=None, performance_ratio_k=None)
        reference = capture(AnalysisPipeline(**options))
        for workers in (1, 2):
            scorer = AgreementScore()
            scorer.calls = 0
            scorer.expected_score = MethodType(StatefulAgreement.expected_score, scorer)
            options.update(scorer=scorer, procs=workers, working_memory_mb=.002)
            self.assertEqual(capture(AnalysisPipeline(**options)), reference)


if __name__ == '__main__':
    unittest.main()
