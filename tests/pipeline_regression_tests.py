"""Pipeline correctness and reproducibility, with independent expectations."""

import contextlib
import io
import random
import tempfile
import unittest

import numpy as np
import pandas as pd

from surveyequivalence import (
    AgreementScore, AnalysisPipeline, CrossEntropyScore, DiscretePrediction,
    DiscreteDistributionPrediction, FrequencyCombiner, PluralityVote,
    load_saved_pipeline,
)


class PipelineRegressionTests(unittest.TestCase):
    def setUp(self):
        self.W = pd.DataFrame([['a', 'a', 'b'], ['a', 'b', 'b'], ['a', 'a', 'a']],
                              columns=['r0', 'r1', 'r2'])

    def pipeline(self, W=None, **kwargs):
        options = dict(combiner=FrequencyCombiner(), scorer=AgreementScore(),
                       allowable_labels=['a', 'b'], anonymous_raters=True,
                       max_K=2, num_bootstrap_item_samples=0, procs=1,
                       verbosity=0, random_state=17)
        options.update(kwargs)
        return AnalysisPipeline(self.W if W is None else W, **options)

    def test_hand_computed_power_curve(self):
        pipeline = self.pipeline()
        self.assertEqual(pipeline.expert_power_curve.values[0], 2 / 3)
        self.assertEqual(pipeline.expert_power_curve.values[1], (.5 + 2 / 3 + .5) / 3)

    def test_single_item_column_and_undefined_equivalences(self):
        from surveyequivalence import AnonymousBayesianCombiner
        ratings = pd.DataFrame([['a']], columns=['r0'])
        predictions = pd.DataFrame({'classifier': [DiscretePrediction('a')]})
        frequency = self.pipeline(ratings, classifier_predictions=predictions)
        self.assertEqual(frequency.expert_power_curve.values[0], 1)
        unsupported = self.pipeline(ratings, classifier_predictions=predictions,
                                     combiner=AnonymousBayesianCombiner())
        self.assertTrue(np.isnan(unsupported.expert_power_curve.values[0]))
        self.assertTrue(np.isnan(unsupported.expert_survey_equivalences.df.iloc[0, 0]))

    def test_all_missing_inputs_produce_missing_scores(self):
        ratings = pd.DataFrame([[None, None], [None, None]], columns=['r0', 'r1'])
        predictions = pd.DataFrame({'classifier': [None, None]})
        pipeline = self.pipeline(ratings, classifier_predictions=predictions)
        self.assertTrue(pipeline.expert_power_curve.df.isna().all().all())
        self.assertTrue(pipeline.classifier_scores.df.isna().all().all())
        self.assertTrue(pipeline.expert_survey_equivalences.df.isna().all().all())

    def test_anonymization_helpers_use_shared_missing_rules(self):
        from surveyequivalence import find_maximal_full_rating_matrix_cols, prep_anonymized_rating_matrix
        ratings = pd.DataFrame([[0, '', pd.NA], [1, None, np.nan]])
        self.assertEqual(find_maximal_full_rating_matrix_cols(ratings), 1)
        result = prep_anonymized_rating_matrix(ratings)
        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.iloc[:, 0].tolist(), [0, 1])
        self.assertTrue(prep_anonymized_rating_matrix(pd.DataFrame([[None, '']])).empty)
        self.assertEqual(prep_anonymized_rating_matrix(ratings, 2).shape, (0, 2))

    def test_item_identifiers_and_repeated_samples(self):
        reference = self.pipeline(item_samples=[[0, 1, 2], [2, 0, 2, 1]])
        for index in (['first', 'middle', 'last'], [30, 10, 90]):
            ratings = self.W.set_axis(index)
            pipeline = self.pipeline(ratings, item_samples=[index, [index[i] for i in [2, 0, 2, 1]]])
            pd.testing.assert_frame_equal(reference.expert_power_curve.df,
                                          pipeline.expert_power_curve.df, check_exact=True)

    def test_classifier_indices_are_aligned(self):
        predictions = pd.DataFrame({'classifier': [DiscretePrediction(v) for v in ['a', 'b', 'a']]})
        pipeline = self.pipeline(classifier_predictions=predictions.iloc[::-1])
        self.assertEqual(pipeline.classifier_scores.values['classifier'], (2 / 3 + 2 / 3 + 1) / 3)

    def test_rater_positions_and_disjoint_amateurs(self):
        ratings = self.W.copy()
        ratings.insert(0, 'unselected', ['b'] * len(ratings))
        actual = self.pipeline(ratings, expert_cols=['r0', 'r1', 'r2'])
        reference = self.pipeline()
        # k=1 uses only the chosen columns; k=0's documented prior uses all W.
        self.assertEqual(actual.expert_power_curve.values[1], reference.expert_power_curve.values[1])
        ratings['amateur0'] = ['a', 'a', 'a']
        ratings['amateur1'] = ['a', 'a', 'a']
        pipeline = self.pipeline(ratings, expert_cols=['r0', 'r1', 'r2'],
                                 amateur_cols=['amateur0', 'amateur1'])
        self.assertEqual(pipeline.amateur_power_curve.values[1], 2 / 3)

    def test_sparse_rows_use_remaining_ratings(self):
        for missing in (None, np.nan, pd.NA, ''):
            ratings = pd.DataFrame([['a', missing, missing], ['b', 'b', missing],
                                    [missing, missing, missing]], columns=['r0', 'r1', 'r2'])
            predictions = pd.DataFrame({'classifier': [DiscretePrediction('a'), DiscretePrediction('b'), None]})
            pipeline = self.pipeline(ratings, classifier_predictions=predictions,
                                     scorer=AgreementScore(num_ref_raters_per_virtual_rater=3))
            self.assertEqual(pipeline.classifier_scores.values['classifier'], 1)
            self.assertTrue(np.isfinite(pipeline.expert_power_curve.df.to_numpy(dtype=float)).all())

    def test_logging_does_not_change_results(self):
        for anonymous in (False, True):
            reference = self.pipeline(anonymous_raters=anonymous)
            with contextlib.redirect_stdout(io.StringIO()):
                verbose = self.pipeline(anonymous_raters=anonymous, verbosity=3)
            pd.testing.assert_frame_equal(reference.expert_power_curve.df,
                                          verbose.expert_power_curve.df, check_exact=True)

    def test_seeded_plurality_is_independent_of_workers_and_caller_rng(self):
        np.random.seed(41)
        random.seed(42)
        before_numpy = np.random.get_state()
        before_python = random.getstate()
        reference = self.pipeline(combiner=PluralityVote(), num_bootstrap_item_samples=2)
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        parallel = self.pipeline(combiner=PluralityVote(), num_bootstrap_item_samples=2, procs=2)
        pd.testing.assert_frame_equal(reference.expert_power_curve.df,
                                      parallel.expert_power_curve.df, check_exact=True)

    def test_invalid_inputs_have_clear_errors(self):
        for ratings in (pd.DataFrame(), self.W.set_axis([0, 0, 1]),
                        self.W.set_axis(['r0', 'r0', 'r2'], axis=1)):
            with self.assertRaises(ValueError):
                self.pipeline(ratings, run_on_creation=False)
        for options in ({'expert_cols': ['unknown']}, {'procs': 0},
                        {'random_state': -1}, {'working_memory_mb': 0},
                        {'item_samples': [[99]]}, {'item_samples': []},
                        {'classifier_predictions': pd.DataFrame({'x': [None]})}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.pipeline(run_on_creation=False, **options)

    def test_input_frame_is_not_modified(self):
        ratings = self.W.copy(deep=True)
        original = ratings.copy(deep=True)
        self.pipeline(ratings)
        pd.testing.assert_frame_equal(ratings, original, check_exact=True)
        self.assertEqual(ratings.attrs, original.attrs)

    def test_saved_configuration_and_exact_results_round_trip(self):
        predictions = pd.DataFrame({'classifier': [DiscreteDistributionPrediction(['a', 'b'], [.37, .63])
                                                   for _ in range(len(self.W))]})
        pipeline = self.pipeline(scorer=CrossEntropyScore(), classifier_predictions=predictions,
                                 num_bootstrap_item_samples=2)
        with tempfile.TemporaryDirectory() as directory:
            pipeline.save(directory, save_results=False)
            loaded = load_saved_pipeline(directory)
        self.assertEqual(loaded.random_state, 17)
        self.assertEqual(loaded.working_memory_mb, 512)
        self.assertTrue(loaded.anonymous_raters)
        pd.testing.assert_frame_equal(pipeline.expert_power_curve.df,
                                      loaded.expert_power_curve.df, check_exact=True)
        pd.testing.assert_frame_equal(pipeline.classifier_scores.df,
                                      loaded.classifier_scores.df, check_exact=True)


if __name__ == '__main__':
    unittest.main()
