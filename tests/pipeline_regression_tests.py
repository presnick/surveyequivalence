"""Consolidated regression cases and independent mathematical checks."""

import contextlib
import io
import math
import numpy as np
import pandas as pd
import random
import tempfile
import unittest
from itertools import product
from surveyequivalence import (
    AgreementScore,
    AnalysisPipeline,
    CrossEntropyScore,
    DiscretePrediction,
    DiscreteDistributionPrediction,
    FrequencyCombiner,
    PluralityVote,
    load_saved_pipeline,
    AnonymousBayesianCombiner,
    AUCScore,
    DMIScore_for_Soft_Classifier,
)
from surveyequivalence._equivalence import supported_equivalence
from surveyequivalence.equivalence import PowerCurve


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


class PaperProtocolTests(unittest.TestCase):
    def test_equal_plateau_is_first_crossing(self):
        curve = PowerCurve(df=pd.DataFrame([{0:.2, 1:.5, 2:.5, 3:.8}]))
        self.assertEqual(curve.compute_one_equivalence(.5), 1.)
        self.assertEqual(curve.compute_one_equivalence_result(.9),
                         dict(status='above_range', estimate=None, bound=3.))
        self.assertTrue(math.isnan(curve.compute_one_equivalence(.5, {0:.2, 2:.8})))
        for missing in (None, np.nan, pd.NA):
            self.assertEqual(curve.compute_one_equivalence_result(missing)['status'], 'undefined')
            self.assertEqual(curve.compute_one_equivalence_result(.5, {0:missing,1:.8})['status'], 'missing_curve')

    def test_below_baseline_is_identified_zero(self):
        self.assertEqual(supported_equivalence(.1, {0:.2, 1:.5}),
                         dict(status='crossing', estimate=0., bound=None))
        self.assertEqual(supported_equivalence(.1, {1:.2, 2:.5})['status'], 'below_range')

    def test_raw_posterior_and_legacy_regularization_are_explicit(self):
        W = np.array([['a']*99+['b']]*3, dtype=object)
        raw = AnonymousBayesianCombiner(extreme_cutoff=0).combine(['a','b'], [], W, 0)
        self.assertEqual(raw.probabilities, [.99,.01])
        self.assertEqual(AnonymousBayesianCombiner().combine(['a','b'], [], W, 0).probabilities, [.98,.02])
        self.assertEqual(FrequencyCombiner(extreme_cutoff=0).combine(['a','b'], [(0,'a')]).probabilities, [1.,0.])

    def test_zero_log_probability_and_zero_weight_terms(self):
        p = DiscreteDistributionPrediction(['a','b'], [1.,0.], extreme_cutoff=0)
        self.assertEqual(CrossEntropyScore.score([p], ['b']), -math.inf)
        self.assertEqual(CrossEntropyScore(num_ref_raters_per_virtual_rater=3)
                         .expected_score_anonymous_raters([p], pd.DataFrame([['a','a','b']])), 0.)

    def test_invalid_probability_vectors_are_rejected(self):
        for values in ([math.nan,.5], [-.1,1.1], [0.,0.]):
            with self.assertRaises(ValueError):
                DiscreteDistributionPrediction(['a','b'], values)

    def test_auc_uses_a_fixed_positive_class(self):
        p = [DiscreteDistributionPrediction(['neg','pos'], [1-v,v]) for v in [.9,.8,.2,.1]]
        self.assertEqual(AUCScore.score(p, ['pos','pos','neg','neg']), 1.)
        p[1] = DiscreteDistributionPrediction(['pos','neg'], [.8,.2])
        self.assertEqual(AUCScore.score(p, ['pos','pos','neg','neg']), 1.)

    def test_auc_matches_independent_pair_enumeration_in_pipeline(self):
        from benchmarks.reference_fixtures import make_case

        class PairwiseAUC(AUCScore):
            @staticmethod
            def score(predictions, labels, **kwargs):
                pairs = [(p,y) for p,y in zip(predictions,labels) if p is not None and y is not None]
                classes = sorted({y for p,y in pairs})
                if len(classes) != 2:
                    return np.nan
                positive = [p.label_probability(classes[-1]) for p,y in pairs if y == classes[-1]]
                negative = [p.label_probability(classes[-1]) for p,y in pairs if y != classes[-1]]
                return sum(float(a>b)+.5*float(a==b) for a,b in product(positive,negative))/(len(positive)*len(negative))

        options=make_case('auc')
        actual=AnalysisPipeline(**options)
        oracle=PairwiseAUC()
        oracle.num_virtual_raters=options['scorer'].num_virtual_raters
        options['scorer']=oracle
        expected=AnalysisPipeline(**options)
        # Pair counting and sklearn integrate the same ROC using different
        # arithmetic; this is an independent oracle, not an exact-parity test.
        np.testing.assert_allclose(actual.expert_power_curve.df,expected.expert_power_curve.df,
                                   rtol=0,atol=2e-15,equal_nan=True)
        np.testing.assert_allclose(actual.classifier_scores.df,expected.classifier_scores.df,
                                   rtol=0,atol=2e-15,equal_nan=True)

    def test_dmi_aligns_named_probabilities(self):
        p = [DiscreteDistributionPrediction(['a','b'], [.8,.2]),
             DiscreteDistributionPrediction(['b','a'], [.8,.2])]
        self.assertAlmostEqual(DMIScore_for_Soft_Classifier.score(p, ['a','b']), .15, delta=1e-15)
        self.assertAlmostEqual(DMIScore_for_Soft_Classifier().expected_score_anonymous_raters(
            p, pd.DataFrame([['a'],['b']])), .15, delta=1e-15)

    def test_external_training_uses_item_identity(self):
        train = pd.DataFrame([['a','a'],['b','b']], index=['train-a','train-b'])
        evaluation = pd.DataFrame([['a','a']]*3, index=['eval-x','eval-y','eval-z'])
        run = AnalysisPipeline(evaluation, combiner=AnonymousBayesianCombiner(W=train),
                               scorer=AgreementScore(), allowable_labels=['a','b'],
                               max_K=1, num_bootstrap_item_samples=0, procs=1, verbosity=0)
        self.assertEqual(run.expert_power_curve.values[0], 1.)

    def test_reordered_training_excludes_matching_item_only(self):
        train=pd.DataFrame([['b','b'],['a','a']],index=['second','first'])
        evaluation=train.iloc[::-1]
        run=AnalysisPipeline(evaluation,combiner=AnonymousBayesianCombiner(W=train),
                             scorer=AgreementScore(),allowable_labels=['a','b'],max_K=1,
                             num_bootstrap_item_samples=0,procs=1,verbosity=0)
        # Each item's other training row has the opposite label.
        self.assertEqual(run.expert_power_curve.values[0],0.)

    def test_paper_mode_does_not_cap_unidentified_equivalence(self):
        W=pd.DataFrame({'human':['a','a','b','b']})
        predictions=pd.DataFrame({'oracle':[DiscretePrediction(y) for y in W.human]})
        run=AnalysisPipeline(W,classifier_predictions=predictions,combiner=AnonymousBayesianCombiner(),
                             scorer=AgreementScore(),allowable_labels=['a','b'],max_K=1,paper_mode=True,
                             num_bootstrap_item_samples=0,procs=1,verbosity=0)
        self.assertTrue(math.isnan(run.expert_survey_equivalences.df.iloc[0,0]))
        self.assertEqual(run.expert_power_curve.compute_one_equivalence_result(1.),
                         dict(status='above_range',estimate=None,bound=0.))

    def test_prepared_zero_probability_matches_scalar(self):
        from surveyequivalence._scoring import prepare_scores, score_prepared
        p=DiscreteDistributionPrediction(['a','b'],[1.,0.],extreme_cutoff=0)
        W=pd.DataFrame([['a','a','b']])
        scorer=CrossEntropyScore(num_ref_raters_per_virtual_rater=3)
        values,valid,mode=prepare_scores(scorer,[p],W,True)
        self.assertEqual(score_prepared(values,valid,np.array([0]),mode),0.)

    def test_paper_partitions_share_balanced_individual_references(self):
        W = pd.DataFrame([['a','a','b'],['a','b','b']]*6)
        predictions = pd.DataFrame({'judge':[DiscretePrediction('a')]*len(W)})
        run = AnalysisPipeline(W, classifier_predictions=predictions,
                               combiner=AnonymousBayesianCombiner(), scorer=AgreementScore(),
                               allowable_labels=['a','b'], max_K=3, max_rater_subsets=3,
                               num_bootstrap_item_samples=2, random_state=31,
                               procs=1, verbosity=0, paper_mode=True)
        self.assertEqual(run.classifier_scores.values['judge'], .5)
        for k, partitions in run.paper_partitions.items():
            refs = [r for b,r in partitions]
            self.assertEqual(sorted(refs), [0,1,2])
            self.assertEqual(len(set((b,r) for b,r in partitions)), len(partitions))
            expected = []
            for b,r in partitions:
                self.assertNotIn(r,b)
                preds = [AnonymousBayesianCombiner(extreme_cutoff=0).combine(
                    ['a','b'], [(j,W.iloc[i,j]) for j in b], W.to_numpy(), i) for i in range(len(W))]
                expected.append(sum(pred.value == W.iloc[i,r] for i,pred in enumerate(preds))/len(W))
            self.assertAlmostEqual(run.expert_power_curve.values[k], sum(expected)/len(expected), delta=2e-15)
        with tempfile.TemporaryDirectory() as directory:
            run.save(directory)
            saved = load_saved_pipeline(directory)
            self.assertTrue(saved.paper_mode)
            self.assertEqual(saved.paper_partitions, run.paper_partitions)
            pd.testing.assert_frame_equal(run.expert_power_curve.df, saved.expert_power_curve.df, check_exact=True)

    def test_paper_mode_rejects_variable_depth(self):
        with self.assertRaisesRegex(ValueError, 'complete'):
            AnalysisPipeline(pd.DataFrame([['a','b'],['b',None]]),
                             combiner=AnonymousBayesianCombiner(), scorer=AgreementScore(),
                             allowable_labels=['a','b'], paper_mode=True, procs=1, verbosity=0)

    def test_paper_mode_rejects_missing_classifier_predictions(self):
        W=pd.DataFrame([['a','b']]*4)
        for missing in (None, np.nan, pd.NA, DiscretePrediction(None)):
            with self.subTest(missing=repr(missing)), self.assertRaisesRegex(ValueError, 'complete classifier'):
                AnalysisPipeline(W,classifier_predictions=pd.DataFrame({'judge':[missing]*4}),
                                 combiner=AnonymousBayesianCombiner(),scorer=AgreementScore(),
                                 allowable_labels=['a','b'],paper_mode=True,procs=1,verbosity=0)


if __name__ == "__main__":
    unittest.main()
