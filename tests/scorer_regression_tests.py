"""Independent sparse-data expectations for scorer correctness fixes."""

import unittest
from math import log2

import numpy as np
import pandas as pd

from surveyequivalence.combiners import DiscretePrediction, DiscreteDistributionPrediction
from surveyequivalence.scoring_functions import (
    AgreementScore, CrossEntropyScore, DMIScore_for_Hard_Classifier,
    DMIScore_for_Soft_Classifier, Scorer_for_Hard_Classifier,
    PrecisionScore, RecallScore, F1Score, AUCScore,
)


class SampledAgreement(Scorer_for_Hard_Classifier):
    score = staticmethod(AgreementScore.score)


class ScorerRegressionTests(unittest.TestCase):
    def setUp(self):
        self.hard = [DiscretePrediction('a'), DiscretePrediction('b')]
        self.soft = [DiscreteDistributionPrediction(['a', 'b'], [.7, .3]),
                     DiscreteDistributionPrediction(['a', 'b'], [.3, .7])]

    def test_missing_labels_are_excluded_from_dmi_vocabulary(self):
        for missing in (None, np.nan, pd.NA, ''):
            with self.subTest(missing=repr(missing)):
                ratings = pd.DataFrame([['a', missing], ['b', missing],
                                        [missing, missing]])
                hard = self.hard + [DiscretePrediction('unused')]
                soft = self.soft + [self.soft[0]]
                # The usable hard joint distribution is diag(.5, .5).
                self.assertEqual(DMIScore_for_Hard_Classifier().expected_score_anonymous_raters(
                    hard, ratings), .25)
                # The soft joint distribution is [[.35, .15], [.15, .35]].
                self.assertAlmostEqual(DMIScore_for_Soft_Classifier().expected_score_anonymous_raters(
                    soft, ratings), .1, places=15)
                labels = ['a', 'b', missing]
                self.assertEqual(DMIScore_for_Hard_Classifier.score(hard, labels), .25)
                self.assertAlmostEqual(DMIScore_for_Soft_Classifier.score(soft, labels), .1, places=15)

    def test_missing_predictions_do_not_add_dmi_labels(self):
        ratings = pd.DataFrame([['outside', 'outside'], ['a', None], ['b', None]])
        self.assertEqual(DMIScore_for_Hard_Classifier().expected_score_anonymous_raters(
            [None] + self.hard, ratings), .25)
        self.assertAlmostEqual(DMIScore_for_Soft_Classifier().expected_score_anonymous_raters(
            [None] + self.soft, ratings), .1, places=15)
        self.assertEqual(DMIScore_for_Hard_Classifier.score(
            [None] + self.hard, ['outside', 'a', 'b']), .25)
        self.assertAlmostEqual(DMIScore_for_Soft_Classifier.score(
            [None] + self.soft, ['outside', 'a', 'b']), .1, places=15)

    def test_reference_panels_are_capped_per_row(self):
        ratings = pd.DataFrame([['a', None, np.nan], ['b', 'a', None],
                                [None, None, pd.NA]])
        hard = self.hard + [DiscretePrediction('unused')]
        scorer = AgreementScore(num_ref_raters_per_virtual_rater=3)
        self.assertEqual(scorer.expected_score_anonymous_raters(hard, ratings), .75)
        soft = [DiscreteDistributionPrediction(['a', 'b'], [.8, .2]),
                DiscreteDistributionPrediction(['a', 'b'], [.4, .6]), None]
        expected = (log2(.8) + (.5 * log2(.6) + .5 * log2(.4))) / 2
        score = CrossEntropyScore(num_ref_raters_per_virtual_rater=3)
        self.assertEqual(score.expected_score_anonymous_raters(soft, ratings), expected)
        # With one observed label per row, a requested panel of three is size one.
        ratings = pd.DataFrame([['a', None, None], ['b', None, None]])
        self.assertEqual(DMIScore_for_Hard_Classifier(
            num_ref_raters_per_virtual_rater=3).expected_score_anonymous_raters(self.hard, ratings), .25)
        self.assertAlmostEqual(DMIScore_for_Soft_Classifier(
            num_ref_raters_per_virtual_rater=3).expected_score_anonymous_raters(self.soft, ratings), .1, places=15)

    def test_sampled_empty_rows_keep_prediction_alignment(self):
        ratings = pd.DataFrame([['a', None], [None, pd.NA], ['b', 'b'], ['a', 'b']])
        predictions = [self.hard[0], DiscretePrediction('wrong'), self.hard[1], None]
        # Every sampled usable row is unanimous; no random tolerance is needed.
        self.assertEqual(SampledAgreement(num_virtual_raters=5).expected_score_anonymous_raters(
            predictions, ratings), 1)

    def test_missing_pairs_are_excluded_by_agreement_and_cross_entropy(self):
        labels = ['a', None, np.nan, pd.NA, '', 'b']
        hard = [self.hard[0]] * 5 + [None]
        soft = [self.soft[0]] * 5 + [None]
        self.assertEqual(AgreementScore.score(hard, labels), 1)
        self.assertEqual(CrossEntropyScore.score(soft, labels), log2(.7))

    def test_no_usable_labels_or_predictions_returns_none(self):
        ratings = pd.DataFrame([[None, np.nan], [pd.NA, '']])
        for scorer, predictions in (
            (AgreementScore(), self.hard), (CrossEntropyScore(), self.soft),
            (DMIScore_for_Hard_Classifier(), self.hard),
            (DMIScore_for_Soft_Classifier(), self.soft),
            (SampledAgreement(), self.hard),
        ):
            with self.subTest(scorer=type(scorer).__name__):
                self.assertIsNone(scorer.expected_score_anonymous_raters(predictions, ratings))
                self.assertIsNone(scorer.expected_score_anonymous_raters(
                    [None, None], pd.DataFrame([['a'], ['b']])))
                self.assertIsNone(scorer.score(predictions, [None, pd.NA]))
        for scorer in (PrecisionScore, RecallScore, F1Score, AUCScore):
            self.assertIsNone(scorer.score(self.soft, [None, pd.NA]))

    def test_declared_empty_string_and_numeric_zero_are_valid_labels(self):
        labels = pd.Series(['', 'a'])
        labels.attrs['allowable_labels'] = ['', 'a']
        self.assertEqual(AgreementScore.score(
            [DiscretePrediction('a'), DiscretePrediction('a')], labels), .5)
        ratings = labels.to_frame()
        self.assertEqual(AgreementScore().expected_score_anonymous_raters(
            [DiscretePrediction('a'), DiscretePrediction('a')], ratings), .5)
        self.assertEqual(AgreementScore.score([DiscretePrediction('')], ['']), 1)
        self.assertEqual(AgreementScore.score([DiscretePrediction(0)], [0]), 1)


if __name__ == '__main__':
    unittest.main()
