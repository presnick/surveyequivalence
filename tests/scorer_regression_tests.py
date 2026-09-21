"""Consolidated regression cases and independent mathematical checks."""

import math
import numpy as np
import pandas as pd
import unittest
from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations
from math import log2, sqrt
from surveyequivalence import (
    scoring_functions as sf,
    AgreementScore,
    CrossEntropyScore,
    DiscreteDistributionPrediction,
)
from surveyequivalence.combiners import (
    DiscretePrediction,
    DiscreteDistributionPrediction,
    NumericPrediction,
)
from surveyequivalence.scoring_functions import (
    AgreementScore,
    CrossEntropyScore,
    DMIScore_for_Hard_Classifier,
    DMIScore_for_Soft_Classifier,
    Scorer_for_Hard_Classifier,
    PrecisionScore,
    RecallScore,
    F1Score,
    AUCScore,
    frac,
    frac_cache,
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


def enumerated_panel_distribution(labels, requested_size):
    """Uniform subsets of individual raters; uniform tie breaking."""
    size = min(requested_size, len(labels))
    panels = list(combinations(range(len(labels)), size))
    distribution = {label: Fraction(0) for label in labels}
    for panel in panels:
        counts = Counter(labels[i] for i in panel)
        winners = [label for label, count in counts.items() if count == max(counts.values())]
        for winner in winners:
            distribution[winner] += Fraction(1, len(panels) * len(winners))
    return distribution


def rational_determinant(matrix):
    result = Fraction(0)
    for order in permutations(range(len(matrix))):
        inversions = sum(order[i] > order[j] for i in range(len(order))
                         for j in range(i + 1, len(order)))
        term = Fraction((-1) ** inversions)
        for row, column in enumerate(order):
            term *= matrix[row][column]
        result += term
    return result


def confusion_metric(predicted, truth, metric, average):
    labels = sorted(set(predicted) | set(truth))
    rows = []
    for label in labels:
        tp = sum(p == label and y == label for p, y in zip(predicted, truth))
        fp = sum(p == label and y != label for p, y in zip(predicted, truth))
        fn = sum(p != label and y == label for p, y in zip(predicted, truth))
        rows.append((tp, fp, fn))

    def score(tp, fp, fn):
        numerator, denominator = {
            'precision': (tp, tp + fp), 'recall': (tp, tp + fn),
            'f1': (2 * tp, 2 * tp + fp + fn),
        }[metric]
        return Fraction(numerator, denominator) if denominator else Fraction(0)

    if average == 'micro':
        return score(*(sum(row[i] for row in rows) for i in range(3)))
    if average == 'macro':
        return sum(score(*row) for row in rows) / len(rows)
    return sum(score(*row) * (row[0] + row[2]) for row in rows) / len(truth)


class ScorerMathAuditTests(unittest.TestCase):
    def setUp(self):
        # The audit probe separately checks the global factorial-cache bug.
        self.previous_factorials = dict(sf.frac_cache)
        sf.frac_cache.clear()

    def tearDown(self):
        sf.frac_cache.clear()
        sf.frac_cache.update(self.previous_factorials)

    def test_binary_panel_scores_match_exhaustive_rater_subsets(self):
        prediction = DiscreteDistributionPrediction(['a', 'b'], [.73, .27])
        for n in range(1, 8):
            for positives in range(n + 1):
                row = ['a'] * positives + ['b'] * (n - positives)
                W = pd.DataFrame([row + [None, np.nan]])
                for size in range(1, n + 2):
                    with self.subTest(n=n, positives=positives, size=size):
                        reference = enumerated_panel_distribution(row, size)
                        agreement = sf.AgreementScore(num_ref_raters_per_virtual_rater=size)
                        actual = agreement.expected_score_anonymous_raters([DiscretePrediction('a')], W)
                        self.assertAlmostEqual(actual, float(reference.get('a', 0)), places=14)
                        entropy = sf.CrossEntropyScore(num_ref_raters_per_virtual_rater=size)
                        expected = sum(float(weight) * log2(prediction.label_probability(label))
                                       for label, weight in reference.items())
                        self.assertAlmostEqual(entropy.expected_score_anonymous_raters(
                            [prediction], W), expected, places=14)

    def test_multiclass_single_rater_expectations_are_item_weighted(self):
        W = pd.DataFrame([['a', 'a', 'b', None], ['b', 'c', None, None],
                          ['c', 'c', 'a', 'b']])
        hard = [DiscretePrediction(label) for label in ['a', 'c', 'b']]
        soft = [DiscreteDistributionPrediction(['a', 'b', 'c'], probabilities)
                for probabilities in [[.6, .3, .1], [.2, .3, .5], [.2, .6, .2]]]
        distributions = [dict(a=Fraction(2, 3), b=Fraction(1, 3)),
                         dict(b=Fraction(1, 2), c=Fraction(1, 2)),
                         dict(a=Fraction(1, 4), b=Fraction(1, 4), c=Fraction(1, 2))]
        self.assertAlmostEqual(sf.AgreementScore().expected_score_anonymous_raters(hard, W),
                               float((Fraction(2, 3) + Fraction(1, 2) + Fraction(1, 4)) / 3), places=14)
        expected = sum(sum(float(weight) * log2(pred.label_probability(label))
                           for label, weight in distribution.items())
                       for pred, distribution in zip(soft, distributions)) / 3
        self.assertAlmostEqual(sf.CrossEntropyScore().expected_score_anonymous_raters(soft, W),
                               expected, places=14)

    def test_precision_recall_f1_match_rational_confusion_counts(self):
        predicted = ['a', 'a', 'b', 'c', 'b', 'c', 'a']
        truth = ['a', 'b', 'b', 'c', 'c', 'a', 'a']
        for metric, scorer in [('precision', sf.PrecisionScore), ('recall', sf.RecallScore),
                               ('f1', sf.F1Score)]:
            for average in ['micro', 'macro', 'weighted']:
                with self.subTest(metric=metric, average=average):
                    self.assertAlmostEqual(scorer.score(
                        [DiscretePrediction(value) for value in predicted] + [None, DiscretePrediction('b')],
                        truth + ['c', None], average=average),
                        float(confusion_metric(predicted, truth, metric, average)), places=14)

    def test_non_anonymous_agreement_averages_raters_not_cells(self):
        W = pd.DataFrame([['a', 'b'], ['a', None], ['a', None]])
        predictions = [DiscretePrediction('a')] * 3
        self.assertEqual(sf.AgreementScore().expected_score_non_anonymous_raters(predictions, W), .5)

    def test_pearson_matches_centered_rational_products(self):
        x, y = [0, 1, 3, 5], [4, 1, 2, 0]
        mx, my = Fraction(sum(x), len(x)), Fraction(sum(y), len(y))
        covariance = sum((a - mx) * (b - my) for a, b in zip(x, y))
        vx = sum((a - mx) ** 2 for a in x)
        vy = sum((b - my) ** 2 for b in y)
        actual = sf.Correlation.score([NumericPrediction(a) for a in x] + [None], y + [99])
        self.assertAlmostEqual(actual, float(covariance) / sqrt(float(vx * vy)), places=14)

    def test_hard_and_soft_dmi_match_rational_determinants(self):
        truth = ['a', 'b', 'c', 'a', 'b', 'c']
        hard_values = ['a', 'b', 'c', 'b', 'b', 'a']
        vocabulary = ['a', 'b', 'c']
        hard_matrix = [[Fraction(0) for _ in vocabulary] for _ in vocabulary]
        for pred, label in zip(hard_values, truth):
            hard_matrix[vocabulary.index(pred)][vocabulary.index(label)] += Fraction(1, len(truth))
        expected = abs(rational_determinant(hard_matrix))
        self.assertAlmostEqual(sf.DMIScore_for_Hard_Classifier.score(
            [DiscretePrediction(value) for value in hard_values], truth), float(expected), places=14)

        vectors = [[Fraction(6, 10), Fraction(3, 10), Fraction(1, 10)],
                   [Fraction(2, 10), Fraction(7, 10), Fraction(1, 10)],
                   [Fraction(1, 10), Fraction(2, 10), Fraction(7, 10)]] * 2
        soft_matrix = [[Fraction(0) for _ in vocabulary] for _ in vocabulary]
        for vector, label in zip(vectors, truth):
            for i, weight in enumerate(vector):
                soft_matrix[i][vocabulary.index(label)] += weight / len(truth)
        predictions = [DiscreteDistributionPrediction(vocabulary, list(map(float, vector))) for vector in vectors]
        self.assertAlmostEqual(sf.DMIScore_for_Soft_Classifier.score(predictions, truth),
                               float(abs(rational_determinant(soft_matrix))), places=14)

    def test_binary_dmi_of_mean_joint_matches_enumerated_panel_distribution(self):
        # This verifies the current population-joint statistic, not E[abs(det)].
        rows = [['a', 'a', 'b'], ['a', 'b'], ['b', 'b', 'b']]
        hard_values = ['a', 'b', 'b']
        for size in [1, 2, 3, 4]:
            matrix = [[Fraction(0), Fraction(0)] for _ in range(2)]
            for pred, row in zip(hard_values, rows):
                distribution = enumerated_panel_distribution(row, size)
                for label, probability in distribution.items():
                    matrix['ab'.index(pred)]['ab'.index(label)] += probability / len(rows)
            actual = sf.DMIScore_for_Hard_Classifier(
                num_ref_raters_per_virtual_rater=size).expected_score_anonymous_raters(
                    [DiscretePrediction(value) for value in hard_values], pd.DataFrame(rows))
            self.assertAlmostEqual(actual, float(abs(rational_determinant(matrix))), places=14)


class CombinatorialRegressionTests(unittest.TestCase):
    def setUp(self):
        frac_cache.clear()

    def tearDown(self):
        frac_cache.clear()

    def test_numpy_factorials_use_unbounded_integer_arithmetic(self):
        self.assertEqual(frac(np.int64(21)), math.factorial(21))
        self.assertIs(type(frac(np.int64(21))), int)

    def test_wide_single_reference_agreement(self):
        ratings = pd.DataFrame([['a'] * 14 + ['b'] * 7])
        actual = AgreementScore().expected_score_anonymous_raters([DiscretePrediction('a')], ratings)
        self.assertEqual(actual, 14 / 21)

    def test_agreement_cannot_contaminate_cross_entropy(self):
        ratings = pd.DataFrame([['a'] * 14 + ['b'] * 7])
        predictions = [DiscreteDistributionPrediction(['a', 'b'], [.8, .2])]
        expected = (14 / 21) * math.log2(.8) + (7 / 21) * math.log2(.2)
        first = CrossEntropyScore().expected_score_anonymous_raters(predictions, ratings)
        frac_cache.clear()
        AgreementScore().expected_score_anonymous_raters([DiscretePrediction('a')], ratings)
        second = CrossEntropyScore().expected_score_anonymous_raters(predictions, ratings)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_legacy_numpy_cache_value_is_recomputed(self):
        frac_cache[21] = np.int64(-1)
        self.assertEqual(frac(np.int64(21)), math.factorial(21))


if __name__ == "__main__":
    unittest.main()
