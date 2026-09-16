"""Independent regression cases for NumPy factorial overflow and cache order."""
import math
import unittest

import numpy as np
import pandas as pd

from surveyequivalence import AgreementScore, CrossEntropyScore, DiscreteDistributionPrediction
from surveyequivalence.combiners import DiscretePrediction
from surveyequivalence.scoring_functions import frac, frac_cache


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
