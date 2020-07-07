import numpy as np
import pandas as pd
import unittest
from surveyequivalence import Correlation, MeanCombiner, NumericPrediction

import unittest

class TestDiscreteDistributionSurveyEquivalence(unittest.TestCase):

    def setUp(self):
        self.datasets = self.make_test_datasets()

    def make_test_datasets(self):
        return []

    def test_mean_combiner(self):
        mean = MeanCombiner()
        pred = mean.combine(labels=[
            ('r1', 3),
            ('r2', 6),
            ('r3', 6)
        ])
        print(pred)
        self.assertEqual(pred.value, 5)

    def test_scoring_functions(self):
        predictions = [NumericPrediction(val) for val in [5, 6, 7, 8, 9]]
        ratings1 = [1, 2, 3, 4, 5]
        ratings2 = [5, 4, 3, 2, 1]
        ratings3 = [6, 5, 7, 8, 9]
        self.assertAlmostEqual(Correlation.score(predictions, ratings1), 1, places=3)
        self.assertAlmostEqual(Correlation.score(predictions, ratings2), -1, places=3)
        self.assertAlmostEqual(Correlation.score(predictions, ratings3), .9, places=3)

    def test_analysis_pipeline(self):
        for dataset in self.datasets:
            pass


if __name__ == '__main__':
    unittest.main()