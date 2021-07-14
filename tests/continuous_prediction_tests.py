import numpy as np

from surveyequivalence import Correlation, MeanCombiner, NumericPrediction, DiscreteState

import unittest

class TestDiscreteDistributionSurveyEquivalence(unittest.TestCase):

    def test_mean_combiner(self):
        mean = MeanCombiner()
        pred = mean.combine(labels=[
            ('r1', 3),
            ('r2', 6),
            ('r3', 6)
        ])
        self.assertEqual(pred.value, 5)

    def test_scoring_functions(self):

        def all_one_ref_rater(label):
            return DiscreteState(state_name='',
                                 labels=[label],
                                 probabilities=[1])

        predictions = [NumericPrediction(val) for val in [5, 6, 7, 8, 9]]
        self.assertAlmostEqual(Correlation.score(predictions, np.array([1, 2, 3, 4, 5])), 1, places=3)
        self.assertAlmostEqual(Correlation.score(predictions, np.array([5, 4, 3, 2, 1])), -1, places=3)
        self.assertAlmostEqual(Correlation.score(predictions, np.array([6, 5, 7, 8, 9])), .9, places=3)


if __name__ == '__main__':
    unittest.main()