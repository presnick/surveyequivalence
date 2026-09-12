"""Independent mathematics for the supported finite, increasing-curve domain.

Known plateau/missing-data defects are reproduced by statistics_audit.py.
These tests do not claim that those separate diagnostics pass.
"""

from fractions import Fraction
from itertools import combinations
import unittest

import pandas as pd
from surveyequivalence.equivalence import ClassifierResults, PowerCurve


class StatisticsMathAuditTests(unittest.TestCase):
    def test_increasing_curves_match_rational_linear_interpolation(self):
        for ordinates in combinations(range(9), 4):
            curve = PowerCurve(df=pd.DataFrame([[y / 8 for y in ordinates]]))
            for target in range(17):
                score = Fraction(target, 16)
                values = [Fraction(y, 8) for y in ordinates]
                if score <= values[0]:
                    expected = Fraction(0)
                elif score >= values[-1]:
                    # Historical endpoint saturation is checked explicitly;
                    # it is not evidence for an observed crossing above range.
                    expected = Fraction(3)
                else:
                    upper = next(k for k, value in enumerate(values) if value >= score)
                    expected = upper - 1 + (score - values[upper - 1]) / (values[upper] - values[upper - 1])
                with self.subTest(ordinates=ordinates, target=target):
                    self.assertAlmostEqual(curve.compute_one_equivalence(float(score)),
                                           float(expected), delta=1e-14)

    def test_finite_reliability_matches_enumerated_wins(self):
        from itertools import product
        for left_values in product((0., 1.), repeat=3):
            for right_values in product((0., 1.), repeat=3):
                left = PowerCurve(df=pd.DataFrame({1: left_values}))
                right = PowerCurve(df=pd.DataFrame({1: right_values}))
                expected = sum(a > b for a, b in zip(left_values, right_values)) / 3
                self.assertEqual(left.reliability_of_difference(right), expected)
                self.assertEqual(left.reliability_of_beating_classifier(right), expected)

    def test_performance_ratio_is_gain_relative_to_baseline(self):
        for baseline, reference in combinations(range(5), 2):
            curve = PowerCurve(df=pd.DataFrame({0: [baseline / 4], 1: [reference / 4]}))
            for target in range(5):
                classifiers = ClassifierResults(df=pd.DataFrame({'c': [target / 4]}))
                expected = Fraction(target - baseline, reference - baseline)
                self.assertEqual(curve.compute_performance_ratio(classifiers, 1).iloc[0, 0], float(expected))
