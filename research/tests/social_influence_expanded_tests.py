"""Regression checks for decimal and signed targets in partisan experiments."""
from itertools import combinations
import unittest

import numpy as np

from research.social_influence_expanded import partisan_errors, panel_mse


class ExpandedSocialInfluenceTests(unittest.TestCase):
    def test_decimal_truth_is_not_truncated(self):
        np.testing.assert_allclose(partisan_errors([4.9, 9.8], 4.9, 1), [0, np.log(2)])
        np.testing.assert_allclose(partisan_errors([-27.8, -13.9], -27.8, 2), [0, -.5])

    def test_signed_relative_errors_match_actual_arithmetic_panels(self):
        estimates, truth = np.array([-70., -27.8, 0., 14.5, 82.]), -27.8
        exact = []
        for k in range(1, len(estimates) + 1):
            exact.append(np.mean([((np.mean(panel) - truth) / truth)**2
                                  for panel in combinations(estimates, k)]))
        calculated = panel_mse(partisan_errors(estimates, truth, 2), np.arange(1, 6))
        np.testing.assert_allclose(calculated, exact, rtol=1e-13)

    def test_positive_log_targets_match_actual_geometric_panels(self):
        estimates, truth = np.array([2., 4.9, 11., 22.]), 4.9
        exact = [np.mean([np.log(np.prod(panel)**(1/k) / truth)**2
                          for panel in combinations(estimates, k)]) for k in range(1, 5)]
        np.testing.assert_allclose(panel_mse(partisan_errors(estimates, truth, 1), np.arange(1, 5)),
                                   exact, rtol=1e-13)

    def test_invalid_scoring_inputs_are_rejected(self):
        for estimates, truth, experiment in [([0], 1, 1), ([-2], 1, 1), ([1], -1, 1),
                                             ([1], 0, 2), ([np.inf], 1, 2), ([1], np.nan, 2),
                                             ([1], 2, 3)]:
            with self.subTest(estimates=estimates, truth=truth, experiment=experiment):
                with self.assertRaises(ValueError):
                    partisan_errors(estimates, truth, experiment)


if __name__ == '__main__':
    unittest.main()
