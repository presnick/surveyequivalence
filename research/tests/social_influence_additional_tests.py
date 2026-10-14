"""Checks for geometric-panel scoring and study-specific range censoring."""
from itertools import combinations
import unittest

import numpy as np

from research.social_influence_additional import log_errors, panel_mse, summarize


class AdditionalSocialInfluenceTests(unittest.TestCase):
    def test_log_score_matches_actual_geometric_panel_predictions(self):
        guesses, truth = np.array([10., 25., 90., 800.]), 50.
        expected = []
        for k in range(1, len(guesses) + 1):
            losses = []
            for panel in combinations(guesses, k):
                prediction = np.prod(panel) ** (1 / k)
                losses.append(np.log(prediction / truth) ** 2)
            expected.append(np.mean(losses))
        np.testing.assert_allclose(panel_mse(log_errors(guesses, truth), [1, 2, 3, 4]), expected, rtol=1e-13)

    def test_log_score_rejects_zero_negative_and_nonfinite_values(self):
        for guesses, truth in [([0., 5.], 2.), ([-1., 5.], 2.), ([np.inf], 2.),
                               ([np.nan], 2.), ([1., 5.], 0.), ([1.], -2.), ([1.], np.nan)]:
            with self.subTest(guesses=guesses, truth=truth), self.assertRaises(ValueError):
                log_errors(guesses, truth)

    def test_unattained_crossings_use_each_studys_supported_range(self):
        for maximum in [12, 100]:
            ks = np.arange(1, maximum + 1)
            mean = np.stack([1 / ks, 2 + 1 / ks])
            boot = np.stack([mean, mean * 2, mean * .5])
            row, _, saved = summarize('fixture', 'condition', ks, mean, boot)
            self.assertEqual(row['equivalent_k'], f'>{maximum}')
            self.assertEqual(row['equivalent_integer_95_high'], f'>{maximum}')
            self.assertEqual(row['bootstrap_fraction_above_range'], 1.)
            self.assertTrue(np.isinf(saved['equivalent_fractional']).all())

    def test_single_report_boundary_is_not_an_invented_fractional_crossing(self):
        ks = np.arange(1, 13)
        mean = np.stack([1 / ks, .05 / ks])
        row, _, _ = summarize('fixture', 'condition', ks, mean, np.stack([mean] * 3))
        self.assertEqual(row['equivalent_k'], 1.)
        self.assertTrue(row['equivalent_at_lower_boundary'])
        self.assertEqual(row['equivalent_integer_k'], 1.)


if __name__ == '__main__':
    unittest.main()
