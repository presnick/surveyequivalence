"""Boundary cases for interpolated empirical rater equivalence."""
import unittest

import numpy as np

from research.panel_power import equivalence_label, first_equivalent_size


class PanelPowerTests(unittest.TestCase):
    def test_fractional_panel_is_a_mixture_of_neighboring_scores(self):
        # A 25% mixture of three-person panels and 75% two-person panels.
        self.assertAlmostEqual(first_equivalent_size([1, 2, 3], [.2, .4, .8], .5), 2.25)

    def test_first_crossing_of_nonmonotone_curve(self):
        self.assertAlmostEqual(first_equivalent_size([1, 2, 3, 4], [.1, .7, .3, .9], .4), 1.5)

    def test_plateau_uses_left_endpoint_including_float_roundoff(self):
        self.assertEqual(first_equivalent_size([1, 2, 3, 4], [.2, .5 - 1e-15, .5, .8], .5), 2)

    def test_unsupported_sizes_are_bounds(self):
        ks = [1, 2, 3]
        low = first_equivalent_size(ks, [.4, .5, .6], .3)
        high = first_equivalent_size(ks, [.4, .5, .6], .8)
        self.assertEqual(equivalence_label(low, ks), '≤1')
        self.assertTrue(np.isinf(high))
        self.assertEqual(equivalence_label(high, ks), '>3')

    def test_loss_negation_and_positive_scaling_preserve_crossing(self):
        loss, target = np.array([9., 4., 1.]), 3.
        self.assertAlmostEqual(first_equivalent_size([1, 2, 3], -loss, -target), 2 + 1/3)
        self.assertAlmostEqual(first_equivalent_size([1, 2, 3], -loss/1000, -target/1000), 2 + 1/3)

    def test_bad_scores_and_size_order_are_rejected(self):
        for ks, scores, target in [([2, 1], [.1, .2], .2), ([1, 2], [.1, np.nan], .2),
                                   ([1, 2], [.1], .2), ([], [], .2), ([1], [.1], np.inf)]:
            with self.subTest(ks=ks, scores=scores, target=target), self.assertRaises(ValueError):
                first_equivalent_size(ks, scores, target)


if __name__ == '__main__':
    unittest.main()
