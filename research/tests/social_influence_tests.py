"""Checks for finite observed-group sampling and censored panel equivalences."""
from itertools import combinations
import unittest

import numpy as np

from research.social_influence_equivalence import (
    first_integer,
    panel_mse,
    quantile_with_censoring,
)


def enumerated_mse(errors, k):
    """Uniformly enumerate panels, without assuming independent responses."""
    return np.mean([np.mean(panel) ** 2 for panel in combinations(errors, k)])


class SocialInfluenceTests(unittest.TestCase):
    def test_every_supported_size_matches_exhaustive_panels(self):
        # Include an extreme response, repeated responses, an exact consensus,
        # and a zero-mean pool; none requires an iid response model.
        for errors in ([0., -2., 3., 15.], [-2., -2., 1., 1., 1.],
                       [3., 3., 3.], [-4., -1., 2., 3.]):
            with self.subTest(errors=errors):
                ks = np.arange(1, len(errors) + 1)
                direct = [enumerated_mse(errors, k) for k in ks]
                np.testing.assert_allclose(panel_mse(errors, ks), direct,
                                           rtol=1e-13, atol=1e-13)

    def test_unequal_group_sizes_keep_distinct_finite_pool_corrections(self):
        groups = [[-3., 1., 6.], [2., 3., 5., 8., 12.]]
        ks = np.arange(1, 4)
        # Select a group with equal probability, then k distinct people within
        # that group. Pooling the eight people would be a different estimand.
        expected = [np.mean([enumerated_mse(g, k) for g in groups]) for k in ks]
        actual = np.mean([panel_mse(g, ks) for g in groups], axis=0)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
        self.assertFalse(np.allclose(actual, panel_mse(np.concatenate(groups), ks)))

    def test_shared_question_normalization_scales_squared_error(self):
        errors, scale = np.array([-4., 2., 6., 8.]), 2.5
        ks = [1, 2, 3, 4]
        np.testing.assert_allclose(panel_mse(errors / scale, ks),
                                   panel_mse(errors, ks) / scale ** 2,
                                   rtol=1e-13)

    def test_invalid_estimates_and_unsupported_sizes_are_rejected(self):
        bad_inputs = [
            ([], [1]), ([1.], [1]), ([[1., 2.]], [1]),
            ([1., np.nan], [1]), ([1., np.inf], [1]),
            ([1., 2., 3.], [0]), ([1., 2., 3.], [-1]),
            ([1., 2., 3.], [4]), ([1., 2., 3.], [1.5]),
            ([1., 2., 3.], [np.nan]), ([1., 2., 3.], [np.inf]),
            ([1., 2., 3.], [[1, 2]]),
        ]
        for errors, ks in bad_inputs:
            with self.subTest(errors=errors, ks=ks), self.assertRaises(ValueError):
                panel_mse(errors, ks)

    def test_integer_crossings_keep_unreached_targets_censored(self):
        curves = [[9., 4., 1.], [9., 8., 7.], [3., 2., 1.]]
        np.testing.assert_array_equal(first_integer(curves, [4., 6., 4.]),
                                      [2., np.inf, 1.])
        # Floating-point disagreement at an equal score must not move the
        # result to the following panel size.
        self.assertEqual(first_integer([9., 4. + 1e-14, 1.], 4.)[0], 2.)

    def test_censored_quantiles_do_not_drop_or_interpolate_infinite_draws(self):
        draws = np.r_[np.arange(1., 39.), np.inf, np.inf]
        self.assertEqual(quantile_with_censoring(draws, .025), 1.)
        self.assertEqual(quantile_with_censoring(draws, .95), 38.)
        self.assertTrue(np.isinf(quantile_with_censoring(draws, .975)))
        self.assertTrue(np.isinf(quantile_with_censoring([np.inf] * 5, .025)))

    def test_empirical_quantiles_retain_endpoints_and_input_order_independence(self):
        draws = [12., 1., 7., 4.]
        self.assertEqual(quantile_with_censoring(draws, 0.), 1.)
        self.assertEqual(quantile_with_censoring(draws, .5), 4.)
        self.assertEqual(quantile_with_censoring(draws, 1.), 12.)
        self.assertEqual(draws, [12., 1., 7., 4.])


if __name__ == '__main__':
    unittest.main()
