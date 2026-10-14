"""Independent checks for exact finite-pool classification panel curves."""
import itertools
import unittest
import numpy as np
from research.sports_survey_equivalence import exact_curve, first_crossing


class SportsEquivalenceTests(unittest.TestCase):
    def test_matches_exhaustive_subsets_including_zero_weights(self):
        for weights in [[1,-1,1,-1,1], [5,-2,1,-4,2], [0,100,-23,17,-100,0]]:
            expected = []
            for k in range(1,len(weights)+1):
                scores = [float(sum(c)>0)+.5*(sum(c)==0)
                          for c in itertools.combinations(weights,k)]
                expected.append(np.mean(scores))
            np.testing.assert_allclose(exact_curve(weights,len(weights)),expected,atol=1e-14)

    def test_reversing_winner_complements_accuracy(self):
        weights = [100,-50,0,34,-1,25]
        np.testing.assert_allclose(exact_curve(weights,6)+exact_curve([-w for w in weights],6),1)

    def test_majority_odd_even_equivalence(self):
        curve = exact_curve([1]*7+[-1]*4)
        np.testing.assert_allclose(curve[::2],curve[1::2],atol=1e-14)
        self.assertEqual(first_crossing(curve,curve[-1]),9)

    def test_first_crossing_preserves_reversals_and_censoring(self):
        curves = np.array([[.7,.9,.6],[.5,.6,.7],[.8,.8,.8]])
        np.testing.assert_array_equal(first_crossing(curves,.8),[2,4,1])


if __name__ == '__main__':
    unittest.main()
