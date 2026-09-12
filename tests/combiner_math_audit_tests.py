"""Independent finite-sample probability checks for the built-in combiners.

The oracle enumerates distinct rater positions. It does not use the library's
counts, falling factorials, cache helpers, or missing-value predicate.
"""

import unittest
from fractions import Fraction
from itertools import combinations_with_replacement, permutations, product
from unittest.mock import patch

import numpy as np

from surveyequivalence._data import PreparedRatings
from surveyequivalence.combiners import (
    AnonymousBayesianCombiner, DiscreteDistributionPrediction,
    FrequencyCombiner, PluralityVote,
)


def ordered_sequence_probability(row, sequence):
    """Chance of a specific label sequence from distinct, uniform positions."""
    observed = [value for value in row if value is not None]
    if len(observed) < len(sequence):
        return None
    draws = list(permutations(range(len(observed)), len(sequence)))
    matches = sum(tuple(observed[i] for i in draw) == tuple(sequence) for draw in draws)
    return Fraction(matches, len(draws))


def conditional_next_label(matrix, vocabulary, sequence, held_out=None):
    """Uniform eligible item, then ordered draws; condition on the prefix."""
    masses = []
    for candidate in vocabulary:
        item_masses = [ordered_sequence_probability(row, tuple(sequence) + (candidate,))
                       for i, row in enumerate(matrix) if i != held_out]
        item_masses = [mass for mass in item_masses if mass is not None]
        if not item_masses:
            return None
        masses.append(sum(item_masses, Fraction(0)) / len(item_masses))
    total = sum(masses, Fraction(0))
    return [mass / total for mass in masses] if total else None


def regularized_probabilities(probabilities, cutoff=Fraction(1, 50)):
    clipped = [max(cutoff, min(1 - cutoff, probability)) for probability in probabilities]
    total = sum(clipped, Fraction(0))
    return [float(probability / total) for probability in clipped]


class CombinerMathAuditTests(unittest.TestCase):
    def assert_matches_oracle(self, matrix, vocabulary, sequences):
        array = np.array(matrix, dtype=object)
        prepared = AnonymousBayesianCombiner()
        prepared._prepare(array, vocabulary, PreparedRatings(array, vocabulary))
        scalar = AnonymousBayesianCombiner()
        for held_out in [None] + list(range(len(matrix))):
            for sequence in sequences:
                expected = conditional_next_label(matrix, vocabulary, sequence, held_out)
                for combiner in (scalar, prepared):
                    prediction = combiner.combine(
                        vocabulary, list(enumerate(sequence)), W=array, item_id=held_out)
                    if expected is None:
                        self.assertIsNone(prediction, (matrix, sequence, held_out))
                    else:
                        np.testing.assert_allclose(
                            prediction.probabilities, regularized_probabilities(expected),
                            rtol=0, atol=2e-15,
                            err_msg=repr((matrix, vocabulary, sequence, held_out)))

    def test_per_item_probability_exhausts_binary_rows_and_ordered_sequences(self):
        combiner = AnonymousBayesianCombiner()
        vocabulary = ['a', 'b']
        sequences = [sequence for size in range(5) for sequence in product(vocabulary, repeat=size)]
        for row in product(['a', 'b', None], repeat=4):
            for sequence in sequences:
                expected = ordered_sequence_probability(row, sequence)
                counts = np.array([sequence.count(label) for label in vocabulary])
                actual, eligible = combiner.probabilityOneItem(
                    counts, np.array(row, dtype=object), vocabulary)
                self.assertEqual(eligible, int(expected is not None))
                self.assertEqual(actual, float(expected) if expected is not None else 0)

    def test_binary_bayesian_conditionals_exhaust_tiny_sparse_matrices(self):
        rows = list(product(['a', 'b', None], repeat=2))
        for matrix in combinations_with_replacement(rows, 2):
            self.assert_matches_oracle(matrix, ['a', 'b'], [(), ('a',), ('b',)])

    def test_multiclass_conditionals_cover_rectangular_and_sparse_support(self):
        matrices = [
            [['a', 'a', 'b'], ['a', 'b', 'c']],
            [['a', 'a', 'b'], ['a', 'b', 'c'], ['c', 'c', 'b'], ['a', None, None]],
            [['a', 'b', None, None], ['a', 'c', 'c', 'c']],
        ]
        for matrix in matrices:
            for vocabulary in (['a', 'b', 'c'], ['c', 'b', 'a']):
                sequences = [sequence for size in range(4)
                             for sequence in product(vocabulary, repeat=size)]
                self.assert_matches_oracle(matrix, vocabulary, sequences)

    def test_prior_weighting_is_explicit_for_unequal_rating_counts(self):
        matrix = np.array([['a', 'a', 'a'], ['b', None, None]], dtype=object)
        # ABC first chooses an item, whereas Frequency chooses a rating.
        self.assertEqual(AnonymousBayesianCombiner(W=matrix).combine(
            ['a', 'b'], []).probabilities, [.5, .5])
        self.assertEqual(FrequencyCombiner().combine(
            ['a', 'b'], [], W=matrix).probabilities, [.75, .25])

    def test_clipping_then_normalization_matches_declared_rule(self):
        binary = DiscreteDistributionPrediction(['a', 'b'], [1, 0])
        self.assertEqual(binary.probabilities, [.98, .02])
        multiclass = DiscreteDistributionPrediction(['a', 'b', 'c'], [1, 0, 0])
        np.testing.assert_allclose(multiclass.probabilities,
                                   [49 / 51, 1 / 51, 1 / 51], rtol=0, atol=2e-16)
        unnormalized = DiscreteDistributionPrediction(
            ['a', 'b', 'c'], [1, 0, 0], extreme_cutoff=.1, normalize=False)
        self.assertEqual(unnormalized.probabilities, [.9, .1, .1])

    def test_prepared_counts_preserve_numeric_and_declared_empty_labels(self):
        matrix = np.array([[0, '', 1], [1, None, 0]], dtype=object)
        prepared = PreparedRatings(matrix, [0, '', 1])
        np.testing.assert_array_equal(prepared.counts, [[1, 1, 1], [1, 0, 1]])
        np.testing.assert_array_equal(prepared.totals, [3, 2])
        np.testing.assert_array_equal(prepared.frequencies, [2, 1, 2])
        for buffer in (prepared.counts, prepared.codes, prepared.valid,
                       prepared.totals, prepared.frequencies):
            self.assertFalse(buffer.flags.writeable)

    def test_plurality_draws_uniform_winners_on_each_call(self):
        combiner = PluralityVote()
        observations = list(enumerate(['a', 'b', 'a', 'b', 'c', None]))
        # random.choice is uniform over its input; verify exactly the tied
        # winners are supplied and that two calls make distinct draws.
        with patch('surveyequivalence.combiners.random.choice', side_effect=['a', 'b']) as choice:
            self.assertEqual(combiner.combine(['a', 'b', 'c'], observations).value, 'a')
            self.assertEqual(combiner.combine(['a', 'b', 'c'], observations).value, 'b')
            self.assertEqual(choice.call_count, 2)
            for call in choice.call_args_list:
                self.assertEqual(call.args[0], ['a', 'b'])
        with patch('surveyequivalence.combiners.random.choice', return_value='c') as choice:
            self.assertEqual(combiner.combine(['a', 'b', 'c'], [(0, None)]).value, 'c')
            choice.assert_called_once_with(['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main()
