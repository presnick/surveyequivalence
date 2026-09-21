"""Consolidated regression cases and independent mathematical checks."""

import numpy as np
import pandas as pd
import unittest
from fractions import Fraction
from itertools import combinations_with_replacement, permutations, product
from surveyequivalence._data import as_rating_array, is_missing, PreparedRatings
from surveyequivalence.combiners import (
    AnonymousBayesianCombiner,
    FrequencyCombiner,
    MeanCombiner,
    PluralityVote,
    DiscreteDistributionPrediction,
)
from unittest.mock import patch


class TestRatingData(unittest.TestCase):
    def test_scalar_missing_values_preserve_zero_and_declared_empty_label(self):
        for value in [None, np.nan, pd.NA, ""]:
            with self.subTest(value=value):
                self.assertTrue(is_missing(value))
        for value in [0, False, "0", "pos"]:
            with self.subTest(value=value):
                self.assertFalse(is_missing(value))
        self.assertFalse(is_missing("", ["", "pos"]))

    def test_matrix_orientation_and_shape_validation(self):
        for shape in [(2, 5), (5, 2), (1, 3), (3, 1), (0, 3), (3, 0)]:
            with self.subTest(shape=shape):
                self.assertEqual(as_rating_array(np.empty(shape)).shape, shape)
        with self.assertRaisesRegex(ValueError, "two dimensions"):
            as_rating_array(["pos", "neg"])
        with self.assertRaisesRegex(ValueError, "pad uneven rows"):
            as_rating_array([["pos"], ["pos", "neg"]])


class TestCombinerMissingData(unittest.TestCase):
    def test_frequency_prior_ignores_every_padding_representation(self):
        W = np.array([["p", "p", None], ["n", np.nan, pd.NA],
                      ["", "", ""]], dtype=object)
        prediction = FrequencyCombiner().combine(["p", "n"], [], W=W)
        self.assertEqual(prediction.probabilities, [2 / 3, 1 / 3])

    def test_frequency_ignores_adjacent_missing_observations(self):
        observations = list(enumerate([None, None, np.nan, pd.NA, "", "p", "n", "p"]))
        prediction = FrequencyCombiner().combine(["p", "n"], observations)
        self.assertEqual(prediction.probabilities, [2 / 3, 1 / 3])

    def test_frequency_no_support_returns_none(self):
        self.assertIsNone(FrequencyCombiner().combine(
            ["p", "n"], [], W=np.array([[None, np.nan]], dtype=object)))

    def test_frequency_numeric_zero_and_declared_empty_label(self):
        for vocabulary in [[0, 1], ["", "p"]]:
            with self.subTest(vocabulary=vocabulary):
                W = np.array([[vocabulary[0], vocabulary[0], vocabulary[1]]], dtype=object)
                prediction = FrequencyCombiner().combine(vocabulary, [], W=W)
                self.assertEqual(prediction.probabilities, [2 / 3, 1 / 3])

    def test_simple_combiners_drop_padding_without_dropping_zero(self):
        observations = list(enumerate([None, np.nan, pd.NA, "", 0, 0, 3]))
        self.assertEqual(MeanCombiner().combine(labels=observations).value, 1)
        self.assertEqual(PluralityVote().combine([0, 3], observations).value, 0)


class TestBayesianCombinerRegression(unittest.TestCase):
    def test_rectangular_missing_matrices_have_hand_computed_prediction(self):
        # P(p,p) = (1 + 0 + 0)/3; P(p,n) = (0 + 1/2 + 0)/3.
        for missing in [None, np.nan, pd.NA, ""]:
            base = [["p", "p", "p"], ["p", "n", missing], ["n", "n", "n"]]
            matrices = [np.array(base * 2, dtype=object),
                        np.array([row + [missing] * 3 for row in base], dtype=object)]
            for W in matrices:
                with self.subTest(missing=missing, shape=W.shape):
                    for cached in [False, True]:
                        combiner = AnonymousBayesianCombiner(W=W if cached else None)
                        prediction = combiner.combine(["p", "n"], [(0, "p")], W=W)
                        self.assertEqual(prediction.probabilities, [2 / 3, 1 / 3])

    def test_no_item_id_averages_items_without_exclusion(self):
        W = np.array([["p", "p"], ["n", None]], dtype=object)
        combiner = AnonymousBayesianCombiner(W=W)
        self.assertEqual(combiner.combine(["p", "n"], []).probabilities, [.5, .5])
        self.assertEqual(combiner.combine(["p", "n"], [], item_id=1).probabilities,
                         [.98, .02])

    def test_explicit_exclusion_matches_physically_removed_row(self):
        W = np.array([["p", "p", "p"], ["p", "n", None], ["n", "n", "n"]], dtype=object)
        for position in range(len(W)):
            with self.subTest(position=position):
                held_out = AnonymousBayesianCombiner(W=W).combine(
                    ["p", "n"], [(0, "p")], item_id=position)
                removed = AnonymousBayesianCombiner(W=np.delete(W, position, axis=0)).combine(
                    ["p", "n"], [(0, "p")])
                self.assertEqual(held_out.probabilities, removed.probabilities)

    def test_missing_observations_and_empty_rows_are_ignored(self):
        W = np.array([["p", "p"], ["p", "n"], ["n", "n"], [None, pd.NA]], dtype=object)
        observations = list(enumerate([None, None, np.nan, pd.NA, "", "p"]))
        prediction = AnonymousBayesianCombiner(W=W).combine(["p", "n"], observations)
        self.assertEqual(prediction.probabilities, [2 / 3, 1 / 3])

    def test_insufficient_joint_support_returns_none(self):
        W = np.array([["p", "p"], ["p", "n"]])
        combiner = AnonymousBayesianCombiner(W=W)
        self.assertIsNone(combiner.combine(["p", "n"], [(0, "p"), (1, "p")]))
        self.assertIsNone(AnonymousBayesianCombiner(W=W[:1]).combine(
            ["p", "n"], [], item_id=0))

    def test_impossible_observation_and_all_missing_data_return_none(self):
        self.assertIsNone(AnonymousBayesianCombiner(W=np.array([["n", "n"], ["n", "n"]])).combine(
            ["p", "n"], [(0, "p")]))
        self.assertIsNone(AnonymousBayesianCombiner(W=np.array([[None, pd.NA]], dtype=object)).combine(
            ["p", "n"], []))

    def test_cache_keys_include_ordered_vocabulary(self):
        W = np.array([["p", "p", "p"], ["p", "n", None], ["n", "n", "n"]], dtype=object)
        combiner = AnonymousBayesianCombiner(W=W)
        for vocabulary in [["p", "n"], ["n", "p"], ["p", "n"]]:
            with self.subTest(vocabulary=vocabulary):
                prediction = combiner.combine(vocabulary, [(0, "p")])
                reference = AnonymousBayesianCombiner().combine(vocabulary, [(0, "p")], W=W)
                self.assertEqual(prediction.probabilities, reference.probabilities)
                self.assertEqual(prediction.label_probability("p"), 2 / 3)

    def test_numeric_zero_and_declared_empty_string_are_labels(self):
        for vocabulary in [[0, 1], ["", "p"]]:
            with self.subTest(vocabulary=vocabulary):
                first, second = vocabulary
                W = np.array([[first, first], [first, second], [second, second]], dtype=object)
                prediction = AnonymousBayesianCombiner(W=W).combine(vocabulary, [(0, first)])
                self.assertEqual(prediction.probabilities, [2 / 3, 1 / 3])

    def test_invalid_item_position_has_clear_error(self):
        combiner = AnonymousBayesianCombiner(W=np.array([["p", "n"]]))
        for position in [-1, 1, "item-a"]:
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, "row position"):
                combiner.combine(["p", "n"], [], item_id=position)


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


if __name__ == "__main__":
    unittest.main()
