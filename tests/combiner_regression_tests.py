import unittest

import numpy as np
import pandas as pd

from surveyequivalence._data import as_rating_array, is_missing
from surveyequivalence.combiners import (
    AnonymousBayesianCombiner, FrequencyCombiner, MeanCombiner, PluralityVote,
)


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


if __name__ == "__main__":
    unittest.main()
