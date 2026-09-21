"""Independent checks of benchmark adapters and reporting, without downloads."""
from types import SimpleNamespace
import unittest

import numpy as np

from research.arena import (bayes_oracle, canonical_mt, cluster_samples,
                                         equivalence_summary, softmax, source_decisions, supported_equivalence)


class ArenaValidationTests(unittest.TestCase):
    def test_source_tie_rules_are_not_argmax(self):
        values = [[3., 3., 1.], [4., 1., 0.], [0., 4., 1.], [0., 1., 4.], [3., 1., 3.]]
        np.testing.assert_array_equal(source_decisions(values), [2, 0, 1, 2, 2])
        np.testing.assert_array_equal(source_decisions(values, binary=True), [1, 0, 1, 1, 0])

    def test_softmax_is_shift_invariant_and_normalized(self):
        p = softmax([[1000, 1000, 1000], [1001, 1000, 999]])
        np.testing.assert_array_equal(p, softmax([[0, 0, 0], [1, 0, -1]]))
        np.testing.assert_allclose(p.sum(axis=1), [1., 1.], rtol=0, atol=2e-16)
        np.testing.assert_array_equal(p[0], [1/3, 1/3, 1/3])
        with self.assertRaises(ValueError):
            softmax([[float('nan'), 0, 0]])

    def test_canonical_items_preserve_response_content_and_flip_winner(self):
        first = [dict(role='user', content='question'), dict(role='assistant', content='answer a')]
        second = [dict(role='user', content='question'), dict(role='assistant', content='answer b')]
        a = SimpleNamespace(question_id=81, turn=1, model_a='z', model_b='a', winner='model_a',
                            conversation_a=first, conversation_b=second)
        b = SimpleNamespace(question_id=81, turn=1, model_a='a', model_b='z', winner='model_b',
                            conversation_a=second, conversation_b=first)
        self.assertEqual(canonical_mt(a), canonical_mt(b))
        b.conversation_b = first[:1] + [dict(role='assistant', content='a different generation')]
        self.assertNotEqual(canonical_mt(a)[0], canonical_mt(b)[0])

    def test_future_turn_is_not_part_of_first_turn_item(self):
        row = SimpleNamespace(question_id=81, turn=1, model_a='a', model_b='b', winner='tie',
                              conversation_a=[{'content': str(i)} for i in range(4)],
                              conversation_b=[{'content': str(i)} for i in range(4)])
        original = canonical_mt(row)
        row.conversation_a[3]['content'] = 'unseen follow-up'
        self.assertEqual(original, canonical_mt(row))

    def test_equivalence_uses_earliest_plateau_and_interpolates(self):
        self.assertEqual(supported_equivalence(.5, {0: .2, 1: .5, 2: .5, 3: .8})['estimate'], 1.)
        self.assertEqual(supported_equivalence(.625, {0: .25, 1: .5, 2: .75})['estimate'], 1.5)

    def test_equivalence_does_not_fabricate_missing_crossings(self):
        for powers in [{0: .2, 1: float('nan'), 2: .8}, {0: .2, 2: .8}]:
            self.assertEqual(supported_equivalence(.5, powers)['status'], 'missing_curve')
        self.assertEqual(supported_equivalence(.9, {0: .2, 1: .8}),
                         dict(status='above_range', estimate=None, bound=1.))
        self.assertEqual(supported_equivalence(.1, {0: .2, 1: .8})['status'], 'crossing')

    def test_cluster_bootstrap_keeps_related_items_together(self):
        samples = cluster_samples(['a', 'a', 'b'], 20, 1729)
        repeated = cluster_samples(['a', 'a', 'b'], 20, 1729)
        for a, b in zip(samples, repeated):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(np.count_nonzero(a == 0), np.count_nonzero(a == 1))
        self.assertTrue(any(len(set(s)) < len(s) for s in samples[1:]))

    def test_censored_interval_retains_out_of_range_replicates(self):
        finite = dict(status='crossing', estimate=.5, bound=None)
        censored = dict(status='above_range', estimate=None, bound=2.)
        result = equivalence_summary([finite] + [finite] * 99 + [censored])
        self.assertEqual(result['interval'], [.5, .5])
        self.assertEqual(result['bootstrap_status_counts']['above_range'], 1)
        self.assertIsNone(equivalence_summary([finite] + [finite] * 50 + [censored] * 50)['interval'])

    def test_rational_bayesian_oracle_holds_out_exact_item(self):
        W = [['A', 'A', 'A'], ['B', 'B', 'B']]
        p = bayes_oracle(W, [], 0)
        self.assertAlmostEqual(p[1], .98 / 1.02, delta=1e-15)
        self.assertAlmostEqual(p[0], .02 / 1.02, delta=1e-15)
        self.assertIsNone(bayes_oracle(W, ['A'], 0))
