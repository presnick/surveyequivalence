"""Offline fixtures, not model evaluations, for the AlpacaEval judge adapter."""
import copy
import json
import math
import unittest

import numpy as np
import pandas as pd

from research.alpaca import (
    INSTRUCTIONS, parse_outputs, request_for, run_analysis)
from surveyequivalence import AgreementScore, CrossEntropyScore
from surveyequivalence._equivalence import supported_equivalence
from research.arena import equivalence_summary


class AlpacaJudgeValidationTests(unittest.TestCase):
    def fixture(self):
        protocol = dict(model='offline-test-model', reasoning_effort='medium', max_output_tokens=2048)
        requests = [request_for(dict(item_id=k, instruction='instruction', response_A='first',
                                    response_B='second'), protocol) for k in ['one', 'two']]
        records = [dict(custom_id=r['custom_id'], error=None, response=dict(status_code=200,
            request_id='fixture-request', body=dict(status='completed', model='offline-test-model',
                metadata=r['body']['metadata'], id='fixture-response', output=[dict(type='message',
                    content=[dict(type='output_text', text=json.dumps({'probability_A': p}))])])))
            for r, p in zip(requests, [.75, .25])]
        return requests, records

    def test_blinding_and_request_identity(self):
        requests, _ = self.fixture()
        body = requests[0]['body']
        self.assertEqual(body['instructions'], INSTRUCTIONS)
        self.assertEqual(set(json.loads(body['input'])), {'instruction', 'response_A', 'response_B'})
        self.assertEqual(body['metadata']['item_id'], 'one')
        with self.assertRaisesRegex(ValueError, 'Blinded'):
            request_for(dict(item_id='x', instruction='i', response_A='a', response_B='b',
                             human_vote='A'), dict(model='fixture', reasoning_effort='medium', max_output_tokens=2048))
        other = request_for(dict(item_id='one', instruction='i', response_A='a', response_B='b'),
                            dict(model='offline-test-model', reasoning_effort='medium', max_output_tokens=2048))
        self.assertNotEqual(body['metadata']['request_sha256'], other['body']['metadata']['request_sha256'])

    def test_unordered_results_are_joined_by_item_not_line_position(self):
        requests, records = self.fixture()
        probabilities, _ = parse_outputs(records[::-1], requests, 'offline-test-model')
        self.assertEqual(probabilities.index.tolist(), ['one', 'two'])
        self.assertEqual(probabilities.tolist(), [.75, .25])

    def test_failures_and_incomplete_cohorts_are_never_dropped(self):
        requests, records = self.fixture()
        variants = [records[:1], records + records[:1]]
        for field, value in [('status', 'incomplete'), ('model', 'another-model'), ('metadata', {})]:
            modified = copy.deepcopy(records)
            modified[0]['response']['body'][field] = value
            variants.append(modified)
        modified = copy.deepcopy(records)
        modified[0]['error'] = {'code': 'batch_expired'}
        variants.append(modified)
        modified = copy.deepcopy(records)
        modified[0]['custom_id'] = 'unknown'
        variants.append(modified)
        for records in variants:
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    parse_outputs(records, requests, 'offline-test-model')

    def test_invalid_probabilities_refusals_and_mixed_models(self):
        requests, records = self.fixture()
        for value in [True, None, '0.5', -1, 1.1, float('nan'), float('inf')]:
            modified = copy.deepcopy(records)
            modified[0]['response']['body']['output'][0]['content'][0]['text'] = json.dumps({'probability_A': value})
            with self.assertRaisesRegex(ValueError, 'probability'):
                parse_outputs(modified, requests, 'offline-test-model')
        modified = copy.deepcopy(records)
        modified[0]['response']['body']['output'][0]['content'] = [dict(type='refusal', refusal='fixture')]
        with self.assertRaisesRegex(ValueError, 'refusal'):
            parse_outputs(modified, requests, 'offline-test-model')
        modified = copy.deepcopy(records)
        modified[0]['response']['body']['model'] += '-snapshot-one'
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            parse_outputs(modified, requests, 'offline-test-model')

    def test_scores_match_independent_ordered_oracle_including_repeats(self):
        W = pd.DataFrame([['A'] * n + ['B'] * (4-n) for n in [0, 1, 2, 3, 4] * 2],
                         index=[f'item-{i}' for i in range(10)])
        p = pd.Series([.25, .5, .75, .25, .75] * 2, index=W.index)
        samples = [np.arange(10), np.array([4, 2, 2, 0, 8, 1, 7, 6, 5, 3])]
        for scorer in [AgreementScore(), CrossEntropyScore()]:
            run = run_analysis(W, p, samples, 1729, scorer)
            terms = []
            for row, probability in zip(W.to_numpy(), p):
                if type(scorer) is AgreementScore:
                    terms.append(list(row).count('A' if probability >= .5 else 'B') / 4)
                else:
                    # Enumerate distinct outcomes in their first-seen order,
                    # matching the score's expected value over a uniform human.
                    terms.append(sum(list(row).count(y) / 4 * math.log2(
                        probability if y == 'A' else 1-probability) for y in dict.fromkeys(row)))
            expected = [sum(terms[i] for i in sample) / len(sample) for sample in samples]
            np.testing.assert_array_equal(run.classifier_scores.df['judge'], expected)
        with self.assertRaisesRegex(ValueError, 'indices'):
            run_analysis(W, p.iloc[::-1], samples, 1729, AgreementScore())

    def test_zero_probabilities_are_not_silently_smoothed(self):
        W = pd.DataFrame([['A', 'A', 'B', 'B']] * 6)
        run = run_analysis(W, pd.Series([0.] * 6), [np.arange(6)], 1729, CrossEntropyScore())
        self.assertEqual(run.classifier_scores.df.iloc[0, 0], -math.inf)

    def test_equivalence_keeps_censoring_instead_of_ranking_everything_at_three(self):
        powers = {0: .5, 1: .625, 2: .75, 3: .875}
        self.assertEqual(supported_equivalence(.6875, powers)['estimate'], 1.5)
        upper = supported_equivalence(.9, powers)
        self.assertEqual(upper, dict(status='above_range', estimate=None, bound=3.))
        summary = equivalence_summary([upper, supported_equivalence(.6875, powers), upper])
        self.assertIsNone(summary['interval'])
        self.assertEqual(summary['bootstrap_status_counts']['above_range'], 1)


if __name__ == '__main__':
    unittest.main()
