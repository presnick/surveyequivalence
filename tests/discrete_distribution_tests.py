import unittest

import numpy as np
import pandas as pd

##TODO: update this to use the generate_labels method; it's no longer a function
from surveyequivalence import DiscreteDistributionOverStates, DiscreteState, \
    DiscretePrediction, DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    MockClassifier, NumericPrediction, synthetic_datasets


class TestDiscreteDistributionSurveyEquivalence(unittest.TestCase):

    def test_leave_one_item_out(self):
        W = np.zeros((9, 15), dtype=str)
        W[0] = ['p', 'p', 'p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
        W[1] = ['p', 'p', 'p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
        W[2] = ['p', 'p', 'p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
        W[3] = ['p', 'p', 'p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
        W[4] = ['p', 'p', 'p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
        W[5] = ['p', 'p', 'p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']
        W[6] = ['p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', '', '', '']
        W[7] = ['p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', '', '', '']
        W[8] = ['p', 'p', 'p', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', '', '', '']

        res = AnonymousBayesianCombiner(W=W).combine(['p', 'n'],
                                                  [('x', 'p'), ('x', 'p'), ('x', 'p'), ('x', 'n'), ('x', 'n'),
                                                   ('x', 'n'), ('x', 'n')], W, 1)
        self.assertAlmostEqual(res.probabilities[0], 0.2002, delta=0.001)

        res = AnonymousBayesianCombiner(W=W).combine(['p', 'n'],
                                                  [('x', 'p'), ('x', 'p'), ('x', 'p'), ('x', 'n'), ('x', 'n'),
                                                   ('x', 'n'),
                                                   ('x', 'n')], W, 7)

        self.assertAlmostEqual(res.probabilities[0], 0.2024, delta=0.001)

    def test_frequency_combiner(self):
        frequency = FrequencyCombiner()
        pred = frequency.combine(['pos', 'neg'], np.array([(1, 'pos'), (2, 'neg'), (4, 'neg')]))
        self.assertEqual(pred.probabilities[0], 0.3333333333333333)
        self.assertEqual(pred.probabilities[1], 0.6666666666666666)

        pred = frequency.combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg'), (4, 'neg')]))
        self.assertAlmostEqual(pred.probabilities[0], 0.0, delta=0.03) #delta of 0.03 because of rounding at the extremes
        self.assertAlmostEqual(pred.probabilities[1], 1.0, delta=0.03)

    def test_ABC_2(self):
        # W: A pandas DataFrame with one row for each item and one column for each rater. Cells are labels.
        Wrows = [
            ['pos', 'pos', 'neg'],
            ['pos', 'pos', 'pos'],
            ['pos', 'pos', 'pos'],
            ['pos', 'neg', 'neg'],
            ['pos', 'neg', 'neg'],
            ['pos', 'neg', 'pos'],
            ['neg', 'pos', 'pos'],
            ['neg', 'neg', 'neg'],
            ['neg', 'neg', 'pos']
        ]
        W = pd.DataFrame(Wrows, columns=['r1', 'r2', 'r3'])
        W_np = W.to_numpy()

        anonymous_bayesian = AnonymousBayesianCombiner(W=W)

        #     r1   r2   r3
        # 0  pos  pos  neg
        # 1  pos  pos  pos
        # 2  pos  pos  pos
        # 3  pos  neg  neg
        # 4  pos  neg  neg
        # 5  pos  neg  pos
        # 6  neg  pos  pos
        # 7  neg  neg  neg
        # 8  neg  neg  pos

        ## two or more positive labels on items 0, 1, 2, 5, 6
        # for items 1 and 2, always get two pos, and then always get a third
        # for items 0, 5, and 6, get two pos 1/3 of the time and, when you do, never get a third.

        # excluding item 0, if we see (pos, pos),
        # the chance that it came from items 1 or 2 is (1 + 1) / (1 + 1 + 1/3 + 1/3) = .75
        pred = anonymous_bayesian.combine(
            allowable_labels = ['pos', 'neg'],
            labels = [(1, 'pos'), (2, 'pos')],
            W = W_np,
            item_id = 0
        )
        self.assertAlmostEqual(pred.probabilities[0], 0.75, delta=0.0001)

        # excluding item 1, if we we see (pos, pos),
        # the chance that it came from item 2 is 1 / (1 + 1/3 + 1/3 + 1/3) = .5
        pred = anonymous_bayesian.combine(
            allowable_labels = ['pos', 'neg'],
            labels = [(1, 'pos'), (2, 'pos')],
            W = W_np,
            item_id = 1
        )
        self.assertAlmostEqual(pred.probabilities[0], 0.5, delta=0.0001)


        ## (pos, neg) is possible on items 0, 3, 4, 5, 6, 7,
        # for all of them, the chance of specifically (pos, neg) for first two is 1/3
        # for items 0, 5, and 6, probability of (pos, neg, pos) is 1/3
        # for other items, probability of (pos, neg, pos) is 0

        # excluding item 0, if we see (pos, neg),
        # the chance that it came from items 0, 5, or 6 is 2/5
        pred = anonymous_bayesian.combine(
            allowable_labels = ['pos', 'neg'],
            labels = [(1, 'pos'), (2, 'neg')],
            W = W_np,
            item_id = 0
        )
        self.assertAlmostEqual(pred.probabilities[0], 0.4, delta=0.0001)

        # excluding item 3, if we we see (pos, neg),
        # the chance that it came from item 0, 5, or 6 is 3/5
        pred = anonymous_bayesian.combine(
            allowable_labels = ['pos', 'neg'],
            labels = [(1, 'pos'), (2, 'neg')],
            W = W_np,
            item_id = 3
        )
        self.assertAlmostEqual(pred.probabilities[0], 0.6, delta=0.0001)

        ## (neg) is possible on rows 0, 3, 4, 5, 6, 7, 8
        # for 0, 5, and 6, pr(neg) is 1/3 and pr(neg, pos) is 1/3
        # for 3, 4, and 8, pr(neg) is 2/3 and pr(neg, pos) is 1/3
        # for 7, pr(neg) is 1 and pr(neg, pos) is 0

        # excluding item 0, if we see (pos),
        # the chance that it came from 5 or 6 is 2/11, and a pos will definitely follow
        # the chance that it came from 3, 4, or 8 is 6/11, and pos will follow with prob 1/2
        # the chance that it came from 7 is 3/11, and pos will definitely not follow
        # so pr((neg, pos) | (neg)) = 2/11 + 6/11*1/2 = 5/11
        pred = anonymous_bayesian.combine(
            allowable_labels = ['pos', 'neg'],
            labels = [(1, 'neg')],
            W = W_np,
            item_id = 0
        )
        self.assertAlmostEqual(pred.probabilities[0], 5/11, delta=0.0001)

        # excluding item 3, if we see (pos),
        # the chance that it came from 0, 5 or 6 is 3/10, and a pos will definitely follow
        # the chance that it came from 3, 4, or 8 is 4/10, and pos will follow with prob 1/2
        # the chance that it came from 7 is 3/10, and pos will definitely not follow
        # so pr((neg, pos) | (neg)) = 3/10 + 4/10*1/2 = 5/10
        pred = anonymous_bayesian.combine(
            allowable_labels = ['pos', 'neg'],
            labels = [(1, 'neg')],
            W = W_np,
            item_id = 3
        )
        self.assertAlmostEqual(pred.probabilities[0], 5/10, delta=0.0001)


        # excluding item 7, if we see (pos),
        # the chance that it came from 0, 5 or 6 is 3/9, and a pos will definitely follow
        # the chance that it came from 3, 4, or 8 is 6/9, and pos will follow with prob 1/2
        # the chance that it came from 7 is 0, and pos will definitely not follow
        # so pr((neg, pos) | (neg)) = 3/9 + 6/9*1/2 = 6/9
        pred = anonymous_bayesian.combine(
            allowable_labels = ['pos', 'neg'],
            labels = [(1, 'neg')],
            W = W_np,
            item_id = 7
        )
        self.assertAlmostEqual(pred.probabilities[0], 6/9, delta=0.0001)

    def test_anonymous_bayesian_combiner(self):
        synth_dataset = synthetic_datasets.make_discrete_dataset_1(num_items_per_dataset=1000)
        anonymous_bayesian = AnonymousBayesianCombiner()
        pred = anonymous_bayesian.combine(['pos', 'neg'],  [(1, 'neg'), (2, 'neg')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.293153527, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], [(1, 'neg'), (2, 'pos')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.6773972603, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], [(1, 'pos'), (2, 'pos')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.8876987131, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)

        anonymous_bayesian = AnonymousBayesianCombiner()
        synth_dataset = synthetic_datasets.make_discrete_dataset_2(num_items_per_dataset=1000)
        pred = anonymous_bayesian.combine(['pos', 'neg'], [(1, 'neg'), (2, 'neg')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.3675675676, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], [(1, 'neg'), (2, 'pos')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.4086956522, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], [(1, 'pos'), (2, 'pos')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.4470588235, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)

        anonymous_bayesian = AnonymousBayesianCombiner()
        synth_dataset = synthetic_datasets.make_discrete_dataset_3(num_items_per_dataset=1000)
        pred = anonymous_bayesian.combine(['pos', 'neg'], [(1, 'neg'), (2, 'neg')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.4818181818, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], [(1, 'neg'), (2, 'pos')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.5702702703, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], [(1, 'pos'), (2, 'pos')], synth_dataset.dataset.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.6463687151, delta=0.04)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)

    def test_cross_entropy_anonymous_score(self):
        three_predictions = [DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]]
        W = pd.DataFrame([['a', 'b', 'b', 'b', None],
                          ['a', 'a', 'a', 'a', None],
                          ['a', 'a', 'b', 'b', 'b']],
                         columns = ['r1', 'r2', 'r3', 'r4', 'r5'])
        score = CrossEntropyScore().expected_score_anonymous_raters(three_predictions, W)
        # correct score 1.076680823, which is mean of:
        # .25*log2(.3) + .75*log2(.7) ==> -0.820171278
        # log2(.4) ==> - 1.321928095
        # .4*log2(.6) + .6*log2(.4) ==> - 1.087943095

        self.assertAlmostEqual(score, -1.076680823, places=3)

    def test_precision_anonymous_score(self):
        three_predictions = [DiscretePrediction(x) for x in ['pos','pos', 'neg']]
        W = pd.DataFrame([['pos', 'neg', 'neg', 'neg', None],
                          ['pos', 'pos', 'pos', 'pos', None],
                          ['pos', 'pos', 'neg', 'neg', 'neg']],
                         columns = ['r1', 'r2', 'r3', 'r4', 'r5'])
        score = PrecisionScore().expected_score_anonymous_raters(three_predictions, W)
        # correct score .625, which is mean of:
        # .25
        # 1

        self.assertAlmostEqual(score, .625, places=3)

    def test_agreement_anonymous_score(self):
        three_predictions = [DiscretePrediction(x) for x in ['pos','pos', 'neg']]
        W = pd.DataFrame([['pos', 'neg', 'neg', 'neg', None],
                          ['pos', 'pos', 'pos', 'pos', None],
                          ['pos', 'pos', 'neg', 'neg', 'neg']],
                         columns = ['r1', 'r2', 'r3', 'r4', 'r5'])
        score = AgreementScore().expected_score_anonymous_raters(three_predictions, W)
        # correct score 1.85/3 = .616667, which is mean of:
        # .25
        # 1
        # .6

        self.assertAlmostEqual(score, .616667, places=3)


    def test_scoring_functions(self):
        small_dataset = [DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]]

        score = AgreementScore.score([i for i in small_dataset], ['a', 'a', 'a'])
        self.assertAlmostEqual(score, 0.33333333, places=3)
        score = AgreementScore.score([i for i in small_dataset], ['b', 'b', 'b'])
        self.assertAlmostEqual(score, 0.66666666, places=3)

        score = CrossEntropyScore.score([i for i in small_dataset], ['a', 'a', 'a'])
        self.assertAlmostEqual(score, -1.26528, places=3)
        score = CrossEntropyScore.score([i for i in small_dataset], ['b', 'b', 'b'])
        self.assertAlmostEqual(score, -0.85782, places=3)

        score = CrossEntropyScore.score([i for i in small_dataset], ['a', 'b', 'a'])
        self.assertAlmostEqual(score, -1.070, places=3)
        score = CrossEntropyScore.score([i for i in small_dataset], ['b', 'a', 'b'])
        self.assertAlmostEqual(score, -1.0528, places=3)

        # TODO: still have not converted precision and recall to accept DiscreteState

        score = PrecisionScore.score(small_dataset, ['b', 'b', 'b'], average='micro')
        self.assertAlmostEqual(score, 0.66666666666, places=3)
        score = PrecisionScore.score(small_dataset, ['a', 'b', 'b'], average='micro')
        self.assertAlmostEqual(score, 0.33333333333, places=3)
        score = PrecisionScore.score(small_dataset, ['b', 'b', 'b'], average='macro')
        self.assertAlmostEqual(score, 0.5, places=3)
        score = PrecisionScore.score(small_dataset, ['a', 'b', 'b'], average='macro')
        self.assertAlmostEqual(score, 0.25, places=3)

        score = RecallScore.score(small_dataset, ['b', 'b', 'b'], average='micro')
        self.assertAlmostEqual(score, 0.66666666666, places=3)
        score = RecallScore.score(small_dataset, ['a', 'b', 'b'], average='micro')
        self.assertAlmostEqual(score, 0.33333333333, places=3)
        score = RecallScore.score(small_dataset, ['b', 'b', 'b'], average='macro')
        self.assertAlmostEqual(score, 0.3333333333, places=3)
        score = RecallScore.score(small_dataset, ['a', 'b', 'b'], average='macro')
        self.assertAlmostEqual(score, 0.25, places=3)

        score = F1Score.score(small_dataset, ['b', 'b', 'b'], average='micro')
        self.assertAlmostEqual(score, 0.66666666666, places=3)
        score = F1Score.score(small_dataset, ['a', 'b', 'b'], average='micro')
        self.assertAlmostEqual(score, 0.33333333333, places=3)
        score = F1Score.score(small_dataset, ['b', 'b', 'b'], average='macro')
        self.assertAlmostEqual(score, 0.4, places=3)
        score = F1Score.score(small_dataset, ['a', 'b', 'b'], average='macro')
        self.assertAlmostEqual(score, 0.25, places=3)

        # score = AUCScore.score(small_dataset, ['b', 'b', 'b'])
        # self.assertAlmostEqual(score, 0.4, places=3)
        # ROC doesn't make sense with only one class
        score = AUCScore.score(small_dataset, ['b', 'b', 'a'])
        self.assertAlmostEqual(score, 0.75, places=3)

    def test_non_full_rating_matrix(self):
        datasets = [synthetic_datasets.make_non_full_dataset_1(num_items_per_dataset=100).dataset]
        for dataset in datasets:
            for combiner in [AnonymousBayesianCombiner(allowable_labels=['pos', 'neg'],W=dataset)]:
                for scorer in [CrossEntropyScore(), AgreementScore()]:
                    if isinstance(combiner, FrequencyCombiner) and isinstance(scorer, CrossEntropyScore):
                        print("Cross entropy not well defined for Frequency combiner - no probabilities")
                        continue
                    if isinstance(combiner, FrequencyCombiner) and isinstance(scorer, AUCScore):
                        print("AUC not well defined for Frequency combiner - no probabilities")
                        continue

                    p = AnalysisPipeline(dataset, combiner=combiner, scorer=scorer,
                                         allowable_labels=['pos', 'neg'], num_bootstrap_item_samples=2, max_K=3)

                    results = pd.concat([p.expert_power_curve.means, p.expert_power_curve.stds], axis=1)
                    results.columns = ['mean', 'std']
                    print("*****RESULTS*****")
                    print(combiner, scorer)
                    print(results)
                    for i in range(15):
                        thresh = results['mean'][0] + .01 * i
                        print(f"\tsurvey equivalence for {thresh} is ", p.expert_power_curve.compute_equivalence_at_actuals(thresh))

    def test_analysis_pipeline(self):
        datasets = [synthetic_datasets.make_discrete_dataset_1(num_items_per_dataset=100).dataset,
                    synthetic_datasets.make_discrete_dataset_2(num_items_per_dataset=100).dataset,
                    synthetic_datasets.make_discrete_dataset_3(num_items_per_dataset=100).dataset]
        for dataset in datasets:
            for combiner in [AnonymousBayesianCombiner(allowable_labels=['pos', 'neg'],W=dataset), FrequencyCombiner(allowable_labels=['pos', 'neg'])]:
                for scorer in [CrossEntropyScore(), AgreementScore(), RecallScore(),
                               AUCScore()]:
                    if isinstance(combiner, FrequencyCombiner) and isinstance(scorer, CrossEntropyScore):
                        print("Cross entropy not well defined for Frequency combiner - no probabilities")
                        continue
                    if isinstance(combiner, FrequencyCombiner) and isinstance(scorer, AUCScore):
                        print("AUC not well defined for Frequency combiner - no probabilities")
                        continue

                    p = AnalysisPipeline(dataset, combiner=combiner, scorer=scorer,
                                         allowable_labels=['pos', 'neg'], num_bootstrap_item_samples=2, max_K=3)

                    results = pd.concat([p.expert_power_curve.means, p.expert_power_curve.stds], axis=1)
                    results.columns = ['mean', 'std']
                    print("*****RESULTS*****")
                    print(combiner, scorer)
                    print(results)
                    for i in range(15):
                        thresh = results['mean'][0] + .01 * i
                        print(f"\tsurvey equivalence for {thresh} is ", p.expert_power_curve.compute_equivalence_at_actuals(thresh))


if __name__ == '__main__':
    unittest.main()
