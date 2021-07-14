import unittest

import numpy as np
import pandas as pd

##TODO: update this to use the generate_labels method; it's no longer a function
from surveyequivalence import DiscreteDistributionOverStates, DiscreteState, \
    DiscreteDistributionPrediction, \
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

        res = AnonymousBayesianCombiner().combine(['p', 'n'],
                                                  [('x', 'p'), ('x', 'p'), ('x', 'p'), ('x', 'n'), ('x', 'n'),
                                                   ('x', 'n'), ('x', 'n')], W, 1)
        self.assertAlmostEqual(res.probabilities[0], 0.2002, delta=0.001)

        res = AnonymousBayesianCombiner().combine(['p', 'n'],
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

    def test_analysis_pipeline(self):
        datasets = [synthetic_datasets.make_discrete_dataset_1(num_items_per_dataset=100).dataset,
                    synthetic_datasets.make_discrete_dataset_2(num_items_per_dataset=100).dataset,
                    synthetic_datasets.make_discrete_dataset_3(num_items_per_dataset=100).dataset]
        for dataset in datasets:
            for combiner in [AnonymousBayesianCombiner(allowable_labels=['pos', 'neg']), FrequencyCombiner(allowable_labels=['pos', 'neg'])]:
                for scorer in [CrossEntropyScore(), AgreementScore(), PrecisionScore(), RecallScore(),
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
