import numpy as np
import pandas as pd
import unittest

from surveyequivalence import generate_labels, State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    MockClassifier, make_discrete_dataset_1, make_discrete_dataset_2, make_discrete_dataset_3

class TestDiscreteDistributionSurveyEquivalence(unittest.TestCase):

    def test_frequency_combiner(self):
        frequency = FrequencyCombiner()
        pred = frequency.combine(['pos', 'neg'], np.array([(1, 'pos'), (2, 'neg'), (4, 'neg')]))
        self.assertEqual(pred.probabilities[0], 0.3333333333333333)
        self.assertEqual(pred.probabilities[1], 0.6666666666666666)

        pred = frequency.combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg'), (4, 'neg')]))
        self.assertEqual(pred.probabilities[0], 0.0)
        self.assertEqual(pred.probabilities[1], 1.0)

    def test_anonymous_bayesian_combiner(self):
        anonymous_bayesian = AnonymousBayesianCombiner()
        item_states, data, mock_classifiers = make_discrete_dataset_1()
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.293153527, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0]+pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'pos')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.6773972603, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'pos'), (2, 'pos')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.8876987131, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)

        anonymous_bayesian = AnonymousBayesianCombiner()
        item_states, data, mock_classifiers = make_discrete_dataset_2()
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.3675675676, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'pos')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.4086956522, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'pos'), (2, 'pos')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.4470588235, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)

        anonymous_bayesian = AnonymousBayesianCombiner()
        item_states, data, mock_classifiers = make_discrete_dataset_3()
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.4818181818, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'pos')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.5702702703, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)
        pred = anonymous_bayesian.combine(['pos', 'neg'], np.array([(1, 'pos'), (2, 'pos')]), data.to_numpy())
        self.assertAlmostEqual(pred.probabilities[0], 0.6463687151, delta=0.03)
        self.assertAlmostEqual(pred.probabilities[0] + pred.probabilities[1], 1.0, delta=0.01)

    def test_scoring_functions(self):
        small_dataset = [DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]]

        score = AgreementScore.score(small_dataset, ['b', 'b', 'b'])
        self.assertAlmostEqual(score, 0.6666666666, places=3)
        score = AgreementScore.score(small_dataset, ['a', 'b', 'b'])
        self.assertAlmostEqual(score, 0.3333333333, places=3)

        score = CrossEntropyScore.score(small_dataset, ['b', 'b', 'b'])
        self.assertAlmostEqual(score, 0.59459709985, places=3)
        score = CrossEntropyScore.score(small_dataset, ['a', 'b', 'b'])
        self.assertAlmostEqual(score, 0.87702971998, places=3)

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
        for dataset in self.datasets:
            for combiner in [FrequencyCombiner(), AnonymousBayesianCombiner()]:
                for scorer in [AgreementScore, CrossEntropyScore, PrecisionScore, RecallScore,
                               AUCScore]:
                    if isinstance(combiner, FrequencyCombiner) and isinstance(scorer(), CrossEntropyScore):
                        print("Cross entropy not well defined for Frequency combiner - no probabilities")
                        continue
                    if isinstance(combiner, FrequencyCombiner) and isinstance(scorer(), AUCScore):
                        print("AUC not well defined for Frequency combiner - no probabilities")
                        continue

                    p = AnalysisPipeline(dataset, combiner, scorer.score, allowable_labels=['pos', 'neg'],
                                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                                         num_runs=2)

                    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
                    results.columns = ['mean', 'ci_width']
                    print("*****RESULTS*****")
                    print(combiner, scorer)
                    print(results)
                    for i in range (15):
                        thresh = .75 + .01*i
                        print(f"\tsurvey equivalence for {thresh} is ", p.power_curve.compute_equivalence(thresh))


if __name__ == '__main__':
    unittest.main()