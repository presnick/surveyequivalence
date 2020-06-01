import numpy as np
import pandas as pd
import unittest

from surveyequivalence import generate_labels, State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore

from test.discrete_distribution_tests import TestDiscreteDistributionSurveyEquivalence

def main():
    x = TestDiscreteDistributionSurveyEquivalence()
    x.setUp()
    x.test_analysis_pipeline()
    exit
    dataset = x.datasets[0]
    combiner = FrequencyCombiner()
    scorer = AgreementScore
    p = AnalysisPipeline(dataset, combiner, scorer.score, allowable_labels=['pos', 'neg'],
                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                         num_runs=2)

    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("*****RESULTS*****")
    print(combiner, scorer)
    print(results)
    for i in range(15):
        thresh = .75 + .01 * i
        print(f"\tsurvey equivalence for {thresh} is ", p.power_curve.compute_equivalence(thresh))

    # p.plot()

if __name__ == '__main__':
    main()