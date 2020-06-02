import numpy as np
import pandas as pd
import unittest

from surveyequivalence import generate_labels, State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    Plot

from test.discrete_distribution_tests import TestDiscreteDistributionSurveyEquivalence

def main():
    x = TestDiscreteDistributionSurveyEquivalence()
    x.setUp()
    # x.test_analysis_pipeline()

    dataset = x.datasets[0]
    mock_classifiers = x.mock_classifiers[0]
    item_states = x.item_state_sequences[0]


    combiner = FrequencyCombiner()
    scorer = CrossEntropyScore()
    pipeline = AnalysisPipeline(dataset, combiner, scorer.score, allowable_labels=['pos', 'neg'],
                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                         num_runs=2)

    colors = ['red', 'blue']
    for c, color in zip(mock_classifiers, colors):
        c.predictions = c.make_predictions(item_states)
        c.score = scorer.score(c.predictions, dataset['r1'])
        c.color = color

        ## call the scoring function with the dataset, to generate a score
    ## pass the results to the plotting function, with classifier.name


    results = pd.concat([pipeline.power_curve.means, pipeline.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("*****RESULTS*****")
    print(combiner, scorer)
    print(results)
    # for i in range(15):
    #     thresh = .75 + .01 * i
    #     print(f"\tsurvey equivalence for {thresh} is ", p.power_curve.compute_equivalence(thresh))

    pl = Plot(pipeline.power_curve, classifiers=mock_classifiers)

    pl.plot()

if __name__ == '__main__':
    main()