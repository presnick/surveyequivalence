import numpy as np
import pandas as pd
import unittest

from surveyequivalence import generate_labels, State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    Plot, make_discrete_dataset_1

def main():

    item_states, dataset, mock_classifiers = make_discrete_dataset_1()

    combiner = AnonymousBayesianCombiner()
    scorer = CrossEntropyScore
    pipeline = AnalysisPipeline(dataset, combiner, scorer.score, allowable_labels=['pos', 'neg'],
                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                         num_runs=2)

    colors = ['red', 'blue', 'yellow']
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