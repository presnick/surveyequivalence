import numpy as np
import pandas as pd
import unittest

from surveyequivalence import generate_labels, State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    Plot, make_discrete_dataset_1, make_noisier_binary_states

def main():
    combiner = AnonymousBayesianCombiner()
    scorer = CrossEntropyScore

    expert_states, expert_dataset, mock_classifiers = make_discrete_dataset_1()

    amateur_states = make_noisier_binary_states(expert_states, 1.1)
    amateur_dataset = generate_labels(amateur_states, 10)

    amateur_pipeline = AnalysisPipeline(amateur_dataset, combiner, scorer.score, allowable_labels=['pos', 'neg'],
                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                         num_runs=2)


    expert_pipeline = AnalysisPipeline(expert_dataset, combiner, scorer.score, allowable_labels=['pos', 'neg'],
                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                         num_runs=2)

    colors = ['red', 'blue', 'yellow']
    for c, color in zip(mock_classifiers, colors):
        c.predictions = c.make_predictions(expert_states)
        c.score = scorer.score(c.predictions, expert_dataset['r1'])
        c.color = color

        ## call the scoring function with the dataset, to generate a score
    ## pass the results to the plotting function, with classifier.name

    results = pd.concat([expert_pipeline.power_curve.means, expert_pipeline.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("*****RESULTS*****")
    print(combiner, scorer)
    print(results)
    # for i in range(15):
    #     thresh = .75 + .01 * i
    #     print(f"\tsurvey equivalence for {thresh} is ", p.power_curve.compute_equivalence(thresh))

    pl = Plot(expert_pipeline.power_curve,
              amateur_power_curve = amateur_pipeline.power_curve,
              classifiers=mock_classifiers)

    pl.plot()

if __name__ == '__main__':
    main()