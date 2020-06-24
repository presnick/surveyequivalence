import numpy as np
import pandas as pd
import unittest

from surveyequivalence import State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    Plot, make_discrete_dataset_1

def main():
    combiner = AnonymousBayesianCombiner()
    scorer = CrossEntropyScore

    ds = make_discrete_dataset_1()

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'purple',
    }
    color_map.update({nm: color for (nm, color) in zip(ds.classifier_predictions.columns, ['red', 'blue', 'navy'])})
    print(color_map)

    amateur_pipeline = AnalysisPipeline(ds.amateur_dataset, combiner, scorer.score,
                         allowable_labels=['pos', 'neg'],
                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                         num_runs=1,
                         legend_label='Higher noise amateurs')


    expert_pipeline = AnalysisPipeline(ds.dataset, combiner, scorer.score,
                         allowable_labels=['pos', 'neg'],
                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                         num_runs=1,
                         legend_label='Expert raters')

    # colors = ['red', 'blue', 'navy']
    # for c, color in zip(mock_classifiers, colors):
    #     c.predictions = c.make_predictions(expert_states)
    #     c.score = scorer.score(c.predictions, expert_dataset['r1'])
    #     c.color = color

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
              classifier_scores = ds.compute_classifier_scores(scorer),
              color_map=color_map,
              y_axis_label='information gain (c_k - c_0)',
              center_on_c0=True
              )

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_amateur_curve=True,
            amateur_equivalences=[2, 8]
            )

if __name__ == '__main__':
    main()