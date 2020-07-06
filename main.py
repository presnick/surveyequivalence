import numpy as np
import pandas as pd
import unittest

from surveyequivalence import State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    Plot, make_discrete_dataset_1, make_perceive_with_noise_datasets

def main():
    scorer = CrossEntropyScore

    # ds = make_discrete_dataset_1()
    #
    # pl = Plot(None,
    #           y_axis_label='information gain (c_k - c_0)',
    #           name='Test'
    #           )
    # pl.add_state_distribution_inset(ds.ds_generator)
    # pl.save_plot()
    # exit()

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'purple',
        'h_infinity': 'red'
    }
    # color_map.update({nm: color for (nm, color) in zip(ds.classifier_predictions.columns, ['red', 'blue', 'navy'])})

    for ds in make_perceive_with_noise_datasets():
        combiner = AnonymousBayesianCombiner()
        expert_pipeline = AnalysisPipeline(ds.dataset,
                                           expert_cols=list(ds.dataset.columns),
                                           amateur_cols=[],
                                           combiner=combiner,
                                           scoring_function=scorer.score,
                                           allowable_labels=['pos', 'neg'],
                                           null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                                           num_runs=2)

        pl = Plot(expert_pipeline.expert_power_curve,
                  classifier_scores=ds.compute_classifier_scores(scorer),
                  color_map=color_map,
                  y_axis_label='information gain (c_k - c_0)',
                  center_on_c0=True,
                  y_range=(0, .65),
                  name=ds.ds_generator.name,
                  legend_label='Expert raters',
                  amateur_legend_label="Lay raters"
                  )

        pl.plot(include_classifiers=True,
                include_classifier_equivalences=True,
                include_droplines=True,
                include_expert_points='all',
                connect_expert_points=True
                )
        pl.add_state_distribution_inset(ds.ds_generator)
        pl.save_plot()

    # expert_pipeline = AnalysisPipeline(ds.dataset, combiner, scorer.score,
    #                      allowable_labels=['pos', 'neg'],
    #                      null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
    #                      num_runs=16,
    #                      legend_label='Expert raters')
    #
    # # amateur_pipeline = AnalysisPipeline(ds.amateur_dataset, combiner, scorer.score,
    # #                      allowable_labels=['pos', 'neg'],
    # #                      null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
    # #                      num_runs=16,
    # #                      legend_label='Higher noise amateurs')
    #
    #
    # # colors = ['red', 'blue', 'navy']
    # # for c, color in zip(mock_classifiers, colors):
    # #     c.predictions = c.make_predictions(expert_states)
    # #     c.score = scorer.score(c.predictions, expert_dataset['r1'])
    # #     c.color = color
    #
    #     ## call the scoring function with the dataset, to generate a score
    # ## pass the results to the plotting function, with classifier.name
    #
    # # results = pd.concat([expert_pipeline.power_curve.means, expert_pipeline.power_curve.cis], axis=1)
    # # results.columns = ['mean', 'ci_width']
    # # print("*****RESULTS*****")
    # # print(combiner, scorer)
    # # print(results)
    # # for i in range(15):
    # #     thresh = .75 + .01 * i
    # #     print(f"\tsurvey equivalence for {thresh} is ", p.power_curve.compute_equivalence(thresh))
    #
    #
    #
    # pl = Plot(expert_pipeline.power_curve,
    #           # amateur_power_curve = amateur_pipeline.power_curve,
    #           classifier_scores = ds.compute_classifier_scores(scorer),
    #           color_map=color_map,
    #           y_axis_label='information gain (c_k - c_0)',
    #           center_on_c0=True,
    #           y_range = (0, .5),
    #           name=ds.ds_generator.name
    #           )
    #
    # pl.plot(include_classifiers=True,
    #         include_classifier_equivalences=True,
    #         include_droplines=True,
    #         include_expert_points='all',
    #         connect_expert_points=True
    #         # include_amateur_curve=True,
    #         # amateur_equivalences=[2, 8]
    #         )
    # pl.add_state_distribution_inset(ds.ds_generator)
    # pl.save_plot()

if __name__ == '__main__':
    main()