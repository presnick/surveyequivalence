import os
import numpy as np
import pandas as pd
import unittest
from matplotlib import pyplot as plt
import matplotlib

from surveyequivalence import State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    Plot, make_discrete_dataset_1, make_perceive_with_noise_datasets, Correlation

def save_plot(fig):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    fig.savefig(f'plots/{self.name}{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.png')

def generate_and_plot_noise_datasets():
    scorer = CrossEntropyScore

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'purple',
        'h_infinity': 'red'
    }

    for ds in make_perceive_with_noise_datasets():
        combiner = AnonymousBayesianCombiner()
        pipeline = AnalysisPipeline(ds.dataset,
                                           expert_cols=list(ds.dataset.columns),
                                           amateur_cols=[],
                                           combiner=combiner,
                                           scoring_function=scorer.score,
                                           allowable_labels=['pos', 'neg'],
                                           null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                                           num_runs=2)

        pl = Plot(pipeline.expert_power_curve,
                  classifier_scores=ds.compute_classifier_scores(scorer),
                  color_map=color_map,
                  y_axis_label='information gain (c_k - c_0)',
                  center_on_c0=True,
                  y_range=(0, .65),
                  name=ds.ds_generator.name,
                  legend_label='Expert raters',
                  )

        pl.plot(include_classifiers=True,
                include_classifier_equivalences=True,
                include_droplines=True,
                include_expert_points='all',
                connect_expert_points=True
                )
        pl.add_state_distribution_inset(ds.ds_generator)
        pl.save_plot()

def generate_and_plot_noisier_amateurs():
    scorer = CrossEntropyScore
    combiner = AnonymousBayesianCombiner()

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'purple',
        'h_infinity': 'red'
    }

    ds = make_discrete_dataset_1()

    # combine the two dataframes

    expert_pipeline = AnalysisPipeline(pd.concat([ds.dataset, ds.amateur_dataset], axis=1),
                                expert_cols=list(ds.dataset.columns),
                                combiner=combiner,
                                scoring_function=scorer.score,
                                allowable_labels=['pos', 'neg'],
                                null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                                num_runs=2)

    amateur_pipeline = AnalysisPipeline(pd.concat([ds.dataset, ds.amateur_dataset], axis=1),
                                expert_cols=list(ds.dataset.columns),
                                amateur_cols=list(ds.amateur_dataset.columns),
                                combiner=combiner,
                                scoring_function=scorer.score,
                                allowable_labels=['pos', 'neg'],
                                null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                                num_runs=2)

    fig, ax = plt.subplots()

    fig.set_size_inches(18.5, 10.5)

    pl = Plot(ax,
              expert_pipeline.expert_power_curve,
              amateur_pipeline.amateur_power_curve,
              color_map=color_map,
              y_axis_label='information gain (c_k - c_0)',
              center_on_c0=True,
              y_range=(0, .65),
              name=ds.ds_generator.name,
              legend_label='Expert raters',
              amateur_legend_label="Lay raters"
              )

    pl.plot(include_classifiers=False,
            include_classifier_equivalences=False,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_amateur_curve=True,
            amateur_equivalences=[2, 8]
            )
    pl.add_state_distribution_inset(ds.ds_generator)
    save_plot(fig)


def main():
    # generate_and_plot_noise_datasets()
    generate_and_plot_noisier_amateurs()




if __name__ == '__main__':
    main()