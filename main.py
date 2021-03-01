import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from surveyequivalence import State, DiscreteState, \
    DistributionOverStates, MixtureOfBetas, \
    DiscreteDistributionPrediction, \
    FrequencyCombiner, PluralityVote, AnonymousBayesianCombiner, \
    AnalysisPipeline, AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, \
    Plot, make_discrete_dataset_1, make_running_example_dataset, make_perceive_with_noise_datasets, Correlation, \
    ClassifierResults, \
    load_saved_pipeline

def save_plot(fig, name, pgf=None):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    fig.savefig(f'plots/{name}{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.png')
    if pgf:
        # Need to get rid of extra linebreaks. This is important
        pgf = pgf.replace('\r', '')
        with open(f'plots/{name}{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.tex', 'w') as tex:
            tex.write(pgf)


def generate_and_plot_noise_datasets():
    scorer = CrossEntropyScore

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'purple',
        'h_infinity: ideal classifier': 'red'
    }

    for ds in make_perceive_with_noise_datasets():
        combiner = AnonymousBayesianCombiner()
        pipeline = AnalysisPipeline(ds.dataset,
                                    expert_cols=list(ds.dataset.columns),
                                    amateur_cols=[],
                                    combiner=combiner,
                                    scoring_function=scorer.score,
                                    allowable_labels=['pos', 'neg'],
                                    num_runs=2,
                                    num_rater_samples=2,
                                    num_bootstrap_item_samples=2,
                                    max_expert_k=3
                                    )

        pl = Plot(pipeline.expert_power_curve,
                  classifier_scores=pipeline.compute_classifier_scores(scorer),
                  color_map=color_map,
                  y_axis_label='information gain ($c_k - c_0$)',
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
    scorer = CrossEntropyScore()
    combiner = AnonymousBayesianCombiner()

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'purple',
        'h_infinity: ideal classifier': 'red'
    }

    ds = make_discrete_dataset_1(num_items_per_dataset=50)

    # combine the two dataframes

    pipeline = AnalysisPipeline(pd.concat([ds.dataset, ds.amateur_dataset], axis=1),
                                       expert_cols=list(ds.dataset.columns),
                                       amateur_cols=list(ds.amateur_dataset.columns),
                                       classifier_predictions = ds.classifier_predictions,
                                       combiner=combiner,
                                       scoring_function=scorer,
                                       allowable_labels=['pos', 'neg'],
                                       num_bootstrap_item_samples=10)

    cs = pipeline.classifier_scores

    # pipeline.expert_power_curve = pipeline.compute_power_curve()

    # pipeline.amateur_power_curve = pipeline.compute_power_curve(amateur_cols= ds.amateur_dataset.columns,
    #                                                             max_k=len(ds.amateur_dataset.columns),
    #                                                             source_name="amateur"
    #                                                             )

    fig, ax = plt.subplots()

    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              pipeline.expert_power_curve,
              pipeline.amateur_power_curve,
              classifier_scores=pipeline.classifier_scores,
              color_map=color_map,
              y_axis_label='information gain ($c_k - c_0$)',
              center_on_c0=True,
              y_range=(0, .65),
              name=ds.ds_generator.name,
              legend_label='Expert raters',
              amateur_legend_label="Lay raters"
              )

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=False,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_amateur_curve=True,
            amateur_equivalences=[2]
            )
    # pl.add_state_distribution_inset(ds.ds_generator)
    save_plot(fig, ds.ds_generator.name)



def main():
    # generate_and_plot_noise_datasets()
    # generate_and_plot_noisier_amateurs()

if __name__ == '__main__':
    main()
