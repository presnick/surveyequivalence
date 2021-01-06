import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from surveyequivalence import State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
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
                                    null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                                    num_runs=2,
                                    num_rater_samples=2,
                                    num_bootstrap_item_samples=2,
                                    max_expert_k=3
                                    )

        pl = Plot(pipeline.expert_power_curve,
                  classifier_scores=ds.compute_classifier_scores(scorer),
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
                                       # null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
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

def generate_and_plot_running_example():
    # scorer = AgreementScore()
    # combiner = PluralityVote(allowable_labels=['pos', 'neg'])

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'green',
        'hard classifier': 'red',
        'mock classifier': 'blue',
        'calibrated hard classifier': 'red'
    }

    # ds = make_running_example_dataset(num_items_per_dataset=1000, num_labels_per_item = 10, minimal=False,
    #                                   include_hard_classifer=True)
    #
    # # print(ds.dataset)
    # # print("---------")
    # # print(ds.classifier_predictions)
    # # print("---------")
    # # print(ds.ds_generator.expert_item_states)
    #
    # pipeline = AnalysisPipeline(ds.dataset,
    #                             expert_cols=list(ds.dataset.columns),
    #                             # amateur_cols=list(ds.dataset.columns[12:]),
    #                             classifier_predictions=ds.classifier_predictions,
    #                             combiner=combiner,
    #                             scorer=scorer,
    #                             allowable_labels=['pos', 'neg'],
    #                             num_bootstrap_item_samples=100,
    #                             max_K=10,
    #                             verbosity=1)
    #
    # pipeline.output_csv('plots/small_running_dataset.csv')
    # cs = pipeline.classifier_scores
    # print("\n----classifier scores-----")
    # print(cs.means)
    # print(cs.stds)
    # print("\n----power curve means-----")
    # print(pipeline.expert_power_curve.means)
    # print(pipeline.expert_power_curve.stds)
    # print("\n----survey equivalences----")
    # equivalences = pipeline.expert_power_curve.compute_equivalences(pipeline.classifier_scores)
    # print(equivalences)
    # print(f"means: {equivalences.mean()}")
    # print(f"medians {equivalences.median()}")
    # print(f"stddevs {equivalences.std()}")
    #
    # fig, ax = plt.subplots()
    #
    # fig.set_size_inches(8.5, 10.5)
    #
    # pl = Plot(ax,
    #           pipeline.expert_power_curve,
    #           # amateur_power_curve=pipeline.amateur_power_curve,
    #           classifier_scores=pipeline.classifier_scores,
    #           color_map=color_map,
    #           y_axis_label='percent agreement with reference rater',
    #           y_range=(0, 1),
    #           name='running example: majority vote + agreement score',
    #           legend_label='k raters',
    #           generate_pgf=True
    #           )
    #
    # pl.plot(include_classifiers=True,
    #         include_classifier_equivalences=True,
    #         include_droplines=True,
    #         include_expert_points='all',
    #         connect_expert_points=True,
    #         include_classifier_cis=True
    #         )
    # # pl.add_state_distribution_inset(ds.ds_generator)
    # pgf = None
    # if pl.generate_pgf:
    #     pgf = pl.template.substitute(**pl.template_dict)
    # save_plot(fig, 'runningexampleABC+cross_entropy', pgf)


    num_items_per_dataset=1000
    num_labels_per_item=10
    num_bootstrap_item_samples = 500

    # pipeline = load_saved_pipeline('saved_analyses/analysis_pipeline/01-01-2021_09-53-28_PM')


    hard_classifiers = ['hard classifier']
    soft_classifiers = ['mock classifier', 'calibrated hard classifier', 'h_infinity: ideal classifier']

    agreement_score = AgreementScore()
    plurality_combiner = PluralityVote(allowable_labels=['pos', 'neg'])

    cross_entropy = CrossEntropyScore()
    abc = AnonymousBayesianCombiner(allowable_labels=['pos', 'neg'])

    freq_combiner = FrequencyCombiner(allowable_labels=['pos', 'neg'])


    ds2 = make_running_example_dataset(minimal=False, num_items_per_dataset=num_items_per_dataset,
                                       num_labels_per_item=num_labels_per_item,
                                       include_soft_classifier=True, include_hard_classifer=True)

    print(f"""mean label counts to use as prior for ABC: {ds2.dataset.apply(
            pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)}""")
    base_pred = abc.combine(['pos', 'neg'], [], W=ds2.dataset.to_numpy(), item_id=1)
    predictions = [base_pred for _ in range(len(ds2.dataset))]
    c_0 = cross_entropy.score_classifier(predictions, ds2.dataset.columns, ds2.dataset)
    print(f"Cross Entropy on base_preds (i.e., c_0) = {c_0}")


    pipeline = AnalysisPipeline(ds2.dataset,
                                expert_cols=list(ds2.dataset.columns),
                                classifier_predictions=ds2.classifier_predictions[hard_classifiers],
                                combiner=plurality_combiner,
                                scorer=agreement_score,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                verbosity = 1)
    pipeline.save(dirname_base = "plurality_plus_agreement",
        msg = f"""
    Running example with {num_items_per_dataset} items and {num_labels_per_item} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    Plurality combiner with agreement score
    """)

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              pipeline.expert_power_curve,
              classifier_scores=ClassifierResults(pipeline.classifier_scores.df[['hard classifier']]),
              color_map=color_map,
              y_axis_label='percent agreement with reference rater',
              y_range=(0, 1),
              name='running example: majority vote + agreement score',
              legend_label='k raters',
              generate_pgf=True
              )

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=True ##change back to false
            )
    # pl.add_state_distribution_inset(ds.ds_generator)
    pgf = None
    if pl.generate_pgf:
        pgf = pl.template.substitute(**pl.template_dict)
    save_plot(fig, 'runningexample_majority_vote_plus_agreement', pgf)


    pipeline2 = AnalysisPipeline(ds2.dataset,
                                expert_cols=list(ds2.dataset.columns),
                                classifier_predictions=ds2.classifier_predictions[soft_classifiers],
                                combiner=abc,
                                scorer=cross_entropy,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                verbosity = 1)

    pipeline2.save(dirname_base = "abc_plus_cross_entropy",
                   msg = f"""
    Running example with {num_items_per_dataset} items and {num_labels_per_item} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    Anonymous Bayesian combiner with cross entropy score
    """)

    pipeline3 = AnalysisPipeline(ds2.dataset,
                                expert_cols=list(ds2.dataset.columns),
                                classifier_predictions=ds2.classifier_predictions[soft_classifiers],
                                combiner=freq_combiner,
                                scorer=cross_entropy,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                verbosity = 1)

    pipeline3.save(dirname_base = "frequency_plus_cross_entropy",
                   msg = f"""
    Running example with {num_items_per_dataset} items and {num_labels_per_item} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    frequency combiner with cross entropy score
    """)

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              pipeline2.expert_power_curve,
              classifier_scores=ClassifierResults(pipeline2.classifier_scores.df[['calibrated hard classifier']]),
              color_map=color_map,
              y_axis_label='information gain ($c_k - c_0$)',
              center_on=pipeline2.expert_power_curve.values[0],
              y_range=(0, 0.4),
              name='running example: ABC + cross entropy',
              legend_label='k raters',
              generate_pgf=True
              )

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=True ##change back to false
            )
    # pl.add_state_distribution_inset(ds.ds_generator)
    pgf = None
    if pl.generate_pgf:
        pgf = pl.template.substitute(**pl.template_dict)
    save_plot(fig, 'runningexampleABC+cross_entropy', pgf)


def main():
    # generate_and_plot_noise_datasets()
    # generate_and_plot_noisier_amateurs()
    generate_and_plot_running_example()

if __name__ == '__main__':
    main()
