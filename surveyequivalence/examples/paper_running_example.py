import pandas as pd
from matplotlib import pyplot as plt
import time

from surveyequivalence import AgreementScore, PluralityVote, CrossEntropyScore, \
    AnonymousBayesianCombiner, FrequencyCombiner,  \
    AnalysisPipeline, Plot, ClassifierResults, DiscretePrediction, DiscreteDistributionPrediction, DMIScore_for_Soft_Classifier

def main(path = f'data/running_example_50_items', num_bootstrap_item_samples=5, nrows=None):

    # read the reference rater labels from file
    if nrows:
        W = pd.read_csv(f"{path}/ref_rater_labels.csv", index_col=0, nrows=nrows)
    else:
        W = pd.read_csv(f"{path}/ref_rater_labels.csv", index_col=0)

    # read the predictions from file
    def str2prediction_instance(s):
        # s will be in format "Prediction: [0.9, 0.1]" or "Prediction: neg"
        suffix = s.split(": ")[1]
        if suffix[0] == '[':
            pr_pos, pr_neg = suffix[1:-1].split(',')
            return DiscreteDistributionPrediction(['pos', 'neg'], [float(pr_pos), float(pr_neg)])
        else:
            return DiscretePrediction(suffix)

    if nrows:
        classifier_predictions = pd.read_csv(f"{path}/predictions.csv", index_col=0, nrows=nrows).applymap(str2prediction_instance)
    else:
        classifier_predictions = pd.read_csv(f"{path}/predictions.csv", index_col=0).applymap(str2prediction_instance)

    hard_classifiers = classifier_predictions.columns[:1] # ['mock hard classifier']
    soft_classifiers = classifier_predictions.columns[1:] # ['calibrated hard classifier', 'h_infinity: ideal classifier']

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'green',
        'mock hard classifier': 'red',
        'calibrated hard classifier': 'red'
    }

    # #### Plurality combiner plus Agreement score ####
    # plurality_combiner = PluralityVote(allowable_labels=['pos', 'neg'])
    # agreement_score = AgreementScore()
    # pipeline = AnalysisPipeline(W,
    #                             expert_cols=list(W.columns),
    #                             classifier_predictions=classifier_predictions[hard_classifiers],
    #                             combiner=plurality_combiner,
    #                             scorer=agreement_score,
    #                             allowable_labels=['pos', 'neg'],
    #                             num_bootstrap_item_samples=num_bootstrap_item_samples,
    #                             verbosity = 1)
    # pipeline.save(path = pipeline.path_for_saving("running_example/plurality_plus_agreement"),
    #     msg = f"""
    # Running example with {len(W)} items and {len(W.columns)} raters per item
    # {num_bootstrap_item_samples} bootstrap itemsets
    # Plurality combiner with agreement score
    # """)
    #
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8.5, 10.5)
    #
    #
    #
    # pl = Plot(ax,
    #           pipeline.expert_power_curve,
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
    # pl.save(pipeline.path_for_saving("running_example/plurality_plus_agreement"), fig=fig)

    def ABC_CE(target_panel_size=1):
        abc = AnonymousBayesianCombiner(allowable_labels=['pos', 'neg'],W=W)
        cross_entropy = CrossEntropyScore(num_ref_raters_per_virtual_rater=target_panel_size)
        # Here we set anonymous_raters to True, so that we will compute expected score against a randomly selected
        # rater for each item, rather than against a randomly selected column
        pipeline2 = AnalysisPipeline(W,
                                expert_cols=list(W.columns),
                                classifier_predictions=classifier_predictions[soft_classifiers],
                                combiner=abc,
                                scorer=cross_entropy,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                anonymous_raters=True,
                                performance_ratio_k = 2,
                                verbosity = 1,
                                procs=1)

        pipeline2.save(path=pipeline2.path_for_saving("running_example/abc_plus_cross_entropy_target_panel_size_"+str(target_panel_size)),
                   msg = f"""
        Running example with {len(W)} items and {len(W.columns)} raters per item
        {num_bootstrap_item_samples} bootstrap itemsets
        Anonymous Bayesian combiner with cross entropy score
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
              generate_pgf=True,
              performance_ratio_k=3,
              )

        pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=True
            )
        pl.save(path=pipeline2.path_for_saving("running_example/abc_plus_cross_entropy_target_panel_size_"+str(target_panel_size)), fig=fig)


    def ABC_DMI(target_panel_size=1):
        # abc+dmi
        abc = AnonymousBayesianCombiner(allowable_labels=['pos', 'neg'],W=W)
        dmi = DMIScore_for_Soft_Classifier(num_ref_raters_per_virtual_rater=target_panel_size)
        # Here we set anonymous_raters to True, so that we will compute expected score against a randomly selected
        # rater for each item, rather than against a randomly selected column
        pipeline4 = AnalysisPipeline(W,
                                expert_cols=list(W.columns),
                                classifier_predictions=classifier_predictions[soft_classifiers],
                                combiner=abc,
                                scorer=dmi,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                anonymous_raters=True,
                                verbosity = 1,
                                performance_ratio_k = 2,
                                procs=1)

        pipeline4.save(path=pipeline4.path_for_saving("running_example/abc_plus_dmi_target_panel_size_"+str(target_panel_size)),
                   msg = f"""
        Running example with {len(W)} items and {len(W.columns)} raters per item
        {num_bootstrap_item_samples} bootstrap itemsets
        Anonymous Bayesian combiner with cross entropy score
        """)

        fig, ax = plt.subplots()
        fig.set_size_inches(8.5, 10.5)

        pl = Plot(ax,
              pipeline4.expert_power_curve,
              classifier_scores=ClassifierResults(pipeline4.classifier_scores.df[['calibrated hard classifier']]),
              color_map=color_map,
              y_axis_label='information gain ($c_k - c_0$)',
              center_on=pipeline4.expert_power_curve.values[0],
              y_range=(0, 0.4),
              name='running example: ABC + dmi',
              legend_label='k raters',
              generate_pgf=True,
              performance_ratio_k=2
              )

        pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=True
            )
        pl.save(path=pipeline4.path_for_saving("running_example/abc_plus_dmi_target_panel_size_"+str(target_panel_size)), fig=fig)

    for i in range(1):
        t0=time.time()
        ABC_CE(i*2+1)
        t1=time.time()
        print("running time: ",t1-t0)
    for i in range(0):
        t0=time.time()
        ABC_DMI(i*2+1)
        t1=time.time()
        print("running time: ",t1-t0)

    # ###### Frequency combiner plus cross entropy ######
    # freq_combiner = FrequencyCombiner(allowable_labels=['pos', 'neg'])
    # pipeline3 = AnalysisPipeline(W,
    #                             expert_cols=list(W.columns),
    #                             classifier_predictions=classifier_predictions[soft_classifiers],
    #                             combiner=freq_combiner,
    #                             scorer=cross_entropy,
    #                             allowable_labels=['pos', 'neg'],
    #                             num_bootstrap_item_samples=num_bootstrap_item_samples,
    #                             verbosity = 1)
    #
    # pipeline3.save(path=pipeline.path_for_saving("running_example/frequency_plus_cross_entropy"),
    #                msg = f"""
    # Running example with {len(W)} items and {len(W.columns)} raters per item
    # {num_bootstrap_item_samples} bootstrap itemsets
    # frequency combiner with cross entropy score
    # """)


if __name__ == '__main__':
    main(path = '../data/running_example', nrows=200)