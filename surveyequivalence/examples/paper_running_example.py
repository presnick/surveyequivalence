import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from surveyequivalence import AgreementScore, PluralityVote, CrossEntropyScore, \
    AnonymousBayesianCombiner, FrequencyCombiner,  \
    AnalysisPipeline, Plot, ClassifierResults, DiscretePrediction, DiscreteDistributionPrediction

from config import ROOT_DIR

def main():

    path = f'{ROOT_DIR}/data/running_example'

    # read the reference rater labels from file
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
    classifier_predictions = pd.read_csv(f"{path}/predictions.csv", index_col=0).applymap(str2prediction_instance)

    hard_classifiers = classifier_predictions.columns[:1] # ['mock hard classifier']
    soft_classifiers = classifier_predictions.columns[1:] # ['calibrated hard classifier', 'h_infinity: ideal classifier']

    num_bootstrap_item_samples = 500
    num_bootstrap_item_samples = 1

    #### Plurality combiner plus Agreement score ####
    plurality_combiner = PluralityVote(allowable_labels=['pos', 'neg'])
    agreement_score = AgreementScore()
    pipeline = AnalysisPipeline(W,
                                expert_cols=list(W.columns),
                                classifier_predictions=classifier_predictions[hard_classifiers],
                                combiner=plurality_combiner,
                                scorer=agreement_score,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                verbosity = 1)
    pipeline.save(dirname_base = "plurality_plus_agreement",
        msg = f"""
    Running example with {len(W)} items and {len(W.columns)} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    Plurality combiner with agreement score
    """)

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'green',
        'mock hard classifier': 'red',
        'calibrated hard classifier': 'red'
    }

    pl = Plot(ax,
              pipeline.expert_power_curve,
              classifier_scores=pipeline.classifier_scores,
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
            include_classifier_cis=True
            )
    pl.save(fig, 'runningexample_majority_vote_plus_agreement')


    #### ABC + CrossEntropy
    abc = AnonymousBayesianCombiner(allowable_labels=['pos', 'neg'])
    cross_entropy = CrossEntropyScore()

    pipeline2 = AnalysisPipeline(W,
                                expert_cols=list(W.columns),
                                classifier_predictions=classifier_predictions[soft_classifiers],
                                combiner=abc,
                                scorer=cross_entropy,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                verbosity = 1)

    pipeline2.save(dirname_base = "abc_plus_cross_entropy",
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
              generate_pgf=True
              )

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=True ##change back to false
            )
    pl.save(fig, 'runningexampleABC+cross_entropy')


    ###### Frequency combiner plus cross entropy ######
    freq_combiner = FrequencyCombiner(allowable_labels=['pos', 'neg'])
    pipeline3 = AnalysisPipeline(W,
                                expert_cols=list(W.columns),
                                classifier_predictions=classifier_predictions[soft_classifiers],
                                combiner=freq_combiner,
                                scorer=cross_entropy,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                verbosity = 1)

    pipeline3.save(dirname_base = "frequency_plus_cross_entropy",
                   msg = f"""
    Running example with {len(W)} items and {len(W.columns)} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    frequency combiner with cross entropy score
    """)


if __name__ == '__main__':
    main()