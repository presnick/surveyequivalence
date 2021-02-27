import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from surveyequivalence import SyntheticDataset, AgreementScore, PluralityVote, CrossEntropyScore, \
    AnonymousBayesianCombiner, FrequencyCombiner, DiscreteDistributionOverStates, DiscreteState, \
    SyntheticBinaryDatasetGenerator, MappedDiscreteMockClassifier, DiscreteDistributionPrediction, \
    DiscretePrediction, MockClassifier, AnalysisPipeline, Plot, ClassifierResults

from surveyequivalence.examples import save_plot

def make_running_example_dataset(num_items_per_dataset = 10, num_labels_per_item=10, minimal=False,
                                 include_hard_classifier=False, include_soft_classifier=False)->SyntheticDataset:
    """
    This generates the running example dataset used in the original Survey Equivalence paper.

    Three states: 70% high = 80/20, 10% med = 50/50; 20% low = 10/90

    Parameters
    ----------
    num_items_per_dataset
    num_labels_per_item
    minimal
        If minimal, use FixedStateGenerator, which generates labels in exact proportion to probabilities specified \
        in the state, rather than each label being an iid draw from the State.
    include_hard_classifier
        Includes a hard classifier which draws labels 90/10 for high state; 50/50 for medium; 05/95 fow low state
    include_soft_classifier
        Includes a soft classifier which runs the hard_classifier to generate a label and then maps it to a calibrated \
        prediction (.7681 when the label is positive; .3226 when the label is negative). Also includes an ideal \
        classifier that always predicts the probability given by the State of the item.
    """

    if minimal:
        state_generator = \
            FixedStateGenerator(states=[DiscreteState(state_name='high',
                                                          labels=['pos', 'neg'],
                                                          probabilities=[.8, .2]),
                                        DiscreteState(state_name='med',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.5, .5]),
                                        DiscreteState(state_name='low',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.1, .9])
                                        ],
                                probabilities=[.7, .1, .2]
                                )
    else:
        state_generator = \
            DiscreteDistributionOverStates(states=[DiscreteState(state_name='high',
                                                                 labels=['pos', 'neg'],
                                                                 probabilities=[.8, .2]),
                                                   DiscreteState(state_name='med',
                                                                 labels=['pos', 'neg'],
                                                                 probabilities=[.5, .5]),
                                                   DiscreteState(state_name='low',
                                                                 labels=['pos', 'neg'],
                                                                 probabilities=[.1, .9])
                                                   ],
                                           probabilities=[.7, .1, .2]
                                           )

    dsg = SyntheticBinaryDatasetGenerator(item_state_generator= state_generator,
                                          num_items_per_dataset=num_items_per_dataset,
                                          num_labels_per_item=num_labels_per_item,
                                          mock_classifiers=None,
                                          name="running example",
                                          )

    if include_hard_classifier:
        dsg.mock_classifiers.append(MappedDiscreteMockClassifier(
            name='mock hard classifier',
            label_predictors={
                'high': DiscreteDistributionPrediction(['pos', 'neg'], [.9, .1]),
                'med': DiscreteDistributionPrediction(['pos', 'neg'], [.5, .5]),
                'low': DiscreteDistributionPrediction(['pos', 'neg'], [.05, .95]),
            },
            prediction_map={'pos': DiscretePrediction('pos'),
                            'neg': DiscretePrediction('neg')
                            }
        ))

    if include_soft_classifier:
        # dsg.mock_classifiers.append(MockClassifier(
        #     name='mock classifier',
        #     label_predictors={
        #         'high': DiscreteDistributionPrediction(['pos', 'neg'], [.9, .1]),
        #         'med': DiscreteDistributionPrediction(['pos', 'neg'], [.5, .5]),
        #         'low': DiscreteDistributionPrediction(['pos', 'neg'], [.05, .95]),
        #     }))

        dsg.mock_classifiers.append(MappedDiscreteMockClassifier(
            name='calibrated hard classifier',
            label_predictors={
                'high': DiscreteDistributionPrediction(['pos', 'neg'], [.9, .1]),
                'med': DiscreteDistributionPrediction(['pos', 'neg'], [.5, .5]),
                'low': DiscreteDistributionPrediction(['pos', 'neg'], [.05, .95]),
            },
            prediction_map = {'pos': DiscreteDistributionPrediction(['pos', 'neg'], [.7681, .2319]),
                              'neg': DiscreteDistributionPrediction(['pos', 'neg'], [.3226, .6774])
                              }
        ))

        dsg.mock_classifiers.append(MockClassifier(
            name='h_infinity: ideal classifier',
            label_predictors={
                'high': DiscreteDistributionPrediction(['pos', 'neg'], [.8, .2]),
                'med': DiscreteDistributionPrediction(['pos', 'neg'], [.5, .5]),
                'low': DiscreteDistributionPrediction(['pos', 'neg'], [.1, .9]),
            }))


    return SyntheticDataset(dsg)

def main():

    num_items_per_dataset=10
    num_labels_per_item=10
    num_bootstrap_item_samples = 1

    # num_items_per_dataset=1000
    # num_labels_per_item=10
    # num_bootstrap_item_samples = 500

    ds2 = make_running_example_dataset(minimal=False, num_items_per_dataset=num_items_per_dataset,
                                       num_labels_per_item=num_labels_per_item,
                                       include_soft_classifier=True, include_hard_classifier=True)

    hard_classifiers = ['mock hard classifier']
    soft_classifiers = ['calibrated hard classifier', 'h_infinity: ideal classifier']

    #### Plurality combiner plus Agreement score ####
    plurality_combiner = PluralityVote(allowable_labels=['pos', 'neg'])
    agreement_score = AgreementScore()
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
            include_classifier_cis=True ##change back to false
            )
    # pl.add_state_distribution_inset(ds.ds_generator)
    pgf = None
    if pl.generate_pgf:
        pgf = pl.template.substitute(**pl.template_dict)
    save_plot(fig, 'runningexample_majority_vote_plus_agreement', pgf)


    #### ABC + CrossEntropy
    abc = AnonymousBayesianCombiner(allowable_labels=['pos', 'neg'])
    cross_entropy = CrossEntropyScore()

    print(f"""mean label counts to use as prior for ABC: {ds2.dataset.apply(
            pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)}""")
    base_pred = abc.combine(['pos', 'neg'], [], W=ds2.dataset.to_numpy(), item_id=1)
    predictions = [base_pred for _ in range(len(ds2.dataset))]
    c_0 = cross_entropy.score_classifier(predictions, ds2.dataset.columns, ds2.dataset)
    print(f"Cross Entropy on base_preds (i.e., c_0) = {c_0}")


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


    ###### Frequency combiner plus cross entropy ######
    freq_combiner = FrequencyCombiner(allowable_labels=['pos', 'neg'])
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

if __name__ == '__main__':
    main()