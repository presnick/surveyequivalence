from random import shuffle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from surveyequivalence import AnalysisPipeline, Plot, DiscreteDistributionPrediction, FrequencyCombiner, \
    CrossEntropyScore, AnonymousBayesianCombiner, AUCScore, Combiner, Scorer, \
    find_maximal_full_rating_matrix_cols


def main():
    """
    This is the main driver for the personal attacks example. The driver function cycles through four different \
    combinations of ScoringFunctions and Combiners
    """

    # These are small values for a quick run through. Values used results reported in paper are provided as comments
    max_k = 10  # 20
    max_items = 20 # 2000
    bootstrap_samples = 0 # 200
    num_processors = 1

    # Next we iterate over various combinations of combiner and scoring functions.
    combiner = AnonymousBayesianCombiner(allowable_labels=['a', 'n'])
    scorer = CrossEntropyScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    combiner = FrequencyCombiner(allowable_labels=['a', 'n'])
    scorer = CrossEntropyScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    combiner = AnonymousBayesianCombiner(allowable_labels=['a', 'n'])
    scorer = AUCScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    combiner = FrequencyCombiner(allowable_labels=['a', 'n'])
    scorer = AUCScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)


def run(combiner: Combiner, scorer: Scorer, max_k: int, max_items: int, bootstrap_samples: int, num_processors: int):
    """
    Run personal attacks example with provided combiner and scorer.

    With personal attacks data we have annotations for if a Wikipedia comment is labeled as a personal attack or not from
    several different raters.

    Parameters
    ----------
    combiner : Combiner
        Combiner function
    scorer : Scorer
        Scoring function
    max_k : int
        Maximum number of raters to use when calculating survey power curve. Lower values dramatically speed up \
        execution of the procedure. No default is set, but this value is typically equal to the average number of \
        raters per item.
    max_items : int
        Maximum items to use from the dataset. Fewer items increases the speed of the procedure by results in loss \
        of statistical power. No default is set. If this value is smaller than the number of items in the dataset then \
        the function will only take the first max_items items from the dataset thereby ignoring some data.
    bootstrap_samples : int
        Number of samples to use when calculating survey equivalence. Like the number of samples in a t-tests, more \
        samples increases the statistical power, but each requires additional computational time. No default is set.
    num_processors : int
        Number of processors to use for parallel processing

    Notes
    -----
    This function uses data collected by Jigsaw's personal attacks platform [4]_ to generate survey equivalence values.

    References
    ----------
    .. [4] Wulczyn, E., Thain, N., & Dixon, L. (2017, April). Ex machina: Personal attacks seen at scale. \
    In Proceedings of the 26th international conference on world wide web (pp. 1391-1399).
    """

    # Load the dataset as a pandas dataframe
    wiki = pd.read_csv(f'data/wiki_attack_labels_and_predictor.csv')
    dataset = dict()

    # X and Y for calibration. These lists are matched
    X = list()
    y = list()

    # Create rating pairs from the dataset
    for index, item in wiki.iterrows():

        raters = list()

        n_raters = int(item['n_labels'])
        n_labelled_attack = int(item['n_labelled_attack'])

        for i in range(n_labelled_attack):
            raters.append('a')
            X.append([item['predictor_prob'], n_raters])
            y.append(1)
        for i in range(n_raters - n_labelled_attack):
            raters.append('n')
            X.append([item['predictor_prob'], n_raters])
            y.append(0)

        shuffle(raters)

        # This is the predictor i.e., personal_attack score for comment. It will be at index 0 in W.
        dataset[index] = [item['predictor_prob']] + raters

    # Determine the number of columns needed in W. This is the max number of raters for an item.
    length = max(map(len, dataset.values()))

    # Pad W with Nones if the number of raters for some item is less than the max.
    padded_dataset = np.array([xi + [None] * (length - len(xi)) for xi in dataset.values()])

    print('##Wiki Personal Attacks - Dataset loaded##', len(padded_dataset))

    # Trim the dataset to only the first max_items and recast W as a dataframe
    W = pd.DataFrame(data=padded_dataset)[:max_items]

    # Recall that index 0 was the classifier output, i.e., personal attacks score. We relabel this to 'soft classifier' to keep
    # track of it.
    W = W.rename(columns={0: 'soft classifier'})

    # Calculate calibration probabilities. Use the current hour as random seed, because these lists need to be matched
    print('Begin Calibration')

    calibrator = CalibratedClassifierCV(LinearSVC(max_iter=1000), method='isotonic').fit(pd.DataFrame([x for x, y in X]), y,
                                                                       sample_weight=[1 / y for x, y in X])

    # Let's keep one classifier uncalibrated
    uncalibrated_classifier = pd.DataFrame(
        [DiscreteDistributionPrediction(['a', 'n'], [attack_prob, 1 - attack_prob], normalize=True)
         for attack_prob
         in W['soft classifier']], columns=['Uncalibrated Jigsaw Personal Attacks Classifier'])

    # Create a calibrated classifier
    calibrated_classifier1 = pd.DataFrame(
        [DiscreteDistributionPrediction(['a', 'n'], [a, b], normalize=True)
         for b, a
         in calibrator.predict_proba(W.loc[:, W.columns == 'soft classifier'])
         ], columns=['Calibrated Jigsaw Personal Attacks Classifier'])

    # The classifier object now holds the classifier predictions. Let's remove this data from W now.
    W = W.drop(['soft classifier'], axis=1)

    classifiers = uncalibrated_classifier.join(calibrated_classifier1, lsuffix='left', rsuffix='right')


    print('Begin Survey Equivalence Analysis Pipeline')

    # AnalysisPipeline is the primary entry point into the SurveyEquivalence package. It takes the dataset W,
    # as well as a combiner, scorer, classifier prediction, max_k, and bootstrap samples. This function will
    # return a power curve.
    p = AnalysisPipeline(W, combiner=combiner, scorer=scorer, allowable_labels=['a', 'n'],
                         num_bootstrap_item_samples=bootstrap_samples, verbosity=1,
                         classifier_predictions=classifiers, max_K=max_k,
                         anonymous_raters=True,
                         procs=num_processors)

    p.save(path=p.path_for_saving(f"personal_attacks/{combiner.__class__.__name__}_plus_{scorer.__class__.__name__}"),
           msg=f"""
        Running personal attacks experiment with {len(W)} items and {len(W.columns)} raters per item
        {bootstrap_samples} bootstrap itemsets {combiner.__class__.__name__} with {scorer.__class__.__name__}
        """)

    # Here we create a prior score. This is the c_0, i.e., the baseline score from which we measure information gain
    # Information gain is only defined from cross entropy, so we only calculate this if the scorer is CrossEntropyScore
    if type(scorer) is CrossEntropyScore:
        # For the prior, we don't need any bootstrap samples and K needs to be only 1. Any improvement will be from k=2
        # k=3, etc.
        prior = AnalysisPipeline(W, combiner=AnonymousBayesianCombiner(allowable_labels=['a', 'n']), scorer=scorer,
                                 allowable_labels=['a', 'n'], num_bootstrap_item_samples=0, verbosity=1,
                                 classifier_predictions=classifiers, max_K=1,
                                 anonymous_raters=True,
                                 procs=num_processors)
    else:
        prior = None


    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              p.expert_power_curve,
              classifier_scores=p.classifier_scores,
              y_axis_label='score',
              color_map={'expert_power_curve': 'black', '0_uncalibrated': 'black', '0_calibrated': 'red'},
              center_on=prior.expert_power_curve.values[0] if prior is not None else None,
              name=f'Personal Attacks {type(combiner).__name__}_plus_{type(scorer).__name__}',
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

    # Save the figure and pgf/tikz if needed.
    pl.save(p.path_for_saving(f"personal_attacks/{type(combiner).__name__}_plus_{type(scorer).__name__}"), fig=fig)

if __name__ == '__main__':
    main()