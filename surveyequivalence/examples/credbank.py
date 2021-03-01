from random import shuffle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import ROOT_DIR
from surveyequivalence import AnalysisPipeline, Plot, DiscreteDistributionPrediction, CrossEntropyScore, \
    AnonymousBayesianCombiner, Combiner, Scorer


def main():
    """
    This is the main driver for the CredBank example. The driver function calls just the AnonymousBayesian and \
    CrossEntropy Scoring Functions
    """

    # These are small values for a quick run through. Values used in experiments are provided in comments
    max_k = 3  # 30
    max_items = 20  # 1400
    bootstrap_samples = 5  # 200
    num_processors = 2

    # Next we iterate over various combinations of combiner and scoring functions.
    combiner = AnonymousBayesianCombiner(allowable_labels=['p', 'n'])
    scorer = CrossEntropyScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)


def run(combiner: Combiner, scorer: Scorer, max_k: int, max_items: int, bootstrap_samples: int, num_processors: int):
    """
    Run CredBank example with provided combiner and scorer.

    With The Credibility study we do not have individual raters, so we simulate them. Since the AnonymousBayesian \
    combiner is anonymous, ie, it doesn't matter who rated what, then we can just simulate these raters.

    In this case, we say that a rating was positive if credibility was rated high or medium high. Otherwise its not.

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
        Number of samples to use when calculating survey equivalence. Like the number of samples in a t-test, more \
        samples increases the statistical power, but each requires additional computational time. No default is set.
    num_processors : int
        Number of processors to use for parallel processing

    Notes
    -----
    This function uses data collected by the crawler of Mitra and Gilbert [2]_ and the classifier predictions were \
    constructed by Mitra et al [3]_

    References
    ----------
    .. [2] Mitra, T., & Gilbert, E. (2015, April). Credbank: A large-scale social media corpus with associated \
    credibility annotations. In Proceedings of the International AAAI Conference on Web and Social Media \
    (Vol. 9, No. 1).

    .. [3] Mitra, T., Wright, G. P., & Gilbert, E. (2017, February). A parsimonious language model of social media \
    credibility across disparate events. In Proceedings of the 2017 ACM conference on computer supported \
    cooperative work and social computing (pp. 126-145).
    """

    # Load the dataset as a pandas dataframe
    cred = pd.read_csv(f'{ROOT_DIR}/data/credweb.csv')

    W = dict()

    # Create rating pairs from the dataset
    for index, item in cred.iterrows():
        # get the x and y in the W
        raters = list()

        low = int(item['credCount-2_ret'])
        lowmed = int(item['credCount-1_ret'])
        med = int(item['credCount0_ret'])
        highmed = int(item['credCount1_ret'])
        high = int(item['credCount2_ret'])

        # Only high ratings get 'p' label, all others get 'n' (just as in the Mitra et al's paper)
        for i in range(high):
            raters.append('p')
        for i in range(low + lowmed + med + highmed):
            raters.append('n')

        # prepend the predictor class
        ret = item['cscw_predictor']
        ca_ret = item['P_ca_ret']

        shuffle(raters)

        # add the predictor class
        W[index] = [ret,ca_ret] + raters

    # Determine the number of columns needed in W. This is the max number of raters for an item.
    x = list(W.values())
    length = max(map(len, x))

    # Pad W with Nones if the number of raters for some item is less than the max.
    W = np.array([xi + [None] * (length - len(xi)) for xi in x])

    print('##CREDWEB - Dataset loaded##', len(W))

    # Trim the dataset to only the first max_items and recast W as a dataframe
    W = pd.DataFrame(data=W)[:max_items]

    # Recall that index 0 and 1 were for the classifier outputs, i.e., the credbank score We relabel this
    # to 'hard classifier' and 'perc' respectively to keep track of them.
    W = W.rename(columns={0: 'hard classifier', 1: 'perc'})

    # Next we calculate calibration probabilities
    calibrated = dict()
    for cred in W['hard classifier']:
        if cred not in calibrated:
            calibrated[cred] = W[W['hard classifier'] == cred]['perc'].mean()

    print(f"Calibrated classes: {calibrated}")

    # Apply calibration probabilities to create a calibrated classifier
    classifier = pd.DataFrame(
        [DiscreteDistributionPrediction(['p', 'n'], [calibrated[cred], 1 - calibrated[cred]], normalize=True) for cred
         in W['hard classifier']])

    # The classifier object now holds the classifier predictions. Let's remove this data from W now.
    W = W.drop(['hard classifier'], axis=1)
    W = W.drop(['perc'], axis=1)

    # Here we create a prior score. This is the c_0, i.e., the baseline score from which we measure information gain
    # Information gain is only defined from cross entropy, so we only calculate this if the scorer is CrossEntropyScore
    if type(scorer) is CrossEntropyScore:
        # For the prior, we don't need any bootstrap samples and K needs to be only 1. Any improvement will be from k=2
        # k=3, etc.
        prior = AnalysisPipeline(W, combiner=AnonymousBayesianCombiner(allowable_labels=['p', 'n']), scorer=scorer,
                                 allowable_labels=['p', 'n'], num_bootstrap_item_samples=0, verbosity=1,
                                 classifier_predictions=classifier, max_K=1, procs=num_processors)
    else:
        prior = None

    p = AnalysisPipeline(W, combiner=combiner, scorer=scorer, allowable_labels=['p', 'n'],
                         num_bootstrap_item_samples=bootstrap_samples, verbosity=1, classifier_predictions=classifier,
                         max_K=max_k, procs=num_processors)

    p.save(path=p.path_for_saving(f"CredWeb/{combiner.__class__.__name__}_plus_{scorer.__class__.__name__}"),
           msg=f"""
        Running CredWeb experiment with {len(W)} items and {len(W.columns)} raters per item
        {bootstrap_samples} bootstrap itemsets {combiner.__class__.__name__} with {scorer.__class__.__name__}.
        Classes are calibrated at: {calibrated}
        """)

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              p.expert_power_curve,
              classifier_scores=p.classifier_scores,
              y_axis_label='score',
              center_on=prior.expert_power_curve.means[0] if prior is not None else None,
              name=f'Cred {type(combiner).__name__}_plus_{type(scorer).__name__}',
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
    pl.save(p.path_for_saving(f"Credweb/{type(combiner).__name__}_plus_{type(scorer).__name__}"), fig=fig)


if __name__ == '__main__':
    main()