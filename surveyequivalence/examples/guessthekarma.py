from random import choice

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import ROOT_DIR
from surveyequivalence import AnalysisPipeline, Plot, DiscreteDistributionPrediction, FrequencyCombiner, \
    CrossEntropyScore, AnonymousBayesianCombiner, PluralityVote, F1Score, AgreementScore, Combiner, Scorer


def main():
    """
    This is the main driver for the Guess The Karma example. The driver function cycles through four different \
    combinations of ScoringFunctions and Combiners
    """

    # These are small values for a quick run through. Values used in experiments are provided in comments
    max_k = 3  # 30
    max_items = 20  # 1400
    bootstrap_samples = 5  # 200
    num_processors = 2

    # Next we iterate over various combinations of combiner and scoring functions.
    combiner = AnonymousBayesianCombiner(allowable_labels=['l', 'r'])
    scorer = CrossEntropyScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    # Frequency Combiner uses Laplace regularization
    combiner = FrequencyCombiner(allowable_labels=['l', 'r'], regularizer=1)
    scorer = CrossEntropyScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    combiner = PluralityVote(allowable_labels=['l', 'r'])
    scorer = F1Score()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    combiner = PluralityVote(allowable_labels=['l', 'r'])
    scorer = AgreementScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)


def run(combiner: Combiner, scorer: Scorer, max_k: int, max_items: int, bootstrap_samples: int, num_processors: int):
    """
    Run GuessTheKarma example with provided combiner and scorer.

    With GuessTheKarma data we have annotations for image pairs (items) from non-anonymous raters. In addition, each
    "correct" item is annotated as 'A' ('B' is therefore the incorrect answer). So to balance the dataset we
    randomly swap 'A' for 'B'.

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
    This function uses data collected by the GuessTheKarma Web game [1]_ to generate survey equivalence values.

    References
    ----------
    .. [1] Glenski, M., Stoddard, G., Resnick, P., & Weninger, T. (2018). Guessthekarma: A game to assess social \
    rating systems. Proceedings of the ACM on Human-Computer Interaction, 2(CSCW), 1-15.
    """

    # Load the dataset as a pandas dataframe
    gtk = pd.read_csv(f'{ROOT_DIR}/data/vote_gtk2.csv')

    prefer_W = dict()
    flip_dict = dict()

    # Create rating pairs from the dataset
    for index, rating in gtk.iterrows():

        # get the x and y in the W. This is the classifier i.e., reddit score. It will be at index 0 in W.
        if rating['image_pair'] not in prefer_W:
            flip_dict[rating['image_pair']] = choice([True, False])
            if flip_dict[rating['image_pair']]:
                prefer_W[rating['image_pair']] = list('r')
            else:
                prefer_W[rating['image_pair']] = list('l')

        # now get the preference
        rater_opinion = rating['opinion_choice']
        if rater_opinion == 'A':
            if flip_dict[rating['image_pair']]:
                prefer_W[rating['image_pair']].append('r')
            else:
                prefer_W[rating['image_pair']].append('l')
        elif rater_opinion == 'B':
            if flip_dict[rating['image_pair']]:
                prefer_W[rating['image_pair']].append('l')
            else:
                prefer_W[rating['image_pair']].append('r')
        else:
            pass

    # Determine the number of columns needed in W. This is the max number of raters for an item.
    x = list(prefer_W.values())
    length = max(map(len, x))

    # Pad W with Nones if the number of raters for some item is less than the max.
    W = np.array([xi + [None] * (length - len(xi)) for xi in x])

    print('##GUESSTHEKARMA - Dataset loaded##', len(W))

    # Trim the dataset to only the first max_items and recast W as a dataframe
    W = pd.DataFrame(data=W)[:max_items]

    # Recall that index 0 was the classifier output, i.e., reddit score. We relabel this to 'hard classifier' to keep
    # track of it.
    W = W.rename(columns={0: 'hard classifier'})

    # Calculate calibration probabilities
    calibrated_predictions_l = W[W['hard classifier'] == 'l'][W.columns.difference(['hard classifier'])].apply(
        pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)
    calibrated_predictions_r = W[W['hard classifier'] == 'r'][W.columns.difference(['hard classifier'])].apply(
        pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)

    # Because left and right are created randomly from the data. These calibrated probabilities should be close
    # to 50/50
    print(calibrated_predictions_l, calibrated_predictions_r)

    # Apply calibration probabilities to create a calibrated classifier
    classifier = pd.DataFrame(
        [DiscreteDistributionPrediction(['l', 'r'], [calibrated_predictions_l['l'], calibrated_predictions_l['r']],
                                        normalize=False) if reddit == 'l' else
         DiscreteDistributionPrediction(['l', 'r'], [calibrated_predictions_r['l'], calibrated_predictions_r['r']],
                                        normalize=False)
         for reddit in W['hard classifier']], columns=['Reddit Scores Classifier'])

    # The classifier object now holds the classifier predictions. Let's remove this data from W now.
    W = W.drop(['hard classifier'], axis=1)

    # Here we create a prior score. This is the c_0, i.e., the baseline score from which we measure information gain
    # Information gain is only defined from cross entropy, so we only calculate this if the scorer is CrossEntropyScore
    if type(scorer) is CrossEntropyScore:
        # For the prior, we don't need any bootstrap samples and K needs to be only 1. Any improvement will be from k=2
        # k=3, etc.
        prior = AnalysisPipeline(W, combiner=AnonymousBayesianCombiner(allowable_labels=['l', 'r']), scorer=scorer,
                                 allowable_labels=['l', 'r'], num_bootstrap_item_samples=0, verbosity=1,
                                 classifier_predictions=classifier, max_K=1, procs=num_processors)
    else:
        prior = None

    # AnalysisPipeline is the primary entry point into the SurveyEquivalence package. It takes the dataset W,
    # as well as a combiner, scorer, classifier prediction, max_k, and bootstrap samples. This functionw will
    # return a power curve.
    p = AnalysisPipeline(W, combiner=combiner, scorer=scorer, allowable_labels=['l', 'r'],
                         num_bootstrap_item_samples=bootstrap_samples, verbosity=1, classifier_predictions=classifier,
                         max_K=max_k, procs=num_processors)

    # Save the output
    p.save(path=p.path_for_saving(f"GTK/{combiner.__class__.__name__}_plus_{scorer.__class__.__name__}"),
           msg=f"""
        Running GuessTheKarma experiment with {len(W)} items and {len(W.columns)} raters per item
        {bootstrap_samples} bootstrap itemsets
        {combiner.__class__.__name__} with {scorer.__class__.__name__}.
        """)

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              p.expert_power_curve,
              classifier_scores=p.classifier_scores,
              y_axis_label='score',
              center_on=prior.expert_power_curve.values[0] if prior is not None else None,
              name=f'GTK {type(combiner).__name__}_plus_{type(scorer).__name__}',
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
    pl.save(p.path_for_saving(f"GTK/{type(combiner).__name__}_plus_{type(scorer).__name__}"), fig=fig)


if __name__ == '__main__':
    main()