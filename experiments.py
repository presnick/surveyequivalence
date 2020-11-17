import os
from datetime import datetime
from random import shuffle, choice

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from surveyequivalence import AnalysisPipeline, Plot, DiscreteDistributionPrediction, FrequencyCombiner, F1Score, \
    CrossEntropyScore, AnonymousBayesianCombiner, PrecisionScore


def save_plot(fig, name, pgf=None):
    if not os.path.isdir('plots'):
        os.mkdir('plots')
    fig.savefig(f'plots/{name}{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.png')
    if pgf:
        # Need to get rid of extra linebreaks. This is important
        pgf = pgf.replace('\r', '')
        with open(f'plots/{name}{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.tex', 'w') as tex:
            tex.write(pgf)

def guessthekarma():
    """
    With GuessTheKarma data we have annotations for image pairs (items) from non-anonymous raters. In addition, each
    the "correct" item is annotated as 'A' ('B' is therefore the incorrect answer). So to balance the dataset we
    randomly swap 'A' for 'B'. In total, W is a len(#max_ratings) by len(#items) matrix.
    :return: None
    """

    gtk = pd.read_csv('./data/vote_gtk2.csv')

    prefer_W = dict()
    flip_dict = dict()

    for index, rating in gtk.iterrows():

        # get the x and y in the W
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
            # print(rater_opinion)

    x = list(prefer_W.values())
    length = max(map(len, x))
    prefer_W = np.array([xi + [None] * (length - len(xi)) for xi in x])

    print('##GUESSTHEKARMA - Dataset loaded##', len(prefer_W))

    prefer_W = pd.DataFrame(data=prefer_W)[:100]
    prefer_W = prefer_W.rename(columns={0: 'hard classifier'})

    calibrated_predictions_l = prefer_W[prefer_W['hard classifier'] == 'l'][
        prefer_W.columns.difference(['hard classifier'])].apply(
        pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)

    calibrated_predictions_r = prefer_W[prefer_W['hard classifier'] == 'r'][
        prefer_W.columns.difference(['hard classifier'])].apply(
        pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)

    print(calibrated_predictions_l, calibrated_predictions_r)

    classifier = pd.DataFrame(
        [DiscreteDistributionPrediction(['l', 'r'], [calibrated_predictions_l['l'], calibrated_predictions_l[
            'r']], normalize=False) if reddit == 'l' else DiscreteDistributionPrediction(['l', 'r'], [calibrated_predictions_r['l'],
                                                                                calibrated_predictions_r['r']], normalize=False) for reddit
         in prefer_W['hard classifier']])
    prefer_W = prefer_W.drop(['hard classifier'], axis=1)

    p = AnalysisPipeline(prefer_W, combiner=FrequencyCombiner(allowable_labels=['l', 'r']),
                         scorer=F1Score(), allowable_labels=['l', 'r'],
                         num_bootstrap_item_samples=10, verbosity=1, classifier_predictions=classifier, max_K=4)

    cs = p.classifier_scores
    print("\nfull dataset\n")
    print("\n----classifier scores-----")
    print(cs.means)
    print(cs.stds)
    print("\n----power curve means-----")
    print(p.expert_power_curve.means)
    print(p.expert_power_curve.stds)
    print("\n----survey equivalences----")
    equivalences = p.expert_power_curve.compute_equivalences(p.classifier_scores)
    print(equivalences)
    print(f"means: {equivalences.mean()}")
    print(f"medians {equivalences.median()}")
    print(f"stddevs {equivalences.std()}")

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              p.expert_power_curve,
              classifier_scores=p.classifier_scores,
              y_axis_label='information gain (c_k - c_0)',
              center_on_c0=True,
              y_range=(0, 0.4),
              name='running example ABC + cross_entropy',
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
    # pl.add_state_distribution_inset(ds.ds_generator)
    pgf = None
    if pl.generate_pgf:
        pgf = pl.template.substitute(**pl.template_dict)
    save_plot(fig, 'gtk+ABC+cross_entropy', pgf)


def wiki_toxicity():
    """
    :return: None
    """
    #rev_id, perc_labelled_attack, n_labelled_attack, n_labels, predictor_prob
    wiki = pd.read_csv('./data/wiki_attack_labels_and_predictor.csv')

    W = dict()

    for index, item in wiki.iterrows():
        # get the x and y in the W
        raters = list()

        n_raters = int(item['n_labels'])
        n_labelled_attack = int(item['n_labelled_attack'])

        for i in range(n_labelled_attack):
            raters.append('a')
        for i in range(n_raters - n_labelled_attack):
            raters.append('n')
        shuffle(raters)

        # add the predictor class
        W[index] = [item['predictor_prob']] + raters

    x = list(W.values())
    length = max(map(len, x))
    W = np.array([xi + [None] * (length - len(xi)) for xi in x])

    print('##Wiki Toxic - Dataset loaded##', len(W))

    W = pd.DataFrame(data=W)[:500]
    W = W.rename(columns={0: 'soft classifier'})

    classifier = pd.DataFrame(
        [DiscreteDistributionPrediction(['a', 'n'], [attack_prob, 1-attack_prob], normalize=True)
         for attack_prob
         in W['soft classifier']])
    W = W.drop(['soft classifier'], axis=1)

    p = AnalysisPipeline(W, combiner=AnonymousBayesianCombiner(allowable_labels=['a', 'n']), scorer=CrossEntropyScore(),
                         allowable_labels=['a', 'n'],
                         num_bootstrap_item_samples=2, verbosity=1, classifier_predictions=classifier, max_K=4,
                         max_rater_subsets=20)

    cs = p.classifier_scores
    print("\nfull dataset\n")
    print("\n----classifier scores-----")
    print(cs.means)
    print(cs.stds)
    print("\n----power curve means-----")
    print(p.expert_power_curve.means)
    print(p.expert_power_curve.stds)
    print("\n----survey equivalences----")
    equivalences = p.expert_power_curve.compute_equivalences(p.classifier_scores)
    print(equivalences)
    print(f"means: {equivalences.mean()}")
    print(f"medians {equivalences.median()}")
    print(f"stddevs {equivalences.std()}")

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              p.expert_power_curve,
              classifier_scores=p.classifier_scores,
              y_axis_label='information gain (c_k - c_0)',
              center_on_c0=True,
              y_range=(0, 0.4),
              name='WikiToxc + ABC + cross_entropy',
              legend_label='k raters',
              )

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=False
            )
    # pl.add_state_distribution_inset(ds.ds_generator)
    save_plot(fig, 'WikiToxc+ABC+cross_entropy')


def cred_web():
    """
    With The Credibility study we do not have individual raters, so we simulate them. Since the Frequency
    and AnonymousBayesian combiners are both anonymous, ie, it doesn't matter who rated what, then we can just simulate
    these raters.
    In this case, we say that a rating was positive if credibility was rated high or medium high. Otherwise its not.
    :return:
    """
    wiki = pd.read_csv('./data/credweb.csv')

    W = dict()

    for index, item in wiki.iterrows():
        # get the x and y in the W
        raters = list()

        low = int(item['credCount-2_ret'])
        lowmed = int(item['credCount-1_ret'])
        med = int(item['credCount0_ret'])
        highmed = int(item['credCount1_ret'])
        high = int(item['credCount2_ret'])

        for i in range(highmed + high):
            raters.append('p')
        for i in range(low + lowmed + med):
            raters.append('n')

        shuffle(raters)

        #prepend the predictor class
        ret = 'n'
        if item['Credibility_Class_Number_ret'] in [1,2]:
            ret = 'p'

        # add the predictor class
        W[index] = [ret] + raters

    x = list(W.values())
    length = max(map(len, x))
    W = np.array([xi + [None] * (length - len(xi)) for xi in x])

    print('##CREDWEB - Dataset loaded##', len(W))

    W = pd.DataFrame(data=W)
    W = W.rename(columns={0: 'hard classifier'})

    calibrated_predictions_p = W[W['hard classifier'] == 'p'][
        W.columns.difference(['hard classifier'])].apply(
        pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)

    calibrated_predictions_n = W[W['hard classifier'] == 'n'][
        W.columns.difference(['hard classifier'])].apply(
        pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)

    print(calibrated_predictions_p, calibrated_predictions_n)

    classifier = pd.DataFrame(
        [DiscreteDistributionPrediction(['p', 'n'], [calibrated_predictions_p['p'], calibrated_predictions_p[
            'n']], normalize=False) if reddit == 'l' else DiscreteDistributionPrediction(['p', 'n'], [calibrated_predictions_n['p'],
                                                                                calibrated_predictions_n['n']], normalize=False) for reddit
         in W['hard classifier']])
    W = W.drop(['hard classifier'], axis=1)


    p = AnalysisPipeline(W, combiner=AnonymousBayesianCombiner(allowable_labels=['p', 'n']), scorer=CrossEntropyScore(),
                         allowable_labels=['p', 'n'],
                         num_bootstrap_item_samples=100, verbosity=1, classifier_predictions=classifier, max_K=20, max_rater_subsets=200)

    cs = p.classifier_scores
    print("\nfull dataset\n")
    print("\n----classifier scores-----")
    print(cs.means)
    print(cs.stds)
    print("\n----power curve means-----")
    print(p.expert_power_curve.means)
    print(p.expert_power_curve.stds)
    print("\n----survey equivalences----")
    equivalences = p.expert_power_curve.compute_equivalences(p.classifier_scores)
    print(equivalences)
    print(f"means: {equivalences.mean()}")
    print(f"medians {equivalences.median()}")
    print(f"stddevs {equivalences.std()}")

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              p.expert_power_curve,
              classifier_scores=p.classifier_scores,
              y_axis_label='information gain (c_k - c_0)',
              center_on_c0=True,
              y_range=(0, 0.4),
              name='Credweb + ABC + cross_entropy',
              legend_label='k raters',
              )

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=False
            )
    # pl.add_state_distribution_inset(ds.ds_generator)
    save_plot(fig, 'credweb+ABC+cross_entropy')


def main():
    #cred_web()
    guessthekarma()
    #wiki_toxicity()


main()
