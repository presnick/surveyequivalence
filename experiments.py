import pandas as pd
import numpy as np
from random import shuffle
from surveyequivalence import AnalysisPipeline, DiscreteDistributionPrediction, FrequencyCombiner, F1Score, CrossEntropyScore, AnonymousBayesianCombiner


def guessthekarma():
    """
    With GuessTheKarma data we have annotations for image pairs (items) from non-anonymous raters. In addition, each
    the "correct" item is annotated as 'A' ('B' is therefore the incorrect answer). So to balance the dataset we
    duplicate it and swap 'A' for 'B'. In total, we W as a len(#raters) by 2*len(#items) matrix.
    :return: None
    """
    gtk = pd.read_csv('./data/vote_gtk2.csv')

    rater_ids = {v:i for (i,v) in enumerate(set(gtk['user_id']))}
    item_ids = {v:i for (i,v) in enumerate(set(gtk['image_pair']))}
    predict_W = np.zeros((len(rater_ids),len(item_ids)), dtype=str)
    prefer_W = np.zeros((len(rater_ids),len(item_ids)), dtype=str)

    for index, rating in gtk.iterrows():
        # get the x and y in the W
        rater_id = rater_ids[rating['user_id']]
        item_id = item_ids[rating['image_pair']]

        # now get the preference
        rater_opinion = rating['opinion_choice']
        if rater_opinion == 'A':
            prefer_W[rater_id, item_id] = 'p'
        elif rater_opinion == 'B':
            prefer_W[rater_id, item_id] = 'n'
        else:
            pass
            # print(rater_opinion)

        rater_prediction = rating['prediction_choice']
        if rater_prediction == 'A':
            predict_W[rater_id, item_id] = 'p'
        elif rater_prediction == 'B':
            predict_W[rater_id, item_id] = 'n'
        else:
            pass
            # print(rater_prediction)

    # prefer might have empty rows because GTK didn't always ask the preference question.
    mask = np.all(prefer_W == '', axis=1)
    prefer_W = prefer_W[~mask]

    print('##GUESSTHEKARMA - Dataset loaded##')

    prefer_W = pd.DataFrame(data=prefer_W)
    # predict_W = pd.DataFrame(data=predict_W)

    p = AnalysisPipeline(prefer_W, AnonymousBayesianCombiner(), CrossEntropyScore.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [.5, .5]), max_k=20,
                         num_runs=2)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, ABC w/ CrossEntropy")
    print(results)

    exit(0)

    p = AnalysisPipeline(prefer_W, AnonymousBayesianCombiner(), F1Score.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [.5, .5]), max_k=20,
                         num_runs=1)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, ABC w/ F1")
    print(results)


    p = AnalysisPipeline(prefer_W, FrequencyCombiner(), CrossEntropyScore.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [.5, .5]), max_k=20,
                         num_runs=1)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, Freq w/ CrossEntropy")
    print(results)

    p = AnalysisPipeline(prefer_W, FrequencyCombiner(), F1Score.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [1, 0]), max_k=20,
                         num_runs=1)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, Freq w/ F1")
    print(results)


    for i in range(15):
        thresh = .65 + .01 * i
        print(f"\tsurvey equivalence for {thresh} is ", p.power_curve.compute_equivalence(thresh))

def wiki_toxicity():
    """
    With The wikipedia toxicity study we do not have individual raters, so we simulate them. Since the Frequency
    and AnonymousBayesian combiners are both anonymous, ie, it doesn't matter who rated what, then we can just simulate
    these raters.
    :return: None
    """
    wiki = pd.read_csv('./data/wiki_toxicity.csv')

    max_annotators = max(wiki[:1000]['n_annotators'])

    W_list = list()
    for index, item in wiki.iterrows():
        if index > 1000: break
        n_attack = int(item['n_attack'])
        n_annotators = int(item['n_annotators'])
        vec = np.zeros(max_annotators, dtype=str)
        for i in range(n_attack):
            vec[i] = 'p'
        for i in range(n_annotators-n_attack):
            vec[i+n_attack] = 'n'
        shuffle(vec)
        W_list.append(vec)
    W = np.stack(W_list)


    W = pd.DataFrame(data=W)



    print('##WIKI TOPXCITIY - Dataset loaded##')

    p = AnalysisPipeline(W, AnonymousBayesianCombiner(), CrossEntropyScore.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [.5, .5]), max_k=10,
                         num_runs=10)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, ABC w/ CrossEntropy")
    print(results)


    p = AnalysisPipeline(W, AnonymousBayesianCombiner(), F1Score.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [.5, .5]), max_k=10,
                         num_runs=10)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, ABC w/ F1")
    print(results)


    p = AnalysisPipeline(W, FrequencyCombiner(), CrossEntropyScore.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [1, 0]), max_k=10,
                         num_runs=10)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, Freq w/ CrossEntropy")
    print(results)


    p = AnalysisPipeline(W, FrequencyCombiner(), F1Score.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [1, 0]), max_k=10,
                         num_runs=10)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, Freq w/ F1")
    print(results)

def cred_web():
    """
    With The Credibility study we do not have individual raters, so we simulate them. Since the Frequency
    and AnonymousBayesian combiners are both anonymous, ie, it doesn't matter who rated what, then we can just simulate
    these raters.
    In this case, we say that a rating was positive if credibility was rated high or medium high. Otherwise its not.
    :return:
    """
    wiki = pd.read_csv('./data/credweb.csv')

    max_annotators = 30 # this is just via the experiment

    W_list = list()
    for index, item in wiki.iterrows():
        low = int(item['credCount-2_ret'])
        lowmed = int(item['credCount-1_ret'])
        med = int(item['credCount0_ret'])
        highmed = int(item['credCount1_ret'])
        high = int(item['credCount2_ret'])

        vec = np.zeros(max_annotators, dtype=str)
        for i in range(highmed+high):
            vec[i] = 'p'
        for i in range(low+lowmed+med):
            vec[i+highmed+high] = 'n'
        shuffle(vec)
        W_list.append(vec)
    W = np.stack(W_list)


    W = pd.DataFrame(data=W)



    print('##CREDWEB - Dataset loaded##')

    p = AnalysisPipeline(W, AnonymousBayesianCombiner(), CrossEntropyScore.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [.5, .5]), max_k=20,
                         num_runs=10)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, ABC w/ CrossEntropy")
    print(results)


    p = AnalysisPipeline(W, AnonymousBayesianCombiner(), F1Score.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [.5, .5]), max_k=20,
                         num_runs=10)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, ABC w/ F1")
    print(results)


    p = AnalysisPipeline(W, FrequencyCombiner(), CrossEntropyScore.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [1, 0]), max_k=20,
                         num_runs=10)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, Freq w/ CrossEntropy")
    print(results)


    p = AnalysisPipeline(W, FrequencyCombiner(), F1Score.score, allowable_labels=['p', 'n'],
                         null_prediction=DiscreteDistributionPrediction(['p', 'n'], [1, 0]), max_k=20,
                         num_runs=10)
    results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print("###10 runs, Freq w/ F1")
    print(results)

def main():
    guessthekarma()
    cred_web()
    wiki_toxicity()

main()