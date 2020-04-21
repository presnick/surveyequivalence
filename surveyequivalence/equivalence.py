from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple, Callable
import numpy as np
import pandas as pd
import random

from .combiners import Prediction
from matplotlib import pyplot as plt
import matplotlib

N = 1000


class PowerCurve:

    def __init__(self, means: Dict[int, float] = None, cis: Dict[int, Tuple[float, float]] = None):
        self.means = means
        self.cis = cis
        self.results = []  # each item will be one dictionary with scores at different k

    def compute_means(self):
        # if we had many runs, get the mean at each k
        pass

    def compute_cis(self, width=.95):
        # if we had many runs, get the 2.5th and 97.5th centiles of the distribution at each k
        pass

    def plot(self, ax: matplotlib.axes.Axes):
        pass



class AnalysisPipeline:

    def __init__(self,
                 W: pd.DataFrame,
                 combiner: Callable[[Sequence[str], np.array, str, str], Prediction],
                 scoring_function: Callable[[Sequence[Prediction], Sequence[str]], float],
                 allowable_labels: Sequence[str]
                 ):
        self.cols = W.columns[:-1]
        self.W = W.to_numpy()
        self.combiner = combiner
        self.scoring_function = scoring_function
        self.allowable_labels = allowable_labels
        self.power_curve = PowerCurve()  # a sequence of results from calling compute_power_curve some number of times

    def array_choice(self, k: int, n: int):
        choice = np.zeros(k, dtype=int)
        arr = np.zeros(n, dtype=int)
        arr[:k] = 1
        np.random.shuffle(arr)
        idx = 0
        for i, c in enumerate(arr):
            if c == 1:
               choice[idx] = i
               idx += 1
        return choice

    def compute_one_power_run(self, K: int) -> Dict[int, float]:
        assert(K>0)

        result = dict()

        N = len(self.W)

        for k in range(1,K+1): #TODO check 1, and K
            predictions = list()
            reference_ratings = list()

            # Sample N rows from the rating matrix W with replacement
            I = self.W[np.random.choice(self.W.shape[0], N, replace=True)]
            #I = self.W.sample(N, replace=True, axis='index')
            predictions = list()
            reference_ratings = list()

            #for each item/row in sample
            for index, item in enumerate(I):
                item = item[:-1]
                true_label = item[-1]

                #sample ratings from nonzero ratings of the item
                nonzero_itm_mask = np.nonzero(item)
                nonzero_itms = item[nonzero_itm_mask]
                nonzero_cols = self.cols[nonzero_itm_mask]
                assert(len(nonzero_itms) == len(nonzero_cols))
                choice_mask = self.array_choice(k+1, len(nonzero_cols))
                sample_ratings = nonzero_itms[choice_mask]
                sample_cols = list(nonzero_cols[choice_mask])

                #sample_ratings = np.random.choice(item[np.nonzero(item)], k+1) # CANT HAAVE MISSINGS
                #sample_ratings = item.sample(k+1)
                rating_tups = list(zip(sample_cols, sample_ratings))
                reference_ratings.append(true_label)
                predictions.append(self.combiner(self.allowable_labels,
                                            rating_tups))

            result[k] = self.scoring_function(predictions, reference_ratings)
        return result



def make_power_curve_graph(expert_scores, amateur_scores, classifier_scores, points_to_show_surveyEquiv=None):

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111)

    # If there are expert_scores and show lines is false:
    if expert_scores and not expert_scores['Show_lines']:
            x = list(expert_scores['Power_curve']['k'])
            y = list(expert_scores['Power_curve']['score'])
            yerr = list(expert_scores['Power_curve']['confidence_radius'])

            ax.errorbar(x, y, yerr=yerr, marker='o',color = expert_scores['color'], elinewidth = 2, capsize = 5,label=expert_scores['name'], linestyle='')


    # If there are expert_scores and show_lines is true:
    if expert_scores and expert_scores['Show_lines']:
        x = list(expert_scores['Power_curve']['k'])
        y = list(expert_scores['Power_curve']['score'])
        yerr = list(expert_scores['Power_curve']['confidence_radius'])

        ax.errorbar(x, y, yerr=yerr, marker='o',color = expert_scores['color'], elinewidth = 2, capsize = 5,label=expert_scores['name'], linestyle='-')



    # If there are amateur_scores show_lines is false
    if amateur_scores and not amateur_scores[0]['Show_lines']:
        x=[]
        y=[]
        yerr=[]

        for i in (range(len(amateur_scores))):
            x.append(list(amateur_scores[i]['Power_curve']['k']))
            y.append(list(amateur_scores[i]['Power_curve']['score']))
            yerr.append(list(amateur_scores[i]['Power_curve']['confidence_radius']))

        for i in range(len(amateur_scores)):
            ax.errorbar(x[i], y[i], yerr=yerr[i], marker='o',color = amateur_scores[i]['color'], label=amateur_scores[i]['name'],elinewidth = 2, capsize = 5, linestyle='')



    # If there are amateur_scores and show_lines is true:
    if amateur_scores and amateur_scores[0]['Show_lines']:
        x=[]
        y=[]
        yerr=[]

        for i in (range(len(amateur_scores))):
            x.append(list(amateur_scores[i]['Power_curve']['k']))
            y.append(list(amateur_scores[i]['Power_curve']['score']))
            yerr.append(list(amateur_scores[i]['Power_curve']['confidence_radius']))

        for i in range(len(amateur_scores)):
             ax.errorbar(x[i] , y[i],yerr=yerr[i],linestyle='-',marker='o',color = amateur_scores[i]['color'],label=amateur_scores[i]['name'],elinewidth = 2, capsize = 5)


    #if classifier_scores has a confidence interval:
    if classifier_scores['confidence_radius'].empty is False:
        ci=[float(i) for i in classifier_scores['confidence_radius'].to_list()]
        for (i, score) in enumerate(classifier_scores['scores']):
            y_=float(classifier_scores['scores'].iloc[i])
            ax.axhline(y=y_,  color=classifier_scores['colors'].iloc[i],linewidth=2, linestyle='dashed',label=classifier_scores['names'].iloc[i])
            ax.axhspan(y_ - ci[i],y_+ ci[i], alpha=0.5, color=classifier_scores['colors'].iloc[i])



    #if classifier_scores has no confidence interval:
    if classifier_scores['confidence_radius'].empty:
        # if there are Classifier_scores:
        if classifier_scores['scores'].empty is False:
            for (i, score) in enumerate(classifier_scores['scores']):
                axhline(y=score,  color=classifier_scores['colors'].iloc[i],linewidth=2, linestyle='dashed',label=classifier_scores['names'].iloc[i])


    #If  Points_to_show_SurveyEquiv exists:
    if points_to_show_surveyEquiv is not None:


        expert_scores_copy = expert_scores['Power_curve'].copy()
        f = expert_scores_copy['k']==0

        expert_scores_copy.where(f, inplace = True)

        expert_score_at_0 = expert_scores_copy.dropna()['score'].iloc[0]

    #if  (score, which is the y value at point x)<expert score at 0 return 0
    #[does this means plot at point 0,0?]-------------------------------
        x_intercepts=[0,54]

        for i in range(len(se)):
                #else: get min(k:where expert score at k>our score)

                #if expert line never above our score, return 1> maximum number of expert raters.

                if se[i]['which_type'] == 'classifier':

                    classifier_copy = classifier_scores.copy()
                    f = classifier_copy['names']==se[i]['name']
                    classifier_copy.where(f, inplace = True)
                    y= float( classifier_copy.dropna()['scores'].iloc[0])
                    x_intercept = np.interp(y, expert_scores['Power_curve']['score'].to_list(), expert_scores['Power_curve']['k'].to_list())

                    x_intercepts.append(x_intercept)
                    plt.scatter(x_intercept, y,c='black')
                    plt.axvline(x=x_intercept,  color='black',linewidth=2, linestyle='dashed',ymax =y)


                if se[i]['which_type'] == 'amateur':

                    for j in amateur_scores:
                        if se[i]['name'] == j['name']:
                            #find the points that make up the line
                            x = j['Power_curve']['k'].tolist()
                            y = j['Power_curve']['score'].tolist()
                            #given x_value, find the corresponding y value for that point on the line
                            y_intercept_at_x_value= np.interp(se[i]['which_x_value'], x,y)
#                             if (y_intercept_at_x_value<expert_score_at_0):
#                                 print('y_intercept_at_x_value<expert_score_at_0')

                            x_intercept = np.interp(y_intercept_at_x_value, expert_scores['Power_curve']['score'].to_list(), expert_scores['Power_curve']['k'].to_list())
                            x_intercepts.append(x_intercept)
                            plt.scatter(x_intercept, y_intercept_at_x_value,c='black')
                            plt.axvline(x=x_intercept,  color='black',linewidth=2, linestyle='dashed',ymax =y_intercept_at_x_value)
        x_intercepts.sort()
        plt.xticks([i for i in x_intercepts])
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))






    ax.axis([0, 54, 0, 1])
    ax.set_xlabel('Number of other journalists', fontsize = 16)
    ax.set_ylabel('Correlation with reference journalist', fontsize = 16)
    plt.legend(loc='upper right')
    plt.show()
