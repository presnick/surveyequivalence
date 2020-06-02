from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple, Callable
import numpy as np
import pandas as pd
import os
import random

from .combiners import Prediction, Combiner
from .scoring_functions import Scorer
from matplotlib import pyplot as plt
import matplotlib
from datetime import datetime


N = 1000


class PowerCurve:

    def __init__(self,
                 runs: Sequence[Dict[int, float]]):
        """each run will be one dictionary with scores at different k
        """
        self.df  = pd.DataFrame(runs)
        self.compute_means_and_cis()

    def compute_means_and_cis(self):
        self.means = self.df.mean()
        self.cis = self.df.std() * 2


    def plot_curve(self, ax: matplotlib.axes.Axes, label=None, color='black', show_lines=True):
        if show_lines:
            linestyle = '-'
        else:
            linestyle = ''

        ax.errorbar(self.means.index, self.means, yerr=self.cis,
                    marker='o', color=color,
                    elinewidth=2, capsize=5,
                    label=label, linestyle=linestyle)

    def plot_equivalence(self, ax: matplotlib.axes.Axes, equivalence_value, color='red'):
        pass

    def compute_equivalence(self, classifier_score: int):
        """
        :param classifier_score:
        :return: number of raters s.t. expected score == classifier_score
        """

        means = self.means.to_dict()
        better_ks = [k for (k, v) in means.items() if v>classifier_score]
        first_better_k = min(better_ks, default=0)
        if len(better_ks) == 0:
            return f">{max([k for (k, v) in means.items()])}"
        elif first_better_k-1 in means:
            dist_to_prev = means[first_better_k] - means[first_better_k-1]
            y_dist = means[first_better_k] - classifier_score
            return first_better_k - (y_dist / dist_to_prev)
        else:
            return f"<{first_better_k}"


class AnalysisPipeline:

    def __init__(self,
                 W: pd.DataFrame,
                 combiner: Combiner,
                 scoring_function: Scorer,
                 allowable_labels: Sequence[str],
                 null_prediction: Prediction,
                 num_runs=1
                 ):
        self.cols = W.columns
        self.W = W.to_numpy()
        self.combiner = combiner
        self.scoring_function = scoring_function
        self.allowable_labels = allowable_labels
        self.null_prediction = null_prediction
        max_raters = self.W.shape[1] - 1
        self.power_curve = PowerCurve([self.compute_one_power_run(max_raters) for _ in range(num_runs)])

    @staticmethod
    def array_choice(k: int, n: int):
        choice = np.zeros(k, dtype=int)
        arr = np.zeros(n, dtype=int)
        arr[:k] = 1
        np.random.shuffle(arr)
        idx = 0
        for i, c in enumerate(arr):
            if c == 1:
               choice[idx] = i
               idx += 1
        np.random.shuffle(choice)
        return choice

    def compute_one_power_run(self, K: int) -> Dict[int, float]:
        assert(K>0)

        result = dict()

        N = len(self.W)

        for k in range(1,K+1):

            # Sample N rows from the rating matrix W with replacement
            I = self.W[np.random.choice(self.W.shape[0], N, replace=True)]

            predictions = list()
            reference_ratings = list()

            # for each item/row in sample
            for index, item in enumerate(I):
                """
                Sample ratings from nonzero ratings of the item. This code needs to randomly choose columns 
                and ratings, but because these are seperate variables, we need to first pick a random mask and 
                then apply that mask to the two arrays so that they align.
                """
                nonzero_itm_mask = np.nonzero(item)
                nonzero_itms = item[nonzero_itm_mask]
                nonzero_cols = self.cols[nonzero_itm_mask]

                assert(len(nonzero_itms) == len(nonzero_cols))
                choice_mask = self.array_choice(k+1, len(nonzero_cols))
                sample_ratings = nonzero_itms[choice_mask]
                sample_cols = list(nonzero_cols[choice_mask])

                rating_tups = list(zip(sample_cols, sample_ratings))
                reference_ratings.append(rating_tups[-1][1])

                if k==0:
                    pred = self.null_prediction
                else:
                    pred = self.combiner.combine(self.allowable_labels, rating_tups[0:-1], self.W, item_id=index)
                predictions.append(pred)

            result[k] = self.scoring_function(predictions, reference_ratings)
        return result


class Plot:
    def __init__(self, expert_power_curve, amateur_power_curve=None, classifiers=[]):
        self.expert_power_curve = expert_power_curve
        self.amateur_power_curve = amateur_power_curve
        self.classifiers = classifiers
        self.x_intercepts = []

    def add_classifier_line(self, ax, name, score, color, ci=None):
        ax.axhline(y=score, color=color, linewidth=2, linestyle='dashed', label=name)
        if ci:
            ax.axhspan(score - ci, score + ci, alpha=0.5, color=color)

    def add_survey_equivalence_point(self, ax, survey_equiv, score, color, drop_line=True):
        print(f'x_intercept: {survey_equiv}; {type(survey_equiv)}')
        print(f'type of score: {type(score)}')
        if (type(survey_equiv) == float):
            print("adding point")
            plt.scatter(survey_equiv, score, c=color)
            if drop_line:
                self.x_intercepts.append(survey_equiv)
                print(f"line from {(survey_equiv, self.ymin)} to {(survey_equiv, score)}")
                plt.vlines(x=survey_equiv, color=color, linewidths=2, linestyles='dashed', ymin=self.ymin, ymax=score)

    def set_ymin(self):
        ymin = min(self.expert_power_curve.means)
        if (self.amateur_power_curve):
            ymin = min(ymin, min(self.amateur_power_curve.means))
        if ymin < 0 or ymin > 1:
            ymin -= 1
        self.ymin = ymin

    def set_ymax(self):
        ymax = max(self.expert_power_curve.means)
        if (self.amateur_power_curve):
            ymax = max(ymax, max(self.amateur_power_curve.means))
        if ymax < 0 or ymax > 1:
            ymax += 1
        self.ymax = ymax

    def set_xmax(self):
        self.xmax = 1 + max(max(self.expert_power_curve.means.index),
                            max(self.amateur_power_curve.means.index) if (self.amateur_power_curve!=None) else 0)

    def plot(self):
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        ax = fig.add_subplot(111)

        xlabel='Number of raters'
        ylabel='Agreement with reference rater'
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)

        self.expert_power_curve.plot_curve(ax)
        if self.amateur_power_curve:
            self.amateur_power_curve.plot_curve(ax)

        self.set_ymax()
        self.set_ymin()
        self.set_xmax()

        for c in self.classifiers:
            self.add_classifier_line(ax, c.name, c.score, c.color)
            self.add_survey_equivalence_point(ax, self.expert_power_curve.compute_equivalence(c.score), c.score, c.color)
            self.add_survey_equivalence_point(ax, 2.3, c.score, c.color)
            self.ymax = max(self.ymax, c.score)
            self.ymin = min(self.ymin, c.score)





        ax.axis([0, self.xmax, self.ymin, self.ymax])

        if len(self.classifiers) > 0:
            plt.legend(loc='upper right')


        plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.1f}'))

        if not os.path.isdir('plots'):
            os.mkdir('plots')
        plt.savefig(f'plots/power_curve{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.png')

        pass


def make_power_curve_graph(expert_scores, amateur_scores, classifier_scores, points_to_show_surveyEquiv=None):

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111)

    # # If there are expert_scores and show lines is false:
    # if expert_scores and not expert_scores['Show_lines']:
    #         x = list(expert_scores['Power_curve']['k'])
    #         y = list(expert_scores['Power_curve']['score'])
    #         yerr = list(expert_scores['Power_curve']['confidence_radius'])
    #
    #         ax.errorbar(x, y, yerr=yerr, marker='o',color = expert_scores['color'], elinewidth = 2, capsize = 5,label=expert_scores['name'], linestyle='')
    #
    #
    # # If there are expert_scores and show_lines is true:
    # if expert_scores and expert_scores['Show_lines']:
    #     x = list(expert_scores['Power_curve']['k'])
    #     y = list(expert_scores['Power_curve']['score'])
    #     yerr = list(expert_scores['Power_curve']['confidence_radius'])
    #
    #     ax.errorbar(x, y, yerr=yerr, marker='o',color = expert_scores['color'], elinewidth = 2, capsize = 5,label=expert_scores['name'], linestyle='-')
    #


    # # If there are amateur_scores show_lines is false
    # if amateur_scores and not amateur_scores[0]['Show_lines']:
    #     x=[]
    #     y=[]
    #     yerr=[]
    #
    #     for i in (range(len(amateur_scores))):
    #         x.append(list(amateur_scores[i]['Power_curve']['k']))
    #         y.append(list(amateur_scores[i]['Power_curve']['score']))
    #         yerr.append(list(amateur_scores[i]['Power_curve']['confidence_radius']))
    #
    #     for i in range(len(amateur_scores)):
    #         ax.errorbar(x[i], y[i], yerr=yerr[i], marker='o',color = amateur_scores[i]['color'], label=amateur_scores[i]['name'],elinewidth = 2, capsize = 5, linestyle='')
    #
    #
    #
    # # If there are amateur_scores and show_lines is true:
    # if amateur_scores and amateur_scores[0]['Show_lines']:
    #     x=[]
    #     y=[]
    #     yerr=[]
    #
    #     for i in (range(len(amateur_scores))):
    #         x.append(list(amateur_scores[i]['Power_curve']['k']))
    #         y.append(list(amateur_scores[i]['Power_curve']['score']))
    #         yerr.append(list(amateur_scores[i]['Power_curve']['confidence_radius']))
    #
    #     for i in range(len(amateur_scores)):
    #          ax.errorbar(x[i] , y[i],yerr=yerr[i],linestyle='-',marker='o',color = amateur_scores[i]['color'],label=amateur_scores[i]['name'],elinewidth = 2, capsize = 5)


    # #if classifier_scores has a confidence interval:
    # if classifier_scores['confidence_radius'].empty is False:
    #     ci=[float(i) for i in classifier_scores['confidence_radius'].to_list()]
    #     for (i, score) in enumerate(classifier_scores['scores']):
    #         y_=float(classifier_scores['scores'].iloc[i])
    #         ax.axhline(y=y_,  color=classifier_scores['colors'].iloc[i],linewidth=2, linestyle='dashed',label=classifier_scores['names'].iloc[i])
    #         ax.axhspan(y_ - ci[i],y_+ ci[i], alpha=0.5, color=classifier_scores['colors'].iloc[i])
    #
    #
    #
    # #if classifier_scores has no confidence interval:
    # if classifier_scores['confidence_radius'].empty:
    #     # if there are Classifier_scores:
    #     if classifier_scores['scores'].empty is False:
    #         for (i, score) in enumerate(classifier_scores['scores']):
    #             axhline(y=score,  color=classifier_scores['colors'].iloc[i],linewidth=2, linestyle='dashed',label=classifier_scores['names'].iloc[i])


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
