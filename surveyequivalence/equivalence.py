from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple, Callable
import numpy as np
import pandas as pd
import os
import random
import math

from .combiners import Prediction, Combiner
from .scoring_functions import Scorer
from surveyequivalence import DiscreteState
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
                 expert_cols: Sequence[str] = [],
                 amateur_cols: Sequence[str] = [],
                 combiner: Combiner=None,
                 scoring_function: Scorer=None,
                 allowable_labels: Sequence[str]=None,
                 null_prediction: Prediction=None,
                 num_runs=1,
                 max_expert_k=None,
                 max_amateur_k=None,
                 min_k=0,
                 verbosity=0
                 ):
        if expert_cols:
            self.expert_cols = expert_cols
        else:
            self.expert_cols = W.columns
        self.amateur_cols = amateur_cols
        # self.W = W.to_numpy()
        self.W = W
        self.W_as_array = W.to_numpy()
        self.combiner = combiner
        self.scoring_function = scoring_function
        self.allowable_labels = allowable_labels
        self.null_prediction = null_prediction
        self.min_k = min_k
        self.verbosity = verbosity

        if max_expert_k is None:
            self.max_expert_k = len(self.expert_cols) - 1
        elif max_expert_k > len(self.expert_cols) - 1:
            raise Exception(f"Only {len(self.expert_cols)} raters. Can't compute power curve up to {max_expert_k}")
        else:
            self.max_expert_k = max_expert_k

        if max_amateur_k is None:
            self.max_amateur_k = len(self.amateur_cols) - 1
        elif max_amateur_k > len(self.amateur_cols) - 1:
            raise Exception(f"Only {len(self.amateur_cols)} raters. Can't compute power curve up to {max_amateur_k}")
        else:
            self.max_amateur_k = max_amateur_k

        self.compute_power_curves(num_runs)


    def compute_power_curves(self, num_runs):
        run_results = []
        amateur_run_results = []
        print(f"{num_runs} runs to go:")
        for i in range(num_runs):
            run_results.append(self.compute_one_power_run())
            if self.amateur_cols:
                amateur_run_results.append(self.compute_one_power_run(amateur=True))
            remaining = num_runs-i
            if remaining % 10 == 0:
                print(remaining)
            else:
                print(".", end='', flush=True)

        self.expert_power_curve = PowerCurve(run_results
                                             )

        if self.amateur_cols:
            self.amateur_power_curve = PowerCurve(amateur_run_results
                                                  )
        print("done")


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

    @staticmethod
    def inverted_array_choice(chosen: Sequence, n: int):
        # return a sequence consisting of those positions not present in chosen
        # useful for selected the rest of the columns when chosen was used as a mask
        return list(set(range(n)) - set(chosen))

    def compute_one_power_run(self, amateur=False) -> Dict[int, float]:
        verbosity = self.verbosity
        if amateur:
            K = self.max_amateur_k
        else:
            K = self.max_expert_k

        result = dict()

        N = len(self.W) # number of rows in the labels dataframe

        print(f'min_k = {self.min_k} K={K}')
        for k in range(self.min_k, K+1):
            if verbosity >= 2:
                print(f'\tk={k}')

            # Sample N rows from the rating matrix W with replacement
            # I = self.W[np.random.choice(self.W.shape[0], N, replace=True)]

            I = self.W.sample(N, replace=True)



            predictions = list()
            reference_ratings = list()

            # for each item/row in sample
            # for index, item in enumerate(available_labels):
            for index, item in I.iterrows():
                """
                Sample ratings from nonzero ratings of the item. This code needs to randomly choose column names 
                and ratings, but because these are separate variables, we need to first pick a random mask and 
                then apply that mask to the two arrays so that they align.
                """

                ## get the available raters with non-missing data
                if amateur:
                    available_raters = item[self.amateur_cols].dropna()
                    min_held_out_raters = 0
                else:
                    available_raters = item[self.expert_cols].dropna()
                    min_held_out_raters = 1

                ## pick a subset of k raters
                ## if not enough available, use the raters available?


                if len(available_raters) - min_held_out_raters <= 0:
                    if verbosity >= 3:
                        print(f"\t\tskipping item {index}: not enough raters available")
                    continue
                elif len(available_raters) - min_held_out_raters < k :
                    if verbosity >= 3:
                        print(f"\t\tUsing fewer raters because only {len(available_raters)} available and {k} needed for item {index}")
                    selected_raters = available_raters.sample(len(available_raters) - min_held_out_raters)
                else:
                    selected_raters = available_raters.sample(k)

                # get the prediction from those k raters
                # rating_tups = zip(sample_cols, sample_ratings)
                rating_tups = list(zip(selected_raters.index, selected_raters))
                pred = self.combiner.combine(self.allowable_labels, rating_tups, self.W_as_array, item_id=index)

                # If k amateurs, all expert labels are available as reference raters
                # If k experts, remaining labels are the reference raters;
                if amateur:
                    reference_raters = item[self.expert_cols].dropna()
                else:
                    reference_raters = available_raters.drop(selected_raters.index).dropna()

                # intuitively: score prediction against each of the reference raters
                # but we might have different number of reference raters for different items
                # so determine proportions of the different labels among the reference raters

                freqs = reference_raters.value_counts()/len(reference_raters)
                if len(freqs) == 0:
                    if verbosity >=3:
                        print(f"\t\tskipping item {index} because no reference raters. k={k}. freqs={freqs}")
                    continue
                ref_rater_dist = DiscreteState(state_name=f'Ref raters for Item {index}',
                                     labels=freqs.index,
                                     probabilities=freqs.tolist(),
                                     num_raters=len(reference_raters))

                predictions.append(pred)
                reference_ratings.append(ref_rater_dist)

            result[k] = self.scoring_function(predictions, reference_ratings, verbosity)
            if verbosity >= 3:
                print(f'\tscore {result[k]}')
        if verbosity == 2:
            print(f'{result}')
        return result

class Plot:
    def __init__(self, expert_power_curve, amateur_power_curve=None, classifier_scores=None,
                 color_map={'expert_power_curve': 'black', 'amateur_power_curve': 'blue', 'classifier': 'green'},
                 y_axis_label = 'Agreement with reference rater',
                 center_on_c0 = False,
                 y_range = None,
                 name = 'powercurve',
                 legend_label='Expert raters',
                 amateur_legend_label="Lay raters"
                 ):
        self.expert_power_curve = expert_power_curve
        self.amateur_power_curve = amateur_power_curve
        self.classifier_scores = classifier_scores
        self.color_map=color_map
        self.y_axis_label = y_axis_label
        self.center_on_c0 = center_on_c0 # whether the subtract out c_0 from all values, to plot gains over baseline
        self.y_range = y_range
        self.name = name
        self.x_intercepts = []
        self.legend_label = legend_label
        self.amateur_legend_label = amateur_legend_label
        self.make_fig_and_axes()

    def make_fig_and_axes(self):
        fig, ax = plt.subplots()

        fig.set_size_inches(18.5, 10.5)

        xlabel = 'Number of raters'
        ylabel = self.y_axis_label
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(self.name)

        self.fig = fig
        self.ax = ax

    def add_state_distribution_inset(self, dataset_generator):
        ymax = self.y_range[1] if self.y_range else self.ymax

        if self.possibly_center_score(self.expert_power_curve.means.iloc[-1]) < .66 *ymax:
            print(f"loc 1. c_k = {self.possibly_center_score(self.expert_power_curve.means.iloc[-1])}; ymax={ymax}")
            loc = 1
        else:
            print("loc 5")
            loc = 5

        inset_ax = inset_axes(self.ax, width='30%', height='20%', loc=loc)
        dataset_generator.make_histogram(inset_ax)

    def save_plot(self):
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        self.fig.savefig(f'plots/{self.name}{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.png')

    def possibly_center_score(self, score):
        if self.center_on_c0 and len(self.expert_power_curve.means)>0:
            return score - self.expert_power_curve.means[0]
        else:
            return score

    def add_classifier_line(self, ax, name, score, color, ci=None):
        ax.axhline(y=score, color=color, linewidth=2, linestyle='dashed', label=name)
        if ci:
            ax.axhspan(score - ci, score + ci, alpha=0.5, color=color)

    def add_survey_equivalence_point(self, ax, survey_equiv, score, color, include_droplines=True):
        # score is already centered before this is called
        print(f"add_survey_equivalence_point {survey_equiv} type {type(survey_equiv)}")
        if (type(survey_equiv) != str):
            plt.scatter(survey_equiv, score, c=color)
            if include_droplines:
                print(f"adding dropline at {survey_equiv} from {self.ymin} to {score}")
                self.x_intercepts.append(survey_equiv)
                plt.vlines(x=survey_equiv, color=color, linewidths=2, linestyles='dashed', ymin=self.ymin, ymax=score)
            else:
                print("include_droplines is False")

    def set_ymin(self):
        ymin = min(self.expert_power_curve.means)
        if (self.amateur_power_curve):
            ymin = min(ymin, min(self.amateur_power_curve.means))
        if self.classifier_scores:
            for score in self.classifier_scores:
                ymin = min(ymin, score)

        self.ymin = self.possibly_center_score(ymin)

    def set_ymax(self):
        ymax = max(self.expert_power_curve.means)
        if (self.amateur_power_curve):
            ymax = max(ymax, max(self.amateur_power_curve.means))
        if self.classifier_scores:
            for score in self.classifier_scores:
                ymax = max(ymax, score)

        self.ymax = self.possibly_center_score(ymax)

    def set_xmax(self):
        self.xmax = 1 + max(max(self.expert_power_curve.means.index),
                            max(self.amateur_power_curve.means.index) if (self.amateur_power_curve!=None) else 0)

    def plot_power_curve(self,
                         ax: matplotlib.axes.Axes,
                         curve: PowerCurve,
                         points,
                         connect,
                         color,
                         legend_label='Power curve',
                         ):


        if connect:
            linestyle = '-'
        else:
            linestyle = ''

        if points=="all":
            points = range(len(curve.means))

        def select_idxs(seq, idxs):
            return [elt for (idx, elt) in enumerate(seq) if idx in idxs]

        ax.errorbar(curve.means.index,
                    [self.possibly_center_score(score) for score in  select_idxs(curve.means, points)],
                    yerr=select_idxs(curve.cis, points),
                    marker='o',
                    color=color,
                    elinewidth=2,
                    capsize=5,
                    label=legend_label,
                    linestyle=linestyle)

    def plot(self,
             include_expert_points='all',
             connect_expert_points=True,
             include_classifiers=True,
             include_classifier_equivalences=True,
             include_droplines=True,
             include_amateur_curve=True,
             amateur_equivalences=[]):

        fig, ax = self.fig, self.ax

        self.plot_power_curve(ax,
                              self.expert_power_curve,
                              points=include_expert_points,
                              connect=connect_expert_points,
                              color=self.color_map['expert_power_curve'],
                              legend_label=self.legend_label
                              )


        if self.amateur_power_curve and include_amateur_curve:
            self.plot_power_curve(ax,
                                  self.amateur_power_curve,
                                  points='all',
                                  connect=True,
                                  color=self.color_map['amateur_power_curve'],
                                  legend_label=self.amateur_legend_label
                                  )


        self.set_ymax()
        self.set_ymin()
        self.set_xmax()
        print(f"y-axis range: {self.ymin}, {self.ymax}")

        if include_classifiers:
            # self.classifier_scores is df with classifier names as column names and single row with values
            for (classifier_name, score) in self.classifier_scores.items():
                color = self.color_map[classifier_name] if classifier_name in self.color_map else 'black'
                self.add_classifier_line(ax, classifier_name, self.possibly_center_score(score), color)
                if include_classifier_equivalences:
                    self.add_survey_equivalence_point(ax,
                                                      self.expert_power_curve.compute_equivalence(score),
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines)

        for idx in amateur_equivalences:
            score = self.amateur_power_curve.means[idx]
            survey_eq = self.expert_power_curve.compute_equivalence(score)
            print(f"k={idx}: score={score} expert equivalence = {survey_eq}")
            survey_eq = survey_eq if type(survey_eq)!=str else 0
            plt.hlines(y=self.possibly_center_score(score),
                       xmin=min(survey_eq, idx),
                       xmax=max(survey_eq, idx),
                       color=self.color_map['amateur_power_curve'],
                       linewidths=2, linestyles='dashed')
            self.add_survey_equivalence_point(ax,
                                              self.expert_power_curve.compute_equivalence(score),
                                              self.possibly_center_score(score),
                                              self.color_map['amateur_power_curve'],
                                              include_droplines=include_droplines)

        # ax.axis([0, self.xmax, self.ymin, self.ymax])
        ax.set(xlim=(0, self.xmax))
        ax.set(ylim = self.y_range if self.y_range else (self.ymin, self.ymax))

        fig.legend(loc='upper right')

        integer_equivs = [int(round(x)) for x in self.x_intercepts]
        print(f"integer_equivs = {integer_equivs}")
        regular_ticks = [i for i in range(0, self.xmax, math.ceil(self.xmax / 10)) if i not in integer_equivs]
        print(f"regular_ticks = {regular_ticks}")


        ticks = sorted(regular_ticks + self.x_intercepts)
        ax.set_xticks(ticks)

        def xtick_formatter(x, pos):
            if math.isclose(x, int(round(x)), abs_tol=.001):
                return f"{x:.0f}"
            else:
                return f"{x:.2f}"
        fig.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xtick_formatter))

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
