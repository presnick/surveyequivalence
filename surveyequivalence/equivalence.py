from abc import ABC, abstractmethod
from typing import Sequence, Dict, Tuple, Callable
import numpy as np
import pandas as pd
import random
import math

from .combiners import Prediction, Combiner
from .scoring_functions import Scorer
from surveyequivalence import DiscreteState
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



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
                 verbosity=1
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

        # Either compute expert or amateur power curve; don't use same pipeline for both
        self.compute_power_curve(num_runs, amateur = len(amateur_cols)>0)

    def compute_power_curve(self, num_runs, amateur=False):

            run_results = []
            if self.verbosity > 0:
                print(f"starting {'amateur' if amateur else 'expert'} power curve: {num_runs} runs to go")
            for i in range(num_runs):
                remaining = num_runs-i
                run_results.append(self.compute_one_power_run(amateur))
                if remaining % 10 == 0 and self.verbosity > 0:
                    print(remaining)
                else:
                    print(".", end='', flush=True)
            if amateur:
                self.amateur_power_curve = PowerCurve(run_results)
            else:
                self.expert_power_curve = PowerCurve(run_results)


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
    def __init__(self,
                 ax,
                 expert_power_curve,
                 amateur_power_curve=None,
                 classifier_scores:Dict =None,
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
        self.center_on_c0 = center_on_c0 # whether to subtract out c_0 from all values, in order to plot gains over baseline
        self.y_range = y_range
        self.name = name
        self.x_intercepts = []
        self.legend_label = legend_label
        self.amateur_legend_label = amateur_legend_label
        self.ax = ax
        self.format_ax()
        # self.make_fig_and_axes()

    def format_ax(self):
        xlabel = 'Number of raters'
        ylabel = self.y_axis_label
        self.ax.set_xlabel(xlabel, fontsize=16)
        self.ax.set_ylabel(ylabel, fontsize=16)
        self.ax.set_title(self.name)

    def add_state_distribution_inset(self, dataset_generator):
        ymax = self.y_range[1] if self.y_range else self.ymax

        if self.possibly_center_score(self.expert_power_curve.means.iloc[-1]) < .66 *ymax:
            print(f"loc 1. c_k = {self.possibly_center_score(self.expert_power_curve.means.iloc[-1])}; ymax={ymax}")
            loc = 1
        else:
            print("loc 5")
            loc = 5

        inset_ax = inset_axes(self.ax, width='30%', height='20%', loc=loc)

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
            ax.scatter(survey_equiv, score, c=color)
            if include_droplines:
                print(f"adding dropline at {survey_equiv} from {self.ymin} to {score}")
                self.x_intercepts.append(survey_equiv)
                ax.vlines(x=survey_equiv, color=color, linewidths=2, linestyles='dashed', ymin=self.y_range_min, ymax=score)
            else:
                print("include_droplines is False")

    @property
    def y_range_min(self):
        return self.y_range[0] if self.y_range else self.ymin

    def set_ymin(self):
        ymin = min(self.expert_power_curve.means)
        if (self.amateur_power_curve):
            ymin = min(ymin, min(self.amateur_power_curve.means))
        if self.classifier_scores:
            for score in self.classifier_scores.values():
                ymin = min(ymin, score)

        self.ymin = self.possibly_center_score(ymin)

    def set_ymax(self):
        ymax = max(self.expert_power_curve.means)
        if (self.amateur_power_curve):
            ymax = max(ymax, max(self.amateur_power_curve.means))
        if self.classifier_scores:
            for score in self.classifier_scores.values():
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
            points = curve.means.index

        ax.errorbar(curve.means[points].index,
                    [self.possibly_center_score(score) for score in  curve.means[points]],
                    yerr=curve.cis[points],
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
             include_classifier_amateur_equivalences=False,
             include_droplines=True,
             include_amateur_curve=True,
             amateur_equivalences=[]):

        ax = self.ax

        if include_expert_points:
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
            # self.classifier_scores is a dictionary with classifier names as keys
            for (classifier_name, score) in self.classifier_scores.items():
                color = self.color_map[classifier_name] if classifier_name in self.color_map else 'black'
                self.add_classifier_line(ax, classifier_name, self.possibly_center_score(score), color)
                if include_classifier_equivalences:
                    self.add_survey_equivalence_point(ax,
                                                      self.expert_power_curve.compute_equivalence(score),
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines)
                if include_classifier_amateur_equivalences:
                    self.add_survey_equivalence_point(ax,
                                                      self.amateur_power_curve.compute_equivalence(score),
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines)
        for idx in amateur_equivalences:
            score = self.amateur_power_curve.means[idx]
            survey_eq = self.expert_power_curve.compute_equivalence(score)
            print(f"k={idx}: score={score} expert equivalence = {survey_eq}")
            survey_eq = survey_eq if type(survey_eq)!=str else 0
            ax.hlines(y=self.possibly_center_score(score),
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

        ax.legend(loc='upper right')

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
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xtick_formatter))
        # fig.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xtick_formatter))

        pass