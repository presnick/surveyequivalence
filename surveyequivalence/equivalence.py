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
from profilehooks import profile



N = 1000


class PowerCurve:

    def __init__(self,
                 runs_actual: Sequence[Dict[int, float]],
                 runs_bootstrap: Sequence[Dict[int, float]]):
        """each run will be one dictionary with scores at different k
        """
        self.actual_df  = pd.DataFrame(runs_actual)
        self.bootstrap_df = pd.DataFrame(runs_bootstrap)
        self.compute_means_and_cis()

    def compute_means_and_cis(self):
        self.means = self.actual_df.mean()
        bootstrap_std = self.bootstrap_df.std()
        bootstrap_means = self.bootstrap_df.mean()
        self.lower_bounds = bootstrap_means - bootstrap_std
        self.upper_bounds = bootstrap_means + bootstrap_std

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


class LabeledItem:
    def __init__(self,
                 item_id,
                 ref_rater_id,
                 ref_rater_label):
        self.item_id = item_id
        self.ref_rater_id = ref_rater_id
        self.ref_rater_label = ref_rater_label

class AnalysisPipeline:

    def __init__(self,
                 W: pd.DataFrame,
                 sparse_raters = False,
                 expert_cols: Sequence[str] = [],
                 amateur_cols: Sequence[str] = [],
                 combiner: Combiner=None,
                 scoring_function: Scorer=None,
                 allowable_labels: Sequence[str]=None,
                 null_prediction: Prediction=None,
                 min_k=0,
                 num_rater_samples=1000,
                 num_item_samples=1000,
                 verbosity=1
                 ):
        if expert_cols:
            self.expert_cols = expert_cols
        else:
            self.expert_cols = W.columns
        self.amateur_cols = amateur_cols
        self.W = W
        self.W_as_array = W.to_numpy()
        self.sparse_raters = sparse_raters
        self.combiner = combiner
        self.scoring_function = scoring_function
        self.allowable_labels = allowable_labels
        self.null_prediction = null_prediction
        self.min_k = min_k
        self.num_rater_samples = num_rater_samples
        self.num_item_samples = num_item_samples
        self.verbosity = verbosity




        self.label_pred_samples = self.gen_ref_rater_samples()

        self.bootstrap_item_samples = self.generate_item_samples(self.num_item_samples, bootstrap=True)
        self.actual_item_samples = self.generate_item_samples(1, bootstrap=False)

        # print(self.bootstrap_item_samples)
        # print(self.actual_item_samples)
        # exit()

        # Either compute expert or amateur power curve; don't use same pipeline for both
        # self.compute_power_curve(num_runs, amateur = len(amateur_cols)>0)

    def gen_ref_rater_samples(self):
        label_pred_samples = {}
        for index, item in self.W.iterrows():
            label_pred_samples[index] = {}
            reference_raters = item[self.expert_cols].dropna()
            for ref_rater, label in reference_raters.items():
                label_pred_samples[index][ref_rater] = {'label': label}
        return label_pred_samples

    def set_pred_samples(self, col_names, min_k, max_k, source_name="expert", sparse=False, exclude_ref_rater=True):
        def sample_raters(available_raters, k):
            selected_raters = available_raters.sample(k)
            if not sparse:
                # raters supposed to have rated every item, but might have missed a few
                # they still count as one of the k raters, but ignore them for this item
                selected_raters = selected_raters.dropna()
            return selected_raters

        def make_prediction(selected_raters):
            # compute the prediction for the selected raters
            rating_tups = list(zip(selected_raters.index, selected_raters))
            return self.combiner.combine(self.allowable_labels, rating_tups, self.W_as_array,
                                         to_predict_for=ref_rater, item_id=index)

        def generate_sample(available_raters, k):
            if len(available_raters) < k:
                if self.verbosity >= 2:
                    print(f"\t\tskipping item {item_id} for k={k}: not enough raters available")
                return None

            samples = [make_prediction(sample_raters(available_raters, k)) for _ in range(self.num_rater_samples)]
            return samples

        for index, item in self.W.iterrows():
            for ref_rater in self.label_pred_samples[index]:
                available_raters = item[col_names]
                if sparse:
                    # always get k raters who labeled the item, rather than any k
                    available_raters = available_raters.dropna()
                if exclude_ref_rater:
                    available_raters = available_raters.drop(ref_rater)

                for k in range(min_k, max_k+1):
                    if k not in self.label_pred_samples[index][ref_rater]:
                        self.label_pred_samples[index][ref_rater][k] = {}
                    self.label_pred_samples[index][ref_rater][k][source_name] = \
                        generate_sample(available_raters, k)

    # def set_expert_pred_samples(self):
    #     for index, item in self.W.iterrows():
    #         for ref_rater in self.label_pred_samples[index]:
    #             available_raters = item[self.expert_cols].dropna().drop(ref_rater)
    #
    #             for k in range(self.min_k, self.max_expert_k):
    #                 if k not in self.label_pred_samples[index][ref_rater]:
    #                     self.label_pred_samples[index][ref_rater][k] = {}
    #                 self.label_pred_samples[index][ref_rater][k]['expert'] = \
    #                     self.generate_pred_samples(available_raters, ref_rater, index, k)
    #
    # def set_amateur_pred_samples(self, col_names, source_name="amateur"):
    #     for index, item in self.W.iterrows():
    #         for ref_rater in self.label_pred_samples[index]:
    #             available_raters = item[col_names].dropna() if self.sparse_raters else item[col_names]
    #             for k in range(self.min_k, self.max_amateur_k):
    #                 if k not in self.label_pred_samples[index][ref_rater]:
    #                     self.label_pred_samples[index][ref_rater][k] = {}
    #                 self.rater_samples[index][ref_rater][k][source_name] = \
    #                     self.generate_pred_samples(available_raters, ref_rater, index, k, remove_na=True)

    def generate_item_samples(self, num_samples=1, bootstrap=False):
        # each set of items has associated ref. raters
        # that way all predictors for a set of item will be compared to the same set of reference raters,
        # eliminating one source of noise in the comparisons

        # fix the choice of ref. rater for each item selected, so that we have coupled randomness when
        # we compare performance of different predictors; all will be compared to same reference rater

        def generate_item_sample(bootstrap):

            ref_raters = [random.choice(self.expert_cols) for _ in range(len(self.W))]
            if bootstrap:
                items = self.W.sample(len(self.W), replace=True).index
            else:
                # use the exact items, no resampling
                items = self.W.index
            return list(zip(items, ref_raters))

        return [generate_item_sample(bootstrap=bootstrap) for _ in range(num_samples)]


    def compute_score(self, item_index, ref_rater):
        null

    def compute_power_curve(self, min_k=None, max_k=None, amateur_cols=None, source_name="expert"):

        if min_k is None:
            min_k = self.min_k

        if max_k is None:
            if amateur_cols is None:
                max_k = len(self.expert_cols) - 1
            else:
                max_k = len(amateur_cols)

        if self.verbosity > 0:
            print(f"starting {source_name} power curve: setting prediction samples")

        if amateur_cols is None:
            self.set_pred_samples(self.expert_cols, min_k, max_k, source_name,
                                  sparse=self.sparse_raters,
                                  exclude_ref_rater=True)

        else:
            self.set_pred_samples(amateur_cols, min_k, max_k, source_name,
                                  sparse=self.sparse_raters,
                                  exclude_ref_rater=False)



        if self.verbosity > 0:
            print(f"starting {source_name} power curve: computing scores")

        def one_item_all_ks(item_index, ref_rater):
            # pick one prediction at random from the available prediction set for each k
            predictions = [random.choice(self.label_pred_samples[item_index][ref_rater][k][source_name]) \
                           for k in range(min_k, max_k + 1)]
            ref_label = self.label_pred_samples[item_index][ref_rater]['label']
            return [ref_label] + predictions


        def compute_scores(predictions_df):
            return {k: self.scoring_function(predictions_df[f'k={k}'],
                                             predictions_df['ref_label'],
                                             self.verbosity) \
                    for k in range(min_k, max_k+1)}


        def compute_one_run(sample):
            predictions_df = pd.DataFrame([one_item_all_ks(item_index, ref_rater) \
                                           for (item_index, ref_rater) in sample],
                                          columns = ['ref_label'] + [f'k={k}' for k in range(min_k, max_k+1)])
            return compute_scores(predictions_df)

        ## Each item sample is one run

        ## mean of scores for non-bootstrap samples is the power at each k
        run_results_actual = [compute_one_run(sample) for sample in self.actual_item_samples]

        ## cis computed from distribution of scores for bootstrap samples
        run_results_bootstrap = [compute_one_run(sample) for sample in self.bootstrap_item_samples]

        return (PowerCurve(run_results_actual, run_results_bootstrap))

    #
    #
    #     #     remaining = len(self.actual_item_samples) - i
    #     #     run_results.append(self.compute_one_power_run(amateur))
    #     #     if remaining % 10 == 0 and self.verbosity > 0:
    #     #         print(remaining)
    #     #     else:
    #     #         print(".", end='', flush=True)
    #     # if amateur:
    #     #     self.amateur_power_curve = PowerCurve(run_results)
    #     # else:
    #     #     self.expert_power_curve = PowerCurve(run_results)
    #
    # def compute_one_power_run(self, amateur=False) -> Dict[int, float]:
    #     verbosity = self.verbosity
    #     if amateur:
    #         K = self.max_amateur_k
    #     else:
    #         K = self.max_expert_k
    #
    #     result = dict()
    #
    #     N = len(self.W) # number of rows in the labels dataframe
    #
    #     print(f'min_k = {self.min_k} K={K}')
    #     for k in range(self.min_k, K+1):
    #         if verbosity >= 2:
    #             print(f'\tk={k}')
    #
    #         # Sample N rows from the rating matrix W with replacement
    #         # I = self.W[np.random.choice(self.W.shape[0], N, replace=True)]
    #
    #         I = self.W.sample(N, replace=True)
    #
    #
    #
    #         predictions = list()
    #         reference_ratings = list()
    #
    #         # for each item/row in sample
    #         # for index, item in enumerate(available_labels):
    #         for index, item in I.iterrows():
    #             """
    #             Sample ratings from nonzero ratings of the item. This code needs to randomly choose column names
    #             and ratings, but because these are separate variables, we need to first pick a random mask and
    #             then apply that mask to the two arrays so that they align.
    #             """
    #
    #             ## get the available raters with non-missing data
    #             if amateur:
    #                 available_raters = item[self.amateur_cols].dropna()
    #                 min_held_out_raters = 0
    #             else:
    #                 available_raters = item[self.expert_cols].dropna()
    #                 min_held_out_raters = 1
    #
    #             ## pick a subset of k raters
    #             ## if not enough available, use the raters available?
    #
    #
    #             if len(available_raters) - min_held_out_raters <= 0:
    #                 if verbosity >= 3:
    #                     print(f"\t\tskipping item {index}: not enough raters available")
    #                 continue
    #             elif len(available_raters) - min_held_out_raters < k :
    #                 if verbosity >= 3:
    #                     print(f"\t\tUsing fewer raters because only {len(available_raters)} available and {k} needed for item {index}")
    #                 selected_raters = available_raters.sample(len(available_raters) - min_held_out_raters)
    #             else:
    #                 selected_raters = available_raters.sample(k)
    #
    #             # get the prediction from those k raters
    #             # rating_tups = zip(sample_cols, sample_ratings)
    #             rating_tups = list(zip(selected_raters.index, selected_raters))
    #             pred = self.combiner.combine(self.allowable_labels, rating_tups, self.W_as_array, item_id=index)
    #
    #             # If k amateurs, all expert labels are available as reference raters
    #             # If k experts, remaining labels are the reference raters;
    #             if amateur:
    #                 reference_raters = item[self.expert_cols].dropna()
    #             else:
    #                 reference_raters = available_raters.drop(selected_raters.index).dropna()
    #
    #             # intuitively: score prediction against each of the reference raters
    #             # but we might have different number of reference raters for different items
    #             # so determine proportions of the different labels among the reference raters
    #
    #             freqs = reference_raters.value_counts()/len(reference_raters)
    #             if len(freqs) == 0:
    #                 if verbosity >=3:
    #                     print(f"\t\tskipping item {index} because no reference raters. k={k}. freqs={freqs}")
    #                 continue
    #             ref_rater_dist = DiscreteState(state_name=f'Ref raters for Item {index}',
    #                                  labels=freqs.index,
    #                                  probabilities=freqs.tolist(),
    #                                  num_raters=len(reference_raters))
    #
    #             predictions.append(pred)
    #             reference_ratings.append(ref_rater_dist)
    #
    #         result[k] = self.scoring_function(predictions, reference_ratings, verbosity)
    #         if verbosity >= 3:
    #             print(f'\tscore {result[k]}')
    #     if verbosity == 2:
    #         print(f'{result}')
    #     return result

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
                    yerr=[[self.possibly_center_score(score) for score in curve.lower_bounds[points]],
                          [self.possibly_center_score(score) for score in curve.upper_bounds[points]]],
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