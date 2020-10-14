import math
import random
from itertools import combinations
from typing import Sequence, Dict, Tuple

import operator
from functools import reduce

import matplotlib
import numpy as np
import pandas as pd
import scipy.special
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .combiners import Prediction, Combiner
from .scoring_functions import Scorer

N = 1000


class ClassifierResults:
    def __init__(self,
                 runs: Sequence[Dict]):
        """
        each run will be one dictionary with scores at different k or scores of different classifiers
        """
        self.df = pd.DataFrame(runs)
        self.compute_means_and_cis()

    def compute_means_and_cis(self):
        self.means = self.df.mean()
        self.stds = self.df.std()
        self.lower_bounds = self.means - 2*self.stds
        self.upper_bounds = self.means + 2*self.stds

    @property
    def max_value(self):
        return max(max(self.means), max(self.upper_bounds))

    @property
    def min_value(self):
        return min(min(self.means), min(self.lower_bounds))


class PowerCurve(ClassifierResults):
    def compute_equivalences(self, classifier_scores: ClassifierResults):
        """
        :param classifier_scores: use dataframe of results for the different bootstrap runs
        :return: number of raters s.t. expected score == classifier_score
        """

        run_results = list()
        for run_idx, row in self.df.iterrows():
            run_equivalences = dict()
            classifier_run = classifier_scores.df.iloc[run_idx]
            for h in classifier_scores.df.columns:
                run_equivalences[h] = self.compute_one_equivalence(classifier_run.loc[h],
                                                                   row.to_dict())
            run_results.append(run_equivalences)
        return pd.DataFrame(run_results)

    def compute_equivalence_at_mean(self, classifier_score):
        means = self.means.to_dict()
        return self.compute_one_equivalence(classifier_score, means)

    @staticmethod
    def compute_one_equivalence(classifier_score, k_powers: Dict):
        better_ks = [k for (k, v) in k_powers.items() if v > classifier_score]
        first_better_k = min(better_ks, default=0)
        if len(better_ks) == 0:
            return max([k for (k, v) in k_powers.items()])
            return f">{max([k for (k, v) in k_powers.items()])}"
        elif first_better_k - 1 in k_powers:
            dist_to_prev = k_powers[first_better_k] - k_powers[first_better_k - 1]
            y_dist = k_powers[first_better_k] - classifier_score
            return first_better_k - (y_dist / dist_to_prev)
        else:
            return 0
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
                 sparse_experts=True,
                 expert_cols: Sequence[str] = [],
                 amateur_cols: Sequence[str] = [],
                 classifier_predictions: pd.DataFrame = None,
                 combiner: Combiner = None,
                 scorer: Scorer = None,
                 allowable_labels: Sequence[str] = None,
                 null_prediction: Prediction = None,
                 min_k=0,
                 num_bootstrap_item_samples=100,
                 max_rater_subsets=200,
                 max_K=10,
                 verbosity=1
                 ):
        if expert_cols:
            self.expert_cols = expert_cols
        else:
            self.expert_cols = W.columns
        self.amateur_cols = amateur_cols
        self.classifier_predictions = classifier_predictions
        self.W = W
        self.W_as_array = W.to_numpy()
        ## sparse_experts=True if we always want k non-null raters for each item
        ## with sparse_experts=False we would get k experts and then remove nulls
        self.sparse_experts = sparse_experts
        self.combiner = combiner
        self.scorer = scorer
        self.allowable_labels = allowable_labels
        self.null_prediction = null_prediction
        self.min_k = min_k
        self.num_bootstrap_item_samples = num_bootstrap_item_samples
        self.verbosity = verbosity

        self.item_samples = self.generate_item_samples(self.num_bootstrap_item_samples)

        if self.classifier_predictions is not None:
            self.classifier_scores = self.compute_classifier_scores()

        self.expert_power_curve = self.compute_power_curve(
            raters=self.expert_cols,
            ref_raters=self.expert_cols,
            min_k=min_k,
            max_k=min(max_K, len(self.expert_cols)) - 1,
            max_rater_subsets=max_rater_subsets)

        if self.amateur_cols is not None and len(self.amateur_cols) > 0:
            self.amateur_power_curve = self.compute_power_curve(
                raters=amateur_cols,
                ref_raters=expert_cols,
                min_k=min_k,
                max_k=min(max_K, len(self.amateur_cols)) - 1,
                max_rater_subsets=max_rater_subsets)

    def output_csv(self, fname):
        # output the dataframe and the expert predictions
        pd.concat([self.classifier_predictions, self.W], axis=1).to_csv(fname)

    def generate_item_samples(self, num_bootstrap_item_samples=0):

        def generate_item_sample():
            return self.W.sample(len(self.W), replace=True).index

        ## return the actual item sample, plus specified number of bootstrap samples
        return [self.W.index] + [generate_item_sample() for _ in range(num_bootstrap_item_samples)]

    def compute_classifier_scores(self):
        if self.verbosity > 0:
            print(f"starting classifiers: computing scores")

        def compute_scores(predictions_df, ref_labels_df):
            return {col_name: self.scorer.score_classifier(predictions_df[col_name],
                                                self.expert_cols,
                                                ref_labels_df) \
                    for col_name in self.classifier_predictions.columns}

        def compute_one_run(idxs):
            predictions_df = self.classifier_predictions.loc[idxs, :].reset_index()
            ref_labels_df = self.W.loc[idxs, :].reset_index()
            return compute_scores(predictions_df, ref_labels_df)

        ## Each item sample is one run
        ## TODO: self.item_samples should just be sets of items, without matched reference raters
        run_results = [compute_one_run(idxs) for idxs in self.item_samples]
        return ClassifierResults(run_results)

    def compute_power_curve(self, raters, ref_raters, min_k, max_k, max_rater_subsets=200):
        ref_raters = set(ref_raters)

        if self.verbosity > 0:
            print(f"\nstarting power curve: computing scores for {raters} with ref_raters {ref_raters}")

        def comb(n, k):
            # from https://stackoverflow.com/a/4941932
            k = min(k, n - k)
            numer = reduce(operator.mul, range(n, n - k, -1), 1)
            denom = reduce(operator.mul, range(1, k + 1), 1)
            return numer // denom

        def rater_subsets(raters, k, max_subsets):
            K = len(raters)
            if comb(K, k) > max_subsets:
                if comb(K, k) > 5 * max_subsets:
                    ## repeatedly grab a random subset and throw it away if it's a duplicate
                    subsets = list()
                    for idx in range(max_subsets):
                        while True:
                            subset = tuple(np.random.choice(raters, k, replace=False))
                            if subset not in subsets:
                                subsets.append(subset)
                                break
                            print(f"repeat rater subset when sampling for idx {idx}; skipping and trying again.")
                    return subsets
                else:
                    ## just enumerate all the subsets and take a sample of max_subsets of them
                    all_k_subsets = list(combinations(raters, k))
                    selected_idxs = np.random.choice(len(all_k_subsets), max_subsets)
                    return [subset for idx, subset in enumerate(all_k_subsets) if idx in selected_idxs]
            else:
                return list(combinations(raters, k))

        def generate_rater_subsets(raters, min_k, max_k, max_subsets) -> Dict[int, Sequence[Tuple[str, ...]]]:
            """
            :param raters: sequence of strings
            :param min_k:
            :param max_k:
            :param max_subsets: integer
            :return: dictionary with k=num_raters as keys; values are sequences of rater tuples, up to max_subsets for each value of k
            """
            retval = dict()
            for k in range(min_k, max_k+1):
                if self.verbosity > 1:
                    print(f"generate_subsets, raters={raters}, k={k}")
                retval[k] = rater_subsets(raters, k, max_subsets)
            return retval

        def generate_predictions(W, ratersets) -> Dict[int, Dict[Tuple[str, ...], Prediction]]:
            if self.verbosity > 0:
                print('\nstarting to precompute predictions for various rater subsets. \nItems processed:')
            predictions = dict()
            item_count = 0
            ## iterate through rows, accumulating predictions for that item
            for idx, row in W.iterrows():
                if self.verbosity > 1:
                    item_count += 1
                    if item_count % 10 == 0:
                        print(item_count, flush=True, end='')
                    else:
                        print(f".", end='', flush=True)

                # max a dictionary with k values as keys
                # special case with k=0 raters: prediction from no labels
                predictions[idx] = {tuple(): self.combiner.combine(
                                    allowable_labels=self.combiner.allowable_labels,
                                    labels=[],
                                    item_id=idx,
                                    W = self.W_as_array)}
                # now get predictions for all non-empty ratersets
                #ratersets = generate_rater_subsets([i for i,x in enumerate(row) if x is not None ], min_k, max_k, max_rater_subsets)

                for k in ratersets:
                    if k > 0:
                        for rater_tup in ratersets[k]:
                            label_vals = row.loc[list(rater_tup)]
                            hasnone = False
                            for val in label_vals.values:
                                if val is None:
                                    hasnone = True
                                    break
                            if hasnone: continue
                            predictions[idx][rater_tup] = self.combiner.combine(
                                allowable_labels=self.combiner.allowable_labels,
                                labels=list(zip(rater_tup, label_vals)),
                                item_id=idx,
                                W = self.W_as_array)

            return predictions

        def compute_one_run(W, idxs, ratersets, predictions, call_count=[0]):
            if self.verbosity > 1:
                call_count[0] += 1
                if call_count[0] % 10 == 0:
                    print(call_count[0], flush=True, end='')
                else:
                    print(f".", end='', flush=True)
            ref_labels_df = W.loc[idxs, :].reset_index()
            power_levels = dict()
            for k in range(min_k, max_k+1):
                if self.verbosity > 2:
                    print(f"compute_one_run, k={k}")
                scores = []
                for raterset in ratersets[k]:
                    preds = list()
                    hasnone = False
                    for idx in idxs:
                        if raterset not in predictions[idx]:
                            hasnone=True
                            break
                        preds.append(predictions[idx][raterset])
                    if hasnone: continue
                    unused_raters = ref_raters - set(raterset)
                    score = self.scorer.score_classifier(
                        preds,
                        unused_raters,
                        ref_labels_df)
                    if score:
                        scores.append(score)
                if len(scores) > 0:
                    power_levels[k] = sum(scores) / len(scores)
                else:
                    power_levels[k] = None
            return power_levels

        ## generate rater samples
        ratersets = generate_rater_subsets(raters, min_k, max_k, max_rater_subsets)

        ## generate predictions
        predictions = generate_predictions(self.W, ratersets)

        ## Each item sample is one run
        if self.verbosity > 0:
            print("\ncomputing power curve results for each bootstrap item sample. Samples processed:")
        run_results = [compute_one_run(self.W, idxs, ratersets, predictions) for idxs in self.item_samples]
        return PowerCurve(run_results)


class Plot:
    def __init__(self,
                 ax,
                 expert_power_curve,
                 amateur_power_curve=None,
                 classifier_scores=None,
                 color_map={'expert_power_curve': 'black', 'amateur_power_curve': 'blue', 'classifier': 'green'},
                 y_axis_label='Agreement with reference rater',
                 center_on_c0=False,
                 y_range=None,
                 name='powercurve',
                 legend_label='Expert raters',
                 amateur_legend_label="Lay raters"
                 ):
        self.expert_power_curve = expert_power_curve
        self.amateur_power_curve = amateur_power_curve
        self.classifier_scores = classifier_scores
        self.color_map = color_map
        self.y_axis_label = y_axis_label
        self.center_on_c0 = center_on_c0  # whether to subtract out c_0 from all values, in order to plot gains over baseline
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

        if self.possibly_center_score(self.expert_power_curve.means.iloc[-1]) < .66 * ymax:
            print(f"loc 1. c_k = {self.possibly_center_score(self.expert_power_curve.means.iloc[-1])}; ymax={ymax}")
            loc = 1
        else:
            print("loc 5")
            loc = 5

        inset_ax = inset_axes(self.ax, width='30%', height='20%', loc=loc)

    def possibly_center_score(self, score):
        if self.center_on_c0 and len(self.expert_power_curve.means) > 0:
            return score - self.expert_power_curve.means[0]
        else:
            return score

    def add_classifier_line(self, ax, name, score, color, ci=None):
        ax.axhline(y=score, color=color, linewidth=2, linestyle='dashed', label=name)
        if ci:
            ax.axhspan(ci[0], ci[1], alpha=0.1, color=color)

    def add_survey_equivalence_point(self, ax, survey_equiv, score, color, include_droplines=True):
        # score is already centered before this is called
        # print(f"add_survey_equivalence_point {survey_equiv} type {type(survey_equiv)}")
        if (type(survey_equiv) != str):
            ax.scatter(survey_equiv, score, c=color)
            if include_droplines:
                # print(f"adding dropline at {survey_equiv} from {self.ymin} to {score}")
                self.x_intercepts.append(survey_equiv)
                ax.vlines(x=survey_equiv, color=color, linewidths=2, linestyles='dashed', ymin=self.y_range_min,
                          ymax=score)
            # else:
            #     print("include_droplines is False")

    @property
    def y_range_min(self):
        return self.y_range[0] if self.y_range else self.ymin

    def set_ymin(self):
        ymin = self.expert_power_curve.min_value
        if (self.amateur_power_curve):
            ymin = min(ymin, self.amateur_power_curve.min_value)
        if self.classifier_scores:
            ymin = min(ymin, self.classifier_scores.min_value)

        self.ymin = self.possibly_center_score(ymin)

    def set_ymax(self):
        ymax = self.expert_power_curve.max_value
        if (self.amateur_power_curve):
            ymax = max(ymax, self.amateur_power_curve.max_value)
        if self.classifier_scores:
            ymax = max(ymax, self.classifier_scores.max_value)

        self.ymax = self.possibly_center_score(ymax)

    def set_xmax(self):
        self.xmax = 1 + max(max(self.expert_power_curve.means.index),
                            max(self.amateur_power_curve.means.index) if (self.amateur_power_curve != None) else 0)

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

        if points == "all":
            points = curve.means.index

        lower_bounds = np.array([self.possibly_center_score(score) for score in curve.lower_bounds[points]])
        upper_bounds = np.array([self.possibly_center_score(score) for score in curve.upper_bounds[points]])
        means = np.array([self.possibly_center_score(score) for score in curve.means[points]])
        lower_error = means - lower_bounds
        upper_error = upper_bounds - means
        ax.errorbar(curve.means[points].index,
                    means,
                    yerr=[lower_error, upper_error],
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
             include_classifier_cis=True,
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
            # self.classifier_scores is an instance of ClassifierResults, with means and cis computed
            for (classifier_name, score) in self.classifier_scores.means.items():
                print(f'{classifier_name} score: {score}, c_0: {self.expert_power_curve.means[0]}')
                color = self.color_map[classifier_name] if classifier_name in self.color_map else 'black'
                print(f'lower bound: {self.classifier_scores.lower_bounds[classifier_name]}')
                print(f'upper bound: {self.classifier_scores.upper_bounds[classifier_name]}')
                self.add_classifier_line(ax,
                                         classifier_name,
                                         self.possibly_center_score(score),
                                         color,
                                         ci=(self.possibly_center_score(self.classifier_scores.lower_bounds[classifier_name]),
                                             self.possibly_center_score(self.classifier_scores.upper_bounds[classifier_name])) \
                                             if include_classifier_cis else None)
                if include_classifier_equivalences:
                    self.add_survey_equivalence_point(ax,
                                                      self.expert_power_curve.compute_equivalence_at_mean(score),
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines)
                if include_classifier_amateur_equivalences:
                    self.add_survey_equivalence_point(ax,
                                                      self.amateur_power_curve.compute_equivalence_at_mean(score),
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines)
        for idx in amateur_equivalences:
            score = self.amateur_power_curve.means[idx]
            survey_eq = self.expert_power_curve.compute_equivalence_at_mean(score)
            print(f"k={idx}: score={score} expert equivalence = {survey_eq}")
            survey_eq = survey_eq if type(survey_eq) != str else 0
            ax.hlines(y=self.possibly_center_score(score),
                      xmin=min(survey_eq, idx),
                      xmax=max(survey_eq, idx),
                      color=self.color_map['amateur_power_curve'],
                      linewidths=2, linestyles='dashed')
            self.add_survey_equivalence_point(ax,
                                              self.expert_power_curve.compute_equivalence_at_mean(score),
                                              self.possibly_center_score(score),
                                              self.color_map['amateur_power_curve'],
                                              include_droplines=include_droplines)

        # ax.axis([0, self.xmax, self.ymin, self.ymax])
        ax.set(xlim=(0, self.xmax))
        ax.set(ylim=self.y_range if self.y_range else (self.ymin, self.ymax))

        ax.legend(loc='upper right')

        integer_equivs = [int(round(x)) for x in self.x_intercepts]
        # print(f"integer_equivs = {integer_equivs}")
        regular_ticks = [i for i in range(0, self.xmax, math.ceil(self.xmax / 10)) if i not in integer_equivs]
        # print(f"regular_ticks = {regular_ticks}")

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
