import math
import random
from itertools import combinations
from typing import Sequence, Dict

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
                 runs_actual: Sequence[Dict[int, float]],
                 runs_bootstrap: Sequence[Dict[int, float]]):
        """each run will be one dictionary with scores at different k
        """
        self.actual_df = pd.DataFrame(runs_actual)
        self.bootstrap_df = pd.DataFrame(runs_bootstrap)
        self.compute_means_and_cis()

    def compute_means_and_cis(self):
        self.means = self.actual_df.mean()
        self.std = self.actual_df.std()
        bootstrap_std = self.bootstrap_df.std()
        bootstrap_means = self.bootstrap_df.mean()
        self.bootstrap_means = bootstrap_means
        self.bootstrap_std = bootstrap_std
        self.lower_bounds = bootstrap_means - bootstrap_std
        self.upper_bounds = bootstrap_means + bootstrap_std

    @property
    def max_value(self):
        return max(max(self.means), max(self.upper_bounds))

    @property
    def min_value(self):
        return min(min(self.means), min(self.lower_bounds))


class PowerCurve(ClassifierResults):
    def compute_equivalence(self, classifier_score: int):
        """
        :param classifier_score:
        :return: number of raters s.t. expected score == classifier_score
        """
        means = self.means.to_dict()
        better_ks = [k for (k, v) in means.items() if v > classifier_score]
        first_better_k = min(better_ks, default=0)
        if len(better_ks) == 0:
            return f">{max([k for (k, v) in means.items()])}"
        elif first_better_k - 1 in means:
            dist_to_prev = means[first_better_k] - means[first_better_k - 1]
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
                 sparse_experts=True,
                 expert_cols: Sequence[str] = None,
                 amateur_cols: Sequence[str] = None,
                 classifier_predictions: pd.DataFrame = None,
                 combiner: Combiner = None,
                 scorer: Scorer = None,
                 allowable_labels: Sequence[str] = None,
                 null_prediction: Prediction = None,
                 min_k=0,
                 num_pred_samples=1000,
                 num_item_samples=1000,
                 same_ref_rater_all_items=False,
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
        self.num_pred_samples = num_pred_samples
        self.num_item_samples = num_item_samples
        self.same_ref_rater_all_items = same_ref_rater_all_items
        self.verbosity = verbosity

        self.label_pred_samples = self.enumerate_ref_raters()

        self.bootstrap_item_samples = self.generate_item_samples(self.num_item_samples,
                                                                 bootstrap=True,
                                                                 same_ref_rater_all_items=same_ref_rater_all_items)
        self.actual_item_samples = self.generate_item_samples(self.num_item_samples,
                                                              bootstrap=False,
                                                              same_ref_rater_all_items=same_ref_rater_all_items)

        if self.classifier_predictions is not None:
            self.classifier_scores = self.compute_classifier_scores()

        self.add_pred_samples(self.expert_cols, min_k,
                              max_k=len(self.expert_cols) - 1,
                              source_name="expert",
                              num_samples=self.num_pred_samples,
                              sparse=self.sparse_experts,
                              exclude_ref_rater=True)

        self.expert_power_curve = self.compute_power_curve(max_k=len(self.expert_cols) - 1)

        if len(self.amateur_cols) > 0:
            self.add_pred_samples(self.amateur_cols, min_k,
                                  max_k=len(self.amateur_cols),
                                  source_name="amateur",
                                  num_samples=self.num_pred_samples,
                                  sparse=False,
                                  exclude_ref_rater=False)
            print("adding amateur_power_curve")
            self.amateur_power_curve = self.compute_power_curve(max_k=len(self.amateur_cols),
                                                                source_name="amateur"
                                                                )
        else:
            print("no amateur power curve")

    def enumerate_ref_raters(self):
        samples = {}
        for index, item in self.W.iterrows():
            samples[index] = {}
            reference_raters = item[self.expert_cols].dropna()
            for ref_rater, label in reference_raters.items():
                samples[index][ref_rater] = {'label': label}
        return samples

    def generate_item_samples(self, num_samples=1, bootstrap=False, same_ref_rater_all_items=True):
        # each set of items has associated ref. raters
        # that way all predictors for a set of items will be compared to the same set of reference raters,
        # eliminating one source of noise in the comparisons.

        def pick_ref_rater(item_index):
            item_d = self.label_pred_samples[item_index]
            if len(item_d) > 0:
                return random.choice(list(item_d.keys()))
            else:
                return None

        def generate_item_sample(bootstrap):
            if bootstrap:
                items = self.W.sample(len(self.W), replace=True).index
            else:
                # use the exact items, no resampling
                items = self.W.index

            if same_ref_rater_all_items:
                # pick reference rater from those available for first item
                # that has at least one reference rater
                for item in items:
                    single_ref_rater = pick_ref_rater(item)
                    if single_ref_rater:
                        break

                # omit items not rated by reference rater
                return [(item_index, single_ref_rater) \
                        for item_index in items \
                        if single_ref_rater in self.label_pred_samples[item_index]]
            else:
                # pick an available ref. rater for each item
                # omit the item if there are no possible reference raters
                return [y for y in ((item_index, pick_ref_rater(item_index)) for item_index in items) if y[1]]

        return [generate_item_sample(bootstrap=bootstrap) for _ in range(num_samples)]

    def add_pred_samples(self, col_names, min_k, max_k, source_name="expert", num_samples=100,
                         sparse=False, exclude_ref_rater=True):

        if self.verbosity >= 2:
            print(
                f"add_pred_samples: {col_names}, min_k = {min_k} max_k={max_k}, {source_name}, exclude_ref_rater={exclude_ref_rater}")

        def sample_raters(available_raters, k):
            selected_raters = available_raters.sample(k)
            if not sparse:
                # raters supposed to have rated every item, but might have missed a few
                # they still count as one of the k raters, but ignore them for this item
                selected_raters = selected_raters.dropna()
            return selected_raters

        def make_prediction(rating_tups):
            # compute the prediction for the selected raters
            return self.combiner.combine(self.allowable_labels, rating_tups, self.W_as_array,
                                         to_predict_for=ref_rater, item_id=index)

        def generate_sample(available_raters, k):
            if len(available_raters) < k:
                if self.verbosity >= 2:
                    print(f"\t\tskipping item {index} for k={k}: not enough raters available")
                return None

            # no need to generate samples if the number of combinations is less than n_choose_r(num_raters, k)
            num_comb = min(num_samples, scipy.special.comb(len(available_raters), k))
            samples = list()
            if num_comb < num_samples:
                for x in list(combinations(zip(available_raters.index, available_raters),k)):
                    if len(x) == 0: x = []
                    samples.append(make_prediction(x))
            else:
                for _ in range(num_samples):
                    selected_raters = sample_raters(available_raters, k)
                    rating_tups = list(zip(selected_raters.index, selected_raters))
                    samples.append(make_prediction(rating_tups))
            # if not enough raters to make a prediction, then omit
            return [s for s in samples if s]

        for index, item in self.W.iterrows():
            ###TODO: error checks for not enough raters
            for ref_rater in self.label_pred_samples[index]:
                available_raters = item[col_names]
                if sparse:
                    # always get k raters who labeled the item, rather than any k
                    available_raters = available_raters.dropna()
                if exclude_ref_rater:
                    available_raters = available_raters.drop(ref_rater)

                for k in range(min_k, max_k + 1):
                    if k <= len(available_raters):
                        if k not in self.label_pred_samples[index][ref_rater]:
                            self.label_pred_samples[index][ref_rater][k] = {}
                        if source_name not in self.label_pred_samples[index][ref_rater][k]:
                            self.label_pred_samples[index][ref_rater][k][source_name] = []
                        self.label_pred_samples[index][ref_rater][k][source_name] += \
                            generate_sample(available_raters, k)

    def compute_classifier_scores(self):
        if self.verbosity > 0:
            print(f"starting classifiers: computing scores")

        def compute_scores(predictions_df):
            return {col_name: self.scorer.score(predictions_df[col_name],
                                                    predictions_df['ref_label'],
                                                    self.verbosity) \
                    for col_name in self.classifier_predictions.columns}

        def compute_one_run(sample):
            idxs = [item_index for (item_index, ref_rater) in sample]
            ref_labels = pd.Series(
                [self.label_pred_samples[item_index][ref_rater]['label'] for (item_index, ref_rater) in sample],
                name='ref_label').reset_index()
            # items may not be the same items from self.classifier_predictions; look up prediction for the right item
            predictions_df = self.classifier_predictions.loc[idxs, :].reset_index()
            return compute_scores(pd.concat([ref_labels, predictions_df], axis=1))

        ## Each item sample is one run

        ## means of scores for non-bootstrap item samples (with different ref. raters) is the overall score
        run_results_actual = [compute_one_run(sample) for sample in self.actual_item_samples]

        ## cis computed from distribution of scores for bootstrap item samples
        run_results_bootstrap = [compute_one_run(sample) for sample in self.bootstrap_item_samples]

        return ClassifierResults(run_results_actual, run_results_bootstrap)

    def compute_power_curve(self, max_k, source_name="expert"):

        min_k = self.min_k

        if self.verbosity > 0:
            print(f"starting {source_name} power curve: computing scores")

        def one_item_all_ks(item_index, ref_rater):

            def pick(item_d, ref_rater, k, source_name):
                try:
                    options = self.label_pred_samples[item_index][ref_rater][k][source_name]
                    return random.choice(options)
                except:
                    print(f"skipping item={item_index}; ref_rater={ref_rater}; k={k}; source_name={source_name}")
                    return None

            # pick one prediction at random from the available prediction set for each k
            predictions = [pick(self.label_pred_samples[item_index], ref_rater, k, source_name) \
                           for k in range(min_k, max_k + 1)]
            ref_label = self.label_pred_samples[item_index][ref_rater]['label']
            return [ref_label] + predictions

        def compute_scores(predictions_df):
            return {k: self.scorer.score(predictions_df[f'k={k}'],
                                             predictions_df['ref_label'],
                                             self.verbosity) \
                    for k in range(min_k, max_k + 1)}

        def compute_one_run(sample):
            predictions_df = pd.DataFrame([one_item_all_ks(item_index, ref_rater) \
                                           for (item_index, ref_rater) in sample],
                                          columns=['ref_label'] + [f'k={k}' for k in range(min_k, max_k + 1)])
            return compute_scores(predictions_df)

        ## Each item sample is one run

        ## mean of scores for non-bootstrap samples (with different ref raters) is the power at each k
        run_results_actual = [compute_one_run(sample) for sample in self.actual_item_samples]

        ## cis computed from distribution of scores for bootstrap samples
        run_results_bootstrap = [compute_one_run(sample) for sample in self.bootstrap_item_samples]

        return PowerCurve(run_results_actual, run_results_bootstrap)


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
            ax.axhspan(ci[0], ci[1], alpha=0.5, color=color)

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
                                         ci=(self.possibly_center_score(
                                             self.classifier_scores.lower_bounds[classifier_name]),
                                             self.possibly_center_score(
                                                 self.classifier_scores.upper_bounds[classifier_name])))
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
            survey_eq = survey_eq if type(survey_eq) != str else 0
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
