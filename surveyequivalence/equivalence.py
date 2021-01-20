import math
from string import Template
import pkgutil
from itertools import combinations
from typing import Sequence, Dict, Tuple

import operator
from functools import reduce

import datetime
import pickle
import os

import matplotlib
import numpy as np
import pandas as pd
import scipy.special
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .combiners import Prediction, Combiner
from .scoring_functions import Scorer

def load_saved_pipeline(path):
    W = pd.read_csv(f'{path}/dataset.csv')

    with open(f'{path}/params.pickle', 'rb') as f:
        params = pickle.load(f)

    predictions = pd.read_csv(f'{path}/predictions.csv', index_col=0)

    classifier_scores = PowerCurve(df=pd.read_csv(f'{path}/classifier_scores.csv', index_col=0))

    epc_df = pd.read_csv(f'{path}/expert_power_curve.csv', index_col=0)
    epc_df.columns = epc_df.columns.astype(int)
    expert_power_curve = PowerCurve(df=epc_df)

    try:
        apc_df = pd.read_csv(f'{path}/amateur_power_curve.csv', index_col=0)
        apc_df.columns = apc_df.columns.astype(int)
        amateur_power_curve = PowerCurve(df=apc_df)

        # amateur_power_curve = PowerCurve(df=pd.read_csv(f'{path}/amateur_power_curve.csv'))
    except FileNotFoundError:
        amateur_power_curve = None

    analysis_pipeline = AnalysisPipeline(run_on_creation=False,
                                         W=W,
                                         classifier_predictions=predictions,
                                         **params)

    analysis_pipeline.classifier_scores = classifier_scores
    analysis_pipeline.expert_power_curve = expert_power_curve
    analysis_pipeline.amateur_power_curve = amateur_power_curve
    return analysis_pipeline

class ClassifierResults:
    def __init__(self,
                 runs: Sequence[Dict]=None,
                 df=None):
        """
        each run will be one dictionary with scores at different k or scores of different classifiers
        Alternatively, a dataframe may be passed in with one row for each run

        First row is special: the values for the actual item sample
        Rest of rows are for bootstrap item samples
        """
        if df is not None:
            self.df = df
        else:
            self.df = pd.DataFrame(runs)
        self.compute_means_and_cis()

    def compute_means_and_cis(self):
        self.means = self.df.mean()
        self.stds = self.df.std()
        self.std_lower_bounds = self.means - 2*self.stds
        self.std_upper_bounds = self.means + 2*self.stds
        self.empirical_lower_bounds = self.df.quantile(.025)
        self.empirical_upper_bounds = self.df.quantile(.975)

    @property
    def lower_bounds(self):
        if len(self.df) < 200:
            return self.std_lower_bounds
        else:
            return self.empirical_lower_bounds

    @property
    def upper_bounds(self):
        if len(self.df) < 200:
            return self.std_upper_bounds
        else:
            return self.empirical_upper_bounds

    @property
    def values(self):
        return self.df.iloc[0, :]

    @property
    def max_value(self):
        return max(max(self.means), max(self.upper_bounds))

    @property
    def min_value(self):
        return min(min(self.means), min(self.lower_bounds))

class PowerCurve(ClassifierResults):
    def compute_equivalences(self, other, columns=None):
        """
        :param other: may either be an instance of ClassifierResults or a PowerCurve. Must have same row
                 indexes as self, one for each item sample
        :param columns: a subset of the columns of other.df; if not specified, use all of them
        :return: a df with one row for each bootstrap run, and columns as specified
                  Cell is a float, the survey equivalence value. That is x s.t. expected score with x raters == classifier_score
        """

        if columns is None:
            columns = other.df.columns
        run_results = list()
        for run_idx, row in self.df.iterrows():
            run_equivalences = dict()
            for h in columns:
                run_equivalences[h] = self.compute_one_equivalence(other.df.loc[run_idx, h],
                                                                   row.to_dict())
            run_results.append(run_equivalences)
        return pd.DataFrame(run_results)

    def compute_equivalence_at_mean(self, classifier_score):
        print("computing equivalence at means")
        return self.compute_one_equivalence(classifier_score, self.means.to_dict())

    def compute_equivalence_at_actuals(self, classifier_score):
        print("computing equivalence at actuals")
        return self.compute_one_equivalence(classifier_score, self.values.to_dict())

    def compute_one_equivalence(self, classifier_score, k_powers: Dict = None):
        """
        :param classifier_score: a number, the classifier's score
        :param k_powers: maps integers k to the expected score for that k
        :return: a float, the survey equivalence value
        """
        if not k_powers:
            # compute it for scores for row 0, the values for the actual item sample
            k_powers = self.values.to_dict()
        better_ks = [k for (k, v) in k_powers.items() if v > classifier_score]
        first_better_k = min(better_ks, default=0)
        if len(better_ks) == 0:
            return max([k for (k, v) in k_powers.items()])
            # return f">{max([k for (k, v) in k_powers.items()])}"
        elif first_better_k - 1 in k_powers:
            dist_to_prev = k_powers[first_better_k] - k_powers[first_better_k - 1]
            y_dist = k_powers[first_better_k] - classifier_score
            return first_better_k - (y_dist / dist_to_prev)
        else:
            return 0
            # return f"<{first_better_k}"

    def reliability_of_difference(self, other, k=1):
        """
        :param other: another PowerCurve
        :param k:
        :return: fraction of bootstrap runs where power@k higher for self than other power curve
        """
        df1 = self.df
        df2 = other.df
        return (df1[k] > df2[k]).sum() / len(df1)

    def reliability_of_beating_classifier(self, other, k=1, other_col=1):
        """
        :param other: the other ClassifierResults or PowerCurve
        :param self_col: the survey size (column) for self
        :param other_col: the survey size (column) for other to compare, with matching bootstrap samples as rows
        :return: fraction of bootstrap runs where self power higher than other power
        """
        return (self.df[k] > other.df[other_col]).sum() / len(self.df)


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
                 ratersets_memo=None,
                 predictions_memo=None,
                 item_samples=None,
                 verbosity=1,
                 run_on_creation = True
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
        self.max_K = max_K
        self.num_bootstrap_item_samples = num_bootstrap_item_samples
        self.max_rater_subsets=max_rater_subsets
        self.verbosity = verbosity

        # initialize memoization cache for rater subsets
        if ratersets_memo:
            self.ratersets_memo = ratersets_memo
        else:
            self.ratersets_memo = dict()

        # initialize memoization cache for predictions for rater subsets
        if predictions_memo:
            self.predictions_memo = predictions_memo
        else:
            self.predictions_memo = dict()

        if item_samples:
            self.item_samples = item_samples
        else:
            self.item_samples = self.generate_item_samples(self.num_bootstrap_item_samples)

        if run_on_creation:
            self.run()

    def run(self):
        if self.classifier_predictions is not None:
            self.classifier_scores = self.compute_classifier_scores()

        self.expert_power_curve = self.compute_power_curve(
            raters=self.expert_cols,
            ref_raters=self.expert_cols,
            min_k=self.min_k,
            max_k=min(self.max_K, len(self.expert_cols)) - 1,
            max_rater_subsets=self.max_rater_subsets)

        if self.amateur_cols is not None and len(self.amateur_cols) > 0:
            if self.verbosity > 0:
                print("\n\nStarting to process amateur raters")
            self.amateur_power_curve = self.compute_power_curve(
                raters=amateur_cols,
                ref_raters=expert_cols,
                min_k=min_k,
                max_k=min(max_K, len(self.amateur_cols)) - 1,
                max_rater_subsets=self.max_rater_subsets)


    def save(self, dirname_base="analysis_pipeline", msg="", save_results = True):
        # make a directory for it
        path = f'saved_analyses/{dirname_base}/{datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}'
        try:
            os.mkdir('saved_analyses')
        except FileExistsError:
            pass
        try:
            os.mkdir(f'saved_analyses/{dirname_base}')
        except FileExistsError:
            pass
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        # save the message as a README file
        with open(f'{path}/README', 'w') as f:
            f.write(msg)

        # save the dataset
        self.W.to_csv(f'{path}/dataset.csv')

        # save parameters
        d = dict(
                expert_cols = self.expert_cols,
                amateur_cols = self.amateur_cols,
                sparse_experts =self.sparse_experts,
                # combiner = self.combiner = combiner,   # combiner and scorer are class instances, so can't save this way
                # scorer = self.scorer,
                allowable_labels = self.allowable_labels,
                null_prediction = self.null_prediction,
                min_k = self.min_k,
                num_bootstrap_item_samples = self.num_bootstrap_item_samples,
                max_rater_subsets = self.max_rater_subsets,
                verbosity = self.verbosity,
                ratersets_memo = self.ratersets_memo,
                item_samples = self.item_samples
        )
        with open(f'{path}/params.pickle', 'wb') as f:
            pickle.dump(d, f)

        # save the classifier predictions
        self.classifier_predictions.to_csv(f'{path}/predictions.csv')

        # save the classifier scores
        self.classifier_scores.df.to_csv(f'{path}/classifier_scores.csv')

        # save the expert power curve
        self.expert_power_curve.df.to_csv(f'{path}/expert_power_curve.csv')

        # save the amateur power_curve
        amateur_power_curve = getattr(self, 'amateur_power_curve', None)
        if amateur_power_curve:
            amateur_power_curve.df.to_csv(f'{path}/amateur_power_curve.csv')

        # write out results summary
        if save_results:
            with open(f'{path}/results_summary.txt', 'w') as f:
                f.write("\n----classifier scores-----")
                f.write(f"\tActual item set score:\n {self.classifier_scores.values}")
                f.write(f"\tmeans:\n{self.classifier_scores.means}")
                f.write(f"\tstds:\n{self.classifier_scores.stds}")

                f.write("\n----power curve means-----")
                f.write(f"\tActual item set score: {self.expert_power_curve.values}")
                f.write(f"\tmeans:\n{self.expert_power_curve.means}")
                f.write(f"\tstds:\n{self.expert_power_curve.stds}")

                f.write("\n----survey equivalences----")
                equivalences = self.expert_power_curve.compute_equivalences(self.classifier_scores)
                f.write(f'{equivalences}')
                f.write(f"\tmeans:\n {equivalences.mean()}")
                f.write(f"\tmedians\n {equivalences.median()}")
                f.write(f"\tstddevs\n {equivalences.std()}")

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
                                                ref_labels_df,
                                                verbosity=self.verbosity) \
                    for col_name in self.classifier_predictions.columns}

        def compute_one_run(idxs):
            predictions_df = self.classifier_predictions.loc[idxs, :].reset_index()
            ref_labels_df = self.W.loc[idxs, :].reset_index()
            return compute_scores(predictions_df, ref_labels_df)

        ## Each item sample is one run
        run_results = [compute_one_run(idxs) for idxs in self.item_samples]
        return ClassifierResults(run_results)

    def compute_power_curve(self, raters, ref_raters, min_k, max_k, max_rater_subsets=200):
        ref_raters = set(ref_raters)

        if self.verbosity > 0:
            print(f"\nstarting power curve")
            if self.verbosity > 1:

                print(f"\tcomputing scores for {raters} with ref_raters {ref_raters}")

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
                    result = subsets
                else:
                    ## just enumerate all the subsets and take a sample of max_subsets of them
                    all_k_subsets = list(combinations(raters, k))
                    selected_idxs = np.random.choice(len(all_k_subsets), max_subsets)
                    result = [subset for idx, subset in enumerate(all_k_subsets) if idx in selected_idxs]
            else:
                result = list(combinations(raters, k))
            return result


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
                    print(f"\tgenerate_subsets, k={k}")
                if self.verbosity > 2:
                    print(f"\t\traters={raters}")
                retval[k] = rater_subsets(raters, k, max_subsets)
            return retval

        def add_predictions(W, ratersets, predictions) -> Dict[int, Dict[Tuple[str, ...], Prediction]]:
            # add additional entries in predictions dictionary, for additional items, as necessary
            if self.verbosity > 0:
                print('\nstarting to precompute predictions for various rater subsets. \nItems processed:')
            # predictions = dict()
            item_count = 0
            ## iterate through rows, accumulating predictions for that item
            for idx, row in W.iterrows():
                item_count += 1
                if idx not in predictions:
                    cached = False
                    preds_label = set([])
                    # make a dictionary with rater_tups as keys and prediction outputted by combiner as values
                    predictions[idx] = {}
                    for k in ratersets:
                        for rater_tup in ratersets[k]:
                            label_vals = row.loc[list(rater_tup)].dropna()
                            predictions[idx][rater_tup] = self.combiner.combine(
                                allowable_labels=self.combiner.allowable_labels,
                                labels=list(zip(rater_tup, label_vals)),
                                W=self.W_as_array)
                            if self.verbosity > 0 and idx == 0:
                                if k == 0:
                                    print(f"baseline score:{predictions[idx][rater_tup]}")
                                if k == 1:
                                    preds_label.add(f"{label_vals.values[0] if len(label_vals)>0 else None }: {predictions[idx][rater_tup]}")
                        if self.verbosity > 0 and idx == 0 and k==1:
                            print(f"scores after 1 rating is {preds_label}")
                else:
                    cached = True

                if self.verbosity > 0:
                    if item_count % 10 == 0:
                        print("\t", item_count, flush=True, end='')
                    else:
                        print(f"{'.' if not cached else ','}", end='', flush=True)
            if self.verbosity > 0:
                print()

            return predictions

        def compute_one_run(W, idxs, ratersets, predictions, call_count=[0]):
            if self.verbosity > 0:
                call_count[0] += 1
                if call_count[0] % 10 == 0:
                    print("\t", call_count[0], flush=True, end='')
                else:
                    print(f".", end='', flush=True)
            ref_labels_df = W.loc[idxs, :].reset_index()
            power_levels = dict()
            for k in range(min_k, max_k+1):
                if self.verbosity > 2:
                    print(f"\t\tcompute_one_run, k={k}")
                scores = []
                for raterset in ratersets[k]:
                    preds = [predictions[idx][raterset] for idx in idxs]
                    unused_raters = ref_raters - set(raterset)
                    score = self.scorer.score_classifier(
                        pd.Series(preds),
                        unused_raters,
                        ref_labels_df,
                        self.verbosity
                    )

                    if score:
                        if np.isnan(score):
                            print(
                                f'!!!!!!!!!Unexpected NaN !!!!!! \n\t\t\preds={preds}\nunused_raters={unused_raters}\nscore={score}\ttype(score)={type(score)}')
                        scores.append(score)
                    else:
                        print("ugh")
                if self.verbosity > 1:
                    print(f'\tscores for k={k}: {scores}')
                if len(scores) > 0:
                    power_levels[k] = sum(scores) / len(scores)
                else:
                    power_levels[k] = None
            return power_levels

        ## get rater samples
        canonical_raters_tuple = tuple(sorted(raters))
        if canonical_raters_tuple not in self.ratersets_memo:
            # add result to the memoized cache
            self.ratersets_memo[canonical_raters_tuple] = generate_rater_subsets(raters, min_k, max_k, max_rater_subsets)
        else:
            if self.verbosity > 1:
                print(f"getting cached rater subsets for {canonical_raters_tuple}")
        ratersets = self.ratersets_memo[canonical_raters_tuple]

        ## get predictions
        if canonical_raters_tuple not in self.predictions_memo:
            self.predictions_memo[canonical_raters_tuple] = dict()
        else:
            if self.verbosity > 1:
                print(f"\tgetting some cached predictions for {canonical_raters_tuple}")
                print(f"\tcached predictions found for items {list(self.predictions_memo[canonical_raters_tuple].keys())}")
        add_predictions(self.W, ratersets, predictions=self.predictions_memo[canonical_raters_tuple])
        predictions = self.predictions_memo[canonical_raters_tuple]

        ## Each item sample is one run
        if self.verbosity > 0:
            print("\n\tcomputing power curve results for each bootstrap item sample. \nSamples processed:")
        run_results = [compute_one_run(self.W, idxs, ratersets, predictions) for idxs in self.item_samples]
        if self.verbosity > 1:
            print(f"\n\t\trun_results={run_results}")
        return PowerCurve(run_results)


class Plot:
    def __init__(self,
                 ax,
                 expert_power_curve,
                 amateur_power_curve=None,
                 classifier_scores=None,
                 color_map={'expert_power_curve': 'black', 'amateur_power_curve': 'blue', 'classifier': 'green'},
                 y_axis_label='Agreement with reference rater',
                 center_on=None,
                 y_range=None,
                 name='powercurve',
                 legend_label='Expert raters',
                 amateur_legend_label="Lay raters",
                 verbosity=1,
                 generate_pgf=False
                 ):
        self.expert_power_curve = expert_power_curve
        self.amateur_power_curve = amateur_power_curve
        self.classifier_scores = classifier_scores
        self.color_map = color_map
        self.y_axis_label = y_axis_label
        self.center_on = center_on  # whether to subtract out c_0 from all values, in order to plot gains over baseline
        self.y_range = y_range
        self.name = name
        self.x_intercepts = []
        self.legend_label = legend_label
        self.amateur_legend_label = amateur_legend_label
        self.verbosity = verbosity
        self.ax = ax
        self.generate_pgf = generate_pgf

        if self.generate_pgf:
            self.template = Template(pkgutil.get_data(__name__, "templates/pgf_template.txt").decode('utf-8'))
            self.template_dict = dict()
        # self.make_fig_and_axes()

        self.format_ax()


    def format_ax(self):
        xlabel = 'Number of raters'
        ylabel = self.y_axis_label
        self.ax.set_xlabel(xlabel, fontsize=16)
        self.ax.set_ylabel(ylabel, fontsize=16)
        self.ax.set_title(self.name)

        if self.generate_pgf:
            self.template_dict['xlabel'] = xlabel
            self.template_dict['ylabel'] = ylabel
            self.template_dict['title'] = self.name

    def add_state_distribution_inset(self, dataset_generator):
        ymax = self.y_range[1] if self.y_range else self.ymax

        if self.possibly_center_score(self.expert_power_curve.means.iloc[-1]) < .66 * ymax:
            if self.verbosity > 2:
                print(f"loc 1. c_k = {self.possibly_center_score(self.expert_power_curve.means.iloc[-1])}; ymax={ymax}")
            loc = 1
        else:
            if self.verbosity > 2:
                print("loc 5")
            loc = 5

        inset_ax = inset_axes(self.ax, width='30%', height='20%', loc=loc)

    def possibly_center_score(self, score):
        if self.center_on is not None and len(self.expert_power_curve.means) > 0:
            return score - self.center_on
        else:
            return score

    def add_classifier_line(self, ax, name, score, color, ci=None):
        ax.axhline(y=score, color=color, linewidth=2, linestyle='dashed', label=name)
        classifier_dict = dict()

        if ci:
            ax.axhspan(ci[0], ci[1], alpha=0.1, color=color)
            classifier_dict['ci'] = ''
            classifier_dict['cicolor'] = color
            classifier_dict['cilower'] = ci[0]
            classifier_dict['ciupper'] = ci[1]
            classifier_dict['cialpha'] = 0.1
        else:
            classifier_dict['ci'] = '%'
            classifier_dict['cicolor'] = ''
            classifier_dict['cilower'] = ''
            classifier_dict['ciupper'] = ''
            classifier_dict['cialpha'] = ''

        if self.generate_pgf:
            classifier_template = Template(pkgutil.get_data(__name__, "templates/classifier_template.txt").decode('utf-8'))
            classifier_dict['score'] = score
            classifier_dict['color'] = color
            classifier_dict['name'] = name
            classifier_dict['linetype'] = 'dashed'
            c = classifier_template.substitute(**classifier_dict)
            if 'classifiers' not in self.template_dict:
                self.template_dict['classifiers'] = ''
            self.template_dict['classifiers'] += c

    def add_survey_equivalence_point(self, ax, survey_equiv, score, color, include_droplines=True):
        # score is already centered before this is called
        # print(f"add_survey_equivalence_point {survey_equiv} type {type(survey_equiv)}")
        if (type(survey_equiv) != str):
            ax.scatter(survey_equiv, score, c=color)

            se_dict = {'surveyequiv':survey_equiv, 'score':score, 'color':color, 'dropline':'%'}

            if include_droplines:
                # print(f"adding dropline at {survey_equiv} from {self.ymin} to {score}")
                self.x_intercepts.append(survey_equiv)
                ax.vlines(x=survey_equiv, color=color, linewidths=2, linestyles='dashed', ymin=self.y_range_min,
                          ymax=score)
                se_dict['dropline'] = ''
                se_dict['dropcolor'] = color
                se_dict['linestyle'] = 'dashed'
                se_dict['x'] = survey_equiv
                se_dict['ymin'] = self.y_range_min
                se_dict['ymax'] = score

            # else:
            #     print("include_droplines is False")

            if self.generate_pgf:
                surveyequiv_template = Template(
                    pkgutil.get_data(__name__, "templates/surveyequiv_template.txt").decode('utf-8'))
                s = surveyequiv_template.substitute(**se_dict)
                if 'surveyequivs' not in self.template_dict:
                    self.template_dict['surveyequivs'] = ''
                self.template_dict['surveyequivs'] += s

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
        actuals = np.array([self.possibly_center_score(score) for score in curve.values[points]])
        lower_error = means - lower_bounds
        upper_error = upper_bounds - means
        ax.errorbar(curve.means[points].index,
                    # means,
                    actuals,
                    yerr=[lower_error, upper_error],
                    marker='o',
                    color=color,
                    elinewidth=2,
                    capsize=5,
                    label=legend_label,
                    linestyle=linestyle)

        if self.generate_pgf:
            plot_template = Template(pkgutil.get_data(__name__, "templates/plot_template.txt").decode('utf-8'))
            plot_dict = dict()
            if linestyle == '-':
                plot_dict['linestyle'] = 'solid'
            else:
                plot_dict['linestyle'] = 'only marks'

            pc = ''
            for i in curve.means[points].index:
                pc += '{0}\t{1}\t{2}\n'.format (i,actuals[i],lower_error[i])
            plot_dict['plot'] = pc
            plot_dict['marker'] = 'o'
            plot_dict['color'] = color
            plot_dict['legend'] = legend_label
            p = plot_template.substitute(**plot_dict)
            if 'plots' not in self.template_dict:
                self.template_dict['plots'] = ''
            self.template_dict['plots'] += p

    def plot(self,
             include_expert_points='all',
             connect_expert_points=True,
             include_classifiers=True,
             include_classifier_equivalences=True,
             include_classifier_amateur_equivalences=False,
             include_droplines=True,
             include_amateur_curve=True,
             include_classifier_cis=True,
             amateur_equivalences=[],
             x_ticks=None,
             legend_loc=None):

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

        if self.verbosity > 3:
            print(f"y-axis range: {self.ymin}, {self.ymax}")

        if include_classifiers:
            # self.classifier_scores is an instance of ClassifierResults, with means and cis computed
            for (classifier_name, score) in self.classifier_scores.values.items():
                if self.verbosity > 1:
                    print(f'{classifier_name} score: {score}')
                color = self.color_map[classifier_name] if classifier_name in self.color_map else 'black'
                if self.verbosity > 1:
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
                                                      self.expert_power_curve.compute_equivalence_at_actuals(score),
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines)
                if include_classifier_amateur_equivalences:
                    self.add_survey_equivalence_point(ax,
                                                      self.amateur_power_curve.compute_equivalence_at_actuals(score),
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines)
        for idx in amateur_equivalences:
            score = self.amateur_power_curve.means[idx]
            survey_eq = self.expert_power_curve.compute_equivalence_at_actuals(score)
            if self.verbosity > 1:
                print(f"k={idx}: score={score} expert equivalence = {survey_eq}")
            survey_eq = survey_eq if type(survey_eq) != str else 0
            ax.hlines(y=self.possibly_center_score(score),
                      xmin=min(survey_eq, idx),
                      xmax=max(survey_eq, idx),
                      color=self.color_map['amateur_power_curve'],
                      linewidths=2, linestyles='dashed')
            self.add_survey_equivalence_point(ax,
                                              survey_eq,
                                              self.possibly_center_score(score),
                                              self.color_map['amateur_power_curve'],
                                              include_droplines=include_droplines)

        # ax.axis([0, self.xmax, self.ymin, self.ymax])
        ax.set(xlim=(0, self.xmax))
        ylims = self.y_range if self.y_range else (self.ymin, self.ymax)
        ax.set(ylim=ylims)
        if self.generate_pgf:
            self.template_dict['xmin'] = 0
            self.template_dict['xmax'] = self.xmax
            self.template_dict['ymin'] = ylims[0]
            self.template_dict['ymax'] = ylims[1]

        # set legend location based on where there is space
        ax.legend(loc=legend_loc if legend_loc else "best")

        if x_ticks:
            regular_ticks = x_ticks
        else:
            regular_ticks = [i for i in range(0, self.xmax, math.ceil(self.xmax / 8))]

        def nearest_tick(ticks, val):
            dists = {x: abs(x - val) for x in ticks}
            return min(dists, key=lambda x: dists[x])

        # remove ticks nearest to survey equivalence points
        ticks_to_use = list(set(regular_ticks) - set([nearest_tick(regular_ticks, x) for x in self.x_intercepts]))

        ticks = sorted(ticks_to_use + self.x_intercepts)
        ax.set_xticks(ticks)
        if self.generate_pgf:
            self.template_dict['xticks'] = ', '.join(map(str, ticks))

        def xtick_formatter(x, pos):
            if math.isclose(x, int(round(x)), abs_tol=.001):
                return f"{x:.0f}"
            else:
                return f"{x:.2f}"

        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xtick_formatter))
        # fig.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(xtick_formatter))

        pass
