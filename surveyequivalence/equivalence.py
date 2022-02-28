import datetime
import math
import operator
import os
import time
import pickle
import pkgutil
import shutil
import tempfile
from functools import reduce
from itertools import combinations
from string import Template
from typing import Sequence, Dict, Tuple

import random
import matplotlib
from matplotlib import figure
import multiprocess.context as ctx
import numpy as np
import pandas as pd
import pathos
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathos.pools import ProcessPool

ctx._force_start_method('spawn')

from .combiners import Prediction, Combiner
from .scoring_functions import Scorer
from .scoring_functions import comb


def load_saved_pipeline(path):
    """Loads dataset, predictions, classifiers scores, and power curve(s) previously saved using \
    :meth:`surveyequivalence.equivalence.AnalysisPipeline.save`"""
    W = pd.read_csv(f'{path}/dataset.csv', index_col=0)

    with open(f'{path}/params.pickle', 'rb') as f:
        params = pickle.load(f)

    try:
        predictions = pd.read_csv(f'{path}/predictions.csv', index_col=0)
    except:
        predictions = None

    try:
        classifier_scores = PowerCurve(df=pd.read_csv(f'{path}/classifier_scores.csv', index_col=0))
    except:
        classifier_scores = None

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


def find_maximal_full_rating_matrix_cols(W) :
    sizes = sorted(W.notnull().sum(axis=1), reverse=True)
    best = 0
    best_s = 0
    for i, s in enumerate(sizes):
        size = (i+1)*s
        if size >= best:
            best = size
            best_s = s

    return best_s

def prep_anonymized_rating_matrix(W, min_ratings_per_item=None):
    """
    Some scoring functions (e.g., correlation) are only well-defined for a set of (prediction, label) pairs for a
    bunch of items. That's why we defined Algorithm 3 the way we did. But what do we want to do if "rater 19"
    has only labeled two items in our whole data set. Surely we don't want to compute the correlation of
    predictions with rater 19 on just those two items, and treat that as equally valid with the correlation
    with "rater 1" who has labeled 1000 items. Conceptually, what would we want to do in that case?

    What we need is a way to distinguish between single-item-based scoring functions and multi-item-based scoring
    functions. This function restricts W so that only raters with more than some minimum number of items are
    included.

    Parameters
    ----------
    W: a Pandas dataframe with raters as columns and items as rows
    min_ratings_per_item: If specified, an integer

    if min_ratings_per_item is None, find the ratings that maximize the size of a full matrix.

    Returns
    -------
    Full anonymized rating matrix with no missing data in cells.
        Items as rows
        Rater-ranks as columns (1, 2, 3, ..., min_ratings_per_item)
    Omit rows (items) with fewer than min_ratings_per_item
    For each row, select min_ratings_per_item items (without replacement) and shuffle them in random order
    """

    if not min_ratings_per_item:
        min_ratings_per_item = find_maximal_full_rating_matrix_cols(W)

    # remove rows with too few items
    W = W.loc[(W.notnull().sum(axis=1) >= min_ratings_per_item)]
    # for each full-enough row, remove Nones and sample if more than min_ratings
    new_w = list()
    idx = list()
    for index, row in W.iterrows():
        new_w.append(row.dropna().sample(min_ratings_per_item).values)
        idx.append(index)
    new_w = pd.DataFrame(new_w)
    new_w.set_index(pd.Index(idx), drop=False, inplace=True)
    return new_w

class Equivalences:
    """
    Contains a dataframe with one row for each bootstrap sample of items and one column for each classifier. \
    Cell contains the survey equivalence value (equivalent number of reference raters whose combined ratings \
    yields the same score as the classifier).
    """
    def __init__(self,
                 df):
        self.df = df
        self.compute_means_and_cis()

    def compute_means_and_cis(self):
        self.means = self.df.mean()
        self.stds = self.df.std()
        self.std_lower_bounds = self.means - 2 * self.stds
        self.std_upper_bounds = self.means + 2 * self.stds
        self.empirical_lower_bounds = self.df.quantile(.025)
        self.empirical_upper_bounds = self.df.quantile(.975)

    @property
    def lower_bounds(self):
        """
        Returns
        -------
        A pandas Series with a lower bound on the survey equivalence for each classifier. \
        Compute based on interval covering 95% of the bootstrap samples of items, if there are more than 200 of them. \
        Otherwise compute based on two standard deviations of the scores on bootstrap samples.
        """
        if len(self.df) < 200:
            return self.std_lower_bounds
        else:
            return self.empirical_lower_bounds

    @property
    def upper_bounds(self):
        """
        Returns
        -------
        A pandas Series with an upper bound on the survey equivalence for each classifier. \
        Compute based on interval covering 95% of the bootstrap samples of items, if there are more than 200 of them. \
        Otherwise compute based on two standard deviations of the scores on bootstrap samples.
        """
        if len(self.df) < 200:
            return self.std_upper_bounds
        else:
            return self.empirical_upper_bounds

class ClassifierResults:
    def __init__(self,
                 runs: Sequence[Dict]=None,
                 df=None):
        """
        each run will be one dictionary with scores for different k (size of survey of reference raters) \
        or scores of different classifiers. \
        Alternatively, a dataframe may be passed in with one row for each run.

        Each column is for one classifier.

        First row is special: the values for the actual item sample. \
        Rest of rows are for bootstrap item samples.
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
        """
        Returns
        -------
        A pandas Series with a lower bound on the survey equivalence for each classifier. \
        Compute based on interval covering 95% of the bootstrap samples of items, if there are more than 200 of them. \
        Otherwise compute based on two standard deviations of the scores on bootstrap samples.
        """
        if len(self.df) < 200:
            return self.std_lower_bounds
        else:
            return self.empirical_lower_bounds

    @property
    def upper_bounds(self):
        """
        Returns
        -------
        A pandas Series with an upper bound on the survey equivalence for each classifier. \
        Compute based on interval covering 95% of the bootstrap samples of items, if there are more than 200 of them. \
        Otherwise compute based on two standard deviations of the scores on bootstrap samples.
        """
        if len(self.df) < 200:
            return self.std_upper_bounds
        else:
            return self.empirical_upper_bounds

    @property
    def values(self):
        """
        Returns
        -------
        Series of classifier scores for the first row, the actual item set, omitting results for all bootstrap item sets.
        """
        return self.df.iloc[0, :]

    @property
    def max_value(self):
        return max(max(self.means), max(self.upper_bounds))

    @property
    def min_value(self):
        return min(min(self.means), min(self.lower_bounds))

class PowerCurve(ClassifierResults):
    """
    A special case of ClassifierResults where there is one column for each integer value k, \
    representing the mean score, over many samples of k raters, of the predictions generated by \
    combining ratings from k raters, scored against a reference rater.
    """
    def compute_equivalences(self, other, columns=None):
        """
        Parameters
        ----------
        self
        other
            The classifier scores that are compared against this PowerCurve to find equivalences
            may either be an instance of ClassifierResults or a PowerCurve. Must have same row \
            indexes as self, one for each item sample
        columns
            a subset of the column names from other.df; if not specified, use all of them
        Returns
        -------
            a df with one row for each bootstrap run, and columns as specified by the columns parameter \
            Each cell is a float, the survey equivalence value for that column from other. \
            That is, the x s.t. expected score with x raters from self == classifier_score from other.
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
        """
        Compute the equivalence of the mean score of the classifier across the bootstrap item samples \
        based on the mean survey power curve computed across the bootstrap item samples
        """
        return self.compute_one_equivalence(classifier_score, self.means.to_dict())

    def compute_equivalence_at_actuals(self, classifier_score):
        """
        Compute the equivalence of the score of the classifier on the actual item sample \
        based on the survey power curve computed for the actual item sample
        """
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
        Parameters
        ----------
        other
            another PowerCurve
        k
            survey size
        Returns
        -------
        fraction of bootstrap runs where power@k higher for self than other power curve
        """
        df1 = self.df
        df2 = other.df
        return (df1[k] > df2[k]).sum() / len(df1)

    def reliability_of_beating_classifier(self, other, k=1, other_col=1):
        """
        Parameters
        ----------
        other
            the other ClassifierResults or PowerCurve
        self_col
            the survey size (column) for self
        other_col
            the survey size (column) for other to compare, with matching bootstrap samples as rows
        Returns
        -------
        fraction of bootstrap runs where self power higher than other power
        """
        return (self.df[k] > other.df[other_col]).sum() / len(self.df)

    def compute_performance_ratio(self, other, K, columns=None):
        """
        Parameters
        ----------
        self
        other
            The classifier scores that are compared against this PowerCurve to find equivalences
            may either be an instance of ClassifierResults or a PowerCurve. Must have same row \
            indexes as self, one for each item sample
        columns
            a subset of the column names from other.df; if not specified, use all of them
        Returns
        -------
            a df with one row for each bootstrap run, and columns as specified by the columns parameter \
            Each cell is a float, the performance_ratio value for that column from other. \
            That is, the classifier's information gain over k raters' information gain.
        """
        if columns is None:
            columns = other.df.columns
        run_results = list()
        for run_idx, row in self.df.iterrows():
            run_ratio = dict()
            for h in columns:
                score = other.df.loc[run_idx, h]
                run_ratio[h] = (score-row[0]) / (row[K]-row[0])
            run_results.append(run_ratio)
        return pd.DataFrame(run_results)


class LabeledItem:
    def __init__(self,
                 item_id,
                 ref_rater_id,
                 ref_rater_label):
        self.item_id = item_id
        self.ref_rater_id = ref_rater_id
        self.ref_rater_label = ref_rater_label


class AnalysisPipeline:
    """The main class for running an analysis

    Parameters
    ----------
    W: pd.DataFrame
        The ratings dataframe with one column for each rater, one row for each item
    sparse_experts: bool
        True (default) if some raters may not have rated all items
    expert_cols: Sequence[str] = []
        A list of column names, one for each potential "reference rater" whose the classifier is trying to \
        predict. These are also the columns used for computing the power curve for survey equivalence
    amateur_cols: Sequence[str] = []
        A list of column names, one for each potential "other rater". Their ratings are not used for evaluating \
        the classifier, but a separate power curve may be computed for them, using surveys of k of them \
        to predict a reference rater's label. Survey equivalences can also be calculated between j "other raters" \
        and k reference raters.
    classifier_predictions: pd.DataFrame = None
        A dataframe with one column for each classifier for which we want to compute survey equivalences. \
        One row for each item; row indexes should be the same as for W
    combiner: Combiner = None
        A combiner that is used to make a prediction about the next label for an item, \
        given labels from some other raters.
    scorer: Scorer = None
        A scorer that takes a vector of predictions and a vector of realized reference rater labels and \
        yields a numeric score.
    allowable_labels: Sequence[str] = None
        A list of the potential label strings that a rater is permitted to assign to an item
    min_k=0
        When computing power curves, the smallest survey size to include
    num_bootstrap_item_samples=100
        When computing error bars, how many bootstrap samples of items to create
    max_rater_subsets=200
        When computing power curves, we compute the average score over predictions made from many subsets \
        of reference raters of size k. When k is small, we choose all subsets of size k. For larger k, \
        we take a sample from the powerset. This parameters determines how many subsets to select.
    max_K=10
        When computing computing curves, the largest survey size to include. Cannot be larger than \
        the number of reference raters in W, minus one.
    ratersets_memo=None
        While running, a dictionary is create to memoize certain computations, for efficiency. A value \
        be passed in in order to reuse the memoized computations from a previous run.
    predictions_memo=None
        While running, a dictionary is create to memoize certain computations, for efficiency. A value \
        be passed in in order to reuse the memoized computations from a previous run.
    item_samples=None
        If specified, the set of bootstrap item samples to use for computing error bars. \
        If not specified, a new set of bootstrap item samples will be created.
    performance_ratio_k=None
        Parameter k for the performance_ratio: classifier's information gain over k raters' information gain
    anonymous_raters=False
        If False, then each column in W represents an individual rater. If True, then raters are anonymous and
        not all labels in a column came from the same rater.
    verbosity=1
        Controls how much information is printed to the console during execution. Set a higher number \
        to help with debugging.
    run_on_creation = True
        Whether to actually run the analysis pipeline
    procs=pathos.helpers.cpu_count() - 1
        How many processors are available for parallel execution"""
    def __init__(self,
                 W: pd.DataFrame,
                 sparse_experts: bool =True,
                 expert_cols: Sequence[str] = [],
                 amateur_cols: Sequence[str] = [],
                 classifier_predictions: pd.DataFrame = None,
                 combiner: Combiner = None,
                 scorer: Scorer = None,
                 allowable_labels: Sequence[str] = None,
                 min_k=0,
                 num_bootstrap_item_samples=100,
                 max_rater_subsets=200,
                 max_K=10,
                 ratersets_memo=None,
                 predictions_memo=None,
                 item_samples=None,
                 performance_ratio_k=None,
                 anonymous_raters=False,
                 verbosity=1,
                 run_on_creation = True,
                 procs=pathos.helpers.cpu_count() - 1
                 ):

        if expert_cols:
            self.expert_cols = expert_cols
        else:
            self.expert_cols = W.columns
        self.amateur_cols = amateur_cols
        self.classifier_predictions = classifier_predictions
        self.W = W
        self.W_as_array = W.to_numpy()
        self.sparse_experts = sparse_experts
        self.combiner = combiner
        self.scorer = scorer
        self.allowable_labels = allowable_labels
        self.min_k = min_k
        self.max_K = max_K
        self.num_bootstrap_item_samples = num_bootstrap_item_samples
        self.max_rater_subsets=max_rater_subsets
        self.verbosity = verbosity
        self.procs = procs
        self.anonymous_raters = anonymous_raters
        self.performance_ratio_k = performance_ratio_k

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
        """Create the power curve(s); normally invoked during __init__ but can be called separately."""

        self.run_timestamp = datetime.datetime.now().strftime("%d-%B-%Y_%I-%M-%S_%p")

        if self.classifier_predictions is not None:
            self.classifier_scores = self.compute_classifier_scores()

        self.expert_power_curve = self.compute_power_curve(
            raters=self.expert_cols,
            ref_raters=self.expert_cols,
            min_k=self.min_k,
            max_k=min(self.max_K, len(self.expert_cols)) - 1,
            procs=self.procs,
            max_rater_subsets=self.max_rater_subsets)

        if self.classifier_predictions is not None:
            self.expert_survey_equivalences = Equivalences(
                self.expert_power_curve.compute_equivalences(self.classifier_scores))

        if self.performance_ratio_k is not None:
            self.performance_ratio = self.expert_power_curve.compute_performance_ratio(self.classifier_scores,K=self.performance_ratio_k)

        if self.amateur_cols is not None and len(self.amateur_cols) > 0:
            if self.verbosity > 0:
                print("\n\nStarting to process amateur raters")
            self.amateur_power_curve = self.compute_power_curve(
                raters=amateur_cols,
                ref_raters=expert_cols,
                min_k=min_k,
                max_k=min(max_K, len(self.amateur_cols)) - 1,
                procs = self.procs,
                max_rater_subsets=self.max_rater_subsets)
            self.amateur_survey_equivalences = Equivalences(
                self.amateur_power_curve.compute_equivalences(self.classifier_scores))

    def path_for_saving(self, dirname_base="analysis_pipeline", include_timestamp=True):
        """

        Parameters
        ----------
        dirname_base
            A name that describes the analysis; / will be treated as a subdirectory
        include_timestamp
            Whether to make a folder indicating the timestamp at which the run was done.

        Returns
        -------
        A path of the form {self.run_timestamp}/{dirname_base}
        If the path does not exist yet, it is created.
        """

        path = f'saved_analyses'
        if include_timestamp:
            path += f'/{self.run_timestamp}'
        path += f'/{dirname_base}'
        if not os.path.isdir(path):
            os.makedirs(path)

        return path

    def save(self, path=None, msg="", save_results = True):
        """Save instance and results to files

        Parameters
        ----------
        dirname_base="analysis_pipeline"
            A subdirectory name in which to store saved results
        msg
            A text string to write in a README file that is generated
        save_results=True
            If True, generates a `results_summary.txt` file with power curve and survey equivalence summary stats
        """

        if not path:
            path = self.path_for_saving()

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
                min_k = self.min_k,
                num_bootstrap_item_samples = self.num_bootstrap_item_samples,
                max_rater_subsets = self.max_rater_subsets,
                verbosity = self.verbosity,
                ratersets_memo = self.ratersets_memo,
                item_samples = self.item_samples
        )
        with open(f'{path}/params.pickle', 'wb') as f:
            pickle.dump(d, f)

        if self.classifier_predictions is not None:
            # save the classifier predictions
            self.classifier_predictions.to_csv(f'{path}/predictions.csv')

            # save the classifier scores
            self.classifier_scores.df.to_csv(f'{path}/classifier_scores.csv')

        # save the expert power curve
        self.expert_power_curve.df.to_csv(f'{path}/expert_power_curve.csv')

        # save the expert equivalences
        if self.classifier_predictions is not None:
            self.expert_survey_equivalences.df.to_csv(f'{path}/expert_survey_equivalences.csv')
        
        # save the k performance ratio
        if self.performance_ratio_k is not None:
            self.performance_ratio.to_csv(f'{path}/{self.performance_ratio_k}_performance_ratio.csv')

        # save the amateur power_curve
        amateur_power_curve = getattr(self, 'amateur_power_curve', None)
        if amateur_power_curve:
            amateur_power_curve.df.to_csv(f'{path}/amateur_power_curve.csv')
            # save the amateur equivalences
            self.amateur_survey_equivalences.df.to_csv(f'{path}/amateur_survey_equivalences.csv')

        # write out results summary
        if save_results:
            with open(f'{path}/results_summary.txt', 'w') as f:
                if self.classifier_predictions is not None:
                    f.write("\n----classifier scores-----\n")
                    f.write(f"\tActual item set score:\n {self.classifier_scores.values}\n")
                    f.write(f"\tmeans:\n{self.classifier_scores.means}\n")
                    f.write(f"\tstds:\n{self.classifier_scores.stds}\n")

                    f.write("\n----classifier score gains-----\n")
                    f.write(f"\tActual item set score:\n {self.classifier_scores.values - self.expert_power_curve.values[0]}\n")
                    f.write(f"\tmeans:\n{self.classifier_scores.means - self.expert_power_curve.values[0]}\n")
                    f.write(f"\tstds:\n{self.classifier_scores.stds - self.expert_power_curve.values[0]}\n")


                f.write("\n----power curve means-----\n")
                f.write(f"\tActual item set score:\n {self.expert_power_curve.values}\n")
                f.write(f"\tmeans:\n{self.expert_power_curve.means}\n")
                f.write(f"\tstds:\n{self.expert_power_curve.stds}\n")

                if 0 in self.expert_power_curve.values:
                    f.write("\n----power curve mean gains-----\n")
                    f.write(f"\tActual item set score:\n {self.expert_power_curve.values - self.expert_power_curve.values[0]}\n")
                    f.write(f"\tmeans:\n{self.expert_power_curve.means - self.expert_power_curve.values[0]}\n")
                    f.write(f"\tstds:\n{self.expert_power_curve.stds - self.expert_power_curve.values[0]}\n")

                f.write("\n----survey equivalences----\n")
                def output_equivalences(f, equivalences):
                    f.write(f"\tmeans:\n {equivalences.df.mean()}\n")
                    f.write(f"\tmedians\n {equivalences.df.median()}\n")
                    f.write(f"\tstddevs\n {equivalences.df.std()}\n")
                    f.write(f"\tlower bounds\n {equivalences.lower_bounds}\n")
                    f.write(f"\tupper bounds\n {equivalences.upper_bounds}\n")

                if self.classifier_predictions is not None:
                    f.write(f'reference rater equivalences\n')
                    output_equivalences(f, self.expert_survey_equivalences)
                if amateur_power_curve:
                    f.write(f'other rater equivalences\n')
                    output_equivalences(f, self.amateur_survey_equivalences)


    def output_csv(self, fname):
        """output the dataframe and the expert predictions"""
        pd.concat([self.classifier_predictions, self.W], axis=1).to_csv(fname)

    def generate_item_samples(self, num_bootstrap_item_samples=0):
        """return the actual item sample, plus specified number of bootstrap samples of items, sampled with replacement"""

        def generate_item_sample():
            return self.W.sample(len(self.W), replace=True).index

        return [self.W.index] + [generate_item_sample() for _ in range(num_bootstrap_item_samples)]

    def compute_classifier_scores(self) -> ClassifierResults:
        """
        Returns
        -------
        ClassifierResults
            instance containing the scores for the classifier(s) on each of the itemsets
        """
        if self.verbosity > 0:
            print(f"starting classifiers: computing scores")

        def compute_scores(predictions_df, ref_labels_df):
            return {col_name: self.scorer.expected_score(predictions_df[col_name],
                                                self.expert_cols,
                                                ref_labels_df,
                                                anonymous=self.anonymous_raters,
                                                verbosity=self.verbosity) \
                    for col_name in self.classifier_predictions.columns}

        def compute_one_run(idxs):
            predictions_df = self.classifier_predictions.loc[idxs, :].reset_index()
            ref_labels_df = self.W.loc[idxs, :].reset_index()
            return compute_scores(predictions_df, ref_labels_df)

        ## Each item sample is one run
        run_results = [compute_one_run(idxs) for idxs in self.item_samples]
        return ClassifierResults(run_results)

    def compute_power_curve(self, raters, ref_raters, min_k, max_k, procs, max_rater_subsets=200) -> PowerCurve:
        """
        The main analysis script

        Returns
        -------
        PowerCurve
            instance containing the scores for surveys of size up to max_k
        """

        # Use index to represent ref_raters
        raters = list(raters)
        ref_rater_idx = set()
        for ref_rater in ref_raters:
            ref_rater_idx.add(raters.index(ref_rater))
        ref_raters = ref_rater_idx

        raters_np = np.array(raters)

        if self.verbosity > 0:
            print(f"\nstarting power curve")
            if self.verbosity > 1:

                print(f"\tcomputing scores for {raters} with ref_raters {ref_raters}")

        def rater_subsets(raters, k, max_subsets):
            K = len(raters)
            raters = range(K)
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
                            if self.verbosity > 1:
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


        def generate_rater_subsets(raters, min_k, max_k, max_subsets) -> Dict[int, Sequence[Tuple[int, ...]]]:
            """
            :param raters: sequence of strings
            :param min_k:
            :param max_k:
            :param max_subsets: integer
            :return: dictionary with k=num_raters as keys; values are sequences of rater tuples, up to max_subsets for each value of k, each tuple contains an index subset of raters
            """

            retval = dict()
            for k in range(min_k, max_k+1):
                if self.verbosity > 1:
                    print(f"\tgenerate_subsets, k={k}")
                if self.verbosity > 2:
                    print(f"\t\traters={raters}")
                retval[k] = rater_subsets(raters, k, max_subsets)
            return retval

        def get_predictions(W, ratersets) -> Dict[int, Dict[Tuple[int, ...], Prediction]]:
            # add additional entries in predictions dictionary, for additional items, as necessary
            if self.verbosity > 0:
                print('\nstarting to precompute predictions for various rater subsets. \n')

            # use a dict to save calculated predictions for memoization
            predicted = dict()

            def make_prediction(idx, row):
                predictions = dict()
                cached = False
                preds_label = set([])
                # make a dictionary with rater_tups as keys and prediction outputted by combiner as values
                predictions[idx] = dict()
                
                for k in ratersets:
                    for rater_tup in ratersets[k]:

                        label_vals = row[list(rater_tup)]

                        # memoization: key is the count of different labels
                        labels=list(zip(rater_tup, label_vals))
                        # delete the empty labels for non_full_rating_matrix cases
                        for label in labels:
                            if label[1] == None:
                                labels.remove(label)
                        predictions[idx][rater_tup] = self.combiner.combine(
                            allowable_labels=self.combiner.allowable_labels,
                            labels=labels,
                            W=self.W_as_array,
                            item_id=idx)

                        if self.verbosity > 1 and idx == 0:
                            if k == 0:
                                print(f"baseline score:{predictions[idx][rater_tup]}")
                            if k == 1:
                                preds_label.add(
                                    f"{label_vals.values[0] if len(label_vals) > 0 else None}: {predictions[idx][rater_tup]}")
                    if self.verbosity > 1 and idx == 0 and k == 1:
                        print(f"scores after 1 rating is {preds_label}")

                if self.verbosity > 0:
                    if idx % 10 == 0:
                        print("\t", idx, flush=True, end='')
                    else:
                        print(f"{'.' if not cached else ','}", end='', flush=True)

                return predictions

            ## iterate through rows, accumulating predictions for that item
            predictions_list = []
            W_np = W.to_numpy()
            idx = 0
            for row in W_np:
                predictions_list.append(make_prediction(idx,row))
                idx += 1
            
            predictions = dict()
            for pred_dict in predictions_list:
                for k,v in pred_dict.items():
                    predictions[k] = v

            if self.verbosity > 0:
                print()

            return predictions

        def compute_one_run(dirpath, call_count):
            W, idxs, ratersets, predictions = pickle.load(open(dirpath + '/state.pickle', 'rb'))

            # get the ith item
            idxs = idxs[call_count]

            if self.verbosity > 0:
                if call_count % 10 == 0:
                    print("\t", call_count, flush=True, end='')
                else:
                    print(f".", end='', flush=True)
            power_levels = dict()
            ref_labels_df = W.loc[idxs, :].reset_index()
            for k in range(min_k, max_k+1):
                if self.verbosity > 2:
                    print(f"\t\tcompute_one_run, k={k}")
                scores = []
                for raterset in ratersets[k]:
                    preds = [predictions[idx][raterset] for idx in idxs]
                    unused_raters = ref_raters - set(raterset)
                    score = self.scorer.expected_score(
                        pd.Series(preds),
                        raters_np[list(unused_raters)],
                        ref_labels_df,
                        self.verbosity
                    )

                    if score is None:
                        print("ugh; no score for classifier this time ")
                    elif pd.isna(score):
                        print(f'!!!!!!!!!Unexpected NaN !!!!!! \n\t\t\preds={preds}\nunused_raters={unused_raters}\nscore={score}\ttype(score)={type(score)}')
                    else:
                        scores.append(score)

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
        predictions = get_predictions(self.W, ratersets)

        ## Each item sample is one run
        if self.verbosity > 0:
            print("\n\tcomputing power curve results for each bootstrap item sample. \n")

        dirpath = tempfile.mkdtemp()

        pool = ProcessPool(nodes=procs)
        pickle.dump((self.W, [idxs for idxs in self.item_samples], ratersets, predictions),
                    open(dirpath + '/state.pickle', 'wb'))
        run_results = pool.imap(compute_one_run, [dirpath for _ in range(0, len(self.item_samples))],
                                 [i for i in range(0, len(self.item_samples))])
        pool.close()
        pool.join()
        pool.clear()        

        shutil.rmtree(dirpath)

        if self.verbosity > 1:
            print(f"\n\t\trun_results={list(run_results)}")
        return PowerCurve(run_results)


class Plot:
    """
    Generates visual display of power curve(s) and classifier scores, as matplotlib objects and as pgf for embedding in latex. \
    First run AnalysisPipeline to generate the PowerCurve and ClassifierResults objects to pass in to constructor.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    expert_power_curve: PowerCurve
        a PowerCurve with scores for combinations of k reference raters in predicting a held-out reference rater
    amateur_power_curve=None: PowerCurve
        a PowerCurve with scores for combinations of k other raters in predicting a held-out reference rater
    classifier_scores=None: ClassifierResults
    color_map={'expert_power_curve': 'black', 'amateur_power_curve': 'blue', 'classifier': 'green'}
        a dictionary specifying colors to use for the different elements of the graph to be pltoted
    y_axis_label='Agreement with reference rater'
    center_on=None: float
        If a value is provided, it will be subtracted from all scores for classifiers and power curve values
    y_range=None
        If specified, a tuple of two values, the min and max y-values for the graph
    name='powercurve'
        A name for the plot
    legend_label='Expert raters'
        Legend label for the power curve for reference raters
    amateur_legend_label="Lay raters"
        Legend label for the power curve for other raters
    verbosity=1
        Controls how much information is printed to the console during execution. Set a higher number \
        to help with debugging.
    performance_ratio_k=None
        Parameter k for the performance_ratio: classifier's information gain over k raters' information gain
    generate_pgf=False
        If True, also populate data to enable create of pgf format, suitable for inclusion in latex \
        after calling `.plot()`, run \
        `self.template.substitute(**self.template_dict)`
    """
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
                 performance_ratio_k=None,
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
        self.performance_ratio_intercepts = []
        self.legend_label = legend_label
        self.amateur_legend_label = amateur_legend_label
        self.verbosity = verbosity
        self.ax = ax
        self.generate_pgf = generate_pgf

        # only center_on is set properly, we can calculate the performance ratio
        if center_on == expert_power_curve.values[0]:
            self.performance_ratio_k = performance_ratio_k
        else:
            self.performance_ratio_k = None

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

    def add_survey_equivalence_point(self, ax, survey_equiv, score, color, include_droplines=True, ci=None):
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
                if ci:
                    ax.axvspan(ci[0], ci[1], alpha=0.1, color=color)
                    # ax.errorbar(x=survey_equiv,
                    #             y=0,
                    #             color=color,
                    #             xerr=[[survey_equiv - ci[0]], [ci[1] - survey_equiv]],
                    #             fmt='none',
                    #             elinewidth=6,
                    #             )
                    se_dict['ci'] = ''
                    se_dict['cicolor'] = color
                    se_dict['cilower'] = ci[0]
                    se_dict['ciupper'] = ci[1]
                    se_dict['cialpha'] = 0.6
                else:
                    se_dict['ci'] = '%'
                    se_dict['cicolor'] = ''
                    se_dict['cilower'] = ''
                    se_dict['ciupper'] = ''
                    se_dict['cialpha'] = ''

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
            # index has one entry for each survey size that we calculated.
            # if we didn't include survey size 0, then we need to index by position when
            # pulling out associated values
            for pos, num_raters in enumerate(curve.means[points].index):
                pc += '{0}\t{1}\t{2}\t{3}\n'.format (num_raters,actuals[pos],lower_error[pos],upper_error[pos])
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
             include_performance_ratio=True,
             other_rater_equivalences_to_include=[],
             include_droplines=True,
             include_amateur_curve=True,
             include_classifier_cis=True,
             include_seq_cis=True,
             x_ticks=None,
             legend_loc=None):
        """
        The method that fills in the contents of the matplotlib Axes object

        Parameters
        ----------
        include_expert_points='all'
            all means to plot all reference rater survey sizes on the x-axis of the power curve
            Or include a list of numbers indicating which survey sizes to include
        connect_expert_points=True
            Whether to draw straight lines connecting the dots for survey power for surveys of successive sizes
        include_classifiers=True
            Whether to include horizontal lines showing the classifier score(s)
        include_classifier_equivalences=True
            Whether to include calculation of the equivalent number of reference raters for each classifier, \
            based on the intersection point of the classifier line and the reference raters' power curve
        include_classifier_amateur_equivalences=False
            Whether to include calculation of the equivalent number of other raters for each classifier, \
            based on the intersection point of the classifier line and the other raters' power curve
        include_performance_ratio=True
            Whether to include performance_ratio
        other_rater_equivalences_to_include=[]
            A list of survey sizes for non-reference raters. \
            For each one, compute the equivalent number of reference raters yielding the same score.
        include_droplines=True
            Whether to include vertical lines from the intersection points (survey equivalences) to the x-axis
        include_amateur_curve=True
            Whether to include a power curve for the other, non-reference raters
        include_classifier_cis=True
            Whether to include error bars around the classifier horizontal lines
        include_seq_cis=True
            Whether to include error bars around the survey equivalence values
        x_ticks=None
            If provided, a  list of x values for which tick marks should be shown. \
            If None, then it will be automatically calculated.
        legend_loc=None
            String indicating where to place the legend (uses default if None). \
            Options as documented for `matplotlib.axes.Axes.legend <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html>`_.
        """
        self.amateur_equivalences = []
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
            
            # add another axis for performance ratio, and set the range
            if self.performance_ratio_k is not None and include_performance_ratio:
                ratio = self.expert_power_curve.compute_performance_ratio(self.classifier_scores,K=self.performance_ratio_k)
                ax2 = ax.twinx()
                ylims = self.y_range if self.y_range else (self.ymin, self.ymax)
                ylims /= (self.expert_power_curve.values[self.performance_ratio_k]-self.expert_power_curve.values[0])
                ax2.set(ylim = ylims)

            survey_equivalences = Equivalences(self.expert_power_curve.compute_equivalences(self.classifier_scores))
            if include_classifier_amateur_equivalences:
                amateur_equivalences = Equivalences(self.amateur_power_curve.compute_equivalences(self.classifier_scores))
            # self.classifier_scores is an instance of ClassifierResults, with means and cis computed
            for (classifier_name, score) in self.classifier_scores.values.items():
                if self.verbosity > 1:
                    print(f'{classifier_name} score: {score}')
                color = self.color_map[classifier_name] if classifier_name in self.color_map else 'black'
                if self.verbosity > 1:
                    print(f'{classifier_name} lower bound: {self.classifier_scores.lower_bounds[classifier_name]}')
                    print(f'{classifier_name} upper bound: {self.classifier_scores.upper_bounds[classifier_name]}')
                self.add_classifier_line(ax,
                                         classifier_name,
                                         self.possibly_center_score(score),
                                         color,
                                         ci=(self.possibly_center_score(self.classifier_scores.lower_bounds[classifier_name]),
                                             self.possibly_center_score(self.classifier_scores.upper_bounds[classifier_name])) \
                                             if include_classifier_cis else None)
                if include_classifier_equivalences:
                    seq_point = survey_equivalences.df.at[0, classifier_name]
                    seq_lower_bound = survey_equivalences.lower_bounds[classifier_name]
                    seq_upper_bound = survey_equivalences.upper_bounds[classifier_name]
                    self.add_survey_equivalence_point(ax,
                                                      seq_point, # self.expert_power_curve.compute_equivalence_at_actuals(score),
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines,
                                                      ci=(seq_lower_bound, seq_upper_bound) \
                                                           if include_seq_cis else None)
                    
                    # find the performance ratio point
                    if self.performance_ratio_k is not None and include_performance_ratio:
                        ratio_point = ratio.at[0,classifier_name]
                        self.performance_ratio_intercepts.append(ratio_point)

                if include_classifier_amateur_equivalences:
                    seq_point = amateur_equivalences.df.at[0, classifier_name]
                    seq_lower_bound = amateur_equivalences.lower_bounds[classifier_name]
                    seq_upper_bound = amateur_equivalences.upper_bounds[classifier_name]
                    self.add_survey_equivalence_point(ax,
                                                      seq_point, #self.amateur_power_curve.compute_equivalence_at_actuals(score)
                                                      self.possibly_center_score(score),
                                                      color,
                                                      include_droplines=include_droplines,
                                                      ci=(seq_lower_bound, seq_upper_bound) \
                                                           if include_seq_cis else None)


                
        for idx in other_rater_equivalences_to_include:
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
            #TODO: calculate confidence interval and pass it to call to add_survey_equivalence
            self.add_survey_equivalence_point(ax,
                                              survey_eq,
                                              self.possibly_center_score(score),
                                              self.color_map['amateur_power_curve'],
                                              include_droplines=include_droplines,
                                              ci=None)

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

        # add the right y-axis for performance_ratio
        if self.performance_ratio_k is not None and include_performance_ratio:
            scale = self.expert_power_curve.values[self.performance_ratio_k]-self.expert_power_curve.values[0]
            regular_ticks = [i/2 for i in range(0, math.ceil(self.ymax/scale)*2+2)]
            ticks_to_use = list(set(regular_ticks) - set([nearest_tick(regular_ticks, y) for y in self.performance_ratio_intercepts]))
            y_ticks = sorted(ticks_to_use+self.performance_ratio_intercepts)
            ax2.set_yticks(y_ticks)
            ax2.set_ylabel(f"{self.performance_ratio_k} performance ratio",fontsize=16)

            # add pgf command
            if self.generate_pgf:
                pratio_dict = {}
                pratio_dict['xmax'] = self.template_dict['xmax']
                pratio_dict['y2max'] = float(self.template_dict['ymax'])/scale
                pratio_dict['y2ticks'] = ', '.join(map(str, y_ticks))
                pratio_dict['y2label'] = str(self.performance_ratio_k)+' performance ratio'

                pratio_template = Template(
                    pkgutil.get_data(__name__, "templates/performance_ratio_template.txt").decode('utf-8'))
                s = pratio_template.substitute(**pratio_dict)
                self.template_dict['performance_ratio'] = s

        pass

    def save(self, path: str, fig: figure, plotname='plot'):
        """
        Wrapper for the matplotlib save_plot function. Saves all data to the ./plots directory as png and tex files.

        Parameters
        ----------
        fig : matplotlib figure object to be saved
        name : Name for the file
        """

        # save the matplotlib figure as a .png and a .pdf
        fig.savefig(f'{path}/{plotname}.png')
        fig.savefig(f'{path}/{plotname}.pdf')
        
        # possibly save figure generator in .tex (pgf) format as well
        if self.generate_pgf:
            pgf = self.template.substitute(**self.template_dict)
            # Need to get rid of extra linebreaks. This is important
            pgf = pgf.replace('\r', '')
            with open(f'{path}/{plotname}.tex', 'w') as tex:
                tex.write(pgf)
        