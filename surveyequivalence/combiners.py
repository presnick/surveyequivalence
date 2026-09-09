import operator
import random
from abc import ABC, abstractmethod
from functools import reduce
from math import factorial
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from ._data import as_rating_array, is_missing


class Prediction(ABC):
    """
    Abstract class that defines a value for many types of Predictions
    """

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def value(self):
        pass

    def __repr__(self):
        return f"Prediction: {self.value}"


class NumericPrediction(Prediction):
    """
    A numeric prediction. value is defined as a number
    """

    def __init__(self, num):
        self.num = num

    @property
    def value(self):
        return self.num


class DiscretePrediction(Prediction):
    """
    A discrete prediction. value is defined as a label
    """

    def __init__(self, label):
        self.label = label

    @property
    def value(self):
        return self.label


class DiscreteDistributionPrediction(Prediction):
    """
    A discrete distribution prediction where labels are associated with probabilities. Value takes the label with the
    highest probability.
    """

    def __init__(self, label_names, probabilities, extreme_cutoff=0.02, normalize=True):
        """
        Constructor for Discrete Distrbution Prediction.

        Parameters
        ----------
        label_names: Labels for the distribution
        probabilities: Probabilities associated with each label
        extreme_cutoff: value that is used to remove extremes of the distribution -- and possibly stop log(0) and divide
        by zero errors in certain scoring functions.
        normalize: If true, then probabilities will be re-normalized, after extreme_cutoff is applied.
        """
        super().__init__()
        self.label_names = label_names
        self.probabilities = [min(1 - extreme_cutoff, max(extreme_cutoff, pr)) for pr in probabilities]

        if normalize:
            s = sum(self.probabilities)
            self.probabilities = [pr / s for pr in self.probabilities]

    def __repr__(self):
        return f"Prediction: {self.probabilities}"

    def label_probability(self, label):
        """
        Returns the probability associated with an input label

        Parameters
        ----------
        label: label to query

        Returns
        -------
        Probability assicated with label.
        """
        return self.probabilities[self.label_names.index(label)]

    @property
    def value(self):
        """
        Return the single label that has the highest predicted probability.
        Break ties by taking the first one

        >>> DiscreteDistributionPrediction(['a', 'b', 'c'], [.3, .4, .3]).value
        'b'
        >>> DiscreteDistributionPrediction(['a', 'b', 'c'], [.4, .4, .2]).value
        'a'

        Returns
        -------
        label with highest probability
        """

        return self.label_names[np.argmax(self.probabilities)]

    @property
    def value_prob(self):
        """
        Return the probability of the majority class

        >>> DiscreteDistributionPrediction(['a', 'b', 'c'], [.3, .4, .3]).value
        .4
        >>> DiscreteDistributionPrediction(['a', 'b', 'c'], [.4, .4, .2]).value
        .4

        Returns
        -------
        highest probability
        """

        return np.max(self.probabilities)

    def draw_discrete_label(self):
        """
        Return one of the labels, drawn according to the distribution

        Returns
        -------
        A label
        """
        return random.choices(
            population=self.label_names,
            weights=self.probabilities
        )[0]


class Combiner(ABC):
    """
    Abstract class defining a combiner.

    A combiner selects a single label from a bag/multiset of labels (and possibly other information) according to some
    function. For example, the PluralityCombiner accepts a bag of labels and returns the label that is most frequent.
    """

    def __init__(self, allowable_labels: Sequence[str] = None, W = None, verbosity=0):
        """
        Constructor

        Parameters
        ----------
        allowable_labels: all labels that can be present in the data set.
        verbosity: verbosity parameter. Takes values 1, 2, or 3 for increasing verbosity.
        W: if W is not None, memoization will be performed
        """
        self.allowable_labels = allowable_labels
        self.verbosity = verbosity
        self.W = W

        #self.W is pandas.dataframe
        #self.W_np is numpy.ndarray
        if W is not None:
            self.W_np = as_rating_array(W)
            if not isinstance(W, pd.DataFrame):
                self.W = pd.DataFrame(self.W_np)
        else:
            self.W_np = None

        #self.memo is a dict to memoize the rusults of SumOfProbabilities
        self.memo = dict()
        #self.memo is a dict to memoize the results of Combine
        self.combined = dict()

    @abstractmethod
    def combine(self, allowable_labels: Sequence[str],
                labels: Sequence[Tuple[str, str]],
                W: np.matrix = None,
                item_id=None,
                to_predict_for=None) -> DiscreteDistributionPrediction:
        pass


class PluralityVote(Combiner):
    """
    Combiner that returns the single label that is most frequent
    """

    def combine(self, allowable_labels: Sequence[str] = None,
                labels: Sequence[Tuple[str, float]] = [],
                W: np.matrix = None,
                item_id=None,
                to_predict_for=None) -> NumericPrediction:
        """
        Returns the single label that is most frequent

        Parameters
        ----------
        allowable_labels: not used in this combiner
        labels: numeric values from particular rater ids; rater ids are ignored
        W: not used in this combiner
        item_id: not used in this combiner
        to_predict_for: not used in this combiner

        Returns
        -------
        The most common label
        """

        labels = [(rater, value) for rater, value in labels
                  if not is_missing(value, allowable_labels)]
        if len(labels) == 0:
            # with no labels, just pick one of the allowable labels at random
            return NumericPrediction(random.choice(allowable_labels))
        else:
            freqs = dict()
            for rater, val in labels:
                freqs[val] = freqs.get(val, 0) + 1
            max_freq = max(freqs.values())
            winners = [k for k in freqs if freqs[k] == max_freq]
            ## return one of the winners, at random
            return NumericPrediction(random.choice(winners))


class MeanCombiner(Combiner):
    """
    Combiner that returns the mean of all the labels.
    """

    def combine(self, allowable_labels: Sequence[str] = None,
                labels: Sequence[Tuple[str, float]] = [],
                W: np.matrix = None,
                item_id=None,
                to_predict_for=None) -> NumericPrediction:
        """
        Returns the single label that is most frequent

        Parameters
        ----------
        allowable_labels: not used in this combiner
        labels: nnumeric values from particular rater ids; rater ids are ignored
        W: not used in this combiner
        item_id: not used in this combiner
        to_predict_for: not used in this combiner

        Returns
        -------
        The mean of the labels
        """

        # ignore any null labels
        non_null_label_values = [val for rater, val in labels
                                 if not is_missing(val, allowable_labels)]

        if len(non_null_label_values) == 0:
            return None
        else:
            return NumericPrediction(sum(non_null_label_values) / len(non_null_label_values))


class FrequencyCombiner(Combiner):
    """
    Returns a vector of frequencies for each label
    """
    def combine(self, allowable_labels: Sequence[str],
                labels: Sequence[Tuple[str, str]],
                W: np.matrix = None,
                item_id=None,
                to_predict_for=None,
                ) -> DiscreteDistributionPrediction:
        """
        Returns the frequency vector for labels

        >>> FrequencyCombiner().combine(['pos', 'neg'], np.array([(1, 'pos'), (2, 'neg'), (4, 'neg')]), ).probabilities
        [0.3333333333333333, 0.6666666666666666]

        >>> FrequencyCombiner().combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg'), (4, 'neg')])).probabilities
        [0.0, 1.0]

        Parameters
        ----------
        allowable_labels: not used in this combiner
        labels: nnumeric values from particular rater ids; rater ids are ignored
        W: not used in this combiner
        item_id: not used in this combiner
        to_predict_for: not used in this combiner

        Returns
        -------
        Frequency vector of labels

        """

        freqs = {k: 0 for k in allowable_labels}

        labels = [(rater, value) for rater, value in labels
                  if not is_missing(value, allowable_labels)]

        if len(labels) > 0:
            # k>0; use the actual labels
            for label in [l[1] for l in labels]:
                freqs[label] += 1

        else:
            # no labels yet; use the Bayesian prior, based on overall frequencies in the dataset
            # for each, loop through all labels
            dataset = self.W_np if self.W_np is not None else W
            if dataset is None:
                return None
            for label in as_rating_array(dataset).flat:
                if not is_missing(label, allowable_labels) and label in freqs:
                    freqs[label] += 1

        tot = sum(freqs.values())
        if tot == 0:
            return None
        return DiscreteDistributionPrediction(allowable_labels, [freqs[k] / tot for k in allowable_labels])


class AnonymousBayesianCombiner(Combiner):
    """
    Anonymous Bayesian Combiner Class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def combine(self, allowable_labels: Sequence[str],
                labels: Sequence[Tuple[str, str]],
                W: np.matrix = None,
                item_id=None,
                to_predict_for=None) -> DiscreteDistributionPrediction:
        """
        Algorithm 6
        Compute the anonymous bayesian combiner. Combines rater labels like frequency_combiner, but this uses the
        information from the item/rating dataset W.

        Parameters
        ----------
        allowable_labels: the set of labels/ratings allowed
        labels: the k ratings
        W: item and rating dataset
        item_id: row position in W to exclude; None uses every eligible item
        to_predict_for: not used currently

        Returns
        -------
        Prediction based on anonymous bayesian combiner
        """

        labels = [(rater, value) for rater, value in labels
                  if not is_missing(value, allowable_labels)]
        W = as_rating_array(self.W_np if self.W_np is not None else W)
        if item_id is not None and (
                not isinstance(item_id, (int, np.integer)) or
                not 0 <= item_id < len(W)):
            raise ValueError("item_id must be a row position in the rating matrix")

        # Include the ordered vocabulary: equal counts can name different labels.
        memo_flag = self.W_np is not None
        if memo_flag:
            freqs = {label: 0 for label in allowable_labels}
            for _, label in labels:
                freqs[label] += 1
            cache_key = (tuple(allowable_labels), tuple(freqs.values()), item_id)
            if cache_key in self.combined:
                return self.combined[cache_key]

        # get number of labels in binary case, it's 2        
        number_of_labels = len(allowable_labels)

        prediction = np.zeros(number_of_labels)
        # go through labels, amnd copute LabelSeqProb of this being the next label
        # at the end, sum of these probabilities will be LabelSeqProb for existing sequence (denominator of line 1 in Alg 6)
        # so we don't have to compute that separately

        for label_idx in range(0, number_of_labels):
            expanded_labels = labels + [('l', allowable_labels[label_idx])]

            probability = self.labelSeqProb(allowable_labels=allowable_labels,
                                            labels=expanded_labels,
                                            W=W,
                                            item_id=item_id)
            if probability is None:
                return None
            prediction[label_idx] = probability

        total = sum(prediction)
        if (not np.all(np.isfinite(prediction)) or
                np.any(prediction < 0) or not np.isfinite(total) or total <= 0):
            return None
        prediction = prediction / total

        output = DiscreteDistributionPrediction(allowable_labels, prediction.tolist())

        if memo_flag:
            self.combined[cache_key] = output
        
        return output


    def labelSeqProb(self,
                     allowable_labels: Sequence[str],
                labels: Sequence[Tuple[str, str]],
                W: np.matrix = None,
                item_id=None,
                to_predict_for=None) -> float:
        """
        Algorithm 5: LabelSeqProb
        """

        W = as_rating_array(self.W_np if self.W_np is not None else W)
        if item_id is not None and (
                not isinstance(item_id, (int, np.integer)) or
                not 0 <= item_id < len(W)):
            raise ValueError("item_id must be a row position in the rating matrix")

        ## compute m_l counts for each label
        ## Line 1 of Algorithm 5: LabelSeqProb
        freqs = {k: 0 for k in allowable_labels}
        for label in [l[1] for l in labels]:
            if not is_missing(label, allowable_labels):
                freqs[label] += 1
        y = np.array([freqs[i] for i in freqs.keys()])
        cache_key = (tuple(allowable_labels), tuple(y))

        # Line 2 of Algorithm 5; get SumofProbabilities
        if self.W_np is not None:
            if cache_key not in self.memo:
                v, num_items = self.sumOfProbabilities(y, W, allowable_labels)
                self.memo[cache_key] = v, num_items
            else:
                v, num_items = self.memo[cache_key]
        else:
            v, num_items = self.sumOfProbabilities(y, W, allowable_labels)

        # Calculate the contribution of the held out item to subtract out at the end
        # line 3 of Algorithm 5
        i_v_excluded, excluded_count = 0, 0
        if item_id is not None:
            i_v_excluded, excluded_count = self.probabilityOneItem(y, W[item_id], allowable_labels)
        if num_items <= excluded_count:
            # Not enough raters to construct a joint distribution of labels
            return None
        return (v - i_v_excluded) / (num_items - excluded_count)


    def probabilityOneItem(self, y: np.array, item: np.array, allowable_labels: Sequence[str]) -> (float, int):
        """
        ProbabilityOneItem function in Algorithm 5. Computes the contribution of a single item to the combiner

        Parameters
        ----------
        y: vector of observed label counts that we are trying to find joint distribution for
        item: The item's labels
        allowable_labels: The set of labels that can be entered by the raters.
        Returns
        -------
        The contribution of this item.
        A 0,1 flag for whether this item had enough labels to be used in this way
        """


        def ank(n, k):
            # (n choose k) * k!
            # selections of k items * permutations of those selected items
            return reduce(operator.mul, range(n, n - k, -1), 1)

        k = sum(y)

        item = [label for label in np.asarray(item).flat
                if not is_missing(label, allowable_labels)]

        # implementing line 2 of SumOfProbabilities; exclude this item if not enough labels
        # also num_labels is |W_i|, the quantity computed on line 3 of ProbabilityOneItem
        num_labels = len(item)
        if num_labels < k:
            return 0, 0

        # line 2 of ProbabilityOneItem
        freqs = {lab: 0 for lab in allowable_labels}
        for label in item:
            freqs[label] += 1
        W_i = np.array([freqs[i] for i in freqs.keys()])

        for label_idx in range(0, len(allowable_labels)):
            if W_i[label_idx] < y[label_idx]:
                # line 8 of the algorithm
                return 0, 1

        product = 1
        for label_idx in range(0, len(allowable_labels)):
            product = product * ank(W_i[label_idx], y[label_idx])
        return product / ank(num_labels, k), 1


    def sumOfProbabilities(self, y: np.array, W: np.matrix, allowable_labels: Sequence[str]) -> (float, int):
        """
        SumOfProbabilities procedure in Algorithm 5 in the paper

        Compute the joint distribution over k anonymous ratings

        Parameters
        ----------
        y: vector of observed label counts that we are trying to find joint distribution for
        W: item and rating dataset
        allowable_labels: The set of labels that can be entered by the raters.

        Returns
        -------
        joint distribution, and num_items
        """

        # lines 1 and 2 are handled in probabilityOneItem rather than here

        W = as_rating_array(W)
        v = 0
        num_items = 0
        for item in W:
            i_v, i_r = self.probabilityOneItem(y, item, allowable_labels)
            v += i_v
            num_items += i_r  # i_r is 0 if this item not usable; else 1

        return v, num_items
