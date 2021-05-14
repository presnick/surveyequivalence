import operator
import random
from abc import ABC, abstractmethod
from functools import reduce
from math import factorial
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


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

    def __init__(self, allowable_labels: Sequence[str] = None, verbosity=0):
        """
        Constructor

        Parameters
        ----------
        allowable_labels: all labels that can be present in the data set.
        verbosity: verbosity parameter. Takes values 1, 2, or 3 for increasing verbosity.
        """
        self.allowable_labels = allowable_labels
        self.verbosity = verbosity

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
        non_null_label_values = [val for rater, val in labels if not pd.isna(val)]

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

        freqs = {k: 1 for k in allowable_labels}

        if len(labels) > 0:
            # k>0; use the actual labels
            for label in [l[1] for l in labels]:
                freqs[label] += 1

        else:
            # no labels yet; use the Bayesian prior, based on overall frequencies in the dataset
            # for each, loop through all labels
            for label in np.nditer(W, flags=['refs_ok']):
                if label in allowable_labels:
                    freqs[str(label)] += 1

        tot = sum(freqs.values())
        return DiscreteDistributionPrediction(allowable_labels, [freqs[k] / tot for k in allowable_labels])


class AnonymousBayesianCombiner(Combiner):
    """
    Anonymous Bayesian Combiner Class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memo = dict()

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
        item_id: item index in W
        to_predict_for: not used currently

        Returns
        -------
        Prediction based on anonymous bayesian combiner
        """

        # get number of labels in binary case, it's 2
        number_of_labels = len(allowable_labels)

        ## compute m_l counts for each label
        # freqs = {k: 0 for k in allowable_labels}
        # for label in [l[1] for l in labels]:
        #    freqs[label] += 1

        # m = np.array([freqs[i] for i in freqs.keys()])

        prediction = np.zeros(number_of_labels)

        freqs = {k: 0 for k in allowable_labels}
        for label in [l[1] for l in labels]:
            freqs[label] += 1
        m = np.array([freqs[i] for i in freqs.keys()])
        k = sum(m) + 1  # +1 because we're adding a reference rater

        for label_idx in range(0, number_of_labels):
            expanded_labels = labels + [('l', str(allowable_labels[label_idx]))]

            # TODO check W[item_id]
            # k = int(np.sum(m + one_hot_label))
            # Calculate the contribution of the held out item
            i_v_onehot, i_r_onehot = AnonymousBayesianCombiner.D_k_item_contribution(expanded_labels, W[item_id],
                                                                                     allowable_labels)

            one_hot_label = np.zeros(number_of_labels)
            one_hot_label[label_idx] = 1

            if str(m + one_hot_label) not in self.memo:
                overall_joint_dist, num_items = AnonymousBayesianCombiner.D_k(expanded_labels, W, allowable_labels)
                # In this case, there are not enough raters to construct a joint distribution for k,
                # so we can't make a prediction
                if num_items <= 1:
                    return None
                self.memo[str(m + one_hot_label)] = overall_joint_dist, num_items
            overall_joint_dist_onehot, num_items = self.memo[str(m + one_hot_label)]

            holdout_joint_dist_onehot = overall_joint_dist_onehot
            if i_r_onehot == 1:
                product = 1
                for idx in range(0, len(allowable_labels)):
                    product = product * factorial(m[idx] + one_hot_label[idx])
                coef = product / factorial(k + 1)  # +1 because we expand labels by 1

                v = overall_joint_dist_onehot * num_items / coef - i_v_onehot
                # In this case, there are not enough raters to construct a joint distribution for k,
                # so we can't make a prediction
                if num_items <= 1:
                    return None
                holdout_joint_dist_onehot = v * coef / (num_items - 1)
            prediction[label_idx] = holdout_joint_dist_onehot

        prediction = prediction / sum(prediction)
        # TODO check that prediction is valid

        output = DiscreteDistributionPrediction(allowable_labels, prediction.tolist())

        return output

    @staticmethod
    def D_k_item_contribution(labels: np.array, item: np.array, allowable_labels: Sequence[str]) -> (float, float):
        """
        ProbabilityOfOneItem function in Algorithm 5. Computes the contribution of a single item to the combiner

        Parameters
        ----------
        labels: item labels from several raters
        item: The item under current consideration
        allowable_labels: The set of labels that can be entered by the raters.

        Returns
        -------
        The contribution of this item.
        """

        def comb(n, k):
            # from https://stackoverflow.com/a/4941932
            k = min(k, n - k)
            numer = reduce(operator.mul, range(n, n - k, -1), 1)
            denom = reduce(operator.mul, range(1, k + 1), 1)
            return numer // denom

        # count number of ratings in the item.
        num_rate = 0

        ## compute m_l counts for each label
        freqs = {k: 0 for k in allowable_labels}
        for label in [l[1] for l in labels]:
            freqs[label] += 1

        m = np.array([freqs[i] for i in freqs.keys()])

        k = sum(m)
        assert (k == len(labels))

        nonzero_itm_mask = np.nonzero(item)
        item = item[nonzero_itm_mask]

        for r in item:
            if r is not None and r != '':
                num_rate += 1
        # only proceed if num_rate < k
        if num_rate < k:
            return 0, 0

        # no_count = 0
        freqs = {lab: 0 for lab in allowable_labels}
        for label in item:
            freqs[label] += 1
        mi = np.array([freqs[i] for i in freqs.keys()])

        for label_idx in range(0, len(allowable_labels)):
            if mi[label_idx] < m[label_idx]:
                # no_count = 1
                return 0, 1

        ki = sum(mi)
        product = 1
        for label_idx in range(0, len(allowable_labels)):
            product = product * comb(mi[label_idx], m[label_idx])

        return product / comb(ki, k), 1

    @staticmethod
    def D_k(labels: np.array, W: np.matrix, allowable_labels: Sequence[str]) -> (float, int):
        """
        Compute the joint distribution over k anonymous ratings

        Parameters
        ----------
        labels: item labels from several raters
        W: item and rating dataset
        allowable_labels: The set of labels that can be entered by the raters.

        Returns
        -------
        joint distribution, and num_items
        """

        ## compute m_l counts for each label
        freqs = {k: 0 for k in allowable_labels}
        for label in [l[1] for l in labels]:
            freqs[label] += 1

        m = np.array([freqs[i] for i in freqs.keys()])

        k = int(np.sum(m))  # the number of raters
        # sample_size = 1000
        # TODO - consider subsampling?

        # Sample rows from the rating matrix W with replacement
        I = W  # [np.random.choice(W.shape[0], sample_size, replace=True)]

        v = 0

        # rating counts for that item i
        mi = np.zeros(len(allowable_labels))
        num_items = 0
        for item in I:
            i_v, i_r = AnonymousBayesianCombiner.D_k_item_contribution(labels, item, allowable_labels)
            v += i_v
            num_items += i_r

        product = 1
        for label_idx in range(0, len(allowable_labels)):
            product = product * factorial(m[label_idx])
        v = v * product / (factorial(k) * num_items)
        return v, num_items
