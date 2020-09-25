from abc import ABC, abstractmethod
from math import factorial
from typing import Sequence, Tuple

import numpy as np


class Prediction(ABC):
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
    def __init__(self, num):
        self.num = num

    @property
    def value(self):
        return self.num

class DiscreteDistributionPrediction(Prediction):
    def __init__(self, label_names, probabilities, no_extremes=True, normalize=True):
        super().__init__()
        self.label_names = label_names
        if no_extremes:
            self.probabilities = [min(.999, max(.001, pr)) for pr in probabilities]
        else:
            self.probabilities = probabilities

        if normalize:
            s = sum(self.probabilities)
            self.probabilities = [pr/s for pr in self.probabilities]

    def __repr__(self):
        return f"Prediction: {self.probabilities}"

    def label_probability(self, label):
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

        """

        return np.max(self.probabilities)


class Combiner(ABC):
    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    @abstractmethod
    def combine(self, allowable_labels: Sequence[str],
            labels: Sequence[Tuple[str, str]],
            W: np.matrix = None,
            item_id=None,
            to_predict_for=None) -> DiscreteDistributionPrediction:
        pass


class MeanCombiner(Combiner):
    def combine(self, allowable_labels: Sequence[str]=None,
            labels: Sequence[Tuple[str, float]]=[],
            W: np.matrix = None,
            item_id=None,
            to_predict_for=None) -> NumericPrediction:

        """
        :param allowable_labels: not used in this combiner
        :param labels: numeric values from particular rater ids; rater ids are ignored
        :param W: ignored in this combiner
        :param item_id: ignored in this combiner
        :param to_predict_for: ignored in this combiner
        :return: the mean of the labels
        """
        if len(labels) == 0:
            return None
        else:
            return NumericPrediction(sum([val for rater, val in labels]) / len(labels))

class FrequencyCombiner(Combiner):
    def __init__(self):
        super().__init__()

    def combine(self, allowable_labels: Sequence[str],
                   labels: Sequence[Tuple[str, str]],
                   W: np.matrix = None,
                   item_id=None,
                   to_predict_for=None) -> DiscreteDistributionPrediction:
        """
        Ignore item_id, rater_ids (first element of each tuple in labels), and rater_id to_predict_for
        return a vector of frequencies with which the allowable labels occur

        >>> FrequencyCombiner().combine(['pos', 'neg'], np.array([(1, 'pos'), (2, 'neg'), (4, 'neg')]), ).probabilities
        [0.3333333333333333, 0.6666666666666666]

        >>> FrequencyCombiner().combine(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg'), (4, 'neg')])).probabilities
        [0.0, 1.0]
        """

        freqs = {k: 0 for k in allowable_labels}

        if len(labels) > 0:
            # k>0; use the actual labels
            for label in [l[1] for l in labels]:
                freqs[label] += 1

        else:
            # no labels yet; use the Bayesian prior, based on overall frequencies in the dataset
            # TODO: loop through items in W
            # for each, loop through all labels
            for label in np.nditer(W, flags=['refs_ok']):
                if label in allowable_labels:
                    freqs[str(label)] += 1

        tot = sum(freqs.values())
        return DiscreteDistributionPrediction(allowable_labels, [freqs[k] / tot for k in allowable_labels])


class AnonymousBayesianCombiner(Combiner):
    def __init__(self):
        super().__init__()
        self.memo = dict()

    def combine(self, allowable_labels: Sequence[str],
                   labels: Sequence[Tuple[str, str]],
                   W: np.matrix = None,
                   item_id=None,
                   to_predict_for=None) -> DiscreteDistributionPrediction:
        """
        Algorithm 5
        Compute the anonymous bayesian combiner. Combines rater labels like frequency_combiner, but this uses the
        information from the item/rating dataset W.

        :param allowable_labels: the set of labels/ratings allowed
        :param labels: the k ratings
        :param W: item and rating dataset
        :param item_id: item index in W
        :param to_predict_for: not used currently
        :return: Prediction based on anonymous bayesian combiner
        """
        # get number of labels in binary case, it's 2
        number_of_labels = len(allowable_labels)

        ## compute m_l counts for each label
        #freqs = {k: 0 for k in allowable_labels}
        #for label in [l[1] for l in labels]:
        #    freqs[label] += 1

        #m = np.array([freqs[i] for i in freqs.keys()])

        prediction = np.zeros(number_of_labels)

        freqs = {k: 0 for k in allowable_labels}
        for label in [l[1] for l in labels]:
            freqs[label] += 1
        m = np.array([freqs[i] for i in freqs.keys()])
        k = sum(m)

        for label_idx in range(0,number_of_labels):
            expanded_labels = labels + [('l', str(allowable_labels[label_idx]))]


            # TODO check W[item_id]
            #k = int(np.sum(m + one_hot_label))
            # Calculate the contribution of the held out item
            i_v_onehot, i_r_onehot = AnonymousBayesianCombiner.D_k_item_contribution(expanded_labels, W[item_id], allowable_labels)


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
                    product = product * factorial(m[idx]+one_hot_label[idx])
                coef = product / factorial(k)

                v = overall_joint_dist_onehot * num_items / coef - i_v_onehot
                # In this case, there are not enough raters to construct a joint distribution for k,
                # so we can't make a prediction
                if num_items <= 1:
                    return None
                holdout_joint_dist_onehot = v * coef / (num_items - 1)
            prediction[label_idx] = holdout_joint_dist_onehot

        i_v_m, i_r_m = AnonymousBayesianCombiner.D_k_item_contribution(labels, W[item_id], allowable_labels)
        if str(m) not in self.memo:
            overall_joint_dist, num_items = AnonymousBayesianCombiner.D_k(labels, W, allowable_labels)
            # In this case, there are not enough raters to construct a joint distribution for k,
            # so we can't make a prediction
            if num_items <= 1:
                return None
            self.memo[str(m)] = overall_joint_dist, num_items
        overall_joint_dist_m, num_items = self.memo[str(m)]
        holdout_joint_dist_m = overall_joint_dist_m
        if i_r_m == 1:
            product = 1
            for idx in range(0, len(allowable_labels)):
                product = product * factorial(m[idx])
            coef = product / factorial(k)

            v = overall_joint_dist_m * num_items / coef - i_v_m
            holdout_joint_dist_m = v * coef / (num_items - 1)

        if holdout_joint_dist_m == 0:
            #TODO - what to do here?
            holdout_joint_dist_m = 0.00001

        prediction = prediction / holdout_joint_dist_m
        # TODO check that prediction is valid

        output = DiscreteDistributionPrediction(allowable_labels, prediction.tolist())

        return output

    @staticmethod
    def D_k_item_contribution(labels: np.array, item: np.array, allowable_labels: Sequence[str]) -> (float, float):
        """

        :param labels:
        :param item:
        :param allowable_labels:
        :param number_of_labels:
        :return: item contribution, and whether it counts towards number of items
        """
        def comb(n, k):
            return factorial(n) / factorial(k) / factorial(n - k)

        # count number of ratings in the item.
        num_rate = 0

        ## compute m_l counts for each label
        freqs = {k: 0 for k in allowable_labels}
        for label in [l[1] for l in labels]:
            freqs[label] += 1

        m = np.array([freqs[i] for i in freqs.keys()])

        k = sum(m)
        assert(k == len(labels))

        nonzero_itm_mask = np.nonzero(item)
        item = item[nonzero_itm_mask]

        for r in item:
            if r is not None and r != '':
                num_rate += 1
        # only proceed if num_rate < k
        if num_rate < k:
            return 0, 0

        #no_count = 0
        freqs = {lab: 0 for lab in allowable_labels}
        for label in item:
            freqs[label] += 1
        mi = np.array([freqs[i] for i in freqs.keys()])

        for label_idx in range(0, len(allowable_labels)):
            if mi[label_idx] < m[label_idx]:
                #no_count = 1
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

        :param labels:
        :param W: item and rating dataset
        :param allowable_labels: the set of labels/ratings allowed
        :return: joint distribution, and num_items
        """

        ## compute m_l counts for each label
        freqs = {k: 0 for k in allowable_labels}
        for label in [l[1] for l in labels]:
            freqs[label] += 1

        m = np.array([freqs[i] for i in freqs.keys()])

        k = int(np.sum(m))  # the number of raters
        #sample_size = 1000
        # TODO - consider subsampling?

        # Sample rows from the rating matrix W with replacement
        I = W#[np.random.choice(W.shape[0], sample_size, replace=True)]

        v = 0

        # rating counts for that item i
        mi = np.zeros(len(allowable_labels))
        num_items = 0
        for item in I:
            i_v,i_r = AnonymousBayesianCombiner.D_k_item_contribution(labels, item, allowable_labels)
            v += i_v
            num_items += i_r

        product = 1
        for label_idx in range(0,len(allowable_labels)):
            product = product * factorial(m[label_idx])
        v = v * product / (factorial(k) * num_items)
        return v, num_items






