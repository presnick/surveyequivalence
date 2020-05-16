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


class DiscreteDistributionPrediction(Prediction):
    def __init__(self, label_names, probabilities):
        super().__init__()
        self.label_names = label_names
        self.probabilities = probabilities

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


class Combiner(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def combine(self, allowable_labels: Sequence[str],
            labels: Sequence[Tuple[str, str]],
            W: np.matrix = None,
            item_id=None,
            to_predict_for=None) -> DiscreteDistributionPrediction:
        pass


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
        for label in [l[1] for l in labels]:
            freqs[label] += 1
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
        Compute the anonymous bayesian combiner. Combines rater labels like frequency_combiner, but this uses the
        information from the item/rating dataset W.

        :param allowable_labels: the set of labels/ratings allowed
        :param labels: the k ratings
        :param W: item and rating dataset
        :param item_id: not used currently
        :param to_predict_for: not used currently
        :return: Prediction based on anonymous bayesian combiner
        """
        # get number of labels in binary case, it's 2
        number_of_labels = len(allowable_labels)

        freqs = {k: 0 for k in allowable_labels}
        for label in [l[1] for l in labels]:
            freqs[label] += 1

        m = np.array([freqs[i] for i in freqs.keys()])

        prediction = np.zeros(number_of_labels)



        for label_idx in range(0,number_of_labels):
            one_hot_label = np.zeros(number_of_labels)
            one_hot_label[label_idx] = 1
            if str(m + one_hot_label) not in self.memo:
                self.memo[str(m + one_hot_label)] = AnonymousBayesianCombiner.D_k(m + one_hot_label, W, allowable_labels)
            prediction[label_idx] = self.memo[str(m + one_hot_label)]

        if str(m) not in self.memo:
            self.memo[str(m)] = AnonymousBayesianCombiner.D_k(m, W, allowable_labels)
        prediction = prediction / self.memo[str(m)]
        # TODO check that prediction is valid

        output = DiscreteDistributionPrediction(allowable_labels, prediction.tolist())

        return output

    @staticmethod
    def D_k(m: np.array, W: np.matrix, allowable_labels: Sequence[str]) -> float:
        """
        Compute the joint distribution over k anonymous ratings

        :param m: rating counts of k anonymous raters
        :param W: item and rating dataset
        :param allowable_labels: the set of labels/ratings allowed
        :return: joint distribution
        """
        number_of_labels = len(allowable_labels)

        k = np.sum(m)  # the number of raters
        sample_size = 1000

        # Sample rows from the rating matrix W with replacement
        # TODO - do we need sub-sampling for efficiency?
        I = W #[np.random.choice(W.shape[0], sample_size, replace=True)]

        v = 0

        def comb(n, k):
            return factorial(n) / factorial(k) / factorial(n - k)

        # rating counts for that item i
        mi = np.zeros(number_of_labels)
        for item in I:
            no_count = 0
            freqs = {lab: 0 for lab in allowable_labels}
            for label in item:
                freqs[label] += 1
            mi = np.array([freqs[i] for i in freqs.keys()])

            for label_idx in range(0,number_of_labels):
                if mi[label_idx] < m[label_idx]:
                    no_count = 1

            ki = sum(mi)
            if no_count == 0:
                product = 1
                for label_idx in range(0,number_of_labels):
                    product = product * comb(mi[label_idx], m[label_idx])

                v = v + product / comb(ki, k)

        product = 1
        for label_idx in range(0,number_of_labels):
            product = product * factorial(m[label_idx])
        v = v * product / (factorial(k) * sample_size)
        return v






