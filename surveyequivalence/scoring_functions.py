from abc import ABC, abstractmethod
from typing import Sequence, Dict
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score, accuracy_score
from scipy.stats import entropy
from sklearn.preprocessing import LabelBinarizer
import scipy
from .combiners import Prediction, DiscreteDistributionPrediction, NumericPrediction
from surveyequivalence import DiscreteState
from math import isclose
import numbers

class Scorer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[DiscreteDistributionPrediction]) -> float:
        pass

class Correlation(Scorer):

    @staticmethod
    def score(classifier_predictions: Sequence[NumericPrediction],
                        rater_labels: Sequence[DiscreteState],
              verbosity=0):
        """
        :param classifier_predictions: numeric values
        :param rater_labels: discrete distribution over labels, which should be numeric values
        :return: Pearson correlation coefficient
        """

        def convert_to_number(label):
            if isinstance(label, numbers.Number):
                return label
            elif label == "pos":
                return 1
            else:
                return 0

        expanded_predictions = []
        expanded_ratings = []

        if len(set([ld.num_raters for ld in rater_labels])) == 1:
            # all sets of reference raters have same length, so probabilities can be turned into integers and we
            # can sample each reference rater exactly once, avoiding some noise from random sampling
            for pred, label_dist in zip(classifier_predictions, rater_labels):
                for label, pr in zip(label_dist.labels, label_dist.probabilities):
                    assert isclose(pr * label_dist.num_raters, int(pr * label_dist.num_raters), abs_tol=.0001)
                    for _ in range(int(pr * label_dist.num_raters)):
                        expanded_predictions.append(convert_to_number(pred.value))
                        expanded_ratings.append(convert_to_number(label))

        else:
            # sample 1000 times from each rater_labels distribution
            if verbosity >= 2:
                print("unequal numbers of reference raters; using sample inside Correlation.score()")
            for pred, label_dist in zip(classifier_predictions, rater_labels):

                try:
                    indexes = scipy.stats.rv_discrete(values=(range(len(label_dist.labels)),
                                                              label_dist.probabilities)).rvs(size=1000)
                except:
                    raise Exception(f"can't sample from {(range(len(label_dist.labels)), label_dist.probabilities)}")
                labels = [label_dist.labels[i] for i in indexes]
                if verbosity >= 4:
                    print(f'labels = {labels[:10]}')
                for label in labels:
                    expanded_predictions.append(convert_to_number(pred.value))
                    expanded_ratings.append(convert_to_number(label))

        if verbosity >= 3:
            print(f'computing correlation of {[expanded_predictions[:10], expanded_ratings[:10]]}')
        return np.corrcoef([expanded_predictions, expanded_ratings])[1,0]

class AgreementScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                        rater_labels: Sequence[str]):
        """
        Resolve predictions to identify the most likely single label;
        Return the fraction where predicted matches actual
        >>> AgreementScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
        0.6666666666666666

        >>> AgreementScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
        0.3333333333333333
        """
        return np.mean([a == b for (a, b) in zip([p.value for p in classifier_predictions],
                                                 rater_labels)])


class CrossEntropyScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str]):
        """
        Calculates the Cross Entropy of the two labels.

        >>> CrossEntropyScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
        0.594597099859

        >>> CrossEntropyScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
        0.87702971998
        """

        # diagnostics
        d = [p.probabilities for p in classifier_predictions]
        bad_predictions = [(p, l) for (p, l) in zip(d, rater_labels) if (p[0] < .001 and l=='pos') or (p[0] > .995 and l=='neg')]
        if len(bad_predictions) > 0:
            for p, l in bad_predictions:
                print(p, l)

        d = [[neg_pr, pos_pr] for (pos_pr, neg_pr) in [p.probabilities_with_extremes_cut_off for p in classifier_predictions]]
        # d = [list([p.probabilities[1], p.probabilities[0]]) for p in classifier_predictions]
        try:
            return -log_loss(rater_labels, d, normalize=True, labels=classifier_predictions[0].label_names)/np.log(2)
        except:
            print([(p, l) for (p, l) in zip(d, rater_labels) if p[0] < .001 or p[0] > .995])
            foobar


class PrecisionScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str], average: str = 'micro') -> float:
        """
        Micro precision score

        >>> PrecisionScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'micro')
        0.6666666666666666
        >>> PrecisionScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'macro')
        0.5

        >>> PrecisionScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'micro')
        0.3333333333333333
        >>> PrecisionScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'macro')
        0.25
        """
        return precision_score(rater_labels, [p.value for p in classifier_predictions], average=average)


class RecallScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str], average: str = 'micro') -> float:
        """
        Recall

        >>> RecallScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'micro')
        0.6666666666666666
        >>> RecallScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'macro')
        0.5

        >>> RecallScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'micro')
        0.3333333333333333
        >>> RecallScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'macro')
        0.25
        """
        return recall_score(rater_labels, [p.value for p in classifier_predictions], average=average)


class F1Score(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str], average: str = 'micro') -> float:
        """
        F1 score

        >>> F1Score.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'micro')
        0.6666666666666666
        >>> F1Score.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'macro')
        0.39759036144

        >>> F1Score.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'micro')
        0.3333333333333333
        >>> F1Score.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'macro')
        0.25
        """
        return f1_score(rater_labels, [p.value for p in classifier_predictions], average=average)


class AUCScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str]) -> float:
        if len(set(rater_labels)) == 1:
            print("AUC isn't defined for single class")
            return 0
        if len(set(rater_labels)) == 2:
            return roc_auc_score(rater_labels, [p.value_prob for p in classifier_predictions])
        if len(set(rater_labels)) > 2:
            return roc_auc_score(rater_labels, [p.probabilities for p in classifier_predictions], multi_class='ovr', labels=classifier_predictions[0].label_names)