from abc import ABC, abstractmethod
from typing import Sequence, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score, accuracy_score
from scipy.stats import entropy
from sklearn.preprocessing import LabelBinarizer
import scipy
from .combiners import Prediction, DiscreteDistributionPrediction, NumericPrediction
from surveyequivalence import DiscreteState
from math import isclose, log2
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
              rater_labels: Sequence[str],
              verbosity=0):
        """
        :param classifier_predictions: numeric values
        :param rater_labels: sequence of labels, which should be numeric values
        :return: Pearson correlation coefficient
        """

        # def convert_to_number(label):
        #     if isinstance(label, numbers.Number):
        #         return label
        #     elif label == "pos":
        #         return 1
        #     elif label == None:
        #         return None
        #     else:
        #         return 0

        # have to remove items where either pred or label is missing
        # note that zip(*tups) unzips a list of tuples
        non_null_preds, non_null_labels = \
            zip(*[(pred.value, label) \
                  for (pred, label) in zip(classifier_predictions, rater_labels) \
                  if pred and label])

        # [convert_to_number(l) for l in rater_labels]
        return np.corrcoef(non_null_preds, non_null_labels)[1, 0]


class AgreementScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[DiscreteState]):
        """
        Resolve predictions to identify the most likely single label;
        Return the fraction where predicted matches actual
        """
        sum = 0
        count = 0

        for pred, label in zip(classifier_predictions, rater_labels):
            # compute weighted mean average
            if pred.value == label:
                sum += pred.value_prob
            count += 1
        return sum / count


class CrossEntropyScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str],
              verbosity=0):
        """
        Calculates the Cross Entropy of the two labels.

        >>> CrossEntropyScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
        0.594597099859

        >>> CrossEntropyScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
        0.87702971998
        """

        assert len(classifier_predictions) == len(rater_labels)

        def item_score(pred, label):
            if pred.value == label:
                return log2(pred.label_probability(label))
            else:
                return (log2(1-pred.label_probability(label)))

        # compute mean score over all items
        tot_score = sum([item_score(pred, label) for (pred, label) in \
                         zip(classifier_predictions, rater_labels)]) / \
                    len(classifier_predictions)

        return tot_score


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
            return roc_auc_score(rater_labels, [p.probabilities for p in classifier_predictions], multi_class='ovr',
                                 labels=classifier_predictions[0].label_names)
