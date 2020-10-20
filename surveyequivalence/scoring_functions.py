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

    def score_classifier(self,
                         classifier_predictions: Sequence,
                         raters: Sequence,
                         W,
                         verbosity=0):
        if verbosity > 2:
            print(f"\t\tScoring predictions = {classifier_predictions} vs. ref raters {raters}")

        if verbosity > 4:
            print(f"ref_ratings = \n{W.loc[:,  list(raters)]}")

        scores = [self.score(classifier_predictions, W[col], verbosity) for col in raters]
        non_null_scores = [score for score in scores if not pd.isna(score)]

        if len(non_null_scores) == 0:
            if verbosity > 2:
                print("\t\t\tNo non-null scores")
            return None

        retval = sum(non_null_scores) / len(non_null_scores)
        if verbosity > 2:
            print(f"\t\tnon_null_scores = {non_null_scores}; returning mean: {retval}")
        return retval

class Correlation(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[NumericPrediction],
              rater_labels: Sequence[str],
              verbosity
              ):
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
        if verbosity > 3:
            print(f'\t\t\tcorrelation: preds={classifier_predictions}, labels={list(rater_labels)}')

        non_null_preds, non_null_labels = \
            zip(*[(pred.value, label) \
                  for (pred, label) in zip(classifier_predictions, rater_labels) \
                  if pred and (not pd.isna(pred.value)) and (not pd.isna(label))])

        if verbosity > 3:
            print(f'\t\t\tcorrelation: non null preds={non_null_preds}, non null labels={list(non_null_labels)}')

        # [convert_to_number(l) for l in rater_labels]
        retval = np.corrcoef(non_null_preds, non_null_labels)[1, 0]
        if verbosity > 2:
            print(f"\t\t\tcorrelation: returning score = {retval}")
        return retval


class AgreementScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[str],
                        rater_labels: Sequence[str],
              verbosity=0):

        assert len(classifier_predictions) == len(rater_labels)
        tot_score = sum([pred == label for (pred, label) in \
                        zip(classifier_predictions, rater_labels)]) / \
               len(classifier_predictions)

        return tot_score

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

        assert len(classifier_predictions) == len(rater_labels);

        if verbosity > 2:
            print(f'\n-------\n\t\tpredictions: {classifier_predictions[:10]}')
            print(f'\n--------\n\t\tlabels: {rater_labels[:10]}')

        def item_score(pred, label):
            if pred is None: return None
            if label is None: return None
            return log2(pred.label_probability(label))
            # if pred.value == label:
            #     return log2(pred.label_probability(label))
            # else:
            #     return (log2(1-pred.label_probability(label)))

        # compute mean score over all items
        tot_score = 0
        cnt = 0
        for (pred, label) in zip(classifier_predictions, rater_labels):
            score = item_score(pred, label)
            if score is not None:
                tot_score += score
                cnt += 1

        return tot_score/cnt


class PrecisionScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str],
              verbosity = 0,
              average: str = 'micro',) -> float:
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
              rater_labels: Sequence[str],
              verbosity = 0,
              average: str = 'micro') -> float:
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
              rater_labels: Sequence[str],
              verbosity = 0,
              average: str = 'micro') -> float:
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
              rater_labels: Sequence[str],
              verbosity=0) -> float:
        if len(set(rater_labels)) == 1:
            print("AUC isn't defined for single class")
            return 0
        if len(set(rater_labels)) == 2:
            return roc_auc_score(rater_labels, [p.value_prob for p in classifier_predictions])
        if len(set(rater_labels)) > 2:
            return roc_auc_score(rater_labels, [p.probabilities for p in classifier_predictions], multi_class='ovr',
                                 labels=classifier_predictions[0].label_names)
