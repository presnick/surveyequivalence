import random
from abc import ABC, abstractmethod
from math import log2
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from .combiners import DiscreteDistributionPrediction, NumericPrediction

def mode(data):
    """
    Calculate the mode in the data.
    """
    freq = dict()
    for i in data:
        if not i in freq:
            freq[i]=1
        else:
            freq[i]+=1
    mx = 0
    argmx = []
    for i in freq:
        if freq[i]>mx:
            mx = freq[i]
            argmx = [i]
        elif freq[i]==mx:
            argmx.append(i)
    return np.random.choice(argmx,1,replace=False)[0]


class Scorer(ABC):
    """
    Scorer class.

    Computes numeric socre for a sequence of classifier predictions:
    - .score() yields actual score against a sequence of reference labels
    - .expected_score() yields expected score against a matrix of reference labels

    Note that the current implementation of survey equivalence centering on c_0 and plotting
    both assume that higher scores are better. Currently, this only affects the CrossEntropy scorer,
    which we have negated from the traditional definition.
    """

    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[DiscreteDistributionPrediction]) -> float:
        pass

    def expected_score_anonymous_raters(self,
                        classifier_predictions,
                        W,
                        num_virtual_raters=100,
                        num_ref_raters_per_virtual_rater=1,
                        ref_rater_combiner="majority_vote",
                        verbosity=0):
        """
        A virtual rater is the combined rating of a randomly selected set of num_ref_raters_per_virtual_rater non-null ratings for each column
            Combine with majority vote for discrete labels; mean for continuous labels
        This implementation generates sample virtual raters, scores each, and takes the mean
        Some scoring functions override this with a closed-form solution for the expectation

        Parameters
        ----------
        classifier_predictions: Scoring predictions
        W: The item and rating dataset
        verbosity: verbosity value from 1 to 4 indicating increased verbosity.

        Returns
        -------
        A scalar expected score
        """
        # create a bunch of virtual raters (samples)
        # for each virtual rater, pick a random combination randomly selected set of num_ref_raters_per_virtual_rater non-null ratings for each column

        # TODO: select num_ref_raters_per_virtual_rater reference raters, and combine them to produce virtual rater label
        virtual_raters_collection = []
        if ref_rater_combiner=="majority_vote":
            for _, virtual_rater_i in W.iterrows():
                vals = virtual_rater_i.dropna().values
                if len(vals) > 0:
                    ratings_for_i = []
                    num = min(len(vals),num_ref_raters_per_virtual_rater)
                    for _ in range(num_virtual_raters):
                        ratings_for_i.append(mode(np.random.choice(vals, num, replace=True)))
                    virtual_raters_collection.append(ratings_for_i)
        else:
            raise NotImplementedError()

        # one row for each item; num_virtual_raters columns
        virtual_raters_matrix = np.array(virtual_raters_collection)

        # iterate through the columns (virtual raters) of samples_matrix, scoring each
        scores = [self.score(classifier_predictions, virtual_rater) for virtual_rater in virtual_raters_matrix.T]
        non_null_scores = [score for score in scores if not pd.isna(score)]

        if len(non_null_scores) == 0:
            if verbosity > 2:
                print("\t\t\tNo non-null scores")
            return None

        # take average score across virtual rateres
        retval = sum(non_null_scores) / len(non_null_scores)
        if verbosity > 2:
            print(f"\t\tnon_null_scores = {non_null_scores}; returning mean: {retval}")
        return retval


    def expected_score_non_anonymous_raters(self,
                        classifier_predictions,
                        W,
                        verbosity=0):
        """
        A virtual rater is a column of W

        Parameters
        ----------
        classifier_predictions: Scoring predictions
        W: The item and rating dataset
        verbosity: verbosity value from 1 to 4 indicating increased verbosity.

        Returns
        -------
        A scalar expected score
        """
        # one sample for each column

        scores = [self.score(classifier_predictions, W[col]) for col in W.columns]
        non_null_scores = [score for score in scores if not pd.isna(score)]

        if len(non_null_scores) == 0:
            if verbosity > 2:
                print("\t\t\tNo non-null scores")
            return None

        retval = sum(non_null_scores) / len(non_null_scores)
        if verbosity > 2:
            print(f"\t\tnon_null_scores = {non_null_scores}; returning mean: {retval}")
        return retval

    def expected_score(self,
                         classifier_predictions: Sequence,
                         raters: Sequence,
                         W,
                         anonymous=False,
                         verbosity=0):
        """
        Computes the expected score of the classifier against a random rater.
        With anonymous flag, compute expected score against a randomly selected label for each item
        With non-anonymous, compute the expected score against a randomly selected column.

        Parameters
        ----------
        classifier_predictions: Predictions to be scored
        raters: Which columns of W to use as reference raters to score the predictions against
        W: The item and rating dataset.
        anonymous: if False, then a random rater is a column from W; if True, then labels in a column are
            not necessarily from the same rater.

        verbosity: verbosity value from 1 to 4 indicating increased printed feedback during execution.

        Returns
        -------
        Expected score of the classifier against a random rater.
        """
        if verbosity > 2:
            print(f"\t\tScoring predictions = {classifier_predictions} vs. ref raters {raters}")

        if verbosity > 4:
            print(f"ref_ratings = \n{W.loc[:, list(raters)]}")


        if not anonymous:
            return self.expected_score_non_anonymous_raters(classifier_predictions, W[raters], verbosity=verbosity)
        else:
            return self.expected_score_anonymous_raters(classifier_predictions, W[raters], verbosity=verbosity)

class Correlation(Scorer):
    """
    Computes the Pearson correlation coefficient.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[NumericPrediction],
              rater_labels: Sequence[str],
              verbosity=0
              ):
        """
        Parameters
        ----------
        classifier_predictions: numeric values
        rater_labels: sequence of labels, which should be numeric values
        verbosity:

        Returns
        -------
        Pearson correlation coefficient
        """

        if verbosity > 3:
            print(f'\t\t\tcorrelation: preds={classifier_predictions}, labels={list(rater_labels)}')

        if len(classifier_predictions) != len(rater_labels):
            print("ALERT: classifier_prediction and rater_labels not of same length; skipping")
            print("")
            return None

        # have to remove items where either pred or label is missing
        good_items = [(pred.value, label) \
                      for (pred, label) in zip(classifier_predictions, rater_labels) \
                      if pred and (not pd.isna(pred.value)) and (not pd.isna(label))]
        if len(good_items) == 0:
            if verbosity > 0:
                print("ALERT: no items with both prediction and label; skipping\n")
            return None
        else:
            # note that zip(*tups) unzips a list of tuples
            non_null_preds, non_null_labels = zip(*good_items)

            if verbosity > 3:
                print(f'\t\t\tcorrelation: non null preds={non_null_preds}, non null labels={list(non_null_labels)}')

            # [convert_to_number(l) for l in rater_labels]
            retval = np.corrcoef(non_null_preds, non_null_labels)[1, 0]
            if verbosity > 2:
                print(f"\t\t\tcorrelation: returning score = {retval}")
            return retval


class AgreementScore(Scorer):
    """
    Agreement Scorer. Discrete labels and predictions
    """
    def __init__(self):
        super().__init__()

    def expected_score_anonymous_raters(self,
                        classifier_predictions,
                        W,
                        num_virtual_raters=None,
                        num_ref_raters_per_virtual_rater=None,
                        ref_rater_combiner="majority_vote",
                        verbosity=0):
        """
        A virtual rater is a randomly selected non-null rating for each column.
        Closed-form solution for the expectation, so we ignore the num_virtual_raters parameter

        Parameters
        ----------
        classifier_predictions: Scoring predictions
        W: The item and rating dataset
        verbosity: verbosity value from 1 to 4 indicating increased verbosity.

        Returns
        -------
        A scalar expected score
        """

        ## TODO: update to handle num_ref_raters_per_virtual_rater parameter

        # iterate through the rows
        # for each row:
        # get the frequency of matches among the ratings

        tot = 0
        ct = 0
        for (row, pred) in zip([row for _, row in W.iterrows()], classifier_predictions):
            # count frequency of each value
            counts = row.dropna().value_counts()
            freqs = counts / sum(counts)
            if len(counts) == 0:
                # no non-null labels for this item; skip it
                continue
            if pred.value in freqs:
                tot += freqs[pred.value]
            else:
                # predicted label never occurred in row, so no agreements to add to tot, but still increment ct
                pass
            ct += 1

        if ct > 0:
            return tot / ct
        else:
            return None

    @staticmethod
    def score(classifier_predictions: Sequence[str],
              rater_labels: Sequence[str],
              verbosity=0):
        """
        Agreement score measures the normalized number of times that the predictor matched the label. Akin to a typical
        accuracy score.

        Parameters
        ----------
        classifier_predictions: numeric values
        rater_labels: sequence of labels, which should be numeric values
        verbosity:

        Returns
        -------
        Agreement score
        """
        assert len(classifier_predictions) == len(rater_labels)
        tot_score = sum([pred.value == label for (pred, label) in \
                         zip(classifier_predictions, rater_labels)]) / \
                    len(classifier_predictions)

        return tot_score


class CrossEntropyScore(Scorer):
    """
    Cross Entropy Scorer
    """
    def __init__(self):
        super().__init__()

    def expected_score_anonymous_raters(self,
                        classifier_predictions,
                        W,
                        num_virtual_raters=None,
                        num_ref_raters_per_virtual_rater=None,
                        ref_rater_combiner="majority_vote",
                        verbosity=0):
        """
        A virtual rater is a randomly selected non-null rating for each column.
        Closed-form solution for the expectation, so we ignore the num_virtual_raters parameter

        Parameters
        ----------
        classifier_predictions: Scoring predictions
        W: The item and rating dataset
        verbosity: verbosity value from 1 to 4 indicating increased verbosity.

        Returns
        -------
        A scalar expected score
        """

        ## TODO: update to handle num_ref_raters_per_virtual_rater parameter

        # iterate through the rows
        # for each row:
        # -- get the probability of each label
        # -- use those as weights, with score for when that label happens

        tot = 0
        ct = 0
        for (row, pred) in zip([row for _, row in W.iterrows()], classifier_predictions):
            # count frequency of each value
            counts = row.dropna().value_counts()
            freqs = counts/sum(counts)
            if len(counts) == 0:
                continue
            item_tot = 0
            for label, freq in freqs.items():
                # We use the negated cross-entropy, so that higher scores will be better,
                # which is true of all the other scoring functions.
                # If we used the standard cross-entropy, scores would be positive, and higher scores would be worse
                # Several things would have to be generalized in equivalence.py to allow for higher scores
                # being worse, including plotting and centering.
                score = freq * log2(pred.label_probability(label))
                item_tot += score

            tot += item_tot
            ct += 1

        if ct > 0:
            return tot / ct
        else:
            return None


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

        Parameters
        ----------
        classifier_predictions: numeric values
        rater_labels: sequence of labels, which should be numeric values
        verbosity:

        Returns
        -------
        Cross Entropy score

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
        seq = list()
        for (pred, label) in zip(classifier_predictions, rater_labels):
            score = item_score(pred, label)
            if score is not None:
                seq.append(score)

        if len(seq) == 0: return None
        return np.mean(seq)  # Scorer.rob_median_of_means(pd.Series(seq), 1)


class PrecisionScore(Scorer):
    """
    Only implemented for binary labels where one of the labels is "pos" and binary predictions.
    Calculate the expected probability of (pos rating | pos prediction).
    (True positives divided by all positives).
    """

    def __init__(self):
        super().__init__()

    def expected_score_anonymous_raters(self,
                        classifier_predictions,
                        W,
                        num_virtual_raters=None,
                        verbosity=0):
        """
        A virtual rater is a randomly selected non-null rating for each column.
        Closed-form solution for the expectation, so we ignore the num_virtual_raters parameter

        Parameters
        ----------
        classifier_predictions: Scoring predictions
        W: The item and rating dataset
        verbosity: verbosity value from 1 to 4 indicating increased verbosity.

        Returns
        -------
        A scalar expected score
        """

        # iterate through the rows
        # for each row:
        # -- count it only if the prediction is "positive"
        # -- get the frequency of positive among the ratings

        tot = 0
        ct = 0
        for (row, pred) in zip([row for _, row in W.iterrows()], classifier_predictions):
            # count frequency of each value
            counts = row.dropna().value_counts()
            freqs = counts/sum(counts)
            if len(counts) == 0:
                # no non-null labels for this item
                continue
            elif pred.value != "pos":
                # no impact on precision if classifier didn't predict positive
                continue
            tot += freqs['pos']
            ct += 1

        if ct > 0:
            return tot / ct
        else:
            return None

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str],
              verbosity=0,
              average: str = 'micro') -> float:
        """
        Precision score. This function uses sklearn's precision function.

        >>> PrecisionScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'micro')
        0.6666666666666666

        >>> PrecisionScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'micro')
        0.3333333333333333

        Parameters
        ----------
        classifier_predictions: numeric values
        rater_labels: sequence of labels, which should be numeric values
        verbosity:
        average: macro or micro averaging

        Returns
        -------
        Precision Score
        """
        assert len(classifier_predictions) == len(rater_labels);

        if verbosity > 2:
            print(f'\n-------\n\t\tpredictions: {classifier_predictions[:10]}')
            print(f'\n--------\n\t\tlabels: {rater_labels[:10]}')

        new_pred = list()
        new_label = list()
        for (pred, label) in zip(classifier_predictions, rater_labels):
            if pred is not None and label is not None:
                new_pred.append(pred)
                new_label.append(label)

        return precision_score(new_label, [p.value for p in new_pred], average=average)


class RecallScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str],
              verbosity=0,
              average: str = 'micro') -> float:
        """
        Recall score. This function uses sklearn's recall function.

        >>> RecallScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'micro')
        0.6666666666666666
        >>> RecallScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'macro')
        0.5

        >>> RecallScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'micro')
        0.3333333333333333
        >>> RecallScore.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'macro')
        0.25

        Parameters
        ----------
        classifier_predictions: numeric values
        rater_labels: sequence of labels, which should be numeric values
        verbosity:
        average: macro or micro averaging

        Returns
        -------
        Recall Score
        """
        assert len(classifier_predictions) == len(rater_labels);

        if verbosity > 2:
            print(f'\n-------\n\t\tpredictions: {classifier_predictions[:10]}')
            print(f'\n--------\n\t\tlabels: {rater_labels[:10]}')

        new_pred = list()
        new_label = list()
        for (pred, label) in zip(classifier_predictions, rater_labels):
            if pred is not None and label is not None:
                new_pred.append(pred)
                new_label.append(label)

        return recall_score(new_label, [p.value for p in new_pred], average=average)


class F1Score(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str],
              verbosity=0,
              average: str = 'micro') -> float:
        """
        F1 score. This function uses sklearn's F1 function.

        >>> F1Score.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'micro')
        0.6666666666666666
        >>> F1Score.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 'macro')
        0.39759036144

        >>> F1Score.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'micro')
        0.3333333333333333
        >>> F1Score.score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 'macro')
        0.25

        Parameters
        ----------
        classifier_predictions: numeric values
        rater_labels: sequence of labels, which should be numeric values
        verbosity:
        average: macro or micro averaging

        Returns
        -------
        F1 Score
        """
        assert len(classifier_predictions) == len(rater_labels);

        if verbosity > 2:
            print(f'\n-------\n\t\tpredictions: {classifier_predictions[:10]}')
            print(f'\n--------\n\t\tlabels: {rater_labels[:10]}')

        new_pred = list()
        new_label = list()
        for (pred, label) in zip(classifier_predictions, rater_labels):
            if pred is not None and label is not None:
                new_pred.append(pred)
                new_label.append(label)

        return f1_score(new_label, [p.value for p in new_pred], average=average)


class AUCScore(Scorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
              rater_labels: Sequence[str],
              verbosity=0) -> float:
        """
        AUC score. This function uses sklearn's AUC function, but does not work in many cases with multiple labels.

        Parameters
        ----------
        classifier_predictions: numeric values
        rater_labels: sequence of labels, which should be numeric values
        verbosity:

        Returns
        -------
        AUC Score
        """
        assert len(classifier_predictions) == len(rater_labels);

        if verbosity > 2:
            print(f'\n-------\n\t\tpredictions: {classifier_predictions[:10]}')
            print(f'\n--------\n\t\tlabels: {rater_labels[:10]}')

        new_pred = list()
        new_label = list()
        for (pred, label) in zip(classifier_predictions, rater_labels):
            if pred is not None and label is not None:
                new_pred.append(pred)
                new_label.append(label)

        if len(set(new_label)) == 1:
            return np.nan
        if len(set(new_label)) == 2:
            return roc_auc_score(new_label, [p.value_prob for p in new_pred])
        if len(set(new_label)) > 2:
            print("multiclass AUC not implemented")
            return np.nan
