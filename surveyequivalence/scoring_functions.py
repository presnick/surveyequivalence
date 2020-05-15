from abc import ABC, abstractmethod
from typing import Sequence, Dict
import numpy as np
from math import log
from .combiners import Prediction, DiscreteDistributionPrediction


def agreement_score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[str]):
    """
    Resolve predictions to identify the most likely single label;
    Return the fraction where predicted matches actual

    >>> agreement_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
    0.6666666666666666

    >>> agreement_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
    0.3333333333333333
    """
    return np.mean([a == b for (a, b) in zip([p.value for p in classifier_predictions],
                                             rater_labels)])


def cross_entropy_score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[str]):
    """
    Calculates the Cross Entropy of the two labels.

    >>> cross_entropy_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
    0.48187545689

    >>> cross_entropy_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
    0.48187545689
    """
    labels = set([p.label_names for p in classifier_predictions])
    assert(len(labels) == 2)
    ents = 0
    for p in classifier_predictions:
        p.probabilities[0] * log(p.probabilities[1], 2)
    return -ents


def macro_precision_score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[str]):
    """
    Macro Precision score: basically the average of precision for each label

    >>> macro_precision_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
    0.3333333333333333

    >>> macro_precision_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
    0.25
    """
    labels = set([p.label_names for p in classifier_predictions])
    return np.sum([a == b for (a, b) in zip([p.value for p in classifier_predictions], rater_labels)])/len(labels)

def micro_precision_score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[str]):
    """
    Micro precision score

    >>> micro_precision_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
    0.6666666666666666

    >>> micro_precision_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
    0.3333333333333333
    """
    tp_sum = 0
    fp_sum = 0
    num_labels = 0
    labels = set([p.label_names for p in classifier_predictions])
    pred_label = [zip([p.value for p in classifier_predictions], rater_labels)]
    for label in labels:
        num_labels+=1
        tp_sum = 0
        for a,b in pred_label:
            if a == label:
                if a == b:
                    tp_sum+=1
                else:
                    fp_sum+=1
    return tp_sum/(tp_sum+fp_sum)


def macro_recall_score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[str]):
    """
    Macro Recall - basically the average of recalls for both labels

    >>> macro_recall_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
    0.5

    >>> macro_recall_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
    0.25
    """
    recalls = 0
    num_labels = 0
    labels = set([p.label_names for p in classifier_predictions])
    pred_label = [zip([p.value for p in classifier_predictions], rater_labels)]
    for label in labels:
        num_labels+=1
        cnt = 0
        correct = 0
        for a,b in pred_label:
            if a == label:
                cnt += 1
                if a == b:
                    correct+=1
        recalls += correct/cnt
    return recalls/num_labels

def micro_recall_score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[str]):
    """
    micro recall

    >>> micro_recall_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
    0.6666666666666666

    >>> micro_recall_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
    0.3333333333333333
    """
    tp_sum = 0
    fn_sum = 0
    num_labels = 0
    labels = set([p.label_names for p in classifier_predictions])
    pred_label = [zip([p.value for p in classifier_predictions], rater_labels)]
    for label in labels:
        num_labels+=1
        fn_sum = 0
        tp_sum = 0
        for a,b in pred_label:
            if a == label:
                if a == b:
                    tp_sum+=1
            else:
                if a == b:
                    fn_sum+=1
    return tp_sum/(tp_sum+fn_sum)


def macro_f1_score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[str], beta=1):
    """
    macro f1 score

    >>> macro_f1_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'], 1)
    0.39759036144

    >>> macro_f1_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'], 1)
    0.25
    """
    prec = macro_precision_score(classifier_predictions, rater_labels)
    rec = macro_recall_score(classifier_predictions, rater_labels)
    return (beta+1) * ((prec*rec)/(prec+rec))

def micro_f1_score(classifier_predictions: Sequence[DiscreteDistributionPrediction],
                    rater_labels: Sequence[str], beta=1):
    """
    micro F1 score

    >>> agreement_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b'])
    0.6666666666666666

    >>> agreement_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b'])
    0.3333333333333333
    """
    prec = micro_precision_score(classifier_predictions, rater_labels)
    rec = micro_recall_score(classifier_predictions, rater_labels)
    return (beta+1) * ((prec*rec)/(prec+rec))
