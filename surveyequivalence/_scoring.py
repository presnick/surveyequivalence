"""Prepared deterministic scores with the original arithmetic order.

This module deliberately recognizes exact built-in classes only. Subclasses,
nonlinear metrics, and Monte Carlo expectations retain the public scorer path.
"""

from math import log2

import numpy as np

from ._data import is_missing
from .combiners import NumericPrediction, DiscretePrediction, DiscreteDistributionPrediction
from .scoring_functions import (
    AgreementScore, CrossEntropyScore, comb, _prediction_is_missing,
    _reference_vocabulary,
)


def ordered_mean(values):
    """Left-to-right float64 addition, starting at zero, then division.

    Unlike np.sum/mean, ufunc.accumulate is an ordered prefix scan. Include
    the initial zero to preserve the reference loop's signed-zero behavior.
    """
    if len(values) == 0:
        return None
    sequence = np.empty(len(values) + 1, dtype=np.float64)
    sequence[0] = 0
    sequence[1:] = values
    np.add.accumulate(sequence, out=sequence)
    return sequence[-1] / len(values)


def score_prepared(values, valid, indices, mode):
    """Score one bootstrap sample, retaining duplicates and sample order."""
    if mode == 'anonymous':
        selected = indices[valid[indices]]
        return ordered_mean(values[selected])
    scores = []
    for column in range(values.shape[1]):
        selected = indices[valid[indices, column]]
        if len(selected) == 0:
            continue
        terms = values[selected, column]
        if mode == 'agreement':
            # Integer counts are exact; this is the original fraction of matches.
            score = int(np.count_nonzero(terms)) / len(terms)
        else:
            # Non-anonymous CrossEntropy.score explicitly uses np.mean.
            score = np.mean(terms)
        scores.append(score)
    return sum(scores) / len(scores) if scores else None


def _majority_probability(count, total, panel):
    probability = 0
    for ii in range(int(panel / 2) + 1):
        i = int((panel + 1) / 2) + ii
        if i * 2 == panel:
            probability += comb(count, i) * comb(total - count, panel - i) / 2
        elif i * 2 > panel:
            probability += comb(count, i) * comb(total - count, panel - i)
    return probability / comb(total, panel)


def prepare_scores(scorer, predictions, W, anonymous):
    """Return (values, valid, mode), or None for an unsupported scorer.

    Values are computed once per original item. No expectation or stochastic
    result is approximated, and no nonlinear score is decomposed into items.
    """
    if (type(scorer) not in (AgreementScore, CrossEntropyScore)
            or any(name in vars(scorer) for name in
                   ("score", "expected_score", "expected_score_anonymous_raters", "expected_score_non_anonymous_raters"))):
        return None
    if scorer.ref_rater_combiner != 'majority_vote':
        return None
    if any(not is_missing(pred) and type(pred) not in
           (NumericPrediction, DiscretePrediction, DiscreteDistributionPrediction) for pred in predictions):
        return None
    vocabulary = _reference_vocabulary(W, predictions)
    ratings = W.to_numpy()
    shape = (len(ratings),) if anonymous else ratings.shape
    values = np.zeros(shape, dtype=np.float64)
    valid = np.zeros(shape, dtype=bool)
    agreement = type(scorer) is AgreementScore
    probabilities = {}
    for position, (row, pred) in enumerate(zip(ratings, predictions)):
        if _prediction_is_missing(pred):
            continue
        if not anonymous:
            for column, label in enumerate(row):
                if is_missing(label, vocabulary):
                    continue
                values[position, column] = (pred.value == label if agreement
                                             else log2(pred.label_probability(label)))
                valid[position, column] = True
            continue
        counts = {}
        for label in row:
            if not is_missing(label, vocabulary):
                counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        if total == 0:
            continue
        panel = min(scorer.num_ref_raters_per_virtual_rater, total)
        if len(counts) > 2 and panel > 1:
            # The public method uses a sampled virtual-rater matrix for the
            # entire subset, even when just one usable row needs sampling.
            return None
        if agreement:
            count = counts.get(pred.value, 0)
            if count:
                # Agreement's pandas value_counts supplies NumPy integer scalars.
                key = (count, total, panel)
                if key not in probabilities:
                    probabilities[key] = _majority_probability(
                        np.int64(count), np.int64(total), panel)
                values[position] = probabilities[key]
        else:
            term = 0
            # Dict insertion order is the original first-seen label order.
            for label, count in counts.items():
                key = (count, total, panel)
                if key not in probabilities:
                    probabilities[key] = _majority_probability(count, total, panel)
                term += probabilities[key] * log2(pred.label_probability(label))
            values[position] = term
        valid[position] = True
    return values, valid, ('anonymous' if anonymous else 'agreement' if agreement else 'cross_entropy')
