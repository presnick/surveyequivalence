from abc import ABC, abstractmethod
from typing import Sequence, Dict
import numpy as np
from .combiners import Prediction

def agreement_score(classifier_predictions: Sequence[Prediction],
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
