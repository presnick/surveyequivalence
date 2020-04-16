from abc import ABC, abstractmethod
from typing import Sequence, Dict
import numpy as np


class Prediction(ABC):
    @abstractmethod
    def __init__(self):
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


def frequency_predictor(allowable_labels: Sequence['str'],
                        labels: np.array,
                        item_id=None,
                        to_predict_for=None):
    """
    Ignore item_id and to_predict_for
    return a vector of frequencies with which the allowable labels occur

    >>> frequency_predictor(['pos', 'neg'], np.array([(1, 'pos'), (2, 'neg'), (4, 'neg')])).probabilities
    {0.3333333333333333, 0.6666666666666666}

    >>> frequency_predictor(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg'), (4, 'neg')])).probabilities
    {0.0, 1.0}
    """
    freqs = {k: 0 for k in allowable_labels}
    for label in labels[:, 1]:
        freqs[label] += 1
    tot = sum(freqs.values())
    return DiscreteDistributionPrediction(allowable_labels, [freqs[k] / tot for k in allowable_labels])


