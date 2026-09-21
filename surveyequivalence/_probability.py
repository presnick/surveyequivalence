"""Log scoring with explicit support for impossible events."""
from math import isfinite, log2


def log_probability(value):
    if not isfinite(value) or value < 0 or value > 1:
        raise ValueError('probabilities must be finite and between zero and one')
    return log2(value) if value else -float('inf')


def aligned_probabilities(prediction, labels):
    if set(prediction.label_names) != set(labels):
        raise ValueError('prediction label vocabularies must match')
    return [prediction.label_probability(label) for label in labels]
