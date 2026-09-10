"""Shared validation for item-by-rater data."""

import numpy as np
import pandas as pd


def is_missing(value, allowable_labels=None):
    """Return whether a scalar rating is missing, including empty padding.

    An empty string can be a real categorical label when explicitly declared.
    Numeric zero and False are ratings, not missing-value sentinels.
    """
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return allowable_labels is None or "" not in allowable_labels
    missing = pd.isna(value)
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError("Rating values must be scalars")
    return bool(missing)


def as_rating_array(W):
    """Return a two-dimensional array: rows are items, columns are raters."""
    if isinstance(W, pd.DataFrame):
        array = W.to_numpy()
    else:
        try:
            array = np.asarray(W)
        except ValueError as exc:
            raise ValueError(
                "Ratings must be a two-dimensional item-by-rater matrix; "
                "pad uneven rows with missing values"
            ) from exc
    if array.ndim != 2:
        raise ValueError("Ratings must have two dimensions: items by raters")
    return array


class PreparedRatings:
    """Integer encodings of an immutable item-by-rater analysis snapshot."""

    def __init__(self, W, allowable_labels):
        self.labels = tuple(allowable_labels)
        self.lookup = {label: i for i, label in enumerate(self.labels)}
        if len(self.lookup) != len(self.labels):
            raise ValueError("allowable_labels must be unique")
        array = as_rating_array(W)
        self.codes = np.full(array.shape, -1, dtype=np.int32)
        self.counts = np.zeros((len(array), len(self.labels)), dtype=np.int64)
        for i, row in enumerate(array):
            for j, label in enumerate(row):
                if is_missing(label, self.labels):
                    continue
                if label not in self.lookup:
                    raise ValueError(f"Rating {label!r} is not in allowable_labels")
                code = self.lookup[label]
                self.codes[i, j] = code
                self.counts[i, code] += 1
        self.valid = self.codes >= 0
        self.totals = self.counts.sum(axis=1)
        self.frequencies = self.counts.sum(axis=0)
        for value in (self.codes, self.counts, self.valid, self.totals, self.frequencies):
            value.flags.writeable = False

    @property
    def nbytes(self):
        return sum(value.nbytes for value in
                   (self.codes, self.counts, self.valid, self.totals, self.frequencies))
