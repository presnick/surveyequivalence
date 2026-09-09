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
