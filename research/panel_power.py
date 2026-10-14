"""Crossings of empirical, higher-is-better panel power curves.

Implements the piecewise-linear presentation in arXiv:2106.01254v3,
Definitions 7 and 9. This does not fit a curve or extrapolate to missing sizes.
"""
import numpy as np


def first_equivalent_size(ks, scores, target):
    """First supported crossing, interpolated in expected-score units.

    A return value at ks[0] is a boundary, not an identified equivalence below
    that size. Infinity means no crossing in the observed range, not proof that
    arbitrarily large panels would fail. Nonmonotone curves are left intact.
    """
    ks = np.asarray(ks, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if (ks.ndim != 1 or scores.shape != ks.shape or len(ks) == 0
            or not np.isfinite(ks).all() or not np.isfinite(scores).all()
            or not np.isfinite(target) or np.any(np.diff(ks) <= 0)):
        raise ValueError('Need finite scores, target, and strictly increasing panel sizes')
    # Roundoff can otherwise move an exactly equal plateau to its far endpoint.
    tolerance = 1e-12 * max(1.0, abs(float(target)))
    reached = np.flatnonzero(scores >= target - tolerance)
    if not len(reached):
        return np.inf
    index = int(reached[0])
    if index == 0 or abs(scores[index] - target) <= tolerance:
        return float(ks[index])
    fraction = (target - scores[index - 1]) / (scores[index] - scores[index - 1])
    return float(ks[index - 1] + fraction * (ks[index] - ks[index - 1]))


def equivalence_label(value, ks):
    """Label unresolved endpoints explicitly; no invented zero-rater score."""
    ks = np.asarray(ks)
    if not np.isfinite(value):
        return f'>{ks[-1]:g}'
    if value <= ks[0]:
        return f'≤{ks[0]:g}'
    return f'{value:.2f}'
