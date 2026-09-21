"""Supported first crossings; bounds are not point estimates."""
import math


def _finite(value):
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def supported_equivalence(score, powers):
    keys = sorted(powers)
    if not keys or not _finite(score):
        return dict(status='undefined', estimate=None, bound=None)
    for pos, k in enumerate(keys):
        value = powers[k]
        if not _finite(value):
            return dict(status='missing_curve', estimate=None, bound=None)
        # A missing earlier interval can conceal the first crossing.
        if pos and k != keys[pos - 1] + 1:
            return dict(status='missing_curve', estimate=None, bound=None)
        if value >= score:
            if pos == 0:
                if k == 0 or value == score:
                    return dict(status='crossing', estimate=float(k), bound=None)
                return dict(status='below_range', estimate=None, bound=float(k))
            previous = keys[pos - 1]
            estimate = k - (value - score) / (value - powers[previous])
            return dict(status='crossing', estimate=float(estimate), bound=None)
    return dict(status='above_range', estimate=None, bound=float(keys[-1]))
