"""Reproduce statistics/reporting defects without modifying numerical code.

Exit status 1 means at least one diagnostic violates its stated contract.
These probes are intentionally separate from the historical regression suite.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from surveyequivalence.equivalence import ClassifierResults, PowerCurve, Plot


def audit():
    findings = []

    def record(identifier, actual, expected, explanation):
        if isinstance(actual, np.generic):
            actual = actual.item()
        findings.append(dict(id=identifier, actual=actual, expected=expected,
                             passes=bool(actual == expected), explanation=explanation))

    plateau = PowerCurve(df=pd.DataFrame([{0: .2, 1: .5, 2: .5, 3: .8}]))
    record('equivalence_plateau', plateau.compute_one_equivalence(.5), 1,
           'The smallest survey size reaching .5 is 1, not the end of the plateau.')

    gap = PowerCurve(df=pd.DataFrame([{0: .2, 1: np.nan, 2: .8}]))
    actual = gap.compute_one_equivalence(.5)
    record('equivalence_missing_crossing_returns_baseline', actual == 0, False,
           'Zero raters score .2, so zero cannot be equivalent to .5. '
           'Return missing unless interpolation over the gap is explicitly defined.')

    left = PowerCurve(df=pd.DataFrame({1: [.8, np.nan, .8]}))
    right = PowerCurve(df=pd.DataFrame({1: [.5, .5, np.nan]}))
    record('reliability_missing_pairs', left.reliability_of_difference(right), 1,
           'Only one comparison is defined, and it is a win; undefined comparisons are not losses.')
    missing = PowerCurve(df=pd.DataFrame({1: [np.nan, np.nan]}))
    actual = missing.reliability_of_difference(PowerCurve(df=pd.DataFrame({1: [.5, .5]})))
    record('reliability_no_pairs', bool(pd.isna(actual)), True,
           'A fraction of wins with no defined comparisons is undefined, not zero.')

    indices = ['actual', 'bootstrap-2']
    power = PowerCurve(df=pd.DataFrame({0: [0., 0.], 1: [1., 1.]}, index=indices))
    classifiers = ClassifierResults(df=pd.DataFrame({'c': [.5, .5]}, index=indices))
    record('equivalence_index_preservation', power.compute_equivalences(classifiers).index.tolist(), indices,
           'Derived results must retain the sample identifiers used to align later comparisons.')
    record('ratio_index_preservation', power.compute_performance_ratio(classifiers, 1).index.tolist(), indices,
           'Ratio results must retain the sample identifiers used to align later comparisons.')

    # Inspect plotted segment endpoints directly, without relying on an image.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    power = PowerCurve(df=pd.DataFrame({0: [.2, .8, .8]}))
    figure, axes = plt.subplots()
    try:
        plot = Plot(axes, power)
        plot.plot_power_curve(axes, power, 'all', True, 'black')
        shown = axes.collections[0].get_segments()[0][:, 1].tolist()
        expected = [float(power.lower_bounds[0]), float(power.upper_bounds[0])]
        findings.append(dict(
            id='plotted_confidence_interval_endpoints', actual=shown, expected=expected,
            passes=bool(np.allclose(shown, expected, atol=1e-14, rtol=0)),
            explanation='Error distances are computed around the bootstrap mean but drawn around the actual score, shifting both endpoints.'))
    finally:
        plt.close(figure)

    return findings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/statistics-audit.json'))
    args = parser.parse_args()
    findings = audit()
    result = dict(scope='statistics and plotted intervals', findings=findings,
                  passed=sum(item['passes'] for item in findings),
                  failed=sum(not item['passes'] for item in findings))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(1 if result['failed'] else 0)
