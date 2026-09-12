"""Reproduce the confirmed mixed-label coercion bug; not a unit-test fixture.

Run: .venv/bin/python tests/combiner_input_audit.py
This exits nonzero while a raw numeric matrix padded with '' loses label types.
The mathematically expected priors are obtained by counting the three observed
ratings (Frequency) or averaging the two item distributions (ABC).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surveyequivalence._data import as_rating_array
from surveyequivalence.combiners import AnonymousBayesianCombiner, FrequencyCombiner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    matrix = [[0, 1], [1, '']]
    print('Raw input:', matrix)
    print('Converted array:', repr(as_rating_array(matrix)))
    failures = []
    findings = []
    for name, combiner, expected in (
        ('Frequency', FrequencyCombiner(), [1 / 3, 2 / 3]),
        ('ABC', AnonymousBayesianCombiner(), [.25, .75]),
    ):
        try:
            prediction = combiner.combine([0, 1], [], W=matrix)
            actual = None if prediction is None else prediction.probabilities
        except Exception as exc:
            actual = f'{type(exc).__name__}: {exc}'
        print(f'{name}: expected {expected}; actual {actual}')
        findings.append(dict(combiner=name, expected=expected, actual=actual,
                             passes=actual == expected))
        if actual != expected:
            failures.append(name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(input=matrix, findings=findings,
                                              failed=len(failures)), indent=2) + '\n')
    if failures:
        raise SystemExit('Confirmed input coercion failures: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
