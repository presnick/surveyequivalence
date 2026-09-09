#!/usr/bin/env python3
"""Run the unchanged historical unittest files against a selected checkout.

Example:
    .venv/bin/python benchmarks/run_legacy_tests.py --repo /tmp/baseline \
        --report benchmarks/results/historical-tests.json
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = args.report.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import numpy as np
    import pandas as pd
    import surveyequivalence

    random.seed(args.seed)
    np.random.seed(args.seed)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # Explicitly enumerate the three historical files so new tests cannot
    # silently enter or leave the comparison suite.
    for filename in ('continuous_prediction_tests.py',
                     'discrete_distribution_tests.py',
                     'scoring_function_tests.py'):
        suite.addTests(loader.discover(str(repo / 'tests'), pattern=filename))
    log = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.with_suffix('.log').write_text(log.getvalue())
    payload = {
        'repo': str(repo),
        'imported_package': str(Path(surveyequivalence.__file__).resolve()),
        'python': sys.version,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'seed': args.seed,
        'elapsed_seconds': time.perf_counter() - started,
        'tests_run': result.testsRun,
        'successful': result.wasSuccessful(),
        'failures': [{'test': str(test), 'traceback': trace}
                     for test, trace in result.failures],
        'errors': [{'test': str(test), 'traceback': trace}
                   for test, trace in result.errors],
        'skipped': [{'test': str(test), 'reason': reason}
                    for test, reason in result.skipped],
    }
    report.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({key: payload[key] for key in
                      ('repo', 'tests_run', 'successful', 'elapsed_seconds')}))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
