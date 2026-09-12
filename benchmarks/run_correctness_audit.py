"""Run independent mathematical/compatibility diagnostics in fresh processes.

This is a failing gate while known correctness findings remain unresolved.
It does not alter or regenerate the frozen historical benchmark fixtures.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('benchmarks/results/correctness-audit'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    checks = []
    for name, script in (
        ('combiners', 'tests/combiner_input_audit.py'),
        ('scorers', 'benchmarks/scorer_math_audit.py'),
        ('statistics', 'benchmarks/statistics_audit.py'),
        ('execution', 'benchmarks/execution_audit.py'),
    ):
        output = directory / (name + '.json')
        command = [sys.executable, str(repo / script), '--output', str(output)]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=repo, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output.with_suffix('.log').write_text(result.stdout)
        record = dict(name=name, script=script, exit_code=result.returncode,
                      elapsed_seconds=time.perf_counter() - started,
                      passed=result.returncode == 0,
                      report=str(output.relative_to(directory)))
        if not output.exists():
            record['error'] = result.stdout
        checks.append(record)
        print('%s: %s (exit %s)' % (name, 'PASS' if record['passed'] else 'FAIL', result.returncode), flush=True)
    summary = dict(
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        python=sys.version, checks=checks, passed=all(check['passed'] for check in checks),
        numerical_source_sha256={str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted((repo / 'surveyequivalence').glob('*.py'))},
        note='Regression-suite success is separate from these independent diagnostics. Each FAIL needs resolution or an explicitly justified contract decision.')
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
