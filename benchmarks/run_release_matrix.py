"""Run the representative scorer matrix sequentially, comparing exact outputs."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/matrix'))
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    benchmark = repo / 'benchmarks/run_benchmarks.py'
    cases = [
        ('sparse1000', ['--workload', 'sparse1000', '--bootstraps', '20', '--panel-size', '3']),
        ('frequency_agreement', ['--combiner', 'frequency', '--scorer', 'agreement', '--bootstraps', '10']),
        ('plurality_agreement', ['--combiner', 'plurality', '--scorer', 'agreement', '--bootstraps', '2']),
        ('soft_dmi', ['--scorer', 'soft_dmi', '--bootstraps', '2']),
        ('hard_dmi', ['--scorer', 'hard_dmi', '--bootstraps', '2']),
        ('auc', ['--scorer', 'auc', '--bootstraps', '1']),
        ('f1', ['--scorer', 'f1', '--bootstraps', '1']),
    ]
    for name, flags in cases:
        reference = args.output / (name + '-corrected.json')
        optimized = args.output / (name + '-optimized.json')
        common = [sys.executable, str(benchmark), '--repeat', str(args.repeat),
                  '--random-state', '1729', '--procs', '1', *flags]
        print('Reference:', name, flush=True)
        subprocess.run([*common, '--repo', str(args.reference.resolve()), '--output', str(reference)], check=True)
        print('Optimized:', name, flush=True)
        subprocess.run([*common, '--repo', str(repo), '--output', str(optimized), '--compare', str(reference)], check=True)


if __name__ == '__main__':
    main()
