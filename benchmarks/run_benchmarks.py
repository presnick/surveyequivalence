#!/usr/bin/env python3
"""Reproducible subprocess benchmarks and exact result comparisons.

The controller imports no scientific packages. Each repetition imports the
selected checkout in a fresh child process. RSS is sampled for that process and
its descendants together; it is a sampled peak, not an allocation measurement.
Profiling is optional because it materially affects execution time.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


def exact_value(value):
    """Encode floating point results losslessly, including nonfinite values."""
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float):
        return {'float_hex': value.hex()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [exact_value(part) for part in value]
    raise TypeError('Unsupported result scalar: %r' % (type(value),))


def frame_result(frame):
    return {
        'index': [exact_value(value) for value in frame.index],
        'columns': [exact_value(value) for value in frame.columns],
        'data': [[exact_value(value) for value in row]
                 for row in frame.to_numpy()],
    }


def pipeline_result(pipeline):
    result = {}
    for name in ('classifier_scores', 'expert_power_curve',
                 'expert_survey_equivalences', 'amateur_power_curve',
                 'amateur_survey_equivalences'):
        value = getattr(pipeline, name, None)
        if value is None:
            continue
        result[name] = {'df': frame_result(value.df)}
        for summary in ('means', 'stds', 'lower_bounds', 'upper_bounds'):
            result[name][summary] = frame_result(
                getattr(value, summary).to_frame(name=summary))
    ratio = getattr(pipeline, 'performance_ratio', None)
    if ratio is not None:
        result['performance_ratio'] = frame_result(ratio)
    return result


def result_digest(result):
    content = json.dumps(result, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(content.encode()).hexdigest()


def build_workload(spec, repo):
    import numpy as np
    import pandas as pd
    from surveyequivalence import DiscreteDistributionPrediction

    rng = np.random.RandomState(spec['seed'] + 1)
    labels = ['pos', 'neg']
    if spec['workload'] == 'repo50':
        path = repo / 'surveyequivalence/data/running_example_50_items'
        ratings = pd.read_csv(path / 'ref_rater_labels.csv', index_col=0)
        if spec['items'] is not None:
            ratings = ratings.iloc[:spec['items']].copy()
        if spec['raters'] is not None:
            ratings = ratings.iloc[:, :spec['raters']].copy()
        ratings = ratings.reset_index(drop=True)
    else:
        count = spec['items'] or {'dense1000': 1000, 'dense2000': 2000,
                                  'sparse1000': 1000, 'sparse2000': 2000}[
                                      spec['workload']]
        raters = spec['raters'] or 10
        probabilities = rng.uniform(.05, .95, size=(count, 1))
        draws = rng.random_sample((count, raters))
        values = np.where(draws < probabilities, 'pos', 'neg').astype(object)
        if spec['workload'].startswith('sparse'):
            absent = rng.random_sample(values.shape) < spec['missing_fraction']
            # Guarantee two observed labels per item while still exercising
            # variable missingness among held-out reference raters.
            absent[:, :min(2, raters)] = False
            values[absent] = None
        ratings = pd.DataFrame(values,
                               columns=['r%d' % i for i in range(raters)])
    probabilities = rng.uniform(.1, .9, size=len(ratings))
    classifiers = pd.DataFrame({
        'fixed_soft_classifier': [DiscreteDistributionPrediction(
            labels, [float(probability), float(1 - probability)])
            for probability in probabilities]
    }, index=ratings.index)
    item_samples = [ratings.index] + [
        pd.Index(rng.choice(ratings.index, size=len(ratings), replace=True))
        for _ in range(spec['bootstraps'])]
    input_payload = {'ratings': frame_result(ratings),
                     'probabilities': [exact_value(float(p)) for p in probabilities],
                     'item_samples': [list(map(int, sample)) for sample in item_samples]}
    return ratings, classifiers, item_samples, labels, result_digest(input_payload)


def worker(spec, output):
    # Insert the target before importing any package or helper from the repo.
    repo = Path(spec['repo']).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import cProfile
    import pstats
    import random
    import resource
    import numpy as np
    import pandas as pd
    import surveyequivalence
    from surveyequivalence import (
        AnalysisPipeline, AnonymousBayesianCombiner, FrequencyCombiner,
        PluralityVote, CrossEntropyScore, AgreementScore,
    )

    random.seed(spec['seed'])
    np.random.seed(spec['seed'])
    stages = {}

    class TimedPipeline(AnalysisPipeline):
        def compute_classifier_scores(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return super().compute_classifier_scores(*args, **kwargs)
            finally:
                stages['classifier_scoring_seconds'] = time.perf_counter() - start

        def compute_power_curve(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return super().compute_power_curve(*args, **kwargs)
            finally:
                stages['power_curve_seconds'] = time.perf_counter() - start

    start = time.perf_counter()
    ratings, classifiers, samples, labels, input_digest = build_workload(spec, repo)
    stages['workload_build_seconds'] = time.perf_counter() - start
    combine_cls = {'abc': AnonymousBayesianCombiner,
                   'frequency': FrequencyCombiner, 'plurality': PluralityVote}[
                       spec['combiner']]
    start = time.perf_counter()
    combiner = combine_cls(allowable_labels=labels,
                          **({'W': ratings} if spec['combiner'] == 'abc' else {}))
    stages['combiner_create_seconds'] = time.perf_counter() - start
    scorer = {'cross_entropy': CrossEntropyScore, 'agreement': AgreementScore}[
        spec['scorer']](num_ref_raters_per_virtual_rater=spec['panel_size'])
    start = time.perf_counter()
    pipeline = TimedPipeline(
        ratings, expert_cols=list(ratings.columns), classifier_predictions=classifiers,
        combiner=combiner, scorer=scorer, allowable_labels=labels,
        num_bootstrap_item_samples=spec['bootstraps'], item_samples=samples,
        max_K=spec['max_k'], max_rater_subsets=spec['max_subsets'],
        anonymous_raters=True, verbosity=1, procs=spec['procs'],
        run_on_creation=False)
    stages['pipeline_create_seconds'] = time.perf_counter() - start
    profiler = cProfile.Profile() if spec['profile'] else None
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    pipeline.run()
    if profiler:
        profiler.disable()
    stages['pipeline_run_seconds'] = time.perf_counter() - start
    values = pipeline_result(pipeline)
    profile_summary = []
    if profiler:
        profiler.dump_stats(str(output.with_suffix('.prof')))
        stats = pstats.Stats(profiler)
        interesting = {'get_predictions', 'make_prediction', 'compute_one_run',
                       'combine', 'sumOfProbabilities', 'probabilityOneItem',
                       'expected_score', 'dump', 'load'}
        for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
            if function in interesting or 'pickle' in function:
                profile_summary.append({'file': filename, 'line': line,
                                        'function': function, 'calls': calls,
                                        'own_seconds': own,
                                        'cumulative_seconds': cumulative})
    units = 1 if sys.platform == 'darwin' else 1024
    payload = {
        'spec': spec,
        'imported_package': str(Path(surveyequivalence.__file__).resolve()),
        'versions': {'python': sys.version, 'numpy': np.__version__,
                     'pandas': pd.__version__},
        'shape': list(ratings.shape),
        'input_digest': input_digest,
        'result_digest': result_digest(values),
        'results': values,
        'stages': stages,
        'resource': {
            'self_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units,
            'children_max_rss_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * units,
        },
        'parent_only_profile': profile_summary,
    }
    output.write_text(json.dumps(payload, indent=2) + '\n')


def process_tree_rss(root_pid):
    """Sample aggregate resident bytes and live process count, using ps."""
    try:
        raw = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    processes = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 3:
            pid, parent, rss = map(int, fields)
            processes[pid] = (parent, rss * 1024)
    included = {root_pid}
    while True:
        descendants = {pid for pid, (parent, _) in processes.items()
                       if parent in included}
        previous = len(included)
        included.update(descendants)
        if len(included) == previous:
            break
    return sum(processes[pid][1] for pid in included if pid in processes), len(included)


def temp_state_sizes(directory):
    total = pickle_bytes = 0
    for parent, _, filenames in os.walk(directory):
        for filename in filenames:
            try:
                size = (Path(parent) / filename).stat().st_size
            except FileNotFoundError:
                continue
            total += size
            if filename.endswith(('.pickle', '.pkl')):
                pickle_bytes += size
    return total, pickle_bytes


def controller(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = {name: getattr(args, name) for name in
            ('workload', 'items', 'raters', 'bootstraps', 'max_k', 'max_subsets',
             'combiner', 'scorer', 'panel_size', 'procs', 'seed', 'profile',
             'missing_fraction')}
    spec['repo'] = str(args.repo.resolve())
    runs = []
    for repetition in range(args.repeat):
        child_output = output.with_name(output.stem + '.run%d.json' % (repetition + 1))
        log_file = child_output.with_suffix('.log')
        with tempfile.TemporaryDirectory(prefix='surveyequivalence-benchmark-') as state_dir:
            env = dict(os.environ)
            env.update({'PYTHONPATH': str(args.repo.resolve()), 'PYTHONHASHSEED': str(args.seed),
                        'TMPDIR': state_dir, 'MPLBACKEND': 'Agg',
                        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'})
            started = time.perf_counter()
            with log_file.open('w') as log:
                process = subprocess.Popen(
                    [str(args.python.resolve()), str(Path(__file__).resolve()), '--worker',
                     json.dumps(spec), '--worker-output', str(child_output)],
                    cwd=spec['repo'], env=env, stdout=log, stderr=subprocess.STDOUT)
                peak_rss = peak_processes = None
                peak_temp = peak_pickle = samples = 0
                while process.poll() is None:
                    rss, process_count = process_tree_rss(process.pid)
                    if rss is not None:
                        peak_rss = max(peak_rss or 0, rss)
                        peak_processes = max(peak_processes or 0, process_count)
                    total, pickle_bytes = temp_state_sizes(state_dir)
                    peak_temp = max(peak_temp, total)
                    peak_pickle = max(peak_pickle, pickle_bytes)
                    samples += 1
                    time.sleep(args.sample_interval)
            if process.returncode:
                raise RuntimeError('Benchmark child failed (%d); see %s' %
                                   (process.returncode, log_file))
            run = json.loads(child_output.read_text())
            run['controller_metrics'] = {
                'subprocess_end_to_end_seconds': time.perf_counter() - started,
                'sampled_tree_peak_rss_bytes': peak_rss,
                'sampled_peak_process_count': peak_processes,
                'sampled_peak_temp_bytes': peak_temp,
                'sampled_peak_pickle_bytes': peak_pickle,
                'samples': samples,
                'sample_interval_seconds': args.sample_interval,
            }
            child_output.write_text(json.dumps(run, indent=2) + '\n')
            runs.append(run)
            print('run %d: %.3fs pipeline, digest %s' %
                  (repetition + 1, run['stages']['pipeline_run_seconds'], run['result_digest']))
    repeatable = len({run['result_digest'] for run in runs}) == 1
    summary = {
        'spec': spec, 'repeat': args.repeat, 'exactly_repeatable': repeatable,
        'median_pipeline_seconds': statistics.median(
            run['stages']['pipeline_run_seconds'] for run in runs),
        'median_end_to_end_seconds': statistics.median(
            run['controller_metrics']['subprocess_end_to_end_seconds'] for run in runs),
        'runs': [str(output.with_name(output.stem + '.run%d.json' % (i + 1)))
                 for i in range(len(runs))],
        'input_digest': runs[0]['input_digest'],
        'result_digest': runs[0]['result_digest'],
        'results': runs[0]['results'],
        'measurement_notes': [
            'RSS and temporary file sizes are sampled; short-lived peaks may be missed.',
            'Child ru_maxrss is a maximum, not the sum across workers.',
            'cProfile describes the parent process only; pickle loads in workers are not counted.',
            'Native math-library threads are capped at one per process for comparable CPU runs.',
        ],
    }
    success = repeatable
    if args.compare:
        reference = json.loads(args.compare.read_text())
        same_input = summary['input_digest'] == reference['input_digest']
        same_results = summary['results'] == reference['results']
        summary['comparison'] = {'reference': str(args.compare.resolve()),
                                 'same_input': same_input, 'exact_results': same_results,
                                 'pipeline_speedup': reference['median_pipeline_seconds'] /
                                     summary['median_pipeline_seconds']}
        success = success and same_input and same_results
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('results', 'measurement_notes', 'spec')}, indent=2))
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--workload', choices=('repo50', 'dense1000', 'dense2000',
                                             'sparse1000', 'sparse2000'), default='repo50')
    parser.add_argument('--items', type=int)
    parser.add_argument('--raters', type=int)
    parser.add_argument('--bootstraps', type=int, default=2)
    parser.add_argument('--max-k', type=int, default=3)
    parser.add_argument('--max-subsets', type=int, default=20)
    parser.add_argument('--combiner', choices=('abc', 'frequency', 'plurality'), default='abc')
    parser.add_argument('--scorer', choices=('cross_entropy', 'agreement'), default='cross_entropy')
    parser.add_argument('--panel-size', type=int, default=1)
    parser.add_argument('--missing-fraction', type=float, default=.2)
    parser.add_argument('--procs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--sample-interval', type=float, default=.1)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/benchmark.json'))
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker), args.worker_output)
        return 0
    if args.repeat < 1 or args.sample_interval <= 0:
        parser.error('--repeat and --sample-interval must be positive')
    if args.combiner == 'plurality' and args.scorer != 'agreement':
        parser.error('plurality requires the agreement scorer')
    return controller(args)


if __name__ == '__main__':
    raise SystemExit(main())
