# Performance and result preservation

The CPU implementation prepares deterministic predictions and score contributions once, then reuses them across bootstrap samples. It supports rectangular item-by-rater matrices with uneven numbers of observed ratings. No GPU dependency is required.

## Usage

```python
from surveyequivalence import AnalysisPipeline, AnonymousBayesianCombiner, CrossEntropyScore

pipeline = AnalysisPipeline(
    W,  # pandas DataFrame: items in rows, raters in columns
    combiner=AnonymousBayesianCombiner(),
    scorer=CrossEntropyScore(),
    allowable_labels=["pos", "neg"],
    anonymous_raters=True,
    num_bootstrap_item_samples=500,
    max_K=3,
    random_state=1729,
    working_memory_mb=512,
    procs=None,
)
```

`procs=None` selects serial execution for prepared deterministic scoring and small workloads. Explicit positive counts select that many workers, capped by the number of samples. On the measured M1 Pro workload, serial execution beats two and four workers after score preparation. Nonlinear and sampled scorers can still benefit from multiple processes on sufficiently large workloads.

The historical `max_K` convention is retained: evaluated survey sizes are `min_k` through `min(max_K, number_of_raters) - 1`, inclusive. Thus `max_K=3` evaluates sizes 0, 1, and 2. `max_rater_subsets` retains the original sampling behavior and is a cap, not a guarantee that exactly that many distinct subsets are produced.

An integer `random_state` supplies independent Python and NumPy streams for sampling, prediction generation, classifier scoring, and each bootstrap scoring task. Built-in seeded results are independent of worker count and subset blocking. `None` retains global RNG use. Stateful custom components retain the generic path; custom random generators outside Python's `random` and NumPy's legacy RNG are outside this seed contract.

`working_memory_mb` budgets prediction and score blocks. Larger analyses process subsets in order, and workers open shared arrays read-only. A single subset larger than the budget uses memory-mapped storage. The setting is not a hard RSS limit: input frames, sample indices, Python objects, native-library memory, and operating-system file caches are excluded. Custom components retain object storage where their interface requires it.

## Correctness changes

- `None`, `NaN`, `pd.NA`, and empty padding are consistently missing. Explicitly declared empty-string labels and numeric zero remain valid.
- A reference panel uses all available ratings when fewer than the requested number remain for an item. Items with no usable ratings/predictions are omitted together, preserving alignment.
- Missing entries no longer add DMI classes, force soft DMI to zero, or cause division by zero in short reference panels.
- An unsupported Bayesian prediction returns `None`, rather than accidentally becoming a uniform distribution. Omitting `item_id` uses all eligible training items; an explicit ID is a row position to exclude.
- Item IDs are separate from row positions, classifier indices are aligned, and expert/amateur columns map to their actual matrix positions.
- Verbosity no longer selects the scoring mode or consumes the results iterator. Undefined curves/scores produce missing equivalences rather than invented endpoints.
- New saves include typed inputs and parse result CSV floats with round-trip precision. Existing saved analyses still load; older files without a seed use `None`.

The pipeline takes its own input snapshot and rebuilds prepared data on `run()`. A built-in combiner constructed with `W` binds a snapshot of that training data. ABC's explicit training matrix remains authoritative; a bare ABC instance learns from the active pipeline matrix. Frequency's explicitly supplied `combine(W=...)` remains authoritative for its prior. Replace a bound combiner to use a different training snapshot.

## Exactness and tests

Exact comparisons are made within the pinned Python/dependency environment. No float32 conversion, floating-point reassociation, alternative logarithm, weighted-bootstrap approximation, or changed nonlinear metric is used.

Bayesian sums retain their original row order and falling-factorial arithmetic. Cross-entropy retains first-seen label order and `math.log2`. Anonymous bootstrap means use an ordered prefix addition beginning at zero; non-anonymous cross-entropy retains `np.mean`. Final pandas statistics are unchanged. Stochastic plurality outcomes are never cached across distinct predictions.

The fixtures in `tests/fixtures` cover four valid deterministic pipelines from the original revision and seventeen cases from the corrected reference. They compare probability-derived outputs, classifier scores, all curve cells, means, standard deviations, both confidence-bound methods, equivalences, and performance ratios using lossless float encodings. Additional tests check scalar/prepared parity, missing-data cases, worker failure cleanup, custom fallbacks, and seeded blocking/worker invariance.

Create the validation environment without changing numerical dependency versions:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r benchmarks/requirements-lock.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m unittest discover -s tests -p '*_tests.py' -v
```

The original 18-test baseline has **17 passes and one existing failure**. The final suite has **72 passes and that same failure across 73 tests**. `test_leave_one_item_out` expects `0.2002 ± 0.001`; the original and updated implementations both return `0.19342359767891681`. That assertion has not been changed, skipped, or marked expected-failure. See `benchmarks/results/historical-tests.json` and `benchmarks/results/final-tests.json`. The full-suite command therefore remains nonzero for this known baseline failure.

## Reproducing measurements

The corrected reference was frozen before count caching, score preparation, and the execution rewrite. It is reproducible from commit `093c94c`; the historical reference is `a4bbe2e`. Reference snapshots used for benchmarks reside outside the working tree, so subsequent implementation edits cannot change them.

```sh
mkdir -p /tmp/surveyequivalence-corrected
git archive 093c94c | tar -x -C /tmp/surveyequivalence-corrected
.venv/bin/python benchmarks/run_benchmarks.py \
  --repo /tmp/surveyequivalence-corrected --workload dense1000 \
  --bootstraps 20 --repeat 3 --output benchmarks/results/reference.json
.venv/bin/python benchmarks/run_benchmarks.py \
  --workload dense1000 --bootstraps 20 --repeat 3 \
  --compare benchmarks/results/reference.json --output benchmarks/results/current.json
```

Use `--procs 1`, `2`, `4`, or `auto`; `--random-state 1729` for reproducible sampled metrics; and `--working-memory-mb` to exercise blocking. The benchmark checks exact input and result digests and exits nonzero on mismatch. `run_release_matrix.py --reference PATH` runs the scorer matrix sequentially. `--profile` creates a separate parent-process profile; profiled timings should not be compared with unprofiled timings.

The harness records pipeline and whole-process wall time, stage timing, sampled aggregate process RSS, native process peak RSS, and temporary-array/pickle storage. OS process-inspection restrictions can make aggregate RSS unavailable; such fields are explicitly null, not estimated. Child `ru_maxrss` is a maximum, not the sum of all workers. Worker array bytes are measured from generated files; they are not a claim about total IPC traffic.

See [the benchmark report](../benchmarks/REPORT.md) for measured configurations, exact-parity checks, and limitations.

## GPU decision

CPU remains the default. This implementation's main improvements eliminate repeated Python/Pandas work and whole-state deserialization. The local M1 Pro's PyTorch MPS backend does not support float64, and floating-point operations are not generally bitwise identical across CPU/GPU backends. Sources: [MPS implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/EmptyTensor.cpp), [numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

No GPU backend was installed or benchmarked. Adoption remains conditional on a remaining dominant batch operation, exact parity, and faster end-to-end execution including setup and transfers. Integer preprocessing can be considered separately; a lower-precision numerical backend is outside this implementation's exact-result contract.
