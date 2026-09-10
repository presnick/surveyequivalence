# CPU optimization validation and measurements

Measured on September 10, 2026, on the local Apple M1 Pro, macOS 26.6.2 arm64, Python 3.9.6. Numerical dependencies remain pinned: NumPy 1.24.4, pandas 2.0.3, SciPy 1.10.1, scikit-learn 1.2.2. The complete environment is in [requirements-lock.txt](requirements-lock.txt). `pip check` reports no broken requirements.

The optimized implementation preserves exact results in the frozen historical/corrected fixtures and every measured comparison. It substantially reduces deterministic Agreement/CrossEntropy work. Sampled AUC and F1 remain approximately unchanged because their metric implementations and sampling budgets are retained.

## Correctness baseline

The untouched historical revision `a4bbe2e` ran all 18 original test methods: **17 passed and one failed**. `test_leave_one_item_out` expects `0.2002 ± 0.001`, but returns `0.19342359767891681`. Both the corrected reference and optimized implementation return that exact same value. The existing assertion and tolerance remain unchanged. See [historical results](results/historical-tests.json) and [corrected results](results/corrected-tests.json).

The corrected reference is commit `093c94c`. Missing-data, alignment, panel-size, and unsupported-prediction fixes were applied before freezing the performance reference. Speedups below compare against that corrected reference, not buggy historical behavior.

The final suite ran **73 methods: 72 passed, one failed**, in 33.131 seconds against implementation commit `6ba59b9`. It includes the original tests, hand-calculated missing-data/panel/indexing regressions, exact differential comparisons, persistence checks, and worker cleanup tests. The only failure is the historical assertion above; the suite intentionally retains a nonzero exit status. See [final validation](results/final-tests.json). The exact command is:

```sh
.venv/bin/python -m unittest discover -s tests -p '*_tests.py' -v
```

Four historical pipelines and seventeen corrected pipelines are frozen using lossless floating-point encodings. Comparisons include classifier scores, every curve cell and missing-value mask, means, standard deviations, confidence bounds, equivalences, and performance ratios. Separate tests compare Bayesian probability vectors and prepared/public scorers exactly. Small-memory blocks and one/two-worker runs preserve seeded results. Additional frozen schedules exercise all three subset-generation branches across four seeds; 1,000 fixed seeds check uniform, independent plurality ties.

## Timing method and scope

Each release timing is the median of **three fresh subprocess runs** with identical inputs, item-sample schedules, dependencies, and native math-library thread limits. Every repetition checks exact repeatability, and each optimized run checks exact inputs/results against its reference. Python/NumPy seeds are fixed at 1729; the scorer matrix also uses `random_state=1729`. Workloads include one fixed classifier and the full-data sample plus the stated number of bootstrap samples.

All timing cases below use **`max_K=3`, `max_rater_subsets=20`**, retaining historical semantics: sizes 0, 1, 2, with sampling possibly producing fewer than 20 subsets for a size. Dense synthetic cases have 10 raters. The uneven case pads approximately 20% of cells outside the first two columns and uses reference panels of size 3. These are representative measurements, not an exhaustive Cartesian sweep of every scorer, panel size, subset budget, matrix size, and worker count. No larger-`max_K` speedup is claimed.

“Pipeline” measures `run()`; “process” includes interpreter/package startup, workload construction, pipeline execution, and result serialization. The largest ratios apply to computation, not import overhead. The machine was not dedicated benchmark hardware; the three observations per case are available in the associated `.run*.json` files.

| Workload | Bootstraps | Workers, both versions | Corrected pipeline | Optimized pipeline | Pipeline speedup | Corrected → optimized process |
|---|---:|---:|---:|---:|---:|---:|
| Packaged 50 items, ABC / CrossEntropy | 10 | 1 | 0.413 s | 0.041 s | 10.15× | 1.706 → 1.283 s |
| Dense 1,000 items, ABC / CrossEntropy | 20 | 1 | 10.813 s | 0.616 s | 17.56× | 12.183 → 1.799 s |
| Dense 2,000 items, ABC / CrossEntropy | 500 | 4 | 331.458 s | 6.465 s | 51.27× | 332.940 → 7.880 s |
| Uneven 1,000 items, ABC / CrossEntropy, panel 3 | 20 | 1 | 13.507 s | 0.535 s | 25.25× | 14.723 → 1.797 s |

Source pairs: [50-item reference](results/corrected-50.json) / [optimized](results/optimized-50.json), [1,000-item reference](results/corrected-1000.json) / [optimized](results/optimized-1000.json), [2,000-item reference](results/corrected-2000-500-p4.json) / [optimized](results/optimized-2000-500-p4.json), [uneven reference](results/matrix/sparse1000-corrected.json) / [optimized](results/matrix/sparse1000-optimized.json).

## Worker selection and memory

Preparing deterministic score contributions makes each bootstrap run inexpensive. One process is faster than two or four for the measured 2,000-item/500-bootstrap case. Automatic selection therefore chooses one process for prepared deterministic workloads and small tasks. Explicit `procs` remains authoritative, capped by available samples.

| Optimized worker setting | Pipeline median | Process median | Median sampled peak aggregate RSS | Shared numeric files |
|---|---:|---:|---:|---:|
| 1 | 1.430 s | 2.957 s | 213.4 MiB | 0 |
| 2 | 4.057 s | 5.639 s | 542.6 MiB | 1,508,512 bytes |
| 4 | 6.465 s | 7.880 s | 837.7 MiB | 1,508,512 bytes |
| Automatic (selected 1) | 1.507 s | 2.937 s | 213.7 MiB | 0 |

All four settings produce the identical result digest `b9c57ba6c9591fa0b8a76baef412399b9de2603ccce435d49b66b5f6c3dcd8a7`. [One worker](results/optimized-2000-500-p1.json), [two](results/optimized-2000-500-p2.json), [four](results/optimized-2000-500-p4.json), [automatic](results/optimized-2000-500-auto.json).

For this workload, changing from corrected four-worker execution to optimized automatic execution reduces pipeline time from 331.458 to 1.507 seconds and whole-process time from 332.940 to 2.937 seconds. This changes **both implementation and worker selection**; the same-worker speedup is the 51.27× comparison above.

RSS is sampled for the benchmark child and descendants every 0.1 seconds. Short-lived peaks can be missed. Aggregate RSS was unavailable under process-inspection restrictions for the first dense reference runs, so those JSON fields are null; no dense-reference aggregate-memory reduction is claimed. The separately measured uneven reference has median sampled peak RSS 262.7 MiB versus 140.5 MiB optimized. Frequency/Agreement measures 230.8 versus 135.3 MiB.

The corrected large run writes a 9,216,010-byte state pickle and reloads full state for each bootstrap task. The optimized large multiprocess run shares 1,508,512 bytes of numeric files read-only and initializes invariant task state once per worker. Tasks carry block descriptors and run IDs; workers open each block once. Temporary files are removed after normal completion and exceptions. These file sizes quantify temporary storage, **not total IPC serialization traffic**. Total IPC bytes and isolated worker-import/startup latency were not separately instrumented; their costs remain included in process/pipeline timing. The worker-count table exposes their combined runtime/memory impact.

`working_memory_mb=512` limits prepared working buffers, not total process RSS. Small budgets are validated with exact blocking/memmap tests. Inputs, Python objects, per-process imports, and OS page caches are outside that budget; custom components retain object storage and complete-run call order.

## Scorer matrix

These packaged 50-item cases use one worker. AUC/F1 retain 100 virtual reference raters and their existing metric calls. Small speed differences for those sampled metrics are within ordinary timing variation; they are not meaningful acceleration claims.

| Combiner / scorer | Bootstraps | Corrected pipeline | Optimized pipeline | Ratio |
|---|---:|---:|---:|---:|
| Frequency / Agreement | 10 | 4.674 s | 0.033 s | 140.08× |
| Plurality / Agreement | 2 | 1.369 s | 0.160 s | 8.54× |
| ABC / soft DMI | 2 | 0.314 s | 0.134 s | 2.34× |
| ABC / hard DMI | 2 | 0.327 s | 0.148 s | 2.21× |
| ABC / AUC | 1 | 8.610 s | 8.305 s | 1.04× |
| ABC / F1 | 1 | 8.688 s | 8.586 s | 1.01× |

All comparisons report exact results and repeatability. Raw pairs are in [results/matrix](results/matrix); [run_release_matrix.py](run_release_matrix.py) reproduces them. Correlation retains its public metric path but has no release timing case in this matrix.

## Remaining CPU work and GPU decision

The final automatic 2,000-item run records 29 prepared subsets, 58,000 prediction requests, 47,442 prediction-cache hits, and 10,558 misses (81.8% hit rate). The first run's unprofiled stage times are approximately 0.035 seconds preparing rating counts, 0.371 generating predictions, 0.804 preparing score contributions, and 0.213 scoring bootstrap samples. Analysis snapshots start with empty numeric caches; these figures include cold preparation and subsequent reuse within one analysis. Re-running an analysis deliberately rebuilds caches to handle changed inputs. Separate persistent cross-analysis warm-cache speedups are not claimed.

A separate [parent-process profile](results/profile-2000-500.run1.json) identifies score preparation/missing-value checks as the dominant remaining work. It takes 2.545 seconds with profiling enabled and is excluded from speedup tables. Count preparation accounts for about 2% of that profiled run, offering no dominant integer-counting batch to move to the GPU.

CPU ships as the default with no GPU dependency. PyTorch's MPS implementation rejects float64, and CPU/GPU floating-point operations are not generally bitwise identical. See the [MPS implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/EmptyTensor.cpp) and [numerical accuracy documentation](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html). No GPU backend was installed, timed, or claimed to improve these workloads. A future GPU path must identify a dominant compatible operation, pass exact parity, and beat end-to-end CPU runtime including initialization, transfer, and synchronization.

Usage, compatibility details, and reproduction commands are in [docs/performance.md](../docs/performance.md).
