# Mathematical and implementation audit

Audit completed September 12, 2026 against runtime revision `b6a9846`, in the existing pinned environment. Production numerical code was not changed during this audit.

**The implementation cannot currently be certified mathematically correct across its advertised inputs.** Independent counterexamples expose incorrect scores, inconsistent estimator definitions, misleading derived results, and reproducibility/identity gaps. Exact parity with historical code preserved some of these pre-existing defects.

The existing suite plus 17 new independent-oracle methods passes **90/90 tests**. Those tests cover specified domains; separate diagnostic programs deliberately return a nonzero status for unresolved findings. Four diagnostic groups currently fail, reporting 25 counterexamples or contract discrepancies and one passing control. Multiple counterexamples can share one root cause. See [regression results](../benchmarks/results/correctness-audit/regression-tests.json), [diagnostic summary](../benchmarks/results/correctness-audit/summary.json), and the case reports linked below.

## Scope and evidence

The audit reviewed built-in combiners, missing-value preparation, scorer definitions, prepared scoring, pipeline alignment, caches, randomness, derived statistics, persistence, and plotted confidence intervals. It used source review, small explicit counterexamples, enumerated rater draws, rational probability calculations, and comparisons with the original scoring module. This is not a formal proof for arbitrary dimensions, every external custom component, every platform, or every example/calibration pipeline.

The new oracle checks include:

- 2,511 item/sequence probability cases: all four-position binary/missing rows and ordered binary sequences of lengths 0–4.
- 45 tiny binary/missing two-row matrices, every eligible holdout, and all prefixes within the tested support; three additional rectangular/missing multiclass matrices with reordered vocabularies.
- Binary reference-panel enumeration through seven observed raters, including ties and requested panels larger than available ratings.
- Rational confusion counts for precision/recall/F1, centered-product Pearson correlation, and permutation-formula determinants for hard/soft DMI in supported configurations.
- 126 strictly increasing four-point curves at 17 target scores, finite paired comparisons, and nonzero-denominator performance ratios.

Mathematical oracles use exact fractions before float conversion; new assertions allow explicitly small rounding errors where a rational result and the implementation follow different floating-point paths. Existing historical/corrected parity assertions still require exact equality and were not relaxed. Enumeration is independent of the runtime's falling-factorial/cache routines. Passing those bounded checks does not cover the failing cases below.

## Confirmed numerical errors: fix first

### 1. Factorial overflow and shared-cache contamination — critical

Location: `scoring_functions.py:45–65`; Agreement feeds NumPy integer counts into `comb`/`frac`.

For one item with 14 `a` ratings and 7 `b` ratings, a prediction of `a`, and a one-rater reference panel, Agreement must be `14/21 = 2/3`. With a fresh cache it instead returns **−8.01560442293844**. The recursive factorial cache can hold NumPy `int64` results; `21!` exceeds that type's range.

This also contaminates subsequent scorers. For predicted probabilities `[.8, .2]`, CrossEntropy should be `2/3 log2(.8) + 1/3 log2(.2) = −0.9885947615540288`. After that Agreement call, it returns **+11.88627681480937**. With a fresh factorial cache, the same CrossEntropy call gives the correct result. Thus results can depend on previous analyses, not just current inputs or seeds.

Proposed correction: ensure combinatorial arithmetic uses Python integers, isolate or eliminate the factorial cache, and test call-order invariance plus wide panels. A wholesale replacement with `math.comb` can change the rounding of valid historical floating-point operations, so first prove exact parity on unaffected inputs or record the deliberate numeric change. Higher-width Python factorial-to-float conversion also needs a stable strategy beyond merely preventing int64 overflow.

### 2. Binary AUC uses confidence in the winning class — high

Location: `scoring_functions.py:950` passes `prediction.value_prob` to `roc_auc_score`.

For truth `[pos, pos, neg, neg]` and `P(pos)=[.9,.8,.2,.1]`, every positive-negative pair is correctly ranked: **AUC = 1**. The implementation supplies `[.9,.8,.8,.9]`, the confidence in whichever class wins, and returns **0.5**.

Proposed correction: use the probability of one explicitly identified positive class for every item. Align that class with the metric's label convention, validate that each prediction supplies it, and test vocabulary reversal, ties, perfect/reversed rankings, and a single-class reference sample. Preserve the current unsupported-multiclass behavior until its averaging convention is explicitly defined.

### 3. Soft DMI changes when per-item label order changes — high

Locations: `scoring_functions.py:1186` and `:1230`, where raw probability vectors are accumulated.

Two semantically equivalent predictions can use different vocabulary orders. With `P(a)=[.8,.2]`, references `[a,b]`, and aligned coordinates, the joint matrix is `[[.4,.1],[.1,.4]]`; its determinant is **.15**. Reordering only the second prediction's vocabulary while retaining the same named probabilities changes the result to **0** in both direct and anonymous scoring.

Proposed correction: choose a canonical vocabulary and retrieve each contribution by label, rather than adding positional vectors. Validate compatible label sets. Test independent per-item permutations and globally reordered vocabularies.

All three root causes above were also reproduced using the scoring source from original revision `a4bbe2e`. That baseline probe loads the historical scoring module with current compatible prediction value classes; it is not an entire historical-environment reinstall. [Scorer evidence](../benchmarks/results/correctness-audit/scorers.json).

## Estimator definitions that need a decision

These are real numerical discrepancies between plausible/publicly described targets. Selecting a correction requires naming the intended statistic; they should not be silently changed as an optimization.

### 4. DMI of the expected joint is not expected DMI

Anonymous DMI computes `abs(det(E[joint]))`. Calling `score()` on each possible reference vector and averaging computes `E[abs(det(joint))]`. These differ because the absolute determinant is nonlinear.

For hard predictions `[a,b]`, with each item independently having reference choices `[a,b]`, four equally likely reference vectors yield expected DMI **.125**; the anonymous method returns **0**. For soft predictions `[.8,.2]` and `[.2,.8]`, the corresponding expectation is **.075**, while the method returns **0**.

If the target is DMI of a population joint distribution, the current plug-in calculation can be deliberate, but `expected_score` does not mean the same thing as in other scorers. If the target is the expectation of the finite-sample metric, use complete reference-vector enumeration for tiny cases and a properly specified Monte Carlo estimator otherwise. Compare estimators, budgets, and convergence explicitly before changing historical outputs.

### 5. Precision changes its averaging convention by path

With one reference column `[pos,pos]` and predictions `[pos,neg]`, direct/default and non-anonymous precision use micro averaging and return **.5**. Anonymous precision returns **1**, conditioning on the positive predictions. There is no reference randomness in this example to justify the difference.

Proposed correction: expose and consistently carry `average` and positive-label settings through direct, anonymous, non-anonymous, and pipeline calls. Decide whether backward compatibility requires a legacy mode. Direct precision/recall/F1 passed the independent confusion-count tests for their explicit averaging choices.

### 6. Reference-panel replacement differs between binary and multiclass paths

Binary closed forms count subsets without replacement. The general sampled path uses `np.random.choice` with replacement. With ratings `[a,a,b,c]` and panel size 3, exact enumeration gives `P(majority=a)=2/3` for distinct-rater subsets, versus **9/16** for the implemented with-replacement samples, including uniform ties.

The row-local size cap remains implemented, but it does not make these two sampling models equivalent. The prior optimization deliberately preserved this historical estimator difference. Specify replacement as part of the scorer contract, then offer a consistent policy with independently tested binary/multiclass cases.

### 7. Anonymous numeric correlation is declared but unimplemented

`Correlation.score()` passed a direct independent Pearson check. Its anonymous path defaults to a mean reference combiner, which the sampling implementation rejects with `NotImplementedError`. Even predictions `[1,2]` and the sole possible reference vector `[1,2]` fail, rather than returning 1.

Implement numeric panel means while retaining complete-vector Pearson scoring, or reject this configuration at construction with a precise supported-capability message. This is an unavailable path, not evidence that the implemented direct Pearson formula is wrong.

Evidence for findings 4–7 is in the [scorer audit](../benchmarks/results/correctness-audit/scorers.json), including the same outcomes in the historical scoring module.

## Derived statistics and presentation

### 8. Equivalence mishandles plateaus and missing crossing points

Location: `equivalence.py:328–338`.

For powers `{0:.2, 1:.5, 2:.5, 3:.8}` and classifier score `.5`, the function returns **2**. The smallest matching survey size is **1**. The stated definition is a smallest survey size meeting the classifier score; the original paper's pseudocode nevertheless uses a strict comparison, so this is also a definition/pseudocode inconsistency to resolve. See [the original paper, §3.4](https://arxiv.org/pdf/2106.01254v1) and [the current formal definition, §5.3](https://arxiv.org/pdf/2106.01254v3).

For `{0:.2, 1:missing, 2:.8}` and target `.5`, the function returns **0**, even though zero raters score .2. Return missing when the crossing is unsupported, or define interpolation over observed neighbors explicitly; bridging those particular neighbors would give 1. Do not silently manufacture a baseline equivalence.

Endpoint saturation above the measured range is historical behavior, but a returned maximum survey size should be identified as a bound, not an observed crossing. Nonzero `min_k` likewise needs lower-bound handling. Equality-aware earliest-crossing logic, explicit censoring, and a missing-point policy need separate regression cases.

### 9. Undefined reliability comparisons count as losses

Locations: `equivalence.py:355` and `:371`.

Comparisons `[(.8,.5), (missing,.5), (.8,missing)]` have one defined pair, which is a win. The function returns **1/3** rather than the conditional fraction **1**. With no defined pairs, it returns **0**. Undefined scores provide no evidence of a loss.

Align paired samples, apply a shared validity mask, divide by the number of defined pairs, and return missing when that number is zero. Report the effective sample count, particularly when missingness might be informative.

### 10. Plot error bars are shifted relative to stored confidence bounds

Location: `equivalence.py:1318–1332`, `Plot.plot_power_curve`.

The function derives error distances from the bootstrap mean but draws them around the actual-data score. For actual `.2` and bootstrap values `[.8,.8]`, the stored bounds are approximately `[-.0928203, 1.2928203]`; the plot instead shows `[-.4928203, .8928203]`. Both endpoints shift by the difference between actual and mean.

Draw the interval endpoints themselves, independent of the point estimate. If the point lies outside a percentile interval, separate the marker and interval segment rather than supplying negative `yerr`. Check both Matplotlib output and exported PGF data. Clipping intervals to a metric's possible range is a separate statistical choice.

### 11. Derived result identities and undefined ratios

`compute_equivalences` and `compute_performance_ratio` recreate a default row index instead of preserving supplied sample identifiers. Inputs indexed `['actual','bootstrap-2']` produce `[0,1]`, which can break later alignment. Preserve the input index and validate identical sample sets before comparisons.

A zero reference information gain makes the ratio denominator zero; a positive classifier gain currently produces infinity. Define whether that result is an explicit unbounded ratio or an undefined value, and ensure persistence/plots handle it intentionally. Small denominators and saturated equivalences can make ratios unstable even when the arithmetic is correct.

The confidence summaries also include the original full-data row among bootstrap replicates, and select percentile bounds using total row count rather than valid count per column. These are legacy statistical conventions, not changes introduced by optimization. Separate the point estimate from bootstrap replicates, document interval assumptions, and assess coverage before changing them. With many missing cells, 200 table rows need not mean 200 usable bootstrap estimates.

[Reproducible statistics and plotted-interval findings](../benchmarks/results/correctness-audit/statistics.json). The ratio-domain and bootstrap-coverage recommendations are source-reviewed limitations rather than conclusions from a coverage simulation.

## Inputs, snapshots, reproducibility, and persistence

### 12. Raw numeric lists with empty padding lose label types

`_data.py:29` uses `np.asarray` without preserving heterogeneous scalar types. Input `[[0,1],[1,'']]` becomes string-valued, although the declared vocabulary is `[0,1]`. Frequency returns `None` instead of `[1/3,2/3]`; ABC raises `KeyError('0')` instead of `[.25,.75]`. Typed object arrays and DataFrames avoid this particular conversion.

Preserve object-valued labels during raw-list normalization and validate vocabulary membership consistently. Include zero, empty padding, declared empty labels, and mixed numeric/categorical input cases. [Input evidence](../benchmarks/results/correctness-audit/combiners.json).

### 13. Reruns can misalign classifier rows; prediction snapshots are shallow

Constructor alignment does not repeat in `run()`. Reversing `pipeline.W` and rerunning retains the old positional classifier order. The probe yields classifier score **−1.216012940543713** versus **−1.1708500876232033** for a freshly constructed analysis with the same reordered data.

Furthermore, DataFrame copying does not copy the prediction objects inside it. Mutating an external prediction's probability list after pipeline construction changes the pipeline's purported snapshot.

Revalidate and align all identifiers at every new analysis snapshot. Snapshot built-in prediction values into immutable storage; define a copy/snapshot protocol for custom predictions. [Execution evidence](../benchmarks/results/correctness-audit/execution.json).

### 14. Global RNG reproducibility changes with blocking and dispatch

With explicit integer `random_state`, tested built-in blocking/worker comparisons pass. With `random_state=None` and identical external Python/NumPy seeds, changing only `working_memory_mb` changes plurality and sampled F1 results because task order changes. For one plurality probe, the actual k=0 score changes from **.5333333333333333** to **.4**.

Multiclass classifier scoring can also consume randomness column-first through prepared dispatch versus run-first through generic dispatch. Two mathematically equivalent scorer implementations then receive different draws.

This is a realization/reproducibility issue, not proof that the random-draw distribution is biased. Preserve legacy call order in that mode or explicitly document that stable execution requires `random_state`. Do not cache stochastic outcomes across distinct logical predictions.

### 15. Save/load does not preserve every allowed output identifier

A classifier named with integer `17` returns as string `'17'` in loaded score/equivalence/ratio table columns, while its input prediction column retains integer `17`. Floating-point round-trip precision is preserved in tested saves; typed output identifiers are not.

Persist typed result schemas or typed result frames and retain legacy CSV loading. Extend save/load checks to integer, string, and mixed identifiers plus derived-table joins. [Execution evidence](../benchmarks/results/correctness-audit/execution.json).

## Performance and resource improvements

Correctness fixes should precede further speed claims. The earlier performance matrix remains valid as a timing comparison, but a fast, exact reproduction of incorrect AUC or an inconsistent estimator is not validation of its scientific meaning. Recompute affected analyses and establish a newly reviewed correctness reference before benchmarking fixes.

The measured CPU improvements remain substantial in the tested configurations: approximately 10×, 18×, and 51× pipeline speedups for the 50/1,000/2,000-item ABC/CrossEntropy cases against the pre-optimization corrected snapshot. These use `max_K=3`, subset cap 20, and fixed worker counts; they are not measurements over the full default survey-size range. The 21-rater overflow example is outside those ten-rater benchmark inputs. [Original performance report](../benchmarks/REPORT.md).

The remaining resource work is concrete:

- Prepared rating arrays are allocated before deciding block size. In the reproducible 500×8 probe, a `.01 MiB` setting corresponds to 10,485 bytes, but base numeric preparation alone occupies **32,016 bytes**. The setting therefore does not strictly bound all prepared numeric storage, even apart from documented RSS exclusions.
- The same probe holds **4,509 prediction-cache entries**, approximately **2.36 MB** of uniquely counted Python cache objects. Cache storage is not bounded by the numeric working-buffer setting.
- Earlier block files remain until the complete analysis finishes: the probe grows from **13,512** to **770,184 bytes** across 57 blocks. After worker synchronization, obsolete block files could be removed instead of accumulating.
- Worker startup and full invariant-state copies can dominate short prepared bootstrap tasks. Automatic serial execution helps; measure the crossover on real nonlinear workloads rather than assuming more workers are faster.
- Score preparation and repeated missing-value checks remain the profiled deterministic bottleneck. Reuse validated masks/reference metadata more directly, preserving label and floating-point operation order.

[Memory probe](../benchmarks/memory_audit.py) / [measurements](../benchmarks/results/correctness-audit/memory.json). Object graph sizes and temporary files are not peak RSS; the instrumentation retains the private combiner to inspect caches already live during normal curve computation.

CPU should remain the default. No GPU path was benchmarked or validated. A GPU experiment should follow correctness repairs and identification of a remaining dominant operation that can satisfy the numerical contract; it should include setup, transfers, synchronization, and memory costs. Neither AUC/F1 acceleration nor a GPU speedup is established here.

## Recommended delivery order

1. **Repair unambiguous score errors:** factorial overflow/cache contamination, AUC positive-class probabilities, and soft-DMI label alignment. Turn the corresponding diagnostics into ordinary regressions. Confirm unaffected outputs remain exactly equal; approve and document affected changes through independent oracles.
2. **Repair result integrity:** rerun alignment, input type preservation, typed result persistence, reliability validity masks, and plotted interval endpoints. Freeze hand-checked expected outputs before implementation.
3. **Resolve statistical contracts:** DMI target, precision averaging, panel replacement, equality/censoring/missing points in equivalences, and bootstrap interval conventions. Publish explicit choices and compatibility behavior.
4. **Make correctness a required automated gate:** both the regular suite and `run_correctness_audit.py` must pass. Add pinned-environment CI and at least one additional platform. A green regression suite alone is insufficient while the diagnostics fail.
5. **Validate actual research workloads:** obtain the user's dataset path and intended combiner/scorer, survey sizes, subset cap, and bootstrap count. No such workload specification has yet been supplied; packaged and synthetic checks cannot stand in for it.
6. **Expand resource/performance measurements:** larger `max_K`, wider/uneven panels, correlation once implemented, nonlinear scorers, worker-count crossover, isolated startup/IPC costs, and budget enforcement. Require three repetitions and identical schedules against the newly reviewed reference.

CI, real-workload validation, the expanded timing grid, and runtime repairs above are proposed next work. This audit adds independent tests, diagnostic tools, evidence, and documentation; it does not silently change estimators or numerical implementations.

## Reproduce the evaluation

```sh
# Existing regressions plus independent-oracle checks: currently 90 pass.
.venv/bin/python -m unittest discover -s tests -p '*_tests.py' -v

# Unresolved independent counterexamples: currently exits 1, intentionally.
.venv/bin/python benchmarks/run_correctness_audit.py

# Optional resource instrumentation; does not claim an RSS limit.
.venv/bin/python benchmarks/memory_audit.py
```

The diagnostic runner executes each group in a fresh process so a corrupted global cache in one probe cannot invalidate another group's evidence. It records the runtime revision and numerical source hashes, and never regenerates the historical parity fixtures.
