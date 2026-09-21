# Results index

Start with the [new repeated-human DICES results](../../docs/dices_validation_results.md), [earlier leaderboard progress](../../docs/arena_leaderboard_progress.md), [paper-conformance audit](../../docs/paper_conformance_audit.md) and
[Arena/MT-Bench results](../../docs/arena_validation_results.md).

## Current artifacts

| Purpose | Location |
| --- | --- |
| DICES uncalibrated human-equivalence result | [PNG](dices-validation/dices_leaderboard.png), [PDF](dices-validation/dices_leaderboard.pdf), [table](dices-validation/leaderboard.csv) |
| Exploratory calibrated DICES leaderboard | [PNG](dices-calibration/dices_calibrated_leaderboard.png), [PDF](dices-calibration/dices_calibrated_leaderboard.pdf), [table](dices-calibration/leaderboard.csv) |
| AlpacaEval human baseline, AI predictions pending | [PNG](alpaca-human-baseline/alpaca_human_curves.png), [manifest](alpaca-human-baseline/manifest.json) |
| AlpacaEval blinded AI batch and tested scoring workflow; inference pending | [Protocol and commands](../../docs/alpaca_judge_validation.md), [verification](alpaca-judge-validation/readiness.json) |
| Held-out Arena calibration (separate experiment) | [PNG](arena-calibration/calibration_comparison.png), [SVG](arena-calibration/calibration_comparison.svg), [results](arena-calibration/results.json) |
| Balanced paper-protocol MT-Bench replay | [Checks and equivalence](paper-protocol/mt_check.json) |
| Arena scores relative to the zero-human prior | [PNG](paper-audit/arena_paper_scorecard.png), [SVG](paper-audit/arena_paper_scorecard.svg) |
| MT-Bench power curve and equivalence interval | [PNG](paper-audit/rater_equivalence_paper_view.png), [SVG](paper-audit/rater_equivalence_paper_view.svg) |
| Paper-specific counterexamples | [Conformance results](paper-audit/conformance.json) |
| Current general mathematical diagnostics | [Summary](paper-protocol/diagnostics/summary.json) |
| Arena zero-equivalence boundary | [Baseline comparison](paper-audit/arena_zero_boundary.json) |
| Real-data numerical results and provenance | [Source manifest](arena-validation/manifest.json), [judge results](arena-validation/arena_results.json), [MT summary](arena-validation/mt_equivalence_summary.json) |
| Current suites: 109 core tests and 25 research tests | [Core suite](../../tests/README.md), [research suite](../../research/README.md); both passed after cleanup |
| CPU benchmark interpretation | [Benchmark report](../REPORT.md) |

The historical Arena result JSON retains the original broad “not identifiable”
status. The paper audit corrects that interpretation: zero log-score equivalence
is identifiable here; positive human-panel equivalence needs additional ratings.

## Consolidation

The superseded `arena_judge_comparison` and `mt_human_equivalence` PNG/PDF files
were removed. Both the validation runner's `--plot-only` option and the dedicated
plot script now use the current figure implementation. PNG and SVG are deliberate
raster/vector exports of each figure.

Byte-identical `combiners.json`, `execution.json`, and `statistics.json` copies
from `correctness-audit`, `arena-validation/baseline-audit`, and
`arena-validation/post-fix-audit` were consolidated under
`shared-audit-evidence`. Historical summaries retain their original run metadata
and point to these immutable reports with SHA-256 checksums. The current audit's
copies remain independently regenerable, so a future fix cannot overwrite the
historical evidence. Distinct scorer reports remain because they capture the
before/after factorial fix.

These are intentionally retained, rather than stale duplicates:

- `*.run1.json`, `*.run2.json`, and `*.run3.json` are independent measured runs.
  Their unnumbered files summarize medians and repeatability.
- Historical test reports and frozen fixtures establish before/after behavior.
- Bayesian and Frequency MT tables record separate analyses whose hard-decision
  results happen to be identical on this cohort.
- Saved analyses contain their own datasets and predictions so the existing
  loader can open each analysis independently; matching input bytes do not make
  the analyses interchangeable.
- The original mathematical audit records historical findings and additional
  resource measurements. The paper audit is the current interpretation.

With the saved results present, regenerate current figures with:

```sh
.venv/bin/python research/arena.py --plot-only
```

The saved zero-human baseline is reused when its input hashes match. Regenerating
a missing or stale baseline requires the validated raw-data cache. New analyses
using a custom output directory write their figures into its `figures` subfolder.
