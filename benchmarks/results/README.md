# Results index

Start with the [paper-conformance audit](../../docs/paper_conformance_audit.md) and
[Arena/MT-Bench results](../../docs/arena_validation_results.md).

## Current artifacts

| Purpose | Location |
| --- | --- |
| Arena scores relative to the zero-human prior | [PNG](paper-audit/arena_paper_scorecard.png), [SVG](paper-audit/arena_paper_scorecard.svg) |
| MT-Bench power curve and equivalence interval | [PNG](paper-audit/rater_equivalence_paper_view.png), [SVG](paper-audit/rater_equivalence_paper_view.svg) |
| Paper-specific counterexamples | [Conformance results](paper-audit/conformance.json) |
| Current general mathematical diagnostics | [Summary](paper-audit/legacy-diagnostics/summary.json) |
| Arena zero-equivalence boundary | [Baseline comparison](paper-audit/arena_zero_boundary.json) |
| Real-data numerical results and provenance | [Source manifest](arena-validation/manifest.json), [judge results](arena-validation/arena_results.json), [MT summary](arena-validation/mt_equivalence_summary.json) |
| Latest regression run | [Test record](paper-audit/regression-tests.json) |
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
.venv/bin/python benchmarks/arena_validation.py --plot-only
```

The saved zero-human baseline is reused when its input hashes match. Regenerating
a missing or stale baseline requires the validated raw-data cache. New analyses
using a custom output directory write their figures into its `figures` subfolder.
