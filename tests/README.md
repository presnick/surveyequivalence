# Core test suite

Run from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -p '*_tests.py' -v
```

The three original files are retained unchanged by the cleanup:

- `continuous_prediction_tests.py`
- `discrete_distribution_tests.py`
- `scoring_function_tests.py`

They contain all 18 original test methods. Earlier justified numerical fixes
remain in place; cleanup did not restore stale expectations or alter assertions.

The five additional suites protect distinct library contracts:

| File | Purpose |
| --- | --- |
| `combiner_regression_tests.py` | Missing/rectangular data, leave-one-out behavior, independent probability oracles. |
| `scorer_regression_tests.py` | Missing/reference alignment, factorial overflow, independent score calculations. |
| `pipeline_regression_tests.py` | Item/rater identity, persistence, reproducibility, paper protocol and supported crossings. |
| `optimization_tests.py` | Exact historical/corrected parity, memory blocks, worker cleanup and custom-component fallback. |
| `statistics_math_audit_tests.py` | Independent checks of supported finite statistical calculations. |

That is **109 core tests across eight files**. The 25 tests of retained research
adapters live separately in `research/tests/`; see [research/README.md](../research/README.md).

Frozen fixtures and their generators are retained to establish exact numerical
parity. The separate `benchmarks/run_correctness_audit.py` diagnostics still
describe unresolved domains; a green test suite does not erase those findings.
