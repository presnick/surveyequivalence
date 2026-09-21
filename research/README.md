# Research workflows

Dataset experiments live here, separate from the library and performance tools.
Saved results remain in `benchmarks/results/` so existing figure and report links
continue to work. Scripts require their documented local data caches; they do not
perform model inference automatically.

| Workflow | Command | Notes |
| --- | --- | --- |
| AlpacaEval human baseline | `.venv/bin/python research/alpaca.py baseline` | Rebuilds human curves and blinded inputs. |
| AlpacaEval AI evaluation | `.venv/bin/python research/alpaca.py prepare` / `score --responses FILE` | [Protocol](../docs/alpaca_judge_validation.md); inference is still pending. |
| Arena / MT-Bench | `.venv/bin/python research/arena.py --help` | Existing evaluation and figure reproduction. |
| DICES | `.venv/bin/python research/dices.py --help` | Released multi-judge predictions. |
| Exploratory DICES calibration | `.venv/bin/python research/dices_calibration.py` | Retained to reproduce the published exploratory leaderboard. |
| Judge’s Verdict ordinal pilot | `.venv/bin/python research/judges_verdict.py` | [Results and graph interpretation](../docs/judges_verdict_results.md); checks its custom scorer with independent enumeration. |

`arena_baseline.py` and `arena_figures.py` support the Arena runner's `--plot-only`
command. The Alpaca human-data builder and judge evaluator are now one script.

Run the 25 dataset-specific tests when changing these workflows:

```sh
.venv/bin/python -m unittest discover -s research/tests -p '*_tests.py' -v
```

The normal `tests/` suite covers library behavior without discovering research
adapters. See [tests/README.md](../tests/README.md).

## Retired tools

These completed investigations no longer need maintained entry points:

- `arena_calibration.py`: exploratory calibration of the single-human Arena
  dataset, which cannot identify positive panel equivalence.
- `arena_pipeline_checks.py` and `paper_protocol_check.py`: one-off real-data
  verification; saved findings remain, and general parity/protocol regressions
  stay in the core suite.
- `plot_dataset_human_curves.py`: candidate-dataset exploration; saved figures,
  curves, and provenance remain. Large generated selection tables and bootstrap
  schedules were removed from the maintained tree.
- `run_legacy_tests.py`: original-baseline capture, already recorded in retained
  reports and fixtures.

Only the six tests specific to the retired Arena calibration and candidate-curve
tools were removed. Independent mathematical tests were consolidated, not dropped.
Performance repetitions, historical fixtures, published results, and unresolved
audit evidence were retained. Recorded JSON manifests keep their historical
source paths and hashes; they are not rewritten to imply new runs.

A complete **local, git-ignored** recovery copy of the pre-cleanup tests,
benchmarks, results, docs, and README is at:

```text
.cache/repo-cleanup/20260920-201405/before-cleanup.tar.gz
```

Its adjacent `changes.json` lists moves and removals; `protected-sha256.json`
records the unchanged original test files and library source. The archive is not
distributed with a fresh clone. Inspect/extract it into a separate directory to
recover retired experiments without overwriting current work.
