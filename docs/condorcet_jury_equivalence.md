# Condorcet jury theory through rater equivalence

Completed October 6, 2026. The [ten-page report](../output/pdf/condorcet_jury_equivalence.pdf)
contains all six planned figures and a support/species appendix. Four released
datasets were analyzed; no new respondents or AI predictions were generated.

The project measures two different quantities: the number of **real reference
raters** needed to match a method, and the number of **hypothetical independent
jurors** needed to match a real panel. The second depends on an explicit
Condorcet model. It does not recover the number of independent people hidden
inside a correlated group.

## Observed growth and the independent-jury benchmark

| Dataset | Items | Largest common panel | One-person accuracy | Observed panel accuracy | Homogeneous iid model at that size | Smallest whole iid jury matching observed panel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NBA forecasts | 465 | 10 | 63.09% | 65.89% | 79.45% | 3 |
| Bluebirds | 108 | 39 | 63.56% | 75.93% | 95.88% | 7 |
| City comparisons | 1,000 | 97 | 67.72% | 76.02% | 99.99% | 5 |
| Ducks | 240 | 40 | 67.26% | 68.75% | 98.74% | 3 |

The model assumes independent correctness with one common success probability,
set to the same dataset's item-averaged single-rater accuracy. Observed panels
sample distinct recorded respondents without replacement. Their membership can
vary across items. All binary ties receive half credit.

The gaps are descriptive departures from this particular model, not identified
correlation penalties. Common task difficulty, differences among raters, and
the sampling design also matter. Adding more voters does not guarantee monotone
improvement in these finite observed pools. For example, Ducks peaks at 70.37%
with seven people, then reaches 68.75% with forty. Those are retrospective
curve features, not out-of-sample optimized panel-size recommendations.

The Ducks species curves make the mechanism tangible: Red-necked Grebe accuracy
falls from 38.85% for one respondent to 16.00% for forty, while Mallard accuracy
rises from 80.60% to 100%. The task asks whether an image contains a duck;
grebes are negatives. This task definition is confirmed in the
[original study](https://proceedings.neurips.cc/paper_files/paper/2010/file/0f9cafd014db7a619ddb4276af0d692c-Paper.pdf).
The numbers here are new calculations on the released labels, with half-credit
ties, rather than an exact reproduction of the publication's majority result.

See [full curves](../benchmarks/results/condorcet-jury-equivalence/panel_curves.csv),
[class and species curves](../benchmarks/results/condorcet-jury-equivalence/class_curves.csv),
and [summary and model-equivalence intervals](../benchmarks/results/condorcet-jury-equivalence/summary.csv).
The CSV includes balanced accuracy and a class-conditional iid benchmark as
additional checks on class imbalance.

## Can selection replace additional raters?

The learned comparisons use training questions to rank people and set weights,
then evaluate on separate questions. The reference is **nine random raters on
the same held-out items**, not the whole-dataset scores above.

| Dataset and method | Whole people reaching the reference | Interpolated crossing | Exploratory 95% whole-count interval | Accuracy at nine people |
| --- | ---: | ---: | --- | ---: |
| Bluebirds: random majority | 9 | 9.00 | 9–9 | 71.48% |
| Bluebirds: training-selected pool | 1 | ≤1 | 1–1 | 79.61% |
| Bluebirds: trained integer weights | 3 | 2.33 | 2–4 | 79.79% |
| Cities: random majority | 9 | 9.00 | Not estimated | 75.06% |
| Cities: training-selected pool | 5 | 4.26 | Not estimated | 77.13% |
| Cities: trained integer weights | 7 | 6.42 | Not estimated | 75.83% |

The selected pool consists of the top half of the recorded raters by training
accuracy: 19 of 39 Bluebirds raters, and 49 of 99 city raters. “One selected
rater” means the average performance of a uniformly selected person from
that pool, not a claim about a particular expert.

Bluebirds uses 54 training and 54 test images, splitting within truth class.
Cities first restricts to 802 complete 99-person items, then splits city
identities into disjoint training and test sets. This gives 200 training pairs,
215 test pairs, and 387 excluded cross-set pairs. The 198 incomplete pairs are
excluded only from the learned comparison; the main city curve retains all
1,000 pairs and supports panels through 97.

Weights use a Beta(2,2) posterior mean for training accuracy. Take three times
its nonnegative log odds, round to integers, and cap at ten. This fixed
convention permits exact subset calculations; it is not a claim of optimal
weights. A zero-weight response still counts toward headcount and contributes
no vote; an all-zero panel receives half credit. The same outcome-based
evaluation and half-credit tie policy apply to every method.

These counts exclude the cost of collecting training labels. Bird intervals
condition on this one trained policy and recorded workers. City comparisons
are descriptive for the held-out city pairs. Neither result establishes a
universal recruitment saving. [Equivalence and paired accuracy differences](../benchmarks/results/condorcet-jury-equivalence/learned_equivalence.csv),
[splits](../benchmarks/results/condorcet-jury-equivalence/learning_splits.csv),
[rater weights](../benchmarks/results/condorcet-jury-equivalence/trained_raters.csv).

The NBA aggregation comparison reproduces the earlier result: confidence
weighting requires ten whole people to match ten winner-only people; peer
agreement requires nine, as does the winner-only baseline itself because of
the odd/even plateau. At ten people the richer methods add 0.14 and 0.43
percentage points, with paired intervals including zero. Their unsupported
bootstrap crossings are retained: 26.13% and 21.16%, respectively.
[NBA equivalence table](../benchmarks/results/condorcet-jury-equivalence/nba_equivalence.csv).

## Figures and supporting data

Every figure has PNG and SVG exports in the results directory.

| Figure | Review image | Numerical source |
| --- | --- | --- |
| Condorcet's theoretical curves | [Figure 1](../benchmarks/results/condorcet-jury-equivalence/01_condorcet_theory.png) | `theory_curves.csv` |
| Real growth versus iid benchmark | [Figure 2](../benchmarks/results/condorcet-jury-equivalence/02_observed_vs_iid.png) | `panel_curves.csv`, `bootstrap_draws.npz` |
| NBA empirical equivalents | [Figure 3](../benchmarks/results/condorcet-jury-equivalence/03_nba_equivalence.png) | `nba_aggregation_curves.csv`, `nba_equivalence.csv` |
| Retrospective item difficulty | [Figure 4](../benchmarks/results/condorcet-jury-equivalence/04_item_difficulty.png) | `item_diagnostics.csv` |
| Held-out expertise versus headcount | [Figure 5](../benchmarks/results/condorcet-jury-equivalence/05_expertise_headcount.png) | `learned_curves.csv`, `learning_bootstrap.npz` |
| Shared-case model illustration | [Figure 6](../benchmarks/results/condorcet-jury-equivalence/06_shared_case_model.png) | `shared_case_model.csv` |
| Full city range and Ducks species | [Appendix](../benchmarks/results/condorcet-jury-equivalence/07_support_and_species.png) | `panel_curves.csv`, `class_curves.csv` |

Figure 6 is analytical: a beta-distributed success probability is shared by
jurors on each item, producing beta-binomial vote counts. It is not fitted to
the data. No simulated respondents are used in the empirical figures, and
the variance-based shortcut `k/[1+(k-1)*rho]` is not used as accuracy equivalence.

## Data audit, inference and validation

Original inputs are cached under `.cache/condorcet-data/`; their hashes match
the earlier source inspection. The normalized vote table stays in that cache.
[Data audit](../benchmarks/results/condorcet-jury-equivalence/data_audit.json)
records sources, hashes, exclusions, coding and denominators.

- Bluebirds and Ducks use the authors' pinned
  [CUBAM release](https://github.com/welinder/cubam/tree/fe5ba700f1adbb489c69af311558d64370d73d36/demo).
  Released bird truth labels were not independently re-adjudicated.
- The [city file](https://osf.io/vyx8a) has 120 raw participant IDs; retain the
  99 with 1,000 trials over two sessions. Remove 213 absent decisions despite
  their `DecisionAcc=0` coding. Normalize screen-position responses to city
  identity and verify correctness against historical populations. This retains
  98,787 votes, with 97–99 votes on each pair. The paper/release raw recruitment
  counts differ; the retained 99-person cohort matches the reanalysis.
- NBA reuses the hash-verified prepared responses and the original paired
  wave-bootstrap schedule. The exact majority calculations independently
  reproduce the saved scores. [NBA methods](sports_survey_equivalence.md).

Ordinary panel curves use hypergeometric majority tails. Learned weighted
panels use integer subset-sum dynamic programming with checked count bounds.
No panel Monte Carlo is needed. All comparisons hold their item sets fixed
over the reported panel sizes.

Bird intervals use 4,000 truth-stratified image resamples; NBA uses 10,000
resamples of 11 whole waves. Both condition on observed people. Each draw
recomputes the matched iid curve and equivalent count. These are pointwise,
exploratory intervals, not simultaneous bands or formal equivalence tests.
City pairs share city identities, so no unsupported iid-item bootstrap is
reported. New-rater or new-domain inference requires additional assumptions.

The theoretical inverse searches whole sizes 1–101 and records censored
crossings rather than inventing larger observed panels. Fractional values
interpolate the complete integer grid. First crossings do not imply that
every subsequent point remains above the target. When a single reference
rater already exceeds a target, `≤1` is a boundary rather than an estimated
fraction below one.

Six new tests plus ten relevant existing tests passed. They cover exhaustive
finite-panel enumeration, Bernoulli-event enumeration, weighted zero/tie
handling, numerical integration of the beta mixture, missing city decisions,
and first-crossing/plateau behavior. The final report was rendered and
visually reviewed. [Verification record](../benchmarks/results/condorcet-jury-equivalence/verification.json).

## Reproduction

The new entry points are [data preparation](../research/condorcet_data.py)
and [analysis/graphics/report](../research/condorcet_equivalence.py). The local
environment uses `.venv` plus optional PDF/YAML dependencies in
`.cache/condorcet-python`; [recorded package versions](../benchmarks/results/condorcet-jury-equivalence/environment.json)
document the completed run.

```sh
# First acquisition only; verifies pinned content hashes.
.venv/bin/python -B research/condorcet_data.py --download

# Offline, complete analysis plus PNG/SVG graphics and PDF.
.venv/bin/python -B research/condorcet_equivalence.py

# Rebuild figures/report from saved numerical tables.
.venv/bin/python -B research/condorcet_equivalence.py --plot-only

.venv/bin/python -B -m unittest research.tests.condorcet_tests research.tests.panel_power_tests research.tests.sports_equivalence_tests -v
```

The original [plan](condorcet_jury_equivalence_plan.md) remains as the design
record. Skin-lesion and RTE datasets remain reserve candidates; their new
analyses were not required for these six figures and are not represented as
completed. No AI-equivalence experiment was run.
