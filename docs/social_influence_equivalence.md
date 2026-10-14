# Independent and socially influenced survey estimates

Also available: [four additional graphs from Lorenz and Kao](social_influence_additional_datasets.md).

Four figures use **real human responses only**, from Becker, Brackbill & Centola
(2017). The experiment includes six candy jars, making it a close match to the
proposed silent-estimate versus shared-estimate comparison. Feedback was shown
on screen, rather than shouted; the candy brand is not established here.
[Original study](https://doi.org/10.1073/pnas.1615978114),
[author-hosted data](https://ndg.asc.upenn.edu/wp-content/uploads/2017/09/NDG_becker_brackbill_centola_collective_intelligence_dataset.zip).

## Figures

The [four-page PDF](../output/pdf/social_influence_equivalence.pdf) contains:

1. [Candy-jar panel curves](../benchmarks/results/social-influence-equivalence/candy_jar_panel_accuracy.png): initial private estimates versus final estimates, separately for private revision, decentralized feedback, and centralized feedback.
2. [All-task panel curves](../benchmarks/results/social-influence-equivalence/all_tasks_panel_accuracy.png): the same comparison across all 34 questions, including candies, pennies, calories, prices and visual percentages.
3. [Candy-jar mean movements](../benchmarks/results/social-influence-equivalence/candy_jar_mean_movement.png): the change in each group's mean relative to the actual count, for every jar and condition.
4. [Agreement versus accuracy](../benchmarks/results/social-influence-equivalence/agreement_vs_accuracy.png): whether tighter agreement coincides with improved full-group accuracy.

Each PNG has a matching editable SVG. These are new panel-score calculations,
not traced versions of published plots. See the [four-paper source comparison](social_influence_sources.md)
for the roles of the other suggested studies.

## What the observed responses show

For the candy jars, **five of six decentralized group means moved closer to the
truth, versus one of six centralized group means**. Three of six independent
control means improved after private revision. These counts describe six jars,
not a general probability of success.

Across all tasks, all 13 decentralized groups and all 13 centralized groups
reduced their average response spread. Nevertheless, full-group squared error
increased in three decentralized groups and five centralized groups. More
agreement does not guarantee more accuracy.

The panel curves answer a different question: **how many final recorded reports
match the expected accuracy of ten initial reports from the same condition?**

| Task subset | Condition | Fractional crossing | Smallest whole panel | Exploratory 95% interval, whole panel |
| --- | --- | ---: | ---: | --- |
| Six candy jars | Private revision, no feedback | 3.54 | 4 | 2-8 |
| Six candy jars | Decentralized feedback | 3.25 | 4 | 3-10 |
| Six candy jars | Centralized feedback | 2.53 | 3 | 2 to >31 |
| All tasks | Private revision, no feedback | 10.53 | 11 | 6-22 |
| All tasks | Decentralized feedback | 2.05 | 3 | 2-3 |
| All tasks | Centralized feedback | 4.97 | 5 | 2 to >31 |

Each row has its **own initial-reference accuracy**, so the crossings are not
a ranking of the three conditions against a common benchmark. A social report
can already contain information from other people in the original 40-person
network. These numbers do **not** establish that a three-person discussion can
replace a ten-person independent survey, or identify a recruitment-cost saving.

The centralized candy curves and the jar-movement counts can differ without
contradiction. Small samples of reports benefit from reduced spread. Full-group
means depend on where the group moved, and the squared-error average also
depends on the magnitude and normalization of each jar's error. Improving one
jar can outweigh several smaller deteriorations.

## Data and missingness

The original CSV has 5,909 person-task rows from 1,360 participants, 34 groups and
34 distinct tasks. We retain 5,626 rows from 1,314 participants with both initial
and final estimates. Within each group-task cell, both rounds use exactly the
same respondents; round two is not required. The source dictionary explains
that missing responses can appear as `NA`, while disconnected subject-task
records can be absent entirely. We do not fill in missing answers.

| Condition | Groups | Paired person-task records | Respondents per paired group-task |
| --- | ---: | ---: | --- |
| Private control | 8 | 1,286 | 35-40 |
| Decentralized | 13 | 2,190 | 33-40 |
| Centralized | 13 | 2,150 | 31-40 |

Candy tasks `3a` through `3f` have actual counts 279, 259, 268, 291, 278 and 278.
They contribute six groups per condition and 665 paired person-task records in
total. All tasks support the same panel range, **k = 1 through 31**, with no
changes to the included group/task mix as k increases.

No responses are trimmed or winsorized. In particular, control jar `3c` includes
initial answers of 9,999 and 3,885, revised to 200 and 136. They help explain the
large control movement in the third graph. Extreme answers also affect squared
errors and the published normalization scales.

## Exact panel calculation

For each group, task and round, let `z = (response - truth) / sd_pool`. The
released `sd_pool` is the sample standard deviation of **all available initial
responses** to that question. The script verifies this independently and keeps
the same scale across conditions and rounds; it does not normalize by the
smaller spread after feedback.

For a uniformly chosen subset of k distinct recorded respondents from a cell
with n paired respondents:

```text
expected_MSE(k) = mean(z)^2 + sample_variance(z) * (1/k - 1/n)
score(k) = -expected_MSE(k)
```

This is an exact finite-population identity, including for socially dependent
responses. It conditions on each intact observed group and does not pool people
from unrelated interacting networks. We average task scores within each group,
then give groups equal weight within each condition. This weighting differs
from simply pooling all person-task records. All-task controls and network
conditions have different task frequencies; no cross-condition causal contrast
is computed from these curves.

The horizontal reference is the same condition's initial score at k=10.
Fractional equivalence uses the existing `panel_power.first_equivalent_size`
function, linearly interpolating **negative MSE**, not RMS error. The smallest
whole-person crossing is retained separately. No fitted logarithm, smoothing or
extrapolation is used. Declining MSE with k follows algebraically for a fixed
pool; it does not itself demonstrate a benefit from recruiting more communicators.

The jar-movement graph uses percentage error for a more concrete axis. It does
not use percentage error to calculate the panel-equivalence numbers.

## Uncertainty and interpretation

5,000 bootstrap draws resample whole observed groups within each condition,
carrying all of a group's tasks, paired respondents and both rounds together.
The initial k=10 target is recomputed in every draw. This preserves dependence
within the original social groups. For the candy subset, each group corresponds
to one jar, so the bootstrap resamples observed group-and-jar clusters; it does
not isolate uncertainty from recruiting new respondents for six fixed jars.

The reported bounds are inverse empirical-CDF quantiles of whole-panel
crossings. Draws with no crossing by k=31 stay in the distribution as `>31`;
20.4% of centralized candy draws and 20.18% of centralized all-task draws have
no supported crossing. `>31` is censored, not a prediction that k=32 will suffice.
With six candy clusters, these intervals are exploratory, not formal
equivalence-test bounds or evidence of generalization to new jars/questions.

Before/after changes include effects of revision as well as feedback. The
private-revision control makes that distinction visible. Scores, metrics and
weighting differ from the paper's principal published results, so these graphs
should be cited as a reanalysis of the released responses.

## Reproduction and checks

```sh
.venv/bin/python research/social_influence_equivalence.py
.venv/bin/python research/social_influence_equivalence.py --plot-only
.venv/bin/python -m unittest research.tests.social_influence_tests research.tests.panel_power_tests -v
```

The full run expects the original CSV at
`.cache/social-influence-data/becker2017/becker_brackbill_centola_supporting_data.csv`.
Download and extract the linked author release there, or supply `--input PATH`.
The script verifies SHA-256
`2c12eac9df56e80bd8d4145b574116a54910e2efa888cf48a190ec1746c16fa1`.
Plot-only mode uses the retained results and rebuilds all PNGs, SVGs and the PDF.
The runner requires NumPy, pandas and Matplotlib, already available in the
project environment.

Checks compare every group-task-round calculation against direct one-report
scoring, exhaustive enumeration of all two-report panels, and the full observed
group mean. Additional tests check exhaustive subsets, invalid ranges and
censoring. Reference self-equivalence is verified for every bootstrap draw.
All four final PDF pages were rendered and visually inspected.

Saved evidence is in
[`benchmarks/results/social-influence-equivalence/`](../benchmarks/results/social-influence-equivalence/):
[panel curves](../benchmarks/results/social-influence-equivalence/panel_curves.csv),
[equivalence table](../benchmarks/results/social-influence-equivalence/equivalent_panels.csv),
[paired responses](../benchmarks/results/social-influence-equivalence/paired_responses.csv),
[per-task moments](../benchmarks/results/social-influence-equivalence/group_task_moments.csv),
[group diagnostics](../benchmarks/results/social-influence-equivalence/group_diagnostics.csv),
[provenance and checks](../benchmarks/results/social-influence-equivalence/results.json),
and compressed bootstrap draws. The original source remains unchanged.
