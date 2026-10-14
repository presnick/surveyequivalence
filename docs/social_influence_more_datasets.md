# More real datasets: discussion, networks, and peer displays

Five new figures use three additional public human-response datasets. The
[five-page PDF](../output/pdf/social_influence_more_datasets.pdf) contains four
panel-size comparisons and a fifth graph of the experimentally varied number
of peers shown. No synthetic participants, simulated social rounds, or fitted
response models are used. Bootstrap resampling measures uncertainty in the
observed data; it does not supply invented responses.

**Initial private estimates mean answers before seeing anyone else's answers.**
The facet titles identify what happens later. The solid curves use the same
respondents' final individual answers after private revision, discussion, or
peer information. An initial estimate can still contain personal bias; private
means no feedback from other participants on that question.

## New datasets and figures

| Study | Actual social interaction | New figures |
| --- | --- | --- |
| [Gurcay, Mellers & Baron (2015), *The power of social influence on estimation accuracy*](https://doi.org/10.1002/bdm.1843) | Private answers, 90-second online chat, then individual answers. Separate individual-pay and group-pay chat conditions, plus private revision. | [Percentage-question panel curves](../benchmarks/results/social-influence-expanded/gurcay2015_panel_accuracy.png) |
| [Becker, Porter & Centola (2019), *The wisdom of partisan crowds*](https://doi.org/10.1073/pnas.1817195116) | Two revision rounds with feedback from four neighbors in politically homogeneous networks; private-revision controls. Experiment 1 conceals shared party affiliation; experiment 2 discloses it and displays partisan symbols. | [Experiment 1](../benchmarks/results/social-influence-expanded/becker2019_1_panel_accuracy.png), [experiment 2](../benchmarks/results/social-influence-expanded/becker2019_2_panel_accuracy.png) |
| [Jayles, Sire & Kurvers (2021), *Impact of sharing full versus averaged social information on social influence and estimation accuracy*](https://doi.org/10.1098/rsif.2021.0231) | Initial estimates, then sorted peer estimates, unsorted peer estimates, or their geometric mean. The number of peers contributing information varies. | [Panel curves](../benchmarks/results/social-influence-expanded/jayles2021_panel_accuracy.png), [number of peers shown](../benchmarks/results/social-influence-expanded/jayles2021_peer_count.png) |

These expand the earlier [Becker 2017 candy-jar analysis](social_influence_equivalence.md)
and [Lorenz 2011 / Kao 2018 analysis](social_influence_additional_datasets.md).
The search selected public records with individual answers, objective truths,
and identifiable before/after social-information phases. Jar illustrations,
aggregate-only results, and model-generated responses were not included.

## What the curves show

Each dashed curve is the expected accuracy of a panel of `k` initial reports.
Each solid curve reaggregates `k` final reports from the same original group
and question. Higher scores mean lower error, with zero perfect. Reports are
sampled without replacement within a group-question cell; different social
groups are never merged into a panel.

The dotted horizontal line is the score of five initial reports for Gurcay and
ten initial reports for the other studies. Every condition has its own initial
reference. A crossing answers: **how many final reports reach that condition's
initial-panel accuracy?** Fractional values interpolate between adjacent
integer panel scores. The whole-report count is the first supported integer
that reaches the reference.

| Dataset / later condition | Initial reference | Final reports required | Exploratory 95% interval for whole reports |
| --- | ---: | ---: | --- |
| Gurcay: private revision | 5 | 7 (interpolated 6.49) | 5 to >7 |
| Gurcay: chat, individual pay | 5 | >7 | 2 to >7 |
| Gurcay: chat, group pay | 5 | >7 | 3 to >7 |
| Becker experiment 1: private revision | 10 | 5 (4.74) | 4 to 9 |
| Becker experiment 1: network feedback | 10 | 2 (1.95) | 2 to 3 |
| Becker experiment 2: private revision | 10 | 13 (12.11) | 10 to 16 |
| Becker experiment 2: network feedback | 10 | 3 (2.70) | 1 to 6 |
| Jayles: sorted estimates | 10 | 6 (5.17) | 4 to 8 |
| Jayles: unsorted estimates | 10 | 5 (4.44) | 4 to 7 |
| Jayles: geometric mean | 10 | 4 (3.72) | 3 to 5 |

`>7` means the reference was not reached in the supported range. It does not
predict where a larger panel would cross. Bootstrap crossings outside the
range are retained as censored observations, not discarded.

In this percentage-question reanalysis, the chat conditions improve the
single-report score but have worse scores than their initial estimates at
larger panel sizes. The paired 95% intervals for the change at `k=5` include
zero in both chat conditions. This is not evidence that discussion generally
harms crowds. The partisan and display datasets show improvements in final
panel accuracy, but their equivalence counts are not a common-benchmark
ranking: their initial errors, questions, and scoring units differ.

The fifth graph holds `k=5` fixed and varies **tau**, the actual number of
peers whose responses contributed information (1, 3, 5, 7, 9, or 11). Its
vertical axis is initial minus final squared log error; positive means an
improvement. It is a different question from the panel-size curves. The wide
intervals do not establish a monotonic benefit from showing more peers.

## Data provenance and cohort choices

**Gurcay.** The [public reanalysis repository](https://github.com/amaatouq/task-dependence/blob/master/data/prior_research_data/gurcay2015.csv)
contains 4,448 person-question records for 278 people across 16 questions.
This analysis selects the seven percentage questions (IDs 9, 10, 12, 13, 14,
15, 16), which share a 0-100 response scale. The nine unbounded questions
have very different scales and extreme responses and are not analyzed here.
Of 1,946 selected records, 1,865 have both estimates: control 644,
individual-pay chat 608, group-pay chat 613. No missing values are imputed.
The control has 92 independent people and complete seven-question records;
the source's `group=40` pools those people and is not an interacting network.
Each chat condition has ten groups. Some group-question cells are entirely
missing and others have seven paired answers, so curves stop at seven.
Questions receive equal weight within each observed group, then groups
receive equal weight within a condition. Thus some chat groups have different
question coverage. These are descriptive, condition-specific comparisons.

**Becker.** Data come from the [original authors' repository](https://github.com/joshua-a-becker/wisdom-of-partisan-crowds)
and are also archived at [Harvard Dataverse](https://doi.org/10.7910/DVN/OE6UMR).
The original CSV preserves decimal truths and negative change targets.
Experiment 1 retains positive, finite answers in all three rounds, following
the author's log-analysis cohort: 4,019 of 4,353 rows. Experiment 2 retains
finite answers in all three rounds and excludes every row carrying the
author's `remove_extreme` flag: 4,114 of 4,468 rows. Its 74 flagged rows
overlap other possible exclusions. Round 1 and round 3 define the plotted
pairs. Requiring the middle round keeps the author's cohort, although it is
not plotted.

Each experiment has eight control groups and 24 social groups, equally split
between parties, with four questions per group. Original networks had 35
participants; retained paired counts vary, so both experiments are plotted
only to `k=25`. IDs are scoped by experiment, set, pair, network condition,
and party; participant codes restart across experiments. Parties contribute
equal group counts, but no mixed-party panels are constructed.

**Jayles.** The [figshare release](https://doi.org/10.6084/m9.figshare.12529571)
contains 9,072 paired records: 216 people, 18 groups of 12, and 42 questions
per person. Each display condition has six groups and 3,024 records. All
guesses and truths are positive and complete, so no rows are excluded.
Questions include marble counts and other factual estimates; the
[question catalog](../benchmarks/results/social-influence-expanded/jayles_questions.csv)
records their exact wording and truths. Each group encounters seven questions
at each tau; across its six groups, every display/tau combination covers all
42 questions. The peers are real participants, not virtual actors.

Raw input SHA-256 hashes, source links, exact inclusion counts, and results are
saved in [results.json](../benchmarks/results/social-influence-expanded/results.json).
The shared [paired response table](../benchmarks/results/social-influence-expanded/paired_responses.csv)
contains the analyzed estimates and errors, without the source's demographic
columns. The data releases and papers remain authoritative for their reuse terms.

## Scoring and uncertainty

For error values `e` in a cell of `n` paired people, the exact expected squared
error of the mean of `k` distinct reports is:

```text
expected MSE(k) = mean(e)^2 + sample_variance(e) * (1/k - 1/n)
plotted score  = -expected MSE(k)
```

- Gurcay: `e = estimate - truth`; arithmetic panels, percentage-point-squared loss.
- Becker experiment 1 and Jayles: `e = ln(estimate / truth)`; geometric panels,
  squared natural-log loss.
- Becker experiment 2: `e = (estimate - truth) / truth`; arithmetic panels,
  squared relative loss. Signed and decimal truths are retained, including
  immigration `-27.8` and unemployment `-46`. Negative targets are not removed
  or converted to logarithms.

Curves average cell losses equally over observed questions within a group and
then equally across groups. They are new analyses of the released responses,
not reproductions of the papers' particular accuracy metrics or effect sizes.

There are 5,000 paired percentile bootstrap replicates, seed 28092028.
Social conditions resample intact groups, keeping all questions and phases
together. Becker resamples within party. Gurcay's independent control
resamples whole people with all seven answers. Both the initial reference
and final curve are recomputed together in each replicate. The same Jayles
group draws are used across tau values. Curve intervals are in the CSV;
the first four pages display crossing intervals instead of shaded bands.
The fifth page shades pointwise gain intervals.

Small numbers of groups make these exploratory intervals. Questions are
treated as fixed; the analysis does not establish accuracy on new tasks or
new recruits. It also does not perform a formal causal comparison across
conditions or a statistical equivalence test. In particular, `k` counts
reports selected **after** the original social experiment; it does not say
that recruiting only `k` people would recreate the feedback or discussion.

## Reproduction and verification

The runner is [research/social_influence_expanded.py](../research/social_influence_expanded.py).
It expects these three files under `.cache/social-influence-data/`, verifies
their recorded hashes, and then prepares the paired response table:

| Local input | Public download |
| --- | --- |
| `task-gurcay2015.csv` | [CSV](https://raw.githubusercontent.com/amaatouq/task-dependence/master/data/prior_research_data/gurcay2015.csv) |
| `partisan-original.csv` | [Author CSV](https://raw.githubusercontent.com/joshua-a-becker/wisdom-of-partisan-crowds/master/Becker%20Centola%20Porter%20-%20Wisdom%20of%20Partisan%20Crowds%20-%20Supplementary%20Dataset.csv) |
| `jayles2021_Data.csv` | [figshare CSV](https://ndownloader.figshare.com/files/27962157) |

```sh
.venv/bin/python research/social_influence_expanded.py
.venv/bin/python research/social_influence_expanded.py --plot-only
.venv/bin/python -m unittest research.tests.social_influence_tests research.tests.social_influence_additional_tests research.tests.social_influence_expanded_tests research.tests.panel_power_tests -v
```

`--plot-only` uses saved results and needs no raw downloads. PNG and SVG exports,
panel curves, equivalence tables, group-question moments, bootstrap draws,
and the source manifest are in
[social-influence-expanded](../benchmarks/results/social-influence-expanded/).
Validation includes direct enumeration of all two-person panels in every
cell and phase, single-report and full-pool checks, bootstrap reference
self-equivalence, signed-target regression tests, and visual review of all
five PDF pages. Additional independent checks are recorded in
[verification.json](../benchmarks/results/social-influence-expanded/verification.json).
