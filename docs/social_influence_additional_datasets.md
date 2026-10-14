# More real datasets: Lorenz and Kao

The [additional four-page PDF](../output/pdf/social_influence_additional_datasets.pdf)
adds two of the other linked studies. All plotted guesses are observed human
responses. The captions explicitly identify **initial answers before feedback**
and the **later feedback condition**.

| Study | Graphs | Data used |
| --- | --- | --- |
| Lorenz et al. (2011) | [Panel accuracy](../benchmarks/results/social-influence-additional/lorenz_panel_accuracy.png), [agreement versus accuracy](../benchmarks/results/social-influence-additional/lorenz_agreement_vs_accuracy.png) | 144 people, 12 sessions, 852 released person-question rows |
| Kao et al. (2018) | [Jar panel accuracy](../benchmarks/results/social-influence-additional/kao_jar_panel_accuracy.png), [individual revisions](../benchmarks/results/social-influence-additional/kao_individual_revisions.png) | 602 paired guesses for a jar containing 659 objects |

Every PNG has a matching SVG in the same directory. The earlier
[Becker candy-jar graphs](social_influence_equivalence.md) remain available.

## What is being compared

**Lorenz:** participants first answered privately, then revised four times.
Depending on the question, they received no feedback, the group average, or the
individual estimates of the group. The same people experienced all three
conditions on different questions. The graph compares rounds 1 and 5, preserving
the original session-question groups of 12.
[Paper](https://doi.org/10.1073/pnas.1008636108).

The respondent CSV comes from the repository associated with the user's fourth
link, Almaatouq et al. (2022). That paper reanalyzes earlier experiments; it is
not an additional newly collected sample. This release has 71 session-question
groups, with the final session's control question 6 absent. It also retains
extreme responses. We do not silently reproduce the original paper's exclusions
or claim to reproduce its published statistics.
[Released CSV](https://raw.githubusercontent.com/amaatouq/task-dependence/master/data/prior_research_data/lorenz2011.csv),
[2022 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530231/).

**Kao:** the same participant made a private guess, saw a displayed number, and
could revise. The displayed number was artificially generated relative to their
first guess and described as the average of previous participants. Thus the
responses are real, but this is a controlled social-cue experiment rather than
mutual information exchange among live peers. The paper's simulated repeated
interactions and fitted bias-correction models are not plotted.
[Paper, experimental methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC5938592/),
[original data](https://ndownloader.figshare.com/files/11094635),
[dictionary](https://ndownloader.figshare.com/files/11094623).

## Calculated panel equivalences

Each row compares final recorded reports with **10 initial reports from that
same condition**, using exactly the same available people and questions.

| Dataset / later condition | Fractional crossing | Smallest whole panel | Exploratory 95% whole-panel interval |
| --- | ---: | ---: | --- |
| Lorenz: no feedback | 9.66 | 10 | 6 to >12 |
| Lorenz: group average | <=1 | 1 | 1-2 |
| Lorenz: all estimates | 4.01 | 5 | 2-9 |
| Kao: displayed artificial cue | 13.68 | 14 | 12-20 |

`<=1` means that one observed final report already exceeds the reference's
expected score. It does not identify a fractional crossing below one report.
`>12` retains bootstrap draws with no crossing in the supported range, rather
than assuming that an unobserved larger panel would eventually match.

The Kao arithmetic-mean curve is less accurate after the cue throughout the
plotted range: about 14 final reports match 10 initial reports under this score.
Among the 602 people, 170 moved closer to the truth, 201 moved farther away,
and 231 left their guess unchanged. The arithmetic mean changed from 724 to
764 objects, while the median changed from 550 to 555 (truth: 659).

In Lorenz's average-information condition, response spread decreased in all
12 sessions, but full-group squared log error increased in two. Under full
information, spread decreased in 11 sessions; five converged while error
increased. These are descriptive session counts from this release.

The row-specific references, score units and aggregation rules differ. These
numbers are **not a ranking across studies or a common-reference ranking of
conditions**. Social reports in Lorenz can incorporate information from other
members of the original session. A panel of k recorded answers does not identify
an experiment or recruitment plan with only k interacting participants.

## Scoring and exact calculation

For a finite observed pool of n errors, the expected squared error of the mean
of k uniformly sampled, distinct reports is:

```text
MSE(k) = mean(error)^2 + sample_variance(error) * (1/k - 1/n)
score(k) = -MSE(k)
```

For **Lorenz**, `error = ln(guess / truth)`. Averaging these errors is equivalent
to scoring a geometric-mean panel prediction by squared natural-log error.
This follows the original study's use of log-transformed estimates while
introducing a new panel-equivalence calculation. It also prevents the very large
raw guesses from dominating as strongly as raw squared error would. It does not
discard those guesses. Arithmetic and geometric aggregation answer different
questions, so the score is named directly on the axes.

Calculate each session-question curve separately for k=1..12. Average questions
within each session-condition, then give the 12 sessions equal weight. The
control session with one released question retains that question; we do not
impute a second question or duplicate its responses. All round-1 and round-5
values are positive and complete. A zero in round 3 remains in the retained
release but is not used in this before/after comparison.

For **Kao**, `error = guess - 659`, so the panel prediction is the arithmetic mean
and the loss is squared error in objects. The curves show k=1..100 from the
602 recorded pairs; no extrapolation is needed. For display only, the score is
divided by 1,000. No outlier trimming or fitted correction is applied. The
individual-revision scatter uses logarithmic axes to show the full range, while
the improvement counts use ordinary absolute count error.

Both datasets use the existing `panel_power.first_equivalent_size` function to
interpolate in expected-score units. The code stores the fractional crossing
and smallest qualifying integer separately.

## Uncertainty

5,000 bootstrap draws use seed 28092027. Each draw recomputes its own initial
k=10 reference.

- **Lorenz:** resample whole sessions, retaining every included question,
  participant and both scored rounds. The same session draws are used across
  conditions because the people recur across conditions. There are only 12
  independent sessions, and the bootstrap resamples the observed session/task
  mix; it does not establish generalization to new questions.
- **Kao:** resample paired participant rows within each claimed cue-size `N`
  stratum, preserving its observed sample count. This keeps the cue mixture fixed
  and is conditional on the one jar and this recruited sample. `N` describes the
  supposed size of the displayed social average; it is not plotted panel size
  `k` and does not count actual observed peers.

Reported whole-panel bounds use inverse empirical-CDF quantiles with unresolved
draws retained. Lorenz has no crossing by k=12 in 23.42% of no-feedback draws
and 0.28% of full-information draws. These are exploratory intervals, not formal
equivalence tests or causal estimates of between-condition effects.

## Files, reproduction and checks

```sh
.venv/bin/python research/social_influence_additional.py
.venv/bin/python research/social_influence_additional.py --plot-only
.venv/bin/python -m unittest research.tests.social_influence_additional_tests research.tests.social_influence_tests research.tests.panel_power_tests -v
```

Full analysis expects the two verified original files under
`.cache/social-influence-data/`: `task-lorenz2011.csv` and
`kao-rsif20180130_si_005.csv`. Download the linked releases to those names if
reproducing from a fresh checkout. The script rejects source files with changed
SHA-256 hashes. Plot-only mode needs only retained results. The runner uses
NumPy, pandas and Matplotlib.

The [results directory](../benchmarks/results/social-influence-additional/)
retains [panel curves](../benchmarks/results/social-influence-additional/panel_curves.csv),
[equivalence results](../benchmarks/results/social-influence-additional/equivalent_panels.csv),
[Lorenz response rows](../benchmarks/results/social-influence-additional/lorenz_released_responses.csv),
[Kao response pairs](../benchmarks/results/social-influence-additional/kao_paired_responses.csv),
group/session diagnostics, bootstrap draws and
[provenance, scoring definitions and checks](../benchmarks/results/social-influence-additional/results.json).
Kao's prepared response table omits demographic fields; `source_row` refers to
file order, not a supplied participant identifier.

Verification covers direct single-response loss, exhaustive two-response panels,
the full observed pool, geometric-mean predictions versus log-error calculations,
unsupported crossing bounds, and reference self-equivalence in every bootstrap
draw. PDF pages are rendered and visually checked before delivery.
