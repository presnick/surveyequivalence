# Risk survey equivalence: calculated results

Calculated September 28, 2026 from released individual response records. These
are exploratory, dataset-specific comparisons of survey formats and aggregation
rules. **People are the raters; formats supply their responses; known values or
observed outcomes supply the scoring reference.** No LLM-generated responses are
used. SPF is an imperfect fit because its IDs may represent forecasting firms.

[Graphs (PNG)](../benchmarks/results/risk-survey-equivalence/panel_accuracy.png)
· [Graphs (SVG)](../benchmarks/results/risk-survey-equivalence/panel_accuracy.svg)
· [All equivalence numbers (CSV)](../benchmarks/results/risk-survey-equivalence/equivalent_panels.csv)
· [Run manifest](../benchmarks/results/risk-survey-equivalence/results.json)
· [Calculation script](../research/risk_survey_equivalence.py)

## What the numbers mean

For each comparison, fix a **10-person reference panel**. The reported number is
the smallest supported integer panel size for the alternative that matches or
improves that reference's expected loss. For example, SPF's estimate of 5 means
five explicit probability responses match ten point-implied decline votes under
the specified scoring rule. It does not mean one forecaster is universally worth
two other forecasters. Each dataset has its own reference and loss scale.

The graphs show **rater equivalence** in the
[paper's power-curve orientation (§5.3, Definitions 7 and 9)](https://arxiv.org/pdf/2106.01254v3):
panel size on a linear horizontal axis and **negative expected loss** on the
vertical axis, so higher is better. Probability judgments and inflation use
negative mean squared error; GDP uses negative Brier loss; decisions use negative
expected regret. Each graph has a single shared target: the expected score of
its specified 10-person reference method, shown by a dashed horizontal line and
a star. Circles and dotted vertical projections show the first matching panel
size for each method.

The graph labels `k*` use fractional first crossings from straight-line
interpolation between observed integer panel sizes, following the paper's
crossing interpretation. The tables below and stored CSV results retain the
smallest supported **integer** counts. `≤1` on a graph means the target is already
met at one person; no zero-person score was estimated. `>50`, for example, means
no crossing within the tested 1–50 range; it is not an estimate of 51 or evidence
that a larger panel will eventually cross. The empirical score curves use no
logarithmic fit, smoothing, or forced monotonicity. In particular, the risk
decision curve can decline as panels grow.

| Dataset / alternative | Reference: 10 people using… | People needed | 95% bootstrap interval |
| --- | --- | ---: | --- |
| Price Exp. 3: grouped icons, percentage wording | Random icons, percentage wording | **1** | 1–3 |
| Price Exp. 3: grouped icons, probability wording | Random icons, percentage wording | **4** | 2–>50 |
| Price Exp. 3: random icons, probability wording | Random icons, percentage wording | **>50** | >50–>50 |
| Price Exp. 3: grouped icons, probability wording | Grouped icons, percentage wording | **>50** | 15–>50 |
| Price Exp. 2: random icons | Numerical percentages, thresholded mean propensity | **1** | 1–>37 |
| Price Exp. 2: grouped icons | Numerical percentages, thresholded mean propensity | **3** | 1–>37 |
| Philadelphia Fed SPF: explicit GDP decline probability | Decline votes implied by GDP point forecasts | **5** | 1–>25 |
| NY Fed SCE: fitted density means, all eligible responses | Direct inflation point estimates | **1** | 1–2 |
| NY Fed SCE: fitted density means, paired bounded-point sensitivity | Direct inflation point estimates, same retained people | **2** | 2–2 |
| Sandia wildfire | Not calculated: individual records unavailable | — | — |

SCE counts and intervals are the same with CPI and PCE outcomes. A 2–2 interval
reflects discretized crossings in this bootstrap, not certainty about a future
population. Wide or censored intervals in the decision and GDP studies do not
establish reliable panel-size savings.

## Price Experiment 3: known probabilities

The [source paper](https://sjdm.org/~baron/journal/21/210615/jdm210615.html) and
[released data](https://osf.io/download/dgksf/) provide 218 people in four cells:
55 random/percentage, 56 random/probability, 55 grouped/percentage, and 52
grouped/probability. Each person has five repeated-trial averages, scored against
known percentages 5, 10, 15, 20, and 25. These are stimulus judgments, not forecasts
of future losses. Panels contain 1–50 people.

Ten random/percentage respondents have RMS error **4.8225 percentage points**.
One grouped/percentage respondent averages **4.6913**, meeting that reference.
At ten people the four methods have errors **4.8225, 8.2935, 1.6441, and 3.4056**,
respectively. The global comparison changes layout and sometimes wording.

To isolate wording, hold layout fixed: probability wording fails to match ten
percentage-wording respondents within 50 people under either layout. Thus this
study supports a layout advantage under the percentage question; it does not
show an advantage for single-event probability wording.

Uncertainty uses 4,000 whole-person bootstrap draws within randomized cells,
retaining all five answers for each sampled person. The five targets stay fixed,
so the intervals do not address generalization to other risk magnitudes.

## Price Experiment 2: risk decisions

The [released data](https://osf.io/download/frbtn/) contain 113 people: 37
numerical, 38 random-icon, and 38 grouped-icon participants. Each `risk_*` value
is a take-risk proportion over five trials. We choose a risk-neutral expected
points benchmark and the following explicit panel rule: take the risk when the
mean proportion exceeds 0.5, pass below 0.5, and randomize on ties.

Use the study's nominal +5, 0, and −5 point expected values. Regret is 5 points
for choosing the inferior action in a nonzero-EV case and zero for either action
in a zero-EV case. Average over all 15 cases. The nominal EVs deliberately follow
the study design; rounded loss amounts in its table can differ slightly.

We calculate the exact probability of the panel's action using integer subset
sums of `2*(5*proportion)-5`, orienting the sign to the optimal action. This avoids
treating a person's repeated trials as independent people. Panels span 1–37;
1,000 whole-person bootstrap draws resample independently within formats.

The 10-person numerical reference has expected regret **1.2230 points per
decision**, compared with **0.9557** for random icons and **1.1159** for grouped
icons at ten people. Their first crossings occur at 1 and 3 people, respectively.
Both intervals extend beyond 37, and both ten-person loss-difference intervals
include zero.

**Even the numerical format first meets its own ten-person benchmark at three
people.** Its curve is nonmonotonic: thresholding can reinforce a shared mistaken
preference as the panel grows. Consequently these are first-crossing counts,
not constant substitution ratios or evidence that all larger panels improve.
The action threshold and utility are analysis choices, not results reported by
the original authors or a universal definition of good risk management.

## Philadelphia Fed SPF: GDP downside forecasts

Join [individual RECESS and RGDP forecasts](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/individual-forecasts)
by year, quarter and ID. Use `RECESS2/100` for next-quarter decline probabilities,
and `RGDP3 < RGDP2` for point-implied binary votes. The latter panel mean is a
vote fraction, not an individually elicited probability. Score both against
negative next-quarter growth in [BEA real GDP via FRED](https://fred.stlouisfed.org/data/GDPC1)
using Brier loss.

There are **4,698 paired records across 137 survey quarters**, 1992Q1–2026Q1,
including 14 negative-growth target quarters. The minimum eligible pool is 25,
which caps the panel range. Forecast quarters 2026Q2 and 2026Q3 are excluded
because their next-quarter outcomes are unavailable in the downloaded snapshot.
GDP decline is not the same event as an NBER recession. Outcomes are latest
revised GDP, not first-release estimates; no vintage sensitivity is included.

The reference Brier loss is **0.092249**. Ten explicit probability responses score
**0.090674**, and the first matching panel contains **5 people**. However, a
4,000-draw circular bootstrap of eight-quarter blocks yields **1–>25**; **39.5%**
of draws do not cross within 25. The ten-person Brier difference is −0.001575
with interval **[−0.016810, 0.012050]**. Treat the apparent advantage as uncertain.
The observed forecaster pools are held fixed in this bootstrap.

## New York Fed SCE: inflation responses

Use the [2020–24 public microdata](https://www.newyorkfed.org/microeconomics/databank),
comparing `Q8v2part2` with released `Q9_mean` for the same person and month.
The point estimates are **already signed** in the released file; their signs
were checked against inflation/deflation direction `Q8v2` without applying a
second sign reversal. Require finite values, complete distribution bins between
0 and 100, and bin totals of 100.

Of 71,976 source rows, 1,327 fail completeness or validity requirements.
Outcome matching leaves **69,597 paired records**, **9,634 person IDs**, and
**59 survey months**. Score the next 12 months' inflation as
`100*(index[month+12]/index[month]-1)` using
[CPIAUCNS](https://fred.stlouisfed.org/data/CPIAUCNS) and, separately,
[PCEPI](https://fred.stlouisfed.org/data/PCEPI). These are research benchmarks for
a generic inflation question, not two different question formats. October 2024
is excluded from both runs because October 2025 CPI is unavailable; see the
[BLS shutdown explanation](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm).

| Outcome / cohort | Point estimates, 10 people: RMS error | Density means, 10 people: RMS error | Density people matching reference |
| --- | ---: | ---: | ---: |
| CPI, all eligible | 17.4859 | 2.8286 | 1 |
| PCE, all eligible | 17.5454 | 2.6388 | 1 |
| CPI, paired absolute point estimate ≤100 | 5.4837 | 2.8282 | 2 |
| PCE, paired absolute point estimate ≤100 | 5.6293 | 2.6376 | 2 |

Errors are percentage points. The primary point responses range from −100 to
10,000. After inspecting these extremes, we added a transparent sensitivity
removing 21 paired records with absolute point estimates above 100 from **both**
methods. This leaves 69,576 records and 9,630 IDs. It changes 1-for-10 to 2-for-10.
The released density means also incorporate the publisher's fitting procedure,
and a best guess need not target the same statistical functional as a mean.
These are therefore not clean causal effects of question wording alone.

We use equal-person means, not official survey weights. Uncertainty uses 4,000
circular moving-block draws of 12 retained months, holding people fixed. There
is one gap in the retained calendar sequence; only 59 months and overlapping
annual outcomes are available. The intervals are exploratory and do not cover
new-person recruitment or all sensitivity to the outcome and aggregation rule.

## Sandia wildfire: unavailable

The [dataset page](https://www.sandia.gov/research/datasets/details/decision-making-under-uncertainty-human-subjects-data-fire-evacuation-task-2025-02-13/)
describes a relevant comparison, but the underlying CSV host again returned
`Request Rejected` HTML rather than individual records. No equivalence number
was calculated. The manifest records this separately from computed results.

## Computation and reproduction

For the squared-loss analyses, the exact expected loss of a uniform panel of
`k` distinct people from an observed pool of `n` is:

```text
(pool_mean - truth)^2 + sample_variance * (1/k - 1/n)
```

Average across cases with equal weights. For probability judgments and inflation,
the tables and stored `score` columns give the square root of this average;
this is the root of expected squared error, not the expected value of a panel's
RMSE. The graphs instead plot the negative of the stored expected squared loss
to express a higher-is-better expected score. For binary GDP truth, the same
formula gives expected Brier loss, negated in the graphs. Experiment 2 uses the
separate exact action calculation described above and plots negative expected
regret. No Monte Carlo approximation of panels is used.

Each bootstrap recomputes its own reference. Equivalence intervals use inverse
empirical-CDF endpoints and retain no-crossing draws as censored, rather than
discarding them. Loss-difference columns in the CSV remain in **squared loss**
units for the RMS summaries; Brier and regret differences retain their native units.
Curves and intervals do not turn these exploratory benchmarks into validated
predictions of savings in a new population.

Raw data and prepared responses remain in the ignored `.cache/risk-survey-data/`.
Source hashes are in the linked inspection and run manifests. Preparation needs
`pyreadstat` (installed locally under `.cache/risk-python`); analysis needs
NumPy and pandas; plotting additionally needs Matplotlib. With the cached inputs:

```sh
# Run preparation under Python with NumPy, pandas, and pyreadstat available.
python research/risk_survey_equivalence.py --prepare
python research/risk_survey_equivalence.py
.venv/bin/python research/risk_survey_equivalence.py --plot-only
.venv/bin/python -m unittest discover -s research/tests -p '*_tests.py' -v
```

The recorded calculation used the bundled Python 3.12 environment; plotting used
the repository environment. All **35 research tests passed**, including six new
checks against exhaustive panel enumeration, threshold decisions, censoring,
block structure, and FRED's extra-row/missing-observation encoding. The exported
PNG was visually inspected. No library behavior was changed.
