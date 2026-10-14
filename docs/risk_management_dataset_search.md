# Risk-management datasets for survey equivalence

Checked September 27, 2026. This search uses the advisor's interpretation:
individual people supply responses, question types define surveys, and a shared
outcome or known answer supports scoring. No new equivalence values were
calculated during this search.

**Update, September 28:** [Calculated equivalence results and graphs](risk_survey_equivalence.md)
are now available for both Price experiments, SPF, and SCE. The historical
search notes below describe the earlier inspection stage. Sandia's raw-data
host still rejects downloads, so that comparison remains uncalculated.

**Best clean question-type pilot: Price et al., Experiment 3. Best real-world
financial-risk extension: Philadelphia Fed Survey of Professional Forecasters,
subject to its forecaster-identity caveat and an outcome-data join.**

| Candidate | Comparison available | Scoring reference | Access and fit |
| --- | --- | --- | --- |
| Price et al. (2022), Experiment 3 | Percentage wording versus single-event probability wording, crossed with random/grouped icon arrays | Five known stimulus probabilities | Individual data downloaded and inspected; strongest question-type design, stylized task |
| Philadelphia Fed SPF | Explicit probability of GDP declining versus decline implied by GDP point forecasts | Actual quarterly real GDP change, to be joined | Individual files downloaded and inspected; relevant to macroeconomic risk, but IDs may follow firms |
| New York Fed SCE | Consumers' inflation point estimates versus probability distributions | Chosen realized inflation series, to be joined | 2020-24 workbook downloaded; fields verified, full completeness check pending |
| Price et al. (2022), Experiment 2 | Numerical percentages versus random/grouped icon arrays in take-risk/pass decisions | Expected payoff from known probabilities and losses | Individual summaries downloaded and inspected; communication-format comparison |
| Sandia wildfire evacuation | Text versus map encodings of burn risk | Experimental outcomes and a documented payoff rule | Public schema and paper verified; CSV server returned a rejection page, so records not inspected |

## 1. Price: question wording and risk communication

[Paper](https://sjdm.org/~baron/journal/21/210615/jdm210615.html),
[OSF](https://osf.io/wtndc/),
[Experiment 3 data](https://osf.io/download/dgksf/),
[Experiment 2 data](https://osf.io/download/frbtn/).

Experiment 3 randomly assigned question wording and array layout. The task
concerned red/blue jelly beans rather than actual losses. The underlying
percentages were 5, 10, 15, 20 and 25. This isolates a risk-communication mechanism;
it does not establish predictive performance on future adverse events.

Direct file inspection found **218 participant records** with `subject`,
`array_type`, `question_type`, `SNS`, and `judgment_5` through `judgment_25`.

| Array layout | Percentage question | Probability question |
| --- | ---: | ---: |
| Random | 55 | 56 |
| Grouped | 55 | 52 |

The five judgment columns contain each person's averages across repeated
presentations, not the complete 20-trial sequence.

Proposed analysis: hold array layout fixed, average k people's five estimates,
and score against the five known values using RMS error. Resample entire people
within each condition. Plot both wording curves before choosing a reference
panel size; report no supported crossing when appropriate. A second comparison
can hold wording fixed and vary layout. There are only five distinct targets,
so respondent uncertainty is not evidence of generalization across many risks.

Experiment 2 has **113 individual records**, with 37 numerical-format, 38
random-array and 38 grouped-array participants. Its 15 `risk_*` columns are
per-person take-risk proportions over five repetitions for each probability/
payoff combination. These are not 15 binary single-trial decisions. An analysis
would first define how to aggregate those proportions into a panel action, then
score expected payoff or regret. A risk-neutral expected-points benchmark must
be labeled as such. Zero-expected-value cases do not have a unique correct action;
lower risk taking is not uniformly better.

## 2. SPF: macroeconomic downside risk

[Individual forecasts](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/individual-forecasts),
[decline-probability definition](https://www.philadelphiafed.org/surveys-and-data/recess),
[documentation](https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/spf-documentation.pdf),
[ID caveats](https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/spf-caveats.pdf).

Downloaded `Individual_RECESS.xlsx` and `Individual_RGDP.xlsx`. Both contain
9,280 respondent-quarter rows. A matched subset from 1992Q1 through 2026Q3 has
**4,745 records over 139 survey quarters**, with **23-51 complete forecaster
responses per quarter**, for the following next-quarter comparison:

- `RECESS2`: explicit probability of a decline next quarter, in percent.
- `RGDP2`, `RGDP3`: current- and next-quarter point predictions for real GDP.
- Join keys: `YEAR`, `QUARTER`, `ID`.

Proposed analysis: for the same k forecasters in each survey quarter, compare
the average explicit probability with the share whose point forecasts imply
`RGDP3 < RGDP2`. Score these aggregate forecasts against an observed indicator
of negative GDP growth. The second quantity is a specified crowd aggregation
rule, not a recovered personal probability or a separately asked yes/no answer.

Before scoring, join a fixed vintage of actual GDP, align target quarters and
remove unresolved forecasts. The 139-quarter coverage count is not a count of
resolved cases. Use a consistent realized GDP series for the outcome comparison;
do not compare levels with incompatible historical base years. Starting in 1992
avoids the earlier GNP/GDP definition change. Block uncertainty over time and
account for repeated forecasters and sparse contraction episodes.

`RECESS` means quarter-over-quarter output decline, not an NBER-dated recession.
The Fed also states that some IDs follow firms when staff move, and historical
IDs may have been reused. Describe panels as **forecaster responses**, not a
guaranteed count of distinct people across decades. These question families are
in one survey; they were not randomly assigned as separate questionnaires.

## 3. SCE: consumer inflation risk

[New York Fed data bank](https://www.newyorkfed.org/microeconomics/databank),
[2020-24 microdata](https://www.newyorkfed.org/medialibrary/interactives/sce/sce/downloads/data/frbny-sce-public-microdata-20-24.xlsx),
[questionnaire](https://www.newyorkfed.org/medialibrary/interactives/sce/sce/downloads/data/frbny-sce-survey-core-module-public-questionnaire.pdf).

The downloaded workbook contains person identifiers (`userid`), dates, weights,
inflation/deflation direction (`Q8v2`), a point estimate (`Q8v2part2`), and ten
probability bins (`Q9_bin1` through `Q9_bin10`). It also supplies fitted density
summaries including `Q9_mean`. These are actual consumer responses, making this
a useful alternative when the SPF's firm/individual distinction is unsuitable.

Proposed analysis: compare panel averages of signed point estimates with panel
averages of distribution-derived estimates against realized inflation over the
same next-12-month period. Check the sign convention and fitted-summary method,
retain common complete respondents, and handle repeated people and overlapping
forecast windows. The question says inflation without specifying CPI or PCE, so
the outcome series is an explicit research choice and should receive sensitivity
analysis. An open-ended probability bin cannot be assigned an arbitrary midpoint
without documenting that assumption. Full row completeness and the outcome join
remain to be checked. This is a comparison of responses within one survey, not
randomized assignment to different questionnaires.

## 4. Sandia: wildfire-risk decisions

[Dataset and public codebook](https://www.sandia.gov/research/datasets/details/decision-making-under-uncertainty-human-subjects-data-fire-evacuation-task-2025-02-13/),
[2025 paper](https://doi.org/10.2312/evs.20251077).

The catalog lists participant IDs, item IDs, stay/evacuate responses, risk
levels, text/map conditions, and scenario outcomes. The paper documents
counterbalanced displays and an experimental reward structure. The released
files include `OrangeMaps_LightBackground.csv` and light/dark tiger-map files.

The public download host returned HTML labeled `Request Rejected`, including
on a direct download attempt. A successful HTTP status was not treated as a
successful CSV download. This is a documented candidate, not analysis-ready
microdata in this workspace.

After access is restored, first verify participant counts, case correspondence
across formats, outcome consistency, and the meaning of `CorrAns`. Score panel
actions using the experiment's payoff rule or explicitly defined expected
regret. Treat realized safe/burn outcomes separately from the best action under
uncertainty. These are experimental wildfire scenarios, not actual incidents.

## Other leads and exclusions

- [Good Judgment Project](https://www.iarpa.gov/newsroom/article/iarpa-announces-publication-of-data-from-the-good-judgment-project): public individual geopolitical forecasts are relevant to risk, but a matched question-format contrast was not verified in this search.
- [COVID-19 expert surveys](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010485): the authors release individual forecasts and truth data, with several answer formats. Some format changes occurred across survey dates, so format and forecasting difficulty cannot be cleanly separated without identifying overlapping targets and information dates.
- [TU Delft expert-judgment database](https://www.sciencedirect.com/science/article/pii/S0951832007000944): relevant expert judgments and known calibration answers, but the cited paper directs researchers to the author for the full data. No matched survey-format experiment was verified.
- Ordinary loan-default, insurance-claim and fraud datasets can supply outcomes or case attributes, but without multiple human responses per common case they do not support the advisor's rater-panel comparison on their own.

Cached originals and inspection metadata are under `.cache/risk-survey-data/`.
The cache is ignored by git. The saved
[inspection record](../benchmarks/results/risk-data-search/inspection.json)
retains source links, hashes and measured coverage outside the cache.
