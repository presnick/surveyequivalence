# Sports forecasting as a survey-equivalence experiment

Checked September 24, 2026. **An open NBA dataset supports a real individual-rater
comparison now.** It records game forecasts rather than betting transactions.
Betting-return analysis additionally requires historical odds and verified game
and forecast-time alignment.

## Verified data

The selected dataset is Study 3 of Rutchick et al. (2020), which recruited from
basketball and sports-betting discussion forums and a sport psychology course.
Participants supplied winner choices, confidence ratings and estimates of other
participants' agreement. The published study covers the 2018-19 NBA season.
[Paper](https://link.springer.com/article/10.1186/s41235-020-00256-z),
[authors' release](https://osf.io/2rzcs/).

Direct inspection of the downloaded workbook found:

| Property | Measured from the release |
| --- | ---: |
| Games with an outcome code | 465 |
| Participant records | 130 |
| Participants with at least one valid prediction | 113 |
| Valid participant-game winner picks | 9,815 |
| Complete pick/confidence/peer-agreement records | 9,788 |
| Complete respondents per game | 10-61 |

Each participant has a stable numeric ID. Each game has three columns with
suffixes `_PRED`, `_CONF`, and `_OTHERS`, plus a value in the final `winner` row.
Pick and winner codes are 1 or 2; confidence is ordinal 1-5; agreement is a
percentage. The loader excludes one invalid pick coded zero and incomplete
triplets. Missing forecasts remain missing, not incorrect forecasts.

The final paper reports 111 active participants, whereas this release contains
113. Its separately released appendix also differs slightly from the final
paper's aggregate results. Any analysis should be described as a reanalysis of
the identified release, not an exact reproduction of the paper. The source
hash matches OSF metadata.

Original files are in `.cache/sports-survey-data/`. Prepared individual records
are in `.cache/sports-survey-data/nba_complete_responses.jsonl`.
[Inspection and checks](../benchmarks/results/sports-data-search/inspection.json)
and [coverage by game](../benchmarks/results/sports-data-search/nba_game_coverage.csv)
are retained outside the cache.

## Calculated results

Exact finite-pool panel calculations use all **465 games, 113 people and 9,788
complete responses**. The reference is the expected **65.8946% winner accuracy
of 10 winner-only respondents**. A panel is a uniformly chosen subset of the
eligible respondents for that game; the results average over all such subsets.

| Response information / aggregation | Smallest panel matching reference | Exploratory 95% bootstrap interval | Accuracy with 10 people |
| --- | ---: | --- | ---: |
| Winner choice / majority vote | 9 | 5-9 | 65.8946% |
| Winner + confidence / weighted vote | 10 | 3 to >10 | 66.0343% |
| Winner + peer agreement / surprisingly popular | 9 | 7 to >10 | 66.3269% |

**These estimates do not establish a respondent saving from the richer
questions.** With half-credit ties, majority voting has identical expected
accuracy at adjacent odd/even panel sizes: 9 and 10 people already match.
Peer agreement reaches the reference at 9, but so does winner-only voting.
At 10 people, confidence adds 0.140 percentage points and peer agreement adds
0.432 percentage points; the paired bootstrap intervals include zero.

The intervals use 10,000 resamples of the **11 entire survey waves**, keeping
all games within each sampled wave and recomputing the reference each time.
They are exploratory and conditional on the observed respondent pools; they
do not capture recruitment of new people or generalization across seasons.
`>10` means no crossing within the supported range, not a measured requirement
of 11 people or evidence that a larger panel must eventually cross. Such
resamples were retained: 26.13% for confidence and 21.16% for peer agreement.
The interval endpoints are inverse empirical-CDF quantiles, not interpolated
fractional people. They are not formal equivalence-test confidence bounds.

[Accuracy and difference figure](../benchmarks/results/sports-survey-equivalence/panel_accuracy.png),
[equivalence table](../benchmarks/results/sports-survey-equivalence/equivalent_panels.csv),
[curves](../benchmarks/results/sports-survey-equivalence/panel_curves.csv),
[provenance and full results](../benchmarks/results/sports-survey-equivalence/results.json).

The figure uses the paper's power-curve presentation: benchmark raters on a
linear horizontal axis and higher-is-better accuracy on the vertical axis.
Its horizontal reference and vertical projections identify the first crossing
after linear interpolation, as in Definitions 15 and 17 of the
[rater-equivalence paper](https://arxiv.org/pdf/2106.01254v3).
The plotted equivalences are **9.00 winner-only, 9.22 confidence-weighted and
8.62 peer-agreement raters**. These fractional crossings differ from the
smallest whole-person thresholds in the table and CSV, which remain 9, 10 and
9. The plot preserves odd/even plateaus and observed declines; no logarithmic
fit or monotonic smoothing is imposed.

This is an objective-outcome adaptation of the power-curve comparison: the
evaluation targets are actual game winners. It does not reproduce the paper's
subjective-label protocol of evaluating predictions against a held-out human.

## Comparison

People are raters, games are cases, and the winner row supplies the outcome.
Use three questionnaire variants based on the available responses:

| Variant | Information from each person | Panel prediction |
| --- | --- | --- |
| A | Winner choice | Majority vote |
| B | Winner choice and confidence | Confidence-weighted vote |
| C | Winner choice and expected agreement from others | Surprisingly popular choice |

For B, weight each choice by its recorded 1-5 confidence, explicitly treating
this as a weighting convention rather than a win probability. For C, orient
every answer to team 1: if a person picks team 1, expected agreement a contributes
a to the predicted fraction selecting team 1; if they pick team 2 it contributes
1-a. Choose team 1 when its observed vote share exceeds its predicted share,
and team 2 when the difference is negative. Give exact ties half credit.
These aggregation rules are described in
[Lee, Danileiko and Vi (2018)](https://doi.org/10.1017/S1930297500009207).

The peer-agreement percentage is a forecast of other people's choices, not a
probability that the team wins. It must not be used directly as a win probability
in a Brier-score comparison. Similarly, confidence category 4 is not automatically
an 80% probability.

All three answers were collected in the same questionnaire. This compares the
value of using different response subsets and aggregation rules, not a
randomized intervention assigning people to different questionnaires. Variant C
requires two answers per person; panel size still counts people, not answers.

## Panel-equivalence protocol

1. Use all 465 games, retaining the same complete-answer respondent pool per game
   for A, B and C. This conditions the result on complete respondents, but keeps
   missingness from changing the comparison between methods.
2. Compare k = 1 through 10, supported on every game. Average over all subsets
   of k distinct people without replacement for each method.
   Membership may vary between games; this measures k available forecasters per
   game rather than one fixed panel that answered the entire season.
3. Score actual winner accuracy, averaged equally over games and possible panels.
   All three curves are calculated by integer subset-sum dynamic programming,
   with no panel Monte Carlo error. Independently verify the baseline on every
   game using hypergeometric probabilities. Treat Excel's 56.9999999999999%
   representation as 57%; all recorded percentages are integer to 1e-9 tolerance.
4. Use the expected accuracy of 10-person variant A as the reference. Report the
   smallest supported k in B or C reaching it, or `>10` when no crossing exists.
   Accuracy need not improve monotonically, so retain the observed curve and
   any reversals rather than imposing a rising curve.
5. Display uncertainty across game blocks conditional on the observed respondent
   pools, with the reference recomputed in each resample. Any claim about new
   people requires additional respondent-level resampling preserving their
   repeated answers and missingness. There are only a limited number of survey
   waves; uncertainty should not be presented as universal across seasons.

This answers: **How many people answering the richer sports questionnaire match
the winner-prediction accuracy of 10 people answering only the winner question?**
The results above implement this protocol, retaining reversals in the curves.

## Extending to betting returns

The companion `nba_538.xlsx` contains 465 dates and team pairs with model
probabilities. Those probabilities are not bookmaker odds. It also lacks an
explicit key linking its rows to the survey's wave/game IDs. Equal row counts
are not sufficient evidence for a join, so no order-based join was assumed.

A cash-return comparison requires verified mapping to the actual matches,
recorded odds available at an appropriate forecast cutoff, and a fixed staking
rule. Score the resulting bets after bookmaker margins and pushes. Winner
accuracy alone does not establish profitability, and a common price such as
-110 must not be assigned to all moneyline picks.

For an against-the-spread experiment, collect direct cover/no-cover answers and
predicted winning margins for the same games and lines. Margin forecasts can
then be compared with the quoted spread to form cover predictions. This is
particularly close to the professor's idea of different substantive questions.
[Simmons et al. (2011)](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Simmons-et-al.-2011.pdf)
studied this elicitation contrast, but a usable raw release for that study was
not located in the bounded search. No author contact was made.

## Secondary NFL candidate

The related [NFL release](https://osf.io/y7z6s/) contains 216 games and 227
participant rows, with the same three response families and a winner row.
It is cached under `.cache/survey-type-data/nonpolitical/` from the prior search.
Confidence values include 6 and 7 despite the paper describing a 1-5 scale,
so that coding requires reconciliation before a confidence-weighted comparison.
NBA is the cleaner first implementation from the checked files.

Run `.venv/bin/python research/sports_survey_data.py` to regenerate the NBA
inspection, game coverage and prepared records from the original workbook.
Then run `.venv/bin/python research/sports_survey_equivalence.py` to regenerate
the exact curves, equivalence table, bootstrap draws and PNG/SVG figures.
Use `.venv/bin/python research/sports_survey_equivalence.py --plot-only` to
rebuild just the figures from saved per-game curves and bootstrap wave draws.
This mode verifies the reconstructed mean curves and percentile intervals
against `panel_curves.csv`, without recalculating panels or rewriting results.
