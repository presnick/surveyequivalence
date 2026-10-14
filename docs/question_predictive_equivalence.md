# Question predictive equivalence: provisional visualizations

**Superseded interpretation:** the professor has clarified that people are the
raters and question types define different surveys. See the
[corrected design and Kalshi data finding](survey_type_equivalence.md).
The figures below retain the earlier questions-as-units illustration; they do
not answer the clarified human-panel question.

This draft explores the interpretation that **questions supply information for
predicting an observed outcome**, and that a question's predictive performance
can be expressed in reference-panel units. The professor's intended panel
definition is still pending. Here the unit is a **question**, not a human.

The [four-page visual draft](../output/pdf/question_predictive_equivalence_draft.pdf)
contains:

1. [Prediction setup](../benchmarks/results/question-equivalence-draft/question_prediction_setup.png):
   question signals, a fitted prediction rule, and held-out outcomes.
2. [Real Kalshi example](../benchmarks/results/question-equivalence-draft/kalshi_question_example.png):
   six related temperature contracts, prices before the outcome day, and their
   eventual settlement. This is a descriptive example of the data structure.
3. [Simulated predictive performance](../benchmarks/results/question-equivalence-draft/synthetic_question_performance.png):
   zero, one, two, and three input questions scored on the same cases.
4. [Simulated equivalence curves](../benchmarks/results/question-equivalence-draft/synthetic_question_equivalence.png):
   a different reference pool for each target question, excluding that question.

**All performance and equivalence estimates in this draft are synthetic.** No
Pew data, LLM outputs, or Kalshi-based equivalence estimates are included.

## Real Kalshi example

The event is `KXHIGHNY-26SEP21`: maximum temperature in New York City on September
21, 2026. It has six mutually exclusive temperature-range contracts. The event
was selected as the most recent listed event with a complete set of resolved
contracts, without filtering by its winning range or forecast performance.

The fixed snapshot is **September 21 at 00:00 America/New_York**, or 04:00 UTC.
The figure uses only hourly candles ending at or before this cutoff. Every
snapshot contract has a candle ending exactly at the cutoff. Dots are raw
closing trade prices; short bars are closing YES bid/ask spreads. Prices are
neither normalized nor assumed to be calibrated probabilities. The six closing
trade prices total 1.04, illustrating why the raw vector should not silently
be called a probability distribution.

Missing trade closes appear as gaps in the history chart; no forward filling
is performed. A common candle boundary does not guarantee that each contract's
last trade is equally recent.

The market records an outcome of **72 F**, and the 72-73 F contract resolved
YES. This is the exchange-reported `expiration_value`, not an independently
reconstructed weather observation. These particular contracts identify **The
Weather Company** as the settlement source; current series metadata or older
NWS examples should not replace the specific contracts' rules.

Sources and provenance:

- [Official event and contract data](https://external-api.kalshi.com/trade-api/v2/events/KXHIGHNY-26SEP21).
- [Official market-data quickstart](https://docs.kalshi.com/getting_started/quick_start_market_data).
- [Official candle schema](https://docs.kalshi.com/api-reference/market/get-market-candlesticks).
- [Historical API routing](https://docs.kalshi.com/getting_started/historical_data).
- [Retained example and provenance](../benchmarks/results/question-equivalence-draft/kalshi_example.json),
  including request URLs, retrieval time, raw-file hashes, snapshot, and retained
  pre-cutoff histories. Raw responses are in `.cache/question-equivalence/kalshi/`.
- [Snapshot table](../benchmarks/results/question-equivalence-draft/kalshi_midnight_snapshot.csv).

This example establishes that the required kind of historical data is
accessible. It does **not** establish predictive equivalence from one day, nor
make six related contracts six independent human opinions.

## Precisely what the simulation measures

Each of 4,000 independent synthetic cases has a latent signal and a binary
outcome drawn with probability `sigmoid(1.8 * latent_signal)`. Three questions
measure the latent signal with a shared noise component and different
independent noise levels. For each question, 120 synthetic response draws
produce an aggregate response count. All data are generated with seed 1729.

The first 2,400 cases train the prediction models; the remaining 1,600 evaluate
them. A standardized, smoothed response logit is the input feature for each
question. A separate L2 logistic regression (`C=1`, no tuning) is fitted for
every nonempty subset. A panel's predictor therefore uses its question signals
jointly. This is not an average of the questions' probabilities or the
repository's anonymous Bayesian combiner.

Every predictor is scored using

`score = 1 - mean((predicted_probability - observed_binary_outcome)^2)`.

Higher is better; 1 is perfect. This score is not classification accuracy. Zero
questions predicts the outcome prevalence estimated on training cases only.

For a target question Q1, the reference pool contains Q2 and Q3:

- `k=0`: the training-prevalence predictor.
- `k=1`: the average **score** of the separately fitted Q2 and Q3 predictors.
- `k=2`: the score of the joint Q2+Q3 predictor.

The target's score intersects this curve at its provisional question-panel
equivalence. The existing supported-first-crossing helper is used. Linear
interpolation has the interpretation of randomizing between adjacent panel
sizes and averaging their scores; a fractional value is not a literal fraction
of a question. If none of the measured sizes reaches the target score, the
figure reports no crossing within 0-2, without extrapolating.

Each target has a different reference pool, so these equivalence numbers do not
form a common-unit leaderboard. The same-case performance plot is the direct
comparison of scores. Curves are not forced to increase.

Shading and whiskers are percentile intervals from 500 paired bootstraps of the
held-out cases (seed 1730), **conditional on the fitted models**. They omit
training-fit uncertainty. They are score intervals, not equivalence intervals.
All bootstrap crossing statuses, including unobserved crossings, are retained
in [the results JSON](../benchmarks/results/question-equivalence-draft/synthetic_results.json).

The [original rater-equivalence paper](https://arxiv.org/abs/2106.01254) compares
predictors with combined human judgments. This draft is a supervised
prediction analogue inspired by that comparison. It does not establish the
paper's optimality assumptions, human-rater equivalence, or full information
equivalence for a population of questions.

## Before an empirical Kalshi equivalence analysis

1. Define the repeated outcome case: for example, one city-day. One recorded
   outcome copied across many markets or timestamps is still one outcome case.
2. Choose a shared prediction target: actual maximum temperature, a defined
   threshold event, or a specified outcome category. Contracts refer to different
   events, so their prices cannot simply be averaged as predictions of the same
   target. A mapping or fitted prediction model is needed.
3. Define stable question roles across cases and retain their numeric strikes.
   Listed temperature bins change across dates; 'third listed bin' is not a
   constant temperature threshold.
4. Freeze a forecast cutoff before the outcome observation window, and handle
   missing or stale prices explicitly. Last trade prices at settlement are not
   usable pre-outcome forecasts.
5. Specify the reference population, panel sampling, scoring rule, and combining
   method. Complementary temperature contracts are structurally dependent.
6. Use a time-respecting train/evaluation split; account for serial and same-day
   dependence in uncertainty. The independent-case simulation does not justify
   an independent bootstrap for real weather or market observations.

## Reproduction and verification

```sh
.venv/bin/python research/question_equivalence.py
```

This command is offline. It regenerates the synthetic data, metrics, figures,
and PDF. It reads the Kalshi cache when present, or the retained
`kalshi_example.json` otherwise; it does not fetch data, run LLM inference, or
access an account. The library and existing experiments are unchanged.

The runner checks probability bounds, optimizer convergence, the Brier score
convention, independent arithmetic for every reference curve, and reconstruction
of each observed crossing. For Kalshi it checks a unique settled winner, a
consistent recorded outcome, complete cutoff-aligned snapshots, and valid
bid/ask prices. PDF pages are rendered separately for visual inspection.
The 25 existing research tests also passed; the core library suite was not rerun
for this isolated visualization workflow.
