# Corrected design: people are raters, question types define surveys

The professor clarified: "You'd need to get individual responses of people"
and "Then the question types would be different surveys."

This supersedes the earlier provisional interpretation in which questions
themselves were treated as rater units. The existing synthetic question-panel
figures are illustrations of that earlier interpretation, not results for the
professor's intended comparison.

## Intended comparison

For each outcome case, collect individual responses under different question
types. Each question type defines a survey. Combine k people's responses within
one survey, predict the outcome, and evaluate that prediction across cases.
Construct a performance-versus-people curve for each survey type.

The comparison is: how many people answering survey A are needed to match a
specified panel of people answering survey B? The panel size counts people;
the question type determines which curve is used. A common outcome target,
scoring rule, aggregation/prediction method, and respondent-population design
still need to be specified.

The data should retain case ID, question/survey type, each person's response,
response time, and the eventual outcome. Anonymous respondent IDs are useful
for identifying distinct people, repeated responses, and dependence. Multiple
contracts or trades must not be assumed to represent distinct respondents.

## Kalshi feasibility finding

Kalshi public market quotes and candlesticks are aggregate market observations.
The documented public trade schema includes a trade ID, contract ticker,
quantity, prices, transaction sides, and timestamp. It does not provide
respondent identifiers or elicited individual survey responses. A transaction
is not equivalent to an individual's stated forecast, and its trade ID is not
a person ID.

Sources: [public trades](https://docs.kalshi.com/api-reference/market/get-trades),
[historical trades](https://docs.kalshi.com/api-reference/historical/get-historical-trades).

Thus the public data checked here cannot directly support the intended
human-panel comparison. Individual forecasting/survey response data would be
needed, whether obtained through a suitable research release or new collection.

## Status of local work

The follow-up [dataset search](survey_type_dataset_search.md) found an open
individual-response pilot and closer election-survey candidates requiring
registration. The Honda response files have been downloaded and checked.
The subsequent [Honda analysis](honda_survey_format_equivalence.md) now contains
real human-panel curves, equivalent panel sizes and question-level comparisons.

- The public-data access probe recovered 1,224 contract candle responses for
  204 NYC daily events, March 1 through September 20, 2026. These are real market
  observations, but they are not individual survey responses.
- Raw data and provenance are retained under `.cache/kalshi-question-pilot/`.
- The unexecuted price-based analysis and protocol were archived under
  `.cache/kalshi-question-pilot/superseded-price-design/` after the clarification.
- No empirical price-based or human-rater equivalence was calculated from
  that multi-event Kalshi collection.
- The earlier draft's equivalence numbers remain synthetic and use the
  superseded questions-as-units interpretation.
