# AlpacaEval AI-judge equivalence: prepared, inference pending

The experiment now has **648 blinded, exact response-pair requests**, four distinct
human votes per pair, frozen prompt-cluster bootstrap samples, and a tested
response importer, scorer, and figure generator. **No Astra judgments have been
generated or scored.** This environment has no configured inference API key.

The implementation is [research/alpaca.py](../research/alpaca.py).
The preparation and verification record is
[readiness.json](../benchmarks/results/alpaca-judge-validation/readiness.json).
The existing [human curves](../benchmarks/results/alpaca-human-baseline/alpaca_human_curves.png)
remain the real-data result to share until model inference is complete.

## Frozen comparison

- Data: AlpacaEval human crossannotations, revision
  `2edc6fad8be6b14ea7230aabfd08188da6b8b814`; 648 response pairs in 605 prompt
  clusters. Dataset terms are separate from this repository's MIT license
  (the dataset card declares CC-BY-NC 4.0).
- Four distinct humans selected and ordered per item with seed 1729; response
  A/B orientation is randomized and shared by human labels and AI inputs.
  A slot is not a globally identical person across items.
- Initial requested judge: `gpt-6-astra`, medium reasoning, one independent
  request per pair, maximum 2,048 output tokens per request including reasoning.
  No tools or previous conversation, and no human labels in the request.
- Elicit the **probability a human prefers A**, rather than confidence in an
  objectively correct answer. Derive the hard choice by argmax, choosing A at
  exactly 0.5. No post hoc calibration or probability clipping.
- Report **log-score equivalence** as the distributional comparison, with
  **agreement equivalence** as a separate comparison of hard decisions.
  These are score-dependent quantities, not interchangeable units or Elo ratings.
- Human benchmarks use the anonymous Bayesian combiner, same-dataset
  leave-one-out training, and a disjoint single human reference, for k=0,1,2,3.
  The AI forecast is scored against a uniformly selected human from the same
  four-person panel. AI forecasts do not consume any of those four labels.
- Pair each judge score with its corresponding human curve in all 500 frozen
  prompt-cluster bootstrap samples. Preserve repeated items and sample order.
  Intervals condition on the selected humans, fitted combiner and one set of AI
  forecasts; they do not measure model-generation or rater-population variability.

Above-range results are `>3`, not a point estimate of three or an extrapolated
rating. Retain censored bootstrap samples. A zero-probability prediction for an
observed outcome yields negative infinity in raw log score and undefined scalar
equivalence; report that failure rather than silently smoothing it.

The public benchmark may have appeared in model training, and a single response
orientation cannot establish freedom from position bias. These are initial
benchmark results, not a contamination-free or universal ranking. A later
prespecified order-reversal experiment can test position sensitivity. Comparing
multiple judges requires the same frozen requests, human cohort, and protocol;
overlapping intervals alone do not provide a formal ranking test.

## Run the prepared experiment

The already prepared local artifacts are:

```text
.cache/alpaca-astra/requests.jsonl
.cache/alpaca-astra/protocol.json
.cache/alpaca-astra/preparation.json
```

The request file follows the documented
[OpenAI Batch API](https://developers.openai.com/api/docs/guides/batch) format,
targets `/v1/responses`, and uses
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Model capabilities were checked against the
[GPT-6 Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra).
Live submission and model access have **not** been tested here.

With provider access and a spending limit configured, upload the JSONL file with
purpose `batch`, create a batch targeting `/v1/responses` with a 24-hour completion
window, and download its output file. Preserve the submitted input, batch ID,
error file, and raw output for provenance. These steps incur API charges; none
have been performed by this implementation. The prepared maximum is 1,327,104
output tokens across 648 requests, plus input tokens; that is a token ceiling,
not an expected bill. A reasoning-token limit can produce incomplete responses.

Then run:

```sh
.venv/bin/python research/alpaca.py score \
  --prepared .cache/alpaca-astra \
  --responses /path/to/downloaded-batch-output.jsonl
```

This writes `benchmarks/results/alpaca-astra/` with prediction and bootstrap-score
CSVs, score/equivalence summaries, API response provenance, and PNG/PDF/SVG
comparison graphs. The output directory must not already exist. It refuses
missing, duplicate, refused, incomplete, incorrectly identified, or mismatched
responses, and mixed resolved model versions. Batch output ordering is irrelevant:
the importer aligns by exact item ID and verifies echoed request metadata.
Resolve failures explicitly; do not drop items or replace outputs with another
model. Before using a different retry protocol, declare that change and its scope.

To prepare another model or a revised protocol, use a fresh directory:

```sh
.venv/bin/python research/alpaca.py prepare \
  --model gpt-6-astra --reasoning-effort medium \
  --max-output-tokens 2048 --output .cache/alpaca-astra-new
```

The preparation command needs the original blinded input cache. It refuses to
overwrite an existing request file. Full prompts and raw response text stay in
the ignored cache; protocol fingerprints are recorded in the results.

## Validation completed

Before repository cleanup, all **140 repository tests passed**, including seven new tests covering blinding,
request identity, response alignment, missing/failed data, invalid probabilities,
refusals, model-version consistency, repeated bootstrap items, explicit zero
probabilities, and censored equivalence. Independent hand-calculated expected
scores match the pipeline exactly in the numerical fixtures. This validates this
adapter's tested path; it does not close unrelated findings in the mathematical
audit.

After cleanup, the same seven Alpaca checks live in `research/tests/alpaca_tests.py`.
The reorganized core suite (109 tests) and research suite (25 tests) both pass;
six checks specific to retired exploratory tools were removed. The prepared
batch and frozen human baseline remain unchanged. See [repository layout and
retired tools](../research/README.md).

An explicitly named **offline constant control** (p(A)=0.5 on every pair) also
passed the complete importer, scorer, bootstrap and figure workflow on all 648
real items. Both human curves replayed **exactly across all 501 rows and four k
values**—4,008 checked cells. The exported figure was inspected visually.
The control is not an AI evaluation and its graph is kept in the ignored cache.

Its agreement is 0.5015432098765432 (ties deterministically choose A), and log2
score is exactly -1. The resulting small positive log-score equivalence
(0.023567133921770567) occurs because a fixed 50/50 forecast slightly beats the
estimated leave-one-out zero-human prior (-1.0009453119319205). It is a useful
reminder that equivalence is relative to the selected fitted human baseline,
not an intrinsic quantity of information or proof that a constant predictor
contains item-specific information.
