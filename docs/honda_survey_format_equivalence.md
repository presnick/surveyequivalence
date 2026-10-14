# Human-panel equivalence across survey response formats

September 23, 2026. These figures use **real individual responses** from Honda,
Kagawa and Shirasuna's 2022 Experiment 1: 901 participants, nine dot-counting
images, seven response formats. People are the raters; formats define the
surveys; the published image counts are the reference answers.

The point estimates indicate that **4-7 scale respondents match the expected
accuracy of 10 typed-number respondents**, under the scoring rule below.
This is a descriptive result for nine images, not general equivalence across
survey questions or future-event forecasts. Uncertainty is substantial for some
formats.

## Figures

- [Three-page PDF](../output/pdf/honda_survey_format_equivalence.pdf).
- [Rater-equivalence power curves](../benchmarks/results/honda-survey-formats/panel_accuracy.png):
  higher-is-better negative expected squared error for panels of 1-100 people,
  with fractional crossings of the 10-person reference score.
- [Equivalent panel sizes](../benchmarks/results/honda-survey-formats/equivalent_panel_sizes.png):
  smallest whole-person panels matching the 10-person typed reference, with
  respondent-bootstrap intervals.
- [Question comparison](../benchmarks/results/honda-survey-formats/question_comparison.png):
  error changes for each image at the same panel size of 10.

SVG versions sit beside the PNGs for vector export.

The power curves follow the presentation in the [rater-equivalence paper,
Section 5.3](https://arxiv.org/pdf/2106.01254v3): panel size on a linear horizontal
axis, expected score on the vertical axis, and equivalence at the first crossing
of the horizontal target. The displayed score is **negative expected MSE**, in
thousands of squared dots; higher is better and zero is perfect. Lines connect
the measured integer panel sizes without fitting a logarithm. An inset expands
the crossings, which use linear interpolation in expected-score units.

These fractional plot labels differ from the **whole-person thresholds** in the
table and on PDF page 2. The original RMS errors and bootstrap results remain in
their original units. This applies the paper's power-curve idea to known image
counts, rather than evaluating agreement with held-out human labels. No
zero-rater baseline is available, and nothing is extrapolated beyond the data.

## Results

The reference panel has **140.3 dots RMS error**. Each row below reports the
minimum integer panel size whose expected error is no greater than that value.

| Response format | Released people | Matching panel size | 95% respondent-bootstrap interval |
| --- | ---: | ---: | --- |
| Typed number | 128 | 10 | 10, by definition |
| Drag, endpoints labeled | 128 | 7 | 3 to >100 |
| Click, endpoints labeled | 133 | 6 | 3 to >100 |
| Drag, midpoint also labeled | 126 | 4 | 2 to 8 |
| Click, midpoint also labeled | 126 | 5 | 3 to 42 |
| Drag, quartiles also labeled | 130 | 4 | 2 to 8 |
| Click, quartiles also labeled | 130 | 4 | 3 to 9 |

Drag scales begin at zero. Click scales have no preset anchor. All scale ranges
are 0-1,000. Conditions contain separate groups of respondents.

For the two endpoint-only formats, 4.50% and 3.45% of bootstrap draws respectively
do not reach the reference within 100 people. Their upper bounds are therefore
shown as >100. No draw is discarded to produce a finite interval. These results
do not establish that every scale design is reliably more efficient.

The heatmap shows the same limitation across items: several scale formats do
worse on Q1 even though their overall error is lower.

## Scoring and exact calculation

For a panel of k distinct people in one format, average their estimates for each
image. Score that panel by averaging squared prediction errors across all nine
images. Then average over all possible panels of that size and take the square
root. This is **the square root of expected panel MSE**, not the mean of panel
RMSEs, and is reported in dots.

For format g, let n be its released participant count, m_q its mean response,
s_q^2 its sample variance (denominator n-1), and y_q the published count for
image q. The exact expected error is:

`MSE_g(k) = mean_q[(m_q - y_q)^2 + s_q^2 * (1/k - 1/n)]`.

This includes the finite-population correction for uniform sampling without
replacement. The same participant rows supply all nine answers in a panel.
No prediction model is fitted, no synthetic responses are generated, and no
Monte Carlo approximation is needed for the point curves.

The primary metric is squared error, replacing the dataset-search note's
proposed absolute error so that complete panel curves can be calculated exactly.
Each image receives equal weight in the squared-error average. This does not
give equal weight to proportional error: larger absolute mistakes count more.
The tables use the first qualifying integer k in 1-100, with no interpolation
or extrapolation. The power-curve figure additionally shows the first fractional
crossing, interpolating negative MSE (not RMS error) between neighboring sizes.
The reference of 10 typed-number respondents was specified before the analysis.

## Uncertainty and scope

2,000 bootstrap draws, seed 23092026, resample whole participant rows with
replacement within each format. A row's nine responses stay together. Each
draw recomputes both the format curves and the typed-number reference. The
reference curve is reused for its self-comparison, so its equivalence is exactly
10 in every draw.

These are percentile intervals conditional on the nine fixed images. They
reflect respondent-sampling uncertainty under the bootstrap assumptions, not
uncertainty about new questions. The nine images were deliberately selected and
do not constitute a broad random sample of survey problems. The released row
keys preserve respondent records but do not link identities across studies.

The original research has already examined crowd accuracy as group size changes.
This analysis adds an explicit equal-accuracy panel-size comparison; it is not
a claim to a new dataset or an independent replication study.

## Verification and reproduction

The source SHA-256 matches the authors' OSF metadata. All 8,109 response cells
are present, finite and in range. For all seven formats, the formula was checked
against direct one-person scoring, exhaustive enumeration of every two-person
panel, and the full-pool mean. Curves are non-increasing, and the reference
self-equivalence is 10 for both the point estimate and every bootstrap draw.

[Results and checks](../benchmarks/results/honda-survey-formats/results.json),
[curve data](../benchmarks/results/honda-survey-formats/panel_curves.csv),
[equivalence table](../benchmarks/results/honda-survey-formats/equivalent_panels.csv),
and [item data](../benchmarks/results/honda-survey-formats/question_errors.csv)
are saved beside the figures. Bootstrap equivalences and reference errors are
retained in `bootstrap_draws.npz`.

Run `.venv/bin/python research/honda_survey_visualizations.py` to regenerate the
data, PNGs and SVGs from the verified cached workbook. Use `--input PATH` for
another location of the same release. Package the figures with
`python research/honda_survey_visualizations.py --pdf-only` using an interpreter
with NumPy and ReportLab. The local bundled runtime supplied ReportLab for this
run. The original workbook remains unchanged.

Use `.venv/bin/python research/honda_survey_visualizations.py --plot-only` to
redraw just the power-curve PNG and SVG from saved results without rerunning the
analysis. Then use `--pdf-only` to refresh the PDF with that figure.

Sources: [paper](https://www.nature.com/articles/s41598-022-11900-7),
[data and supporting material](https://osf.io/sxdf9/),
[original Experiment 1 workbook](https://osf.io/download/uxrhd/),
[prior file inspection](../benchmarks/results/survey-type-data-search/inspection.json).
