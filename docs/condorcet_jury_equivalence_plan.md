# Rater equivalence and Condorcet jury theory

Research and graphics plan · October 6, 2026

**Executed:** the [results and reproduction guide](condorcet_jury_equivalence.md)
and [ten-page report](../output/pdf/condorcet_jury_equivalence.pdf) now contain
the four-dataset analysis, all six planned figures, and a support/species
appendix. The text below is the original design record.

**Recommended project:** measure how many reference jurors are needed to match
the accuracy of a real panel, and investigate why increasing panel size yields
less—or more—benefit than a simple Condorcet model predicts. Start with the NBA
data already analyzed here, then use Bluebirds for a clean import pilot and
city-population comparisons for larger juries. Ducks provides a useful
species-stratified replication.

The deliverable from this search is this plan and one
[NBA preview graphic](../benchmarks/results/condorcet-plan-preview/condorcet_nba_preview.png)
([SVG](../benchmarks/results/condorcet-plan-preview/condorcet_nba_preview.svg)).
The new dataset analyses and uncertainty calculations below are proposed work.

## 1. The connection

Condorcet's classical model asks how majority accuracy changes as independent,
equally competent voters are added. Rater equivalence reverses that question:
**how many voters from a specified reference population match a given score?**
This makes it possible to compare a single expert, a different aggregation
rule, or another panel in units of reference jurors.

There are two useful, distinct comparisons:

| Comparison | Reference curve | Question answered |
| --- | --- | --- |
| Empirical jury equivalence | Measured accuracy of random panels of real raters | How many ordinary raters match this expert, method, or panel? |
| Independent-jury model equivalence | Condorcet majority accuracy at a specified competence | How many hypothetical independent jurors match the observed accuracy? |

The original [rater-equivalence paper](https://arxiv.org/abs/2106.01254v3)
addresses human judgments that may lack definitive truth. Here the primary
score is accuracy against independently established answers or realized
outcomes. This is an objective-outcome adaptation, like the existing NBA
workflow. Agreement with another human is a different target. Preference and
safety-label datasets, including existing AI-judge examples, do not establish
the Condorcet correctness target just by having majority labels.

The research concerns the jury theorem, rather than Condorcet cycles among
three or more ranked alternatives. Literal courtroom verdict data are also
a difficult starting point: verdicts alone usually lack individual independent
votes and an external correctness label.

## 2. Datasets to use

“Inspected” below means individual released records were parsed, not merely
that a paper reports a dataset. The NBA counts come from the repository's
existing inspection; Bluebirds, Ducks, and city comparisons were inspected
during this search. [Source hashes and inspection record](../benchmarks/results/condorcet-dataset-search/inspection.json).

| Priority and dataset | Available observations | Proposed application | Readiness and limitation |
| --- | --- | --- | --- |
| **1. NBA winner forecasts**, Rutchick et al. | 465 games; 113 active people; 9,788 complete pick/confidence/peer-agreement records; 10–61 respondents per game | Immediate observed-versus-Condorcet graphic; compare majority, confidence weighting, and surprisingly popular aggregation | Already prepared and scored here. Common all-game support is k=1–10. Respondent membership changes across games; only 11 waves. [Study and release links](https://link.springer.com/article/10.1186/s41235-020-00256-z), [local inspection](../benchmarks/results/sports-data-search/inspection.json). |
| **2. Bluebirds**, Welinder et al./CUBAM | **108 images × 39 raters = 4,212 binary labels**; complete matrix; released truth labels | Cleanest new pilot for full panel-growth curves, individual skill, and shared mistakes | Original `labels.yaml` and `gt.yaml` inspected. Task distinguishes Indigo Bunting from Blue Grosbeak. Only 108 items; source labels still require a provenance check. [Author release](https://github.com/welinder/cubam/tree/public/demo/bluebirds). |
| **3. City population comparisons**, Yu et al., reanalyzed by Kurvers et al. | **1,000 city pairs, 99 retained people, 98,787 usable votes; 97–99 votes per pair** | Largest verified binary jury study in this shortlist; expertise, confidence, and headcount | Original `S3_CityTask.csv` inspected. Exclude 213 missing decisions encoded as accuracy zero. City reuse creates dependence across questions. [Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6957221/), [original file](https://osf.io/vyx8a). |
| **4. Ducks**, Welinder et al./CUBAM | **240 images, 53 workers, 9,600 labels; 40 ratings on every image** | Replication with species-specific error patterns and varying worker coverage | Original `labels.txt` and `classes.yaml` inspected. Binary duck/not-duck responses; truth can be constructed from the five released classes. Keep species in the analysis. [Author release](https://github.com/welinder/cubam/tree/public/demo/ducks). |
| **5. Skin-lesion decisions**, Argenziano/Kurvers et al. | Paper reports **40 dermatologists × 108 cases = 4,320 decisions**, with confidence; histopathology reference | Expert-panel equivalence and confidence-based aggregation | Public supplementary dataset advertised; raw file not yet audited. Report sensitivity and specificity as well as aggregate accuracy; 27/108 cases are melanomas. [Primary study and supplement](https://pmc.ncbi.nlm.nih.gov/articles/PMC4978286/). |

For Ducks, the released species mapping identifies Mallard and American Black
Duck as ducks; Canada Goose, Non Bird, and Red-necked Grebe are the negative
class. Preserve original IDs and verify this coding against the author's
loader before scoring. These are **two different CUBAM datasets**: do not use
the 108-image Bluebirds count for the 240-image Ducks release.

CUBAM's repository has a BSD software license; a separate data-specific reuse
statement was not located in this search. Record exact source revisions and
terms with the imports; do not infer image redistribution rights from the
software license. Labels alone suffice for the first human-panel analyses.
The associated [Welinder et al. paper](https://proceedings.neurips.cc/paper_files/paper/2010/file/0f9cafd014db7a619ddb4276af0d692c-Paper.pdf)
provides the methodological background for the two bird tasks.

For the city task, the raw release has 109,500 rows and 120 participant IDs;
99 have 1,000 trials each, matching the reanalysis cohort. Of their 99,000
trials, 213 have no recorded decision despite `DecisionAcc=0`. Excluding those
leaves 15 items with 97 votes, 183 with 98, and 802 with 99. Use all 1,000
items through k=97; use a separately declared 802-item subset if studying
k=98–99. Canonicalize answers to city identity because screen position varies,
and derive truth from the released historical population fields. The paper's
raw recruitment count and release differ; report the actual cohort rule.
The OSF node has no explicit license in its metadata.

A useful reserve is Recognizing Textual Entailment: the crowdsourcing catalog
reports 800 items with ten labels each and worker IDs. Its original release
route needs repair or a provenance-checked mirror, and entailment gold labels
are less objective than realized sports outcomes. [Dataset catalog](https://github.com/TrentoCrowdAI/crowdsourced-datasets).

The existing CIFAR-10H cache offers many repeated ratings but has ten classes.
Use it for a later multiclass extension or a substantively justified binary
task. Treating every wrong class as one vote for a single alternative would
misrepresent the original plurality decision. Existing numerical social-
influence datasets are useful extensions only after defining a meaningful
binary question and threshold; their continuous-error curves are not direct
tests of classical binary majority voting.

## 3. A concrete first result to visualize

The saved NBA winner-only curve already gives:

| Quantity | Accuracy |
| --- | ---: |
| One randomly selected eligible forecaster | 63.09% |
| Nine- or ten-person observed majority | 65.89% |
| Three independent model jurors, each 63.09% accurate | 69.19% |
| Nine or ten independent model jurors, each 63.09% accurate | 79.45% |

Thus **three whole model jurors are the smallest panel reaching the observed
ten-person accuracy** under this particular homogeneous independence model.
The preview's fractional crossing is 2.46 using interpolation over every
integer size, including the 1–2 plateau. Interpolating only odd sizes would
give a different fractional answer; use whole counts as the main result and
always state the interpolation convention.

This is a descriptive projection. The 13.56 percentage-point gap at k=10
does not identify a causal effect of social dependence. Shared game difficulty,
heterogeneous competence, respondent selection, and finite-pool sampling also
differ from the simple model. The preview adds no new uncertainty interval.
[Exact values and source hashes](../benchmarks/results/condorcet-plan-preview/preview_numerics.json).

## 4. Graphics storyboard

Build the first four figures as the initial report. Add the last two after the
new datasets pass inspection and the evaluation design is fixed.

| Figure | Layout and axes | Data and comparison | Intended takeaway |
| --- | --- | --- | --- |
| **1. What Condorcet predicts** | Line chart: panel size k=1,3,…,101; probability majority is correct | Analytical curves for individual competence p=.45, .50, .55, .65, .75; mark an illustrative target | More voters help under the stated above-chance independence assumptions. Label the entire plot “theoretical.” |
| **2. Real jury growth versus the simple model** | Small multiples for NBA, Bluebirds, city comparisons, and Ducks; x=headcount, y=accuracy; lower strip=model minus observed, in percentage points | Exact empirical majority curves versus a homogeneous binomial curve matched to each dataset's single-rater accuracy | Measure departures from the model without assigning them to one cause. The NBA preview illustrates the layout. |
| **3. How many jurors is this worth?** | Empirical power curves with horizontal target and vertical first-crossing projections; companion count table | Initially NBA only: compare the three existing aggregators against a ten-person majority. After Figure 5, extend to trained methods on the new datasets | Translate accuracy into a clearly specified reference panel size. Show whole counts, interpolated values, and unresolved bounds separately. |
| **4. Which questions defeat the crowd?** | Correct-vote-share histogram with 50% line, beside per-item panel gain from k=1 to k=9 | Use the same items as the main curves; optionally facet Ducks by species | Show how gains on some items coexist with confidently wrong majorities on others. Label outcome-derived groups “retrospective.” |
| **5. Expertise versus headcount** | Held-out accuracy curves for random panels, training-selected raters, and regularized weighted voting | Start on the complete Bluebirds matrix; repeat on city comparisons with city-aware splits and explicit missing-response handling | Test whether selection or weighting substitutes for additional raters. Fit selection and weights on separate questions. |
| **6. Why a crowd can plateau** | Theoretical panel curves and heatmap of accuracy-equivalent independent size as shared-case variation increases | Explicit beta-binomial model with matched mean competence; no invented empirical observations | Explain one mechanism producing dependent errors. Label all values “model illustration,” separate from measured results. |

Use a linear headcount axis, visible integer markers, and consistent colors for
empirical majority, alternative aggregation, and the theoretical reference.
Keep plateaus and decreases. Export PNG for quick review and SVG/PDF for reuse;
each plotted series should have a corresponding CSV and a caption stating its
target, pool, score, tie rule, and uncertainty unit. Do not rank datasets by
equivalent count: each has a different task and reference population.

## 5. Analysis protocol

**Freeze the target and panel design.** Use one row per
`dataset, item_id, rater_id, vote, truth`, with optional confidence, species,
wave/session, and exposure/condition columns. Check unique item/rater pairs,
label coding, truth coverage, missingness, and per-item depth. Keep items and
eligible rater pools fixed across k and comparable methods. Do not fill absent
votes with errors. Sampling k available raters separately per item estimates
an average panel policy; it is not necessarily one fixed jury across all items.

**Compute exact baseline curves.** For item i with N_i recorded responses and
C_i correct responses, the number of correct votes in a uniformly chosen
k-person subset has a hypergeometric distribution. Compute its majority tail
and give ties half credit, then average equally over items. This calculation
does not assume the original responses were independent. Start at k=1–10 for
NBA, 1–39 for Bluebirds, 1–97 for city comparisons, and 1–40 for Ducks, subject
to the import audit. First render the city curve through k=31 for readability,
then show the full supported range in an appendix.
The deepest sizes describe these finite observed pools, not unlimited new
recruitment.

**Add reference models in stages.** First use the homogeneous binomial
benchmark, with p equal to the same item-weighted single-rater accuracy.
Then inspect performance by truth class and task subgroup. If class accuracy
differs, compare against separate class-conditional binomial curves and combine
them with the same evaluation weights. For independent voters with unequal
competence, use the Poisson-binomial tail for the specified panel rather than
substituting mean competence into a binomial formula.

**Keep explanations separate from descriptions.** Average competence above
one half does not establish competence on every kind of item:
`mean[B_k(p_item)]` generally differs from `B_k(mean[p_item])`. As an analytical
counterexample, equal numbers of items with p=.4 and p=.9 give individual
accuracy .65, yet their average majority accuracy tends to .5 as k grows.
See [Dietrich and Spiekermann on defensible jury-theorem premises](https://eprints.lse.ac.uk/46542/1/Dietrich_Spiekermann_Epistemic-democracy-with-defensible-premises_2013.pdf).

Use residual shared-error measures only as model-dependent diagnostics.
Pairwise correlation alone does not specify the whole majority distribution.
If estimating item difficulty prospectively from pilot ratings, separate pilot
raters from evaluation raters and count that annotation cost. Do not erase
item difficulty by shuffling each worker's errors across questions and then
call the resulting gain a pure independence effect.

**Separate fitting from evaluation.** Learn rater skill, weights, and panel
selection on training questions, then score all candidate methods on the same
held-out questions. Preserve task families and repeated city identities in
splits where feasible; use past waves for forecasting claims. Report missing
coverage after selection. Confidence categories and predicted peer agreement
are not automatically calibrated probabilities of correctness.
For learned policies, distinguish intervals conditional on fixed trained
weights/selection from intervals that resample and refit the entire training
procedure; the latter includes training variability.

**Quantify uncertainty in the full comparison.** For NBA, reuse paired
whole-wave resampling and recompute p, model curve, target, and crossing in
each draw. Its eleven waves support exploratory inference conditional on the
observed people. For image tasks, resample independent images or original
image clusters, retaining every vote for each sampled item; stratify by truth
class for class-balanced targets. These intervals remain conditional on the
observed workers. For city comparisons, start with an explicitly fixed-item
descriptive result; any generalization across questions must account for
their shared cities, rather than treating 1,000 pairs as independent facts.
Claims about new recruits need a separate design or
respondent-aware analysis. Retain bootstrap draws that never cross.

**Report identified limits.** Use the first crossing without imposing
monotonicity. Report `≤1` when one reference rater already reaches the target
and `>K` when none of the supported sizes does. The latter is a bound within
the study, not proof that every larger panel fails. A first crossing also does
not imply every larger panel remains above the target. These are performance
matching estimates, not formal statistical equivalence tests.

## 6. Implementation sequence and completion criteria

1. **Import and provenance.** Add a proposed `research/condorcet_data.py`
   adapter for Bluebirds, city comparisons, and Ducks. Cache original files; pin versions and
   hashes; save the schema/coverage audit and label mapping. The gate is
   verified external truth plus enough distinct raters per common item.
2. **Exact accuracy engine.** Add a proposed `research/condorcet_equivalence.py`
   runner, reusing `research/panel_power.py` for first crossings and the NBA
   workflow's conventions. Check the hypergeometric result against exhaustive
   tiny-panel enumeration and single-rater/full-pool endpoints. Check the
   binomial implementation against an independent probability routine.
3. **Initial report.** Produce Figures 1–4, curve tables, paired bootstrap
   draws, and equivalence bounds under
   `benchmarks/results/condorcet-jury-equivalence/`. Include the support limits
   and both class-specific and overall scores where relevant.
4. **Extensions.** Only after the first report, add held-out selection and
   weighting, the explicit shared-case model illustration, and one further
   domain if its raw release is accessible. [Weighted-voting theory](https://arxiv.org/abs/1312.0451)
   motivates this extension; empirical performance still needs held-out tests.
5. **Graphics review.** Inspect every rendered figure at normal viewing size,
   check labels and crossings against saved numbers, then assemble a short
   PDF report. Every figure must clearly distinguish recorded responses from
   analytical model outputs.

No new human collection or AI inference is needed for the first report.
An AI-versus-human extension needs predictions on these exact items; an
unmatched published benchmark score cannot supply an equivalent-jury estimate.

The completed preview can be regenerated now:

```sh
.venv/bin/python -B research/condorcet_preview.py
```

## Technical definitions for the runner

For a majority with independent correctness indicators of common probability
p, including half credit for ties:

```text
B_k(p) = P[Binomial(k,p) > k/2] + 0.5 P[Binomial(k,p) = k/2]
A(k)   = mean_i {P[Hypergeom(N_i,C_i,k) > k/2]
                 + 0.5 P[Hypergeom(N_i,C_i,k) = k/2]}
e_R(q) = smallest supported integer k with reference_accuracy(k) >= q
```

The equality term is zero at odd k. For the classical improvement result,
use p>1/2; at p=1/2 the model stays at chance. State any class-conditional
competence assumptions explicitly. A majority-class rule can have high
overall accuracy while failing completely on the minority truth class.

For Figure 6, an explicit model is:

```text
theta_i ~ Beta(alpha,beta)
correct_ij | theta_i ~ independently Bernoulli(theta_i)
alpha = p*(1/rho - 1); beta = (1-p)*(1/rho - 1), for 0<rho<1
```

This yields beta-binomial majority tails and marginal correctness correlation
rho. Handle rho=0 with the binomial limit. It models shared case conditions;
it is not evidence of communication among respondents. Correlated-jury theory
requires specifying such a joint model; see
[Ladha's correlated-vote analysis](https://doi.org/10.1016/0167-2681(94)00068-P).

The familiar expression `k/[1+(k-1)*rho]` matches the variance of an average
under exchangeable, equal-variance responses. It is **not** the exact
accuracy-equivalent jury size. Use the actual majority tail and invert the
chosen reference curve for the rater-equivalence analysis.
