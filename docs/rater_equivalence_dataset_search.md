# Suitable datasets for the human-equivalence leaderboard

**Follow-up completed September 17:** all 16 DICES AI files and exact conversation joins have now been verified, experiments run, and graphs generated. AlpacaEval human baselines and a blinded AI input batch are also ready. See [results](dices_validation_results.md); the inspection scope below records the earlier search stage.

Checked September 16, 2026. **Existing datasets can supply the repeated human labels we need. New human annotation is not required to start.** Matching AI predictions remain a separate requirement.

For the closest version of the proposed general-preference leaderboard, start with **AlpacaEval human cross-annotations**. For a deeper test of the human-panel mathematics, use **DICES-350**, explicitly interpreting the leaderboard as safety-judgment equivalence. These measure different human tasks and should have separate leaderboards.

## Shortlist

| Dataset | Human coverage | Best use | Main limitation |
| --- | --- | --- | --- |
| AlpacaEval human cross-annotations | Our file inspection: 648 exact response pairs, all with at least four distinct raters | First pairwise preference leaderboard | Matching AI judgments must be obtained or generated; the inspected standard GPT-4 file has only four matching items |
| DICES-350 | 350 conversations × 123 released raters; published quality filtering retains 104 | Deeper human-panel curves and population sensitivity | Safety judgment, not general response preference |
| MultiPref | About 10k pairs, two general crowdworkers and two domain experts each | Larger preference study; released GPT-4 baseline | Four votes come from two deliberately different populations |
| SummEval | 1,600 summaries, five crowd and three expert ratings each | Separate summarization-quality validation | Different task; keep crowd and expert populations separate |

Sources: [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval#data-release), [DICES](https://github.com/google-research-datasets/dices-dataset/blob/main/350/README.md), [MultiPref](https://huggingface.co/datasets/allenai/multipref), [SummEval](https://github.com/Yale-LILY/SummEval#human-annotations).

## What was actually verified

The [machine-readable inspection record](../benchmarks/results/dataset-search/inspection.json) stores pinned revisions, downloaded-file hashes, counts and the limits of these checks. Raw files are in the ignored `.cache/dataset-search` directory. No software behavior was changed, and no model inference was run.

**AlpacaEval:** downloaded the individual human votes and grouped them by exact `(instruction, output_1, output_2)` text. The 2,596 votes form **648 distinct items**: 645 have four raters, two have five, and one has six. There are 16 distinct annotator IDs and no repeated item/rater records. All human preferences are binary 1/2. These are direct file measurements, explaining why the documentation's approximate 650 items should not be used as an exact denominator. The dataset metadata specifies CC-BY-NC-4.0.

We also downloaded the released `alpaca_eval_annotations_alpaca_eval_gpt4.json`: 10,465 AI annotations, but only **four exact overlaps** with the human cross-annotation items, even allowing response reversal. This file must not be joined by prompt alone. Published evaluator-agreement tables establish that multiple judges were evaluated, but do not by themselves supply the required item-level outputs. The current GitHub file inventory did not contain the evaluator-cache JSONs we sought. Other releases/history have not been exhaustively audited.

**DICES:** inspected JUDGE-BENCH's converted crowd file and confirmed **350 unique item IDs, each with 123 individual safety labels**. For implementation, use the original DICES rows with rater IDs, not anonymous list positions in this conversion. The source documents the 19 excluded raters for reproducing the 104-rater analysis and releases the full pool. Choose that quality rule before comparing AI scores. The data are CC-BY-4.0. [Original schema and quality exclusions](https://github.com/google-research-datasets/dices-dataset/blob/main/350/README.md).

A newer [perspective-taking study](https://github.com/shasanamin/llm-perspective-taking/blob/main/data/README.md) advertises DICES outputs for 16 model configurations and ten generations per item. I verified the existence of 16 corresponding JSONL files in its repository. **Their contents and exact alignment are not yet verified.** They predict demographic-group percentages, so they require a target-definition audit; repeated generations cannot be silently averaged when evaluating the value of a single AI judgment. Its data README also reports item counts that need reconciliation with the original 350-item release.

**MultiPref:** the official default schema exposes individual votes, annotator IDs and prompt/comparison IDs. It also advertises a GPT-4 preference configuration. Avoid `human_overall_binarized`, which replaces individual votes with a majority outcome and resolves ties randomly. Only metadata/schema were reviewed here; full-file joins are pending. With two ratings per homogeneous population, a separate expert or crowd benchmark directly supports only k=0 and 1 against another human. Pooling all four requires an explicit mixed-population design; it is not automatically four exchangeable human draws. [Dataset card](https://huggingface.co/datasets/allenai/multipref).

**SummEval:** the original release documents independent ratings across coherence, consistency, fluency and relevance. Select one dimension and one rater population for each analysis. JUDGE-BENCH includes this task, but its aggregate model correlations are insufficient to reconstruct item-level judge predictions. [Original release](https://github.com/Yale-LILY/SummEval#human-annotations), [JUDGE-BENCH](https://github.com/dmg-illc/JUDGE-BENCH).

## Candidates not recommended as the primary test

HelpSteer2/3 do release individual annotation information, but retain the most mutually agreeing annotations after collection. That changes the disagreement distribution we want to measure, so I would use them only with an explicit selection-conditioned interpretation. They are not evidence of an unfiltered independent human population. [HelpSteer2 disagreements](https://huggingface.co/datasets/nvidia/HelpSteer2#disagreements-new---1-oct-2024), [HelpSteer3 methodology](https://huggingface.co/datasets/nvidia/HelpSteer3#annotation-methodology-overview).

The previously audited Arena data still have the one-vote problem. The previously audited MT-Bench cohort remains exploratory because of uncertain human exposure to GPT-4 feedback.

## Recommended next experiment

1. Use AlpacaEval's verified repeated-human items. Select four distinct raters per item using a fixed, label-independent seed so every k uses the same cohort. Preserve the original item/rater identities; rating slots are not globally consistent people.
2. Freeze a small set of AI judges, rubric, response-order handling and one-judgment policy. First seek matching released predictions; otherwise generate judgments for the exact existing response pairs. API spending/model execution would be a separate step, not part of this search.
3. Build the human curve at k=0,1,2,3 against a disjoint human, and compare each judge against the same references. Four human votes permit this design; they do not guarantee every judge's crossing lies below three.
4. Use prompt-group splits for any calibration and paired prompt-group bootstraps for uncertainty. Check Bayesian support and population assumptions before claiming an identified equivalence. Report `>3` for above-range results instead of extrapolating.
5. Repeat on DICES with a prespecified safety target and rater population, initially k=0 through 5 or 10 after support checks. This tests larger panel sizes without paying for new human ratings. Evaluate humans held out from the predictor as an additional control.

The key correction to our earlier discussion is that **additional human collection is one option, not a prerequisite for validating the system**. We can reuse existing multi-rater data; fresh AI judgments may be the only new labels needed for a multi-model leaderboard.

## Additional datasets verified for the Astra request (September 17)

The user redirected this work toward dataset discovery rather than model-access setup. Two additional datasets have now been downloaded and checked. The [inspection record](../benchmarks/results/dataset-search/astra_dataset_search.json) includes pinned revisions, file hashes, counts and search limitations. No Astra inference was run, and no matching released Astra predictions were found in this search; that is not proof that none exist.

| Candidate | Verified human data | Supported use |
| --- | --- | --- |
| [HateXplain](https://github.com/hate-alert/HateXplain/tree/01d742279dac941981f53806154481c0e15ee686/Data) | 20,148 posts; exactly three distinct annotators per post; 60,444 individual votes; 253 annotator IDs | Text classification: hate speech / offensive / normal. k=0,1,2 against a held-out human. Preserve disagreements rather than replacing them with majority labels. |
| [CIFAR-10H](https://github.com/jcpeterson/cifar-10h/tree/389f22d9d300ec71a4c44735d46f550c94b957ab/data) | 10,000 images; 511,000 retained votes; 2,555 retained annotators; 47–63 votes per image | Ten-class visual categorization with substantially deeper human panels. Human annotation files verified; original image files still need acquisition. |
| AlpacaEval, already prepared | 648 exact response pairs with four selected distinct humans per pair | General response preference; existing human curves and blinded input batch are ready. |

**Recommendation:** use HateXplain for the next small text experiment, CIFAR-10H for a distinct vision task, and AlpacaEval for response preference. Keep DICES as the already implemented safety task. Report separate equivalences for each; these populations, task difficulties and label spaces do not define a common universal human unit.

For CIFAR-10H, the raw file includes 25,710 attention-check trials and 514,200 ordinary votes. Excluding attention-check trials and retaining annotators with attention-check accuracy at least 80% reproduces **every cell** of the released 10,000×10 count matrix exactly. This is an empirical reconciliation of the released data, not independent validation of that quality criterion. The actual image-index column is `cifar10_test_test_idx`, although the README calls it `cifar10_test_set_idx`; an adapter must handle the actual spelling. No retained item/annotator duplicates were found. Numeric label zero is a valid class. The original CIFAR-10 test-image order must match the annotation indices. [Publisher's schema](https://github.com/jcpeterson/cifar-10h/blob/master/README.md)

For integration, retain real annotator identities. Our current fixed-depth anonymous paper protocol can use label-independent selection and per-item random slot assignment; slots must not be presented as globally consistent humans. CIFAR-10H permits selecting 47 distinct retained ratings per image without discarding items, but this does not guarantee Bayesian support or a finite crossing at every panel size. Start with a small k range, check support, and report bounds where a crossing is not observed. The fixed-depth selection and its seed should be frozen before scoring Astra.

**ChaosNLI is promising but not ready from the checked sources.** Its authors document 100 human labels per item across SNLI, MNLI and abductive inference, with label-count vectors suitable for anonymous calculations. In this session, both advertised Dropbox archives returned a “File Deleted” preview; the direct download host returned 404. The GitHub releases endpoint was empty. The newer CoT-HLV repository explicitly omits generated traces and data. Do not describe these files as downloaded or Astra predictions as available. Counts alone also do not recover rater identities: count-based sampling needs a validated without-replacement implementation rather than fixed, label-sorted pseudo-rater columns. [ChaosNLI release](https://github.com/easonnie/ChaosNLI), [CoT-HLV availability statement](https://github.com/mainlp/CoT-HLV)

The required human-side data are available for the recommended tasks. An **Astra-specific** equivalence still requires Astra's labels or probability vectors on those exact inputs; a general benchmark score or another model's predictions cannot substitute for them.
