# Real-world validation with Arena and MT-Bench

**Paper-conformance follow-up:** the September 16 [paper audit](paper_conformance_audit.md) updates this plan. **Positive** equivalence requires repeated human labels, but the zero boundary can be identified with one label per item when the prior already outperforms the AI. That occurs for all 16 verified released Arena probability forecasts under the existing log-score protocol. See the [updated results](arena_validation_results.md) before interpreting the original plan below.

Proposed September 15, 2026. **Execution update, September 16:** supported real-data tests and graphs are now available in [the results report](arena_validation_results.md). The Arena join and 16 judge baselines passed; one judge was quarantined. MT-Bench supports an exploratory Agreement-based GPT-4 equivalence, while the intended multi-model probability-based human-equivalence leaderboard still requires additional verified repeated-human data. The stages below record the full plan and its acceptance gates, including work that remains deferred.

Use real human annotations and released AI judgments as the evaluation evidence. Keep small mathematical oracle tests as implementation checks. Real-world agreement alone cannot establish mathematical correctness or make human preference an objective answer key.

**Assignment reference: an Arena-style leaderboard**

The user clarified that the assignment is essentially [OpenLM's Chatbot Arena page](https://openlm.ai/chatbot-arena/) using our rater-equivalence framework, and that the desired ranking is which model corresponds to the greatest number of humans. Operationalize that as a leaderboard of **models acting as preference judges**, ranked by human-panel equivalence.

The primary question is: **How many human judgments per battle are needed to match this model's ability to predict an independent human preference?** Each model predicts A, B, or tie probabilities for the same response pairs. Compare its score with predictions constructed from k human votes, using the same reference population and scoring rule. Find the smallest supported human-panel size that reaches the model's score, with the specified interpolation/censoring policy. Larger equivalence means a model matches the predictive performance of a larger human panel; it does not establish literal equality of probability distributions.

Use CrossEntropy as the primary distribution-sensitive score, subject to the numerical and probability-handling gates below; Agreement is a secondary hard-decision score. Primary leaderboard columns are judge identity, score, human-panel equivalence, uncertainty, and evaluation coverage. Compare models on a shared cohort and human reference curve. The Elo/Bradley–Terry analysis below is optional supporting context rather than the primary deliverable.

Do not label a generator's answer quality with its evaluator's human-equivalence score. Do not count one AI vote as N independent human votes merely because its evaluator has equivalence N. These do not follow from the metric. Repeated independent human annotations remain necessary for the primary leaderboard; single-vote Arena data supports the adapter and agreement validation first.

**The revised experiment**

| Track | Question | Inputs | Interpretation |
| --- | --- | --- | --- |
| Battle agreement | Does an AI judge predict the observed human preference? | Matched human and AI labels for each exact response pair | Agreement with the sampled human population |
| Model rankings | Do humans and AI judges rank and separate the competing models similarly? | Two label sets on the same retained battles | Agreement between aggregate preference rankings |
| Human-panel equivalence | How many human judgments achieve a specified AI judge's predictive performance? | Multiple independent human judgments of each exact item, plus AI predictions | Equivalence for the measured human population, task distribution, combiner, and scoring rule |

The first two tracks are feasible with one human vote per battle. The third is not identifiable from those votes alone. Comparing a k-human predictor with one held-out human needs at least k+1 distinct human judgments per item; a disjoint q-person reference needs k+q. More battles, more AI judges, and bootstrap duplicates cannot supply the missing human repetitions.

An item means the exact prompt/conversation, response pair, and evaluated turn. Two battles involving the same model names are different items. Model Elo/Bradley–Terry ratings are aggregate outputs, not a replacement for the item-by-rater input matrix of `AnalysisPipeline`.

**Verified sources and remaining data questions**

The official [Arena human-preference release](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) exposes a battle ID, model identities, conversation fields, and one recorded winner per row. Its visible outcome columns are A, B, and tie; do not assume that “both bad” can be recovered separately from this export.

The [AI-judge dataset card](https://huggingface.co/datasets/potsawee/chatbot-arena-llm-judges) describes 49,938 single-turn examples. `train.single-turn.json` contains source battles and human outcomes; AI logits are in `llm-judges`, with swapped presentations in `llm-judges-reversed`. Loading only the default split would not provide the AI comparison. The card reports Llama-3-70B-Instruct at 51.37% three-way and 70.81% two-way forward accuracy. These are baseline candidates to reproduce, not acceptance thresholds. The [file tree also provides averaged predictions](https://huggingface.co/datasets/potsawee/chatbot-arena-llm-judges/tree/main/llm-judges-prob-avg); verify their construction against forward/reversed files before using them.

The often-cited 80% depends on evaluation conditions. MT-Bench Table 5 reports GPT-4 pairwise/human agreement of 85% with non-ties and 66% with ties included, for each turn. Judge identity, tie handling, order reconciliation, and denominators matter. Reproduce the matching experiment, rather than requiring every judge to reach 80%. [Original paper, agreement evaluation](https://arxiv.org/html/2306.05685v4#S4.T5).

For repeated human labels, inspect [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) and its co-published `gpt4_pair` split. [Release metadata](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/raw/main/README.md) lists 3,355 human and 2,400 model judgments. Repeated human judgments are visible, but usable independent-rater counts and complete content alignment remain unverified. This is a curated task population, distinct from Arena users. The paper's annotation procedure includes GPT-4 feedback after disagreements: establish whether released votes are initial or revised, and distinguish external experts from authors before claiming an independent human baseline. [Annotation procedure](https://arxiv.org/html/2306.05685v4#A3.SS1).

**1. Freeze the experiment and resolve relevant correctness gates**

Record the runtime commit, dependency lock, dataset revisions and SHA-256 hashes, judge file names, ordered label vocabulary, seed, bootstrap schedule, tie policy, and all filtering rules. Start with released predictions; no model inference, paid API calls, or GPU is needed for this design.

Run the existing checks in the pinned environment:

```sh
.venv/bin/python -m unittest discover -s tests -p '*_tests.py' -v
.venv/bin/python benchmarks/run_correctness_audit.py
```

The [current correctness audit](correctness_audit.md) records a passing 90-method regression suite and failing separate diagnostics. Do not interpret the first command alone as a green correctness gate.

Repair and independently test factorial overflow/cache contamination before general panel scoring. Resolve equivalence plateaus, missing crossings, endpoint censoring, valid-pair reliability, and confidence-interval reporting before track 3. Verify immutable input alignment and sample identifiers for the experiment runner. Preserve unaffected exact-parity fixtures; isolate and document intentional correctness changes.

Initially use Agreement and single-human-reference CrossEntropy. Keep AUC and DMI outside the primary validation until their audited direction, vocabulary, and estimator issues are repaired or explicitly specified. Keep multi-human reference panels out of the first run because binary and multiclass replacement policies currently differ. Ranking metrics belong in an independently tested evaluation module; they are not existing survey-equivalence outputs.

Deliverable: a metric-by-metric gate report stating what is validated, repaired, deferred, or unsupported. The data audit can proceed while numerical repairs are underway.

**2. Audit and join the real battles**

Parse the original conversation arrays, retain exact text, and restrict the primary Arena cohort to verified single-turn overlap. Join by battle ID, then assert model identities, response order, and exact parsed prompt/response content. Normalize serialization only; do not fuzzy-match text. Quarantine conflicts and explain exclusions. An A/B reversal requires swapping both model/response identities and outcome/probability coordinates.

Construct two tables: an item table with stable IDs, content hashes, model versions, turn and available cluster metadata; and a judgment table keyed by item, judge, source and presentation order. Keep unavailable or invalid predictions missing rather than turning them into ties. Validate unique keys, exclusive winner flags, finite logits, and complete label coordinates. Repeated observations must remain distinguishable from duplicated records.

Report overlap and missingness by judge, outcome, model, and available task/language metadata. Use the same battles for each paired human/AI comparison. For direct comparisons among judges, use a common cohort and disclose its exclusions; provide each judge's larger available cohort as a separate sensitivity analysis.

For MT-Bench, match question, turn, model version and response content; count unique human judges per item after deduplication and order normalization. Do not merge different response generations or model versions. Report annotation-depth histograms and how selection by depth changes the task population.

Deliverable: a data manifest, exclusion ledger, join assertions, and coverage report. Do not claim that all 49,938 battles align until these assertions run.

**3. Reproduce dataset-specific agreement before using the pipeline**

Start with forward judgments from a prespecified small judge set: Llama-3-70B-Instruct, Llama-3-8B-Instruct, and dolphin-2.1-mistral-7b. This spans the published table without choosing judges after inspecting our results. Add the remaining judges after the adapter passes.

Reproduce three-way accuracy and the source's two-way calculation with exact counts and denominators. Establish whether the latter merely filters human ties or also forces an A/B decision; those are different statistics. Match argmax ties, rounding and invalid-output handling. Reproduce the MT-Bench paper protocol separately if the released annotation provenance permits it. Unexplained mismatches stop downstream interpretation.

Keep logits and probabilities distinct. Verify the output schema and class-token mapping, then normalize logits with a documented stable softmax when appropriate. These are judge verdict probabilities, not automatically calibrated probabilities of human preference. Map reversed outputs back to the original orientation before averaging probabilities. Do not average raw logits and call it the same operation. Treat forward and reversed presentations as paired measurements of one judge.

Deliverable: reproducible baseline tables, including raw integer counts, provenance and any unresolved differences from published rounded values.

**4. Validate item-level scoring through SurveyEquivalence**

Use human-only `W`, with battle IDs as the index. Put AI judges and prespecified AI ensembles into `classifier_predictions`. Do not insert AI columns into `W` merely as `amateur_cols`: the default Bayesian/Frequency training prior can then incorporate AI votes into the human baseline. Competitor model names belong in item metadata, not rater columns.

For a single human column, compare the pipeline's classifier Agreement against an independent count of matching outcomes. Compare CrossEntropy with an independent ordered mean of `log2(p[human_outcome])`. The package returns negative cross-entropy, so higher is better; report conventional positive log loss separately with explicit sign and units.

`DiscreteDistributionPrediction` clips probabilities at a default `extreme_cutoff=0.02` and then normalizes. Preserve raw judge probabilities separately. Initially score the pipeline with this documented transformation, applied consistently to human and AI predictive distributions, and compare like-for-like probabilities in parity tests. Report raw probability metrics separately. External raw log loss may be infinite at zero observed-label probability; the current pipeline's `math.log2(0)` raises instead. Disabling clipping alone is therefore insufficient: raw-probability pipeline scoring requires strictly positive inputs or a separately tested scorer change. Do not present default-clipped scores as raw model calibration.

Report three-way agreement, confusion counts, outcome-specific agreement and tie rates. Add Brier score and calibration summaries in the evaluation module when probability semantics are established. Any learned calibration, ensemble weighting, or decision thresholds must be fit on separate prompt groups and frozen before final evaluation.

With one human column, the current pipeline measures only k=0 on the human power curve. Suppress a numerical “N humans” interpretation for that cohort.

**5. Optional: compare human and AI model rankings**

Use batch Bradley–Terry as the primary fit and sequential Elo as a sensitivity analysis. The former avoids Elo's battle-order dependence. Pin and independently check the chosen [Arena reference rating implementation](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/elo_analysis.py) and its tie/weight conventions; do not compare this subset against an unmatched live leaderboard.

Fit human labels and each judge's hard labels separately on identical battle IDs and exact competitor versions. Keep ties under the same specified rule, initially the pinned reference convention. A tie-aware Davidson model and soft-probability fits are separate sensitivity analyses, not substitutes for baseline reproduction.

Use the same rating scale and a fixed common zero-point constraint, such as zero mean over the same competitor set. Check comparison-graph connectivity, sparse models, optimization convergence and separation. Disconnected graphs or unbounded fits require explicit restrictions or a prespecified regularized analysis; do not manufacture comparable scores. Apply the same policy to bootstrap replicates and report failures.

Compare Kendall tau-b/Spearman rank agreement, per-model signed rating differences, MAE/RMSE on the common scale, and pairwise predicted win-probability differences. Report model coverage and ties alongside ranks. Distinguish spread **between competing models** from bootstrap uncertainty **about each model**. An optional Wasserstein comparison of rating histograms loses model identities and cannot replace per-model differences.

Resample the same battles or prompt/conversation clusters for both label sets and refit both ratings in every replicate. Use paired differences for intervals; overlapping marginal intervals do not establish equivalence. Predefine practical margins before making a claim of sufficiently similar rankings or ratings. When user/session identifiers are unavailable, disclose the remaining dependence that cannot be modeled.

Deliverable: aligned rating tables, paired intervals, rank plots, a model-level difference plot, and a comparison-coverage heatmap.

**6. Run human-panel equivalence only on supported repeated labels**

First try MT-Bench's verified independent repeated-label subset. If annotation depth or provenance is insufficient, specify additional independent, blinded human annotation of real Arena battles. A concrete expansion is 500–1,000 battles with six humans each, supporting k=0 through 5 against one held-out human; final size depends on pilot uncertainty. Do not interpret MT-Bench experts as interchangeable with Arena users.

Use real judge identities where applicable, or anonymous rating slots for exchangeable independent human draws. Use a one-human reference initially. Set `max_K=K+1` to evaluate k=0 through K under the existing exclusive convention. For unequal coverage, report the effective predictor/reference counts at every k; a primary fixed-depth cohort prevents changes in item composition from masquerading as gains from extra raters. Exercise the row-local reference cap in a separate uneven-coverage check.

Compare prespecified Frequency, Plurality and Bayesian human predictors against the same held-out human labels used to score AI predictions. Verify no reference vote enters its predictor and that Bayesian leave-one-out removes the intended item. Keep held-out prompt groups separate for any learned configuration; distinguish the package's item-level leave-one-out contract from stricter prompt-group generalization.

Audit the actual reference weights and shared item masks, not just the bootstrap indices. The current pipeline scores classifiers against all expert columns, while each human subset uses its complement and may lose unsupported Bayesian predictions. A fixed-depth cohort with exhaustive symmetric subsets can make reference weighting equivalent, but missingness, capped subsets and failed predictions can break that equivalence. The validation runner must compare both predictors on a shared reference schedule and valid-item mask, report prediction coverage, and verify when native pipeline scores target that same comparison. Do not silently compare scores over different surviving populations.

Use matched bootstrap samples across human curves and AI scores. For repeated questions/pairs/turns, add a validated cluster-sampling adapter rather than silently relying on the pipeline's ordinary item bootstrap. Define whether inference is conditional on the observed raters or includes rater resampling. Keep the original-data estimate separate from bootstrap draws.

Report each score's curve, effective sample counts, uncertainty and supported crossing. For above-range AI performance, report “exceeds the measured range”; for an unsupported crossing, report unavailable. Do not force a finite answer or enforce monotonically increasing empirical curves without an explicitly separate estimator.

**7. Establish controls, parity, and completion criteria**

Use fixed subsets of actual battles at 50, 1,000 and 2,000 items, with full-data analysis after pilot gates. Run 50 bootstrap replicates during development and 500 for release comparisons; assess interval stability before interpreting fine differences. Start with `procs=1`, `random_state=1729`, and `working_memory_mb=512`; compare two workers and small memory blocks on the fixed fixtures. Profile full-data execution before increasing parallelism.

Apply controlled transformations to real data: identical human/AI labels must give perfect hard agreement and identical rating fits under identical settings; swapping all A/B fields and labels must preserve semantic results under a label-equivariant tie rule; shuffling AI labels should agree with a permutation null preserving outcome frequencies, not an assumed 50% chance rate. Isolate exact argmax ties when reproducing a source's first-coordinate tie rule, since that rule itself depends on coordinate order. Test missing judgment alignment, duplicate rejection, bootstrap repetitions, input reordering, label-coordinate changes, and save/load.

Require exact equality between corrected generic and optimized pipeline paths for predictions, valid masks, classifier scores, power-curve cells, and derived statistics in the pinned environment with frozen schedules. Independent count/rational oracles establish the mathematical target. Cross-library floating-point optimization and logarithmic calculations may require declared numerical tolerances; they must not weaken the repository's exact runtime-parity contract.

Check rating code independently using tiny enumerated win tables, label reversal, disconnected graphs, ties, and unchanged fits from equivalent count representations. An identity or null control is an implementation diagnostic, not evidence that human and AI judges agree on the unmodified data.

Deliver a versioned validation runner, small licensed real-data fixtures, manifests and schedules, baseline reproduction report, metric gate report, and real-workload timing/memory results. Download full datasets during an explicit integration run rather than ordinary unit tests. The final report must distinguish software correctness checks, successful source reproduction, empirical human/AI agreement, and remaining limits on generalization. A result below 80%, or genuine rank disagreement, can be a valid scientific result.
