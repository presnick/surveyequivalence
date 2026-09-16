# Real-data Arena and MT-Bench validation

Run September 16, 2026 in the existing pinned environment. The experiments use released human judgments and AI predictions; no new model inference or GPU was used. Raw data is cached outside version control. Dataset revisions, hashes, runtime-source hashes and seed are recorded in the [manifest](../benchmarks/results/arena-validation/manifest.json).

**The supported experiments ran successfully. Positive human-panel equivalence for multiple Arena judges is still not identifiable from these releases alone.** Arena supports a comparison of many AI judges against one observed human vote per battle. MT-Bench supports an exploratory repeated-human comparison, but only one released AI judge, and its votes' independence from judge feedback is unresolved.

**Paper-audit correction, September 16:** one vote per battle can identify **zero** empirical equivalence when the zero-human prior already beats the classifier. The new leave-one-out prior scores **−1.5836467 bits**, above all 16 verified judges' released, uncalibrated and clipped probability forecasts. Each therefore has empirical **log-score equivalence 0**, including all 500 paired bootstrap samples. This does not imply zero agreement or no information. Positive Agreement equivalence remains unmeasured. The historical JSON status strings used an overly broad “not identifiable” description; the [paper audit and updated figures](paper_conformance_audit.md#arena-a-correction-to-the-previous-conclusion) supersede that interpretation. Production code was not changed by the audit.

![Arena scores relative to the paper's zero-human benchmark](../benchmarks/results/paper-audit/arena_paper_scorecard.png)

**Current graphs**

[Download the Arena graph as SVG](../benchmarks/results/paper-audit/arena_paper_scorecard.svg).

![MT-Bench human-panel equivalence](../benchmarks/results/paper-audit/rater_equivalence_paper_view.png)

[Download the MT-Bench graph as SVG](../benchmarks/results/paper-audit/rater_equivalence_paper_view.svg).
The older PNG/PDF graphs have been superseded by these paper-audit figures.

**Arena: verified real-world agreement**

All **49,938** single-turn battles matched the **57,477-row** original human release exactly by ID, model identity, parsed prompt, responses and outcome. There were zero content mismatches. The matched set contains 17,312 A wins, 16,985 B wins and 15,641 ties, grouped into 44,256 distinct question-text clusters. [Data audit](../benchmarks/results/arena-validation/arena_data_audit.json).

Sixteen of seventeen judges reproduced their dataset card's forward three-way and two-way percentages at the published precision. **Starling-LM-7B-alpha was excluded from the main comparison**: its released forward predictions yield 45.9850% three-way agreement and 64.3497% two-way agreement, versus 45.34% and 64.68% in the card. The discrepancy remains unresolved; it was not silently corrected. [Reproduction records](../benchmarks/results/arena-validation/baseline_reproduction.json).

| Judge | Three-way agreement | 95% question-bootstrap interval |
| --- | ---: | ---: |
| Qwen2-72B-Instruct | 52.06% | 51.62–52.54% |
| Meta-Llama-3-70B-Instruct | 51.37% | 50.89–51.81% |
| Hermes-3-Llama-3.1-70B | 50.83% | 50.38–51.34% |
| Athene-70B | 50.04% | 49.58–50.49% |

These are descriptive estimates, not claims of statistically separated ranks. All judges use identical battles and 500 paired question-cluster bootstrap schedules. Voter/session identity was unavailable, so the intervals cannot account for that additional dependence. [Full leaderboard](../benchmarks/results/arena-validation/arena_leaderboard.csv), [bootstrap scores](../benchmarks/results/arena-validation/arena_bootstraps.csv).

The adapter uncovered an important tie convention. The released three-way rule chooses A or B only when its logit strictly exceeds both others; otherwise it reports a tie, including A/B equal maxima when the tie-class logit is smaller. The two-way calculation removes human ties, forces A/B, and chooses B on equal A/B logits. The runner reproduces these rules explicitly instead of assuming ordinary first-coordinate argmax. Hard decisions and probability scores therefore use separate prediction objects.

For the three prespecified pilot judges, reversed-order outputs were mapped back to the original orientation. Reconstructed probability averages matched the released averaged files to a maximum absolute difference of 2.22e-16. An identical-label control scored exactly 1. A 100-permutation control for dolphin-2.5-mixtral-8x7b averaged 34.5312%, versus 34.5316% expected from the label marginals. [Controls](../benchmarks/results/arena-validation/arena_controls.json).

**Probability scores need careful interpretation**

The plot also reports the package's mean log2 probability of the human outcome, with higher values better. Logits were converted to float64 softmax probabilities, then passed through the package's existing .02 clipping and normalization. These are **uncalibrated verdict probabilities**, not established human-vote probabilities.

The best log score was the base Meta-Llama-3-8B model at −1.6357 bits, despite only 37.49% hard agreement. Every evaluated judge scored below the uniform-probability baseline of −log2(3) = −1.5850. This shows why hard-decision agreement and probabilistic prediction quality cannot be conflated: confident wrong judgments are costly under log scoring. It does not establish that the base model is the best calibrated judge. Temperature calibration on separate prompt groups is a future experiment; no calibration was fitted on this evaluation set.

For every downloaded judge, public pipeline classifier scores matched independent agreement counts and ordered `math.log2` reductions exactly. This validates the used scoring path on these data, not every advertised scorer.

**MT-Bench: exploratory human-panel equivalence**

The release contains 3,355 human annotations and 2,400 GPT-4 pairwise judgments. After exact content matching through the evaluated turn and A/B normalization, all 1,814 human-rated items matched a model judgment. One duplicate record by the same human on the same item was deduplicated. Authors were excluded from the primary cohort.

There were **227 items across 57 question clusters** with at least three distinct external experts. Exactly three experts per item were selected with seed 1729, before calculating performance. Predictor panels of size 0, 1 and 2 were enumerated exhaustively and evaluated against their held-out complements. This keeps the item population fixed and reference weights balanced. The AI judge was evaluated against the same selected human population. No Bayesian prediction was unsupported. [Coverage and protocol](../benchmarks/results/arena-validation/mt_data_audit.json), [selected items](../benchmarks/results/arena-validation/mt_items.csv), [ratings](../benchmarks/results/arena-validation/mt_human_ratings.csv).

| Predictor | Agreement with held-out human |
| --- | ---: |
| Zero-human prior | 40.68% |
| One-human predictor | 61.38% |
| Two-human predictor | 67.11% |
| Released GPT-4 judge | 60.06% |

Bayesian and Frequency combiners produced identical hard-decision curves on this selected cohort. Their probability predictions need not be identical. Bayesian uses leave-one-item-out; the built-in Frequency zero-rater prior uses the full dataset. Bayesian is the primary reference.

The interpolated **Agreement-based equivalence is 0.94 human judgments**, with a **95% paired percentile interval of 0.68–1.70**. There were 499 supported bootstrap crossings and one above the measured two-human range. That censored replicate was retained; both percentile endpoint ranks remain within observed crossings. This is not an interval computed after dropping out-of-range results. [Equivalence summary](../benchmarks/results/arena-validation/mt_equivalence_summary.json).

These results are **exploratory and specific to the selected released annotations**. The original annotation procedure describes GPT-4 feedback after disagreement, and the release does not establish whether each stored vote is initial or revised. Distinct judge IDs alone do not establish independent, unexposed human judgments. The interval is conditional on the selected experts, observed votes and fitted combiner; it resamples questions, not raters or model training. It does not include uncertainty from all those sources.

The released GPT-4 split contains hard verdicts only, so no genuine probability-based GPT-4 human-equivalence estimate was manufactured. Its Agreement result cannot be transferred to the other Arena judges or population.

The separate, full expert-annotation comparison reproduces the MT-Bench Table 5 denominators: 1,343/1,325 votes by turn with ties and 859/864 decisive comparisons. Observed agreement is 65.97%/65.74% including ties and 84.63%/84.61% excluding them, matching the paper's rounded 66%/85%. The 227-item repeated-vote cohort is a different, selected population, explaining why its result is not required to equal those percentages.

**Correctness and execution checks**

The original 90-method regression suite passed before this work. Four new tests first reproduced NumPy factorial overflow and cross-scorer cache contamination, then passed after a narrow runtime fix: factorials now use unbounded Python integers and invalid legacy NumPy cache entries are recomputed. `comb` retains its existing floating-point division order. This does not solve every large-panel numerical limitation.

Nine additional adapter/reporting methods cover exact content identity, response order, tie rules, rational Bayesian draws, cluster sampling, missing/plateau crossings and censored intervals. The final regression suite contains **103 passing methods**. [Test records](../benchmarks/results/arena-validation/validation_checks.json).

The real-data checks also established:

- Exact optimized-versus-generic parity on 50-, 1,000- and 2,000-item Arena subsets, each with 500 bootstraps, including score cells, k=0 curve cells, summary statistics and save/load. These one-human curves are parity checks, not usable human-equivalence estimates.
- Exact MT-Bench score/curve parity against separately enumerated subset/reference contributions, plus rational Bayesian probability checks with a declared 1e-14 cross-arithmetic tolerance.
- Exact MT-Bench parity across one/two workers, small memory blocks and reordered classifier rows on a fixed six-sample schedule.
- Exact result-table byte hashes across a repeated real-data run; the manifest and timing fields intentionally change.

[Real-data parity records](../benchmarks/results/arena-validation/real_data_pipeline_parity.json), [MT-Bench oracle records](../benchmarks/results/arena-validation/mt_oracle_checks.json). Recorded timings are single runs against generic dispatch, not three-repetition speedup claims against the original repository.

The broader [post-fix mathematical audit](../benchmarks/results/arena-validation/post-fix-audit/summary.json) **still fails** for previously documented AUC, DMI, statistics, input and execution cases. Those paths are not certified by this report. The graphs use explicit reporting helpers for supported crossings/censoring and bootstrap-only percentile intervals, rather than the problematic legacy derived-statistic/plot behavior. Existing historical fixtures and tolerances were not weakened.

**Reproduction and remaining work**

With the downloads already present:

```sh
.venv/bin/python -m unittest discover -s tests -p '*_tests.py' -v
.venv/bin/python benchmarks/arena_validation.py --bootstraps 500
.venv/bin/python benchmarks/arena_pipeline_checks.py --bootstraps 500
.venv/bin/python benchmarks/arena_validation.py --plot-only
```

For a fresh download, create `.cache/arena-validation`, save the pinned AI dataset README from revision `638a7e3be3078e55d2eb0b789d34749747822c49` as `arena-judges-README.md`, run `arena_validation.py --download-config`, and use `curl -L --fail --config .cache/arena-validation/download.conf`. The generated source manifest lists every pinned file. MT-Bench's optional Parquet reader was installed separately using `.venv/bin/python -m pip install --no-deps --target .cache/arena-reader pyarrow==14.0.2`; the project dependency lock was unchanged.

To measure positive equivalences for the intended multi-model leaderboard, obtain multiple independent, blinded human votes on the same Arena battles as the AI predictions, or obtain multiple AI judges' probability predictions on a verified repeated-human cohort. Confirm vote provenance, then evaluate calibration on separate prompt groups and repeat the equivalence analysis. Remaining global audit repairs are a separate prerequisite for broader production use. The [paper-conformance audit](paper_conformance_audit.md) specifies the mathematical corrections and recommended leaderboard design.

Sources: [Arena human release](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k), [AI judge release](https://huggingface.co/datasets/potsawee/chatbot-arena-llm-judges), [MT-Bench judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments), [original paper](https://arxiv.org/html/2306.05685v4).
