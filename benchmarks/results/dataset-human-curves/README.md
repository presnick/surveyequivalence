# Human-panel curves for candidate datasets

[PNG](human_panel_curves.png) · [PDF](human_panel_curves.pdf) · [SVG](human_panel_curves.svg) · [Numbers](curve_summary.csv)

These are human Bayesian predictor curves under Agreement scoring, not Astra scores or Astra equivalences. No Astra judgments were available. The figure uses separate vertical scales for three distinct tasks; human units are not transferable between them.

HateXplain uses all 20,148 posts and all three distinct humans per post. CIFAR-10H uses all 10,000 images and selects 47 distinct retained human ratings per image, without conditioning on their labels, after reproducing the published annotation counts using the attention-check filter. Real selected rater IDs are stored separately; rating slots are independently randomized per item and are not global identities. AlpacaEval reuses the previously saved 648-item, four-human curves exactly.

Each curve uses the existing unclipped anonymous Bayesian paper protocol, leave-one-item-out training, disjoint single-human references, and 500 bootstrap samples. HateXplain resamples groups of identical token sequences; CIFAR-10H resamples image indices; AlpacaEval resamples prompt groups. Intervals condition on the fitted combiner, selected ratings and sampled rater partitions. k is limited to 0–2 for the new tasks; 47 ratings do not mean a 46-human curve was computed. Every possible k≤2 input-label histogram has support in another training row in the new datasets.

**CIFAR-10H's unusually low zero-human result needs care.** It is a leave-one-item-out empirical-prior classifier, not uniform random guessing. The two leading full-data label totals differ by only 39 votes, while each item contributes 47 votes. Removing many horse-labeled items switches the prior's winner from horse to dog, so it systematically predicts another class on those items. A separate prior replay verifies the plotted number and that every selected label maximizes the exact remaining class counts. Exact count ties can be resolved differently by floating-point sequence arithmetic; that existing arithmetic was preserved. This artifact makes an independent-training prior a useful future sensitivity check before relying on CIFAR-10H equivalence near zero.

The candidate-dataset plotting experiment is complete. Its standalone generator,
selection tables, and large bootstrap schedules were retired from the maintained
tree during cleanup. A local recovery archive retains them; see
[research tooling](../../../research/README.md). The saved figures, summary curves,
partitions, and manifests remain historical evidence. The graph's original input
hashes are in [figure_provenance.json](figure_provenance.json); some inputs now
exist only in the archive and verified raw-data cache.
