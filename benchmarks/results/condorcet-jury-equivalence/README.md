# Condorcet jury / rater-equivalence results

See [results, methods, sources, and reproduction](../../../docs/condorcet_jury_equivalence.md)
and the [ten-page PDF](../../../output/pdf/condorcet_jury_equivalence.pdf).

- `01_*` through `06_*`: the six planned figures, PNG and SVG.
- `07_*`: full city support and Ducks species appendix.
- `summary.csv`: maximum-supported observed accuracy and hypothetical iid-jury equivalents.
- `panel_curves.csv`, `class_curves.csv`: exact majority curves and benchmark models.
- `nba_equivalence.csv`, `learned_equivalence.csv`: empirical performance-matching counts.
- `learning_design.json`, `learning_splits.csv`, `trained_raters.csv`: held-out design and fitted policies.
- `items.csv`, `item_diagnostics.csv`, `*_item_curves.npz`: retained items and exact item-level scores.
- `bootstrap_draws.npz`, `learning_bootstrap.npz`: paired resampling weights and curves.
- `theory_curves.csv`, `shared_case_model.csv`: analytical model illustrations, not empirical responses.
- `data_audit.json`, `results.json`, `environment.json`, `verification.json`: provenance, interpretation and validation.

Raw sources and normalized respondent records are cached outside this directory.
No missing vote is imputed. Ordinary finite-pool and trained weighted scores are
exact, with half credit for ties. City results have no iid-item confidence bands.
Performance-equivalent counts depend on their stated task, reference population,
score, and support; they are not a universal unit of human intelligence.
