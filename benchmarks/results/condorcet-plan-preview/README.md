# Condorcet / rater-equivalence preview

A descriptive preview based on the existing NBA winner-only panel curve. No source data or existing results are modified.

Reproduce from the repository root:

```sh
.venv/bin/python -B research/condorcet_preview.py
```

- `condorcet_nba_preview.png`: review image.
- `condorcet_nba_preview.svg`: scalable graphic.
- `preview_numerics.json`: exact plotted values, crossing, source hashes and verification.

Observed accuracy at 9 and 10 people is 65.894616%; the homogeneous iid model with p=0.630949 reaches 79.454556% at those sizes. The model reaches the observed target at 2.459101 interpolated jurors, or 3 whole jurors. The interpolated number is a model projection, not a measured count of independent real people.

The gap does not identify social dependence: item difficulty, voter heterogeneity and finite sampling also differ. No new uncertainty analysis is performed. Odd/even plateaus and the full integer grid are retained.
