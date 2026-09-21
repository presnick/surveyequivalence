"""Audit the identifiable zero-human boundary on the released Arena predictions.

Uses the existing validated sources and bootstrap schedule. No new labels,
inference, calibration, or production changes are made.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
from research.arena import LABELS, cluster_samples, digest, softmax
from surveyequivalence.combiners import AnonymousBayesianCombiner
from surveyequivalence._data import PreparedRatings
from surveyequivalence._scoring import ordered_mean


def main(source=None, cache=None, output=None):
    source = Path(source) if source is not None else ROOT / 'benchmarks/results/arena-validation'
    cache = Path(cache) if cache is not None else ROOT / '.cache/arena-validation'
    output = Path(output) if output is not None else ROOT / 'benchmarks/results/paper-audit'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    expected_hashes = {r['local']:r['sha256'] for r in manifest['sources']}
    items = pd.read_csv(source/'arena_items.csv')
    models = json.loads((source/'arena_results.json').read_text())
    boot = pd.read_csv(source/'arena_bootstraps.csv', index_col=0, float_precision='round_trip')
    first_name = models[0]['judge']+'.forward.jsonl'
    assert digest(cache/first_name) == expected_hashes[first_name]
    labels_by_id = {row['id']:row['winner_gt'] for row in map(json.loads, (cache/first_name).read_text().splitlines())}
    truth = np.array([LABELS.index(labels_by_id[int(i)]) for i in items.battle_id])
    W = np.asarray([[LABELS[i]] for i in truth], dtype=object)
    counts = np.bincount(truth, minlength=3)
    abc = AnonymousBayesianCombiner()
    abc._prepare(W, LABELS, PreparedRatings(W, LABELS))
    # At k=0, rows with the same held-out label have exactly the same posterior.
    by_label, raw = [], []
    for label in range(3):
        position = int(np.flatnonzero(truth == label)[0])
        prediction = abc.combine(LABELS, [], W=W, item_id=position)
        expected = (counts - np.eye(3, dtype=int)[label]) / (len(truth)-1)
        np.testing.assert_allclose(prediction.probabilities, expected, rtol=0, atol=2e-15)
        by_label.append(prediction.probabilities)
        raw.append(expected.tolist())
    terms = np.array([math.log2(by_label[y][y]) for y in truth])
    agreement_terms = np.array([float(np.argmax(by_label[y]) == y) for y in truth])
    samples = cluster_samples(items.prompt_sha256, len(boot), manifest['seed'])
    schedule_hash = hashlib.sha256()
    for sample in samples:
        schedule_hash.update(len(sample).to_bytes(8,'little')); schedule_hash.update(sample.tobytes())
    data_audit = json.loads((source/'arena_data_audit.json').read_text())
    assert schedule_hash.hexdigest() == data_audit['bootstrap_schedule_sha256']
    baseline = ordered_mean(terms)
    bootstrap_baselines = np.array([ordered_mean(terms[s]) for s in samples[1:]])
    results = []
    for model in models:
        judge = model['judge']
        file = cache/(judge+'.forward.jsonl')
        assert digest(file) == expected_hashes[file.name]
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        indexed = {r['id']:r for r in rows}
        assert len(indexed) == len(items) and set(indexed) == set(items.battle_id)
        rows = [indexed[int(i)] for i in items.battle_id]
        assert [r['winner_gt'] for r in rows] == [LABELS[y] for y in truth]
        probabilities = softmax([[r['logit_A'], r['logit_B'], r['logit_C']] for r in rows])
        raw_score = ordered_mean(np.array([math.log2(float(p[y])) for p,y in zip(probabilities, truth)]))
        differences = boot[judge+'/cross_entropy'].to_numpy() - bootstrap_baselines
        gain = model['cross_entropy'] - baseline
        # Definition 17: if p(0)>=CES the minimum over nonnegative sizes is 0.
        estimate = 0. if gain <= 0 else None
        results.append(dict(judge=judge, log_score=model['cross_entropy'],
                            information_gain_vs_loo_prior_bits=gain,
                            paired_gain_interval=np.quantile(differences,[.025,.975]).tolist(),
                            empirical_equivalence=estimate,
                            status='zero_at_baseline' if estimate == 0 else 'above_observed_range',
                            bootstrap_zero_count=int(np.count_nonzero(differences<=0)),
                            bootstrap_total=len(differences), raw_softmax_log_score=raw_score,
                            rows_affected_by_clipping=int(np.count_nonzero(np.any((probabilities<.02)|(probabilities>.98),axis=1)))))
    report = dict(paper_reference='Definition 17 and Remark 18, p21',
                  items=len(items), human_votes_per_item=1, seed=manifest['seed'],
                  sample_schedule_sha256=schedule_hash.hexdigest(),
                  zero_human_log_score=baseline, zero_human_agreement=ordered_mean(agreement_terms),
                  prior_vectors_by_held_out_label=by_label,
                  raw_prior_vectors_by_held_out_label=raw,
                  zero_human_score_interval=np.quantile(bootstrap_baselines,[.025,.975]).tolist(),
                  judges=results,
                  interpretation='All estimates refer to the released uncalibrated, clipped verdict probabilities on this cohort. Zero equivalence is identifiable; positive equivalence is not identified with one rating. It does not mean absence of information or zero hard-decision agreement.',
                  original_files_note='Historical Arena result files retain their original blanket not-identifiable status; this report supersedes that statement for the zero boundary.',
                  input_hashes={name:digest(source/name) for name in ['manifest.json','arena_items.csv','arena_results.json','arena_bootstraps.csv']},
                  script_sha256=digest(__file__))
    (output/'arena_zero_boundary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    pd.DataFrame({'zero_human_log_score':bootstrap_baselines},index=boot.index).to_csv(output/'arena_prior_bootstraps.csv')
    print(json.dumps({k:report[k] for k in ['items','zero_human_log_score','zero_human_agreement']},indent=2))
    print([(r['judge'],r['empirical_equivalence'],r['bootstrap_zero_count']) for r in results])


if __name__ == '__main__':
    main()
