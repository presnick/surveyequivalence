"""Offline real-data validation. See docs/arena_validation_plan.md.

Generate a pinned curl download list with --download-config, download it, then
run without that flag. Raw downloads live in the ignored .cache directory.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))

import numpy as np
import pandas as pd
from surveyequivalence import (AnalysisPipeline, AgreementScore, CrossEntropyScore,
                              AnonymousBayesianCombiner, FrequencyCombiner)
from surveyequivalence.combiners import DiscretePrediction, DiscreteDistributionPrediction
from surveyequivalence._scoring import ordered_mean

LABELS = ['A', 'B', 'tie']
PILOT_JUDGES = ['Meta-Llama-3-70B-Instruct', 'Meta-Llama-3-8B-Instruct', 'dolphin-2.1-mistral-7b']
REVISIONS = {
    'potsawee/chatbot-arena-llm-judges': '638a7e3be3078e55d2eb0b789d34749747822c49',
    'lmarena-ai/arena-human-preference-55k': '18c298340948c0e7f7727399fd459cca6ce0ca6f',
    'lmsys/mt_bench_human_judgments': 'f7d2896d2cc5d80f8b55c2bbc722613555233c25',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def published_baselines(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        cells = [x.strip() for x in line.split('|')]
        if len(cells) >= 8:
            try:
                values[cells[1]] = [float(x) for x in cells[2:8]]
            except ValueError:
                pass
    values.pop('Random Guessing', None)
    return values


def download_config(cache):
    """Emit fixed-host URLs, never execute commands from downloaded metadata."""
    cache.mkdir(parents=True, exist_ok=True)
    judges = list(published_baselines(cache / 'arena-judges-README.md'))
    paths = []
    def add(repo, remote, local):
        paths.append(dict(url=f'https://huggingface.co/datasets/{repo}/resolve/{REVISIONS[repo]}/{remote}',
                          local=local, repo=repo, revision=REVISIONS[repo]))
    add('lmarena-ai/arena-human-preference-55k', 'train.csv', 'arena-human.csv')
    add('potsawee/chatbot-arena-llm-judges', 'train.single-turn.json', 'arena-single-turn.json')
    for judge in judges:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', judge):
            raise ValueError('Unexpected judge filename')
        add('potsawee/chatbot-arena-llm-judges', f'llm-judges/{judge}.jsonl', f'{judge}.forward.jsonl')
    for judge in PILOT_JUDGES:
        for folder, suffix in [('llm-judges-reversed', 'reversed'), ('llm-judges-prob-avg', 'averaged')]:
            add('potsawee/chatbot-arena-llm-judges', f'{folder}/{judge}.jsonl', f'{judge}.{suffix}.jsonl')
    for split, name in [('human', 'human-00000-of-00001-25f4910818759289.parquet'),
                        ('gpt4', 'gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet')]:
        add('lmsys/mt_bench_human_judgments', 'data/' + name, f'mt-{split}.parquet')
    write_json(cache / 'sources.json', paths)
    config = []
    for entry in paths:
        target = cache / entry['local']
        if not target.exists():
            config += [f'url = "{entry["url"]}"', f'output = "{target}"']
    (cache / 'download.conf').write_text('\n'.join(config) + '\n')
    print(f'{len(config)//2} downloads listed in {cache / "download.conf"}', flush=True)


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError('Expected three finite logits per judgment')
    exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponents / exponents.sum(axis=1, keepdims=True)


def source_decisions(scores, binary=False):
    """Released Arena policy: strict winner, otherwise tie; binary equality -> B."""
    scores = np.asarray(scores)
    if binary:
        return (scores[:, 1] >= scores[:, 0]).astype(int)
    return np.where((scores[:, 0] > scores[:, 1]) & (scores[:, 0] > scores[:, 2]), 0,
                    np.where((scores[:, 1] > scores[:, 0]) & (scores[:, 1] > scores[:, 2]), 1, 2))


def canonical_mt(row):
    """Exact content through the evaluated turn; future turns are not inputs."""
    a, b = row.model_a, row.model_b
    ca = json.dumps(list(row.conversation_a[:2 * row.turn]), sort_keys=True, ensure_ascii=False)
    cb = json.dumps(list(row.conversation_b[:2 * row.turn]), sort_keys=True, ensure_ascii=False)
    outcome = {'model_a': 'A', 'model_b': 'B', 'tie': 'tie', 'tie (inconsistent)': 'tie'}[row.winner]
    if a > b:
        a, b, ca, cb = b, a, cb, ca
        outcome = {'A': 'B', 'B': 'A', 'tie': 'tie'}[outcome]
    key = (int(row.question_id), int(row.turn), a, b,
           hashlib.sha256(ca.encode()).hexdigest(), hashlib.sha256(cb.encode()).hexdigest())
    return key, outcome


from surveyequivalence._equivalence import supported_equivalence


def cluster_samples(groups, count, seed):
    """Paired cluster bootstrap; retain every item in each selected cluster."""
    members = defaultdict(list)
    for i, group in enumerate(groups):
        members[group].append(i)
    blocks = [np.asarray(v, dtype=np.int64) for v in members.values()]
    rng = np.random.RandomState(seed)
    samples = [np.arange(len(groups), dtype=np.int64)]
    for _ in range(count):
        samples.append(np.concatenate([blocks[i] for i in rng.randint(len(blocks), size=len(blocks))]))
    return samples


def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError('Intervals require finite bootstrap replicates')
    return [float(x) for x in np.quantile(values, [.025, .975])]


def equivalence_summary(entries):
    """Retain censored samples; report percentiles only if their ranks are known."""
    lower, upper = [], []
    result = dict(actual=entries[0], bootstrap_status_counts=dict(Counter(x['status'] for x in entries[1:])),
                  interval=None, interval_note='Percentile endpoints are not identified by the supported crossings.')
    for entry in entries[1:]:
        if entry['status'] == 'crossing':
            lower.append(entry['estimate']); upper.append(entry['estimate'])
        elif entry['status'] == 'above_range':
            lower.append(entry['bound']); upper.append(float('inf'))
        elif entry['status'] == 'below_range':
            lower.append(float('-inf')); upper.append(entry['bound'])
        else:
            return result
    def quantile(values, q):
        values = sorted(values)
        position = (len(values) - 1) * q
        a, b = values[math.floor(position)], values[math.ceil(position)]
        if not math.isfinite(a) or not math.isfinite(b):
            return None
        return a + (position - math.floor(position)) * (b - a)
    if len(lower) < 2:
        return result
    endpoints = []
    for q in [.025, .975]:
        low, high = quantile(lower, q), quantile(upper, q)
        if low is None or high is None or low != high:
            return result
        endpoints.append(float(low))
    result.update(interval=endpoints,
                  interval_note='Paired percentile interval; all censored samples retained. Both endpoint ranks are within observed crossings.')
    return result


def pipeline(W, predictions, combiner, scorer, samples, *, procs=1, memory=512, seed=1729):
    return AnalysisPipeline(W, classifier_predictions=predictions, combiner=combiner, scorer=scorer,
                            allowable_labels=LABELS, anonymous_raters=True, verbosity=0,
                            max_K=W.shape[1], max_rater_subsets=200,
                            item_samples=[W.index[s] for s in samples], random_state=seed,
                            procs=procs, working_memory_mb=memory)


def exact_frame(left, right):
    pd.testing.assert_frame_equal(left, right, check_exact=True)


def audit_arena(cache):
    battles = json.loads((cache / 'arena-single-turn.json').read_text())
    human = pd.read_csv(cache / 'arena-human.csv').set_index('id', verify_integrity=True)
    seen = set()
    for row in battles:
        key = row['id']
        if key in seen:
            raise ValueError(f'Duplicate battle {key}')
        seen.add(key)
        source = human.loc[key]
        for field in ['model_a', 'model_b', 'winner_model_a', 'winner_model_b', 'winner_tie']:
            if row[field] != source[field]:
                raise ValueError(f'Battle mismatch: {key} {field}')
        for field, original in [('question', 'prompt'), ('answer_a', 'response_a'), ('answer_b', 'response_b')]:
            decoded = json.loads(source[original])
            if len(decoded) != 1 or row[field] != decoded[0]:
                raise ValueError(f'Content mismatch: {key} {field}')
        if sum(row[k] for k in ['winner_model_a', 'winner_model_b', 'winner_tie']) != 1:
            raise ValueError('Non-exclusive winner')
    labels = [LABELS[next(i for i, k in enumerate(['winner_model_a', 'winner_model_b', 'winner_tie']) if row[k])]
              for row in battles]
    questions = [hashlib.sha256(row['question'].encode()).hexdigest() for row in battles]
    report = dict(human_rows=len(human), matched_single_turn_rows=len(battles),
                  outcome_counts=dict(Counter(labels)), unique_prompt_clusters=len(set(questions)),
                  content_mismatches=0, human_votes_per_battle=1,
                  equivalence_identifiable=False)
    return battles, labels, questions, report


def read_judge(path, ids, labels):
    rows = [json.loads(line) for line in path.open()]
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError(f'Duplicate or unmatched predictions in {path.name}')
    rows = [by_id[i] for i in ids]
    if [r['winner_gt'] for r in rows] != list(labels):
        raise ValueError(f'Ground-truth alignment mismatch in {path.name}')
    logits = np.asarray([[r['logit_A'], r['logit_B'], r['logit_C']] for r in rows])
    predictions = np.asarray([LABELS.index(r['winner_pred']) for r in rows])
    if not np.array_equal(source_decisions(logits), predictions):
        raise ValueError('Stored decisions differ from the released strict-winner/tie policy')
    return rows, logits, softmax(logits), predictions


def run_arena(args):
    started = time.perf_counter()
    battles, labels, groups, audit = audit_arena(args.cache)
    ids = [r['id'] for r in battles]
    truth = np.asarray([LABELS.index(x) for x in labels])
    W = pd.DataFrame({'human': labels}, index=pd.Index(ids, name='battle_id'))
    baselines = published_baselines(args.cache / 'arena-judges-README.md')
    samples = cluster_samples(groups, args.bootstraps, args.seed)
    schedule_hash = hashlib.sha256()
    for sample in samples:
        schedule_hash.update(len(sample).to_bytes(8, 'little'))
        schedule_hash.update(sample.tobytes())
    audit['bootstrap_schedule_sha256'] = schedule_hash.hexdigest()
    audit['bootstrap_unit'] = 'exact question text; all battles in a selected question retained'
    results, checks, bootstrap_columns, controls = [], [], {}, []
    for judge, baseline in baselines.items():
        rows, logits, raw, decisions = read_judge(args.cache / f'{judge}.forward.jsonl', ids, labels)
        swapped = np.asarray([1, 0, 2])[source_decisions(logits[:, [1, 0, 2]])]
        if not np.array_equal(swapped, decisions):
            raise AssertionError('A/B relabeling changed the released three-way decision')
        probabilities = [DiscreteDistributionPrediction(LABELS, [float(p) for p in row]) for row in raw]
        predictions = pd.DataFrame({judge: [DiscretePrediction(LABELS[i]) for i in decisions]}, index=W.index)
        analysis = AnalysisPipeline(W, classifier_predictions=predictions, combiner=FrequencyCombiner(),
                                    scorer=AgreementScore(), allowable_labels=LABELS, anonymous_raters=True,
                                    item_samples=[W.index], max_K=1, procs=1, random_state=args.seed,
                                    verbosity=0, run_on_creation=False)
        agreement = float(analysis.compute_classifier_scores().df.iloc[0, 0])
        matches = (decisions == truth).astype(float)
        if agreement != float(matches.sum() / len(matches)):
            raise AssertionError('Pipeline/direct agreement mismatch')
        analysis.scorer = CrossEntropyScore()
        analysis.classifier_predictions = pd.DataFrame({judge: probabilities}, index=W.index)
        cross_entropy = float(analysis.compute_classifier_scores().df.iloc[0, 0])
        log_terms = np.asarray([math.log2(pred.label_probability(y)) for pred, y in zip(probabilities, labels)])
        if cross_entropy != ordered_mean(log_terms):
            raise AssertionError('Pipeline/independent ordered log-score mismatch')
        decisive = truth != 2
        binary = float(np.mean(source_decisions(logits[decisive], binary=True) == truth[decisive]))
        baseline_ok = abs(agreement * 100 - baseline[0]) <= .0051 and abs(binary * 100 - baseline[1]) <= .0051
        checks.append(dict(judge=judge, published_three_way=baseline[0], reproduced_three_way=agreement * 100,
                           published_two_way=baseline[1], reproduced_two_way=binary * 100,
                           exact_both_scorers_vs_independent=True, semantic_swap_invariant=True, passed=baseline_ok))
        if not baseline_ok:
            write_json(args.output / 'baseline_reproduction.json', checks)
            print(f'QUARANTINED {judge}: source baseline differs; see baseline_reproduction.json', flush=True)
            continue
        boot_agreement = np.asarray([ordered_mean(matches[s]) for s in samples[1:]])
        boot_ce = np.asarray([ordered_mean(log_terms[s]) for s in samples[1:]])
        bootstrap_columns[judge + '/agreement'] = boot_agreement
        bootstrap_columns[judge + '/cross_entropy'] = boot_ce
        lo, hi = interval(boot_agreement)
        cl, ch = interval(boot_ce)
        confusion = np.zeros((3, 3), dtype=int)
        np.add.at(confusion, (truth, decisions), 1)
        results.append(dict(judge=judge, n=len(ids), agreement=agreement, agreement_low=lo, agreement_high=hi,
                            cross_entropy=cross_entropy, cross_entropy_low=cl, cross_entropy_high=ch,
                            binary_forced_agreement=binary, binary_n=int(decisive.sum()),
                            brier_raw=float(np.mean(np.sum((raw - np.eye(3)[truth])**2, axis=1))),
                            predicted_tie_rate=float(np.mean(decisions == 2)),
                            equivalence_status='not_identifiable_one_human_vote_per_battle',
                            confusion=confusion.tolist()))
        if judge in PILOT_JUDGES:
            reverse_labels = [{'A':'B', 'B':'A', 'tie':'tie'}[x] for x in labels]
            _, reversed_logits, reverse_probs, reverse_decisions = read_judge(args.cache / f'{judge}.reversed.jsonl', ids, reverse_labels)
            reverse_probs = reverse_probs[:, [1, 0, 2]]
            average = (raw + reverse_probs) / 2
            released = [json.loads(line) for line in (args.cache / f'{judge}.averaged.jsonl').open()]
            lookup = {r['id']: r for r in released}
            if len(lookup) != len(released) or set(lookup) != set(ids):
                raise AssertionError('Averaged judgment IDs mismatch')
            expected = np.asarray([[lookup[i][k] for k in ['prob_A', 'prob_B', 'prob_C']] for i in ids])
            for i, y in zip(ids, labels):
                if lookup[i]['winner_gt'] != y:
                    raise AssertionError('Averaged ground-truth mismatch')
            difference = float(np.max(np.abs(average - expected)))
            # This is a cross-implementation softmax check, not runtime parity.
            if difference > 2e-7:
                raise AssertionError(f'Unexpected forward/reverse averaging: {difference}')
            controls.append(dict(judge=judge, averaged_probability_max_abs_error=difference,
                                 reversed_agreement=float(np.mean(np.asarray([1, 0, 2])[reverse_decisions] == truth)),
                                 averaged_agreement=float(np.mean(np.asarray([LABELS.index(lookup[i]['winner_pred']) for i in ids]) == truth))))
        print(f'Arena {judge}: {agreement:.4%}; published baseline PASS', flush=True)
    rng = np.random.RandomState(args.seed + 1)
    last_decisions = decisions.copy()
    shuffled = [float(np.mean(rng.permutation(last_decisions) == truth)) for _ in range(100)]
    chance = float(np.dot(np.bincount(last_decisions, minlength=3) / len(truth),
                          np.bincount(truth, minlength=3) / len(truth)))
    controls.append(dict(judge=judge, null_expected_from_marginals=chance,
                         null_mean=float(np.mean(shuffled)), null_range=[min(shuffled), max(shuffled)],
                         identical_labels_agreement=AgreementScore.score([DiscretePrediction(x) for x in labels], labels)))
    audit['runtime_seconds'] = time.perf_counter() - started
    audit['bootstrap_replicates'] = args.bootstraps
    audit['excluded_judges'] = [r['judge'] for r in checks if not r['passed']]
    audit['probability_rule'] = 'softmax float64; pipeline clips each probability to [.02,.98] and renormalizes'
    audit['hard_decision_rule'] = 'released strict A or B winner, otherwise tie; two-way forces A/B with equality -> B'
    audit['uncertainty_limits'] = 'question-cluster bootstrap; no voter/session IDs; fixed judges, no training uncertainty'
    write_json(args.output / 'arena_data_audit.json', audit)
    write_json(args.output / 'baseline_reproduction.json', checks)
    write_json(args.output / 'arena_controls.json', controls)
    write_json(args.output / 'arena_results.json', results)
    pd.DataFrame(results).drop(columns='confusion').to_csv(args.output / 'arena_leaderboard.csv', index=False)
    pd.DataFrame(bootstrap_columns).to_csv(args.output / 'arena_bootstraps.csv', index_label='bootstrap_id')
    # Save an order-sensitive ID list; the seed + locked NumPy version recreates the schedule.
    pd.DataFrame({'battle_id': ids, 'prompt_sha256': groups}).to_csv(args.output / 'arena_items.csv', index=False)
    return results, audit


def bayes_oracle(W, labels, exclude, raw=False):
    """Rational enumeration of ordered draws, independent of runtime caches."""
    def likelihood(row, sequence):
        counts = Counter(row)
        value = Fraction(1)
        for draw, label in enumerate(sequence):
            value *= Fraction(counts[label], len(row) - draw)
            counts[label] -= 1
            if value == 0:
                break
        return value
    numerators = [sum((likelihood(row, labels + [label]) for i, row in enumerate(W) if i != exclude), Fraction(0))
                  for label in LABELS]
    total = sum(numerators)
    if total == 0:
        return None
    probabilities = [float(n / total) for n in numerators]
    return probabilities if raw else DiscreteDistributionPrediction(LABELS, probabilities).probabilities


def load_mt(cache, seed):
    sys.path.insert(0, str(ROOT / '.cache/arena-reader'))
    human = pd.read_parquet(cache / 'mt-human.parquet')
    model = pd.read_parquet(cache / 'mt-gpt4.parquet')
    votes, gpt = defaultdict(dict), {}
    duplicate_humans = 0
    for row in human.itertuples():
        key, label = canonical_mt(row)
        if row.judge in votes[key]:
            duplicate_humans += 1
            if votes[key][row.judge] != label:
                raise ValueError('Conflicting duplicate human judgment')
        votes[key][row.judge] = label
    for row in model.itertuples():
        key, label = canonical_mt(row)
        if key in gpt:
            raise ValueError('Duplicate exact model judgment')
        gpt[key] = label
    experts = {key: {j: y for j, y in panel.items() if j.startswith('expert_')}
               for key, panel in votes.items() if key in gpt}
    depth = Counter(len(panel) for panel in experts.values())
    rng = np.random.RandomState(seed)
    selected, metadata = [], []
    for key, panel in sorted(experts.items()):
        if len(panel) < 3:
            continue
        names = sorted(panel)
        chosen = [names[i] for i in rng.choice(len(names), size=3, replace=False)]
        item = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        selected.append([panel[j] for j in chosen])
        metadata.append(dict(item_id=item, question_id=key[0], turn=key[1], model_a=key[2], model_b=key[3],
                             human_judges=chosen, available_experts=len(panel), gpt4=gpt[key]))
    W = pd.DataFrame(selected, index=pd.Index([m['item_id'] for m in metadata], name='item_id'),
                     columns=['human_1', 'human_2', 'human_3'])
    predictions = pd.DataFrame({'GPT-4 (released pairwise judge)': [DiscretePrediction(m['gpt4']) for m in metadata]}, index=W.index)
    # Published MT-Bench comparison is annotation-weighted, not our item-weighted cohort.
    paper = []
    for turn in [1, 2]:
        pairs = [(y, gpt[key]) for key, panel in experts.items() if key[1] == turn for y in panel.values()]
        decisive = [(a, b) for a, b in pairs if a != 'tie' and b != 'tie']
        paper.append(dict(turn=turn, expert_annotations=len(pairs),
                          agreement_with_ties=sum(a == b for a, b in pairs) / len(pairs),
                          decisive_annotations=len(decisive),
                          agreement_non_ties=sum(a == b for a, b in decisive) / len(decisive)))
    for row, expected_n, expected_decisive in zip(paper, [1343, 1325], [859, 864]):
        if (row['expert_annotations'] != expected_n or row['decisive_annotations'] != expected_decisive
                or round(row['agreement_with_ties'] * 100) != 66
                or round(row['agreement_non_ties'] * 100) != 85):
            raise AssertionError('MT-Bench Table 5 expert baseline did not reproduce')
    report = dict(raw_human_annotations=len(human), raw_gpt4_annotations=len(model),
                  human_items=len(votes), content_matched_human_items=sum(k in gpt for k in votes),
                  duplicate_human_records=duplicate_humans, expert_depth_histogram=dict(sorted(depth.items())),
                  selected_items=len(W), selected_humans_per_item=3,
                  unique_question_clusters=len(set(m['question_id'] for m in metadata)),
                  reference_policy='one held-out human; exhaustive predictor subsets; equal weight per item',
                  status='exploratory released expert annotations; initial-versus-feedback-revised vote provenance unresolved',
                  probability_limitation='GPT-4 split contains hard verdicts, no calibrated class probabilities',
                  paper_table5_reproduced=True,
                  paper_comparison=paper)
    return W, predictions, metadata, report


def run_mt(args):
    W, predictions, metadata, audit = load_mt(args.cache, args.seed)
    samples = cluster_samples([m['question_id'] for m in metadata], args.bootstraps, args.seed)
    print(f'MT-Bench: {len(W)} exact items, three distinct experts each, {audit["unique_question_clusters"]} question clusters', flush=True)
    runs, equivalences, oracle_checks = {}, {}, []
    for name, factory in [('Bayesian', AnonymousBayesianCombiner), ('Frequency', FrequencyCombiner)]:
        analysis = pipeline(W, predictions, factory(), AgreementScore(), samples, seed=args.seed)
        runs[name] = analysis
        scalar = factory(LABELS, W)
        array = W.to_numpy()
        expected = {}
        for k in range(3):
            subset_terms = []
            for subset in combinations(range(3), k):
                remaining = [j for j in range(3) if j not in subset]
                terms = []
                for i, row in enumerate(array):
                    observed = [(j, row[j]) for j in subset]
                    pred = scalar.combine(LABELS, observed, W=array, item_id=i)
                    if pred is None:
                        raise AssertionError('Unsupported prediction would change the paired cohort')
                    if name == 'Bayesian':
                        independent = bayes_oracle(array, [row[j] for j in subset], i)
                        if independent is None or not np.allclose(pred.probabilities, independent, rtol=0, atol=1e-14):
                            raise AssertionError('Bayesian probability differs from rational enumeration')
                    terms.append(float(Fraction(sum(row[j] == pred.value for j in remaining), len(remaining))))
                subset_terms.append(np.asarray(terms))
            expected[k] = [sum(ordered_mean(t[s]) for t in subset_terms) / len(subset_terms) for s in samples]
        exact_frame(analysis.expert_power_curve.df, pd.DataFrame(expected, dtype=float))
        ai_terms = np.asarray([sum(y == p.value for y in row) / 3 for row, p in zip(array, predictions.iloc[:, 0])])
        ai_expected = pd.DataFrame({predictions.columns[0]: [ordered_mean(ai_terms[s]) for s in samples]})
        exact_frame(analysis.classifier_scores.df, ai_expected)
        oracle_checks.append(dict(combiner=name, probability_oracle='rational ordered draws' if name == 'Bayesian' else 'frequency counts',
                                  exact_curve_cells=int(analysis.expert_power_curve.df.size),
                                  exact_classifier_cells=len(samples), all_predictions_supported=True))
        entries = [supported_equivalence(float(ai_expected.iloc[i, 0]), analysis.expert_power_curve.df.iloc[i].to_dict())
                   for i in range(len(samples))]
        equivalences[name] = equivalence_summary(entries)
        pd.DataFrame(entries).to_csv(args.output / f'mt_{name.lower()}_equivalences.csv', index_label='sample_id')
        analysis.expert_power_curve.df.to_csv(args.output / f'mt_{name.lower()}_power.csv', index_label='sample_id')
        print(f'{name} human curve: {analysis.expert_power_curve.df.iloc[0].to_dict()}; GPT-4 {ai_expected.iloc[0,0]:.4f}; {entries[0]}', flush=True)
    # Exact process-count, memory-block, reordered-index, and persistence checks on a fixed schedule.
    reference = pipeline(W, predictions, AnonymousBayesianCombiner(), AgreementScore(), samples[:6], seed=args.seed)
    parallel = pipeline(W, predictions.iloc[::-1], AnonymousBayesianCombiner(), AgreementScore(), samples[:6],
                        procs=2, memory=.01, seed=args.seed)
    exact_frame(reference.expert_power_curve.df, parallel.expert_power_curve.df)
    exact_frame(reference.classifier_scores.df, parallel.classifier_scores.df)
    for i, row in enumerate(reference.expert_power_curve.df.to_numpy()):
        if not np.isfinite(row).all():
            raise AssertionError('Unexpected missing curve')
    audit['worker_and_block_exact_parity'] = True
    audit['frequency_k0_note'] = 'Built-in Frequency uses the full-dataset prior; Bayesian primary uses leave-one-item-out throughout.'
    audit['human_population'] = 'Released external experts only, at least three votes per exact item; authors excluded'
    # Store a replayable cluster schedule without object arrays or pickle.
    np.savez_compressed(args.output / 'mt_sample_schedule.npz',
                        indices=np.concatenate(samples), offsets=np.cumsum([0] + [len(s) for s in samples]))
    runs['Bayesian'].classifier_scores.df.to_csv(args.output / 'mt_classifier_scores.csv', index_label='sample_id')
    W.to_csv(args.output / 'mt_human_ratings.csv')
    pd.DataFrame(metadata).to_csv(args.output / 'mt_items.csv', index=False)
    write_json(args.output / 'mt_data_audit.json', audit)
    write_json(args.output / 'mt_oracle_checks.json', oracle_checks)
    write_json(args.output / 'mt_equivalence_summary.json', equivalences)
    return runs, audit, equivalences


def plot_results(output, cache):
    """Render the current paper-audit figures through the single plot implementation."""
    from research.arena_baseline import main as build_baseline
    from research.arena_figures import main as render
    output = Path(output).resolve()
    figures = (ROOT / 'benchmarks/results/paper-audit'
               if output == ROOT / 'benchmarks/results/arena-validation'
               else output / 'figures')
    prior = figures / 'arena_zero_boundary.json'
    current = False
    if prior.exists():
        record = json.loads(prior.read_text())
        current = all(digest(output / name) == value
                      for name, value in record.get('input_hashes', {}).items())
        current = current and bool(record.get('input_hashes'))
    if not current:
        build_baseline(source=output, cache=cache, output=figures)
    render(source=output, output=figures)
    print(f'Current graphs saved to {figures}', flush=True)


def run_validation(args):
    started = time.perf_counter()
    arena, arena_audit = run_arena(args)
    runs, mt_audit, equivalences = run_mt(args)
    sources = json.loads((args.cache / 'sources.json').read_text())
    for source in sources:
        source['sha256'] = digest(args.cache / source['local'])
        source['bytes'] = (args.cache / source['local']).stat().st_size
    write_json(args.output / 'manifest.json', dict(sources=sources, seed=args.seed, bootstrap_replicates=args.bootstraps,
               python=sys.version, numpy=np.__version__, pandas=pd.__version__,
               revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), *sorted((ROOT / 'surveyequivalence').glob('*.py'))]},
               elapsed_seconds=time.perf_counter() - started,
               result_status='Arena agreement validation and exploratory MT-Bench equivalence; full mathematical audit not certified'))
    plot_results(args.output, args.cache)
    print(f'Results saved to {args.output}', flush=True)


def replot(args):
    """Re-render the latest figures without rewriting saved numerical results."""
    plot_results(args.output, args.cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / '.cache/arena-validation')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/results/arena-validation')
    parser.add_argument('--download-config', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--bootstraps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1729)
    args = parser.parse_args()
    if args.download_config:
        download_config(args.cache)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        replot(args)
        return
    run_validation(args)


if __name__ == '__main__':
    main()
