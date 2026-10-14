"""Pinned inputs and audited binary votes for the Condorcet jury experiment.

Run with --download once to obtain public source files; normal runs are offline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.cache/condorcet-python'))
import yaml

CACHE = ROOT / '.cache/condorcet-data'
OUT = ROOT / 'benchmarks/results/condorcet-jury-equivalence'
SEARCH = ROOT / 'benchmarks/results/condorcet-dataset-search/inspection.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(download=False):
    inspected = json.loads(SEARCH.read_text())
    inputs = []
    for entry in inspected['datasets']:
        for f in entry.get('files', []):
            inputs.append(dict(name=entry['id'] + '_' + f['url'].split('/')[-1],
                               url=f['url'], sha256=f['sha256']))
        if entry['id'] == 'city_comparisons':
            inputs.append(dict(name=entry['file_name'], url=entry['download'], sha256=entry['sha256']))
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in inputs:
        path = CACHE / f['name']
        if not path.exists() and download:
            with urllib.request.urlopen(f['url'], timeout=90) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != f['sha256']:
                raise ValueError('Downloaded source differs from inspected version: ' + f['name'])
            path.write_bytes(content)
        if not path.exists() or digest(path) != f['sha256']:
            raise ValueError('Missing or changed input: ' + str(path))
        f['bytes'] = path.stat().st_size
    return inputs


def city_votes(raw):
    """Canonicalize screen-position answers; never turn absent responses into votes."""
    counts = raw.groupby('Subject').size()
    retained = counts.index[counts == 1000]
    d = raw[raw.Subject.isin(retained)].copy()
    assert d.groupby('Subject').Session.nunique().eq(2).all()
    assert (d.CityResponse.fillna('MISSING') == d.DecisionResponse.fillna('MISSING')).all()
    missing = d.CityResponse.isna()
    assert d.loc[missing, 'DecisionAcc'].eq(0).all()
    d = d[~missing].copy()
    assert d.CityResponse.isin(['CityLeftBox', 'CityRightBox']).all()
    left = d.CityLeft.str.strip()
    right = d.CityRight.str.strip()
    assert (left != right).all() and (d.CityLeftPop != d.CityRightPop).all()
    a = np.where(left < right, left, right)
    b = np.where(left < right, right, left)
    chosen = np.where(d.CityResponse == 'CityLeftBox', left, right)
    correct = np.where(d.CityLeftPop > d.CityRightPop, left, right)
    assert np.array_equal(chosen == correct, d.DecisionAcc.to_numpy(bool))
    assert np.array_equal(d.CorrectResponse == 'CityLeftBox', d.CityLeftPop > d.CityRightPop)
    votes = pd.DataFrame(dict(dataset='cities', item_id=[x + ' | ' + y for x, y in zip(a, b)],
        rater_id=d.Subject.astype(str), vote=(chosen == b).astype(int), truth=(correct == b).astype(int),
        group='city pairs', city_a=a, city_b=b, confidence=pd.to_numeric(d.Conf, errors='coerce')))
    audit = dict(raw_rows=len(raw), raw_raters=int(raw.Subject.nunique()),
        retained_raters=len(retained), retained_trials=int(counts.loc[retained].sum()),
        missing_decisions_excluded=int(missing.sum()), valid_votes=len(votes),
        rule='Keep subjects with 1000 trials over two sessions; exclude absent decisions; no confidence filter.',
        orientation='Vote/truth 1 means lexicographically second city; use recorded historical populations.',
        dependence='Repeated city identities; main curve is fixed-item descriptive, without item-iid intervals.')
    return votes, audit


def prepare(download=False):
    inputs = sources(download)
    frames, audit = [], {}
    labels = yaml.safe_load((CACHE / 'bluebirds_labels.yaml').read_text())
    truth = yaml.safe_load((CACHE / 'bluebirds_gt.yaml').read_text())
    frames.append(pd.DataFrame([dict(dataset='bluebirds', item_id=str(i), rater_id=str(w),
        vote=int(v), truth=int(truth[i]), group='truth ' + str(int(truth[i])))
        for w, responses in labels.items() for i, v in responses.items()]))
    audit['bluebirds'] = dict(truth_source='Original gt.yaml, not derived from crowd vote.',
        task='Indigo Bunting versus Blue Grosbeak; original Boolean label coding retained.')
    classes = yaml.safe_load((CACHE / 'ducks_classes.yaml').read_text())
    arr = np.loadtxt(CACHE / 'ducks_labels.txt', skiprows=1, dtype=int)
    class_map = classes['labels']
    names = {0: 'Mallard', 1: 'Canada Goose', 2: 'American Black Duck', 3: 'Non Bird', 4: 'Red-necked Grebe'}
    frames.append(pd.DataFrame([dict(dataset='ducks', item_id=str(i), rater_id=str(w), vote=int(v),
        truth=int(class_map[int(i)] in [0, 2]), group=names[class_map[int(i)]]) for i, w, v in arr]))
    audit['ducks'] = dict(truth_source='Original species classes; Mallard and American Black Duck are positive.',
                         species_names=names)
    cities, audit['cities'] = city_votes(pd.read_csv(CACHE / 'S3_CityTask.csv'))
    frames.append(cities)
    path = ROOT / '.cache/sports-survey-data/nba_complete_responses.jsonl'
    nba = pd.read_json(path, lines=True)
    old = json.loads((ROOT / 'benchmarks/results/sports-survey-equivalence/results.json').read_text())
    assert digest(path) == old['prepared_sha256']
    frames.append(pd.DataFrame(dict(dataset='nba', item_id=nba.game_id, rater_id=nba.participant_id.astype(str),
        vote=nba.pick_code - 1, truth=nba.winner_code - 1, group=nba.study_wave, confidence=nba.confidence_ordinal)))
    inputs.append(dict(name=str(path.relative_to(ROOT)), sha256=digest(path), url=old['source_url']))
    audit['nba'] = dict(rule='Existing complete three-answer cohort; game codes retained.',
                       truth_source='Actual winner code from released workbook.', waves=11)
    votes = pd.concat(frames, ignore_index=True)
    assert not votes.duplicated(['dataset', 'item_id', 'rater_id']).any()
    assert votes.vote.isin([0, 1]).all() and votes.truth.isin([0, 1]).all()
    assert votes.groupby(['dataset', 'item_id']).truth.nunique().eq(1).all()
    votes['correct'] = (votes.vote == votes.truth).astype(int)
    items = votes.groupby(['dataset', 'item_id'], sort=True).agg(
        truth=('truth', 'first'), n=('vote', 'size'), c=('correct', 'sum'), group=('group', 'first'),
        city_a=('city_a', 'first'), city_b=('city_b', 'first')).reset_index()
    expected = {'nba': (465, 113, 9788, 10, 61), 'bluebirds': (108, 39, 4212, 39, 39),
                'ducks': (240, 53, 9600, 40, 40), 'cities': (1000, 99, 98787, 97, 99)}
    for key, d in votes.groupby('dataset'):
        it = items[items.dataset == key]
        actual = (len(it), d.rater_id.nunique(), len(d), it.n.min(), it.n.max())
        assert actual == expected[key], (key, actual)
        audit[key].update(dict(items=len(it), raters=int(d.rater_id.nunique()), votes=len(d),
            min_depth=int(it.n.min()), max_depth=int(it.n.max()),
            truth_counts={str(k): int(v) for k, v in it.truth.value_counts().items()}))
    OUT.mkdir(parents=True, exist_ok=True)
    votes.to_csv(CACHE / 'prepared_votes.csv', index=False)
    items.to_csv(OUT / 'items.csv', index=False)
    result = dict(sources=inputs, datasets=audit, prepared_sha256=digest(CACHE / 'prepared_votes.csv'),
        license_note='Source provenance retained; CUBAM software license does not establish image rights; city OSF license unspecified.',
        input_adapter_sha256=digest(__file__))
    (OUT / 'data_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(audit, indent=2))
    return votes, items, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    prepare(parser.parse_args().download)
