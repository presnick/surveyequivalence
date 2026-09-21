"""Reproducible, offline DICES safety-flag equivalence experiment.

See docs/dices_validation_protocol.md. Downloads are not executed by this script.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
from surveyequivalence import (AnalysisPipeline, AnonymousBayesianCombiner,
                              AgreementScore, CrossEntropyScore,
                              DiscreteDistributionPrediction)
from research.arena import (cluster_samples, digest, write_json,
                                        supported_equivalence, equivalence_summary)

REVISION = '5f5e6bfd859ebdd25c8e558e65c06a6ce27a0bad'
TARGET = 'people with education college degree or higher'
LABELS = ['nonflag', 'flag']  # First label wins a 50% tie.
EXCLUDED = set(map(int, '''297514565398139 297515609163939 297515750682315
297515617432733 297541515566649 297541515769980 297515629971478
297059995361243 297541522412126 297540556928761 297541321453321
297540562350921 297540983991638 297060365288109 297514543980607
297515729806999 297541271027233 296709611112092 296709543131761'''.split()))


def human_panel(raw, comments, quality=True):
    """Validate complete item identity before translating any label."""
    if raw.duplicated(['item_id', 'rater_id']).any():
        raise ValueError('Duplicate item/rater vote')
    for col in ['context', 'response']:
        if raw.groupby('item_id')[col].nunique(dropna=False).max() != 1:
            raise ValueError('Conflicting item content')
    items = raw.drop_duplicates('item_id').reset_index(drop=True)
    comments = comments.set_index('comment_id', verify_integrity=True)
    if set(comments.index) != set(range(len(items))):
        raise ValueError('Comment IDs do not cover the original item order')
    expected = items.context.str.replace('LAMDA:', 'AI:', regex=False) + '\nAI: ' + items.response
    if not np.array_equal(expected.to_numpy(), comments.loc[items.index, 'comment'].to_numpy()):
        raise ValueError('Exact conversation join failed')
    if not raw.Q_overall.isin(['Yes', 'No', 'Unsure']).all():
        raise ValueError('Unknown human label')
    for col in ['rater_education', 'rater_gender', 'rater_race', 'rater_age']:
        if raw.groupby('rater_id')[col].nunique(dropna=False).max() != 1:
            raise ValueError('Conflicting rater demographics')
    cohort = raw[raw.rater_education == 'College degree or higher'].copy()
    if quality:
        cohort = cohort[~cohort.rater_id.isin(EXCLUDED)].copy()
    cohort['label'] = np.where(cohort.Q_overall == 'Yes', 'flag', 'nonflag')
    W = cohort.pivot(index='item_id', columns='rater_id', values='label').loc[items.item_id]
    if W.empty or W.isna().any().any():
        raise ValueError('Expected a complete fixed human panel')
    W.columns = W.columns.map(str)
    meta = pd.DataFrame({'item_id': items.item_id, 'comment_id': items.index,
                         'context_group': [hashlib.sha256(s.encode()).hexdigest() for s in items.context],
                         'conversation_sha256': [hashlib.sha256(s.encode()).hexdigest() for s in expected]})
    return W, meta, dict(items=len(W), raters=W.shape[1], votes=int(W.size),
                        context_clusters=items.context.nunique(),
                        original_labels=dict(Counter(cohort.Q_overall)),
                        exact_conversation_matches=len(W), quality_filter=quality)


def single_generation(records, item_count):
    selected = [r for r in records if r.get('target_group') == TARGET and r.get('generation_index') == 0]
    if not selected:
        raise ValueError('No requested target/generation')
    names = {r['model_name'] for r in selected}
    if len(names) != 1:
        raise ValueError('Mixed model configurations')
    ids = [r['comment_id'] for r in selected]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate model/item/generation')
    if any(i not in range(item_count) for i in ids):
        raise ValueError('Unknown AI item')
    values = np.full(item_count, np.nan)
    for record in selected:
        if record.get('dataset') != 'dices' or record.get('experiment_group') != 'main':
            raise ValueError('Unexpected source experiment')
        p = record.get('parsed_percentage')
        if isinstance(p, (float, int)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 100:
            values[record['comment_id']] = p / 100
    return next(iter(names)), values, dict(records=len(selected), valid=int(np.isfinite(values).sum()),
                                         missing_or_invalid=int((~np.isfinite(values)).sum()))


def predictions_from_percentages(percentages, variants=True):
    columns = {}
    for model in percentages:
        for epsilon in ([0, .005, .01, .02] if variants else [0]):
            p = percentages[model].clip(epsilon, 1-epsilon)
            key = f'{model}|clip={epsilon:g}'
            columns[key] = [DiscreteDistributionPrediction(LABELS, [1-v, v], extreme_cutoff=0) for v in p]
    return pd.DataFrame(columns, index=percentages.index)


def independent_scores(W, predictions, scores, metric, samples):
    """Separate per-reference enumeration, with no production scorer helpers."""
    largest_error = 0.
    for name in predictions:
        terms = []
        for i, prediction in enumerate(predictions[name]):
            row = W.iloc[i].tolist()
            if metric == 'agreement':
                terms.append(sum(prediction.value == y for y in row) / len(row))
            else:
                total = 0.
                for label in LABELS:
                    weight = row.count(label) / len(row)
                    probability = prediction.label_probability(label)
                    if weight:
                        total += weight * math.log2(probability) if probability else -math.inf
                terms.append(total)
        for j, indices in enumerate(samples):
            expected = sum(terms[i] for i in indices) / len(indices)
            actual = scores.iloc[j][name]
            if math.isinf(expected):
                if actual != expected:
                    raise AssertionError('Infinite score was not preserved')
            else:
                error = abs(actual - expected)
                largest_error = max(largest_error, error)
                if error > 2e-14:
                    raise AssertionError(f'Independent score mismatch: {name}: {error}')
    return largest_error


def run_analysis(W, predictions, samples, seed, k_max, procs=1, multiplier=10):
    return AnalysisPipeline(W, classifier_predictions=predictions,
        combiner=AnonymousBayesianCombiner(), scorer=CrossEntropyScore(),
        allowable_labels=LABELS, anonymous_raters=True, paper_mode=True,
        item_samples=[W.index[s] for s in samples], random_state=seed,
        max_K=k_max+1, max_rater_subsets=multiplier*W.shape[1],
        procs=procs, verbosity=0)


def summarize(run, population, metric):
    rows, detail = [], {}
    for name in run.classifier_scores.df:
        model, clipping = name.rsplit('|clip=', 1)
        peers = [c for c in run.classifier_scores.df if c.endswith('|clip='+clipping)]
        bootstrap_scores = run.classifier_scores.df[peers].iloc[1:]
        ranks = bootstrap_scores.rank(axis=1, ascending=False, method='average')[name]
        rank_low,rank_high = np.quantile(ranks,[.025,.975])
        entries = [supported_equivalence(score, curve.to_dict()) for score, (_, curve) in
                   zip(run.classifier_scores.df[name], run.expert_power_curve.df.iterrows())]
        summary = equivalence_summary(entries)
        detail[name] = summary
        low, high = summary['interval'] or [None, None]
        actual = summary['actual']
        rows.append(dict(population=population, metric=metric, model=model, clipping=float(clipping),
                         score=run.classifier_scores.df.iloc[0][name],
                         baseline=run.expert_power_curve.df.iloc[0, 0],
                         status=actual['status'], equivalence=actual['estimate'], bound=actual['bound'],
                         ci_low=low, ci_high=high,
                         descriptive_rank_low=rank_low, descriptive_rank_high=rank_high,
                         above_range_bootstraps=summary['bootstrap_status_counts'].get('above_range', 0)))
    return rows, detail


def save_run(run, directory):
    directory.mkdir(parents=True, exist_ok=True)
    run.expert_power_curve.df.to_csv(directory/'human_curve.csv', index_label='sample_id')
    run.classifier_scores.df.to_csv(directory/'judge_scores.csv', index_label='sample_id')
    write_json(directory/'partitions.json', {str(k): [[list(map(int,b)),int(r)] for b,r in v]
                                            for k,v in run.paper_partitions.items()})


def figures(table, runs, output):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    main = table[(table.population=='quality') & (table.metric=='log2') & (table.clipping==.01)]
    main = main.sort_values(['equivalence','score'], ascending=False).reset_index(drop=True)
    fig, (left,right) = plt.subplots(1,2,figsize=(14, max(6.2,len(main)*.38+1.7)), gridspec_kw={'width_ratios':[1.25,1]})
    for i,row in main.iterrows():
        x = row.equivalence if row.status=='crossing' else row.bound
        if pd.notna(x):
            left.scatter(x,i,color='#2166ac',marker='o' if row.status=='crossing' else '>',zorder=3)
            if pd.notna(row.ci_low) and pd.notna(row.ci_high):
                left.plot([row.ci_low,row.ci_high],[i,i],color='#2166ac',lw=2)
            left.annotate(f'{x:.2f}' if row.status=='crossing' else f'>{x:g}',(x,i),xytext=(6,7),textcoords='offset points',fontsize=8)
    left.set_yticks(range(len(main)),main.model);left.invert_yaxis()
    left.set_xlabel('Human-rater equivalence (95% conditional bootstrap interval)')
    left.set_title('One released AI prediction, clipped to 1–99%')
    left.grid(axis='x',alpha=.2);left.set_xlim(-.12,max(1.,float(main.equivalence.max())*1.2))
    run = runs['quality_log2'];curve=run.expert_power_curve.df
    x=curve.columns.astype(int);low,high=np.quantile(curve.iloc[1:].to_numpy(),[.025,.975],axis=0)
    right.fill_between(x,low,high,alpha=.15,color='black')
    right.plot(x,curve.iloc[0],'-o',color='black',label='Bayesian human panel')
    for color, (_, row) in zip(['#2166ac','#d95f02','#1b9e77'], main.head(3).iterrows()):
        right.axhline(row.score,ls='--',lw=1.3,label=row.model,color=color)
    right.set(xlabel='Number of input human ratings (k)',ylabel='Expected log₂ score (higher is better)',title='Human curve and leading AI scores')
    right.legend(fontsize=8,loc='lower right');right.grid(alpha=.2)
    fig.suptitle('DICES safety-flag prediction • 350 conversations • college-educated human reference group',fontsize=13)
    fig.text(.5,.025,'Equivalence is task-, population- and scoring-specific. Zero means no advantage over the k=0 prior.\nIntervals condition on the human pool, fitted combiner, sampled panels and generation 0; ranks are descriptive.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.94])
    for ext in ['png','svg','pdf']:fig.savefig(output/f'dices_leaderboard.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,max(5.5,len(main)*.35+1.6)))
    for j,(label,population,clip) in enumerate([('0.5% floor','quality',.005),('1% floor','quality',.01),('2% floor','quality',.02),('Full demographic pool, 1%','all',.01)]):
        t=table[(table.population==population)&(table.metric=='log2')&(table.clipping==clip)].set_index('model').loc[main.model]
        ax.scatter(t.equivalence,np.arange(len(main))+(j-1.5)*.12,label=label,s=23)
    ax.set_yticks(range(len(main)),main.model);ax.invert_yaxis();ax.set_xlabel('Human-rater equivalence');ax.grid(axis='x',alpha=.2)
    ax.set_title('Sensitivity to probability clipping and human quality filtering')
    ax.legend(fontsize=8);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(output/f'dices_sensitivity.{ext}',dpi=180)
    plt.close(fig)


def refresh_reports(output):
    runs={};rows=[]
    for key,population,metric in [('quality_log2','quality','log2'),('all_log2','all','log2'),('quality_agreement','quality','agreement')]:
        curve=pd.read_csv(output/key/'human_curve.csv',index_col=0,float_precision='round_trip')
        curve.columns=curve.columns.astype(int)
        scores=pd.read_csv(output/key/'judge_scores.csv',index_col=0,float_precision='round_trip')
        run=SimpleNamespace(expert_power_curve=SimpleNamespace(df=curve),classifier_scores=SimpleNamespace(df=scores))
        runs[key]=run;r,detail=summarize(run,population,metric);rows.extend(r)
        write_json(output/key/'equivalences.json',detail)
    table=pd.DataFrame(rows);table.to_csv(output/'leaderboard.csv',index=False);figures(table,runs,output)
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['reporting_script_sha256']=digest(Path(__file__))
    manifest['report_input_sha256']={str(p.relative_to(output)):digest(p) for key in runs for p in [output/key/'human_curve.csv',output/key/'judge_scores.csv']}
    write_json(output/'manifest.json',manifest)


def download_config():
    """Use checked-in provenance to describe public downloads; never execute them."""
    from urllib.parse import quote
    manifest=json.loads((ROOT/'benchmarks/results/dices-validation/manifest.json').read_text())
    cache=ROOT/'.cache/dataset-search';ai=ROOT/'.cache/dices-validation'
    cache.mkdir(parents=True,exist_ok=True);ai.mkdir(parents=True,exist_ok=True)
    base=f'https://raw.githubusercontent.com/shasanamin/llm-perspective-taking/{REVISION}/'
    entries=[(base+quote(v['path']),ai/name) for name,v in manifest['ai_sources'].items()]
    entries += [(base+remote,cache/local) for remote,local in [
        ('data/dices/original/dices350_raw.csv','dices-raw.csv'),
        ('data/dices/original/dices_comments.csv','dices-comments.csv'),
        ('src/datasets.py','dices-datasets-source.py'),('src/prompts.py','dices-prompts.py')]]
    entries += [(f'https://api.github.com/repos/shasanamin/llm-perspective-taking/git/trees/{REVISION}?recursive=1',cache/'perspective-tree.json'),
        ('https://raw.githubusercontent.com/google-research-datasets/dices-dataset/main/350/README.md',cache/'dices-original-readme.md')]
    lines=[]
    for url,path in entries:
        lines.extend([f'url = "{url}"',f'output = "{path}"'])
    (ai/'download.conf').write_text('\n'.join(lines)+'\n')
    print(ai/'download.conf')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstraps',type=int,default=500)
    parser.add_argument('--max-k',type=int,default=5)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/dices-validation')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--download-config',action='store_true')
    args=parser.parse_args()
    if args.download_config:download_config();return
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    if args.report_only:refresh_reports(output);return
    started=time.perf_counter();cache=ROOT/'.cache/dataset-search';ai_cache=ROOT/'.cache/dices-validation'
    raw=pd.read_csv(cache/'dices-raw.csv');comments=pd.read_csv(cache/'dices-comments.csv')
    W,meta,audit=human_panel(raw,comments)
    if len(W)!=350:raise ValueError('Unexpected dataset size')
    meta.to_csv(output/'item_manifest.csv',index=False)
    arrays={};coverage={};sources={}
    tree=json.loads((cache/'perspective-tree.json').read_text())['tree']
    expected={Path(e['path']).name:e for e in tree if e['path'].startswith('data/llm_annotations/dices/main/') and e['path'].endswith('.jsonl')}
    for filename,entry in sorted(expected.items()):
        path=ai_cache/filename
        content=path.read_bytes()
        git_blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if len(content)!=entry['size'] or git_blob!=entry['sha']:raise ValueError(f'Incomplete or changed download: {path}')
        model,values,info=single_generation([json.loads(l) for l in content.splitlines()],len(W))
        if model in coverage:raise ValueError('Duplicate model name')
        coverage[model]=info;sources[filename]=dict(sha256=digest(path),git_blob_sha1=git_blob,path=entry['path'])
        if info['valid']==len(W):arrays[model]=values
    write_json(output/'coverage.json',coverage)
    if not arrays:raise ValueError('No complete model configurations')
    percentages=pd.DataFrame(arrays,index=W.index);percentages.to_csv(output/'judge_probabilities.csv')
    samples=cluster_samples(meta.context_group,args.bootstraps,args.seed)
    np.savez_compressed(output/'sample_schedule.npz',indices=np.concatenate(samples),offsets=np.cumsum([0]+[len(x) for x in samples]))
    runs={};rows=[];checks={};populations={}
    for population,quality in [('quality',True),('all',False)]:
        W,_,pop_audit=human_panel(raw,comments,quality=quality);populations[population]=pop_audit
        W.to_csv(output/f'human_ratings_{population}.csv')
        predictions=predictions_from_percentages(percentages)
        print(f'Running {population}: {W.shape}, {len(arrays)} complete judges, {args.bootstraps} bootstraps',flush=True)
        run=run_analysis(W,predictions,samples,args.seed,args.max_k)
        if not np.isfinite(run.expert_power_curve.df.to_numpy()).all():raise ValueError('Human curve has nonfinite scores')
        key=f'{population}_log2';runs[key]=run;save_run(run,output/key)
        checks[key]=independent_scores(W,predictions,run.classifier_scores.df,'log2',samples[:4])
        r,detail=summarize(run,population,'log2');rows.extend(r);write_json(output/key/'equivalences.json',detail)
        if quality:
            options=dict(W=W,classifier_predictions=predictions_from_percentages(percentages,False),
                combiner=AnonymousBayesianCombiner(),scorer=AgreementScore(),allowable_labels=LABELS,
                anonymous_raters=True,paper_mode=True,item_samples=[W.index[s] for s in samples],
                random_state=args.seed,max_K=args.max_k+1,max_rater_subsets=10*W.shape[1],procs=1,verbosity=0)
            agreement=AnalysisPipeline(**options);runs['quality_agreement']=agreement;save_run(agreement,output/'quality_agreement')
            checks['quality_agreement']=independent_scores(W,options['classifier_predictions'],agreement.classifier_scores.df,'agreement',samples[:4])
            r,detail=summarize(agreement,population,'agreement');rows.extend(r);write_json(output/'quality_agreement/equivalences.json',detail)
            # A bounded replay tests execution parity, not a second full experiment.
            replay_samples=samples[:4];replay_predictions=predictions.iloc[:,:1]
            serial=run_analysis(W,replay_predictions,replay_samples,args.seed,1,multiplier=1)
            parallel=run_analysis(W,replay_predictions,replay_samples,args.seed,1,procs=2,multiplier=1)
            pd.testing.assert_frame_equal(serial.expert_power_curve.df,parallel.expert_power_curve.df,check_exact=True)
            pd.testing.assert_frame_equal(serial.classifier_scores.df,parallel.classifier_scores.df,check_exact=True)
    table=pd.DataFrame(rows);table.to_csv(output/'leaderboard.csv',index=False)
    figures(table,runs,output)
    source_paths=[cache/'dices-raw.csv',cache/'dices-comments.csv',cache/'dices-datasets-source.py',cache/'dices-prompts.py',cache/'dices-original-readme.md',cache/'perspective-tree.json']
    write_json(output/'manifest.json',dict(revision=REVISION,target=TARGET,populations=populations,
        generation_index=0,bootstraps=args.bootstraps,seed=args.seed,k_max=args.max_k,
        source_files={p.name:digest(p) for p in source_paths},ai_sources=sources,
        independent_score_max_absolute_errors=checks,independent_score_tolerance=2e-14,
        exact_serial_parallel_replay=True,replay_scope='k=0,1; 4 samples; first raw judge; one subset/reference',
        seconds=time.perf_counter()-started,protocol_sha256=digest(ROOT/'docs/dices_validation_protocol.md'),
        implementation_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),*sorted((ROOT/'surveyequivalence').glob('*.py'))]}))
    print(table[(table.population=='quality')&(table.metric=='log2')&(table.clipping==.01)].sort_values('score',ascending=False).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
