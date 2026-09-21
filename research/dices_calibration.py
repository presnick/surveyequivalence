"""Exploratory held-out calibration follow-up; does not replace the primary run."""
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from research.dices import (predictions_from_percentages,run_analysis,save_run,summarize,
    independent_scores,LABELS)
from research.arena import cluster_samples,digest,write_json


def fit_calibrator(probability,human_fraction):
    """Convex binomial log loss, nonnegative slope; all fitting uses training rows."""
    p=np.asarray(probability,dtype=float);y=np.asarray(human_fraction,dtype=float)
    if p.shape!=y.shape or p.ndim!=1 or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all() or np.any((p<0)|(p>1)|(y<0)|(y>1)):
        raise ValueError('Expected finite aligned probabilities')
    p=np.clip(p,.005,.995);x=np.log(p/(1-p))
    def objective(theta):
        z=theta[0]+theta[1]*x
        residual=expit(z)-y
        return float(np.mean(np.logaddexp(0,z)-y*z)),np.array([residual.mean(),np.mean(residual*x)])
    prior=np.clip(y.mean(),1e-4,1-1e-4)
    fit=minimize(objective,[math.log(prior/(1-prior)),0.],jac=True,
                 bounds=[(-10.,10.),(0.,10.)],method='L-BFGS-B',
                 options={'ftol':1e-14,'gtol':1e-10,'maxiter':1000})
    if not fit.success:raise ValueError(f'Calibration optimization failed: {fit.message}')
    null_loss=objective([math.log(prior/(1-prior)),0.])[0]
    if fit.fun>null_loss+1e-12:raise AssertionError('Fit worse than included constant predictor')
    return dict(intercept=float(fit.x[0]),slope=float(fit.x[1]),training_loss=float(fit.fun),
                constant_training_loss=null_loss,iterations=int(fit.nit))


def apply_calibrator(probability,parameters):
    p=np.clip(np.asarray(probability,dtype=float),.005,.995)
    return expit(parameters['intercept']+parameters['slope']*np.log(p/(1-p)))


def group_split(groups,seed=1729):
    unique=np.array(sorted(set(groups)));rng=np.random.RandomState(seed)
    shuffled=unique[rng.permutation(len(unique))]
    train_groups=set(shuffled[:int(.3*len(unique))])
    training=np.array([g in train_groups for g in groups])
    return training,~training


def main():
    source=ROOT/'benchmarks/results/dices-validation';out=ROOT/'benchmarks/results/dices-calibration';out.mkdir(parents=True,exist_ok=True)
    W=pd.read_csv(source/'human_ratings_quality.csv',index_col=0)
    p=pd.read_csv(source/'judge_probabilities.csv',index_col=0,float_precision='round_trip').loc[W.index]
    meta=pd.read_csv(source/'item_manifest.csv').set_index('item_id').loc[W.index]
    train,test=group_split(meta.context_group)
    if set(meta.context_group[train])&set(meta.context_group[test]):raise AssertionError('Group leakage')
    parameters={};calibrated={};human_fraction=W.eq('flag').mean(axis=1)
    for model in p:
        fit=fit_calibrator(p[model][train],human_fraction[train]);parameters[model]=fit
        calibrated[model]=apply_calibrator(p[model][test],fit)
    calibrated=pd.DataFrame(calibrated,index=W.index[test]);predictions=predictions_from_percentages(calibrated,False)
    raw=predictions_from_percentages(p.loc[W.index[test]])
    predictions=pd.concat([predictions,raw[[c for c in raw if c.endswith('|clip=0.01')]]],axis=1)
    samples=cluster_samples(meta.context_group[test],500,1729)
    print(f'Calibration: {int(train.sum())} train items; {int(test.sum())} test items',flush=True)
    run=run_analysis(W.loc[W.index[test]],predictions,samples,1729,5)
    save_run(run,out);rows,detail=summarize(run,'heldout','log2');table=pd.DataFrame(rows)
    table['variant']=np.where(table.clipping==0,'heldout_calibrated','uncalibrated_1_percent')
    table.to_csv(out/'leaderboard.csv',index=False);write_json(out/'equivalences.json',detail)
    calibrated.to_csv(out/'calibrated_probabilities.csv')
    pd.DataFrame({'item_id':W.index,'split':np.where(train,'calibration','evaluation'),'context_group':meta.context_group}).to_csv(out/'split.csv',index=False)
    np.savez_compressed(out/'sample_schedule.npz',indices=np.concatenate(samples),offsets=np.cumsum([0]+[len(s) for s in samples]))
    independent=independent_scores(W.loc[W.index[test]],predictions,run.classifier_scores.df,'log2',samples[:4])
    import matplotlib.pyplot as plt
    main_table=table[table.variant=='heldout_calibrated'].sort_values('score',ascending=False).reset_index(drop=True)
    fig,(ax,curveax)=plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios':[1.3,1]})
    for i,r in main_table.iterrows():
        value=r.equivalence if r.status=='crossing' else r.bound
        if pd.notna(value):
            ax.scatter(value,i,color='#2166ac',marker='o' if r.status=='crossing' else '>')
            if pd.notna(r.ci_low) and pd.notna(r.ci_high):ax.plot([r.ci_low,r.ci_high],[i,i],color='#2166ac',lw=2)
            ax.annotate(f'{value:.2f}' if r.status=='crossing' else f'>{value:g}',(value,i),xytext=(5,7),textcoords='offset points',fontsize=8)
    ax.set_yticks(range(len(main_table)),main_table.model);ax.invert_yaxis();ax.set_xlabel('Human-rater equivalence with conditional 95% interval')
    ax.set_title('One AI forecast + calibration fitted on separate contexts');ax.grid(axis='x',alpha=.2)
    curve=run.expert_power_curve.df;lo,hi=np.quantile(curve.iloc[1:].to_numpy(),[.025,.975],axis=0)
    curveax.fill_between(curve.columns,lo,hi,color='black',alpha=.15)
    curveax.plot(curve.columns,curve.iloc[0],'-o',color='black',label='Bayesian human panel')
    for color,(_,r) in zip(['#2166ac','#d95f02','#1b9e77'],main_table.head(3).iterrows()):
        curveax.axhline(r.score,ls='--',label=r.model,color=color)
    curveax.legend(fontsize=8,loc='lower right');curveax.grid(alpha=.2)
    curveax.set(xlabel='Number of input human ratings (k)',ylabel='Expected log₂ score (higher is better)',title='Human and calibrated AI scores on held-out items')
    fig.suptitle(f'Exploratory DICES follow-up • {train.sum()} calibration items / {test.sum()} evaluation items • 64 human raters',fontsize=13)
    fig.text(.5,.025,'This evaluates the calibrated judge system, not the original raw forecasts. Calibration and evaluation contexts are disjoint.\nIntervals condition on fitted calibration, human pool, fitted combiner, sampled panels and generation 0; ordering is descriptive.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,.94])
    for ext in ['png','svg','pdf']:fig.savefig(out/f'dices_calibrated_leaderboard.{ext}',dpi=180)
    plt.close(fig)
    write_json(out/'manifest.json',dict(exploratory=True,train_items=int(train.sum()),test_items=int(test.sum()),
        train_contexts=meta.context_group[train].nunique(),test_contexts=meta.context_group[test].nunique(),
        disjoint_contexts=True,seed=1729,bootstraps=500,raters=W.shape[1],k_max=5,
        parameters=parameters,independent_score_max_error=independent,
        training='Equal-item binomial cross entropy; logit input clipped to [.005,.995]; intercept [-10,10], slope [0,10]',
        input_sha256={name:digest(source/name) for name in ['human_ratings_quality.csv','judge_probabilities.csv','item_manifest.csv']},
        source_revision=json.loads((source/'manifest.json').read_text())['revision'],
        implementation_sha256={str(path.relative_to(ROOT)):digest(path) for path in [Path(__file__),ROOT/'research/dices.py',*sorted((ROOT/'surveyequivalence').glob('*.py'))]}))
    print(main_table.to_string(index=False),flush=True)


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
