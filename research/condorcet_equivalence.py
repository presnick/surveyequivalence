"""Execute the Condorcet jury analysis and six-figure report.

Offline after condorcet_data.py --download. --plot-only uses saved results.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, hypergeom, betabinom

try:
    from .condorcet_data import ROOT, CACHE, OUT, prepare, digest
    from .panel_power import first_equivalent_size, equivalence_label
    from .sports_survey_equivalence import exact_curve
except ImportError:
    from condorcet_data import ROOT, CACHE, OUT, prepare, digest
    from panel_power import first_equivalent_size, equivalence_label
    from sports_survey_equivalence import exact_curve

SEED, B = 20261006, 4000
ORDER = ['nba', 'bluebirds', 'cities', 'ducks']
NAMES = dict(nba='NBA forecasts', bluebirds='Bluebirds', cities='City comparisons', ducks='Ducks')
KMAX = dict(nba=10, bluebirds=39, cities=97, ducks=40)
PDF = ROOT / 'output/pdf/condorcet_jury_equivalence.pdf'
TEAL, ORANGE, PURPLE, INK, GRAY = '#157e83', '#b76428', '#7853a0', '#223343', '#657585'


def binomial_curve(ks, p):
    k = np.asarray(ks)
    return binom.sf(k // 2, k, p) + np.where(k % 2 == 0, .5 * binom.pmf(k // 2, k, p), 0)


def finite_curves(n, c, kmax):
    n, c = np.asarray(n, dtype=int), np.asarray(c, dtype=int)
    if np.any(c < 0) or np.any(c > n) or np.any(n < kmax):
        raise ValueError('Invalid finite pool or unsupported panel size')
    k = np.arange(1, kmax + 1)[None, :]
    return hypergeom.sf(k // 2, n[:, None], c[:, None], k) + np.where(
        k % 2 == 0, .5 * hypergeom.pmf(k // 2, n[:, None], c[:, None], k), 0)


def whole_crossing(curve, target):
    """K+1 is a censoring code; never interpret it as an observed crossing."""
    good = np.asarray(curve) >= np.asarray(target)[..., None] - 1e-12
    return np.where(good.any(axis=-1), good.argmax(axis=-1) + 1, good.shape[-1] + 1)


def label_count(k, maximum):
    return '>' + str(maximum) if k > maximum else str(int(k))


def draw_weights(items, dataset, rng):
    """Conditional-on-raters uncertainty: waves for NBA, stratified images for birds."""
    n = len(items)
    if dataset == 'cities':
        return None
    if dataset == 'nba':
        waves = sorted(items.group.unique())
        old = ROOT / 'benchmarks/results/sports-survey-equivalence'
        meta = json.loads((old / 'results.json').read_text())
        assert waves == meta['waves']
        with np.load(old / 'bootstrap_draws.npz') as saved:
            draws = saved['wave_multiplicities']
        w = draws[:, [waves.index(v) for v in items.group]].astype(float)
        return w / w.sum(axis=1)[:, None]
    w = np.zeros((B, n))
    for truth in [0, 1]:
        idx = np.flatnonzero(items.truth.to_numpy() == truth)
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)), size=B)
    return w / n


def score_datasets(votes, items):
    rng = np.random.default_rng(SEED)
    curve_rows, class_rows, summary, item_rows, draws = [], [], [], [], {}
    for ds in ORDER:
        it = items[items.dataset == ds].reset_index(drop=True)
        ks = np.arange(1, KMAX[ds] + 1)
        per = finite_curves(it.n, it.c, KMAX[ds])
        np.testing.assert_allclose(per[:, 0], it.c/it.n, atol=1e-13)
        np.testing.assert_allclose(per[:, :2*(len(ks)//2):2], per[:, 1::2], atol=1e-13)
        curve = per.mean(axis=0)
        ideal = binomial_curve(ks, curve[0])
        conditional = np.zeros_like(curve)
        balanced = np.zeros_like(curve)
        groups = {'truth 0': it.truth.eq(0), 'truth 1': it.truth.eq(1)}
        if ds == 'ducks':
            groups.update({x: it.group.eq(x) for x in sorted(it.group.unique())})
        for group, mask in groups.items():
            cc = per[mask].mean(axis=0)
            model = binomial_curve(ks, cc[0])
            if group.startswith('truth'):
                conditional += mask.mean() * model
                balanced += .5 * cc
            for k, a, m in zip(ks, cc, model):
                class_rows.append(dict(dataset=ds, group=group, items=int(mask.sum()), k=k,
                                       accuracy=a, iid_class_accuracy=m))
        w = draw_weights(it, ds, rng)
        boot = w @ per if w is not None else None
        if boot is not None:
            bmodel = binomial_curve(ks[None, :], boot[:, :1])
            blo, bhi = np.quantile(boot, [.025, .975], axis=0)
            dlo, dhi = np.quantile(bmodel-boot, [.025, .975], axis=0)
            draws[ds+'_weights'] = w
            draws[ds+'_accuracy'] = boot
            draws[ds+'_iid'] = bmodel
        else:
            blo = bhi = dlo = dhi = np.full(len(ks), np.nan)
        reference_grid = np.arange(1, 102)
        model_grid = binomial_curve(reference_grid, curve[0])
        e = int(whole_crossing(model_grid, curve[-1]))
        fraction = first_equivalent_size(reference_grid, model_grid, curve[-1])
        if boot is not None:
            model_boot = binomial_curve(reference_grid[None, :], boot[:, :1])
            eq = whole_crossing(model_boot, boot[:, -1])
            admissible = boot[:, 0] > .5
            eq = eq[admissible]
            intervals = np.quantile(eq, [.025, .975], method='inverted_cdf').astype(int)
            draws[ds+'_iid_equivalence'] = eq
            lo, hi = [label_count(x, 101) for x in intervals]
            censor = float(np.mean(eq > 101))
        else:
            lo = hi = None
            censor = None
        summary.append(dict(dataset=ds, items=len(it), raters=int(votes[votes.dataset == ds].rater_id.nunique()),
            max_panel=int(ks[-1]), single_accuracy=float(curve[0]), max_panel_accuracy=float(curve[-1]),
            iid_accuracy_at_max=float(ideal[-1]), gap_pp=float(100*(ideal[-1]-curve[-1])),
            model_whole_equivalent=label_count(e,101), model_interpolated_equivalent=equivalence_label(fraction,reference_grid),
            model_equivalent_95_lower=lo, model_equivalent_95_upper=hi, model_no_crossing_fraction=censor,
            bootstrap_below_chance_fraction=float(np.mean(boot[:,0] <= .5)) if boot is not None else None,
            majority_wrong_items=int((2*it.c < it.n).sum()), majority_tied_items=int((2*it.c == it.n).sum()),
            uncertainty='Fixed-item descriptive; none estimated' if ds == 'cities' else
              ('10000 paired wave resamples; conditional on people' if ds == 'nba' else '4000 class-stratified image resamples; conditional on workers')))
        for j,k in enumerate(ks):
            curve_rows.append(dict(dataset=ds,k=int(k),accuracy=curve[j],lower=blo[j],upper=bhi[j],
                iid_accuracy=ideal[j],class_conditional_iid=conditional[j],balanced_accuracy=balanced[j],gap_pp=100*(ideal[j]-curve[j]),
                gap_lower_pp=100*dlo[j],gap_upper_pp=100*dhi[j]))
        for i,row in it.iterrows():
            item_rows.append(dict(dataset=ds,item_id=row.item_id,group=row.group,n=int(row.n),c=int(row.c),
                single_accuracy=per[i,0],accuracy_k9=per[i,8],gain_k9_pp=100*(per[i,8]-per[i,0])))
        np.savez_compressed(OUT / (ds+'_item_curves.npz'), item_ids=it.item_id.to_numpy(str), curves=per)
        print(ds, json.dumps(summary[-1]), flush=True)
    pd.DataFrame(curve_rows).to_csv(OUT/'panel_curves.csv',index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'class_curves.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(item_rows).to_csv(OUT/'item_diagnostics.csv',index=False)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**draws)
    return summary


def nba_aggregation():
    old = ROOT/'benchmarks/results/sports-survey-equivalence'
    data = pd.read_csv(old/'game_panel_curves.csv')
    items = pd.read_csv(OUT/'items.csv')
    it = items[items.dataset == 'nba'].reset_index(drop=True)
    with np.load(OUT/'bootstrap_draws.npz') as b:
        w = b['nba_weights']
    rows, equivalences = [], []
    arrays = {}
    for method in ['winner_only','confidence_weighted','surprisingly_popular']:
        x = data[data.method == method].pivot(index='game_id',columns='k',values='accuracy').reindex(it.item_id).to_numpy()
        assert np.isfinite(x).all()
        arrays[method] = (x.mean(axis=0), w@x)
    reference, bref = arrays['winner_only'][0][-1], arrays['winner_only'][1][:,-1]
    fresh = pd.read_csv(OUT/'panel_curves.csv').query("dataset == 'nba'").accuracy
    np.testing.assert_allclose(fresh,arrays['winner_only'][0],atol=1e-13)
    for method, (curve,boot) in arrays.items():
        eq = whole_crossing(boot,bref)
        lower,upper = np.quantile(boot,[.025,.975],axis=0)
        elo,ehi = np.quantile(eq,[.025,.975],method='inverted_cdf')
        frac = first_equivalent_size(np.arange(1,11),curve,reference)
        delta = 100*(boot[:,-1]-bref)
        equivalences.append(dict(dataset='nba',method=method,target_panel=10,target_accuracy=reference,
            whole=label_count(int(whole_crossing(curve,reference)),10),fractional=equivalence_label(frac,np.arange(1,11)),
            lower=label_count(elo,10),upper=label_count(ehi,10),no_crossing_fraction=float(np.mean(eq>10)),
            delta_at_target_pp=float(100*(curve[-1]-reference)),
            delta_lower_pp=float(np.quantile(delta,.025)),delta_upper_pp=float(np.quantile(delta,.975))))
        for j in range(10):rows.append(dict(method=method,k=j+1,accuracy=curve[j],lower=lower[j],upper=upper[j]))
    pd.DataFrame(rows).to_csv(OUT/'nba_aggregation_curves.csv',index=False)
    pd.DataFrame(equivalences).to_csv(OUT/'nba_equivalence.csv',index=False)
    return equivalences


def learned_panels(votes,items):
    rows, eqs, split_rows, weights_rows, meta = [], [], [], [], {}
    rng = np.random.default_rng(SEED+5)
    saved_boot = {}
    for ds in ['bluebirds','cities']:
        d = votes[votes.dataset == ds]
        matrix = d.pivot(index='item_id',columns='rater_id',values='correct').sort_index().dropna()
        it = items[items.dataset == ds].set_index('item_id').loc[matrix.index]
        train = np.zeros(len(it),bool)
        test = np.zeros(len(it),bool)
        if ds == 'bluebirds':
            for truth in [0,1]:
                idx = np.flatnonzero(it.truth.to_numpy() == truth)
                rng.shuffle(idx)
                train[idx[:len(idx)//2]] = True
            test = ~train
        else:
            cities = np.unique(np.concatenate([it.city_a.to_numpy(),it.city_b.to_numpy()]))
            rng.shuffle(cities)
            train_cities = set(cities[:len(cities)//2])
            train = (it.city_a.isin(train_cities) & it.city_b.isin(train_cities)).to_numpy()
            test = (~it.city_a.isin(train_cities) & ~it.city_b.isin(train_cities)).to_numpy()
            assert not set(it[test].city_a).union(it[test].city_b).intersection(train_cities)
        x = matrix.to_numpy(int)
        skill = (x[train].sum(axis=0)+2)/(train.sum()+4)
        # Finite, nonnegative log-odds convention, discretized for exact subset DP.
        weights = np.clip(np.rint(3*np.maximum(0,np.log(skill/(1-skill)))),0,10).astype(int)
        selected = np.argsort(-skill,kind='stable')[:x.shape[1]//2]
        xt = x[test]
        kmax = 15
        assert math.comb(x.shape[1],kmax) < np.iinfo(np.int64).max
        base = finite_curves(np.full(len(xt),x.shape[1]),xt.sum(axis=1),kmax)
        select = finite_curves(np.full(len(xt),len(selected)),xt[:,selected].sum(axis=1),kmax)
        weighted = np.stack([exact_curve((2*y-1)*weights,kmax) for y in xt])
        methods = {'random':base,'training_selected':select,'trained_weighted':weighted}
        test_items = it[test].reset_index()
        bw = draw_weights(test_items,ds,rng)
        if bw is not None:saved_boot[ds+'_weights']=bw
        np.savez_compressed(OUT/(ds+'_learned_item_curves.npz'), item_ids=matrix.index[test].to_numpy(str), **methods)
        target = base[:,8].mean()
        bref = bw@base[:,8] if bw is not None else None
        for method, per in methods.items():
            curve = per.mean(axis=0)
            if bw is not None:
                boot = bw@per
                lo,hi = np.quantile(boot,[.025,.975],axis=0)
                eq = whole_crossing(boot,bref)
                elo,ehi = np.quantile(eq,[.025,.975],method='inverted_cdf')
                lower,upper=label_count(elo,kmax),label_count(ehi,kmax)
                censor=float(np.mean(eq>kmax))
                delta=100*(boot[:,8]-bref)
                delta_lo,delta_hi=np.quantile(delta,[.025,.975])
                saved_boot[ds+'_'+method]=boot
            else:
                lo=hi=np.full(kmax,np.nan);lower=upper=None;censor=None
                delta_lo=delta_hi=None
            frac=first_equivalent_size(np.arange(1,kmax+1),curve,target)
            eqs.append(dict(dataset=ds,method=method,target_panel=9,target_accuracy=target,
                whole=label_count(int(whole_crossing(curve,target)),kmax),fractional=equivalence_label(frac,np.arange(1,kmax+1)),
                lower=lower,upper=upper,no_crossing_fraction=censor,accuracy_k9=float(curve[8]),
                delta_k9_pp=float(100*(curve[8]-target)),delta_lower_pp=delta_lo,delta_upper_pp=delta_hi))
            for j in range(kmax):rows.append(dict(dataset=ds,method=method,k=j+1,accuracy=curve[j],lower=lo[j],upper=hi[j]))
        for i,item in enumerate(matrix.index):
            split_rows.append(dict(dataset=ds,item_id=item,split='train' if train[i] else 'test' if test[i] else 'cross_city_excluded'))
        for j,rater in enumerate(matrix.columns):
            weights_rows.append(dict(dataset=ds,rater_id=rater,training_accuracy=float(x[train,j].mean()),
                smoothed_accuracy=skill[j],weight=int(weights[j]),selected=bool(j in selected)))
        meta[ds]=dict(complete_items=len(it),train_items=int(train.sum()),test_items=int(test.sum()),
            cross_city_excluded=int((~(train|test)).sum()),incomplete_items_excluded=int(d.item_id.nunique()-len(it)),
            raters=x.shape[1],selected_raters=len(selected),kmax=kmax,
            weighting='round(3*max(0,logit((training_correct+2)/(training_n+4)))) clipped to [0,10]; exact integer subset DP.',
            uncertainty='Conditional on trained policy and workers; image resampling only' if ds=='bluebirds' else 'Fixed test city-pairs; no confidence intervals',
            split='50% within each truth class' if ds=='bluebirds' else 'Disjoint city identities: within-train-city pairs for training, within-test-city pairs for test; cross pairs excluded')
        print('LEARNED',ds,json.dumps(meta[ds]),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'learned_curves.csv',index=False)
    pd.DataFrame(eqs).to_csv(OUT/'learned_equivalence.csv',index=False)
    pd.DataFrame(split_rows).to_csv(OUT/'learning_splits.csv',index=False)
    pd.DataFrame(weights_rows).to_csv(OUT/'trained_raters.csv',index=False)
    (OUT/'learning_design.json').write_text(json.dumps(meta,indent=2)+'\n')
    np.savez_compressed(OUT/'learning_bootstrap.npz',**saved_boot)
    return meta


def theory_tables():
    ks=np.arange(1,102)
    rows=[]
    for p in [.45,.5,.55,.65,.75]:
        for k,a in zip(ks,binomial_curve(ks,p)):
            rows.append(dict(p=p,k=int(k),accuracy=a))
    pd.DataFrame(rows).to_csv(OUT/'theory_curves.csv',index=False)
    rho_values=[0,.02,.05,.1,.2,.4]
    rows=[]
    for p in [.55,.65,.75]:
        reference=binomial_curve(ks,p)
        for rho in rho_values:
            if rho==0:a=reference
            else:
                alpha,beta=p*(1/rho-1),(1-p)*(1/rho-1)
                a=betabinom.sf(ks//2,ks,alpha,beta)+np.where(ks%2==0,.5*betabinom.pmf(ks//2,ks,alpha,beta),0)
            for k,acc in zip(ks,a):
                e=first_equivalent_size(ks,reference,acc)
                rows.append(dict(p=p,rho=rho,k=int(k),accuracy=acc,
                    equivalent_iid_fractional=float(e),equivalent_iid_whole=int(whole_crossing(reference,acc))))
    pd.DataFrame(rows).to_csv(OUT/'shared_case_model.csv',index=False)


def run_analysis():
    votes,items,audit=prepare()
    summary=score_datasets(votes,items)
    nba=nba_aggregation()
    learned=learned_panels(votes,items)
    theory_tables()
    manifest=dict(seed=SEED,bootstrap_replicates=B,nba_bootstrap_replicates=10000,summary=summary,
        nba_equivalence=nba,learning=learned,script_sha256=digest(__file__),
        data_audit_sha256=digest(OUT/'data_audit.json'),
        interpretation='Objective-outcome adaptation of rater equivalence; iid-model gaps do not isolate dependence.',
        conditional_on='Recorded rater pools. City curves have no item-iid confidence intervals.',
        reference='Empirical finite-pool random juries or explicitly hypothetical iid jurors; never interchangeable.',
        bootstrap_scope='Bird images stratified by truth; NBA entire waves; learned-policy bands conditional on fixed training.',
        numerical_method='Exact hypergeometric majority tails; exact integer subset DP for trained weights; no panel Monte Carlo.',
        source_files=audit['sources'])
    (OUT/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')


def setup_plot():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/condorcet-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold',
        'svg.fonttype':'none','axes.grid':True,'grid.alpha':.2,'grid.color':GRAY})
    return plt,PercentFormatter


def figure_frame(plt,title,subtitle,nrows=1,ncols=1,figsize=(12,7.6),**kw):
    fig,axes=plt.subplots(nrows,ncols,figsize=figsize,**kw)
    fig.subplots_adjust(left=.075,right=.965,top=.79,bottom=.19,hspace=.48,wspace=.28)
    fig.suptitle(title,x=.075,y=.96,ha='left',fontsize=20,fontweight='bold')
    fig.text(.075,.89,subtitle,fontsize=10.5,va='top',color=GRAY,linespacing=1.5)
    return fig,axes


def finish(fig,name,note):
    fig.text(.075,.095,note,fontsize=9,color=GRAY,va='top',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=180,facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)


def plots():
    plt,PercentFormatter=setup_plot()
    theory=pd.read_csv(OUT/'theory_curves.csv')
    fig,ax=figure_frame(plt,'1. What Condorcet predicts','THEORETICAL  |  Independent, equally competent jurors; binary decisions; majority rule.')
    colors=['#b44848',GRAY,'#6b9e50',TEAL,PURPLE]
    for color,(p,g) in zip(colors,theory.groupby('p')):
        g=g[g.k%2==1]
        ax.plot(g.k,g.accuracy,label=f'Individual accuracy {p:.0%}',color=color,lw=2)
    ax.axhline(.9,color=GRAY,ls=':',lw=1)
    g=theory[theory.p==.65]
    match=int(g[g.accuracy>=.9].k.iloc[0])
    ax.scatter([match],[g[g.k==match].accuracy.iloc[0]],color=TEAL,zorder=4)
    ax.annotate(f'90% target: {match} jurors at p=65%',(match,.9),xytext=(match+10,.78),arrowprops={'arrowstyle':'->','color':GRAY},fontsize=10)
    ax.set(xlabel='Jurors (odd sizes shown)',ylabel='Probability majority is correct',xlim=(1,101),ylim=(0,1.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='lower left',frameon=False,fontsize=9)
    finish(fig,'01_condorcet_theory','Analytical binomial tails, not recorded responses. At p=50% the curve stays at chance; below 50%, larger majorities worsen.\nSource: classical Condorcet model; formulas and assumptions in the accompanying methods.')

    curves=pd.read_csv(OUT/'panel_curves.csv')
    fig,axes=figure_frame(plt,'2. Real panels gain less than the simple model predicts',
        'RECORDED RESPONSES + MODEL  |  Solid: exact observed-pool majority accuracy. Dashed: iid model matched to single-rater accuracy.\nShading: pointwise 95% resampling intervals where supported; the city curve is descriptive.',2,4,figsize=(14,8),gridspec_kw={'height_ratios':[2.2,1]})
    for col,ds in enumerate(ORDER):
        g=curves[curves.dataset==ds];display=g[g.k<=31] if ds=='cities' else g
        a,b=axes[:,col]
        a.plot(display.k,display.accuracy,color=TEAL,lw=2,label='Observed pools')
        a.plot(display.k,display.iid_accuracy,color=ORANGE,ls='--',lw=1.8,label='Homogeneous iid')
        if ds!='cities':a.fill_between(display.k,display.lower,display.upper,color=TEAL,alpha=.13)
        a.set(title=NAMES[ds],ylim=(.45,1.02),xlabel='People per panel');a.yaxis.set_major_formatter(PercentFormatter(1))
        b.plot(display.k,display.gap_pp,color=ORANGE)
        if ds!='cities':b.fill_between(display.k,display.gap_lower_pp,display.gap_upper_pp,color=ORANGE,alpha=.15)
        b.axhline(0,color=GRAY,lw=.8);b.set(xlabel='People per panel',ylabel='Model gap (pp)')
    axes[0,0].legend(frameon=False,fontsize=8,loc='lower right')
    finish(fig,'02_observed_vs_iid','NBA: 11-wave bootstrap. Birds: truth-stratified image bootstrap. All intervals condition on the recorded people.\nCity view stops at k=31; all 1,000 items support k=97 (appendix). Gaps can reflect task difficulty, voter differences and sampling, not only dependence.')

    nba=pd.read_csv(OUT/'nba_aggregation_curves.csv');eq=pd.read_csv(OUT/'nba_equivalence.csv')
    target=float(eq.target_accuracy.iloc[0])
    fig,(ax,tableax)=figure_frame(plt,'3. Can richer answers replace additional people?',
        'NBA RECORDED RESPONSES  |  Match the accuracy of ten winner-only respondents on the same 465 games.',1,2,
        gridspec_kw={'width_ratios':[1.55,1]})
    labels={'winner_only':'Winner only','confidence_weighted':'Confidence weighted','surprisingly_popular':'Peer agreement'}
    for color,(method,g) in zip([TEAL,ORANGE,PURPLE],nba.groupby('method',sort=False)):
        ax.plot(g.k,g.accuracy,marker='o',ms=3,color=color,label=labels[method])
        r=eq[eq.method==method].iloc[0];e=float(r.fractional)
        ax.vlines(e,.616,target,color=color,ls=':',lw=1)
    ax.axhline(target,color=GRAY,ls='--');ax.set(xlabel='People per panel',ylabel='Winner accuracy',ylim=(.615,.672),xticks=np.arange(1,11))
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1));ax.legend(frameon=False,fontsize=9,loc='lower right')
    tableax.axis('off')
    text='Reference accuracy: '+f'{target:.2%}'+'\n\n'
    for _,r in eq.iterrows():
        text+=f'{labels[r.method]}\n  Whole people: {r.whole}  |  Interpolated: {r.fractional}\n  95% whole-count interval: {r.lower} to {r.upper}\n\n'
    tableax.text(0,1,text,va='top',fontsize=11,linespacing=1.5)
    finish(fig,'03_nba_equivalence','Paired whole-wave bootstrap recomputes the target and each crossing; >10 draws are retained.\nAll three formats were recorded together. At k=10, both richer-method improvement intervals include zero; no established respondent saving.')

    diag=pd.read_csv(OUT/'item_diagnostics.csv')
    fig,axes=figure_frame(plt,'4. Which questions defeat the crowd?',
        'RETROSPECTIVE DIAGNOSTIC  |  Top: fraction of available votes that are correct. Bottom: exact benefit of nine versus one.',2,4,figsize=(14,8))
    for col,ds in enumerate(ORDER):
        d=diag[diag.dataset==ds];a,b=axes[:,col]
        a.hist(d.single_accuracy,bins=np.linspace(0,1,21),color=TEAL,alpha=.8)
        a.axvline(.5,color=ORANGE,ls='--');a.set(title=NAMES[ds],ylabel='Items',xlim=(0,1))
        b.scatter(d.single_accuracy,d.gain_k9_pp,s=10,alpha=.5,color=TEAL)
        b.axhline(0,color=GRAY,lw=.8);b.axvline(.5,color=ORANGE,ls='--');b.set(ylabel='Accuracy gain (pp)',xlim=(0,1),ylim=(-30,30))
        a.set_xlabel('Correct-vote fraction');b.set_xlabel('Correct-vote fraction')
    finish(fig,'04_item_difficulty','Each point uses recorded responses on one item; positive means improvement at k=9. These groups use known outcomes.\nThe relationship is partly mathematical: majority voting amplifies an observed pool leaning right or wrong. It is not a prospective skill test.')

    learned=pd.read_csv(OUT/'learned_curves.csv');leq=pd.read_csv(OUT/'learned_equivalence.csv');design=json.loads((OUT/'learning_design.json').read_text())
    fig,axes=figure_frame(plt,'5. Expertise versus headcount on held-out questions',
        'RECORDED RESPONSES  |  Selection and weights use training questions only. Every method is scored on the same held-out items.\nThe target is the held-out accuracy of nine random raters from that dataset.',1,2)
    methods={'random':'Random majority','training_selected':'Top half by training accuracy','trained_weighted':'Trained integer weights'}
    for ax,ds in zip(axes,['bluebirds','cities']):
        for (method,label),color in zip(methods.items(),[TEAL,PURPLE,ORANGE]):
            g=learned[(learned.dataset==ds)&(learned.method==method)]
            ax.plot(g.k,g.accuracy,color=color,label=label,lw=2,marker='o',ms=2)
            if ds=='bluebirds':ax.fill_between(g.k,g.lower,g.upper,color=color,alpha=.08)
        q=leq[leq.dataset==ds].target_accuracy.iloc[0]
        ax.axhline(q,color=GRAY,ls='--',lw=1)
        m=design[ds]
        ax.set(title=f'{NAMES[ds]}: {m["train_items"]} train / {m["test_items"]} test',xlabel='People per panel',ylabel='Held-out accuracy')
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,fontsize=8,loc='lower right')
    finish(fig,'05_expertise_headcount','Exact subset scores for all methods; trained weights are discretized nonnegative log odds, not a claim of optimality.\nBird bands condition on fixed training. City split has disjoint city identities and uses complete items; no city confidence bands.')

    model=pd.read_csv(OUT/'shared_case_model.csv')
    fig,(ax,heat)=figure_frame(plt,'6. Shared case conditions can make accuracy plateau',
        'THEORETICAL MODEL  |  Beta-binomial correctness: jurors share a latent item-specific success probability.\nMean individual accuracy remains 65%; rho denotes marginal correctness correlation.',1,2)
    for rho,g in model[model.p==.65].groupby('rho'):
        g=g[g.k%2==1];ax.plot(g.k,g.accuracy,label=f'rho = {rho:g}',lw=1.8)
    ax.set(xlabel='Jurors (odd sizes shown)',ylabel='Majority accuracy',ylim=(.5,1.02));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,fontsize=8,ncol=2)
    h=model[(model.p==.65)&model.k.isin([5,15,31,51,101])].pivot(index='rho',columns='k',values='equivalent_iid_whole')
    heat.imshow(h.to_numpy(),aspect='auto',cmap='YlGnBu',vmin=1,vmax=101)
    heat.set(xticks=np.arange(len(h.columns)),xticklabels=h.columns,yticks=np.arange(len(h)),yticklabels=[f'{x:g}' for x in h.index],
             xlabel='Actual panel size in model',ylabel='rho',title='Smallest iid jury reaching\nthe same accuracy')
    for i in range(len(h)):
        for j in range(len(h.columns)):
            v=int(h.iloc[i,j]);heat.text(j,i,str(v),ha='center',va='center',color='white' if v>50 else INK,fontsize=11)
    finish(fig,'06_shared_case_model','Every value on this page is analytical, not fitted to the datasets. A specified joint model is essential: pairwise correlation alone is insufficient.\nThis mechanism need not involve communication. The variance shortcut k / [1 + (k - 1) rho] is not used as accuracy equivalence.')

    fig,(ax,other)=figure_frame(plt,'Appendix. Full support and class-specific performance',
        'RECORDED RESPONSES + CLASS-CONDITIONAL MODEL  |  Use the full city range and inspect the five Ducks species.',1,2)
    city=curves[curves.dataset=='cities']
    for col,label,color in [('accuracy','Observed city panels',TEAL),('iid_accuracy','Homogeneous iid',ORANGE),('class_conditional_iid','Class-conditional iid',PURPLE)]:
        ax.plot(city.k,city[col],color=color,label=label,ls='-' if col=='accuracy' else '--')
    ax.set(xlabel='People per panel',ylabel='Accuracy',title='Same 1,000 city pairs, k=1 to 97');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,fontsize=8)
    cc=pd.read_csv(OUT/'class_curves.csv');cc=cc[(cc.dataset=='ducks')&~cc.group.str.startswith('truth')]
    for group,g in cc.groupby('group'):other.plot(g.k,g.accuracy,label=group)
    other.set(xlabel='People per panel',ylabel='Accuracy',title='Ducks: different species, different errors');other.yaxis.set_major_formatter(PercentFormatter(1));other.legend(frameon=False,fontsize=8)
    finish(fig,'07_support_and_species','All curves average over fixed observed items. Every city item has at least 97 votes; every Ducks image has 40.\nClass-specific curves for all datasets and matched class-conditional model curves are included in the accompanying CSV tables.')


def pdf_report():
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    summary=pd.read_csv(OUT/'summary.csv')
    PDF.parent.mkdir(parents=True,exist_ok=True)
    c=canvas.Canvas(str(PDF),pagesize=(864,612))
    c.setTitle('Condorcet jury theory and rater equivalence')
    c.setAuthor('SurveyEquivalence research')
    body=ParagraphStyle('body',fontName='Helvetica',fontSize=11,leading=16,textColor=HexColor(INK))
    small=ParagraphStyle('small',parent=body,fontSize=9,leading=13)
    def paragraph(text,x,y,w,style=body):
        p=Paragraph(text,style);pw,ph=p.wrap(w,500);p.drawOn(c,x,y-ph);return y-ph
    def footer(page):
        c.setFillColor(HexColor(GRAY));c.setFont('Helvetica',9)
        c.drawString(42,23,'SurveyEquivalence | October 6, 2026 | Objective-outcome research reanalysis')
        c.drawRightString(822,23,str(page))
    c.setFillColor(HexColor(INK));c.setFont('Helvetica-Bold',26)
    c.drawString(42,556,'How many jurors is a panel worth?')
    y=paragraph('Condorcet jury theory and rater equivalence',42,532,780)
    y=paragraph('Four datasets, six principal figures, and a reproducible comparison of observed panel accuracy with explicitly hypothetical independent jurors.',42,y-20,780)
    y-=30
    c.setFont('Helvetica-Bold',10)
    for x,t in [(42,'Dataset'),(230,'Items'),(312,'Panel'),(405,'Single'),(490,'Observed panel'),(620,'iid model'),(717,'iid jury needed')]:c.drawString(x,y,t)
    y-=22
    for _,r in summary.iterrows():
        c.setFont('Helvetica',10)
        for x,t in [(42,NAMES[r.dataset]),(230,str(r['items'])),(312,str(r.max_panel)),(405,f'{r.single_accuracy:.2%}'),(490,f'{r.max_panel_accuracy:.2%}'),(620,f'{r.iid_accuracy_at_max:.2%}'),(745,str(r.model_whole_equivalent))]:c.drawString(x,y,t)
        y-=24
    y=paragraph('The last column is the smallest whole model jury matching the observed panel at the displayed size, using that dataset\'s single-rater accuracy. It is a model projection, not a recovered count of independent real people.',42,y-10,780,small)
    y=paragraph('<b>Main interpretation.</b> More recorded votes can improve accuracy, but the simple homogeneous independence model can substantially overpredict those gains. This gap does not identify a causal effect of dependence: common item difficulty, heterogeneous skill and finite-pool sampling also matter.',42,y-20,780)
    y=paragraph('<b>Scope.</b> Majority curves use exact sampling without replacement. NBA uncertainty resamples 11 whole waves; bird uncertainty resamples images within truth classes. City results are descriptive because questions share city identities. All results condition on the observed people.',42,y-16,780)
    footer(1);c.showPage()
    c.setFillColor(HexColor(INK));c.setFont('Helvetica-Bold',24)
    c.drawString(42,556,'Empirical substitutions: fewer people, same target')
    y=paragraph('These comparisons use real reference panels. They are distinct from the hypothetical iid equivalents on the previous page.',42,528,780)
    y-=32
    c.setFont('Helvetica-Bold',10)
    for x,t in [(42,'Evaluation set'),(215,'Method'),(410,'Reference'),(510,'Whole people'),(625,'95% interval'),(735,'Fractional')]:c.drawString(x,y,t)
    y-=24
    leq=pd.read_csv(OUT/'learned_equivalence.csv',keep_default_na=False)
    method_names={'training_selected':'Training-selected pool','trained_weighted':'Trained integer weights'}
    for _,r in leq[leq.method!='random'].iterrows():
        vals=[NAMES[r.dataset],method_names[r.method],'9 random',str(r.whole),
              str(r.lower)+' to '+str(r.upper) if str(r.lower) else 'Not estimated',str(r.fractional).replace('\u2264','<=')]
        c.setFont('Helvetica',10)
        for x,t in zip([42,215,410,510,625,735],vals):c.drawString(x,y,t)
        y-=25
    y=paragraph('Bluebirds: 54 training and 54 test images. Cities: 200 training and 215 test pairs with disjoint city identities; 198 incomplete pairs and 387 cross-set pairs excluded. The selected pool contains the top half of observed raters by training accuracy. One selected rater means an average over that pool, not one specific expert.',42,y-10,780,small)
    y=paragraph('Bird intervals resample test images conditional on fixed training and workers. City counts are descriptive for the test pairs. These headcounts do not include the cost of acquiring training labels, and this single split does not establish universal savings.',42,y-14,780,small)
    y-=30
    c.setFont('Helvetica-Bold',13);c.drawString(42,y,'NBA: richer answers do not establish a saving')
    y=paragraph('Ten winner-only respondents score 65.89%. Winner-only majority first reaches this score at nine people because of the odd/even tie plateau. Confidence weighting first reaches it at ten, and peer-agreement aggregation at nine. Their k=10 improvements are +0.14 and +0.43 percentage points, respectively; both paired intervals include zero.',42,y-16,780)
    y=paragraph('For confidence weighting, 26.13% of bootstrap draws do not cross within k=1..10; for peer agreement, 21.16% do not cross. Those draws are retained as bounds. Fractional first crossings are 9.22 and 8.62; these should not be mistaken for whole recruited people.',42,y-15,780,small)
    assert y>45,y
    footer(2);c.showPage()
    names=['01_condorcet_theory','02_observed_vs_iid','03_nba_equivalence','04_item_difficulty','05_expertise_headcount','06_shared_case_model','07_support_and_species']
    for page,name in enumerate(names,3):
        path=OUT/(name+'.png');im=Image.open(path);iw,ih=im.size
        scale=min(830/iw,555/ih);w,h=iw*scale,ih*scale
        c.drawImage(ImageReader(im),(864-w)/2,40+(555-h)/2,width=w,height=h)
        footer(page);c.showPage()
    c.setFillColor(HexColor(INK));c.setFont('Helvetica-Bold',23);c.drawString(42,556,'Methods, sources and limits')
    texts=[
        '<b>Accuracy and equivalence.</b> A k-person panel is a uniform subset of distinct available respondents on an item. Binary majority ties receive half credit. Hypergeometric tails give exact expected accuracy; whole-count equivalence is the first supported crossing. Fractional values interpolate every integer size, retaining even/odd plateaus. A &gt;K result is censored at the supported range, not an estimate of K+1.',
        '<b>Training.</b> Bluebirds uses a seeded, truth-stratified half split of images. Cities uses complete 99-person items and disjoint sets of city identities; pairs crossing those sets are excluded. Rater selection keeps the top half by smoothed training accuracy. Weights are nonnegative, rounded log odds with a Beta(2,2) prior, capped at 10. All weighted subset scores are exact. These conventions were fixed before evaluating test outcomes.',
        '<b>Uncertainty.</b> 4,000 stratified bird-image resamples and the existing 10,000 NBA wave resamples. The iid benchmark is refitted in every draw. Learned-policy bird bands condition on the fixed training result. City curves have no confidence intervals. This does not measure uncertainty from recruiting new people or establish causal effects of panel composition.',
        '<b>Data handling.</b> Bluebirds: 108 images, 39 raters. Ducks: 240 images, 53 raters, 40 votes per image. Cities: 99 retained people, 1,000 pairs, 98,787 valid votes; 213 absent decisions encoded as incorrect are excluded. NBA: 465 games, 113 people, 9,788 complete records. Source hashes and all scoring tables accompany the report.',
        '<b>Sources.</b> Rutchick et al. (2020), doi.org/10.1186/s41235-020-00256-z; NBA release osf.io/2rzcs/. Welinder et al. (2010), The Multidimensional Wisdom of Crowds; github.com/welinder/cubam (pinned commit in data_audit.json). Yu et al. city study release: osf.io/cuzqm/, file osf.io/vyx8a; Kurvers et al. reanalysis: doi.org/10.1126/sciadv.aaw9011.',
        '<b>Framework.</b> Resnick et al., arxiv.org/abs/2106.01254v3 (rater equivalence); Dietrich and Spiekermann, Epistemic democracy with defensible premises (2013); Ladha, doi.org/10.1016/0167-2681(94)00068-P (correlated voting). This report adapts the human-label framework to known outcomes; it does not substitute human consensus for truth.'
    ]
    y=526
    for t in texts:y=paragraph(t,42,y,780,small)-15
    assert y>45,y
    footer(10);c.save()
    print('PDF',PDF,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--no-pdf',action='store_true')
    parser.add_argument('--pdf-only',action='store_true')
    args=parser.parse_args()
    if not args.pdf_only:
        if not args.plot_only:run_analysis()
        plots()
    if not args.no_pdf:pdf_report()
