"""Risk survey panel equivalence from released human responses and observed truth.

Prepare with bundled Python + pyreadstat in .cache/risk-python; analyze with
numpy/pandas; plot with --plot-only in the repository environment. Raw data stays
in the ignored cache. No wagers, model inference, or synthetic human responses.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

try:
    from .panel_power import equivalence_label, first_equivalent_size
except ImportError:
    from panel_power import equivalence_label, first_equivalent_size

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/risk-survey-data'
OUT = ROOT / 'benchmarks/results/risk-survey-equivalence'
SEED = 20260927
B = 4000
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_xlsx_columns(path, wanted, header_name):
    """Stream only needed columns; reject spreadsheet errors as missing."""
    with ZipFile(path) as z:
        strings = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        records, columns = [], None
        with z.open('xl/worksheets/sheet1.xml') as stream:
            context = ET.iterparse(stream, events=('start', 'end'))
            _, root = next(context)
            for event, row in context:
                if event != 'end' or row.tag != '{'+NS['s']+'}row':
                    continue
                cells = {}
                for c in row.findall('s:c', NS):
                    col = c.get('r').rstrip('0123456789')
                    if columns is not None and col not in columns:
                        continue
                    v = c.find('s:v', NS)
                    if v is None:
                        continue
                    if c.get('t') == 's':
                        value = strings[int(v.text)]
                    elif c.get('t') == 'e':
                        value = np.nan
                    else:
                        value = float(v.text)
                    cells[col] = value
                if columns is None and header_name in cells.values():
                    columns = {k: v for k, v in cells.items() if v in wanted}
                    assert set(columns.values()) == set(wanted)
                elif columns is not None and cells:
                    records.append({name: cells.get(col, np.nan) for col, name in columns.items()})
                root.clear()
        return pd.DataFrame(records, columns=wanted)


def prepare():
    inspection = json.loads((ROOT/'benchmarks/results/risk-data-search/inspection.json').read_text())
    for info in list(inspection['price'].values()) + inspection['spf']['files'] + [inspection['sce']]:
        p = ROOT/info.get('file', info.get('path', ''))
        assert sha(p) == info['sha256'], str(p)
    sys.path.insert(0, str(ROOT/'.cache/risk-python'))
    import pyreadstat
    for exp in [2, 3]:
        d, _ = pyreadstat.read_sav(str(CACHE/f'price_exp{exp}.sav'))
        d.to_csv(CACHE/f'price_exp{exp}_prepared.csv', index=False)
    for name, cols in [('recess', ['YEAR','QUARTER','ID','RECESS2']),
                       ('rgdp', ['YEAR','QUARTER','ID','RGDP2','RGDP3'])]:
        d = read_xlsx_columns(CACHE/f'spf_individual_{name}.xlsx', cols, 'YEAR')
        d.to_csv(CACHE/f'spf_{name}_prepared.csv', index=False)
    cols = ['date','userid','Q8v2','Q8v2part2','Q9_mean'] + [f'Q9_bin{i}' for i in range(1,11)]
    d = read_xlsx_columns(CACHE/'sce_public_microdata_2020_2024.xlsx', cols, 'userid')
    d.to_csv(CACHE/'sce_prepared.csv', index=False)
    print('SCE prepared:', len(d), 'rows; date range', d.date.min(), d.date.max(),
          'direction counts', d.Q8v2.value_counts(dropna=False).to_dict(),
          'point range', (d.Q8v2part2.min(), d.Q8v2part2.max()), flush=True)
    files = sorted(CACHE.glob('*_prepared.csv'))
    (CACHE/'prepared_hashes.json').write_text(json.dumps({p.name: sha(p) for p in files},indent=2)+'\n')


def finite_mse(values, truth, ks):
    """Exact squared loss of uniform k-person means, rows people, columns cases."""
    a = np.asarray(values, dtype=float)
    ks = np.asarray(ks, dtype=float)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Need at least two complete numeric respondents')
    if np.any(ks < 1) or np.any(ks > len(a)):
        raise ValueError('Unsupported panel size')
    return (a.mean(0)-truth)**2 + (1/ks[:,None]-1/len(a))*a.var(0,ddof=1)


def crossing(curves, target):
    eligible = np.asarray(curves) <= np.asarray(target)[...,None] + 1e-12
    return np.where(eligible.any(-1), eligible.argmax(-1)+1, eligible.shape[-1]+1)


def block_indices(n, length, rng, reps=B):
    """Circular moving-block bootstrap, truncated to exactly n cases."""
    starts = rng.integers(0,n,size=(reps,int(np.ceil(n/length))))
    return ((starts[...,None]+np.arange(length)) % n).reshape(reps,-1)[:,:n]


def fred_table(filename):
    text = (CACHE/filename).read_text()
    body = text.split('<table id="data-table-observations"',1)[1].split('</table>',1)[0]
    pairs = re.findall(r'<th[^>]*>(\d{4}-\d{2}-\d{2})</th>\s*<td[^>]*>\s*([\d.,-]+)\s*</td>',body)
    # FRED keeps observations after the first 1,000 in a second HTML container.
    if '<div id="extra-rows">' in text:
        extra=text.split('<div id="extra-rows">',1)[1].split('</div>',1)[0]
        pairs+=re.findall(r'#(\d{4}-\d{2}-\d{2})\|([\d.,-]+)',extra)
    assert len(pairs) > 200
    pairs=[(d,v) for d,v in pairs if v not in ['.','-']]
    assert len({d for d,_ in pairs})==len(pairs)
    return pd.Series({pd.Timestamp(d): float(v.replace(',','')) for d,v in pairs}).sort_index()


def write_csv(path, records):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def analyze_experiment(name, methods, per_case, boot, reference, metadata, score='rms'):
    """Save full curves plus equivalence to a specified 10-person method."""
    # per_case: cases x methods x panel sizes. boot: draws x methods x panel sizes.
    mean = per_case.mean(0)
    kmax = mean.shape[-1]
    target = mean[reference,9]
    eq = crossing(mean,target)
    beq = crossing(boot,boot[:,reference,9,None])
    bounds = np.quantile(beq,[.025,.975],axis=0,method='inverted_cdf').astype(int)
    display = np.sqrt if score=='rms' else lambda x:x
    label = lambda k: str(int(k)) if k <= kmax else f'>{kmax}'
    table=[]
    for i, method in enumerate(methods):
        delta = boot[:,i,9]-boot[:,reference,9]
        table.append(dict(analysis=name,method=method,reference_method=methods[reference],reference_people=10,
                          people_to_match=label(eq[i]),bootstrap_95_lower=label(bounds[0,i]),
                          bootstrap_95_upper=label(bounds[1,i]),
                          bootstrap_no_crossing_fraction=float(np.mean(beq[:,i]>kmax)),
                          score_at_1=float(display(mean[i,0])),score_at_10=float(display(mean[i,9])),
                          reference_score=float(display(target)),
                          loss_difference_at_10=float(mean[i,9]-target),
                          loss_difference_95_lower=float(np.quantile(delta,.025)),
                          loss_difference_95_upper=float(np.quantile(delta,.975))))
    low,high = np.quantile(display(boot),[.025,.975],axis=0)
    folder = OUT/name;folder.mkdir(parents=True,exist_ok=True)
    write_csv(folder/'equivalent_panels.csv',table)
    write_csv(folder/'panel_curves.csv',[dict(method=m,k=k+1,loss=float(mean[i,k]),score=float(display(mean[i,k])),
              bootstrap_95_lower=float(low[i,k]),bootstrap_95_upper=float(high[i,k]))
              for i,m in enumerate(methods) for k in range(kmax)])
    score_label={'rms':'RMS error (percentage points)','brier':'Expected Brier score',
                 'regret':'Expected regret (points per decision)'}[score]
    result=dict(analysis=name,score=score_label,
                lower_is_better=True,panel_range=[1,kmax],reference_method=methods[reference],reference_people=10,
                formula='mean_cases[(population_mean-truth)^2 + sample_variance*(1/k - 1/n)]',
                equivalence='Smallest supported integer k matching or improving the reference loss; no extrapolation',
                panel_sampling='Uniform distinct respondents without replacement; equal case weights and equal respondent weights',
                seed=SEED,bootstrap_replicates=len(boot),results=table)
    result.update(metadata)
    (folder/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(folder/'bootstrap_draws.npz',equivalence=beq,reference_loss=boot[:,reference,9])
    print(name,[(r['method'],r['people_to_match'],r['score_at_10']) for r in table],flush=True)
    return table


def price():
    d=pd.read_csv(CACHE/'price_exp3_prepared.csv')
    cols=[f'judgment_{v}' for v in [5,10,15,20,25]]
    truth=np.array([5,10,15,20,25]);ks=np.arange(1,51)
    groups=[d[(d.array_type==a)&(d.question_type==q)][cols].to_numpy() for a in [1,2] for q in [1,2]]
    assert [len(a) for a in groups]==[55,56,55,52]
    per_case=np.stack([finite_mse(a,truth,ks).T for a in groups],axis=1)
    rng=np.random.default_rng(SEED);boot=np.empty((B,4,len(ks)))
    for g,a in enumerate(groups):
        for b in range(B):
            sampled=a[rng.integers(0,len(a),size=len(a))]
            boot[b,g]=finite_mse(sampled,truth,ks).mean(-1)
    methods=['random_percentage','random_probability','grouped_percentage','grouped_probability']
    meta=dict(people=218,targets=truth.tolist(),cases=5,group_counts=[len(a) for a in groups],
              uncertainty='Whole-person bootstrap within each randomized cell; five fixed targets; reference recomputed per draw',
              data_granularity='Each response is a person-level average over repeated presentations at a target probability',
              limits=['Known stimulus values, not future adverse outcomes.','Only five target levels; uncertainty is conditional on these targets.'])
    rows=analyze_experiment('price_all_formats',methods,per_case,boot,0,meta)
    grouped_meta={**meta,'people':107,'group_counts':[55,52]}
    rows+=analyze_experiment('price_grouped_wording',methods[2:],per_case[:,2:],boot[:,2:],0,grouped_meta)
    return rows


def price_decisions():
    # The released proportions summarize five decisions per person and case.
    # Threshold the panel's average propensity; do not pretend these are raw votes.
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from research.sports_survey_equivalence import exact_curve
    d=pd.read_csv(CACHE/'price_exp2_prepared.csv')
    cols=[f'risk_{p}_{ev}' for p in [5,10,15,20,25] for ev in ['pos','neg']]
    signs=np.tile([1,-1],5)
    groups=[]
    for condition in [1,2,3]:
        a=d[d.condition==condition][cols].to_numpy()
        np.testing.assert_allclose(a*5,np.rint(a*5),atol=1e-9)
        assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
        groups.append((2*np.rint(a*5).astype(int)-5)*signs)
    assert [len(a) for a in groups]==[37,38,38]
    # Nominal +/-5 point EV from the study design; zero-EV cases have zero regret.
    # Ten scored cases plus five zeros: 5 * (1-accuracy) * 10/15.
    kmax=37
    def losses(a):
        return (1-np.array([exact_curve(a[:,j],kmax) for j in range(10)]))*5*10/15
    pc=np.stack([losses(a) for a in groups],axis=1)
    rng=np.random.default_rng(SEED);boot=np.empty((1000,3,kmax))
    for g,a in enumerate(groups):
        for b in range(len(boot)):
            boot[b,g]=losses(a[rng.integers(0,len(a),size=len(a))]).mean(0)
        print('Price decisions bootstrap group',g+1,'complete',flush=True)
    meta=dict(people=113,group_counts=[37,38,38],cases=15,informative_cases=10,
              formula='Exact subset-sum enumeration of thresholded panel-mean take-risk proportions; nominal +/-5 point expected values; five zero-EV cases contribute zero regret',
              aggregation='Take risk if mean released propensity > 0.5; pass if < 0.5; random action on ties',
              uncertainty='1000 whole-person bootstrap draws within each randomized condition; fixed cases; reference recomputed',
              limits=['Risk-neutral expected-points utility chosen for this analysis, not a measure of all risk preferences.',
                      'Uses repeated-decision summaries, not raw single-trial panel votes.',
                      'Nominal study EVs are used; rounded loss amounts in the paper can differ slightly.',
                      'Thresholding is an analyst-specified aggregation rule; curves may be nonmonotonic.'])
    return analyze_experiment('price_decisions',['numerical_percentage','random_icons','grouped_icons'],pc,boot,0,meta,'regret')


def spf():
    left=pd.read_csv(CACHE/'spf_recess_prepared.csv');right=pd.read_csv(CACHE/'spf_rgdp_prepared.csv')
    keys=['YEAR','QUARTER','ID']
    assert not left.duplicated(keys).any() and not right.duplicated(keys).any()
    d=left.merge(right,on=keys,validate='one_to_one')
    d=d[d.YEAR>=1992].copy();initial=len(d)
    d=d.dropna(subset=['RECESS2','RGDP2','RGDP3'])
    d=d[d.RECESS2.between(0,100)&(d.RGDP2>0)&(d.RGDP3>0)]
    invalid_count=initial-len(d)
    d['survey']=pd.PeriodIndex(d.YEAR.astype(int).astype(str)+'Q'+d.QUARTER.astype(int).astype(str),freq='Q')
    gdp=fred_table('GDPC1.txt');gdp.index=gdp.index.to_period('Q')
    actual=(gdp/gdp.shift(1)-1).dropna()
    targets={p:float(actual[p+1]<0) for p in sorted(d.survey.unique()) if p+1 in actual.index}
    excluded=[str(p) for p in sorted(d.survey.unique()) if p not in targets]
    d=d[d.survey.isin(targets)]
    counts=d.groupby('survey').size();kmax=int(counts.min());ks=np.arange(1,kmax+1)
    arrays=[];case_rows=[]
    for p,g in d.groupby('survey',sort=True):
        a=np.column_stack([(g.RGDP3<g.RGDP2).astype(float),g.RECESS2.to_numpy()/100])
        arrays.append(finite_mse(a,targets[p],ks).T)
        case_rows.append(dict(survey=str(p),target=str(p+1),people=len(g),decline=int(targets[p]),
                              realized_growth_percent=100*float(actual[p+1])))
    pc=np.stack(arrays) # cases x methods x k
    rng=np.random.default_rng(SEED);idx=block_indices(len(pc),8,rng)
    boot=pc[idx].mean(1)
    meta=dict(cases=len(pc),responses=len(d),unique_forecaster_ids=int(d.ID.nunique()),
              first_survey=case_rows[0]['survey'],last_survey=case_rows[-1]['survey'],
              negative_growth_cases=sum(r['decline'] for r in case_rows),
              incomplete_or_invalid_response_rows=invalid_count,
              unresolved_survey_quarters_excluded=excluded,
              outcome_source='https://fred.stlouisfed.org/data/GDPC1',outcome_snapshot='2026-09-27 download; latest revised series, not first-release truth',
              uncertainty='Circular moving-block bootstrap of 8 consecutive survey quarters; observed respondent pools fixed',
              limits=['Forecaster IDs can follow firms rather than individual people.','Output decline is not an NBER recession.','Vote fraction from point estimates is an aggregation rule, not individually elicited probabilities.','Latest revised GDP outcomes; no first-release vintage sensitivity.'])
    rows=analyze_experiment('spf_gdp_decline',['point_implied_decline_votes','explicit_decline_probability'],pc,boot,0,meta,'brier')
    write_csv(OUT/'spf_gdp_decline/cases.csv',case_rows)
    return rows


def sce(trim=False):
    d=pd.read_csv(CACHE/'sce_prepared.csv');initial=len(d)
    assert not d.duplicated(['date','userid']).any()
    cols=['Q8v2','Q8v2part2','Q9_mean']+[f'Q9_bin{i}' for i in range(1,11)]
    d=d.dropna(subset=cols).copy()
    bins=d[[f'Q9_bin{i}' for i in range(1,11)]].to_numpy()
    # Released point estimates already carry the deflation sign. Confirm it;
    # applying the questionnaire's sign a second time would reverse deflation.
    signs=((d.Q8v2==1)&(d.Q8v2part2>=0))|((d.Q8v2==2)&(d.Q8v2part2<=0))
    valid=signs&np.isfinite(d[cols]).all(axis=1)&((bins>=0)&(bins<=100)).all(axis=1)&np.isclose(bins.sum(1),100,atol=1e-6)
    d=d[valid].copy()
    valid_count=len(d)
    if trim:
        d=d[d.Q8v2part2.abs()<=100].copy()
    trim_removed=valid_count-len(d)
    d['point']=d.Q8v2part2
    d['month']=pd.to_datetime(d.date.astype(int).astype(str),format='%Y%m').dt.to_period('M')
    assert d.month.min()==pd.Period('2020-01','M') and d.month.max()==pd.Period('2024-12','M')
    levels={}
    for series,filename in [('CPIAUCNS','CPIAUCNS.html'),('PCEPI','PCEPI.html')]:
        level=fred_table(filename);level.index=level.index.to_period('M');levels[series]=level
    common=[m for m in sorted(d.month.unique()) if all(m in s.index and m+12 in s.index for s in levels.values())]
    missing_outcomes=[str(m) for m in sorted(d.month.unique()) if m not in common]
    d=d[d.month.isin(common)].copy()
    # Equal respondents define panel equivalence, not population-weighted survey estimates.
    ks=np.arange(1,101);rows=[]
    for series,level in levels.items():
        pc=[];cases=[]
        for month,g in d.groupby('month',sort=True):
            assert month in level.index and month+12 in level.index
            truth=100*(level[month+12]/level[month]-1)
            a=g[['point','Q9_mean']].to_numpy()
            assert len(a)>=100
            pc.append(finite_mse(a,truth,ks).T)
            cases.append(dict(month=str(month),target_month=str(month+12),people=len(g),
                              realized_inflation_percent=float(truth),point_mean=float(a[:,0].mean()),density_mean=float(a[:,1].mean())))
        pc=np.stack(pc)
        idx=block_indices(len(pc),12,np.random.default_rng(SEED));boot=pc[idx].mean(1)
        meta=dict(cases=len(pc),responses=len(d),unique_person_ids=int(d.userid.nunique()),
                  initial_response_rows=initial,excluded_incomplete_or_invalid_rows=initial-valid_count,
                  sensitivity_filter='abs(point estimate) <= 100 percentage points' if trim else 'none',
                  sensitivity_rows_removed_before_outcome_join=trim_removed,
                  survey_months_excluded_missing_common_outcomes=missing_outcomes,
                  point_estimate_range=[float(d.point.min()),float(d.point.max())],
                  min_people_per_case=min(c['people'] for c in cases),max_people_per_case=max(c['people'] for c in cases),
                  first_survey=cases[0]['month'],last_survey=cases[-1]['month'],
                  outcome_source=f'https://fred.stlouisfed.org/data/{series}',outcome_snapshot='2026-09-27 download',
                  outcome_definition='100*(index at survey month + 12 / index at survey month - 1)',
                  uncertainty='Circular moving-block bootstrap of 12 retained survey months; observed people fixed; at most 60 months',
                  summary='Q9_mean is the released fitted density mean; Q8v2part2 is already signed in the released file (validated against Q8v2)',
                  limits=['Generic inflation question does not specify CPI/PCE; both reported as research benchmarks.',
                          'Equal-person panels, not the official survey-weighted estimates.',
                          'Extreme point estimates are retained in the primary analysis; paired abs(point)<=100 sensitivity reported separately.',
                          'Density elicitation plus publisher fitting is compared with a best guess; these can target different functionals.',
                          'Overlapping 12-month outcomes and repeat respondents; bootstrap does not cover recruitment of new people.'])
        name='sce_'+series.lower()+('_bounded_points' if trim else '')
        rows+=analyze_experiment(name,['direct_point_estimate','elicited_density_mean'],pc,boot,0,meta)
        write_csv(OUT/name/'cases.csv',cases)
    return rows


def plot():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(15,16))
    configs=[('price_all_formats','Known probabilities: four survey variants'),
             ('price_grouped_wording','Known probabilities: wording within grouped arrays'),
             ('price_decisions','Risk decisions: thresholded panel preferences'),
             ('spf_gdp_decline','GDP decline: point-implied votes vs probabilities'),
             ('sce_cpiaucns','Inflation / CPI: all eligible point estimates'),
             ('sce_cpiaucns_bounded_points','Inflation / CPI: paired sensitivity, |point| ≤ 100')]
    colors=['#526478','#187c83','#ba6334','#8a589c']
    method_labels={
        'random_percentage':'Random icons, percentage wording',
        'random_probability':'Random icons, probability wording',
        'grouped_percentage':'Grouped icons, percentage wording',
        'grouped_probability':'Grouped icons, probability wording',
        'numerical_percentage':'Numerical percentages',
        'random_icons':'Random icons',
        'grouped_icons':'Grouped icons',
        'point_implied_decline_votes':'Point-implied decline votes',
        'explicit_decline_probability':'Explicit decline probabilities',
        'direct_point_estimate':'Direct point estimates',
        'elicited_density_mean':'Fitted density means',
    }
    score_labels={
        'RMS error (percentage points)':'Expected score: −MSE (percentage points²)',
        'Expected Brier score':'Expected score: −Brier loss',
        'Expected regret (points per decision)':'Expected score: −regret (points per decision)',
    }
    for ax,(name,title) in zip(axes.flat,configs):
        meta=json.loads((OUT/name/'results.json').read_text());d=pd.read_csv(OUT/name/'panel_curves.csv')
        reference=meta['reference_method'];reference_people=meta['reference_people']
        reference_row=d.loc[(d.method==reference)&(d.k==reference_people)].iloc[0]
        target=-float(reference_row.loss)
        crossings=[]
        for i,(method,g) in enumerate(d.groupby('method',sort=False)):
            g=g.sort_values('k')
            score=-g.loss.to_numpy()
            eq=first_equivalent_size(g.k.to_numpy(),score,target)
            eq_label=equivalence_label(eq,g.k.to_numpy())
            eq_label='k*'+('' if eq_label.startswith(('≤','>')) else '=')+eq_label
            label=method_labels[method]+(' [reference]' if method==reference else '')
            ax.plot(g.k,score,label=f'{label}; {eq_label}',color=colors[i],lw=2)
            crossings.append((eq,float(score[0]),colors[i]))
        ax.axhline(target,ls='--',color='#444',lw=1.2,zorder=1)
        ax.scatter([reference_people],[target],s=65,marker='*',color='#222',zorder=6)
        ax.set(title=title,xlabel='People per panel (k)',ylabel=score_labels[meta['score']])
        ax.set_xlim(float(d.k.min()),float(d.k.max()))
        ymin=ax.get_ylim()[0]
        for eq,first_score,color in crossings:
            if not np.isfinite(eq):
                continue
            # A method already above the target at k=1 has no supported crossing.
            # Mark its observed endpoint and report the boundary as <=1.
            y=first_score if eq==d.k.min() else target
            ax.vlines(eq,ymin,y,color=color,ls=':',lw=1.4,alpha=.8,zorder=3)
            ax.scatter([eq],[y],s=40,facecolors='white',edgecolors=color,lw=1.7,zorder=5,clip_on=False)
        ax.set_ylim(bottom=ymin)
        ax.grid(alpha=.15)
        ax.legend(fontsize=9,frameon=False,loc='upper left',bbox_to_anchor=(0,-.21),borderaxespad=0)
    fig.suptitle('Risk survey rater equivalence',y=.98,fontsize=19,fontweight='bold')
    fig.text(.5,.955,'Higher expected score is better · read equivalent panel size where a curve first reaches the shared target',
             ha='center',fontsize=11)
    fig.text(.075,.025,
             'Dashed line / star: the reference method at 10 people. Dotted projections / circles: each method’s first matching panel size.\n'
             'Scores are negative expected losses; each panel shares its own 10-person target. k* uses linear interpolation; tables retain integer counts.\n'
             '≤1: already meets the target at one person (no k=0 estimate). >K: no match through K. Empirical curves: no log fits or forced monotonicity.\n'
             'Uncertainty is in the tables. GDP equivalence is uncertain; inflation results depend on extreme points and density fitting.',fontsize=9.5)
    fig.subplots_adjust(left=.075,right=.98,bottom=.15,top=.91,wspace=.30,hspace=.70)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'panel_accuracy.{ext}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        prepare();return
    if args.plot_only:
        plot();return
    hashes=json.loads((CACHE/'prepared_hashes.json').read_text())
    for name,digest in hashes.items():
        assert sha(CACHE/name)==digest
    OUT.mkdir(parents=True,exist_ok=True)
    rows=price()+price_decisions()+spf()+sce()+sce(trim=True)
    write_csv(OUT/'equivalent_panels.csv',rows)
    manifest={'run_date':'2026-09-28','source_inspection':'../risk-data-search/inspection.json',
              'prepared_hashes':hashes,'outcome_hashes':{n:sha(CACHE/n) for n in ['GDPC1.txt','CPIAUCNS.html','PCEPI.html']},
              'sandia':{'status':'blocked: public CSV host returns Request Rejected HTML','equivalence':None,
                        'retry_file_sha256':sha(CACHE/'sandia_retry.txt')},'results':rows}
    (OUT/'results.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
