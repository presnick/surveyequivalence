"""Five real-response figures from three additional social-influence studies.

Gurcay 2015: percentage questions and actual chat. Becker 2019: two partisan
network experiments. Jayles 2021: full versus averaged peer information.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .social_influence_equivalence import panel_mse, first_integer, quantile_with_censoring
    from .panel_power import first_equivalent_size, equivalence_label
except ImportError:
    from social_influence_equivalence import panel_mse, first_integer, quantile_with_censoring
    from panel_power import first_equivalent_size, equivalence_label

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache/social-influence-data'
OUT = ROOT / 'benchmarks/results/social-influence-expanded'
PDF = ROOT / 'output/pdf/social_influence_more_datasets.pdf'
B, SEED = 5000, 28092028
SOURCES = {
    'task-gurcay2015.csv': dict(sha256='a8bab890fbe5f7f5f76e7b998faa13698253c788d3e5b2e22a573d2561b7700f',
        url='https://raw.githubusercontent.com/amaatouq/task-dependence/master/data/prior_research_data/gurcay2015.csv',
        paper='https://doi.org/10.1002/bdm.1843'),
    'partisan-original.csv': dict(sha256='33596e95e4a02b015a9b971f44af80878d9ed9ce9fc9de0103394ce075539c97',
        url='https://raw.githubusercontent.com/joshua-a-becker/wisdom-of-partisan-crowds/master/Becker%20Centola%20Porter%20-%20Wisdom%20of%20Partisan%20Crowds%20-%20Supplementary%20Dataset.csv',
        paper='https://doi.org/10.1073/pnas.1817195116'),
    'jayles2021_Data.csv': dict(sha256='aea0349a4ef1941b8a657a28e2d72a38fde565ee08a161eda03b2283307e723a',
        url='https://ndownloader.figshare.com/files/27962157',
        paper='https://doi.org/10.1098/rsif.2021.0231')}
CONFIG = {
    'Gurcay2015': dict(arms=['C', 'I', 'G'], titles=['Later: private revision', 'Later: chat / individual pay', 'Later: chat / group pay'],
        kmax=7, reference=5, metric='-MSE (percentage points squared)',
        title='Private estimates, then a 90-second discussion',
        subtitle='Seven percentage questions: private revision versus chat, with individual or group incentives.\nAll dashed lines show private estimates BEFORE discussion; solid lines show the subsequent individual answers.',
        footer='Gurcay, Mellers & Baron (2015)  |  doi.org/10.1002/bdm.1843  |  Data: Almaatouq et al. repository',
        note='Reference = 5 initial reports in the same condition. Some original groups have only 7 paired responses, so k stops at 7.\nControl uncertainty resamples 92 independent people; chat uncertainty resamples 10 intact groups per incentive condition.'),
    'Becker2019_1': dict(arms=['Control', 'Social'], titles=['Later: private revision', 'Later: feedback from four peers'],
        kmax=25, reference=10, metric='-squared log error',
        title='Partisan networks: experiment 1',
        subtitle='Four factual questions about unemployment, taxes, California and the electorate.\nEach social group contained one political party; participants were not told their peers shared that affiliation.',
        footer='Becker, Porter & Centola (2019)  |  doi.org/10.1073/pnas.1817195116  |  Original author release',
        note='Panel estimate = geometric mean. Equal weight across groups; no mixed-party panels are constructed.\nBootstrap resamples intact groups within party, retaining both rounds and all questions.'),
    'Becker2019_2': dict(arms=['Control', 'Social'], titles=['Later: private revision', 'Later: feedback from four peers'],
        kmax=25, reference=10, metric='-squared relative error',
        title='Partisan networks: experiment 2',
        subtitle='Four new questions, including signed changes in unemployment and immigration.\nHere, shared party affiliation was disclosed and partisan symbols were shown; final answers follow two revisions.',
        footer='Becker, Porter & Centola (2019)  |  doi.org/10.1073/pnas.1817195116  |  Original author release',
        note='Panel estimate = arithmetic mean; errors divided by each question\'s true value, retaining its sign and decimals.\nAuthor-flagged extreme rows are excluded; uncertainty resamples intact groups within party.'),
    'Jayles2021': dict(arms=['Sorted', 'Unsorted', 'Aggregated'],
        titles=['Later: sorted peer guesses', 'Later: unsorted peer guesses', 'Later: peers\' geometric mean'],
        kmax=12, reference=10, metric='-squared log error',
        title='How should other people\'s guesses be displayed?',
        subtitle='216 people answered 42 questions in 18 groups of 12. All panels average over the same 42 questions.\nDashed lines: initial private estimates. Solid lines: revised estimates after the named information display.',
        footer='Jayles, Sire & Kurvers (2021)  |  doi.org/10.1098/rsif.2021.0231  |  Data: figshare 12529571',
        note='Panel estimate = geometric mean. Each display has 6 independent groups; intervals resample whole groups.\nCurves average over 1, 3, 5, 7, 9 or 11 shared peer estimates. The next page separates that experimental factor.')}
INK, MUTED, GRID, BEFORE = '#202C3B', '#586879', '#DEE5EB', '#34495E'
COLORS = ['#657386', '#138575', '#B66735']


def load(name):
    p = CACHE / name
    if hashlib.sha256(p.read_bytes()).hexdigest() != SOURCES[name]['sha256']:
        raise ValueError(f'Input changed: {name}')
    return pd.read_csv(p)


def partisan_errors(estimates, truth, experiment):
    """Preserve decimal truths and negative change targets; never int-cast."""
    estimates, truth = np.asarray(estimates, float), np.asarray(truth, float)
    if not np.isfinite(estimates).all() or not np.isfinite(truth).all() or np.any(truth == 0):
        raise ValueError('Need finite estimates and nonzero truth')
    if experiment == 1:
        if np.any(estimates <= 0) or np.any(truth <= 0):
            raise ValueError('Log scores need positive values')
        return np.log(estimates / truth)
    if experiment == 2:
        return (estimates - truth) / truth
    raise ValueError('Unknown experiment')


def prepare():
    frames, audit = [], {}
    g = load('task-gurcay2015.csv')
    selected = g[g['question.type'] == 'percentage']
    valid = selected.dropna(subset=['est1', 'est2']).copy()
    assert valid[['est1', 'est2']].ge(0).all().all() and valid[['est1', 'est2']].le(100).all().all()
    x = pd.DataFrame(dict(study='Gurcay2015', arm=valid.condition, cluster=valid.group.astype(str),
                          person=valid['subject.no'].astype(str), task=valid['question.no'].astype(str),
                          before=valid.est1, after=valid.est2, truth=valid['true values'], stratum='all', tau=0))
    x['error_before'], x['error_after'] = x.before - x.truth, x.after - x.truth
    frames.append(x)
    audit['Gurcay2015'] = dict(raw_rows=len(g), selected_percentage_rows=len(selected), paired_rows=len(valid),
        missing_selected_rows=len(selected) - len(valid), selected_questions=sorted(valid['question.no'].unique().tolist()),
        scope='Seven percentage questions; nine unbounded questions not analyzed. All available paired cells retained, no imputation or outlier deletion.',
        condition_mapping='C=private individual pay; I=chat individual pay; G=chat group pay',
        control_group='Source group 40 is a pooled independent control sample, not a single interacting group.')
    b = load('partisan-original.csv')
    for exp in [1, 2]:
        selected = b[b.experiment == exp]
        keep = np.isfinite(selected[['response_1', 'response_2', 'response_3']]).all(axis=1)
        if exp == 1:
            keep &= selected[['response_1', 'response_2', 'response_3']].gt(0).all(axis=1)
        else:
            keep &= selected.remove_extreme.isna()
        valid = selected[keep].copy()
        cluster = valid[['set', 'pair_id', 'network', 'party']].astype(str).agg('|'.join, axis=1)
        x = pd.DataFrame(dict(study=f'Becker2019_{exp}', arm=valid.network, cluster=cluster,
                              person=valid.user_id.astype(str), task=valid.q, before=valid.response_1,
                              after=valid.response_3, truth=valid.truth, stratum=valid.party, tau=0))
        x['error_before'] = partisan_errors(x.before, x.truth, exp)
        x['error_after'] = partisan_errors(x.after, x.truth, exp)
        frames.append(x)
        audit[f'Becker2019_{exp}'] = dict(raw_rows=len(selected), paired_rows=len(valid), excluded_rows=len(selected) - len(valid),
            all_rounds_required=True, cleaning='Positive finite all three rounds, following author log-analysis cohort.' if exp == 1 else 'Finite all three rounds and no author remove_extreme flag, following author cleaned cohort.',
            author_flagged_rows=int(selected.remove_extreme.notna().sum()),
            target_values={str(k):float(v) for k,v in valid.groupby('q').truth.first().items()})
    j = load('jayles2021_Data.csv')
    assert len(j) == 9072 and j.groupby(['group', 'question_ID']).size().eq(12).all()
    assert j.groupby(['group', 'question_ID']).tau.nunique().max() == 1
    assert j.groupby('group').treatment.nunique().max() == 1
    assert j[['personal_estimate', 'second_estimate', 'true_value']].gt(0).all().all()
    x = pd.DataFrame(dict(study='Jayles2021', arm=j.treatment, cluster=j.group.astype(str),
                          person=j.participant.astype(str), task=j.question_ID.astype(str), before=j.personal_estimate,
                          after=j.second_estimate, truth=j.true_value, stratum='all', tau=j.tau))
    x['error_before'], x['error_after'] = np.log(x.before / x.truth), np.log(x.after / x.truth)
    frames.append(x)
    j[['question_ID', 'question', 'true_value']].drop_duplicates().to_csv(OUT/'jayles_questions.csv', index=False)
    audit['Jayles2021'] = dict(raw_rows=len(j), paired_rows=len(j), excluded_rows=0, people=216, groups=18, questions=42,
                             tau_values=sorted(j.tau.unique().tolist()), source_description='Actual estimates from real peers; no virtual participants or simulated rounds in these files.')
    prepared = pd.concat(frames, ignore_index=True)
    assert not prepared.duplicated(['study', 'cluster', 'person', 'task']).any()
    assert np.isfinite(prepared[['before', 'after', 'truth', 'error_before', 'error_after']]).all().all()
    prepared.to_csv(OUT/'paired_responses.csv', index=False)
    return prepared, audit


def finite_bound(v, maximum):
    return float(v) if np.isfinite(v) else f'>{maximum}'


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = prepare()
    moments, checks = [], []
    for (study, arm, cluster, task), cell in data.groupby(['study', 'arm', 'cluster', 'task']):
        for phase in ['before', 'after']:
            e = cell['error_' + phase].to_numpy()
            assert len(e) >= CONFIG[study]['kmax']
            ii, jj = np.triu_indices(len(e), 1)
            direct = [np.mean(e**2), np.mean(((e[ii] + e[jj]) / 2)**2), e.mean()**2]
            value = panel_mse(e, [1, 2, len(e)])
            np.testing.assert_allclose(value, direct, rtol=1e-12, atol=1e-12)
            checks.append(float(np.max(np.abs(value - direct))))
            moments.append(dict(study=study, arm=arm, cluster=cluster, task=task, phase=phase,
                stratum=cell.stratum.iloc[0], tau=int(cell.tau.iloc[0]), n=len(e),
                mean_error=float(e.mean()), variance=float(e.var(ddof=1)),
                group_mse=float(e.mean()**2), sd=float(e.std(ddof=1))))
    moments = pd.DataFrame(moments)
    moments.to_csv(OUT/'group_task_moments.csv', index=False)
    rng = np.random.default_rng(SEED)
    curves, summaries, boot_save, peer_rows = [], [], {}, []
    for study, cfg in CONFIG.items():
        ks = np.arange(1, cfg['kmax'] + 1)
        for arm in cfg['arms']:
            selected = moments[(moments.study == study) & (moments.arm == arm)]
            groups = sorted(selected.cluster.unique())
            group_curves = []
            for group in groups:
                by_phase = []
                for phase in ['before', 'after']:
                    m = selected[(selected.cluster == group) & (selected.phase == phase)]
                    by_phase.append((m.group_mse.to_numpy()[:, None] + m.variance.to_numpy()[:, None] *
                                     (1 / ks - 1 / m.n.to_numpy()[:, None])).mean(axis=0))
                group_curves.append(by_phase)
            group_curves = np.asarray(group_curves)
            mean = group_curves.mean(axis=0)
            prefix = study + '_' + arm
            if study == 'Gurcay2015' and arm == 'C':
                # The control pool contains independent people, not 92 members
                # of one discussion. Keep all seven answers from each person.
                c = data[(data.study == study) & (data.arm == arm)]
                before = c.pivot(index='person', columns='task', values='error_before').sort_index()
                after = c.pivot(index='person', columns='task', values='error_after').reindex(index=before.index, columns=before.columns)
                a = np.stack([before.values, after.values], axis=1)
                assert a.shape == (92, 2, 7) and np.isfinite(a).all()
                boot = np.empty((B, 2, len(ks)))
                for start in range(0, B, 100):
                    count = min(100, B-start)
                    sample = a[rng.integers(0, len(a), size=(count, len(a)))]
                    bias = (sample.mean(axis=1)**2).mean(axis=2)
                    var = sample.var(axis=1, ddof=1).mean(axis=2)
                    boot[start:start+count] = bias[:, :, None] + var[:, :, None] * (1/ks - 1/len(a))
                resampling_units, unit_label = 92, 'independent people'
            else:
                strata = selected.groupby('cluster').stratum.first().reindex(groups).to_numpy()
                draws = np.concatenate([rng.choice(np.flatnonzero(strata == s), size=(B, int((strata == s).sum())))
                                        for s in sorted(set(strata))], axis=1)
                boot = group_curves[draws].mean(axis=1)
                boot_save[prefix+'_sampled_groups'] = np.asarray(groups)[draws]
                resampling_units, unit_label = len(groups), 'groups'
                if study == 'Jayles2021':
                    taus = [1, 3, 5, 7, 9, 11]
                    gain = np.empty((len(groups), len(taus)))
                    for gi, group in enumerate(groups):
                        for ti, tau in enumerate(taus):
                            losses = []
                            for phase in ['before', 'after']:
                                z = selected[(selected.cluster == group) & (selected.tau == tau) & (selected.phase == phase)]
                                assert len(z) == 7
                                losses.append((z.group_mse + z.variance*(1/5 - 1/z.n)).mean())
                            gain[gi, ti] = losses[0] - losses[1]
                    gain_boot = gain[draws].mean(axis=1)
                    for ti, tau in enumerate(taus):
                        peer_rows.append(dict(display=arm, peers_shown=tau, panel_k=5,
                            accuracy_gain=float(gain[:,ti].mean()), lower=float(np.quantile(gain_boot[:,ti],.025)),
                            upper=float(np.quantile(gain_boot[:,ti],.975))))
            ref = cfg['reference']
            target, boot_target = mean[0, ref-1], boot[:, 0, ref-1]
            eq = first_equivalent_size(ks, -mean[1], -target)
            integers = first_integer(boot[:, 1], boot_target)
            np.testing.assert_array_equal(first_integer(boot[:, 0], boot_target), ref)
            delta = boot[:, 1, ref-1] - boot_target
            summary = dict(study=study, condition=arm, reference_k=ref, max_k=int(ks[-1]),
                reference_mse=float(target), revised_mse_at_reference_k=float(mean[1,ref-1]),
                equivalent_k=finite_bound(eq,ks[-1]), equivalent_at_lower_boundary=bool(eq==1),
                equivalent_integer=finite_bound(first_integer(mean[1],target)[0],ks[-1]),
                equivalent_integer_95_low=finite_bound(quantile_with_censoring(integers,.025),ks[-1]),
                equivalent_integer_95_high=finite_bound(quantile_with_censoring(integers,.975),ks[-1]),
                unreached_bootstrap_fraction=float(np.isinf(integers).mean()),
                difference_95_low=float(np.quantile(delta,.025)), difference_95_high=float(np.quantile(delta,.975)),
                resampling_units=resampling_units, unit_label=unit_label,
                paired_rows=int(selected[selected.phase=='before'].n.sum()),
                observed_group_tasks=int((selected.phase=='before').sum()))
            summaries.append(summary)
            lo, hi = np.quantile(boot,[.025,.975],axis=0)
            for r, phase in enumerate(['before','after']):
                for ki,k in enumerate(ks):
                    curves.append(dict(study=study, condition=arm, phase=phase, k=int(k),
                        expected_mse=float(mean[r,ki]), score=float(-mean[r,ki]),
                        mse_95_low=float(lo[r,ki]),mse_95_high=float(hi[r,ki])))
            boot_save[prefix+'_reference_mse']=boot_target
            boot_save[prefix+'_equivalent_integer']=integers
            boot_save[prefix+'_difference']=delta
    pd.DataFrame(curves).to_csv(OUT/'panel_curves.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'equivalent_panels.csv',index=False)
    pd.DataFrame(peer_rows).to_csv(OUT/'jayles_peer_count.csv',index=False)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**boot_save)
    meta=dict(data_kind='Observed human responses only; no synthetic raters or simulated social rounds',
        sources=SOURCES, preparation=audit, bootstrap_replicates=B,seed=SEED,results=summaries,
        checks=dict(cell_rounds=len(checks),maximum_absolute_enumeration_error=max(checks),
                    exact='Single report, exhaustive pairs, full pool; bootstrap reference self-equivalence'),
        scoring='Exact expected MSE of k distinct reports within intact group-task cells; negative loss plotted. Geometric panels for log errors, arithmetic panels otherwise.',
        weighting='Equal observed questions within group, then equal groups within condition. Party strata have equal counts in Becker. No mixed-group panels.',
        uncertainty='Paired percentile bootstrap: intact groups (within party for Becker), except Gurcay independent control resamples whole person rows over all seven fixed questions. Unattained crossings retained.',
        limitations=['Each condition has its own initial-score reference; equivalences are not a common-benchmark ranking.',
            'Social k counts selected reports from existing groups, not the number of people needed to reproduce a discussion.',
            'No causal between-arm contrast or formal equivalence test is computed.',
            'Gurcay is restricted to seven percentage questions and observed paired cells; question coverage varies in some chat groups.',
            'Becker source participant codes restart across experiments; group and participant identifiers are scoped by experiment.',
            'Different plots use different error units; numerical scores must not be compared across studies.',
            'Jayles tau is the actual experimental number of peers shown; it is distinct from reaggregation panel size k.',
            'Small numbers of groups limit bootstrap inference; results do not establish performance on new tasks or recruits.'])
    (OUT/'results.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    return meta


def plot_all():
    os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':INK,'axes.labelcolor':INK,
        'axes.edgecolor':GRID,'xtick.color':MUTED,'ytick.color':MUTED,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
    meta=json.loads((OUT/'results.json').read_text())
    curves=pd.read_csv(OUT/'panel_curves.csv')
    PDF.parent.mkdir(parents=True,exist_ok=True)

    def page(title,subtitle,footer,number):
        fig=plt.figure(figsize=(14,9))
        fig.text(.065,.945,title,fontsize=23,weight='bold',va='top')
        fig.text(.065,.878,subtitle,fontsize=11,color=MUTED,va='top',linespacing=1.55)
        fig.text(.94,.978,f'REAL STUDY RESPONSES  /  {number:02d}',fontsize=9,color=MUTED,ha='right')
        fig.text(.065,.026,footer,fontsize=8.5,color=MUTED)
        return fig

    def save(fig,stem,pdf):
        fig.savefig(OUT/f'{stem}.png',dpi=180)
        fig.savefig(OUT/f'{stem}.svg')
        pdf.savefig(fig)
        plt.close(fig)

    def label(x):
        return f'{x:g}' if isinstance(x,(float,int)) else x

    with PdfPages(PDF,metadata={'Title':'More real datasets: discussion, partisan networks and peer displays'}) as pdf:
        for number,(study,cfg) in enumerate(CONFIG.items(),1):
            fig=page(cfg['title'],cfg['subtitle'],cfg['footer'],number)
            c=curves[curves.study==study]
            floor=float(c.score.min())*1.08
            n=len(cfg['arms']);step=.90/n;width=step-.045
            for i,arm in enumerate(cfg['arms']):
                x=.075+i*step
                ax=fig.add_axes([x,.33,width,.405])
                entry=next(r for r in meta['results'] if r['study']==study and r['condition']==arm)
                for phase,color,style in [('before',BEFORE,'--'),('after',COLORS[i],'-')]:
                    line=c[(c.condition==arm)&(c.phase==phase)].sort_values('k')
                    ax.plot(line.k,line.score,color=color,ls=style,lw=2.5,
                        label='Initial private estimates' if phase=='before' else 'Final individual estimates')
                target=-entry['reference_mse'];eq=entry['equivalent_k']
                eq_num=float(eq) if isinstance(eq,(float,int)) else np.inf
                ax.axhline(target,color='#9A865A',ls=':',lw=1.1)
                ax.scatter(cfg['reference'],target,color=BEFORE,marker='s',s=36,zorder=5)
                if np.isfinite(eq_num):
                    score=float(c[(c.condition==arm)&(c.phase=='after')&(c.k==1)].score.iloc[0]) if eq_num==1 else target
                    ax.vlines(eq_num,floor,score,color=COLORS[i],ls=':',lw=1.2)
                    ax.scatter(eq_num,score,color=COLORS[i],s=60,edgecolors='white',zorder=6)
                ticks=list(range(1,8)) if cfg['kmax']==7 else ([1,5,10,15,20,25] if cfg['kmax']==25 else [1,3,6,10,12])
                ax.set(xlim=(.85,cfg['kmax']+.15),ylim=(floor,0),xticks=ticks,xlabel='Recorded reports in panel (k)')
                ax.set_title(cfg['titles'][i],fontsize=11,weight='bold',loc='left',pad=18)
                if i==0:ax.set_ylabel('Expected score ('+cfg['metric']+')\nHigher is more accurate; zero is perfect')
                else:ax.tick_params(labelleft=False)
                ax.grid(axis='y',color=GRID,lw=.7)
                ax.legend(loc='lower right',fontsize=8,frameon=True,facecolor='white',edgecolor='white',framealpha=1)
                eq_label=equivalence_label(eq_num,np.arange(1,cfg['kmax']+1))
                text=f'{eq_label} final reports match {cfg["reference"]} initial'
                if eq_num==1:text='One final report exceeds the reference'
                if not np.isfinite(eq_num):text=f'No match within {cfg["kmax"]} final reports'
                fig.text(x,.265,text,fontsize=11.5,color=COLORS[i],weight='bold')
                fig.text(x,.217,f'Whole reports: {label(entry["equivalent_integer"])}  |  95% interval: '
                    f'{label(entry["equivalent_integer_95_low"])} to {label(entry["equivalent_integer_95_high"])}\n'
                    f'{entry["resampling_units"]} {entry["unit_label"]}; {entry["paired_rows"]:,} paired person-question rows',
                    fontsize=9,color=MUTED,linespacing=1.65)
            fig.text(.065,.132,cfg['note'],fontsize=9.2,color=MUTED,linespacing=1.65)
            fig.text(.065,.076,'Every facet uses its own initial reference. Dashed = before feedback; solid = after revision.\n'
                'k counts reports sampled from existing groups. It is not the number of participants needed to run the social experiment.',
                fontsize=9.2,color=MUTED,linespacing=1.65)
            save(fig,study.lower()+'_panel_accuracy',pdf)
        cfg=CONFIG['Jayles2021']
        fig=page('Does seeing more peers change the benefit?',
            'Jayles experiment: participants actually saw 1, 3, 5, 7, 9 or 11 peers\' estimates, or their geometric mean.\n'
            'All results below use five recorded reports per panel. Positive values mean lower error after social information.',cfg['footer'],5)
        peers=pd.read_csv(OUT/'jayles_peer_count.csv')
        ax=fig.add_axes([.105,.275,.79,.46])
        for i,arm in enumerate(cfg['arms']):
            d=peers[peers.display==arm].sort_values('peers_shown')
            ax.plot(d.peers_shown,d.accuracy_gain,marker='o',lw=2.3,color=COLORS[i],label=arm)
            ax.fill_between(d.peers_shown.to_numpy(),d.lower.to_numpy(),d.upper.to_numpy(),color=COLORS[i],alpha=.12)
        ax.axhline(0,color=INK,ls=':',lw=1.2)
        ax.set(xticks=[1,3,5,7,9,11],xlim=(.7,11.3),xlabel='Number of peers whose estimates were shown (tau)',
               ylabel='Accuracy gain at panel k=5\n(initial squared log error - final squared log error)')
        ax.grid(axis='y',color=GRID,lw=.7)
        ax.legend(loc='best',frameon=True,facecolor='white',edgecolor='white')
        fig.text(.105,.193,'This horizontal axis is an experimental manipulation, not a resampled panel size.',fontsize=12,weight='bold')
        fig.text(.065,.13,'Lines join observed condition means; no response model is fitted. Shading: exploratory pointwise 95% whole-group bootstrap intervals.\n'
            'Each display has only 6 groups. Questions vary within group across tau; the 42-question coverage is balanced across groups.',
            fontsize=9.3,color=MUTED,linespacing=1.65)
        fig.text(.065,.073,'These are paired before/after gains, not a formal causal comparison of display types or peer-count levels.',fontsize=9.3,color=MUTED)
        save(fig,'jayles2021_peer_count',pdf)
    print(PDF)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if not args.plot_only:
        result=analyze()
        print(json.dumps(result['results'],indent=2))
    plot_all()
