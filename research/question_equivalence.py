"""Provisional question-panel illustration, not a human-rater measurement.

Offline reproduction: .venv/bin/python research/question_equivalence.py
All performance/equivalence values are simulated. Optional Kalshi inputs are
read from a separately acquired public-data cache and never used in simulation.
"""
from itertools import combinations
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from surveyequivalence._equivalence import supported_equivalence

OUTPUT = ROOT / 'benchmarks/results/question-equivalence-draft'
PDF = ROOT / 'output/pdf/question_predictive_equivalence_draft.pdf'
SEED, TRAIN, TEST, BOOTSTRAPS = 1729, 2400, 1600, 500
COLORS = ['#126D91', '#B45B32', '#7063A3']
INK, MUTED, GRID = '#172A39', '#536674', '#DCE4E9'
SUBSETS = [s for k in range(4) for s in combinations(range(3), k)]


def name(subset):
    return ' + '.join(f'Q{j+1}' for j in subset) if subset else 'No questions'


def simulate():
    rng = np.random.default_rng(SEED)
    latent = rng.normal(size=TRAIN + TEST)
    outcomes = rng.binomial(1, expit(1.8 * latent))
    shared_noise = .25 * rng.normal(size=TRAIN + TEST)
    signals = (latent[:, None] + shared_noise[:, None]
               + rng.normal(size=(TRAIN + TEST, 3)) * [.55, .85, 1.4])
    counts = rng.binomial(120, expit(signals))
    responses = (counts + .5) / 121
    features = logit(responses)
    probabilities, coefficients = {}, {}
    train_prior = float(outcomes[:TRAIN].mean())
    for subset in SUBSETS:
        if not subset:
            probabilities[subset] = np.full(TEST, train_prior)
            continue
        predictor = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        predictor.fit(features[:TRAIN, list(subset)], outcomes[:TRAIN])
        probabilities[subset] = predictor.predict_proba(features[TRAIN:, list(subset)])[:, 1]
        scaler, model = predictor.steps[0][1], predictor.steps[1][1]
        assert model.n_iter_.max() < 2000
        coefficients[name(subset)] = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                                        intercept=model.intercept_.tolist(), weights=model.coef_.tolist())
    y = outcomes[TRAIN:]
    predictions = np.column_stack([probabilities[s] for s in SUBSETS])
    terms = 1 - (predictions - y[:, None]) ** 2
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    # Check the probability scoring convention against known values.
    assert 1 - (.5 - 1) ** 2 == .75
    assert 1 - (1 - 1) ** 2 == 1
    assert 1 - (0 - 1) ** 2 == 0
    draws = np.random.default_rng(SEED + 1).integers(0, TEST, size=(BOOTSTRAPS, TEST))
    scores = np.vstack([terms.mean(axis=0), terms[draws].mean(axis=1)])
    curves, equivalences = {}, {}
    for target in range(3):
        refs = [j for j in range(3) if j != target]
        curve = np.column_stack([scores[:, 0],
                                 scores[:, [SUBSETS.index((j,)) for j in refs]].mean(axis=1),
                                 scores[:, SUBSETS.index(tuple(refs))]])
        target_scores = scores[:, SUBSETS.index((target,))]
        # Independent arithmetic: score each eligible panel, then average scores.
        expected = []
        for k in range(3):
            expected.append(np.mean([
                sum(1 - (float(p) - int(t)) ** 2 for p, t in zip(probabilities[s], y)) / TEST
                for s in combinations(refs, k)]))
        np.testing.assert_allclose(curve[0], expected, atol=1e-13, rtol=0)
        estimates = [supported_equivalence(float(t), dict(enumerate(c)))
                     for t, c in zip(target_scores, curve)]
        actual = estimates[0]
        if actual['status'] == 'crossing' and actual['estimate'] > 0:
            x = actual['estimate']
            left, right = int(np.floor(x)), int(np.ceil(x))
            recovered = curve[0, left] if left == right else np.interp(x, [left, right], curve[0, [left, right]])
            np.testing.assert_allclose(recovered, target_scores[0], atol=1e-13, rtol=0)
        curves[target] = curve
        equivalences[target] = dict(reference_questions=[f'Q{j+1}' for j in refs],
            actual=actual, bootstrap_status_counts={s: sum(e['status'] == s for e in estimates[1:])
                for s in sorted({e['status'] for e in estimates[1:]})},
            bootstrap_equivalences=estimates[1:])
    frame = pd.DataFrame(dict(case_id=[f'sim-{i:04d}' for i in range(TRAIN + TEST)],
                              split=['train'] * TRAIN + ['test'] * TEST,
                              outcome=outcomes))
    for j in range(3):
        frame[f'Q{j+1}_yes_responses'] = counts[:, j]
        frame[f'Q{j+1}_respondents'] = 120
    frame.to_csv(OUTPUT / 'synthetic_cases.csv', index=False)
    pd.DataFrame(predictions, columns=[name(s) for s in SUBSETS]).assign(outcome=y).to_csv(
        OUTPUT / 'synthetic_test_predictions.csv', index=False)
    pd.DataFrame(scores, columns=[name(s) for s in SUBSETS]).to_csv(
        OUTPUT / 'synthetic_bootstrap_scores.csv', index_label='replicate_0_is_original')
    rows = []
    for target, curve in curves.items():
        for k in range(3):
            lo, hi = np.quantile(curve[1:, k], [.025, .975])
            rows.append(dict(target=f'Q{target+1}', reference_pool=name(tuple(j for j in range(3) if j != target)),
                             panel_size=k, score=curve[0, k], lower=lo, upper=hi))
    pd.DataFrame(rows).to_csv(OUTPUT / 'synthetic_reference_curves.csv', index=False)
    manifest = dict(data_kind='SYNTHETIC; no Pew or Kalshi estimates', seed=SEED,
                    train_cases=TRAIN, test_cases=TEST, bootstrap_replicates=BOOTSTRAPS,
                    scoring='1 - mean((predicted_probability - binary_outcome)^2)',
                    model='standardized response logits -> L2 logistic regression, C=1; fit on train only',
                    panel_unit='question; not individual respondent or trader',
                    combination='separate supervised predictor fitted for every nonempty subset',
                    reference_sampling='uniform average of scores across same-size subsets excluding the target question',
                    fractional_interpretation='linear interpolation; randomized mixture of adjacent panel sizes',
                    uncertainty='paired test-case bootstrap, conditional on fixed fitted models; no equivalence CI reported',
                    training_prior=train_prior, models=coefficients, equivalences=equivalences,
                    checked={'probability_bounds': True, 'optimizer_converged': True,
                             'reference_curve_independent_arithmetic': True, 'crossing_reconstruction': True,
                             'train_test_disjoint': True})
    (OUTPUT / 'synthetic_results.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return scores, curves, equivalences


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
        'axes.labelcolor': INK, 'axes.edgecolor': GRID, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
        'axes.titlepad': 15, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'svg.fonttype': 'none', 'pdf.fonttype': 42})


def header(fig, title, subtitle, tag):
    fig.text(.07, .945, title, fontsize=22, weight='bold', va='top')
    fig.text(.07, .887, subtitle, fontsize=11, color=MUTED, va='top')
    fig.text(.93, .973, tag, fontsize=9, ha='right', color=MUTED, weight='bold')


def footer(fig, text):
    fig.text(.07, .038, text, fontsize=9, color=MUTED, va='bottom', linespacing=1.5)


def box(ax, x, y, w, h, title, body, color=INK, fill='#F1F5F7'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=.015',
                               lw=1.1, edgecolor=color, facecolor=fill))
    ax.text(x + w/2, y + h*.69, title, ha='center', va='center', color=color, fontsize=12, weight='bold')
    ax.text(x + w/2, y + h*.30, body, ha='center', va='center', color=INK, fontsize=10, linespacing=1.5)


def arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.6,
                                                       shrinkA=5, shrinkB=5))


def setup_figure():
    fig = plt.figure(figsize=(12, 8))
    header(fig, 'How predictive is a question?',
           'Working interpretation: questions supply signals; a later observed outcome is the target.',
           'CONCEPTUAL DRAFT / PANEL DEFINITION PENDING')
    ax = fig.add_axes([.07, .22, .86, .60]); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    for j, y in enumerate([.75, .43, .11]):
        box(ax, .015, y, .25, .19, f'Question {j+1}', 'Responses or a market price\nrecorded before the outcome', COLORS[j])
        arrow(ax, (.28, y+.09), (.39, .55))
    box(ax, .40, .42, .23, .26, 'Predict the same outcome', 'One question or a subset\nPrediction rule fitted on\ntraining cases only')
    arrow(ax, (.645, .55), (.75, .55))
    box(ax, .76, .42, .225, .26, 'Score on held-out cases', 'Predicted probability\ncompared with\nobserved outcome')
    box(ax, .76, .03, .225, .22, 'Observed outcome', 'Collected after\nthe forecast cutoff', '#426B60', '#EFF6F2')
    arrow(ax, (.87, .26), (.87, .41))
    ax.text(.40, .85, 'Repeat over many independent outcome cases', fontsize=12, weight='bold')
    ax.text(.40, .23, 'Compare a single question with\npanels drawn from the other questions.', fontsize=12, linespacing=1.6)
    fig.text(.07, .155, 'Equivalence asks: what reference-panel size matches the question\'s predictive score?',
             fontsize=13, weight='bold')
    footer(fig, 'Provisional unit: one question. A market price aggregates traders; it is not one human rating.\n'
           'Real data must define the outcome cases, forecast cutoff, question identities, and reference pool before scoring.')
    return fig


def performance_figure(scores):
    fig = plt.figure(figsize=(12, 8))
    header(fig, 'Compare predictive performance on the same cases',
           f'{TRAIN:,} training cases / {TEST:,} held-out cases / a separately fitted predictor for every question subset',
           'SIMULATED DATA / NOT PEW OR KALSHI RESULTS')
    ax = fig.add_axes([.23, .22, .64, .57])
    lo, hi = np.quantile(scores[1:], [.025, .975], axis=0)
    for i, subset in enumerate(SUBSETS):
        c = COLORS[subset[0]] if len(subset) == 1 else INK
        ax.plot([lo[i], hi[i]], [i, i], color=c, lw=2)
        ax.plot(scores[0, i], i, 'o', ms=8, color=c)
        ax.annotate(f'{scores[0,i]:.3f}', (hi[i], i), xytext=(10, 0), textcoords='offset points', va='center', fontsize=10)
    ax.axvline(scores[0, 0], color=MUTED, ls=':', lw=1.2)
    ax.set(yticks=range(len(SUBSETS)), yticklabels=[name(s) for s in SUBSETS], ylim=(7.65, -.65),
           xlabel='Predictive score: 1 - Brier loss (higher is better)')
    pad = (hi.max() - lo.min()) * .15
    ax.set_xlim(lo.min()-pad*.4, hi.max()+pad)
    ax.grid(axis='x', color=GRID, lw=.8); ax.set_axisbelow(True)
    for y in [.5, 3.5, 6.5]:
        ax.axhline(y, color=GRID, lw=.7)
    footer(fig, 'Dots: held-out scores. Lines: 95% paired test-case bootstrap intervals, conditional on fixed fitted models.\n'
           'Zero questions uses the training outcome frequency. These synthetic differences describe the simulation only.\n'
           'Panels use jointly fitted logistic predictors; no human-equivalence or information-equivalence claim is made.')
    return fig


def equivalence_figure(scores, curves, equivalences):
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))
    fig.subplots_adjust(left=.09, right=.94, top=.72, bottom=.28, wspace=.30)
    header(fig, 'Express each question in reference-panel units',
           'Each benchmark excludes the question being evaluated. The reference pool changes between panels.',
           'SIMULATED DATA / QUESTION-PANEL EQUIVALENCE')
    all_values = []
    for target in range(3):
        all_values.extend(np.quantile(curves[target][1:], [.025, .975], axis=0).ravel())
        all_values.extend(np.quantile(scores[1:, SUBSETS.index((target,))], [.025, .975]))
    low, high = min(all_values), max(all_values); pad = (high-low)*.12
    for target, ax in enumerate(axes):
        curve = curves[target]; c = COLORS[target]
        target_values = scores[:, SUBSETS.index((target,))]
        lo, hi = np.quantile(curve[1:], [.025, .975], axis=0)
        slo, shi = np.quantile(target_values[1:], [.025, .975])
        ax.axhspan(slo, shi, color=c, alpha=.08)
        ax.fill_between(range(3), lo, hi, color=INK, alpha=.10)
        ax.plot(range(3), curve[0], color=INK, lw=2, marker='o', label='Reference panel')
        ax.axhline(target_values[0], color=c, lw=2, ls='--', label=f'Q{target+1} alone')
        eq = equivalences[target]['actual']
        if eq['status'] == 'crossing':
            x = eq['estimate']
            ax.vlines(x, low-pad, target_values[0], color=c, lw=1.3, ls=':')
            ax.plot(x, target_values[0], marker='*', color=c, markersize=14, zorder=4)
            result = f'{x:.2f} reference questions\n(interpolated)'
        elif eq['status'] == 'above_range':
            result = 'Above reference curve\n(maximum 2 questions)'
        else:
            result = 'No crossing in 0-2\nreference questions'
        refs = ' + '.join(equivalences[target]['reference_questions'])
        ax.set(title=f'Q{target+1} vs. {refs}', xticks=[0, 1, 2], xlim=(-.06, 2.06),
               ylim=(low-pad, high+pad), xlabel='Reference questions (k)')
        if target == 0:
            ax.set_ylabel('Predictive score: 1 - Brier loss')
        ax.grid(color=GRID, lw=.7); ax.set_axisbelow(True)
        ax.text(.5, -.23, result, transform=ax.transAxes, fontsize=11, ha='center', va='top', color=c, weight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(-.02, 1.28), frameon=False, fontsize=9)
    footer(fig, 'At k=1, average the scores of the two eligible single-question predictors; at k=2, use their joint predictor.\n'
           'Fractional k is interpolation (a mixture of adjacent panel sizes), not a literal fraction of a question. No extrapolation.\n'
           'Shading: conditional 95% score intervals, not equivalence intervals. Different reference pools do not give a common ranking.')
    return fig


def load_kalshi_example():
    """Keep real market prices separate from all synthetic performance results."""
    cache = ROOT / '.cache/question-equivalence/kalshi'
    saved = OUTPUT / 'kalshi_example.json'
    if not (cache / 'manifest.json').exists():
        return json.loads(saved.read_text()) if saved.exists() else None
    manifest = json.loads((cache / 'manifest.json').read_text())
    ticker = manifest['selected_event_ticker']
    event = json.loads((cache / f'event_{ticker}.json').read_text())
    cutoff = int(datetime.fromisoformat(manifest['pre_event_cutoff_utc']).timestamp())
    markets = sorted(event['markets'], key=lambda m: m.get('floor_strike')
                     if m.get('floor_strike') is not None else -np.inf)
    values = {m['expiration_value'] for m in markets}
    assert len(values) == 1 and sum(m['result'] == 'yes' for m in markets) == 1
    rows, histories = [], []
    def dollars(candle, field):
        value = candle[field].get('close_dollars', candle[field].get('close'))
        return float(value) if value is not None else None
    for market in markets:
        candles = json.loads((cache / f"candles_{market['ticker']}.json").read_text())['candlesticks']
        before = sorted((c for c in candles if c['end_period_ts'] <= cutoff), key=lambda c:c['end_period_ts'])
        snapshot = before[-1]
        # An exact candle boundary avoids mixing different lookback ages.
        assert snapshot['end_period_ts'] == cutoff
        last, bid, ask = [dollars(snapshot, f) for f in ['price', 'yes_bid', 'yes_ask']]
        assert all(v is not None and 0 <= v <= 1 for v in [last, bid, ask]) and bid <= ask
        if market['strike_type'] == 'less':
            label = f"{int(market['cap_strike'])-1} F or below"
        elif market['strike_type'] == 'greater':
            label = f"{int(market['floor_strike'])+1} F or above"
        else:
            label = f"{int(market['floor_strike'])}-{int(market['cap_strike'])} F"
        rows.append(dict(ticker=market['ticker'], label=label, result=market['result'],
                         last_trade_price_dollars=last, yes_bid_dollars=bid, yes_ask_dollars=ask,
                         candle_end_ts=cutoff, floor_strike=market.get('floor_strike'),
                         cap_strike=market.get('cap_strike'), strike_type=market['strike_type']))
        histories.append(dict(ticker=market['ticker'], label=label,
                              candles=[dict(end_period_ts=c['end_period_ts'],
                                            last_trade_price_dollars=dollars(c, 'price')) for c in before]))
    for request in manifest['requests']:
        request['sha256'] = hashlib.sha256((ROOT / request['file']).read_bytes()).hexdigest()
    result = dict(data_kind='REAL PUBLIC KALSHI DATA; descriptive single-event example only',
                  event_ticker=ticker, title=event['event']['title'],
                  observed_temperature_f=float(next(iter(values))),
                  settlement_sources=event['event']['settlement_sources'],
                  snapshot=rows, histories=histories, provenance=manifest,
                  transformation='raw dollar prices; not normalized or treated as calibrated probabilities',
                  equivalence_estimated=False)
    saved.write_text(json.dumps(result, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(OUTPUT / 'kalshi_midnight_snapshot.csv', index=False)
    return result


def kalshi_figure(data):
    import matplotlib.dates as mdates
    colors = ['#AAB5BD', '#667684', '#126D91', '#B45B32', '#426B60', '#7063A3']
    fig = plt.figure(figsize=(12, 8))
    header(fig, 'Kalshi: several questions, one observed outcome',
           'NYC maximum temperature / September 21, 2026 / six related temperature-range contracts',
           'REAL MARKET DATA / DESCRIPTIVE EXAMPLE')
    left = fig.add_axes([.09, .29, .43, .44]); right = fig.add_axes([.72, .29, .20, .44])
    ny = ZoneInfo('America/New_York')
    all_times = []
    for row, history, c in zip(data['snapshot'], data['histories'], colors):
        times = [datetime.fromtimestamp(v['end_period_ts'], ny) for v in history['candles']]
        prices = [100*v['last_trade_price_dollars'] if v['last_trade_price_dollars'] is not None else np.nan
                  for v in history['candles']]
        all_times.extend(times)
        left.plot(times, prices, color=c, lw=2, label=row['label'])
    left.set(ylabel='YES last-trade price (cents)', ylim=(-1, 68),
             xlabel='September 20 local time / 00:00 begins Sep 21', title='Price history before the forecast cutoff')
    left.xaxis.set_major_locator(mdates.HourLocator(byhour=[12, 16, 20, 0], tz=ny))
    left.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=ny))
    left.set_xlim(min(all_times), max(all_times)); left.grid(color=GRID, lw=.7); left.set_axisbelow(True)
    for i, (row, c) in enumerate(zip(data['snapshot'], colors)):
        last, bid, ask = (row[k]*100 for k in ['last_trade_price_dollars', 'yes_bid_dollars', 'yes_ask_dollars'])
        right.plot([bid, ask], [i, i], color=c, lw=4, solid_capstyle='round')
        right.plot(last, i, 'o', ms=7, color=c)
        right.annotate(f'{last:.0f}c', (max(ask,last), i), xytext=(8, 0), textcoords='offset points', va='center', fontsize=10)
        if row['result'] == 'yes':
            right.axhspan(i-.35, i+.35, color='#426B60', alpha=.09)
            right.text(.98, i-.36, 'RESOLVED YES', transform=right.get_yaxis_transform(),
                       ha='right', color='#426B60', fontsize=8, weight='bold')
    right.set(yticks=range(len(data['snapshot'])), yticklabels=[r['label'] for r in data['snapshot']],
              ylim=(5.7, -.7), xlim=(-1, 68), xticks=[0, 20, 40, 60],
              xlabel='YES price (cents)', title='Midnight snapshot')
    right.grid(axis='x', color=GRID, lw=.7); right.set_axisbelow(True)
    fig.legend(*left.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.08, .83),
               ncol=3, fontsize=9, frameon=False)
    winning_label = next(r['label'] for r in data['snapshot'] if r['result'] == 'yes')
    fig.text(.09, .18, f"Recorded outcome: {data['observed_temperature_f']:.0f} F / the {winning_label} contract resolved YES",
             fontsize=13, color='#426B60', weight='bold')
    total = sum(r['last_trade_price_dollars'] for r in data['snapshot'])
    footer(fig, f'Raw hourly closing trade prices; snapshot at Sep 21, 00:00 America/New_York. Short bars show the closing bid/ask spread.\n'
           f'Snapshot YES prices total ${total:.2f}; no normalization. Mutually exclusive contracts are not independent forecasters.\n'
           'Gaps indicate no reported trade close. Last-trade ages can differ even at a common candle boundary.\n'
           f'Source: public Kalshi API, {data["event_ticker"]}; settlement source: The Weather Company. One event does not estimate equivalence.')
    return fig


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); PDF.parent.mkdir(parents=True, exist_ok=True)
    scores, curves, equivalences = simulate(); style()
    figures = [('question_prediction_setup', setup_figure())]
    kalshi = load_kalshi_example()
    if kalshi:
        figures.append(('kalshi_question_example', kalshi_figure(kalshi)))
    figures.extend([('synthetic_question_performance', performance_figure(scores)),
               ('synthetic_question_equivalence', equivalence_figure(scores, curves, equivalences))]
    )
    with PdfPages(PDF, metadata={'Title': 'Question predictive equivalence - provisional visualizations',
                                'Subject': 'Simulated methodology demonstration; no human-rater-equivalence claim'}) as pdf:
        for filename, fig in figures:
            fig.savefig(OUTPUT / f'{filename}.png', dpi=170)
            fig.savefig(OUTPUT / f'{filename}.svg')
            pdf.savefig(fig)
            plt.close(fig)
    print(json.dumps({'pdf': str(PDF), 'figures': len(figures),
                      'equivalences': {f'Q{k+1}': v['actual'] for k, v in equivalences.items()}}, indent=2))


if __name__ == '__main__':
    main()
