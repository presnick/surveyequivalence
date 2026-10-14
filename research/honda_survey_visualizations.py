"""Observed human-panel accuracy across Honda et al.'s response formats.

No model fitting or generated human responses. The primary curves are exact
finite-pool expectations under uniform sampling of distinct people. Bootstrap
uncertainty resamples whole participant rows, keeping the nine images fixed.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

try:
    from .panel_power import equivalence_label, first_equivalent_size
except ImportError:  # Direct script execution.
    from panel_power import equivalence_label, first_equivalent_size

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/honda-survey-formats"
PDF = ROOT / "output/pdf/honda_survey_format_equivalence.pdf"
INPUT = ROOT / ".cache/survey-type-data/nonpolitical/honda_Experiment1.xlsx"
SOURCE_HASH = "eab733c6545403e8eba3c63bdac9501d95225d81ceb87cf0ccf3311a25f63f5f"
TRUTH = np.array([183, 287, 360, 453, 554, 633, 719, 807, 986], dtype=float)
GROUPS = ["Number"] + [f"Scale {i}" for i in range(1, 7)]
COUNTS = [128, 128, 133, 126, 126, 130, 130]
LABELS = ["Type a number", "Drag / endpoints", "Click / endpoints",
          "Drag / midpoint", "Click / midpoint", "Drag / quartiles", "Click / quartiles"]
COLORS = ["#202C3B", "#3485B1", "#84B4CD", "#127E69", "#6AAF9D", "#975AAF", "#C29BCD"]
STEMS = ["panel_accuracy", "equivalent_panel_sizes", "question_comparison"]
KS = np.arange(1, 101)
REFERENCE_K = 10
SEED = 23092026
BOOTSTRAPS = 2000
INK, MUTED, GRID = "#202C3B", "#586879", "#DEE5EB"


def load_data(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError("Source workbook differs from the verified OSF release")
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as z:
        strings = ["".join(n.itertext()) for n in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    grouped = {name: [] for name in GROUPS}
    for row in sheet.findall("s:sheetData/s:row", ns):
        if int(row.get("r")) < 4:
            continue
        cells = {}
        for c in row.findall("s:c", ns):
            value = c.find("s:v", ns)
            if value is not None:
                cells[c.get("r").rstrip("0123456789")] = (
                    strings[int(value.text)] if c.get("t") == "s" else float(value.text))
        if cells:
            grouped[cells["S"]].append([cells[chr(65 + j)] for j in range(9)])
    arrays = [np.asarray(grouped[name]) for name in GROUPS]
    assert [len(a) for a in arrays] == COUNTS
    assert all(a.shape[1] == 9 and np.isfinite(a).all() and
               ((0 <= a) & (a <= 1000)).all() for a in arrays)
    return arrays


def item_mse(a, ks):
    """Exact E[(mean of k distinct rows - truth)^2], one column per image."""
    return (a.mean(axis=0) - TRUTH) ** 2 + np.outer(
        1 / np.asarray(ks, dtype=float) - 1 / len(a), a.var(axis=0, ddof=1))


def needed(curves, targets):
    """First supported integer k meeting the target; inf denotes >100."""
    curves = np.atleast_2d(curves)
    targets = np.asarray(targets).reshape(-1, 1)
    qualifies = curves <= targets + 1e-8
    return np.where(qualifies.any(axis=1), qualifies.argmax(axis=1) + 1, np.inf)


def extended_interval(values):
    """Order-statistic bounds retain censored +inf, without dropping draws."""
    values = np.sort(values)
    return values[int(np.floor(.025 * (len(values) - 1)))], values[int(np.ceil(.975 * (len(values) - 1)))]


def serial_bound(x):
    return int(x) if np.isfinite(x) else ">100"


def check_formula(arrays, curves):
    checks = {}
    for name, a, curve in zip(GROUPS, arrays, curves):
        direct_one = ((a - TRUTH) ** 2).mean()
        ii, jj = np.triu_indices(len(a), k=1)
        direct_two = (((a[ii] + a[jj]) / 2 - TRUTH) ** 2).mean()
        full_pool = ((a.mean(axis=0) - TRUTH) ** 2).mean()
        np.testing.assert_allclose(curve[:2], [direct_one, direct_two], rtol=1e-12)
        np.testing.assert_allclose(item_mse(a, [len(a)]).mean(), full_pool, rtol=1e-12)
        assert np.all(np.diff(curve) <= 1e-8)
        checks[name] = {"k1_direct": float(direct_one), "k2_all_pairs": float(direct_two),
                        "all_person_pairs": len(ii), "full_pool_mse": float(full_pool)}
    return checks


def analyze(arrays):
    per_item = np.stack([item_mse(a, KS) for a in arrays])
    curves = per_item.mean(axis=2)
    checks = check_formula(arrays, curves)
    target = curves[0, REFERENCE_K - 1]
    equivalent = needed(curves, np.full(7, target))
    assert equivalent[0] == REFERENCE_K
    # Each draw resamples people within formats. All nine answers travel with
    # the selected person. The same draw's Number target is used for all formats.
    rng = np.random.default_rng(SEED)
    boot_curves = np.empty((BOOTSTRAPS, 7, len(KS)))
    for g, a in enumerate(arrays):
        for start in range(0, BOOTSTRAPS, 100):
            count = min(100, BOOTSTRAPS - start)
            sampled = a[rng.integers(0, len(a), size=(count, len(a)))]
            bias_sq = ((sampled.mean(axis=1) - TRUTH) ** 2).mean(axis=1)
            variance = sampled.var(axis=1, ddof=1).mean(axis=1)
            boot_curves[start:start+count, g] = (bias_sq[:, None] +
                variance[:, None] * (1 / KS[None, :] - 1 / len(a)))
    boot_targets = boot_curves[:, 0, REFERENCE_K - 1]
    boot_equiv = np.column_stack([needed(boot_curves[:, g], boot_targets) for g in range(7)])
    assert np.all(boot_equiv[:, 0] == REFERENCE_K)
    intervals = [extended_interval(boot_equiv[:, g]) for g in range(7)]
    rmse_lo, rmse_hi = np.quantile(np.sqrt(boot_curves), [.025, .975], axis=0)
    rows = []
    for g in range(7):
        rows.append({"format": GROUPS[g], "description": LABELS[g], "people": COUNTS[g],
                     "rms_error_k1": float(np.sqrt(curves[g, 0])),
                     "rms_error_k10": float(np.sqrt(curves[g, 9])),
                     "rms_error_k100": float(np.sqrt(curves[g, 99])),
                     "needed_to_match_10_number": serial_bound(equivalent[g]),
                     "bootstrap_95_low": serial_bound(intervals[g][0]),
                     "bootstrap_95_high": serial_bound(intervals[g][1]),
                     "bootstrap_fraction_above_100": float(np.isinf(boot_equiv[:, g]).mean())})
    metadata = {"data_kind": "Observed individual human responses; no synthetic labels",
        "source": "https://osf.io/sxdf9/", "paper": "https://www.nature.com/articles/s41598-022-11900-7",
        "download": "https://osf.io/download/uxrhd/", "source_sha256": SOURCE_HASH,
        "people": sum(COUNTS), "items": 9, "truth": TRUTH.tolist(),
        "score": "sqrt(E_panel[mean_image((panel_mean - truth)^2)]) in dots; lower is better",
        "formula": "MSE_g(k) = mean_q[(mean_gq-truth_q)^2 + sample_variance_gq*(1/k - 1/n_g)]",
        "sampling": "Uniform without replacement among distinct released participant rows; same people across all nine images",
        "weighting": "Equal weight for each of nine images; all formats share the same targets",
        "reference": {"format": "Number", "panel_size": REFERENCE_K, "rms_error": float(np.sqrt(target))},
        "equivalence": "Smallest integer k with expected MSE no greater than Number at k=10; no extrapolation beyond k=100",
        "uncertainty": "2000 percentile bootstrap replicates of whole people within format, nine images fixed; target recomputed in each draw; order-statistic bounds retain >100 draws",
        "seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "format_results": rows,
        "checks": checks,
        "limitations": ["Finite observed respondent pools and only nine deliberately chosen images",
                        "Bootstrap does not measure uncertainty across new images",
                        "Response formats differ, not substantive question content",
                        "Distinct row records cannot establish identities across experiments",
                        "No causal assignment claim or future-event forecasting claim",
                        "Primary RMS scoring differs from the earlier proposed MAE; chosen for exact finite-pool curves"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    with (OUT / "panel_curves.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["format", "panel_size", "expected_mse", "rms_error", "bootstrap_95_low", "bootstrap_95_high"])
        for g in range(7):
            for j, k in enumerate(KS):
                writer.writerow([GROUPS[g], k, curves[g, j], np.sqrt(curves[g, j]), rmse_lo[g, j], rmse_hi[g, j]])
    with (OUT / "equivalent_panels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "question_errors.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["format", "question", "truth", "panel_size", "rms_error", "change_vs_number_percent"])
        for g in range(7):
            for q in range(9):
                error = np.sqrt(per_item[g, 9, q])
                writer.writerow([GROUPS[g], f"Q{q+1}", int(TRUTH[q]), 10, error,
                                 100 * (error / np.sqrt(per_item[0, 9, q]) - 1)])
    np.savez_compressed(OUT / "bootstrap_draws.npz", panel_equivalences=boot_equiv,
                        reference_rms=np.sqrt(boot_targets))
    return curves, per_item, metadata, intervals


def plot_all(curves, per_item, meta, intervals, panel_only=False):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none"})

    def page(title, subtitle, number):
        fig = plt.figure(figsize=(13, 8.5))
        fig.text(.07, .94, title, fontsize=23, weight="bold", va="top")
        fig.text(.07, .881, subtitle, fontsize=11, color=MUTED, va="top", linespacing=1.55)
        fig.text(.93, .978, f"HONDA 2022  /  EXPERIMENT 1  /  {number:02d}",
                 fontsize=9, color=MUTED, ha="right")
        return fig

    def footer(fig, note):
        fig.text(.07, .071, note, fontsize=9, color=MUTED, va="top", linespacing=1.55)
        fig.text(.07, .023, "Source: Honda, Kagawa & Shirasuna (2022)  |  Data: osf.io/sxdf9  |  Analysis: 23 Sep 2026",
                 fontsize=8.5, color=MUTED)

    def save(fig, stem):
        fig.savefig(OUT / f"{stem}.png", dpi=200)
        fig.savefig(OUT / f"{stem}.svg")
        plt.close(fig)

    fig = page("Rater-equivalence power curves",
               "901 people estimated the same nine dot counts. Higher score means a more accurate panel.\n"
               "Match the score of 10 typed responses; read each format's equivalent panel size at its first crossing.", 1)
    ax = fig.add_axes([.09, .20, .49, .56])
    target = meta["reference"]["rms_error"]
    # Negate the averaged squared loss before interpolation. Negating RMS would
    # preserve integer thresholds but change the paper's fractional crossings.
    scores = -curves / 1000
    target_score = scores[0, REFERENCE_K - 1]
    equivalents = [first_equivalent_size(KS, score, target_score) for score in scores]
    zoom = ax.inset_axes([.36, .10, .60, .43])
    for chart in [ax, zoom]:
        for g, score in enumerate(scores):
            chart.plot(KS, score, color=COLORS[g], lw=2.5 if g == 0 else 1.8,
                       ls="--" if g in [2, 4, 6] else "-", zorder=5 if g == 0 else 3)
        chart.axhline(target_score, color="#AB7444", lw=1.3, ls="--", zorder=2)
        chart.grid(axis="y", color=GRID, lw=.7)
    ax.set(xlim=(1, 100), ylim=(np.min(scores) * 1.06, 0),
           xlabel="People in the panel (k)", ylabel="Expected score (-MSE; thousands of dots²)",
           xticks=[1, 20, 40, 60, 80, 100])
    zoom.set(xlim=(2.5, 11), ylim=(-31, -6), xticks=[3, 5, 7, 10], yticks=[-30, -20, -10])
    zoom.tick_params(labelsize=8)
    zoom.set_title("First crossings (detail)", fontsize=9, loc="left", pad=6)
    for g, eq in enumerate(equivalents):
        if np.isfinite(eq):
            for chart in [ax, zoom]:
                chart.vlines(eq, chart.get_ylim()[0], target_score, color=COLORS[g],
                             linestyle=":", linewidth=1, alpha=.7)
                chart.scatter(eq, target_score, s=32, color=COLORS[g],
                              edgecolors="white", linewidths=.5, zorder=7)
    fig.text(.635, .746, "RESPONSE FORMAT", fontsize=10, weight="bold", color=MUTED)
    fig.text(.94, .746, "EQUIVALENT k", fontsize=10, weight="bold", color=MUTED, ha="right")
    for g in range(7):
        y = .695 - .057 * g
        fig.add_artist(Line2D([.635, .665], [y, y], transform=fig.transFigure,
                            color=COLORS[g], lw=3, ls="--" if g in [2, 4, 6] else "-"))
        fig.text(.677, y, LABELS[g], fontsize=9.8, va="center")
        fig.text(.94, y, equivalence_label(equivalents[g], KS), fontsize=11,
                 weight="bold", va="center", ha="right", color=COLORS[g])
    fig.text(.635, .272, "Reference: 10 typed responses", fontsize=12, weight="bold")
    fig.text(.635, .233, f"Score {target_score:.2f}  |  RMS error {target:.1f} dots",
             fontsize=10, color="#946234")
    fig.text(.635, .184, "Fractional k: linear interpolation in expected score.\n"
             "The next page reports whole-person thresholds.", fontsize=9.5, color=MUTED, linespacing=1.5)
    footer(fig, "Score = negative expected squared error; zero is perfect. Straight segments connect exact finite-pool scores at k = 1-100.\n"
           "Paper-style crossings applied to known truths; no fitted logarithm or extrapolation. Source: arXiv:2106.01254v3, Sec. 5.3.")
    save(fig, STEMS[0])
    if panel_only:
        return

    fig = page("How many people match 10 typed responses?",
               f"Target: no more than {target:.1f} dots RMS error. Dots show the smallest qualifying whole-person panel.\n"
               "Lines show 95% respondent-bootstrap intervals, including uncertainty in the reference target.", 2)
    ax = fig.add_axes([.28, .26, .56, .49])
    cap = 100
    for g, result in enumerate(meta["format_results"]):
        lo, hi = intervals[g]
        value = result["needed_to_match_10_number"]
        ax.plot([min(lo, cap), min(hi, cap)], [g, g], color=COLORS[g], lw=3, alpha=.65)
        ax.scatter(value if isinstance(value, int) else cap, g, color=COLORS[g], s=85,
                   edgecolors="white", zorder=3)
        if not np.isfinite(hi) or hi > cap:
            ax.annotate("", xy=(cap, g), xytext=(80, g), arrowprops={"arrowstyle": "->", "color": COLORS[g], "lw": 2})
        interval_label = "Reference" if g == 0 else f"{serial_bound(lo)} to {serial_bound(hi)}"
        ax.text(1.045, g, f"{value} people", transform=ax.get_yaxis_transform(), va="center", fontsize=11, weight="bold")
        ax.text(1.045, g + .27, interval_label, transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=MUTED)
    ax.axvline(10, color=INK, lw=1, ls=":", alpha=.65)
    ax.set(ylim=(6.55, -.65), xscale="log", xlim=(1, 110),
           xlabel="People needed to match the reference (logarithmic scale)")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels([1, 2, 5, 10, 20, 50, 100])
    ax.minorticks_off()
    ax.set_yticks(range(7))
    ax.set_yticklabels(LABELS)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    fig.text(.28, .151, "Point estimates: 4-7 scale respondents match 10 typed-number respondents.",
             fontsize=12, weight="bold")
    footer(fig, "2,000 bootstrap draws resample whole people; the nine images stay fixed. Arrows indicate an upper bound beyond 100.\n"
           "Intervals do not describe new questions. Bootstrap draws without a crossing are retained, not extrapolated.")
    save(fig, STEMS[1])

    fig = page("The comparison changes by question",
               "All panels have 10 people. Cells show the percent change in RMS error relative to typing a number.\n"
               "Negative values favor the scale format; positive values favor typing a number.", 3)
    ratio = (np.sqrt(per_item[1:, 9, :]).T / np.sqrt(per_item[0, 9, :])[:, None] - 1) * 100
    vmax = max(50, np.ceil(np.abs(ratio).max() / 10) * 10)
    cmap = LinearSegmentedColormap.from_list("errors", ["#127E69", "#F5F6F7", "#B95739"])
    ax = fig.add_axes([.18, .23, .60, .51])
    im = ax.imshow(ratio, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    for q in range(9):
        for g in range(6):
            label = f"{ratio[q, g]:+.1f}%" if abs(ratio[q, g]) < 1 else f"{ratio[q, g]:+.0f}%"
            ax.text(g, q, label, ha="center", va="center", fontsize=11,
                    color="white" if abs(ratio[q, g]) > .6 * vmax else INK)
    ax.set_yticks(range(9))
    ax.set_yticklabels([f"Q{q+1}   /   {int(v)} dots" for q, v in enumerate(TRUTH)])
    ax.set_xticks(range(6))
    ax.set_xticklabels(["Drag\nendpoints", "Click\nendpoints", "Drag\nmidpoint", "Click\nmidpoint", "Drag\nquartiles", "Click\nquartiles"], fontsize=9.5)
    ax.tick_params(length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for q in range(9):
        ax.text(1.07, q, f"{np.sqrt(per_item[0, 9, q]):.1f}", transform=ax.get_yaxis_transform(), va="center", ha="center", fontsize=10)
    fig.text(.822, .757, "Typed-number\nRMS error (dots)", ha="center", fontsize=9, color=MUTED)
    cax = fig.add_axes([.28, .134, .4, .018])
    colorbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Change in error (%)", fontsize=9, labelpad=1)
    footer(fig, "These are descriptive comparisons for nine fixed images, not nine independent experiments.\n"
           "The overall curve gives each image equal weight before taking the square root.")
    save(fig, STEMS[2])


def write_pdf():
    from reportlab.pdfgen import canvas
    PDF.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(PDF), pagesize=(13 * 72, 8.5 * 72))
    c.setTitle("Human panel equivalence across survey response formats")
    c.setAuthor("SurveyEquivalence research")
    for stem in STEMS:
        c.drawImage(str(OUT / f"{stem}.png"), 0, 0, width=13*72, height=8.5*72)
        c.linkURL("https://osf.io/sxdf9/", (4.5*72, .12*72, 8*72, .34*72), relative=0)
        c.showPage()
    c.save()
    print(PDF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--plot-only", action="store_true", help="Redraw the power curve from saved results")
    parser.add_argument("--pdf-only", action="store_true", help="Package existing figures using ReportLab")
    args = parser.parse_args()
    if args.pdf_only:
        write_pdf()
    elif args.plot_only:
        metadata = json.loads((OUT / "results.json").read_text())
        with (OUT / "panel_curves.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        curves = []
        for name in GROUPS:
            group = sorted((r for r in rows if r["format"] == name), key=lambda r: int(r["panel_size"]))
            np.testing.assert_array_equal([int(r["panel_size"]) for r in group], KS)
            curves.append([float(r["expected_mse"]) for r in group])
        plot_all(np.asarray(curves), None, metadata, None, panel_only=True)
    else:
        arrays = load_data(args.input)
        curves, per_item, metadata, intervals = analyze(arrays)
        plot_all(curves, per_item, metadata, intervals)
        print(json.dumps(metadata["format_results"], indent=2))
