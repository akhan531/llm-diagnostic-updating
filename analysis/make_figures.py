"""Figures from results/derived/ (scored.csv, wbayes.csv). Requires matplotlib.

  fig_stated_vs_target.{pdf,png}  stated vs exact posttest in log-odds, one panel per model,
                                  colored AND shaped by result sign, probability tick labels
  fig_wbayes.{pdf,png}            fitted a and b with 95% scenario-bootstrap CIs, per model x
                                  condition, reference lines at 1 (exact Bayes)
Usage: python analysis/make_figures.py [--derived results/derived] [--out paper/figures]
Refuses to write into paper/figures if any input was synthetic.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, display_name, guard_output, guard_pilot, read_run_set, logit_clipped, read_csv, read_provenance  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # F3 (AAAI-27 kit): no Type 3 fonts in figures; embed TrueType (Type 42) Helvetica-metric text, >= 9 pt
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                                "font.sans-serif": ["Arimo", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
                                "font.size": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
except ImportError as e:  # pragma: no cover
    sys.exit(f"matplotlib is required for figures (pip install matplotlib): {e}")

CONDS = ["V", "N", "B", "S", "E"]
FIT_CONDS = ["V", "N", "B", "S"]
# Okabe-Ito (color-blind safe); marker and lightness differ too, so the plot survives grayscale
COLOR = {"V": "#000000", "N": "#0072B2", "B": "#D55E00", "S": "#009E73", "E": "#CC79A7"}
MARK = {"V": "o", "N": "s", "B": "^", "S": "D", "E": "v"}
LABEL = {"V": "V clean vignette", "N": "N note", "B": "B result buried", "S": "S shorthand",
         "E": "E negatives omitted"}


# W6 (committee #15): Figure 1 is coloured by result sign (the variable that matters), not by note
# condition (the null variable); marker shape still differs by sign so it survives grayscale.
RES_COLOR = {"pos": "#D55E00", "neg": "#0072B2"}
RES_MARK = {"pos": "o", "neg": "s"}
RES_LABEL = {"pos": "after a positive result", "neg": "after a negative result"}
PROB_TICKS = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]


def _prob_label(p):
    pct = 100 * p
    return f"{pct:g}%"


LIM = 8.0  # axis half-range in log-odds; points beyond it are drawn clipped to the edge with an "x" marker


def _clip(v):
    return max(-LIM, min(LIM, v))


def fig_stated_vs_target(scored, out):
    # (H1): the A5 calibration cases (condition V only) are not plotted; they are a separate table
    scored = [r for r in scored if r.get("calibration", "0") != "1"]
    models = sorted({r["model"] for r in scored})
    ncol = min(3, len(models))
    nrow = math.ceil(len(models) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 4.4), squeeze=False,
                             sharex=True, sharey=True)
    ticks = [math.log(p / (1 - p)) for p in PROB_TICKS]
    labels = [_prob_label(p) for p in PROB_TICKS]
    npts = 0
    n_clip = 0
    for ax, m in zip(axes.flat, models):
        for res in ("neg", "pos"):
            xs, ys = [], []
            for r in scored:
                if r["model"] == m and r["result"] == res and r["condition"] in CONDS and r["parse_ok"] == "1":
                    y = logit_clipped(float(r["posttest_stated"]) / 100.0)
                    xs.append(y - float(r["logodds_err"]))  # target log-odds
                    ys.append(y)
            npts += len(xs)
            inr = [i for i in range(len(xs)) if abs(xs[i]) <= LIM and abs(ys[i]) <= LIM]
            out_ = [i for i in range(len(xs)) if i not in set(inr)]
            n_clip += len(out_)
            ax.scatter([xs[i] for i in inr], [ys[i] for i in inr], s=9, marker=RES_MARK[res],
                       color=RES_COLOR[res], alpha=0.45, linewidths=0, label=RES_LABEL[res])
            if out_:  # out of range: clipped to the axis edge, drawn as an x in the same colour
                ax.scatter([_clip(xs[i]) for i in out_], [_clip(ys[i]) for i in out_], s=14, marker="x",
                           color=RES_COLOR[res], alpha=0.8, linewidths=0.7, clip_on=False,
                           label="_nolegend_")
        ax.plot([-12, 12], [-12, 12], color="0.35", lw=0.8, ls="--")
        ax.set_title(display_name(m), fontsize=9)
        ax.set_xlim(-LIM - 0.3, LIM + 0.3)
        ax.set_ylim(-LIM - 0.3, LIM + 0.3)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=9)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    for ax in list(axes.flat)[len(models) - ncol:len(models)]:  # bottom visible axes only
        ax.set_xlabel("Exact posttest probability", fontsize=9)
        ax.tick_params(axis="x", labelbottom=True)
        for t in ax.get_xticklabels():
            t.set_rotation(45)
    for ax in axes[:, 0]:
        ax.set_ylabel("Stated posttest probability", fontsize=9)
    # the empty sixth panel (5 models) holds the legend
    h, l = axes.flat[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], marker="x", ls="", color="0.3", ms=4, mew=0.8))
    l.append(f"outside plotted range, drawn at edge")
    if len(models) < nrow * ncol:
        list(axes.flat)[len(models)].legend(h, l, loc="center", fontsize=9, frameon=False, markerscale=1.5)
        fig.tight_layout()
    else:
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, -0.01), markerscale=1.4)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_stated_vs_target")
    return npts, n_clip


def fig_wbayes(wb, out):
    fits = [r for r in wb if r["variant"] == "published" and r["condition"] in FIT_CONDS
            and r["a"] != ""]
    assert fits, "no published-variant fits to plot"
    models = sorted({r["model"] for r in fits})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    w = 0.8 / len(FIT_CONDS)
    n = 0
    for ax, p, title in zip(axes, "ab", ("a (weight on prior; <1 base-rate neglect)",
                                         "b (weight on LR; <1 conservatism)")):
        for k, c in enumerate(FIT_CONDS):
            xs, ys, lo, hi = [], [], [], []
            for i, m in enumerate(models):
                r = [x for x in fits if x["model"] == m and x["condition"] == c]
                if not r or r[0][p] == "":
                    continue
                r = r[0]
                x = i - 0.4 + w * (k + 0.5)
                xs.append(x)
                ys.append(float(r[p]))
                lo.append(float(r[p]) - float(r[p + "_lo"]) if r[p + "_lo"] != "" else 0.0)
                hi.append(float(r[p + "_hi"]) - float(r[p]) if r[p + "_hi"] != "" else 0.0)
            n += len(xs)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[c], color=COLOR[c], ms=4.5, capsize=2,
                        lw=1, label=LABEL[c], mfc=COLOR[c] if c != "V" else "white")
        ax.axhline(1.0, color="0.3", lw=0.9, ls="--")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([display_name(m) for m in models], rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Fitted coefficient (exact Bayes = 1)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l[:len(FIT_CONDS)], loc="lower center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig_wbayes")
    return n


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_fig_macros(gen, total, clipped):
    os.makedirs(gen, exist_ok=True)
    with open(os.path.join(gen, "figure_macros.tex"), "w") as f:
        f.write("% GENERATED by analysis/make_figures.py from results/derived/scored.csv. Do not edit.\n"
                f"\\newcommand{{\\figOnePointsTotal}}{{{total}}}\n"
                f"\\newcommand{{\\figOnePointsDrawn}}{{{total - clipped}}}\n"
                f"\\newcommand{{\\figOnePointsClipped}}{{{clipped}}}\n"
                f"\\newcommand{{\\figOneAxisLim}}{{{int(LIM)}}}\n"
                f"\\newcommand{{\\figOneClippedPct}}{{{100.0 * clipped / total:.1f}}}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default=os.path.join(ROOT, "results", "derived"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures"))
    ap.add_argument("--generated", default=None, help="default: <out>/../generated")
    a = ap.parse_args(argv)
    syn = read_provenance(a.derived)
    guard_output(a.out, syn)
    guard_pilot(a.out, read_run_set(a.derived))
    scored = read_csv(os.path.join(a.derived, "scored.csv"))
    wb = read_csv(os.path.join(a.derived, "wbayes.csv"))
    n_ok = sum(1 for r in scored if r["parse_ok"] == "1" and r.get("calibration", "0") != "1")
    assert n_ok > 0, "no parsed records to plot"
    n1, n_clip = fig_stated_vs_target(scored, a.out)
    assert n1 == n_ok, (n1, n_ok)  # every parsed answer is plotted exactly once (in range or clipped)
    assert 0 <= n_clip < n1
    gen = a.generated or os.path.join(os.path.dirname(os.path.abspath(a.out)), "generated")
    write_fig_macros(gen, n1, n_clip)
    n2 = fig_wbayes(wb, a.out)
    assert n2 > 0
    for f in ("fig_stated_vs_target", "fig_wbayes"):
        for ext in ("pdf", "png"):
            assert os.path.getsize(os.path.join(a.out, f"{f}.{ext}")) > 0
    print(f"wrote figures to {a.out}: {n1} points in fig 1, {n2} fit markers in fig 2")


if __name__ == "__main__":
    main()
