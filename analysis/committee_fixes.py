"""Committee-review analyses.

Reads ONLY results/raw/*.jsonl (via the same allow-list and latest-attempt dedup as score.py) and
data/ (cases, clinician baseline). Never writes to results/raw/. Writes:
  results/derived/cf_*.csv                     one CSV per analysis
  paper/generated/committee_macros.tex         \\cf... macros (two decimals unless a count)
  paper/generated/table_hindsight.tex, table_decomp.tex, table_runsettings.tex
  paper/figures/fig_ratio_vs_dor.{pdf,png}     F1 figure (skipped with --no-figures)
The pre-specified analyses (score.py, fit_wbayes.py, make_tables.py) are untouched; everything here
is post hoc, prompted by the simulated committee review.

Usage: python analysis/committee_fixes.py [--raw GLOB] [--cases DIR] [--out DIR] [--generated DIR]
       [--figures DIR] [--n-boot 2000] [--no-figures] [--set paper|pilot] [--run-ids ...]
"""
import argparse
import itertools
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, add_run_args, display_name, guard_output, guard_pilot, is_calibration,  # noqa: E402
                    is_synthetic, load_cases, load_raw, logit_clipped, resolve_run_ids, select_records,
                    write_csv)
from fit_wbayes import fit_boot, ols  # noqa: E402
from make_tables import model_tag, tex_num  # noqa: E402
from score import band_error, score_record  # noqa: E402

SEED = 20261111
FIT_CONDS = ["V", "N", "B", "S"]
UTI = "A4_uti"
FLOOR_CLIP = (0.005, 0.995)  # F7: answers clipped to [0.5%, 99.5%]
SUB1 = 0.01  # F7: targets below 1%
BASELINE = os.path.join(ROOT, "data", "baselines", "morgan2021_clinicians.json")


# ---------------------------------------------------------------- loading
def load_rows(raw_glob, cases_dir, allowed, cfg):
    """Return (cases, final raw records, analysis rows, synthetic flag).

    Analysis rows: parse-ok, non-calibration scored rows with numeric fields added
    (pre, post as probabilities; prior, lr, target from the case file)."""
    cases = load_cases(cases_dir)
    recs, _ = load_raw(raw_glob)
    syn = is_synthetic(recs, raw_glob)
    final, _ = select_records(recs, allowed, cfg, quiet=True)
    assert final, f"no records selected for {allowed}"
    rows = []
    for r in final:
        c = cases[r["case_id"]]
        s = score_record(r, c)
        if not s["parse_ok"] or is_calibration(c):
            continue
        t = c["target"]
        s.update(pre=s["pretest_stated"] / 100.0, post=s["posttest_stated"] / 100.0,
                 prior=t["pretest_point"], lr=t["lr_point"], target=t["posttest_point"])
        rows.append(s)
    return cases, final, rows, syn


def models_of(rows):
    return sorted({r["model"] for r in rows})


# ---------------------------------------------------------------- small exact tests
def binom_cdf(k, n):
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n


def sign_test_p(n_pos, n):
    """Exact two-sided sign test (ties already removed), p = min(1, 2 * smaller tail)."""
    if n == 0:
        return None
    lo = binom_cdf(n_pos, n)
    hi = 1.0 - binom_cdf(n_pos - 1, n) if n_pos > 0 else 1.0
    return min(1.0, 2.0 * min(lo, hi))


def median_ci(vals, level=0.95):
    """Exact distribution-free CI for the median from order statistics.

    Returns (lo, hi, coverage) with the narrowest symmetric order-statistic interval whose exact
    coverage is at least `level`; (None, None, None) if n is too small."""
    v = sorted(vals)
    n = len(v)
    best = None
    for k in range(1, n // 2 + 1):  # interval (v[k-1], v[n-k])
        cov = 1.0 - 2.0 * binom_cdf(k - 1, n)
        if cov >= level:
            best = (v[k - 1], v[n - k], cov)
    return best if best else (None, None, None)


def signflip_p(d, delta=0.0):
    """Exact two-sided sign-flip permutation p for mean(d - delta) = 0 (all 2^n flips)."""
    x = [v - delta for v in d]
    n = len(x)
    obs = abs(sum(x)) - 1e-12
    cnt = 0
    for signs in itertools.product((1, -1), repeat=n):
        if abs(sum(s * v for s, v in zip(signs, x))) >= obs:
            cnt += 1
    return cnt / 2 ** n


def signflip_ci(d, alpha=0.05, tol=1e-4):
    """Confidence interval for the mean by inverting the exact sign-flip test: all delta with p > alpha.
    The bounds are the smallest and largest mean effects the data do NOT exclude."""
    m = sum(d) / len(d)
    span = max(abs(v - m) for v in d) + 1.0

    def bound(direction):
        inside, outside = m, m + direction * span
        while signflip_p(d, outside) > alpha:
            outside += direction * span
        while abs(outside - inside) > tol:
            mid = (inside + outside) / 2
            if signflip_p(d, mid) > alpha:
                inside = mid
            else:
                outside = mid
        return inside
    if 2.0 / 2 ** len(d) > alpha:  # too few clusters for any rejection
        return None, None
    return bound(-1), bound(1)


# ---------------------------------------------------------------- F1 within-scenario ratio
def ratio_cells(rows, cases, conds, by_condition=False):
    """One row per model x scenario (x condition): r = (mean logit p+ - mean logit p-) / (log LR+ - log LR-)."""
    g = {}
    for r in rows:
        if r["condition"] not in conds:
            continue
        key = (r["model"], r["scenario"]) + ((r["condition"],) if by_condition else ())
        g.setdefault(key, {"pos": [], "neg": []})[r["result"]].append(logit_clipped(r["post"]))
    lr = {}
    for c in cases.values():
        if not is_calibration(c):
            lr.setdefault(c["scenario"], {})[c["result"]] = c["target"]
    out = []
    for key, d in sorted(g.items()):
        if not d["pos"] or not d["neg"]:
            continue
        t = lr[key[1]]
        dx = math.log(t["pos"]["lr_point"]) - math.log(t["neg"]["lr_point"])
        dy = statistics.mean(d["pos"]) - statistics.mean(d["neg"])
        row = {"model": key[0], "scenario": key[1], "condition": key[2] if by_condition else "VNBS",
               "n_pos": len(d["pos"]), "n_neg": len(d["neg"]), "dy": dy, "log_dor": dx, "ratio": dy / dx,
               "target_neg_pct": 100 * t["neg"]["posttest_point"], "target_pos_pct": 100 * t["pos"]["posttest_point"]}
        out.append(row)
    return out


def fe_slope(cells):
    """Scenario fixed-effects slope from the two scenario-cell means: sum(dy*dx)/sum(dx^2)."""
    den = sum(c["log_dor"] ** 2 for c in cells)
    return sum(c["dy"] * c["log_dor"] for c in cells) / den


def ratio_summary(cells, group_keys=("model",)):
    groups = {}
    for c in cells:
        groups.setdefault(tuple(c[k] for k in group_keys), []).append(c)
    out = []
    for key, cs in sorted(groups.items()):
        rs = [c["ratio"] for c in cs]
        lo, hi, cov = median_ci(rs)
        n_gt = sum(r > 1 for r in rs)
        n_ne = sum(r != 1 for r in rs)
        loso = [fe_slope([c for c in cs if c["scenario"] != s]) for s in sorted({c["scenario"] for c in cs})]
        le1 = [c for c in cs if c["ratio"] <= 1]
        row = dict(zip(group_keys, key))
        row.update(n_scen=len(cs), median=statistics.median(rs), med_lo=lo, med_hi=hi, med_cov=cov,
                   n_gt1=n_gt, sign_p=sign_test_p(n_gt, n_ne), b_fe=fe_slope(cs),
                   loso_min=min(loso) if len(cs) > 1 else None, loso_max=max(loso) if len(cs) > 1 else None,
                   b_fe_no_uti=fe_slope([c for c in cs if c["scenario"] != UTI]) if any(c["scenario"] != UTI for c in cs) else None,
                   median_no_uti=statistics.median([c["ratio"] for c in cs if c["scenario"] != UTI]) if any(c["scenario"] != UTI for c in cs) else None,
                   n_le1=len(le1), n_le1_sub1_neg_target=sum(c["target_neg_pct"] < 100 * SUB1 for c in le1),
                   le1_scenarios=";".join(c["scenario"] for c in le1),
                   max_ratio_scenario=max(cs, key=lambda c: c["ratio"])["scenario"])
        out.append(row)
    return out


RATIO_CELL_FIELDS = ["model", "scenario", "condition", "n_pos", "n_neg", "dy", "log_dor", "ratio", "target_pos_pct",
                     "target_neg_pct"]
RATIO_SUM_FIELDS = ["model", "condition", "n_scen", "median", "med_lo", "med_hi", "med_cov", "n_gt1", "sign_p", "b_fe",
                    "loso_min", "loso_max", "b_fe_no_uti", "median_no_uti", "n_le1", "n_le1_sub1_neg_target",
                    "le1_scenarios", "max_ratio_scenario"]


def f1(rows, cases, mac):
    cells = ratio_cells(rows, cases, FIT_CONDS)
    cells_c = ratio_cells(rows, cases, FIT_CONDS, by_condition=True)
    summ = ratio_summary(cells)
    for s in summ:
        s["condition"] = "VNBS"
    summ_c = ratio_summary(cells_c, ("model", "condition"))
    allm = ratio_summary([dict(c, model="ALL") for c in cells])[0]
    for s in summ:
        t = "cfRatio" + model_tag(s["model"])
        mac[t + "Med"] = tex_num(s["median"])
        mac[t + "MedLo"] = tex_num(s["med_lo"])
        mac[t + "MedHi"] = tex_num(s["med_hi"])
        mac[t + "NGtOne"] = str(s["n_gt1"])
        mac[t + "NScen"] = str(s["n_scen"])
        mac[t + "SignP"] = fmt_p(s["sign_p"])
        mac[t + "Fe"] = tex_num(s["b_fe"])
        mac[t + "LosoMin"] = tex_num(s["loso_min"])
        mac[t + "LosoMax"] = tex_num(s["loso_max"])
        mac[t + "MaxScen"] = s["max_ratio_scenario"].replace("_", "\\_")
    for s in summ_c:
        t = "cfRatio" + model_tag(s["model"]) + s["condition"]
        mac[t + "Med"] = tex_num(s["median"])
        mac[t + "NGtOne"] = str(s["n_gt1"])
        mac[t + "Fe"] = tex_num(s["b_fe"])
    mac["cfRatioMedCoverage"] = f"{100 * summ[0]['med_cov']:.1f}"
    mac["cfRatioAllMed"] = tex_num(allm["median"])
    mac["cfRatioAllNCells"] = str(allm["n_scen"])
    mac["cfRatioAllNGtOne"] = str(allm["n_gt1"])
    mac["cfRatioAllNLeOne"] = str(allm["n_le1"])
    mac["cfRatioAllNLeOneSubOneNeg"] = str(allm["n_le1_sub1_neg_target"])
    mac["cfRatioMedMin"] = tex_num(min(s["median"] for s in summ))
    mac["cfRatioMedMax"] = tex_num(max(s["median"] for s in summ))
    mac["cfRatioFeMin"] = tex_num(min(s["b_fe"] for s in summ))
    mac["cfRatioFeMax"] = tex_num(max(s["b_fe"] for s in summ))
    mac["cfRatioNGtOneMin"] = str(min(s["n_gt1"] for s in summ))
    mac["cfRatioNGtOneMax"] = str(max(s["n_gt1"] for s in summ))
    mac["cfRatioLosoMin"] = tex_num(min(s["loso_min"] for s in summ))
    mac["cfRatioLosoMax"] = tex_num(max(s["loso_max"] for s in summ))
    return cells, cells_c, summ + summ_c + [dict(allm, condition="VNBS")]


def fmt_p(p):
    if p is None:
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def fig_ratio(cells, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("matplotlib missing: F1 figure skipped")
        return None
    models = sorted({c["model"] for c in cells})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000", "#E69F00"]
    marks = ["o", "s", "^", "D", "v", "P"]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    scen = sorted({(c["log_dor"], c["scenario"]) for c in cells})
    offs = {m: (i - (len(models) - 1) / 2) * 0.05 for i, m in enumerate(models)}
    for i, m in enumerate(models):
        cs = [c for c in cells if c["model"] == m]
        ax.scatter([c["log_dor"] + offs[m] for c in cs], [c["ratio"] for c in cs], s=16,
                   color=colors[i % 6], marker=marks[i % 6], label=display_name(m), alpha=0.9,
                   edgecolors="none")
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5, 2, 3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    rmin, rmax = min(c["ratio"] for c in cells), max(c["ratio"] for c in cells)
    assert rmin > 0.6 and rmax < 3.4, "F1 figure: a ratio falls outside the fixed axis; widen it"
    ax.set_ylim(0.6, 3.6)
    ax.set_xlabel("published log diagnostic odds ratio, ln(LR+/LR-)", fontsize=7)
    ax.set_ylabel("within-scenario update ratio r (log scale)", fontsize=7)
    ax.tick_params(labelsize=6)
    groups = []  # merge scenario labels whose x positions nearly coincide (e.g. B2 and B3)
    for x, s in scen:
        if groups and x - groups[-1][0] < 0.12:
            groups[-1][1].append(s.split("_")[0])
        else:
            groups.append([x, [s.split("_")[0]]])
    for x, names in groups:
        ax.annotate("/".join(names), (x, 3.5), fontsize=5, ha="center", va="top", color="0.35")
    ax.legend(fontsize=5, frameon=False, loc="lower left")
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_ratio_vs_dor.{ext}"), dpi=300)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- F2 hindsight in the stated pretest
def hindsight(rows, conds=FIT_CONDS):
    """Per model x scenario (conditions pooled) and model x scenario x condition: mean stated pretest after
    the positive version minus after the negative version, in pp and in log-odds."""
    g = {}
    for r in rows:
        if r["condition"] not in conds:
            continue
        for key in ((r["model"], r["scenario"], "VNBS"), (r["model"], r["scenario"], r["condition"])):
            g.setdefault(key, {"pos": [], "neg": []})[r["result"]].append(r["pre"])
    cells = []
    for (m, s, c), d in sorted(g.items()):
        if not d["pos"] or not d["neg"]:
            continue
        cells.append({"model": m, "scenario": s, "condition": c, "n_pos": len(d["pos"]), "n_neg": len(d["neg"]),
                      "pre_pos_pct": 100 * statistics.mean(d["pos"]), "pre_neg_pct": 100 * statistics.mean(d["neg"]),
                      "gap_pp": 100 * (statistics.mean(d["pos"]) - statistics.mean(d["neg"])),
                      "gap_lod": statistics.mean(logit_clipped(x) for x in d["pos"]) -
                      statistics.mean(logit_clipped(x) for x in d["neg"])})
    summ = []
    for m in sorted({c["model"] for c in cells}) + ["ALL"]:
        pooled = [c for c in cells if c["condition"] == "VNBS" and (m == "ALL" or c["model"] == m)]
        per_c = [c for c in cells if c["condition"] != "VNBS" and (m == "ALL" or c["model"] == m)]
        if m == "ALL":  # scenario-level: one value per scenario (mean over models), then sign test over scenarios
            bys = {}
            for c in pooled:
                bys.setdefault(c["scenario"], []).append(c)
            pooled = [{"gap_pp": statistics.mean(x["gap_pp"] for x in v),
                       "gap_lod": statistics.mean(x["gap_lod"] for x in v)} for _, v in sorted(bys.items())]
        gaps = [c["gap_pp"] for c in pooled]
        npos = sum(x > 0 for x in gaps)
        nne = sum(x != 0 for x in gaps)
        mr = [r for r in rows if r["condition"] in conds and (m == "ALL" or r["model"] == m)]
        summ.append({"model": m, "n_cells": len(pooled), "n_gap_pos": npos, "n_gap_neg": sum(x < 0 for x in gaps),
                     "sign_p": sign_test_p(npos, nne), "gap_pp_med": statistics.median(gaps),
                     "gap_pp_mean": statistics.mean(gaps),
                     "gap_lod_med": statistics.median(c["gap_lod"] for c in pooled),
                     "pre_pos_mean_pct": 100 * statistics.mean(r["pre"] for r in mr if r["result"] == "pos"),
                     "pre_neg_mean_pct": 100 * statistics.mean(r["pre"] for r in mr if r["result"] == "neg"),
                     "n_cond_cells": len(per_c), "n_cond_cells_pos": sum(c["gap_pp"] > 0 for c in per_c),
                     "n_cond_cells_neg": sum(c["gap_pp"] < 0 for c in per_c)})
    return cells, summ


HIND_CELL_FIELDS = ["model", "scenario", "condition", "n_pos", "n_neg", "pre_pos_pct", "pre_neg_pct", "gap_pp", "gap_lod"]
HIND_SUM_FIELDS = ["model", "n_cells", "n_gap_pos", "n_gap_neg", "sign_p", "gap_pp_med", "gap_pp_mean", "gap_lod_med",
                   "pre_pos_mean_pct", "pre_neg_mean_pct", "n_cond_cells", "n_cond_cells_pos", "n_cond_cells_neg"]


def f2(rows, mac, gen_dir):
    cells, summ = hindsight(rows)
    for s in summ:
        t = "cfHind" + ("All" if s["model"] == "ALL" else model_tag(s["model"]))
        mac[t + "PrePos"] = f"{s['pre_pos_mean_pct']:.1f}"
        mac[t + "PreNeg"] = f"{s['pre_neg_mean_pct']:.1f}"
        mac[t + "GapMedPp"] = tex_num(s["gap_pp_med"])
        mac[t + "GapMeanPp"] = tex_num(s["gap_pp_mean"])
        mac[t + "GapMedLod"] = tex_num(s["gap_lod_med"])
        mac[t + "NPos"] = str(s["n_gap_pos"])
        mac[t + "NScen"] = str(s["n_cells"])
        mac[t + "SignP"] = fmt_p(s["sign_p"])
        mac[t + "CondCellsPos"] = str(s["n_cond_cells_pos"])
        mac[t + "CondCells"] = str(s["n_cond_cells"])
    ms = [s for s in summ if s["model"] != "ALL"]
    mac["cfHindNPosMin"] = str(min(s["n_gap_pos"] for s in ms))
    mac["cfHindGapMedPpMin"] = tex_num(min(s["gap_pp_med"] for s in ms))
    mac["cfHindGapMedPpMax"] = tex_num(max(s["gap_pp_med"] for s in ms))
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Model & Pretest after pos. & after neg. & Median gap (pp) & Scenarios gap $>0$ & Sign test $p$ \\\\",
             "\\midrule"]
    for s in ms:
        lines.append(f"{display_name(s['model'])} & {s['pre_pos_mean_pct']:.1f} & {s['pre_neg_mean_pct']:.1f} & "
                     f"{tex_num(s['gap_pp_med'])} & {s['n_gap_pos']} of {s['n_cells']} & {fmt_p(s['sign_p'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(gen_dir, "table_hindsight.tex"), lines)
    return cells, summ


# ---------------------------------------------------------------- fits (shared by F3, F6, F7)
def fit_rows(rows, post_clip=None):
    """{scenario: [(logit prior, log LR, logit stated post)]} for the published-prior fit."""
    cl = {}
    for r in rows:
        p = r["post"] if post_clip is None else min(max(r["post"], post_clip[0]), post_clip[1])
        cl.setdefault(r["scenario"], []).append((logit_clipped(r["prior"]), math.log(r["lr"]), logit_clipped(p)))
    return cl


def fit_cell(rows, n_boot, seed_key, post_clip=None):
    cl = fit_rows(rows, post_clip)
    rng = random.Random(f"{SEED}-{seed_key}")
    res = fit_boot(cl, n_boot, rng) if len(cl) >= 2 else None
    return res


FIT_FIELDS = ["analysis", "model", "condition", "n_obs", "n_clusters", "a", "a_lo", "a_hi", "b", "b_lo", "b_hi", "c",
              "n_boot_valid"]


# ---------------------------------------------------------------- F3 sensitivity without A4 (UTI)
def overestimation(rows):
    out = []
    for m in models_of(rows):
        pos = [r for r in rows if r["model"] == m and r["result"] == "pos"]
        le = [logit_clipped(r["post"]) - logit_clipped(r["target"]) for r in pos]
        out.append({"model": m, "n": len(pos), "median_signed_lod": statistics.median(le),
                    "pct_above_target": 100 * sum(x > 0 for x in le) / len(le),
                    "abs_lod_med_all": statistics.median(abs(logit_clipped(r["post"]) - logit_clipped(r["target"]))
                                                         for r in rows if r["model"] == m)})
    return out


def clinician_values(cases_dir_cases):
    with open(BASELINE) as f:
        b = json.load(f)
    vals = {}
    for s in b["scenarios"]:
        for item in s["items"].values():
            if item is None or item.get("pooled") is None:
                continue
            res = item["applies_to_result"]
            if res in ("pos", "neg"):
                vals[s["case_ids"][res]] = float(item["pooled"]["median"])
    return vals


def clinician_closer(final_recs, cases, exclude_scen=()):
    """Set A: per model x condition, cases where the model median posttest has a smaller band error than the
    clinician pooled median (same definitions as compare_clinicians.py)."""
    clin = clinician_values(cases)
    g = {}
    for r in final_recs:
        c = cases[r["case_id"]]
        if r["case_id"] not in clin or c["scenario"] in exclude_scen or is_calibration(c):
            continue
        pa = r.get("parsed_answer")
        if r.get("parse_status") != "ok" or not isinstance(pa, dict) or pa.get("posttest") is None or pa.get("pretest") is None:
            continue
        g.setdefault((r["model"], r["condition"], r["case_id"]), []).append(float(pa["posttest"]))
    agg = {}
    for (m, cond, cid), v in g.items():
        band = cases[cid]["target"]["posttest_band_primary"]
        me = abs(band_error(statistics.median(v), band))
        ce = abs(band_error(clin[cid], band))
        a = agg.setdefault((m, cond), {"model": m, "condition": cond, "n_cases": 0, "n_model_closer": 0,
                                       "n_tie": 0, "n_clin_closer": 0})
        a["n_cases"] += 1
        a["n_model_closer"] += int(me < ce)
        a["n_tie"] += int(me == ce)
        a["n_clin_closer"] += int(me > ce)
    return [agg[k] for k in sorted(agg)]


def f3(rows, final_recs, cases, cells_ratio, n_boot, mac):
    noA4 = [r for r in rows if r["scenario"] != UTI]
    over_all, over_no = overestimation(rows), overestimation(noA4)
    out_over = [dict(o, sample="all") for o in over_all] + [dict(o, sample="no_A4") for o in over_no]
    for o in over_no:
        t = "cfNoUti" + model_tag(o["model"])
        mac[t + "OverMedLod"] = tex_num(o["median_signed_lod"])
        mac[t + "OverPctAbove"] = f"{o['pct_above_target']:.0f}"
        mac[t + "AbsLodMed"] = tex_num(o["abs_lod_med_all"])
    for o in over_all:
        mac["cfAll" + model_tag(o["model"]) + "AbsLodMed"] = tex_num(o["abs_lod_med_all"])
    mac["cfNoUtiOverMedLodMin"] = tex_num(min(o["median_signed_lod"] for o in over_no))
    mac["cfNoUtiOverMedLodMax"] = tex_num(max(o["median_signed_lod"] for o in over_no))
    mac["cfNoUtiOverPctAboveMin"] = f"{min(o['pct_above_target'] for o in over_no):.0f}"
    mac["cfNoUtiOverPctAboveMax"] = f"{max(o['pct_above_target'] for o in over_no):.0f}"
    # weighted-Bayes fits (published prior) without A4: per model x condition and per model pooled over V N B S
    fits = []
    for sample, rs in (("all", rows), ("no_A4", noA4)):
        for m in models_of(rows):
            for cond in FIT_CONDS + ["VNBS"]:
                sel = [r for r in rs if r["model"] == m and (r["condition"] == cond or
                                                             (cond == "VNBS" and r["condition"] in FIT_CONDS))]
                if sample == "all" and cond != "VNBS":
                    continue  # the pre-specified per-condition fits live in wbayes.csv
                f = fit_cell(sel, n_boot, f"{sample}-{m}-{cond}")
                if f:
                    fits.append(dict(f, analysis=f"wbayes_published_{sample}", model=m, condition=cond))
    for f in fits:
        if f["analysis"] != "wbayes_published_no_A4":
            continue
        t = "cfNoUtiWb" + model_tag(f["model"]) + f["condition"]
        for p in "ab":
            mac[t + "Par" + p.upper() + "Pt"] = tex_num(f[p])
            mac[t + "Par" + p.upper() + "Lo"] = tex_num(f[p + "_lo"])
            mac[t + "Par" + p.upper() + "Hi"] = tex_num(f[p + "_hi"])
    nu = [f for f in fits if f["analysis"] == "wbayes_published_no_A4" and f["condition"] != "VNBS"]
    mac["cfNoUtiWbACells"] = str(len(nu))
    mac["cfNoUtiWbAMin"] = tex_num(min(f["a"] for f in nu))
    mac["cfNoUtiWbAMax"] = tex_num(max(f["a"] for f in nu))
    mac["cfNoUtiWbBMin"] = tex_num(min(f["b"] for f in nu))
    mac["cfNoUtiWbBMax"] = tex_num(max(f["b"] for f in nu))
    mac["cfNoUtiWbACiBelowOneN"] = str(sum(f["a_hi"] < 1 for f in nu))
    mac["cfNoUtiWbBCiAboveOneN"] = str(sum(f["b_lo"] > 1 for f in nu))
    mac["cfNoUtiWbBGtOneN"] = str(sum(f["b"] > 1 for f in nu))
    # ratio without A4 (F1 estimand)
    rno = ratio_summary([c for c in cells_ratio if c["scenario"] != UTI])
    for s in rno:
        t = "cfNoUtiRatio" + model_tag(s["model"])
        mac[t + "Med"] = tex_num(s["median"])
        mac[t + "Fe"] = tex_num(s["b_fe"])
        mac[t + "NGtOne"] = str(s["n_gt1"])
        mac[t + "NScen"] = str(s["n_scen"])
        mac[t + "SignP"] = fmt_p(s["sign_p"])
    mac["cfNoUtiRatioMedMin"] = tex_num(min(s["median"] for s in rno))
    mac["cfNoUtiRatioMedMax"] = tex_num(max(s["median"] for s in rno))
    # clinician comparison without A4
    cc_all = clinician_closer(final_recs, cases)
    cc_no = clinician_closer(final_recs, cases, exclude_scen=(UTI,))
    clin_rows = [dict(c, sample="all") for c in cc_all] + [dict(c, sample="no_A4") for c in cc_no]
    for c in cc_no:
        mac["cfNoUtiClinCloser" + model_tag(c["model"]) + c["condition"]] = str(c["n_model_closer"])
    if cc_no:
        mac["cfNoUtiClinNCases"] = str(cc_no[0]["n_cases"])
        assert len({c["n_cases"] for c in cc_no}) == 1, "unequal Set A case counts"
    mac["cfNoUtiClinCloserSum"] = str(sum(c["n_model_closer"] for c in cc_no))
    mac["cfNoUtiClinTieSum"] = str(sum(c["n_tie"] for c in cc_no))
    mac["cfNoUtiClinClinCloserSum"] = str(sum(c["n_clin_closer"] for c in cc_no))
    mac["cfNoUtiClinCasesSum"] = str(sum(c["n_cases"] for c in cc_no))
    mac["cfAllClinCloserSum"] = str(sum(c["n_model_closer"] for c in cc_all))
    mac["cfAllClinCasesSum"] = str(sum(c["n_cases"] for c in cc_all))
    return out_over, fits, rno, clin_rows


OVER_FIELDS = ["sample", "model", "n", "median_signed_lod", "pct_above_target", "abs_lod_med_all"]
CLIN_FIELDS = ["sample", "model", "condition", "n_cases", "n_model_closer", "n_tie", "n_clin_closer"]


def write_tex(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/committee_fixes.py. Do not edit.\n" + "\n".join(lines) + "\n")


def write_macros(path, mac):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for k, v in mac.items():
        assert k.isalpha(), f"macro name not letters only: {k}"
        assert v not in ("", None), f"empty macro {k}"
        assert "\u2014" not in str(v)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/committee_fixes.py from results/raw and data/. Do not edit.\n"
                "% Post hoc analyses prompted by a simulated review.\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


# ---------------------------------------------------------------- F9 prior error + update error
def decomposition(rows, conds=FIT_CONDS):
    """Signed log-odds error = prior error (logit stated pretest - logit published prior)
    + update error (the remainder = log implied LR - log published LR). Exact identity per answer."""
    out, per = [], []
    for m in models_of(rows) + ["ALL"]:
        for res in ("pos", "neg"):
            rs = [r for r in rows if r["condition"] in conds and r["result"] == res and (m == "ALL" or r["model"] == m)]
            tot = [logit_clipped(r["post"]) - logit_clipped(r["target"]) for r in rs]
            pri = [logit_clipped(r["pre"]) - logit_clipped(r["prior"]) for r in rs]
            upd = [t - p for t, p in zip(tot, pri)]
            for r, p, u in zip(rs, pri, upd):  # identity check against the implied LR
                implied = logit_clipped(r["post"]) - logit_clipped(r["pre"])
                exact = logit_clipped(r["target"]) - logit_clipped(r["prior"])
                assert abs(u - (implied - exact)) < 1e-9
            assert abs(statistics.mean(tot) - statistics.mean(pri) - statistics.mean(upd)) < 1e-9
            out.append({"model": m, "result": res, "n": len(rs), "total_mean": statistics.mean(tot),
                        "prior_mean": statistics.mean(pri), "update_mean": statistics.mean(upd),
                        "total_med": statistics.median(tot), "prior_med": statistics.median(pri),
                        "update_med": statistics.median(upd),
                        "prior_share_of_mean": statistics.mean(pri) / statistics.mean(tot) if statistics.mean(tot) else None})
    return out


DECOMP_FIELDS = ["model", "result", "n", "total_mean", "prior_mean", "update_mean", "total_med", "prior_med",
                 "update_med", "prior_share_of_mean"]


def f9(rows, mac, gen_dir):
    dec = decomposition(rows)
    for d in dec:
        t = "cfDec" + ("All" if d["model"] == "ALL" else model_tag(d["model"])) + d["result"].capitalize()
        for k, tag in (("total", "Tot"), ("prior", "Prior"), ("update", "Upd")):
            mac[t + tag + "Mean"] = tex_num(d[k + "_mean"])
            mac[t + tag + "Med"] = tex_num(d[k + "_med"])
        mac[t + "N"] = str(d["n"])
    ms = [d for d in dec if d["model"] != "ALL" and d["result"] == "pos"]
    mac["cfDecPosPriorMedMin"] = tex_num(min(d["prior_med"] for d in ms))
    mac["cfDecPosPriorMedMax"] = tex_num(max(d["prior_med"] for d in ms))
    mac["cfDecPosUpdMedMin"] = tex_num(min(d["update_med"] for d in ms))
    mac["cfDecPosUpdMedMax"] = tex_num(max(d["update_med"] for d in ms))
    mac["cfDecPosPriorMeanMin"] = tex_num(min(d["prior_mean"] for d in ms))
    mac["cfDecPosPriorMeanMax"] = tex_num(max(d["prior_mean"] for d in ms))
    mac["cfDecPosUpdMeanMin"] = tex_num(min(d["update_mean"] for d in ms))
    mac["cfDecPosUpdMeanMax"] = tex_num(max(d["update_mean"] for d in ms))
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{Positive test} & \\multicolumn{3}{c}{Negative test} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Model & Total & Prior & Update & Total & Prior & Update \\\\", "\\midrule"]
    for m in models_of(rows):
        p = [d for d in dec if d["model"] == m and d["result"] == "pos"][0]
        n = [d for d in dec if d["model"] == m and d["result"] == "neg"][0]
        lines.append(f"{display_name(m)} & " + " & ".join(tex_num(x[k + '_mean']) for x in (p, n)
                                                           for k in ("total", "prior", "update")) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(gen_dir, "table_decomp.tex"), lines)
    return dec


# ---------------------------------------------------------------- F6 paired log-odds contrasts, sets pooled
CONTRASTS = [("NvsV", "N", "V", "abs", "H1"), ("BvsN", "B", "N", "abs", "H2"), ("SvsN", "S", "N", "abs", "H3"),
             ("EvsN", "E", "N", "signed", "H4")]


def scenario_contrasts(rows, X, Y, kind):
    """{model: {scenario: d}}: per case, seed-paired X minus Y difference in (absolute or signed) log-odds
    error averaged over seeds; per scenario, the mean of its positive and negative case values."""
    val = {}
    for r in rows:
        e = logit_clipped(r["post"]) - logit_clipped(r["target"])
        val[(r["model"], r["case_id"], r["condition"], r["seed"])] = (abs(e) if kind == "abs" else e, r["scenario"])
    case_d = {}
    for (m, cid, cond, seed), (v, scen) in val.items():
        if cond != X or (m, cid, Y, seed) not in val:
            continue
        case_d.setdefault((m, scen, cid), []).append(v - val[(m, cid, Y, seed)][0])
    scen_d = {}
    for (m, scen, cid), ds in case_d.items():
        scen_d.setdefault(m, {}).setdefault(scen, []).append(statistics.mean(ds))
    return {m: {s: statistics.mean(v) for s, v in d.items()} for m, d in scen_d.items()}


def num2(x):
    """Two decimals always (L10); a value that rounds to zero prints 0.00, never -0.00."""
    if float(f"{float(x):.2f}") == 0.0:
        return "0.00"
    return tex_num(x, 2)


def f6(rows, n_boot, mac):
    out = []
    for name, X, Y, kind, hyp in CONTRASTS:
        sc = scenario_contrasts(rows, X, Y, kind)
        models = sorted(sc)
        allm = {}
        for m in models:
            for s, v in sc[m].items():
                allm.setdefault(s, []).append(v)
        groups = [(m, sc[m]) for m in models] + [("ALL", {s: statistics.mean(v) for s, v in allm.items()})]
        for m, d in groups:
            vals = [d[s] for s in sorted(d)]
            lo, hi = signflip_ci(vals)
            row = {"contrast": name, "hypothesis": hyp, "metric": "abs_logodds_err" if kind == "abs" else "signed_logodds_err",
                   "model": m, "n_scen": len(vals), "mean": statistics.mean(vals), "lo": lo, "hi": hi,
                   "p_signflip": signflip_p(vals), "n_scen_pos": sum(v > 0 for v in vals)}
            out.append(row)
            t = "cfCtr" + name + ("All" if m == "ALL" else model_tag(m))
            mac[t + "Mean"] = num2(row["mean"])
            mac[t + "Lo"] = num2(lo)
            mac[t + "Hi"] = num2(hi)
            mac[t + "P"] = fmt_p(row["p_signflip"])
            mac[t + "N"] = str(len(vals))
            mac[t + "NPos"] = str(row["n_scen_pos"])
        ms = [r for r in out if r["contrast"] == name and r["model"] != "ALL"]
        mac["cfCtr" + name + "MaxHi"] = num2(max(r["hi"] for r in ms))
        mac["cfCtr" + name + "MinLo"] = num2(min(r["lo"] for r in ms))
        mac["cfCtr" + name + "NExclZero"] = str(sum(r["lo"] > 0 or r["hi"] < 0 for r in ms))
        mac["cfCtr" + name + "MinP"] = fmt_p(min(r["p_signflip"] for r in ms))
    h13 = [r for r in out if r["hypothesis"] in ("H1", "H2", "H3") and r["model"] != "ALL"]
    mac["cfCtrHOneToThreeMaxHi"] = num2(max(r["hi"] for r in h13))
    mac["cfCtrHOneToThreeMinLo"] = num2(min(r["lo"] for r in h13))
    mac["cfCtrHOneToThreeNExclZero"] = str(sum(r["lo"] > 0 or r["hi"] < 0 for r in h13))
    mac["cfCtrHOneToThreeNCells"] = str(len(h13))
    # b (and a) per model pooled over V N B S, published prior, all scenarios
    fits = []
    for m in models_of(rows):
        f = fit_cell([r for r in rows if r["model"] == m and r["condition"] in FIT_CONDS], n_boot, f"pooled-{m}")
        fits.append(dict(f, analysis="wbayes_published_pooled_VNBS", model=m, condition="VNBS"))
        t = "cfPool" + model_tag(m)
        for p in "ab":
            mac[t + "Par" + p.upper() + "Pt"] = tex_num(f[p])
            mac[t + "Par" + p.upper() + "Lo"] = tex_num(f[p + "_lo"])
            mac[t + "Par" + p.upper() + "Hi"] = tex_num(f[p + "_hi"])
    mac["cfPoolParBMin"] = tex_num(min(f["b"] for f in fits))
    mac["cfPoolParBMax"] = tex_num(max(f["b"] for f in fits))
    mac["cfPoolParBCiAboveOneN"] = str(sum(f["b_lo"] > 1 for f in fits))
    mac["cfPoolParACiBelowOneN"] = str(sum(f["a_hi"] < 1 for f in fits))
    mac["cfPoolNModels"] = str(len(fits))
    return out, fits


CTR_FIELDS = ["contrast", "hypothesis", "metric", "model", "n_scen", "mean", "lo", "hi", "p_signflip", "n_scen_pos"]


# ---------------------------------------------------------------- F7 response floor
def fclip(p):
    return min(max(p, FLOOR_CLIP[0]), FLOOR_CLIP[1])


def f7(rows, n_boot, mac):
    fits = []
    for m in models_of(rows):
        sel = [r for r in rows if r["model"] == m and r["condition"] in FIT_CONDS]
        f = fit_cell(sel, n_boot, f"clip-{m}", post_clip=FLOOR_CLIP)
        fits.append(dict(f, analysis="wbayes_published_pooled_VNBS_clip05", model=m, condition="VNBS"))
        t = "cfClip" + model_tag(m)
        for p in "ab":
            mac[t + "Par" + p.upper() + "Pt"] = tex_num(f[p])
            mac[t + "Par" + p.upper() + "Lo"] = tex_num(f[p + "_lo"])
            mac[t + "Par" + p.upper() + "Hi"] = tex_num(f[p + "_hi"])
        for cond in FIT_CONDS:
            g = ols([x for xs in fit_rows([r for r in sel if r["condition"] == cond], FLOOR_CLIP).values() for x in xs])
            fits.append({"analysis": "wbayes_published_clip05_point", "model": m, "condition": cond,
                         "n_obs": sum(r["condition"] == cond for r in sel), "a": g[0], "b": g[1], "c": g[2]})
    share = []
    for m in models_of(rows) + ["ALL"]:
        rs = [r for r in rows if m == "ALL" or r["model"] == m]
        err = [abs(logit_clipped(r["post"]) - logit_clipped(r["target"])) for r in rs]
        # floor-aware error: answer AND target clipped to [0.5%, 99.5%], so a "1%" answer to a
        # 0.03% target counts as 0.69 log-odds, not 3.6
        errc = [abs(logit_clipped(fclip(r["post"])) - logit_clipped(fclip(r["target"]))) for r in rs]
        sub = [r["target"] < SUB1 for r in rs]
        tot = sum(err)
        row = {"model": m, "n": len(rs), "n_sub1_target": sum(sub), "pct_answers_sub1_target": 100 * sum(sub) / len(rs),
               "pct_abs_err_from_sub1": 100 * sum(e for e, s in zip(err, sub) if s) / tot,
               "pct_abs_err_from_sub1_clipped": 100 * sum(e for e, s in zip(errc, sub) if s) / sum(errc),
               "pct_answers_at_or_below_1pct": 100 * sum(r["post"] <= 0.01 for r in rs) / len(rs),
               "abs_err_med": statistics.median(err), "abs_err_med_clipped": statistics.median(errc)}
        share.append(row)
        t = "cfFloor" + ("All" if m == "ALL" else model_tag(m))
        mac[t + "PctAnswersSubOne"] = f"{row['pct_answers_sub1_target']:.0f}"
        mac[t + "PctErrSubOne"] = f"{row['pct_abs_err_from_sub1']:.0f}"
        mac[t + "PctErrSubOneClip"] = f"{row['pct_abs_err_from_sub1_clipped']:.0f}"
        mac[t + "PctAtOrBelowOne"] = f"{row['pct_answers_at_or_below_1pct']:.0f}"
        mac[t + "AbsErrMedClip"] = tex_num(row["abs_err_med_clipped"])
    ms = [s for s in share if s["model"] != "ALL"]
    mac["cfFloorPctErrSubOneMin"] = f"{min(s['pct_abs_err_from_sub1'] for s in ms):.0f}"
    mac["cfFloorPctErrSubOneMax"] = f"{max(s['pct_abs_err_from_sub1'] for s in ms):.0f}"
    cl = [f for f in fits if f["analysis"] == "wbayes_published_pooled_VNBS_clip05"]
    mac["cfClipParBMin"] = tex_num(min(f["b"] for f in cl))
    mac["cfClipParBMax"] = tex_num(max(f["b"] for f in cl))
    mac["cfClipParAMin"] = tex_num(min(f["a"] for f in cl))
    mac["cfClipParAMax"] = tex_num(max(f["a"] for f in cl))
    mac["cfClipLo"] = "0.5"
    mac["cfClipHi"] = "99.5"
    return fits, share


FLOOR_FIELDS = ["model", "n", "n_sub1_target", "pct_answers_sub1_target", "pct_abs_err_from_sub1",
                "pct_abs_err_from_sub1_clipped", "pct_answers_at_or_below_1pct", "abs_err_med", "abs_err_med_clipped"]


# ---------------------------------------------------------------- F10 midpoint error, clinicians vs models
def f10(rows, cases, mac):
    clin = {k: v for k, v in clinician_values(cases).items() if not is_calibration(cases[k])}
    out = []
    mids = {}
    for cid, v in sorted(clin.items()):
        lo, hi = cases[cid]["target"]["posttest_band_primary"]
        mids[cid] = 100 * (lo + hi) / 2
        out.append({"who": "clinicians_pooled_median", "condition": "", "case_id": cid, "value_pct": v,
                    "midpoint_pct": mids[cid], "abs_mid_err_pp": abs(v - mids[cid])})
    for m in models_of(rows):
        for cond in ["V", "N", "B", "S", "E"]:
            for cid in sorted(clin):
                v = [100 * r["post"] for r in rows if r["model"] == m and r["condition"] == cond and r["case_id"] == cid]
                if v:
                    med = statistics.median(v)
                    out.append({"who": m, "condition": cond, "case_id": cid, "value_pct": med,
                                "midpoint_pct": mids[cid], "abs_mid_err_pp": abs(med - mids[cid])})
    summ = []
    for sample, excl in (("all", ()), ("no_A4", (UTI,))):
        keep = [o for o in out if cases[o["case_id"]]["scenario"] not in excl]
        cl = [o["abs_mid_err_pp"] for o in keep if o["who"] == "clinicians_pooled_median"]
        summ.append({"sample": sample, "who": "clinicians_pooled_median", "condition": "", "n_cases": len(cl),
                     "mean_abs_mid_err_pp": statistics.mean(cl), "median_abs_mid_err_pp": statistics.median(cl)})
        for m in models_of(rows):
            for cond in ["V", "N", "B", "S", "E"]:
                e = [o["abs_mid_err_pp"] for o in keep if o["who"] == m and o["condition"] == cond]
                if e:
                    summ.append({"sample": sample, "who": m, "condition": cond, "n_cases": len(e),
                                 "mean_abs_mid_err_pp": statistics.mean(e), "median_abs_mid_err_pp": statistics.median(e)})
    for s in summ:
        t = "cfMid" + ("NoUti" if s["sample"] == "no_A4" else "") + \
            ("Clin" if s["who"] == "clinicians_pooled_median" else model_tag(s["who"]) + s["condition"])
        mac[t + "Mae"] = f"{s['mean_abs_mid_err_pp']:.1f}"
        mac[t + "Med"] = f"{s['median_abs_mid_err_pp']:.1f}"
        mac[t + "N"] = str(s["n_cases"])
    v = [s for s in summ if s["sample"] == "all" and s["who"] != "clinicians_pooled_median" and s["condition"] == "V"]
    mac["cfMidModelVMaeMin"] = f"{min(s['mean_abs_mid_err_pp'] for s in v):.1f}"
    mac["cfMidModelVMaeMax"] = f"{max(s['mean_abs_mid_err_pp'] for s in v):.1f}"
    return out, summ


MID_CASE_FIELDS = ["who", "condition", "case_id", "value_pct", "midpoint_pct", "abs_mid_err_pp"]
MID_SUM_FIELDS = ["sample", "who", "condition", "n_cases", "mean_abs_mid_err_pp", "median_abs_mid_err_pp"]


# ---------------------------------------------------------------- F11 run settings from raw records
REASONING_KEYS = ("reasoning", "reasoning_effort", "thinking", "include_reasoning", "effort")
PROVIDER_KEYS = ("provider", "provider_name", "upstream_provider")


def run_settings(final_recs, all_recs, cases):
    out, trunc = [], []
    for m in sorted({r["model"] for r in final_recs}):
        rs = [r for r in final_recs if r["model"] == m]
        dps = sorted({json.dumps(r.get("decoding_params"), sort_keys=True) for r in rs})
        mx = {(r.get("decoding_params") or {}).get("max_tokens") for r in rs}
        temp = {(r.get("decoding_params") or {}).get("temperature") for r in rs}
        reasoning = sorted({k for r in rs for k in list(r) + list(r.get("decoding_params") or {}) if k in REASONING_KEYS})
        provider = sorted({str(r[k]) for r in rs for k in PROVIDER_KEYS if r.get(k)})
        hit = [r for r in rs if r.get("completion_tokens") is not None and (r.get("decoding_params") or {}).get("max_tokens")
               and float(r["completion_tokens"]) >= float(r["decoding_params"]["max_tokens"])]
        hit_all = [r for r in all_recs if r["model"] == m and r.get("completion_tokens") is not None
                   and (r.get("decoding_params") or {}).get("max_tokens")
                   and float(r["completion_tokens"]) >= float(r["decoding_params"]["max_tokens"])]
        out.append({"model": m, "n_final": len(rs), "decoding_params": " | ".join(dps),
                    "max_tokens": ";".join(str(x) for x in sorted(mx, key=str)),
                    "temperature": ";".join(str(x) for x in sorted(temp, key=str)),
                    "reasoning_settings": ";".join(reasoning) if reasoning else "none recorded",
                    "provider": ";".join(provider) if provider else "not recorded",
                    "n_length_stopped_final": len(hit),
                    "n_length_stopped_unparsed_final": sum(r.get("parse_status") != "ok" for r in hit),
                    "n_length_stopped_all_attempts": len(hit_all),
                    "max_completion_tokens": max(float(r.get("completion_tokens") or 0) for r in rs)})
        sc = {}
        for r in hit:
            k = (cases[r["case_id"]]["scenario"], r["condition"])
            sc[k] = sc.get(k, 0) + 1
        for (s, c), n in sorted(sc.items()):
            trunc.append({"model": m, "scenario": s, "condition": c, "n_length_stopped": n})
    return out, trunc


RUN_FIELDS = ["model", "n_final", "decoding_params", "max_tokens", "temperature", "reasoning_settings", "provider",
              "n_length_stopped_final", "n_length_stopped_unparsed_final", "n_length_stopped_all_attempts",
              "max_completion_tokens"]


def f11(final, raw_glob, allowed, cases, mac, gen_dir):
    recs, _ = load_raw(raw_glob)
    sel = [r for r in recs if r.get("run_id") in set(allowed)]
    rs, tr = run_settings(final, sel, cases)
    for r in rs:
        t = "cfRun" + model_tag(r["model"])
        mac[t + "MaxTokens"] = r["max_tokens"]
        mac[t + "Temperature"] = r["temperature"]
        mac[t + "NTrunc"] = str(r["n_length_stopped_final"])
        mac[t + "NTruncUnparsed"] = str(r["n_length_stopped_unparsed_final"])
        mac[t + "NTruncAllAttempts"] = str(r["n_length_stopped_all_attempts"])
        mac[t + "Reasoning"] = r["reasoning_settings"]
        mac[t + "Provider"] = r["provider"]
    mac["cfRunNTruncTotal"] = str(sum(r["n_length_stopped_final"] for r in rs))
    mac["cfRunNTruncAllAttemptsTotal"] = str(sum(r["n_length_stopped_all_attempts"] for r in rs))
    mac["cfRunProviderRecorded"] = "no" if all(r["provider"] == "not recorded" for r in rs) else "yes"
    mac["cfRunReasoningRecorded"] = "no" if all(r["reasoning_settings"] == "none recorded" for r in rs) else "yes"
    lines = ["\\begin{tabular}{lrrrrl}", "\\toprule",
             "Model & Temp. & Max tokens & Replies (final attempts) & Stopped at limit & Provider \\\\", "\\midrule"]
    for r in rs:
        lines.append(f"{display_name(r['model'])} & {r['temperature']} & {r['max_tokens']} & {r['n_final']} & "
                     f"{r['n_length_stopped_final']} & {r['provider']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(gen_dir, "table_runsettings.tex"), lines)
    return rs, tr


TRUNC_FIELDS = ["model", "scenario", "condition", "n_length_stopped"]


def extra(rows, final, cases, a, mac):
    res = {}
    dec = f9(rows, mac, a.generated)
    write_csv(os.path.join(a.out, "cf_decomposition.csv"), dec, DECOMP_FIELDS)
    ctr, pfits = f6(rows, a.n_boot, mac)
    write_csv(os.path.join(a.out, "cf_contrasts_signflip.csv"), ctr, CTR_FIELDS)
    cfits, share = f7(rows, a.n_boot, mac)
    write_csv(os.path.join(a.out, "cf_floor.csv"), share, FLOOR_FIELDS)
    mcases, msum = f10(rows, cases, mac)
    write_csv(os.path.join(a.out, "cf_midpoint_cases.csv"), mcases, MID_CASE_FIELDS)
    write_csv(os.path.join(a.out, "cf_midpoint_summary.csv"), msum, MID_SUM_FIELDS)
    rs, tr = f11(final, a.raw, a.allowed, cases, mac, a.generated)
    write_csv(os.path.join(a.out, "cf_run_settings.csv"), rs, RUN_FIELDS)
    write_csv(os.path.join(a.out, "cf_truncated_by_scenario.csv"), tr, TRUNC_FIELDS)
    res.update(dec=dec, ctr=ctr, floor=share, mid=msum, run=rs, trunc=tr, extra_fits=pfits + cfits)
    return res


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "derived"))
    ap.add_argument("--generated", default=os.path.join(ROOT, "paper", "generated"))
    ap.add_argument("--figures", default=os.path.join(ROOT, "paper", "figures"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-figures", action="store_true")
    add_run_args(ap)
    a = ap.parse_args(argv)
    allowed, run_set, cfg = resolve_run_ids(a)
    cases, final, rows, syn = load_rows(a.raw, a.cases, allowed, cfg)
    for d in (a.out, a.generated, a.figures):
        guard_output(d, syn)
    guard_pilot(a.generated, run_set)
    guard_pilot(a.figures, run_set)
    mac = {}
    res = {}
    cells, cells_c, rsum = f1(rows, cases, mac)
    write_csv(os.path.join(a.out, "cf_ratio_cells.csv"), cells + cells_c, RATIO_CELL_FIELDS)
    write_csv(os.path.join(a.out, "cf_ratio_summary.csv"), rsum, RATIO_SUM_FIELDS)
    if not a.no_figures:
        fig_ratio(cells, a.figures)
    res["ratio"] = rsum
    hc, hs = f2(rows, mac, a.generated)
    write_csv(os.path.join(a.out, "cf_hindsight_cells.csv"), hc, HIND_CELL_FIELDS)
    write_csv(os.path.join(a.out, "cf_hindsight_summary.csv"), hs, HIND_SUM_FIELDS)
    res["hindsight"] = hs
    over, fits, rno, clin = f3(rows, final, cases, cells, a.n_boot, mac)
    write_csv(os.path.join(a.out, "cf_no_uti_overestimation.csv"), over, OVER_FIELDS)
    write_csv(os.path.join(a.out, "cf_no_uti_clinician.csv"), clin, CLIN_FIELDS)
    res.update(over=over, fits=fits, clin=clin)
    a.allowed = allowed
    res.update(extra(rows, final, cases, a, mac))
    write_csv(os.path.join(a.out, "cf_fits.csv"), res["fits"] + res["extra_fits"], FIT_FIELDS)
    write_macros(os.path.join(a.generated, "committee_macros.tex"), mac)
    print(f"committee_fixes: {len(rows)} analysed rows, {len(mac)} macros")
    return res, mac


if __name__ == "__main__":
    main()
