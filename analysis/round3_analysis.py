"""Round-3 simulated-review analyses.
All post hoc except the pre-specified per-set contrasts of item 5. Reads ONLY results/raw/*.jsonl (same allow-list and
latest-attempt dedup as score.py) and data/. Never writes to results/raw/. Original 10 scenarios only (B7, B8 are never
included; same exclusions as analysis/ext_analysis.py E1). Writes:
  results/derived/r3_blind_prior_cells.csv     item 1: blind pretest vs published prior, model x scenario (V N S pooled)
  results/derived/r3_blind_prior_summary.csv   item 1: per model and pooled, all 10 scenarios and without A4
  results/derived/r3_blind_range_cells.csv     item 1: blind pretest against the published range or CI, per cell
  results/derived/r3_blind_range_summary.csv   item 1: counts above / inside / below the range, cell and answer level
  results/derived/r3_decomp.csv                item 2: prior part + update part, with and without A4, pos / neg / both
  results/derived/r3_decomp_a4share.csv        item 2: A4's share of the mean prior part
  results/derived/r3_belief_fits.csv           item 3: b against the published and the believed LR, E2 vs main1 N
  results/derived/r3_belief_ratio.csv          item 3: r against the published and the believed LR, E2 vs main1 N
  results/derived/r3_belief_share.csv          item 3: how much of the drop in r is the believed LR and how much is different answers
  results/derived/r3_noa4_headlines.csv        item 4: hindsight, pooled r, H1 to H4 contrasts without A4
  results/derived/r3_prespec_contrasts.csv     item 5: pre-specified per-set contrasts (same code as make_tables)
  paper/generated/round3_macros.tex            \\rthree... macros (letters only, two decimals)
  paper/generated/table_prespec.tex            item 5: one-column table of the pre-specified H1 to H3 contrasts
Interpretation:.

Usage: python analysis/round3_analysis.py [--raw GLOB] [--cases DIR] [--out DIR] [--generated DIR] [--n-boot 2000]
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import committee_fixes as cf  # noqa: E402
import ext_analysis as ext  # noqa: E402
import make_tables as mt  # noqa: E402
from common import (ROOT, add_run_args, display_name, guard_output, guard_pilot, load_cases, load_raw,  # noqa: E402
                    logit_clipped, resolve_run_ids, write_csv)
from fit_wbayes import fit_boot  # noqa: E402
from make_tables import model_tag  # noqa: E402
from round2_analysis import ORIGINAL_SCENARIOS, summarize_ratio  # noqa: E402

UTI = cf.UTI
CONDS = cf.FIT_CONDS
PREFIX = "rthree"


def lod(p):
    return logit_clipped(p)


def tag(m):
    return "All" if m == "ALL" else model_tag(m)


def hw(h):
    """H1 -> HOne (macro names are letters only)."""
    return "".join(mt.DIGITS[int(ch)] if ch.isdigit() else ch for ch in h)


def n2(x):
    return "n/a" if x is None else cf.num2(x)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


# Intervals printed in the case files for Set B point priors (source text, never invented). Each entry is
# (low, high, kind, string that must appear in the case file). Set A uses target.pretest_range (Morgan 2021 Table 3).
# B4 prints no interval for its stratum prevalence: point only. B3's printed CI belongs to the 40.6% figure that the
# paper itself prints inconsistently for 24 of 64; the target uses 24/64 = 37.5%, which lies inside it.
SET_B_INTERVALS = {
    "B1_pe_low": (0.005, 0.027, "source_ci", "CI 0.5-2.7"),
    "B2_pe_mod": (0.125, 0.206, "source_ci", "CI 12.5-20.6"),
    "B3_pe_high": (0.287, 0.537, "source_ci", "CI 28.7-53.7"),
    "B5_appy_ct": (0.220, 0.251, "source_ci", "95% CI 22.0-25.1"),
    "B6_appy_us": (0.220, 0.251, "source_ci", "95% CI 22.0-25.1"),
}


def prior_ranges(cases):
    """{scenario: (lo, hi, kind)} with kind morgan_range, source_ci or point_only (lo = hi = None)."""
    out = {}
    for c in cases.values():
        if c["result"] != "pos" or c["scenario"] not in ORIGINAL_SCENARIOS or c["scenario"] == "A5_hypothetical":
            continue
        s = c["scenario"]
        if c["set"] == "A":
            lo, hi = c["target"]["pretest_range"]
            assert lo < hi
            assert "range" in c["target_inputs"]["pretest"]
            out[s] = (lo, hi, "morgan_range")
        elif s in SET_B_INTERVALS:
            lo, hi, kind, needle = SET_B_INTERVALS[s]
            assert needle in json.dumps(c), f"{s}: interval text {needle!r} not in the case file"
            assert lo <= c["target"]["pretest_point"] <= hi, s
            out[s] = (lo, hi, kind)
        else:
            out[s] = (None, None, "point_only")
    return out


# ---------------------------------------------------------------- item 1
def blind_prior(cells, models):
    """Per model and pooled (mean over models within scenario): blind pretest minus published prior."""
    pooled = [c for c in cells if c["condition"] == "VNS"]
    cell_rows, summ = [], []
    for c in pooled:
        cell_rows.append({"model": c["model"], "scenario": c["scenario"], "blind_pct": c["blind_pct"],
                          "published_prior_pct": c["published_prior_pct"],
                          "gap_pp": c["blind_pct"] - c["published_prior_pct"], "gap_lod": c["blind_prior_err_lod"]})
    for sample in ("all", "no_A4"):
        for m in models + ["ALL"]:
            cs = [c for c in cell_rows if (m == "ALL" or c["model"] == m) and (sample == "all" or c["scenario"] != UTI)]
            if m == "ALL":
                by = {}
                for c in cs:
                    by.setdefault(c["scenario"], []).append(c)
                cs = [{"scenario": s, "gap_pp": mean(x["gap_pp"] for x in v), "gap_lod": mean(x["gap_lod"] for x in v)}
                      for s, v in sorted(by.items())]
            lod_v = [c["gap_lod"] for c in cs]
            pp_v = [c["gap_pp"] for c in cs]
            lo, hi = cf.signflip_ci(lod_v)
            nab = sum(v > 0 for v in lod_v)
            summ.append({"sample": sample, "model": m, "n_scen": len(cs), "n_above": nab,
                         "sign_test_p": cf.sign_test_p(nab, sum(v != 0 for v in lod_v)),
                         "med_pp": statistics.median(pp_v), "mean_pp": mean(pp_v),
                         "med_lod": statistics.median(lod_v), "mean_lod": mean(lod_v), "lod_lo": lo, "lod_hi": hi,
                         "signflip_p": cf.signflip_p(lod_v)})
    return cell_rows, summ


def blind_range(cell_rows, bl, ranges, models):
    """Cell level: blind mean pretest vs the published range; answer level: share of individual blind answers."""
    cells = []
    for c in cell_rows:
        lo, hi, kind = ranges[c["scenario"]]
        cells.append(dict(c, kind=kind, range_lo_pct=None if lo is None else 100 * lo,
                          range_hi_pct=None if hi is None else 100 * hi, position=_pos(c["blind_pct"] / 100, lo, hi)))
    out = []
    for sample in ("all", "no_A4"):
        for m in models + ["ALL"]:
            keep = lambda s: sample == "all" or s != UTI  # noqa: E731
            cs = [c for c in cells if (m == "ALL" or c["model"] == m) and keep(c["scenario"])]
            if m == "ALL":     # scenario-level: mean blind pretest over models, then position
                by = {}
                for c in cs:
                    by.setdefault(c["scenario"], []).append(c["blind_pct"])
                pos = [_pos(mean(v) / 100, *ranges[s][:2]) for s, v in sorted(by.items())]
            else:
                pos = [c["position"] for c in cs]
            ans = [(p, ranges[k[1]][:2]) for k, ps in bl.items() if (m == "ALL" or k[0] == m) and keep(k[1]) for p in ps]
            ans_pos = [_pos(p, *rg) for p, rg in ans if rg[0] is not None]
            n_r = len(ans_pos)
            out.append({"sample": sample, "model": m, "n_scen": len(pos),
                        "n_point_only": sum(p == "point_only" for p in pos),
                        "n_range": sum(p != "point_only" for p in pos),
                        "n_above": sum(p == "above" for p in pos), "n_inside": sum(p == "inside" for p in pos),
                        "n_below": sum(p == "below" for p in pos), "n_answers_with_range": n_r,
                        "pct_ans_above": 100 * sum(p == "above" for p in ans_pos) / n_r,
                        "pct_ans_inside": 100 * sum(p == "inside" for p in ans_pos) / n_r,
                        "pct_ans_below": 100 * sum(p == "below" for p in ans_pos) / n_r})
    return cells, out


def _pos(p, lo, hi):
    if lo is None:
        return "point_only"
    return "above" if p > hi else ("below" if p < lo else "inside")


# ---------------------------------------------------------------- item 2
def decomp_parts(rs):
    tot = [lod(r["post"]) - lod(r["target"]) for r in rs]
    pri = [lod(r["pre"]) - lod(r["prior"]) for r in rs]
    upd = [t - p for t, p in zip(tot, pri)]
    return tot, pri, upd


def decomp(rows, models):
    out, share = [], []
    for sample in ("all", "no_A4"):
        for m in models + ["ALL"]:
            for res in ("pos", "neg", "both"):
                rs = [r for r in rows if r["condition"] in CONDS and (m == "ALL" or r["model"] == m)
                      and (res == "both" or r["result"] == res) and (sample == "all" or r["scenario"] != UTI)]
                tot, pri, upd = decomp_parts(rs)
                assert abs(mean(tot) - mean(pri) - mean(upd)) < 1e-9
                out.append({"sample": sample, "model": m, "result": res, "n": len(rs), "total_mean": mean(tot),
                            "prior_mean": mean(pri), "update_mean": mean(upd), "prior_med": statistics.median(pri),
                            "update_med": statistics.median(upd)})
    for m in models + ["ALL"]:
        for res in ("pos", "neg", "both"):
            rs = [r for r in rows if r["condition"] in CONDS and (m == "ALL" or r["model"] == m)
                  and (res == "both" or r["result"] == res)]
            _, pri, _ = decomp_parts(rs)
            a4 = [lod(r["pre"]) - lod(r["prior"]) for r in rs if r["scenario"] == UTI]
            share.append({"model": m, "result": res, "n": len(rs), "n_a4": len(a4), "prior_mean_all": mean(pri),
                          "a4_contribution": sum(a4) / len(rs), "a4_share_of_prior_mean": sum(a4) / sum(pri),
                          "a4_share_of_answers": len(a4) / len(rs)})
    return out, share


# ---------------------------------------------------------------- item 3
def fits_for(rows, models, variants, n_boot, tagname):
    """variants: {name: (x_prior(r), x_lr(r))}. Clusters = scenarios; pooled over models for ALL."""
    out = []
    for m in models + ["ALL"]:
        rs = [r for r in rows if m == "ALL" or r["model"] == m]
        for name, (xp, xl) in variants.items():
            cl = {}
            for r in rs:
                cl.setdefault(r["scenario"], []).append((xp(r), xl(r), lod(r["post"])))
            f = fit_boot(cl, n_boot, random.Random(f"{cf.SEED}-r3-{tagname}-{name}-{m}"))
            out.append(dict(f, model=m, arm=tagname, variant=name))
    return out


def belief(rows, cases, e2_final, models, n_boot):
    e2r, _ = ext.e2_rows(e2_final, cases)
    mn = [r for r in rows if r["condition"] == "N"]
    own = lambda r: lod(r["pre"])        # noqa: E731
    pubp = lambda r: lod(r["prior"])     # noqa: E731
    lpub = lambda r: math.log(r["lr"])   # noqa: E731
    lbel = lambda r: math.log(r["lr_bel"])  # noqa: E731
    fits = fits_for(e2r, models, {"own_pretest_published_lr": (own, lpub), "own_pretest_believed_lr": (own, lbel),
                                  "published_prior_published_lr": (pubp, lpub)}, n_boot, "ext2_N")
    fits += fits_for(mn, models, {"own_pretest_published_lr": (own, lpub), "published_prior_published_lr": (pubp, lpub)},
                     n_boot, "main1_N")
    # r (same estimand as the main analysis: within-scenario, mean logit posttest difference over log DOR)
    sc, _, _ = ext.e2(e2_final, cases, 2, {})
    cells_e2 = {"published_lr": [], "believed_lr": []}
    for s in sc:
        if s["ratio_pub"] is not None:
            cells_e2["published_lr"].append({"model": s["model"], "scenario": s["scenario"], "ratio": s["ratio_pub"],
                                             "n_censored": 0})
        if s["ratio_own"] is not None:
            cells_e2["believed_lr"].append({"model": s["model"], "scenario": s["scenario"], "ratio": s["ratio_own"],
                                            "n_censored": 0})
    cells_main = [dict(c, n_censored=0) for c in cf.ratio_cells(rows, cases, ["N"])]
    arms = {("ext2_N", "published_lr"): cells_e2["published_lr"], ("ext2_N", "believed_lr"): cells_e2["believed_lr"],
            ("main1_N", "published_lr"): cells_main}
    rat = []
    for (arm, lrk), cells in arms.items():
        for s in summarize_ratio(cells, models):
            rat.append(dict(s, arm=arm, lr=lrk))
    # paired scenario difference in r: ext2 (published LR) minus main1 N
    def scen_vals(cells, m):
        by = {}
        for c in cells:
            if m == "ALL" or c["model"] == m:
                by.setdefault(c["scenario"], []).append(c["ratio"])
        return {s: statistics.median(v) for s, v in by.items()}
    diffs = []
    for m in models + ["ALL"]:
        a, b = scen_vals(cells_e2["published_lr"], m), scen_vals(cells_main, m)
        d = [a[s] - b[s] for s in sorted(set(a) & set(b))]
        lo, hi = cf.signflip_ci(d)
        diffs.append({"model": m, "n_scen": len(d), "mean_diff": mean(d), "median_diff": statistics.median(d),
                      "lo": lo, "hi": hi, "p": cf.signflip_p(d), "n_pos": sum(x > 0 for x in d)})
    med = {(r["arm"], r["lr"], r["model"]): r["median"] for r in rat}
    shares = []
    for m in models + ["ALL"]:
        rm, rp, rb = med[("main1_N", "published_lr", m)], med[("ext2_N", "published_lr", m)], med[("ext2_N", "believed_lr", m)]
        tot = rm - rb          # drop in r from main1 N (published LR) to ext2 N (believed LR)
        shares.append({"model": m, "r_main_n": rm, "r_ext2_published": rp, "r_ext2_believed": rb, "total_drop": tot,
                       "share_from_believed_lr": (rp - rb) / tot if tot > 0 else None,
                       "share_from_different_answers": (rm - rp) / tot if tot > 0 else None})
    return fits, rat, diffs, len(e2r), shares


# ---------------------------------------------------------------- item 4
def noa4_headlines(rows, cases, models):
    rn = [r for r in rows if r["scenario"] != UTI]
    out = []
    hs = cf.hindsight(rn)[1]
    for h in hs:
        out.append({"headline": "hindsight_pos_minus_neg_stated_pretest", "model": h["model"], "n_scen": h["n_cells"],
                    "n_pos": h["n_gap_pos"], "estimate": h["gap_pp_med"], "estimate_unit": "median pp", "lo": None,
                    "hi": None, "p": h["sign_p"]})
    cells = [dict(c, n_censored=0) for c in cf.ratio_cells(rn, cases, CONDS)]
    for s in summarize_ratio(cells, models):
        out.append({"headline": "ratio_r_median", "model": s["model"], "n_scen": s["n_scen"], "n_pos": s["n_gt1"],
                    "estimate": s["median"], "estimate_unit": "median r", "lo": s["med_lo"], "hi": s["med_hi"],
                    "p": s["sign_p"]})
    for name, X, Y, kind, hyp in cf.CONTRASTS:
        sc = cf.scenario_contrasts(rn, X, Y, kind)
        allm = {}
        for m in models:
            for s, v in sc[m].items():
                allm.setdefault(s, []).append(v)
        for m, d in [(m, sc[m]) for m in models] + [("ALL", {s: mean(v) for s, v in allm.items()})]:
            vals = [d[s] for s in sorted(d)]
            lo, hi = cf.signflip_ci(vals)
            out.append({"headline": f"contrast_{name}_{hyp}", "model": m, "n_scen": len(vals),
                        "n_pos": sum(v > 0 for v in vals), "estimate": mean(vals), "estimate_unit": "mean log-odds",
                        "lo": lo, "hi": hi, "p": cf.signflip_p(vals)})
    return out


# ---------------------------------------------------------------- item 5
def prespec(rows, n_boot):
    """The pre-specified analysis (protocol Section 6): paired case contrasts per model and set, median of paired
    differences, percentile bootstrap over scenarios. Same code as make_tables.contrast_table, fed from raw."""
    scored = [dict(r, parse_ok="1", calibration="0") for r in rows]
    con = mt.contrast_table(scored, n_boot)
    return [c for c in con if c["result"] == "all" and c["hypothesis"] in ("H1", "H2", "H3")
            and c["metric"] in ("band_err_pp", "abs_logodds_err")]


def fmt_cell(row, d):
    if row.get("ci_lo") is None or row.get("ci_lo") == "":
        return "n/a"
    def f(x):
        s = f"{float(x):.{d}f}"
        if float(s) == 0.0:
            s = s.lstrip("-")
        return s.replace("-", "$-$")
    return f"{f(row['median_diff'])} [{f(row['ci_lo'])}, {f(row['ci_hi'])}]"


def table_prespec(con, models):
    """Rows = model x hypothesis (so the table fits one column); cells = median paired difference [95% bootstrap
    interval over scenarios]. Set A in percentage points of band error, Set B in log-odds."""
    idx = {(c["model"], c["set"], c["hypothesis"], c["metric"]): c for c in con}
    lines = ["\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llcc}", "\\toprule",
             "Model & Contrast & Set A (pp) & Set B (log-odds) \\\\", "\\midrule"]
    for k, m in enumerate(models):
        for h, lab in (("H1", "H1: N$-$V"), ("H2", "H2: B$-$N"), ("H3", "H3: S$-$N")):
            name = display_name(m).split()[0] if h == "H1" else ""
            lines.append(f"{name} & {lab} & {fmt_cell(idx[(m, 'A', h, 'band_err_pp')], 1)} & "
                         f"{fmt_cell(idx[(m, 'B', h, 'abs_logodds_err')], 2)} \\\\")
        if k < len(models) - 1:
            lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


# ---------------------------------------------------------------- macros
def macros(bsum, rsum, dec, share, bfits, brat, bdiff, noa, con, models, n_e2, bshare):
    mac = {}
    P = PREFIX
    for s in bsum:
        t = P + "Blind" + ("NoUti" if s["sample"] == "no_A4" else "") + tag(s["model"])
        mac[t + "NScen"], mac[t + "NAbove"] = str(s["n_scen"]), str(s["n_above"])
        for k, nm in (("med_pp", "MedPp"), ("mean_pp", "MeanPp"), ("med_lod", "MedLod"), ("mean_lod", "MeanLod"),
                      ("lod_lo", "Lo"), ("lod_hi", "Hi")):
            mac[t + nm] = n2(s[k])
        mac[t + "SignflipP"] = cf.fmt_p(s["signflip_p"])
        mac[t + "SignTestP"] = cf.fmt_p(s["sign_test_p"])
    for s in rsum:
        t = P + "Range" + ("NoUti" if s["sample"] == "no_A4" else "") + tag(s["model"])
        for k, nm in (("n_scen", "NScen"), ("n_point_only", "NPointOnly"), ("n_range", "NRange"), ("n_above", "NAbove"),
                      ("n_inside", "NInside"), ("n_below", "NBelow"), ("n_answers_with_range", "NAnswers")):
            mac[t + nm] = str(s[k])
        for k, nm in (("pct_ans_above", "PctAnsAbove"), ("pct_ans_inside", "PctAnsInside"),
                      ("pct_ans_below", "PctAnsBelow")):
            mac[t + nm] = n2(s[k])
    for d in dec:
        t = P + "Dec" + ("NoUti" if d["sample"] == "no_A4" else "") + tag(d["model"]) + d["result"].capitalize()
        for k, nm in (("total_mean", "TotMean"), ("prior_mean", "PriorMean"), ("update_mean", "UpdMean"),
                      ("prior_med", "PriorMed"), ("update_med", "UpdMed")):
            mac[t + nm] = n2(d[k])
        mac[t + "N"] = str(d["n"])
    for s in share:
        t = P + "AFourShare" + tag(s["model"]) + s["result"].capitalize()
        mac[t + "Prior"] = n2(s["a4_share_of_prior_mean"])
        mac[t + "Contrib"] = n2(s["a4_contribution"])
        mac[t + "Answers"] = n2(s["a4_share_of_answers"])
    for f in bfits:
        t = P + "Fit" + {"ext2_N": "ETwo", "main1_N": "MainN"}[f["arm"]] + {
            "own_pretest_published_lr": "OwnPub", "own_pretest_believed_lr": "OwnBel",
            "published_prior_published_lr": "PubPub"}[f["variant"]] + tag(f["model"])
        for p in "ab":
            mac[t + p.upper() + "Pt"], mac[t + p.upper() + "Lo"], mac[t + p.upper() + "Hi"] = (
                n2(f[p]), n2(f.get(p + "_lo")), n2(f.get(p + "_hi")))
    for r in brat:
        t = P + "R" + {"ext2_N": "ETwo", "main1_N": "MainN"}[r["arm"]] + {"published_lr": "Pub", "believed_lr": "Bel"}[
            r["lr"]] + tag(r["model"])
        mac[t + "Med"], mac[t + "Lo"], mac[t + "Hi"] = n2(r["median"]), n2(r["med_lo"]), n2(r["med_hi"])
        mac[t + "NGtOne"], mac[t + "NScen"], mac[t + "SignP"] = str(r["n_gt1"]), str(r["n_scen"]), cf.fmt_p(r["sign_p"])
    for d in bdiff:
        t = P + "RDiffETwoMinusMainN" + tag(d["model"])
        mac[t + "Mean"], mac[t + "Lo"], mac[t + "Hi"] = n2(d["mean_diff"]), n2(d["lo"]), n2(d["hi"])
        mac[t + "P"], mac[t + "NPos"], mac[t + "NScen"] = cf.fmt_p(d["p"]), str(d["n_pos"]), str(d["n_scen"])
    mac[P + "ETwoNParsed"] = str(n_e2)
    for d in bshare:
        t = P + "Share" + tag(d["model"])
        mac[t + "BelievedLr"], mac[t + "DifferentAnswers"] = n2(d["share_from_believed_lr"]), n2(d["share_from_different_answers"])
        mac[t + "RMainN"], mac[t + "RETwoPub"], mac[t + "RETwoBel"] = (n2(d["r_main_n"]), n2(d["r_ext2_published"]),
                                                                        n2(d["r_ext2_believed"]))
    for h in noa:
        t = P + "NoUti" + "".join(hw(w).capitalize() if not w.startswith("H") else hw(w) for w in h["headline"].split("_")) + tag(h["model"])
        mac[t + "Est"] = n2(h["estimate"])
        mac[t + "NScen"], mac[t + "NPos"], mac[t + "P"] = str(h["n_scen"]), str(h["n_pos"]), cf.fmt_p(h["p"])
        if h["lo"] is not None:
            mac[t + "Lo"], mac[t + "Hi"] = n2(h["lo"]), n2(h["hi"])
    for c in con:
        t = P + "Prespec" + ("A" if c["set"] == "A" else "B") + hw(c["hypothesis"]) + tag(c["model"])
        mac[t + "Med"] = n2(c["median_diff"])
        if c.get("ci_lo") not in (None, ""):
            mac[t + "Lo"], mac[t + "Hi"] = n2(c["ci_lo"]), n2(c["ci_hi"])
        mac[t + "NClusters"] = str(c["n_clusters"])
    return mac


def write_macros(path, mac, existing):
    for k, v in mac.items():
        assert k.isalpha() and k.startswith(PREFIX), f"bad macro name {k}"
        assert v not in ("", None) and "\u2014" not in str(v), k
        assert k not in existing, f"macro {k} already defined elsewhere"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/round3_analysis.py from results/raw and data/. Do not edit.\n"
                "% Round-3 post hoc analyses; table_prespec is the pre-specified analysis.\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


def existing_macro_names(gen_dir, skip):
    import re
    names = set()
    if os.path.isdir(gen_dir):
        for fn in os.listdir(gen_dir):
            if fn.endswith(".tex") and fn != skip:
                with open(os.path.join(gen_dir, fn)) as f:
                    names |= set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read()))
    return names


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "derived"))
    ap.add_argument("--generated", default=os.path.join(ROOT, "paper", "generated"))
    ap.add_argument("--n-boot", type=int, default=2000)
    add_run_args(ap)
    a = ap.parse_args(argv)
    allowed, run_set, cfg = resolve_run_ids(a)
    cases, final, rows, syn = cf.load_rows(a.raw, a.cases, allowed, cfg)
    cases = {k: v for k, v in cases.items() if v["scenario"] in ORIGINAL_SCENARIOS}   # original 10 scenarios only
    rows = [r for r in rows if r["scenario"] in ORIGINAL_SCENARIOS]
    for d in (a.out, a.generated):
        guard_output(d, syn)
    guard_pilot(a.generated, run_set)
    models = cf.models_of(rows)
    recs, _ = load_raw(a.raw)
    ids = ext.present_ids(recs)
    e1_id, e2_id = ext.DEFAULT_IDS["E1"], ext.DEFAULT_IDS["E2"]
    assert e1_id in ids and e2_id in ids, "ext1_pretest and ext2_belief are required"
    e1_final, e2_final = ext.final_of(recs, e1_id, cfg), ext.final_of(recs, e2_id, cfg)
    o = lambda n: os.path.join(a.out, n)  # noqa: E731

    # item 1
    cells, _, _ = ext.e1(e1_final, rows, cases, 2, {})
    bl, _, _ = ext.blind_cells(e1_final, ext.scenario_targets(cases))
    cell_rows, bsum = blind_prior(cells, models)
    ranges = prior_ranges(cases)
    rcells, rsum = blind_range(cell_rows, bl, ranges, models)
    write_csv(o("r3_blind_prior_cells.csv"), cell_rows, ["model", "scenario", "blind_pct", "published_prior_pct",
                                                         "gap_pp", "gap_lod"])
    write_csv(o("r3_blind_prior_summary.csv"), bsum, ["sample", "model", "n_scen", "n_above", "sign_test_p", "med_pp",
                                                      "mean_pp", "med_lod", "mean_lod", "lod_lo", "lod_hi",
                                                      "signflip_p"])
    write_csv(o("r3_blind_range_cells.csv"), rcells, ["model", "scenario", "blind_pct", "published_prior_pct", "kind",
                                                      "range_lo_pct", "range_hi_pct", "position", "gap_pp", "gap_lod"])
    write_csv(o("r3_blind_range_summary.csv"), rsum, ["sample", "model", "n_scen", "n_point_only", "n_range", "n_above",
                                                      "n_inside", "n_below", "n_answers_with_range", "pct_ans_above",
                                                      "pct_ans_inside", "pct_ans_below"])
    # item 2
    dec, share = decomp(rows, models)
    write_csv(o("r3_decomp.csv"), dec, ["sample", "model", "result", "n", "total_mean", "prior_mean", "update_mean",
                                        "prior_med", "update_med"])
    write_csv(o("r3_decomp_a4share.csv"), share, ["model", "result", "n", "n_a4", "prior_mean_all", "a4_contribution",
                                                  "a4_share_of_prior_mean", "a4_share_of_answers"])
    old = {(d["model"], d["result"]): d for d in cf.decomposition(rows)}      # same identity as F9 (sample = all)
    for d in dec:
        if d["sample"] == "all" and d["result"] != "both":
            assert abs(old[(d["model"], d["result"])]["prior_mean"] - d["prior_mean"]) < 1e-9
            assert old[(d["model"], d["result"])]["n"] == d["n"]
    # item 3
    bfits, brat, bdiff, n_e2, bshare = belief(rows, cases, e2_final, models, a.n_boot)
    write_csv(o("r3_belief_fits.csv"), bfits, ["arm", "variant", "model", "n_obs", "n_clusters", "a", "a_lo", "a_hi",
                                               "b", "b_lo", "b_hi", "c", "n_boot_valid"])
    write_csv(o("r3_belief_ratio.csv"), brat + [dict(d, arm="ext2_N_minus_main1_N", lr="published_lr") for d in bdiff],
              ["arm", "lr", "model", "n_scen", "median", "med_lo", "med_hi", "n_gt1", "sign_p", "n_cells_gt1", "n_cells",
               "n_censored", "mean_diff", "median_diff", "lo", "hi", "p", "n_pos"])
    write_csv(o("r3_belief_share.csv"), bshare, ["model", "r_main_n", "r_ext2_published", "r_ext2_believed",
                                                 "total_drop", "share_from_believed_lr", "share_from_different_answers"])
    # item 4
    noa = noa4_headlines(rows, cases, models)
    write_csv(o("r3_noa4_headlines.csv"), noa, ["headline", "model", "n_scen", "n_pos", "estimate", "estimate_unit", "lo",
                                                "hi", "p"])
    # item 5
    con = prespec(rows, a.n_boot)
    write_csv(o("r3_prespec_contrasts.csv"), con, mt.CON_FIELDS)
    cf.write_tex(os.path.join(a.generated, "table_prespec.tex"), table_prespec(con, models))
    with open(os.path.join(a.generated, "table_prespec.tex")) as f:
        txt = f.read().replace("GENERATED by analysis/committee_fixes.py", "GENERATED by analysis/round3_analysis.py")
    with open(os.path.join(a.generated, "table_prespec.tex"), "w") as f:
        f.write(txt)
    mac = macros(bsum, rsum, dec, share, bfits, brat, bdiff, noa, con, models, n_e2, bshare)
    write_macros(os.path.join(a.generated, "round3_macros.tex"), mac,
                 existing_macro_names(a.generated, "round3_macros.tex"))
    print(f"round3_analysis: {len(rows)} analysed rows, {len(mac)} macros")
    return {"models": models, "blind": bsum, "range": rsum, "range_cells": rcells, "decomp": dec, "share": share,
            "fits": bfits, "ratio": brat, "rdiff": bdiff, "bshare": bshare, "noa4": noa, "prespec": con, "n_e2": n_e2}, mac


if __name__ == "__main__":
    main()
