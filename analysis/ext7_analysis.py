"""I3 extension: "12 scenarios" = the original 10 analysable scenarios plus B7 (strep pharyngitis,
rapid antigen) and B8 (septic arthritis, synovial WBC). Post hoc extension; the original 10 scenarios stay the
pre-specified analysis and no existing macro or CSV is changed by this script.

Reads ONLY results/raw/*.jsonl (run ids main1 and ext7_newcases, plus ext1..ext4 for the blind-pretest and Holm pieces,
same allow-list and latest-attempt dedup as score.py) and data/cases. Never writes to results/raw/. Writes:
  results/derived/ext7_new_error.csv       new scenarios alone: error by scenario x model x condition (x result)
  results/derived/ext7_new_pretest.csv     new scenarios alone: stated pretest vs the published prior, per scenario x model
  results/derived/ext7_new_hindsight.csv   new scenarios alone: hindsight (pos minus neg stated pretest) and vs blind pretest
  results/derived/ext7_new_ratio.csv       new scenarios alone: update ratio r per model x scenario
  results/derived/ext7_parse.csv           every unparsed record of ext7_newcases with its status
  results/derived/ext7_headlines.csv       headline quantities, arms Ten / Twelve / SeedTwelve, per model and ALL
  results/derived/ext7_scenario_cells.csv  per scenario x model cells (r, hindsight gap) behind the 12-scenario estimates
  results/derived/ext7_contrasts.csv       format contrasts per arm (scenario-level sign-flip CI)
  results/derived/ext7_holm.csv            Holm within hypothesis (per model) and pooled family (m = 7), Ten vs Twelve
  results/derived/ext7_verdicts.csv        stronger / weaker / unchanged per headline, with the rule values
  paper/generated/ext7_macros.tex          \\extSeven... macros (letters only; none redefines an existing macro)
  paper/generated/table_ext7_rows.tex      rows for table_ext.tex style tables ("12 scen.", columns = models)

Arms. Ten = main1 only (the pre-specified scenario set; must reproduce the existing r, hindsight, stated-prior,
contrast and Holm outputs exactly, asserted). Twelve = main1 (all its seeds, as the headline uses them) plus
ext7_newcases. SeedTwelve = Twelve restricted to seeds 1 to 3 in both runs (main1 has 5 seeds on some cells, ext7 has 3).
Seed handling: every estimator here is a scenario-level summary (one value per model x scenario, pooled over conditions
V N B S and seeds exactly as the headline code does), so the number of seeds changes only the precision of a scenario's
value, never the unit of analysis (12 clusters). The seed-matched arm checks that the unequal seed counts do not drive
the result. We did not take a median over seeds because the headline estimators pool answers within a cell.

Headline quantities (same code as the paper, see the module names): inflated stated pretest = per scenario median stated
pretest (V N B S, parse-ok, positive and negative versions pooled) against the published prior
(make_tables.stated_prior_table logic); hindsight = committee_fixes.hindsight (stated pretest after the positive version
minus after the negative, scenario level, exact sign test; Holm within the 5 models and pooled in the m = 7 family of
round2_analysis); update ratio r = committee_fixes.ratio_cells (median over scenarios per model; ALL = median across
models per scenario, then median; exact median CI; sign test); format contrasts = committee_fixes.scenario_contrasts
(H1 N-V, H2 B-N, H3 S-N pre-specified, absolute log-odds error; B-V and S-V added post hoc), sign-flip CI over scenarios.

Materiality (fixed before looking at the 12-scenario numbers). A headline changes MATERIALLY from 10 to 12 scenarios if
any of: (a) a pooled p value crosses 0.05; (b) the number of models with median r > 1 changes; (c) pooled median r
changes by more than 0.15; (d) the pooled hindsight median gap changes by more than 2 pp, or the number of models whose
within-hypothesis Holm-adjusted hindsight p is below 0.05 changes; (e) the number of scenarios whose median stated
pretest exceeds the published prior is not all of them (share below 100%), or the median excess changes by more than
2 pp; (f) a pooled format-contrast CI changes whether it contains 0, or the largest per-model upper bound over
H1 to H3 changes by more than 0.10 log-odds. Direction label, applied per headline on the evidence statistic
(pooled sign p for r and hindsight, -log10 scale; share exceeding for stated pretest; CI width of the pooled N-V
contrast for format): "stronger" if the evidence statistic improves by more than 0.30 in -log10 p (or the CI narrows by
more than 10%), "weaker" if it worsens by the same margin (or widens by more than 10%), else "unchanged".
The label "mixed" was ADDED AFTER the first look at the 12-scenario numbers: for the format null, a narrower CI that comes
with a rise of more than 0.15 in -log10 p of the N-V contrast (more evidence of a worsening) is labelled mixed, not
stronger, because the pooled N-V interval then sits at the edge of zero.

Usage: python analysis/ext7_analysis.py [--raw GLOB] [--cases DIR] [--out DIR] [--generated DIR] [--no-check]
"""
import argparse
import math
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import committee_fixes as cf  # noqa: E402
import ext_analysis as ext  # noqa: E402
from common import (CONFIG_PATH, ROOT, guard_output, is_calibration, is_synthetic, load_cases, load_config,  # noqa: E402
                    load_raw, logit_clipped, write_csv)
from make_tables import model_tag  # noqa: E402
from round2_analysis import ORIGINAL_SCENARIOS, holm, summarize_ratio  # noqa: E402

RUN_ID = "ext7_newcases"
NEW = {"B7_strep": "Strep", "B8_septic": "Septic"}
CONDS = cf.FIT_CONDS                 # V N B S
ALL_CONDS = ["V", "N", "B", "S", "E"]
ARMS = ["Ten", "Twelve", "SeedTwelve"]
SEEDS3 = (1, 2, 3)
MAT_P = 0.05
MAT_R = 0.15
MAT_HIND_PP = 2.0
MAT_EXCESS_PP = 2.0
MAT_HI = 0.10
EV_MARGIN = 0.30
WIDTH_MARGIN = 0.10
MIXED_EV = 0.15


def tag(m):
    return "All" if m == "ALL" else model_tag(m)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def n2(x, d=2):
    return ext.num(x, d)


def lod(p):
    return logit_clipped(p)


# ------------------------------------------------------------------ loading
def load_all(raw_glob, cases_dir, cfg):
    cases = load_cases(cases_dir)
    recs, _ = load_raw(raw_glob)
    syn = is_synthetic(recs, raw_glob)
    orig = {k: v for k, v in cases.items() if v["scenario"] in ORIGINAL_SCENARIOS}
    new = {k: v for k, v in cases.items() if v["scenario"] in NEW}
    assert len(new) == 4 and len(orig) == 22, (len(new), len(orig))
    main_final = ext.final_of(recs, cfg["paper_run_ids"][0], cfg)
    main_rows = ext.v1_rows([r for r in main_final if r["case_id"] in orig], orig)
    assert RUN_ID in ext.present_ids(recs), f"{RUN_ID} not in raw"
    e7_final = ext.final_of(recs, RUN_ID, cfg)
    v1 = [r for r in e7_final if r.get("prompt_version") == "v1"]
    blind = [r for r in e7_final if r.get("prompt_version") == "v2_pretest_only"]
    assert len(v1) + len(blind) == len(e7_final)
    assert all(r["case_id"] in new for r in v1) and all(r["case_id"] in NEW for r in blind)
    e7_rows = ext.v1_rows(v1, new)
    e1_final = ext.final_of(recs, ext.DEFAULT_IDS["E1"], cfg)
    return {"cases": cases, "orig": orig, "new": new, "recs": recs, "syn": syn, "main_final": main_final,
            "main_rows": main_rows, "e7_final": e7_final, "v1": v1, "blind": blind, "e7_rows": e7_rows,
            "e1_final": e1_final}


# ------------------------------------------------------------------ new scenarios alone
def new_error_rows(rows):
    out = []
    for scen in list(NEW) + ["BOTH"]:
        for m in sorted({r["model"] for r in rows}) + ["ALL"]:
            for cond in ALL_CONDS + ["VNBS"]:
                for res in ("all", "pos", "neg"):
                    rs = [r for r in rows if (scen == "BOTH" or r["scenario"] == scen) and
                          (m == "ALL" or r["model"] == m) and
                          (cond == "VNBS" and r["condition"] in CONDS or r["condition"] == cond) and
                          (res == "all" or r["result"] == res)]
                    if not rs:
                        continue
                    out.append({"scenario": scen, "model": m, "condition": cond, "result": res, "n": len(rs),
                                "mean_abs_lod_err": mean(r["abs_logodds_err"] for r in rs),
                                "mean_signed_lod_err": mean(r["logodds_err"] for r in rs),
                                "mean_band_err_pp": mean(r["band_err_pp"] for r in rs),
                                "mean_abs_pp_err_vs_target": mean(abs(r["post"] - r["target"]) * 100 for r in rs),
                                "median_post_pct": statistics.median(r["post"] * 100 for r in rs),
                                "target_pct": 100 * rs[0]["target"] if res != "all" and scen != "BOTH" else ""})
    return out


def new_pretest_rows(rows, cases):
    pub = {s: next(c["target"]["pretest_point"] for c in cases.values() if c["scenario"] == s) for s in NEW}
    out = []
    for scen in NEW:
        for m in sorted({r["model"] for r in rows}) + ["ALL"]:
            rs = [r for r in rows if r["scenario"] == scen and r["condition"] in CONDS and (m == "ALL" or r["model"] == m)]
            med = statistics.median(r["pre"] for r in rs)
            out.append({"scenario": scen, "model": m, "n": len(rs), "published_prior_pct": 100 * pub[scen],
                        "median_stated_pretest_pct": 100 * med, "median_minus_published_pp": 100 * (med - pub[scen]),
                        "median_minus_published_lod": lod(med) - lod(pub[scen]),
                        "mean_stated_pretest_pct": 100 * mean(r["pre"] for r in rs),
                        "exceeds_published": int(med > pub[scen])})
    return out


# ------------------------------------------------------------------ the headline set for one arm
def ratio_block(rows, cases, models):
    cells = [dict(c, n_censored=0) for c in cf.ratio_cells(rows, cases, CONDS)]
    return cells, summarize_ratio(cells, models)


def stated_block(rows, cases, models):
    """Per scenario: median stated pretest (V N B S, parse-ok) vs the published prior; per model counts."""
    pub = {}
    for c in cases.values():
        if not is_calibration(c):
            pub.setdefault(c["scenario"], set()).add(c["target"]["pretest_point"])
    scens = sorted({r["scenario"] for r in rows})
    out = {"pooled": [], "bymodel": {m: [] for m in models}}
    for s in scens:
        assert len(pub[s]) == 1, s
        p = next(iter(pub[s]))
        rs = [r["pre"] for r in rows if r["scenario"] == s and r["condition"] in CONDS]
        med = statistics.median(rs)
        out["pooled"].append({"scenario": s, "published": p, "median": med, "n": len(rs), "exceeds": int(med > p)})
        for m in models:
            mm = statistics.median(r["pre"] for r in rows if r["scenario"] == s and r["condition"] in CONDS
                                   and r["model"] == m)
            out["bymodel"][m].append({"scenario": s, "published": p, "median": mm, "exceeds": int(mm > p)})
    return out


def contrasts_block(rows, full):
    out = []
    spec = [("NvsV", "N", "V", "abs", "H1"), ("BvsN", "B", "N", "abs", "H2"), ("SvsN", "S", "N", "abs", "H3"),
            ("BvsV", "B", "V", "abs", "post hoc"), ("SvsV", "S", "V", "abs", "post hoc")]
    for name, X, Y, kind, hyp in spec:
        sc = cf.scenario_contrasts(rows, X, Y, kind)
        models = sorted(sc)
        allm = {}
        for m in models:
            for s, v in sc[m].items():
                allm.setdefault(s, []).append(v)
        groups = [("ALL", {s: statistics.mean(v) for s, v in allm.items()})]
        if hyp != "post hoc":
            groups += [(m, sc[m]) for m in models]
        for m, d in groups:
            vals = [d[s] for s in sorted(d)]
            lo, hi = cf.signflip_ci(vals) if full else (None, None)
            out.append({"contrast": name, "hypothesis": hyp, "model": m, "n_scen": len(vals),
                        "mean": statistics.mean(vals), "lo": lo, "hi": hi, "p_signflip": cf.signflip_p(vals),
                        "n_scen_pos": sum(v > 0 for v in vals)})
    return out


def blind_block(e1_final, rows, cases):
    """ext1-style blind-pretest cells for the given rows (e1_final may hold ext1 and ext7 blind answers)."""
    cells, summ, _fits = ext.e1(e1_final, rows, cases, 2, {})
    return cells, summ


def arm_headlines(rows, cases, models, e1_final, full):
    r_cells, r_sum = ratio_block(rows, cases, models)
    h_cells, h_sum = cf.hindsight(rows)
    st = stated_block(rows, cases, models)
    ctr = contrasts_block(rows, full)
    res = {"r_cells": r_cells, "r_sum": r_sum, "h_cells": h_cells, "h_sum": h_sum, "stated": st, "ctr": ctr}
    if full:
        b_cells, b_sum = blind_block(e1_final, rows, cases)
        res.update(b_cells=b_cells, b_sum=b_sum)
    return res


def pick(summ, m):
    return next(s for s in summ if s["model"] == m)


def holm_tables(h, models, long_unnamed_p):
    """Within-hypothesis Holm over the 5 models (Ratio, Hind, BlindPos, BlindNeg) and the pooled m = 7 family."""
    fam = []
    sets = [("Ratio", [(m, pick(h["r_sum"], m)["sign_p"]) for m in models + ["ALL"]]),
            ("Hind", [(m, pick(h["h_sum"], m)["sign_p"]) for m in models + ["ALL"]]),
            ("BlindPos", [(m, pick(h["b_sum"], m)["gap_pos_lod_p"]) for m in models + ["ALL"]]),
            ("BlindNeg", [(m, pick(h["b_sum"], m)["gap_neg_lod_p"]) for m in models + ["ALL"]])]
    pooled = {}
    for name, ps in sets:
        per = [(m, p) for m, p in ps if m != "ALL"]
        adj = holm([p for _, p in per])
        for (m, p), a in zip(per, adj):
            fam.append({"family": name, "model": m, "p": p, "p_holm_within": a})
        pooled[name] = next(p for m, p in ps if m == "ALL")
    pooled.update(long_unnamed_p)
    order = ["Ratio", "Hind", "BlindPos", "BlindNeg", "Long", "UnnamedN", "UnnamedL"]
    adj = holm([pooled[k] for k in order])
    for k, a in zip(order, adj):
        fam.append({"family": "Pooled", "model": k, "p": pooled[k], "p_holm_within": a})
    return fam


# ------------------------------------------------------------------ macros
def head_macros(arm, h, fam, models, mac):
    A = "extSeven" + arm
    for m in models + ["ALL"]:
        r, hs = pick(h["r_sum"], m), pick(h["h_sum"], m)
        t = A + tag(m)
        mac[t + "RMed"], mac[t + "RLo"], mac[t + "RHi"] = n2(r["median"]), n2(r["med_lo"]), n2(r["med_hi"])
        mac[t + "RNGtOne"], mac[t + "RNScen"] = str(r["n_gt1"]), str(r["n_scen"])
        mac[t + "RSignP"] = cf.fmt_p(r["sign_p"])
        mac[t + "HindMedPp"] = n2(hs["gap_pp_med"], 1)
        mac[t + "HindNPos"], mac[t + "HindNScen"] = str(hs["n_gap_pos"]), str(hs["n_cells"])
        mac[t + "HindSignP"] = cf.fmt_p(hs["sign_p"])
    pm = [pick(h["r_sum"], m) for m in models]
    mac[A + "RModelsGtOne"] = str(sum(s["median"] > 1 for s in pm))
    mac[A + "RModelsN"] = str(len(pm))
    mac[A + "RMedMin"], mac[A + "RMedMax"] = n2(min(s["median"] for s in pm)), n2(max(s["median"] for s in pm))
    mac[A + "RScenGtOneMin"], mac[A + "RScenGtOneMax"] = str(min(s["n_gt1"] for s in pm)), str(max(s["n_gt1"] for s in pm))
    st = h["stated"]["pooled"]
    mac[A + "StatedNScen"], mac[A + "StatedNExceed"] = str(len(st)), str(sum(s["exceeds"] for s in st))
    mac[A + "StatedMedExcessPp"] = n2(100 * statistics.median(s["median"] - s["published"] for s in st), 1)
    mac[A + "StatedMedExcessLod"] = n2(statistics.median(lod(s["median"]) - lod(s["published"]) for s in st))
    for m in models:
        mac[A + tag(m) + "StatedNExceed"] = str(sum(s["exceeds"] for s in h["stated"]["bymodel"][m]))
    for c in h["ctr"]:
        t = A + "Ctr" + c["contrast"] + tag(c["model"])
        mac[t + "Mean"], mac[t + "P"] = n2(c["mean"]), cf.fmt_p(c["p_signflip"])
        mac[t + "NPos"], mac[t + "N"] = str(c["n_scen_pos"]), str(c["n_scen"])
        if c["lo"] is not None:
            mac[t + "Lo"], mac[t + "Hi"] = n2(c["lo"]), n2(c["hi"])
    h13 = [c for c in h["ctr"] if c["hypothesis"] in ("H1", "H2", "H3") and c["model"] != "ALL"]
    if h13 and h13[0]["lo"] is not None:
        mac[A + "CtrHOneToThreeMaxHi"] = n2(max(c["hi"] for c in h13))
        mac[A + "CtrHOneToThreeMinLo"] = n2(min(c["lo"] for c in h13))
        mac[A + "CtrHOneToThreeNExclZero"] = str(sum(c["lo"] > 0 or c["hi"] < 0 for c in h13))
    if fam:
        for f in fam:
            if f["family"] == "Pooled":
                mac[A + "HolmPooled" + f["model"] + "RawP"] = cf.fmt_p(f["p"])
                mac[A + "HolmPooled" + f["model"] + "AdjP"] = cf.fmt_p(f["p_holm_within"])
            else:
                t = A + "Holm" + f["family"] + tag(f["model"])
                mac[t + "RawP"], mac[t + "AdjWithin"] = cf.fmt_p(f["p"]), cf.fmt_p(f["p_holm_within"])
        for name in ("Ratio", "Hind", "BlindPos", "BlindNeg"):
            sub = [f for f in fam if f["family"] == name]
            mac[A + "Holm" + name + "NSurviveWithin"] = str(sum(f["p_holm_within"] < 0.05 for f in sub))
            mac[A + "Holm" + name + "NRaw"] = str(sum(f["p"] < 0.05 for f in sub))
            mac[A + "Holm" + name + "NTests"] = str(len(sub))
        for k in ("BlindPos", "BlindNeg"):
            for s in h["b_sum"]:
                t = A + tag(s["model"]) + k
                key = "gap_pos" if k == "BlindPos" else "gap_neg"
                mac[t + "Lod"], mac[t + "P"] = n2(s[key + "_lod_mean"]), cf.fmt_p(s[key + "_lod_p"])
                mac[t + "NPos"], mac[t + "NScen"] = str(s[key + "_npos"]), str(s["n_scen"])


def evidence(p):
    return -math.log10(max(p, 1e-12))


def verdicts(h10, h12, f10, f12, models):
    """Apply the pre-declared rules (module docstring). Returns rows and an overall material flag."""
    rows = []

    def add(head, stat, v10, v12, material, label, note=""):
        rows.append({"headline": head, "statistic": stat, "ten": v10, "twelve": v12,
                     "material": "yes" if material else "no", "direction": label, "note": note})

    def lab(delta, margin=EV_MARGIN):
        return "stronger" if delta > margin else "weaker" if delta < -margin else "unchanged"

    r10, r12 = pick(h10["r_sum"], "ALL"), pick(h12["r_sum"], "ALL")
    g10 = sum(pick(h10["r_sum"], m)["median"] > 1 for m in models)
    g12 = sum(pick(h12["r_sum"], m)["median"] > 1 for m in models)
    cross = (r10["sign_p"] < MAT_P) != (r12["sign_p"] < MAT_P)
    mat = cross or g10 != g12 or abs(r12["median"] - r10["median"]) > MAT_R
    add("update ratio r above 1 (pooled)", "pooled sign p (r median, models with r>1)",
        f"p={r10['sign_p']:.4f}; r={r10['median']:.3f}; {g10} of {len(models)}",
        f"p={r12['sign_p']:.4f}; r={r12['median']:.3f}; {g12} of {len(models)}", mat,
        lab(evidence(r12["sign_p"]) - evidence(r10["sign_p"])))
    s10, s12 = pick(h10["h_sum"], "ALL"), pick(h12["h_sum"], "ALL")
    w10 = next(f for f in f10 if f["family"] == "Pooled" and f["model"] == "Hind")
    w12 = next(f for f in f12 if f["family"] == "Pooled" and f["model"] == "Hind")
    n10 = sum(f["p_holm_within"] < 0.05 for f in f10 if f["family"] == "Hind")
    n12 = sum(f["p_holm_within"] < 0.05 for f in f12 if f["family"] == "Hind")
    cross = ((s10["sign_p"] < MAT_P) != (s12["sign_p"] < MAT_P)) or ((w10["p_holm_within"] < MAT_P) != (w12["p_holm_within"] < MAT_P))
    mat = cross or abs(s12["gap_pp_med"] - s10["gap_pp_med"]) > MAT_HIND_PP or n10 != n12
    add("hindsight in the stated pretest (pooled)", "pooled sign p (median gap pp, models surviving Holm within)",
        f"p={s10['sign_p']:.4f}; adj={w10['p_holm_within']:.4f}; gap={s10['gap_pp_med']:.2f}; {n10} of {len(models)}",
        f"p={s12['sign_p']:.4f}; adj={w12['p_holm_within']:.4f}; gap={s12['gap_pp_med']:.2f}; {n12} of {len(models)}", mat,
        lab(evidence(s12["sign_p"]) - evidence(s10["sign_p"])))
    b10, b12 = pick(h10["b_sum"], "ALL"), pick(h12["b_sum"], "ALL")
    mat = ((b10["gap_pos_lod_p"] < MAT_P) != (b12["gap_pos_lod_p"] < MAT_P))
    add("hindsight vs blind pretest after positive (pooled)", "pooled sign-flip p",
        f"p={b10['gap_pos_lod_p']:.4f}", f"p={b12['gap_pos_lod_p']:.4f}", mat,
        lab(evidence(b12["gap_pos_lod_p"]) - evidence(b10["gap_pos_lod_p"])))
    p10, p12 = h10["stated"]["pooled"], h12["stated"]["pooled"]
    sh10, sh12 = mean(s["exceeds"] for s in p10), mean(s["exceeds"] for s in p12)
    ex10 = 100 * statistics.median(s["median"] - s["published"] for s in p10)
    ex12 = 100 * statistics.median(s["median"] - s["published"] for s in p12)
    mat = sh12 < 1.0 or abs(ex12 - ex10) > MAT_EXCESS_PP
    add("inflated stated pretest", "share of scenarios with median stated pretest above published (median excess pp)",
        f"{sum(s['exceeds'] for s in p10)} of {len(p10)}; {ex10:.2f}", f"{sum(s['exceeds'] for s in p12)} of {len(p12)}; {ex12:.2f}",
        mat, "weaker" if sh12 < sh10 else "stronger" if len(p12) > len(p10) and sh12 == 1.0 else "unchanged",
        "stronger here means the same all-scenarios pattern holds in more clusters")
    c10 = next(c for c in h10["ctr"] if c["contrast"] == "NvsV" and c["model"] == "ALL")
    c12 = next(c for c in h12["ctr"] if c["contrast"] == "NvsV" and c["model"] == "ALL")
    w_10, w_12 = c10["hi"] - c10["lo"], c12["hi"] - c12["lo"]
    incl10, incl12 = c10["lo"] <= 0 <= c10["hi"], c12["lo"] <= 0 <= c12["hi"]
    h13_10 = max(c["hi"] for c in h10["ctr"] if c["hypothesis"] in ("H1", "H2", "H3") and c["model"] != "ALL")
    h13_12 = max(c["hi"] for c in h12["ctr"] if c["hypothesis"] in ("H1", "H2", "H3") and c["model"] != "ALL")
    pooled_incl = [(c["lo"] <= 0 <= c["hi"]) for c in h10["ctr"] if c["model"] == "ALL" and c["hypothesis"] != "post hoc"]
    pooled_incl12 = [(c["lo"] <= 0 <= c["hi"]) for c in h12["ctr"] if c["model"] == "ALL" and c["hypothesis"] != "post hoc"]
    mat = pooled_incl != pooled_incl12 or abs(h13_12 - h13_10) > MAT_HI
    rel = (w_12 - w_10) / w_10
    dev = evidence(c12["p_signflip"]) - evidence(c10["p_signflip"])   # evidence of a worsening N vs V (> 0 = more)
    lab_w = "stronger" if rel < -WIDTH_MARGIN else "weaker" if rel > WIDTH_MARGIN else "unchanged"
    if lab_w == "stronger" and dev > MIXED_EV:   # added after first look at the numbers: see module docstring
        lab_w = "mixed"
    add("format contrasts null (pooled N-V CI width)", "pooled N-V mean [CI] (largest per-model upper bound H1 to H3)",
        f"{c10['mean']:.3f} [{c10['lo']:.3f}, {c10['hi']:.3f}]; p={c10['p_signflip']:.4f}; max hi {h13_10:.3f}",
        f"{c12['mean']:.3f} [{c12['lo']:.4f}, {c12['hi']:.3f}]; p={c12['p_signflip']:.4f}; max hi {h13_12:.3f}", mat,
        lab_w, "stronger = narrower interval around the null; mixed = narrower but the evidence of a worsening rose by "
        f"more than {MIXED_EV} in -log10 p; zero in pooled N-V CI: " + ("yes" if incl10 else "no") + " -> " +
        ("yes" if incl12 else "no"))
    return rows, any(r["material"] == "yes" for r in rows)


# ------------------------------------------------------------------ main build
def build(D, cfg, raw_glob, cases_dir):
    models = sorted({r["model"] for r in D["main_rows"]})
    assert len(models) == 5
    cases = D["cases"]
    rows10 = D["main_rows"]
    rows12 = rows10 + D["e7_rows"]
    rows12s = [r for r in rows12 if r["seed"] in SEEDS3]
    arms = {"Ten": rows10, "Twelve": rows12, "SeedTwelve": rows12s}
    e1_ten = D["e1_final"]
    e1_twelve = D["e1_final"] + [r for r in D["blind"]]
    heads = {}
    for arm in ("Ten", "Twelve"):
        c = {k: v for k, v in cases.items() if arm == "Twelve" or v["scenario"] in ORIGINAL_SCENARIOS}
        heads[arm] = arm_headlines(arms[arm], c, models, e1_ten if arm == "Ten" else e1_twelve, True)
    c12 = cases
    heads["SeedTwelve"] = arm_headlines(rows12s, c12, models, None, False)
    # Long / UnnamedN / UnnamedL pooled p values are 10-scenario results (no new-scenario data exist for E3, E4)
    with tempfile.TemporaryDirectory() as tmp:
        orig_dir = os.path.join(tmp, "cases")
        os.makedirs(orig_dir)
        for k, c in D["orig"].items():
            import json
            with open(os.path.join(orig_dir, k + ".json"), "w") as f:
                json.dump(c, f)
        eres, _emac, _have = ext.main(["--raw", raw_glob, "--cases", orig_dir, "--out", tmp, "--generated", tmp,
                                       "--n-boot", "2", "--config", cfg["_path"]])
    lu = {}
    for x in eres["E3"][0]:
        if x["metric"] == "abs" and x["result"] == "all" and x["model"] == "ALL":
            lu["Long"] = x["p_signflip"]
    for x in eres["E4"][0]:
        if x["metric"] == "abs" and x["result"] == "all" and x["model"] == "ALL" and x["contrast"] in ("NunVsN", "LunVsL"):
            lu["UnnamedN" if x["contrast"] == "NunVsN" else "UnnamedL"] = x["p_signflip"]
    assert set(lu) == {"Long", "UnnamedN", "UnnamedL"}, lu
    fams = {a: holm_tables(heads[a], models, lu) for a in ("Ten", "Twelve")}
    return {"models": models, "arms": arms, "heads": heads, "fams": fams, "lu": lu}


def make_macros(D, B):
    models, heads, fams = B["models"], B["heads"], B["fams"]
    mac = {}
    for arm in ARMS:
        head_macros(arm, heads[arm], fams.get(arm), models, mac)
    # data accounting
    e7 = D["e7_final"]
    bad = [r for r in e7 if r.get("parse_status") != "ok" or not (
        ext.parse_ok(r, ["pretest"]) if r.get("prompt_version") == "v2_pretest_only"
        else ext.parse_ok(r, ["pretest", "posttest"]))]
    mac["extSevenNTotal"], mac["extSevenNParsed"] = str(len(e7)), str(len(e7) - len(bad))
    mac["extSevenNUnparsed"] = str(len(bad))
    mac["extSevenNTotalVOne"], mac["extSevenNTotalBlind"] = str(len(D["v1"])), str(len(D["blind"]))
    mac["extSevenNCallError"] = str(sum(r.get("parse_status") == "call_error" for r in bad))
    mac["extSevenNTrunc"] = str(sum(ext.length_stopped(r) for r in bad))
    mac["extSevenNAnalysedRows"] = str(len(D["e7_rows"]))
    mac["extSevenNScenarios"] = "12"
    # new scenarios alone
    models_new = sorted({r["model"] for r in D["e7_rows"]})
    for row in B["new_error"]:
        if row["result"] != "all" or row["condition"] not in ALL_CONDS + ["VNBS"]:
            continue
        s = "Both" if row["scenario"] == "BOTH" else NEW[row["scenario"]]
        t = f"extSeven{s}{tag(row['model'])}{'Pool' if row['condition'] == 'VNBS' else row['condition']}"
        mac[t + "AbsLod"], mac[t + "BandPp"] = n2(row["mean_abs_lod_err"]), n2(row["mean_band_err_pp"], 1)
        mac[t + "SignedLod"] = n2(row["mean_signed_lod_err"])
    for row in B["new_pretest"]:
        t = f"extSeven{NEW[row['scenario']]}{tag(row['model'])}"
        mac[t + "MedPrePct"] = n2(row["median_stated_pretest_pct"], 1)
        mac[t + "MedExcessPp"] = n2(row["median_minus_published_pp"], 1)
        mac[t + "PreExceeds"] = "yes" if row["exceeds_published"] else "no"
        if row["model"] == "ALL":
            mac[f"extSeven{NEW[row['scenario']]}PubPriorPct"] = n2(row["published_prior_pct"], 1)
    for row in B["new_hind"]:
        t = f"extSeven{NEW[row['scenario']]}{tag(row['model'])}"
        mac[t + "HindPp"] = n2(row["gap_posneg_pp"], 1)
        mac[t + "BlindPosPp"], mac[t + "BlindNegPp"] = n2(row["gap_pos_pp"], 1), n2(row["gap_neg_pp"], 1)
        mac[t + "BlindPp"] = n2(row["blind_pct"], 1)
    for row in B["new_ratio"]:
        t = f"extSeven{NEW[row['scenario']]}{tag(row['model'])}"
        mac[t + "R"] = n2(row["ratio"])
    for s in NEW:
        rr = [r["ratio"] for r in B["new_ratio"] if r["scenario"] == s and r["model"] != "ALL"]
        mac[f"extSeven{NEW[s]}RModelsGtOne"] = str(sum(v > 1 for v in rr))
        mac[f"extSeven{NEW[s]}RAll"] = n2(statistics.median(rr))
    # verdicts
    vr, material = verdicts(heads["Ten"], heads["Twelve"], fams["Ten"], fams["Twelve"], models)
    names = ["Ratio", "Hindsight", "BlindPos", "Stated", "Format"]
    for n, v in zip(names, vr):
        mac[f"extSevenVerdict{n}Direction"] = v["direction"]
        mac[f"extSevenVerdict{n}Material"] = v["material"]
    mac["extSevenMaterial"] = "yes" if material else "no"
    mac["extSevenNMaterialHeadlines"] = str(sum(v["material"] == "yes" for v in vr))
    # seed-matched agreement with the all-seed 12-scenario arm
    a, b = pick(heads["Twelve"]["r_sum"], "ALL"), pick(heads["SeedTwelve"]["r_sum"], "ALL")
    mac["extSevenSeedMatchedDeltaR"] = n2(b["median"] - a["median"])
    mac["extSevenSeedMatchedModelsGtOne"] = mac["extSevenSeedTwelveRModelsGtOne"]
    ha, hb = pick(heads["Twelve"]["h_sum"], "ALL"), pick(heads["SeedTwelve"]["h_sum"], "ALL")
    mac["extSevenSeedMatchedDeltaHindPp"] = n2(hb["gap_pp_med"] - ha["gap_pp_med"], 1)
    return mac, vr


def check(D, B, mac, vr):
    """Assertions on key numbers. Part 1: the Ten arm must reproduce the existing 10-scenario outputs."""
    import csv
    der = os.path.join(ROOT, "results", "derived")

    def rd(name):
        with open(os.path.join(der, name)) as f:
            return list(csv.DictReader(f))
    models, heads = B["models"], B["heads"]
    h10, h12 = heads["Ten"], heads["Twelve"]
    # 1) r and scenario-level counts equal r2_scenario_level.csv
    for row in rd("r2_scenario_level.csv"):
        s = pick(h10["r_sum"], row["model"])
        assert s["n_gt1"] == int(row["n_gt1"]) and s["n_scen"] == int(row["n_scen"]), row["model"]
        assert abs(s["median"] - float(row["median"])) < 1e-9 and abs(s["sign_p"] - float(row["sign_p"])) < 1e-9
    # 2) hindsight equals cf_hindsight_summary.csv
    for row in rd("cf_hindsight_summary.csv"):
        s = pick(h10["h_sum"], row["model"])
        assert s["n_gap_pos"] == int(row["n_gap_pos"]) and abs(s["gap_pp_med"] - float(row["gap_pp_med"])) < 1e-9, row["model"]
        assert abs(s["sign_p"] - float(row["sign_p"])) < 1e-9
    # 3) stated prior equals table_stated_prior.csv
    sp = {r["scenario"]: r for r in rd("table_stated_prior.csv")}
    assert len(sp) == 10 and len(h10["stated"]["pooled"]) == 10
    for s in h10["stated"]["pooled"]:
        assert abs(100 * s["median"] - float(sp[s["scenario"]]["median_stated_pretest_pct"])) < 1e-9, s["scenario"]
        assert s["exceeds"] == int(sp[s["scenario"]]["median_exceeds_published"])
    # 4) contrasts equal cf_contrasts_signflip.csv
    for row in rd("cf_contrasts_signflip.csv"):
        if row["contrast"] not in ("NvsV", "BvsN", "SvsN"):
            continue
        c = next(x for x in h10["ctr"] if x["contrast"] == row["contrast"] and x["model"] == row["model"])
        assert abs(c["mean"] - float(row["mean"])) < 1e-9 and abs(c["p_signflip"] - float(row["p_signflip"])) < 1e-9, row
        assert abs(c["hi"] - float(row["hi"])) < 1e-3 and abs(c["lo"] - float(row["lo"])) < 1e-3, row
    # 5) Holm (Ten) equals r2_holm.csv
    for row in rd("r2_holm.csv"):
        fam = row["family"]
        if fam == "Pooled":
            nm = {"r_gt_1_sign_test": "Ratio", "hindsight_pre_pos_gt_neg": "Hind", "blind_gap_pos": "BlindPos",
                  "blind_gap_neg": "BlindNeg", "long_vs_short_abs": "Long", "NunVsN_abs": "UnnamedN",
                  "LunVsL_abs": "UnnamedL"}[row["test"]]
            f = next(x for x in B["fams"]["Ten"] if x["family"] == "Pooled" and x["model"] == nm)
        elif fam in ("Ratio", "Hind", "BlindPos", "BlindNeg"):
            f = next(x for x in B["fams"]["Ten"] if x["family"] == fam and x["model"] == row["model"])
        else:
            continue
        assert abs(f["p"] - float(row["p"])) < 1e-9, (fam, row["test"], row["model"])
        assert abs(f["p_holm_within"] - float(row["p_holm_within"])) < 1e-9, (fam, row["test"], row["model"])
    # 6) data facts for the new run (390 records: 300 v1 + 90 blind; 2 flagged DeepSeek records)
    assert mac["extSevenNTotal"] == "390" and mac["extSevenNTotalVOne"] == "300" and mac["extSevenNTotalBlind"] == "90"
    assert mac["extSevenNUnparsed"] == "2" and mac["extSevenNCallError"] == "1", mac["extSevenNUnparsed"]
    assert mac["extSevenNAnalysedRows"] == "298"
    assert all("deepseek" in r["model"] for r in D["e7_final"] if r.get("flagged"))
    assert len({r["case_id"] for r in D["v1"]}) == 4 and {r["condition"] for r in D["v1"]} == set(ALL_CONDS)
    # 7) cluster counts
    for m in models + ["ALL"]:
        assert pick(h10["r_sum"], m)["n_scen"] == 10 and pick(h12["r_sum"], m)["n_scen"] == 12, m
        assert pick(h12["h_sum"], m)["n_cells"] == 12
    assert len(h12["stated"]["pooled"]) == 12
    # 8) the blind pretest covers both new scenarios for every model
    assert {c["scenario"] for c in h12["b_cells"] if c["condition"] == "VNS"} >= set(NEW)
    # 9) every B7/B8 cell present for every model
    assert len({(c["model"], c["scenario"]) for c in h12["r_cells"] if c["scenario"] in NEW}) == 10
    # 10) Twelve differs from Ten only through the two new scenarios: the original 10 cells are identical
    k = lambda c: (c["model"], c["scenario"])  # noqa: E731
    r10 = {k(c): c["ratio"] for c in h10["r_cells"]}
    for c in h12["r_cells"]:
        if c["scenario"] not in NEW:
            assert abs(r10[k(c)] - c["ratio"]) < 1e-12
    assert len(vr) == 5


# ------------------------------------------------------------------ outputs
def write_macros(path, mac):
    for k, v in mac.items():
        assert k.isalpha() and k.startswith("extSeven"), f"bad macro name {k}"
        assert v not in ("", None) and "\u2014" not in str(v), k
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/ext7_analysis.py from results/raw and data/. Do not edit.\n"
                "% I3 extension: 12 scenarios (original 10 + B7 strep + B8 septic arthritis). Post hoc; arms Ten, Twelve, SeedTwelve.\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


def f2(x):
    s = f"{float(x):.2f}"
    if float(s) == 0.0:
        s = "0.00"
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def table_rows(B):
    models, heads = B["models"], B["heads"]
    h10, h12 = heads["Ten"], heads["Twelve"]
    lines = ["% GENERATED by analysis/ext7_analysis.py. Do not edit. Rows in the style of table_ext.tex (columns = models, sorted slugs).",
             "% 12 scen. = the original 10 scenarios plus strep (B7) and septic arthritis (B8); post hoc extension."]
    lines.append("E7 $r$, 12 scen. & " + " & ".join(f2(pick(h12["r_sum"], m)["median"]) for m in models) + " \\\\")
    lines.append("E7 hind. gap, 12 scen. (pp) & " + " & ".join(f"{pick(h12['h_sum'], m)['gap_pp_med']:.1f}" for m in models) + " \\\\")
    lines.append("% Same quantities with the original 10 scenarios (pre-specified set), for comparison:")
    lines.append("% E7 $r$, 10 scen. & " + " & ".join(f2(pick(h10["r_sum"], m)["median"]) for m in models) + " \\\\")
    lines.append("% E7 hind. gap, 10 scen. (pp) & " + " & ".join(f"{pick(h10['h_sum'], m)['gap_pp_med']:.1f}" for m in models) + " \\\\")
    return lines


def flat_headline_rows(B):
    out = []
    for arm in ARMS:
        h = B["heads"][arm]
        for m in B["models"] + ["ALL"]:
            r, hs = pick(h["r_sum"], m), pick(h["h_sum"], m)
            row = {"arm": arm, "model": m, "r_median": r["median"], "r_lo": r["med_lo"], "r_hi": r["med_hi"],
                   "r_n_gt1": r["n_gt1"], "r_n_scen": r["n_scen"], "r_sign_p": r["sign_p"],
                   "hind_gap_pp_med": hs["gap_pp_med"], "hind_n_pos": hs["n_gap_pos"], "hind_n_scen": hs["n_cells"],
                   "hind_sign_p": hs["sign_p"]}
            if m != "ALL":
                sb = h["stated"]["bymodel"][m]
                row.update(stated_n_exceed=sum(s["exceeds"] for s in sb), stated_n_scen=len(sb),
                           stated_med_excess_pp=100 * statistics.median(s["median"] - s["published"] for s in sb))
            else:
                sb = h["stated"]["pooled"]
                row.update(stated_n_exceed=sum(s["exceeds"] for s in sb), stated_n_scen=len(sb),
                           stated_med_excess_pp=100 * statistics.median(s["median"] - s["published"] for s in sb))
            out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "derived"))
    ap.add_argument("--generated", default=os.path.join(ROOT, "paper", "generated"))
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cfg["_path"] = a.config
    D = load_all(a.raw, a.cases, cfg)
    for d in (a.out, a.generated):
        guard_output(d, D["syn"])
    B = build(D, cfg, a.raw, a.cases)
    B["new_error"] = new_error_rows(D["e7_rows"])
    B["new_pretest"] = new_pretest_rows(D["e7_rows"], D["cases"])
    # new scenarios alone: hindsight and ratio from the Twelve arm cells restricted to B7 and B8
    h12 = B["heads"]["Twelve"]
    hc = [c for c in h12["h_cells"] if c["scenario"] in NEW and c["condition"] == "VNBS"]
    bc = {(c["model"], c["scenario"]): c for c in h12["b_cells"] if c["scenario"] in NEW and c["condition"] == "VNS"}
    B["new_hind"] = []
    for s in NEW:
        sub = [bc[(m, s)] for m in B["models"]]
        for m in B["models"] + ["ALL"]:
            cs = [bc[(m, s)]] if m != "ALL" else sub
            B["new_hind"].append({"scenario": s, "model": m,
                                  "gap_posneg_pp": mean(c["gap_posneg_pp"] for c in cs),
                                  "gap_pos_pp": mean(c["gap_pos_pp"] for c in cs),
                                  "gap_neg_pp": mean(c["gap_neg_pp"] for c in cs),
                                  "blind_pct": mean(c["blind_pct"] for c in cs),
                                  "pre_pos_pct": mean(c["pre_pos_pct"] for c in cs),
                                  "pre_neg_pct": mean(c["pre_neg_pct"] for c in cs),
                                  "published_prior_pct": cs[0]["published_prior_pct"]})
    # the VNBS stated-pretest gap per model, for the table (all four conditions, hindsight())
    for row in B["new_hind"]:
        if row["model"] != "ALL":
            row["gap_posneg_pp_VNBS"] = next(c["gap_pp"] for c in hc if c["model"] == row["model"] and c["scenario"] == row["scenario"])
    B["new_ratio"] = []
    for s in NEW:
        cs = [c for c in h12["r_cells"] if c["scenario"] == s]
        B["new_ratio"] += [dict(c) for c in cs]
        B["new_ratio"].append({"model": "ALL", "scenario": s, "ratio": statistics.median(c["ratio"] for c in cs),
                               "dy": mean(c["dy"] for c in cs), "log_dor": cs[0]["log_dor"], "n_pos": "", "n_neg": ""})
    mac, vr = make_macros(D, B)
    if not a.no_check:
        check(D, B, mac, vr)
    o = a.out
    write_csv(os.path.join(o, "ext7_new_error.csv"), B["new_error"],
              ["scenario", "model", "condition", "result", "n", "mean_abs_lod_err", "mean_signed_lod_err",
               "mean_band_err_pp", "mean_abs_pp_err_vs_target", "median_post_pct", "target_pct"])
    write_csv(os.path.join(o, "ext7_new_pretest.csv"), B["new_pretest"],
              ["scenario", "model", "n", "published_prior_pct", "median_stated_pretest_pct", "median_minus_published_pp",
               "median_minus_published_lod", "mean_stated_pretest_pct", "exceeds_published"])
    write_csv(os.path.join(o, "ext7_new_hindsight.csv"), B["new_hind"],
              ["scenario", "model", "gap_posneg_pp", "gap_pos_pp", "gap_neg_pp", "blind_pct", "pre_pos_pct", "pre_neg_pct",
               "published_prior_pct", "gap_posneg_pp_VNBS"])
    write_csv(os.path.join(o, "ext7_new_ratio.csv"), B["new_ratio"],
              ["scenario", "model", "n_pos", "n_neg", "dy", "log_dor", "ratio", "target_pos_pct", "target_neg_pct"])
    parse = [{"model": r["model"], "case_id": r["case_id"], "condition": r["condition"], "seed": r.get("seed"),
              "prompt_version": r.get("prompt_version"), "parse_status": r.get("parse_status"),
              "flagged": r.get("flagged"), "length_stopped": int(ext.length_stopped(r)),
              "completion_tokens": r.get("completion_tokens"), "error": r.get("error")}
             for r in D["e7_final"] if r.get("parse_status") != "ok"]
    write_csv(os.path.join(o, "ext7_parse.csv"), parse,
              ["model", "case_id", "condition", "seed", "prompt_version", "parse_status", "flagged", "length_stopped",
               "completion_tokens", "error"])
    write_csv(os.path.join(o, "ext7_headlines.csv"), flat_headline_rows(B),
              ["arm", "model", "r_median", "r_lo", "r_hi", "r_n_gt1", "r_n_scen", "r_sign_p", "hind_gap_pp_med",
               "hind_n_pos", "hind_n_scen", "hind_sign_p", "stated_n_exceed", "stated_n_scen", "stated_med_excess_pp"])
    cells = []
    for arm in ("Ten", "Twelve"):
        h = B["heads"][arm]
        hmap = {(c["model"], c["scenario"]): c for c in h["h_cells"] if c["condition"] == "VNBS"}
        for c in h["r_cells"]:
            hc_ = hmap.get((c["model"], c["scenario"]), {})
            cells.append({"arm": arm, "model": c["model"], "scenario": c["scenario"], "ratio": c["ratio"], "dy": c["dy"],
                          "log_dor": c["log_dor"], "n_pos": c["n_pos"], "n_neg": c["n_neg"],
                          "hind_gap_pp": hc_.get("gap_pp", "")})
    write_csv(os.path.join(o, "ext7_scenario_cells.csv"), cells,
              ["arm", "model", "scenario", "ratio", "dy", "log_dor", "n_pos", "n_neg", "hind_gap_pp"])
    ctr = [dict(c, arm=arm) for arm in ARMS for c in B["heads"][arm]["ctr"]]
    write_csv(os.path.join(o, "ext7_contrasts.csv"), ctr,
              ["arm", "contrast", "hypothesis", "model", "n_scen", "mean", "lo", "hi", "p_signflip", "n_scen_pos"])
    hol = [dict(f, arm=arm) for arm in ("Ten", "Twelve") for f in B["fams"][arm]]
    write_csv(os.path.join(o, "ext7_holm.csv"), hol, ["arm", "family", "model", "p", "p_holm_within"])
    write_csv(os.path.join(o, "ext7_verdicts.csv"), vr,
              ["headline", "statistic", "ten", "twelve", "material", "direction", "note"])
    write_macros(os.path.join(a.generated, "ext7_macros.tex"), mac)
    with open(os.path.join(a.generated, "table_ext7_rows.tex"), "w") as f:
        f.write("\n".join(table_rows(B)) + "\n")
    print(f"ext7_analysis: {len(mac)} macros; material change in any headline: {mac['extSevenMaterial']}")
    return B, mac, vr


if __name__ == "__main__":
    main()
