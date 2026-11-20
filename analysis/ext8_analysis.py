"""I5 extra model family: ext8_family = qwen/qwen4-max, a ROBUSTNESS ROW, never pooled into the 5-model headline.

Run: conditions V (22 cases incl. the A5 calibration case), N and E (20 cases), seeds 1 to 3, prompt v1, temperature 1.0,
max_tokens 8000, same notes and prompt as main1. Reads ONLY results/raw/*.jsonl (same allow-list and latest-attempt
dedup as score.py: the one call_error record is superseded by its later successful attempt) and data/cases. Never
writes to results/raw/. Original scenario ids only (B7/B8 never included); A5 is excluded from every error, contrast,
ratio, hindsight and stated-pretest cell (ext.v1_rows drops calibration cases) but counted in the parse counts.
Post hoc, robustness only. Writes:
  results/derived/ext8_parse.csv       parse counts (records, keys, parsed, unparsed, retried) and every unparsed key
  results/derived/ext8_error.csv       mean abs log-odds error vs the exact target and mean band error (pp) per model x condition
                                       (Qwen and, for comparison, the 5-model main1 roster; roster given with all seeds and seeds 1-3)
  results/derived/ext8_contrasts.csv   paired N minus V and E minus N (abs log-odds error), scenario-level sign-flip CI,
                                       Qwen and each roster model (same code as ext6 / main per-model contrasts)
  results/derived/ext8_ratio.csv       update ratio r per model x scenario (V and N pooled), plus summary rows
  results/derived/ext8_hindsight.csv   stated-pretest gap, positive minus negative result (main F2 definition), per model
  results/derived/ext8_pretest.csv     per scenario median stated pretest vs the published prior, Qwen and roster
  paper/generated/ext8_macros.tex      \\extEight... macros (letters only; none redefines an existing macro)
  paper/generated/table_ext8_rows.tex  one row "E8 Qwen" for the Table 3 body (see the comment in the file)

Definitions (same code as the paper):
  abs log-odds error = |logit(stated posttest) - logit(exact target)| (score.py abs_logodds_err; clipped logits);
  band error = distance in pp outside the primary evidence band (score.py band_err_pp).
  contrasts = cf-style paired difference by case and seed, averaged to scenarios (ext.contrast_rows), 10 scenario clusters,
  exact sign-flip CI and p (cf.signflip_ci, cf.signflip_p).
  r = committee_fixes.ratio_cells on V and N (no B or S exist in ext8): within scenario (mean logit p+ - mean logit p-)
  / (log LR+ - log LR-), median over scenarios per model (round2_analysis.summarize_ratio, exact median CI, sign test).
  hindsight = committee_fixes.hindsight (stated pretest after the positive version minus after the negative version,
  scenario level, exact sign test). Computed on N (headline) and on V+N. NOT the blind-pretest hindsight: ext8 has no
  blind-pretest run (prompt v2_pretest_only), so "stated pretest after vs blind" is NOT computable for Qwen.
  stated pretest vs prior = per scenario median stated pretest (V and N answers; the published-prior conditions of main
  that exist in ext8) against target.pretest_point; the main count (10 of 10) uses V N B S, so the roster is recomputed
  here on V and N for a like-for-like comparison.

Usage: python analysis/ext8_analysis.py [--raw GLOB] [--cases DIR] [--out DIR] [--generated DIR] [--no-check]
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import committee_fixes as cf  # noqa: E402
import ext_analysis as ext  # noqa: E402
from common import (CONFIG_PATH, ROOT, guard_output, is_calibration, is_synthetic, load_cases, load_config,  # noqa: E402
                    load_raw, write_csv)
from make_tables import model_tag  # noqa: E402
from round2_analysis import ORIGINAL_SCENARIOS, summarize_ratio  # noqa: E402

RUN_ID = "ext8_family"
QWEN = "qwen/qwen4-max"
CONDS = ["V", "N", "E"]
RCONDS = ["V", "N"]
SEEDS3 = (1, 2, 3)


def n2(x, d=2):
    return ext.num(x, d)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


# ------------------------------------------------------------------ loading
def load_all(raw_glob, cases_dir, cfg):
    cases_all = load_cases(cases_dir)
    cases = {k: v for k, v in cases_all.items() if v["scenario"] in ORIGINAL_SCENARIOS}
    assert len(cases) == 22, len(cases)
    recs, _ = load_raw(raw_glob)
    syn = is_synthetic(recs, raw_glob)
    assert RUN_ID in ext.present_ids(recs), f"{RUN_ID} not in raw"
    raw8 = [r for r in recs if r.get("run_id") == RUN_ID]
    final8 = ext.final_of(recs, RUN_ID, cfg)
    assert all(r["case_id"] in cases for r in final8), "ext8 holds a case outside the original scenarios"
    main_final = ext.final_of(recs, cfg["paper_run_ids"][0], cfg)
    main_final = [r for r in main_final if r["case_id"] in cases]
    q_rows = ext.v1_rows([r for r in final8 if r.get("prompt_version") == "v1"], cases)
    m_rows = ext.v1_rows([r for r in main_final if r.get("prompt_version", "v1") in ("v1", "v4")], cases)
    return {"cases": cases, "raw8": raw8, "final8": final8, "main_final": main_final, "q_rows": q_rows,
            "m_rows": m_rows, "syn": syn}


# ------------------------------------------------------------------ pieces
def parse_block(D):
    raw8, final = D["raw8"], D["final8"]
    key = lambda r: (r["case_id"], r["condition"], r["model"], r.get("seed"))  # noqa: E731
    final_keys = {key(r) for r in final}
    superseded = [r for r in raw8 if r not in final]
    bad = [r for r in final if not ext.parse_ok(r, ["pretest", "posttest"])]
    out = {"n_records": len(raw8), "n_keys": len(final), "n_parsed": len(final) - len(bad), "n_unparsed": len(bad),
           "n_superseded": len(superseded),
           "n_retried_call_error": sum(r.get("parse_status") == "call_error" and key(r) in final_keys for r in superseded),
           "n_final_call_error": sum(r.get("parse_status") == "call_error" for r in final),
           "n_trunc": sum(ext.length_stopped(r) for r in bad),
           "n_analysed": len(D["q_rows"]), "n_a5": sum(is_calibration(D["cases"][r["case_id"]]) for r in final),
           "unparsed": [{"case_id": r["case_id"], "condition": r["condition"], "seed": r.get("seed"),
                         "parse_status": r.get("parse_status"), "length_stopped": int(ext.length_stopped(r)),
                         "completion_tokens": r.get("completion_tokens")} for r in bad]}
    return out


def error_block(q_rows, m_rows, models):
    out = []
    arms = [("Qwen", q_rows, [QWEN]), ("Roster", m_rows, models),
            ("RosterSeedsOneToThree", [r for r in m_rows if r["seed"] in SEEDS3], models)]
    for arm, rows, ms in arms:
        for m in ms + ["ALL"]:
            for cond in CONDS:
                rs = [r for r in rows if r["condition"] == cond and (m == "ALL" or r["model"] == m)]
                out.append({"arm": arm, "model": m, "condition": cond, "n": len(rs),
                            "mean_abs_lod_err": mean(r["abs_logodds_err"] for r in rs),
                            "mean_band_err_pp": mean(r["band_err_pp"] for r in rs),
                            "mean_signed_lod_err": mean(r["logodds_err"] for r in rs)})
    return out


def contrast_block(q_rows, m_rows, models):
    out = []
    for arm, rows, ms in (("Qwen", q_rows, [QWEN]), ("Roster", m_rows, models)):
        for name, X, Y in (("NvsV", "N", "V"), ("EvsN", "E", "N")):
            rx = [r for r in rows if r["condition"] == X]
            ry = [r for r in rows if r["condition"] == Y]
            for c in ext.contrast_rows(name, rx, ry):
                if c["metric"] == "abs" and c["result"] == "all" and (c["model"] in ms or c["model"] == "ALL"):
                    if arm == "Qwen" and c["model"] == "ALL":
                        continue                    # a single model: ALL would duplicate it
                    out.append(dict(c, arm=arm))
    # R4 M1: the SIGNED E minus N shift (H4 estimand, cfCtrEvsNAll method) for Qwen, kept under its own arm label
    rx = [r for r in q_rows if r["condition"] == "E"]
    ry = [r for r in q_rows if r["condition"] == "N"]
    for c in ext.contrast_rows("EvsN", rx, ry):
        if c["metric"] == "signed" and c["result"] == "all" and c["model"] == QWEN:
            out.append(dict(c, arm="QwenSigned"))
    return out


def ratio_block(q_rows, m_rows, models, cases):
    out, summ = [], []
    for arm, rows, ms in (("Qwen", q_rows, [QWEN]), ("Roster", m_rows, models)):
        for label, conds in (("VN", RCONDS), ("N", ["N"])):
            cells = [dict(c, n_censored=0) for c in cf.ratio_cells(rows, cases, conds)]
            for c in cells:
                out.append({"arm": arm, "conds": label, "kind": "cell", "model": c["model"], "scenario": c["scenario"],
                            "n_pos": c["n_pos"], "n_neg": c["n_neg"], "dy": c["dy"], "log_dor": c["log_dor"],
                            "ratio": c["ratio"]})
            for s in summarize_ratio(cells, ms):
                if arm == "Qwen" and s["model"] == "ALL":
                    continue
                summ.append(dict(s, arm=arm, conds=label))
                out.append({"arm": arm, "conds": label, "kind": "summary", "model": s["model"], "scenario": "",
                            "n_pos": s["n_scen"], "n_neg": s["n_gt1"], "dy": s["med_lo"], "log_dor": s["med_hi"],
                            "ratio": s["median"], "sign_p": s["sign_p"]})
    return out, summ


def hind_block(q_rows, m_rows):
    out, summ = [], []
    for arm, rows in (("Qwen", q_rows), ("Roster", m_rows)):
        for label, conds in (("N", ["N"]), ("VN", ["V", "N"])):
            cells, s = cf.hindsight(rows, conds=conds)
            for x in s:
                if arm == "Qwen" and x["model"] == "ALL":
                    continue
                summ.append(dict(x, arm=arm, conds=label))
                out.append({"arm": arm, "conds": label, "model": x["model"], "n_scen": x["n_cells"],
                            "n_gap_pos": x["n_gap_pos"], "n_gap_neg": x["n_gap_neg"], "sign_p": x["sign_p"],
                            "gap_pp_med": x["gap_pp_med"], "gap_pp_mean": x["gap_pp_mean"],
                            "pre_pos_mean_pct": x["pre_pos_mean_pct"], "pre_neg_mean_pct": x["pre_neg_mean_pct"]})
    return out, summ


def pretest_block(q_rows, m_rows, models, cases):
    pub = {}
    for c in cases.values():
        if not is_calibration(c):
            pub.setdefault(c["scenario"], set()).add(c["target"]["pretest_point"])
    out = []
    for arm, rows, ms in (("Qwen", q_rows, [QWEN]), ("Roster", m_rows, models)):
        for label, conds in (("VN", RCONDS), ("N", ["N"])):
            for m in ms:
                for s in sorted(pub):
                    assert len(pub[s]) == 1, s
                    p = next(iter(pub[s]))
                    rs = [r["pre"] for r in rows if r["model"] == m and r["scenario"] == s and r["condition"] in conds]
                    med = statistics.median(rs)
                    out.append({"arm": arm, "conds": label, "model": m, "scenario": s, "n": len(rs),
                                "published_prior_pct": 100 * p, "median_stated_pretest_pct": 100 * med,
                                "median_minus_published_pp": 100 * (med - p), "exceeds_published": int(med > p)})
    return out


def pick(rows, **kw):
    r = [x for x in rows if all(x[k] == v for k, v in kw.items())]
    assert len(r) == 1, (kw, len(r))
    return r[0]


# ------------------------------------------------------------------ macros
def make_macros(P, err, ctr, rsum, hsum, pre, models):
    mac = {}
    for k, name in (("n_records", "NRecords"), ("n_keys", "NKeys"), ("n_parsed", "NParsed"), ("n_unparsed", "NUnparsed"),
                    ("n_superseded", "NSuperseded"), ("n_retried_call_error", "NRetried"),
                    ("n_final_call_error", "NFinalCallError"), ("n_trunc", "NTrunc"), ("n_analysed", "NAnalysed"),
                    ("n_a5", "NCalibration")):
        mac["extEightQwen" + name] = str(P[k])
    for e in err:
        t = "extEight" + ("Qwen" if e["arm"] == "Qwen" else e["arm"]) + ("" if e["model"] in (QWEN,) else
                                                                      ("All" if e["model"] == "ALL" else model_tag(e["model"])))
        t += e["condition"]
        mac[t + "AbsLod"] = n2(e["mean_abs_lod_err"])
        mac[t + "BandPp"] = n2(e["mean_band_err_pp"], 1)
        mac[t + "N"] = str(e["n"])
    for c in ctr:
        if c["arm"] == "QwenSigned":
            t = "extEightCtrEvsNSignedQwen"
            mac[t + "Mean"], mac[t + "Lo"], mac[t + "Hi"] = n2(c["mean"]), n2(c["lo"]), n2(c["hi"])
            mac[t + "P"] = cf.fmt_p(c["p_signflip"])
            mac[t + "NPos"], mac[t + "NScen"] = str(c["n_scen_pos"]), str(c["n_scen"])
            continue
        who = "Qwen" if c["arm"] == "Qwen" else "Roster" + ("All" if c["model"] == "ALL" else model_tag(c["model"]))
        t = f"extEightCtr{c['contrast']}{who}"
        mac[t + "Mean"], mac[t + "Lo"], mac[t + "Hi"] = n2(c["mean"]), n2(c["lo"]), n2(c["hi"])
        mac[t + "P"] = cf.fmt_p(c["p_signflip"])
        mac[t + "NPos"], mac[t + "NScen"] = str(c["n_scen_pos"]), str(c["n_scen"])
    for s in rsum:
        who = "Qwen" if s["arm"] == "Qwen" else "Roster" + ("All" if s["model"] == "ALL" else model_tag(s["model"]))
        t = f"extEightR{s['conds']}{who}"
        mac[t + "Med"], mac[t + "Lo"], mac[t + "Hi"] = n2(s["median"]), n2(s["med_lo"]), n2(s["med_hi"])
        mac[t + "NGtOne"], mac[t + "NScen"] = str(s["n_gt1"]), str(s["n_scen"])
        mac[t + "SignP"] = cf.fmt_p(s["sign_p"])
    for h in hsum:
        who = "Qwen" if h["arm"] == "Qwen" else "Roster" + ("All" if h["model"] == "ALL" else model_tag(h["model"]))
        t = f"extEightHind{h['conds']}{who}"
        mac[t + "MedPp"] = n2(h["gap_pp_med"], 1)
        mac[t + "NPos"], mac[t + "NScen"] = str(h["n_gap_pos"]), str(h["n_cells"])
        mac[t + "SignP"] = cf.fmt_p(h["sign_p"])
    for arm, ms in (("Qwen", [QWEN]), ("Roster", models)):
        for label in ("VN", "N"):
            for m in ms:
                rows = [x for x in pre if x["arm"] == arm and x["conds"] == label and x["model"] == m]
                who = "Qwen" if arm == "Qwen" else "Roster" + model_tag(m)
                t = f"extEightPre{label}{who}"
                mac[t + "NExceed"] = str(sum(x["exceeds_published"] for x in rows))
                mac[t + "NScen"] = str(len(rows))
                mac[t + "MedExcessPp"] = n2(statistics.median(x["median_minus_published_pp"] for x in rows), 1)
    for label in ("VN", "N"):
        ex = [sum(x["exceeds_published"] for x in pre if x["arm"] == "Roster" and x["conds"] == label and x["model"] == m)
              for m in models]
        mac[f"extEightPre{label}RosterMinNExceed"], mac[f"extEightPre{label}RosterMaxNExceed"] = str(min(ex)), str(max(ex))
    pm = [pick(rsum, arm="Roster", conds="VN", model=m)["median"] for m in models]
    mac["extEightRVNRosterModelsGtOne"] = str(sum(v > 1 for v in pm))
    mac["extEightRVNRosterModelsN"] = str(len(pm))
    return mac


def write_macros(path, mac):
    for k, v in mac.items():
        assert k.isalpha() and k.startswith("extEight"), f"bad macro name {k}"
        assert v not in ("", None) and "\u2014" not in str(v), k
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/ext8_analysis.py from results/raw and data/. Do not edit.\n"
                "% I5 extra family (qwen/qwen4-max, run ext8_family): robustness row, post hoc, never pooled into the 5-model headline.\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


def f2(x):
    s = f"{float(x):.2f}"
    if float(s) == 0.0:
        s = "0.00"
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def table_rows(ctr, rsum, hsum):
    nv = pick(ctr, arm="Qwen", contrast="NvsV")["mean"]
    en = pick(ctr, arm="Qwen", contrast="EvsN")["mean"]
    r = pick(rsum, arm="Qwen", conds="VN", model=QWEN)["median"]
    h = pick(hsum, arm="Qwen", conds="N", model=QWEN)["gap_pp_med"]
    return ["% GENERATED by analysis/ext8_analysis.py. Do not edit.",
            "% ONE row for Qwen (qwen/qwen4-max), post hoc, not one of the five roster columns. Cells, in order:",
            "% N minus V and E minus N change in mean |log-odds error| (paired, same as the E6 rows), update ratio r (V and N),",
            "% median hindsight gap on N (pp, stated pretest after positive minus after negative); last column does not apply.",
            f"E8 Qwen & {f2(nv)} & {f2(en)} & {f2(r)} & {h:.1f} & -- \\\\"]


# ------------------------------------------------------------------ checks
def check(D, P, err, ctr, rsum, hsum, pre, models, mac):
    der = os.path.join(ROOT, "results", "derived")

    def rd(name):
        with open(os.path.join(der, name)) as f:
            return list(csv.DictReader(f))
    # data facts: 187 records, 186 keys (66 V incl. 6 A5, 60 N, 60 E), one call_error superseded by a later success
    assert P["n_records"] == 187 and P["n_keys"] == 186, P
    assert P["n_superseded"] == 1 and P["n_retried_call_error"] == 1 and P["n_final_call_error"] == 0, P
    assert P["n_parsed"] + P["n_unparsed"] == P["n_keys"]
    assert P["n_a5"] == 6 and P["n_analysed"] == P["n_parsed"] - 6 - 0, P
    assert {r["model"] for r in D["final8"]} == {QWEN} and {r["seed"] for r in D["final8"]} == set(SEEDS3)
    conds = {r["condition"] for r in D["final8"]}
    assert conds == set(CONDS), conds
    assert len({r["case_id"] for r in D["final8"] if r["condition"] == "V"}) == 22
    assert len({r["case_id"] for r in D["final8"] if r["condition"] in ("N", "E")}) == 20
    assert not any(r["case_id"].startswith(("B7", "B8")) for r in D["final8"])
    assert len(models) == 5 and QWEN not in models
    # the roster comparison reproduces the existing main1 per-model contrasts and ratios
    for row in rd("cf_contrasts_signflip.csv"):
        if row["contrast"] != "NvsV" or row["model"] == "ALL":
            continue
        c = pick(ctr, arm="Roster", contrast="NvsV", model=row["model"])
        assert abs(c["mean"] - float(row["mean"])) < 1e-9 and abs(c["p_signflip"] - float(row["p_signflip"])) < 1e-9, row
    # 10 scenario clusters everywhere
    for c in ctr:
        assert c["n_scen"] == 10, c
    for s in rsum:
        assert s["n_scen"] == 10, s
    for h in hsum:
        assert h["n_cells"] == 10, h
    assert len([x for x in pre if x["arm"] == "Qwen" and x["conds"] == "VN"]) == 10
    # every macro defined for Qwen
    for k in ("extEightCtrNvsVQwenMean", "extEightCtrEvsNQwenMean", "extEightRVNQwenMed", "extEightHindNQwenNPos",
              "extEightPreVNQwenNExceed"):
        assert k in mac, k


# ------------------------------------------------------------------ main
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
    D = load_all(a.raw, a.cases, cfg)
    for d in (a.out, a.generated):
        guard_output(d, D["syn"])
    cases = D["cases"]
    models = sorted({r["model"] for r in D["m_rows"]})
    q_rows = D["q_rows"]
    P = parse_block(D)
    err = error_block(q_rows, D["m_rows"], models)
    ctr = contrast_block(q_rows, D["m_rows"], models)
    rout, rsum = ratio_block(q_rows, D["m_rows"], models, cases)
    hout, hsum = hind_block(q_rows, D["m_rows"])
    pre = pretest_block(q_rows, D["m_rows"], models, cases)
    mac = make_macros(P, err, ctr, rsum, hsum, pre, models)
    if not a.no_check:
        check(D, P, err, ctr, rsum, hsum, pre, models, mac)
    o = a.out
    write_csv(os.path.join(o, "ext8_parse.csv"),
              [{"kind": "summary", **{k: v for k, v in P.items() if k != "unparsed"}}] +
              [{"kind": "unparsed", **u} for u in P["unparsed"]],
              ["kind", "n_records", "n_keys", "n_parsed", "n_unparsed", "n_superseded", "n_retried_call_error",
               "n_final_call_error", "n_trunc", "n_analysed", "n_a5", "case_id", "condition", "seed", "parse_status",
               "length_stopped", "completion_tokens"])
    write_csv(os.path.join(o, "ext8_error.csv"), err,
              ["arm", "model", "condition", "n", "mean_abs_lod_err", "mean_band_err_pp", "mean_signed_lod_err"])
    write_csv(os.path.join(o, "ext8_contrasts.csv"), ctr, ["arm"] + ext.CTR_FIELDS)
    write_csv(os.path.join(o, "ext8_ratio.csv"), rout,
              ["arm", "conds", "kind", "model", "scenario", "n_pos", "n_neg", "dy", "log_dor", "ratio", "sign_p"])
    write_csv(os.path.join(o, "ext8_hindsight.csv"), hout,
              ["arm", "conds", "model", "n_scen", "n_gap_pos", "n_gap_neg", "sign_p", "gap_pp_med", "gap_pp_mean",
               "pre_pos_mean_pct", "pre_neg_mean_pct"])
    write_csv(os.path.join(o, "ext8_pretest.csv"), pre,
              ["arm", "conds", "model", "scenario", "n", "published_prior_pct", "median_stated_pretest_pct",
               "median_minus_published_pp", "exceeds_published"])
    write_macros(os.path.join(a.generated, "ext8_macros.tex"), mac)
    with open(os.path.join(a.generated, "table_ext8_rows.tex"), "w") as f:
        f.write("\n".join(table_rows(ctr, rsum, hsum)) + "\n")
    print(f"ext8_analysis: {len(mac)} macros")
    return {"P": P, "err": err, "ctr": ctr, "rsum": rsum, "hsum": hsum, "pre": pre, "models": models}, mac


if __name__ == "__main__":
    main()
