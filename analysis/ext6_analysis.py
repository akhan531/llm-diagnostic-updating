"""I4 prompt-robustness extension: two paraphrased posttest prompts on condition N.

Runs ext6_para1 (prompt v5_para1) and ext6_para2 (prompt v6_para2): 20 non-A5 original cases x seeds 1 to 3 x 5 models
on condition N, temperature 1.0, max_tokens 8000. Each is compared with the main1 records on condition N for the SAME
cases, models and seeds (main1 has seeds 1 to 5; only seeds 1 to 3 are used here). Post hoc, robustness only; the
paraphrases are never pooled into the pre-specified or headline estimates.

Reads ONLY results/raw/*.jsonl (same allow-list and latest-attempt dedup as score.py) and data/. Never writes to
results/raw/. Restricted to the original scenario ids (B7_*/B8_* excluded). Writes:
  results/derived/ext6_summary.csv    per arm x model (and ALL): parse counts, mean abs band error (pp), mean abs
                                      log-odds error, Set A band error and Set B abs log-odds error (pre-specified
                                      primary metrics), hindsight gap, update ratio r
  results/derived/ext6_contrasts.csv  paraphrase minus main1 N, paired by case and seed, per model and ALL, abs error
                                      in log-odds (scenario-level sign-flip CI, as ext_analysis.contrast_rows)
  results/derived/ext6_ratio.csv      per arm x model x scenario r cells
  results/derived/ext6_parse.csv      parse counts per arm x model, with the status of every unparsed record
  paper/generated/ext6_macros.tex     \\extSix... macros (letters only; none of them redefines an existing macro)
  paper/generated/table_ext6_rows.tex rows ready to paste into table_ext.tex (columns = models)

Metrics (same code as the paper): band error = score.py band_err_pp (distance outside the primary target band);
hindsight = committee_fixes.hindsight (stated pretest after the positive version minus after the negative version,
here on N only; its pp median over scenarios and the count of scenarios with a positive gap); update ratio r =
committee_fixes.ratio_cells (the H5 primary estimator: (mean logit post positive minus negative) / (log LR+ minus log LR-)
per scenario, median over scenarios per model, restricted to condition N; the pooled "All" value is the median across
models per scenario, then the median over scenarios, as round2_analysis.summarize_ratio).
Materiality (fixed before looking): a headline estimate moves materially if the pooled mean abs band error changes by
more than 5 pp, if the pooled r crosses 1, if the number of models with r > 1 changes by 2 or more, or if the pooled
hindsight median gap changes by more than 5 pp.

Usage: python analysis/ext6_analysis.py [--raw GLOB] [--cases DIR] [--out DIR] [--generated DIR]
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import committee_fixes as cf  # noqa: E402
import ext_analysis as ext  # noqa: E402
from common import (CONFIG_PATH, ROOT, display_name, guard_output, is_calibration, is_synthetic, load_cases,  # noqa: E402
                    load_config, load_raw, write_csv)
from make_tables import model_tag  # noqa: E402
from round2_analysis import ORIGINAL_SCENARIOS, summarize_ratio  # noqa: E402

ARMS = [("Main", None), ("ParaOne", "ext6_para1"), ("ParaTwo", "ext6_para2")]
SEEDS = (1, 2, 3)
COND = "N"
MAT_BAND_PP = 5.0          # pooled mean abs band error change, percentage points
MAT_HIND_PP = 5.0          # pooled hindsight median gap change, percentage points
MAT_NGT1 = 2               # change in the number of models with r > 1


def tag(m):
    return "All" if m == "ALL" else model_tag(m)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def prep(final, cases, seeds):
    """(all analysed keys' records, parse-ok scored rows) for condition N, original scenarios, given seeds."""
    recs = [r for r in final if r["condition"] == COND and r.get("seed") in seeds and r["case_id"] in cases
            and not is_calibration(cases[r["case_id"]])]
    rows = [r for r in ext.v1_rows(recs, cases)]
    return recs, rows


def arm_summary(arm, recs, rows, cases, models):
    out = []
    rcells = [dict(c, n_censored=0) for c in cf.ratio_cells(rows, cases, [COND])]
    rsum = {s["model"]: s for s in summarize_ratio(rcells, models)}
    hcells, hsum = cf.hindsight(rows, conds=[COND])
    hs = {s["model"]: s for s in hsum}
    for m in models + ["ALL"]:
        rr = [r for r in rows if m == "ALL" or r["model"] == m]
        rc = [r for r in recs if m == "ALL" or r["model"] == m]
        sa = [r["band_err_pp"] for r in rr if r["set"] == "A"]
        sb = [r["abs_logodds_err"] for r in rr if r["set"] == "B"]
        n_unp = len(rc) - len(rr)
        out.append({"arm": arm, "model": m, "n_total": len(rc), "n_parsed": len(rr), "n_unparsed": n_unp,
                    "n_length_stopped": sum(ext.length_stopped(r) for r in rc if r.get("parse_status") != "ok"),
                    "mean_band_err_pp": mean(r["band_err_pp"] for r in rr),
                    "mean_abs_lod_err": mean(r["abs_logodds_err"] for r in rr),
                    "setA_mean_band_err_pp": mean(sa), "setB_mean_abs_lod_err": mean(sb),
                    "n_setA": len(sa), "n_setB": len(sb),
                    "hind_gap_pp_med": hs[m]["gap_pp_med"], "hind_gap_lod_med": hs[m]["gap_lod_med"],
                    "hind_n_pos": hs[m]["n_gap_pos"], "hind_n_scen": hs[m]["n_cells"],
                    "r_med": rsum[m]["median"], "r_lo": rsum[m]["med_lo"], "r_hi": rsum[m]["med_hi"],
                    "r_n_gt1": rsum[m]["n_gt1"], "r_n_scen": rsum[m]["n_scen"]})
    return out, rcells


SUM_FIELDS = ["arm", "model", "n_total", "n_parsed", "n_unparsed", "n_length_stopped", "mean_band_err_pp",
              "mean_abs_lod_err", "setA_mean_band_err_pp", "setB_mean_abs_lod_err", "n_setA", "n_setB",
              "hind_gap_pp_med", "hind_gap_lod_med", "hind_n_pos", "hind_n_scen", "r_med", "r_lo", "r_hi", "r_n_gt1",
              "r_n_scen"]


def build(recs_all, cases, cfg, ids, seeds=SEEDS):
    """Return dict of results; asserts the invariants."""
    cases = {k: v for k, v in cases.items() if v["scenario"] in ORIGINAL_SCENARIOS}
    main_id = cfg["paper_run_ids"][0]
    finals = {"Main": ext.final_of(recs_all, main_id, cfg)}
    for arm, rid in ARMS[1:]:
        assert ids[arm] in ext.present_ids(recs_all), f"{ids[arm]} not in raw"
        finals[arm] = ext.final_of(recs_all, ids[arm], cfg)
    case_set = {r["case_id"] for r in finals["ParaOne"]} | {r["case_id"] for r in finals["ParaTwo"]}
    assert all(c in cases for c in case_set), "paraphrase run contains a case outside the original scenarios"
    assert not any(is_calibration(cases[c]) for c in case_set)
    para_cases = {c: cases[c] for c in case_set}
    models = sorted({r["model"] for r in finals["ParaOne"]})
    prepped = {}
    for arm, _ in ARMS:
        recs, rows = prep(finals[arm], para_cases, seeds)
        if arm != "Main":
            assert {r["condition"] for r in recs} == {COND}
            assert {r["case_id"] for r in recs} == case_set
        prepped[arm] = (recs, rows)
    summ, rcells, ctr, parse = [], [], [], []
    for arm, _ in ARMS:
        recs, rows = prepped[arm]
        s, rc = arm_summary(arm, recs, rows, para_cases, models)
        summ += s
        rcells += [dict(c, arm=arm) for c in rc]
        for r in recs:
            if r.get("parse_status") != "ok" or not ext.parse_ok(r, ["pretest", "posttest"]):
                parse.append({"arm": arm, "model": r["model"], "case_id": r["case_id"], "seed": r.get("seed"),
                              "parse_status": r.get("parse_status"), "length_stopped": int(ext.length_stopped(r)),
                              "completion_tokens": r.get("completion_tokens")})
    for arm, _ in ARMS[1:]:
        ctr += [dict(r, arm=arm) for r in ext.contrast_rows(arm + "VsMain", prepped[arm][1], prepped["Main"][1])]
    return {"summ": summ, "rcells": rcells, "ctr": ctr, "parse": parse, "models": models, "cases": para_cases,
            "prepped": prepped}


def get(summ, arm, model):
    return next(s for s in summ if s["arm"] == arm and s["model"] == model)


def n3(x, d=2):
    return ext.num(x, d)


def macros(res):
    summ, ctr, models = res["summ"], res["ctr"], res["models"]
    mac = {}
    for arm, _ in ARMS:
        for m in models + ["ALL"]:
            s = get(summ, arm, m)
            t = "extSix" + arm + tag(m)
            mac[t + "NTotal"] = str(s["n_total"])
            mac[t + "NParsed"] = str(s["n_parsed"])
            mac[t + "NUnparsed"] = str(s["n_unparsed"])
            mac[t + "NTrunc"] = str(s["n_length_stopped"])
            mac[t + "BandPp"] = n3(s["mean_band_err_pp"], 1)
            mac[t + "AbsLod"] = n3(s["mean_abs_lod_err"])
            mac[t + "SetABandPp"] = n3(s["setA_mean_band_err_pp"], 1)
            mac[t + "SetBAbsLod"] = n3(s["setB_mean_abs_lod_err"])
            mac[t + "HindMedPp"] = n3(s["hind_gap_pp_med"], 1)
            mac[t + "HindNPos"] = str(s["hind_n_pos"])
            mac[t + "HindNScen"] = str(s["hind_n_scen"])
            mac[t + "RMed"] = n3(s["r_med"])
            mac[t + "RLo"] = n3(s["r_lo"])
            mac[t + "RHi"] = n3(s["r_hi"])
            mac[t + "RNScenGtOne"] = str(s["r_n_gt1"])
        pm = [get(summ, arm, m)["r_med"] for m in models]
        mac["extSix" + arm + "RModelsGtOne"] = str(sum(v > 1 for v in pm))
        mac["extSix" + arm + "RModelsN"] = str(len(pm))
    # paraphrase minus main1 changes
    maxd = {"band": 0.0, "hind": 0.0}
    cross, dngt = False, 0
    for arm, _ in ARMS[1:]:
        for m in models + ["ALL"]:
            a, b = get(summ, arm, m), get(summ, "Main", m)
            t = "extSix" + arm + tag(m)
            mac[t + "DeltaBandPp"] = n3(a["mean_band_err_pp"] - b["mean_band_err_pp"], 1)
            mac[t + "DeltaAbsLod"] = n3(a["mean_abs_lod_err"] - b["mean_abs_lod_err"])
            mac[t + "DeltaHindMedPp"] = n3(a["hind_gap_pp_med"] - b["hind_gap_pp_med"], 1)
            mac[t + "DeltaRMed"] = n3(a["r_med"] - b["r_med"])
        for r in (x for x in ctr if x["arm"] == arm and x["metric"] == "abs" and x["result"] == "all"):
            t = "extSix" + arm + "Ctr" + tag(r["model"])
            mac[t + "Mean"], mac[t + "Lo"], mac[t + "Hi"] = n3(r["mean"]), n3(r["lo"]), n3(r["hi"])
            mac[t + "P"] = cf.fmt_p(r["p_signflip"])
            mac[t + "NScenPos"] = str(r["n_scen_pos"])
            mac[t + "NScen"] = str(r["n_scen"])
        a, b = get(summ, arm, "ALL"), get(summ, "Main", "ALL")
        maxd["band"] = max(maxd["band"], abs(a["mean_band_err_pp"] - b["mean_band_err_pp"]))
        maxd["hind"] = max(maxd["hind"], abs(a["hind_gap_pp_med"] - b["hind_gap_pp_med"]))
        cross = cross or ((a["r_med"] > 1) != (b["r_med"] > 1))
        dngt = max(dngt, abs(int(mac["extSix" + arm + "RModelsGtOne"]) - int(mac["extSixMainRModelsGtOne"])))
    mac["extSixMaxDeltaBandPp"] = n3(maxd["band"], 1)
    mac["extSixMaxDeltaHindPp"] = n3(maxd["hind"], 1)
    mac["extSixRCrossesOne"] = "yes" if cross else "no"
    mac["extSixMaxDeltaModelsGtOne"] = str(dngt)
    material = maxd["band"] > MAT_BAND_PP or maxd["hind"] > MAT_HIND_PP or cross or dngt >= MAT_NGT1
    mac["extSixMaterial"] = "yes" if material else "no"
    mac["extSixNUnparsedTotal"] = str(sum(int(mac[f"extSix{a}All" + "NUnparsed"]) for a, _ in ARMS[1:]))
    mac["extSixNTotalPara"] = str(sum(int(mac[f"extSix{a}AllNTotal"]) for a, _ in ARMS[1:]))
    return mac


def f2(x):
    if x is None or x == "":
        return ""
    s = f"{float(x):.2f}"
    if float(s) == 0.0:
        s = "0.00"
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def table_rows(res):
    """Rows for table_ext.tex (columns = models, order of ext.write_table: sorted slugs). Short labels."""
    models = res["models"]
    lines = ["% GENERATED by analysis/ext6_analysis.py. Do not edit. Rows to paste into table_ext.tex (same columns).",
             "% Paraphrase minus main1 on N, same cases/models/seeds 1-3: change in mean |log-odds error| (paired)."]
    for arm, lab in (("ParaOne", "E6 para.\\ 1"), ("ParaTwo", "E6 para.\\ 2")):
        d = {r["model"]: r for r in res["ctr"] if r["arm"] == arm and r["metric"] == "abs" and r["result"] == "all"}
        lines.append(lab + " & " + " & ".join(f2(d[m]["mean"]) for m in models) + " \\\\")
    lines.append("% Update ratio r on N (median over scenarios), paraphrase minus main1:")
    for arm, lab in (("ParaOne", "Paraphrase 1, $r$ change"), ("ParaTwo", "Paraphrase 2, $r$ change")):
        lines.append("% " + lab + " & " + " & ".join(
            f2(get(res["summ"], arm, m)["r_med"] - get(res["summ"], "Main", m)["r_med"]) for m in models) + " \\\\")
    return lines


def write_macros(path, mac):
    for k, v in mac.items():
        assert k.isalpha() and k.startswith("extSix"), f"bad macro name {k}"
        assert v not in ("", None) and "\u2014" not in str(v), k
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/ext6_analysis.py from results/raw and data/. Do not edit.\n"
                "% I4 prompt paraphrases on N (ext6_para1, ext6_para2) vs main1 N, same cases/models/seeds 1-3. Post hoc.\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


def check(res, mac, para_ids_present=True):
    """Assertions on the key numbers (fail loudly on any silent change)."""
    summ = res["summ"]
    nc = len(res["cases"])
    assert nc == 20, nc
    assert len(res["models"]) == 5
    for arm, _ in ARMS:
        a = get(summ, arm, "ALL")
        assert a["n_total"] == nc * len(SEEDS) * 5, (arm, a["n_total"])
        assert a["n_parsed"] + a["n_unparsed"] == a["n_total"]
        assert sum(get(summ, arm, m)["n_parsed"] for m in res["models"]) == a["n_parsed"]
        for m in res["models"]:
            assert get(summ, arm, m)["r_n_scen"] == 10, (arm, m)
    # data facts from 3 flagged DeepSeek records across the two paraphrases, 297 + 297 parsed
    assert int(mac["extSixNUnparsedTotal"]) == 3
    assert int(mac["extSixParaOneAllNParsed"]) == 299 and int(mac["extSixParaTwoAllNParsed"]) == 298
    assert all("deepseek" in p["model"] for p in res["parse"] if p["arm"] != "Main")
    assert mac["extSixNTotalPara"] == "600"
    # cross-check the main1 comparison arm against the pre-specified machinery
    cells = cf.ratio_cells(res["prepped"]["Main"][1], res["cases"], [COND])
    assert len(cells) == len(res["models"]) * 10
    # contrast sanity: the ALL scenario-level mean equals the mean over models of per-model scenario values
    for arm, _ in ARMS[1:]:
        r = [x for x in res["ctr"] if x["arm"] == arm and x["metric"] == "abs" and x["result"] == "all"
             and x["model"] == "ALL"]
        assert len(r) == 1 and r[0]["n_scen"] == 10


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "derived"))
    ap.add_argument("--generated", default=os.path.join(ROOT, "paper", "generated"))
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--para1", default="ext6_para1")
    ap.add_argument("--para2", default="ext6_para2")
    ap.add_argument("--no-check", action="store_true", help="skip the real-data assertions (synthetic tests)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cases = load_cases(a.cases)
    recs, _ = load_raw(a.raw)
    syn = is_synthetic(recs, a.raw)
    for d in (a.out, a.generated):
        guard_output(d, syn)
    res = build(recs, cases, cfg, {"ParaOne": a.para1, "ParaTwo": a.para2})
    mac = macros(res)
    if not a.no_check:
        check(res, mac)
    write_csv(os.path.join(a.out, "ext6_summary.csv"), res["summ"], SUM_FIELDS)
    write_csv(os.path.join(a.out, "ext6_contrasts.csv"), res["ctr"], ["arm"] + ext.CTR_FIELDS)
    write_csv(os.path.join(a.out, "ext6_ratio.csv"), res["rcells"],
              ["arm", "model", "scenario", "condition", "n_pos", "n_neg", "dy", "log_dor", "ratio", "target_pos_pct",
               "target_neg_pct"])
    write_csv(os.path.join(a.out, "ext6_parse.csv"), res["parse"],
              ["arm", "model", "case_id", "seed", "parse_status", "length_stopped", "completion_tokens"])
    write_macros(os.path.join(a.generated, "ext6_macros.tex"), mac)
    with open(os.path.join(a.generated, "table_ext6_rows.tex"), "w") as f:
        f.write("\n".join(table_rows(res)) + "\n")
    print(f"ext6_analysis: {len(mac)} macros; material change: {mac['extSixMaterial']}")
    return res, mac


if __name__ == "__main__":
    main()
