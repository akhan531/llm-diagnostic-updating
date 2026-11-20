"""R1 extension analyses E1 to E5 (paid batches ext1_pretest, ext2_belief, ext3_long, ext4_unnamed, ext5_dsretry).

Post hoc, prompted by the simulated committee review. Reads ONLY results/raw/*.jsonl and data/.
Never writes to results/raw/. The pre-specified main1 analyses are untouched: main1 records are selected
exactly as score.py selects them (allow-list, latest attempt), and the ext run ids are separate run ids
(analysis/config.json `ext_run_ids`, never scored by score.py).

  E1 hindsight-free prior (ext1_pretest): pretest elicited from result-redacted notes (V, N, S; B is identical
     to N after redaction) vs the stated pretest after the positive and after the negative version in
     main1 (same condition); weighted-Bayes refit with the blind prior as regressor; blind-prior error.
  E2 believed accuracy (ext2_belief, N, prompt v3): believed sensitivity/specificity -> believed LR+/LR-;
     knowledge error (believed vs published LR) vs updating error (b against the model's own believed LR).
  E3 long chart (ext3_long, L, prompt v1) vs main1 N; split by result sign (the older same-test result is
     always negative-type); parse rate; does the answer follow the current or the older result.
  E4 test not named (ext4_unnamed, prompt v4): N-unnamed vs main1 N-named and L-unnamed vs ext3 L-named.
  E5 DeepSeek truncation retry (ext5_dsretry): sensitivity in which the retry draws fill the main1 keys whose
     final main1 record stopped at the token limit; main1 as pre-specified is unchanged.

A batch whose raw records are absent is skipped: its CSVs are not written (stale ones are removed), its
macros are not defined, and `\\extHave<E>` is 0. If no batch is present, ext_macros.tex and table_ext.tex are
removed, so a paper slot guarded by \\IfFileExists shows its fallback.

Usage: python analysis/ext_analysis.py [--raw GLOB] [--cases DIR] [--out DIR] [--generated DIR] [--n-boot 2000]
       [--main-run-id main1] [--ext1 ext1_pretest] [--ext2 ...] [--ext3 ...] [--ext4 ...] [--ext5 ...]
"""
import argparse
import glob
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import committee_fixes as cf  # noqa: E402
from common import (CONFIG_PATH, ROOT, clip, display_name, guard_output, is_calibration, is_synthetic,  # noqa: E402
                    load_cases, load_config, load_raw, logit_clipped, select_records, write_csv)
from fit_wbayes import fit_boot  # noqa: E402
from make_tables import model_tag, tex_num  # noqa: E402
from score import score_record  # noqa: E402

SEED = cf.SEED
BATCHES = ["E1", "E2", "E3", "E4", "E5"]
WORD = {"E1": "EOne", "E2": "ETwo", "E3": "EThree", "E4": "EFour", "E5": "EFive"}
DEFAULT_IDS = {"E1": "ext1_pretest", "E2": "ext2_belief", "E3": "ext3_long", "E4": "ext4_unnamed",
               "E5": "ext5_dsretry"}
BLIND_CONDS = ["V", "N", "S"]           # ext1 conditions; B redacted == N redacted
BLIND_COND_OF = {"V": "V", "N": "N", "B": "N", "S": "S"}
CSV_FILES = {
    "E1": ["ext_e1_cells.csv", "ext_e1_summary.csv", "ext_e1_fits.csv"],
    "E2": ["ext_e2_scenarios.csv", "ext_e2_summary.csv", "ext_e2_fits.csv"],
    "E3": ["ext_e3_contrasts.csv", "ext_e3_parse.csv", "ext_e3_tracking.csv"],
    "E4": ["ext_e4_contrasts.csv", "ext_e4_parse.csv", "ext_e4_tracking.csv"],
    "E5": ["ext_e5_retry.csv", "ext_e5_headline.csv"],
}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def lod(p):
    return logit_clipped(p)


def tag(m):
    return "All" if m == "ALL" else model_tag(m)


def num(x, d=2):
    return "n/a" if x is None else tex_num(x, d)


# ---------------------------------------------------------------- loading
def present_ids(recs):
    return {r.get("run_id") for r in recs}


def final_of(recs, run_id, cfg):
    final, _ = select_records(recs, [run_id], cfg, quiet=True)
    return final


def v1_rows(final, cases):
    """Parse-ok, non-calibration rows (prompt v1 or v4 answers) with pre/post/prior/lr/target, as cf.load_rows."""
    rows = []
    for r in final:
        c = cases[r["case_id"]]
        if is_calibration(c):
            continue
        s = score_record(r, c)
        if not s["parse_ok"]:
            continue
        t = c["target"]
        s.update(pre=s["pretest_stated"] / 100.0, post=s["posttest_stated"] / 100.0,
                 prior=t["pretest_point"], lr=t["lr_point"], target=t["posttest_point"])
        rows.append(s)
    return rows


def scenario_targets(cases):
    """{scenario: {"pos": target dict, "neg": target dict, "set": ..}} for non-calibration cases."""
    out = {}
    for c in cases.values():
        if not is_calibration(c):
            out.setdefault(c["scenario"], {})[c["result"]] = c["target"]
    return out


def parse_ok(r, fields):
    pa = r.get("parsed_answer")
    return r.get("parse_status") == "ok" and isinstance(pa, dict) and all(pa.get(f) is not None for f in fields)


# ---------------------------------------------------------------- E1 hindsight-free prior
def blind_cells(ext1_final, scen_t):
    """{(model, scenario, cond): [blind pretest probabilities]} over parse-ok ext1 answers."""
    g, n_total, n_ok = {}, 0, 0
    for r in ext1_final:
        if r["case_id"] not in scen_t:   # calibration or unknown scenario: not analysed
            continue
        n_total += 1
        if not parse_ok(r, ["pretest"]):
            continue
        n_ok += 1
        g.setdefault((r["model"], r["case_id"], r["condition"]), []).append(float(r["parsed_answer"]["pretest"]) / 100.0)
    return g, n_total, n_ok


def length_stopped(r):
    """True if the reply used its whole max_tokens budget (same rule as parse_rates)."""
    return bool(r.get("completion_tokens") is not None and (r.get("decoding_params") or {}).get("max_tokens") and
                float(r["completion_tokens"]) >= float(r["decoding_params"]["max_tokens"]))


def e1(ext1_final, main_rows, cases, n_boot, mac):
    scen_t = scenario_targets(cases)
    bl, n_total, n_ok = blind_cells(ext1_final, scen_t)
    cells = []
    for pooled in (False, True):
        g = {}
        for (m, s, c), ps in bl.items():
            k = (m, s, "VNS" if pooled else c)
            g.setdefault(k, {"blind": [], "pos": [], "neg": []})["blind"].extend(ps)
        for r in main_rows:
            if r["condition"] not in BLIND_CONDS:
                continue
            k = (r["model"], r["scenario"], "VNS" if pooled else r["condition"])
            if k in g:
                g[k][r["result"]].append(r["pre"])
        for (m, s, c), d in sorted(g.items()):
            if not d["blind"] or not d["pos"] or not d["neg"]:
                continue
            bm, pm, nm = (mean(lod(x) for x in d[k]) for k in ("blind", "pos", "neg"))
            pub = scen_t[s]["pos"]["pretest_point"]
            cells.append({"model": m, "scenario": s, "condition": c, "n_blind": len(d["blind"]),
                          "n_pos": len(d["pos"]), "n_neg": len(d["neg"]),
                          "blind_pct": 100 * mean(d["blind"]), "pre_pos_pct": 100 * mean(d["pos"]),
                          "pre_neg_pct": 100 * mean(d["neg"]), "published_prior_pct": 100 * pub,
                          "gap_pos_pp": 100 * (mean(d["pos"]) - mean(d["blind"])),
                          "gap_neg_pp": 100 * (mean(d["neg"]) - mean(d["blind"])),
                          "gap_posneg_pp": 100 * (mean(d["pos"]) - mean(d["neg"])),
                          "gap_pos_lod": pm - bm, "gap_neg_lod": nm - bm, "gap_posneg_lod": pm - nm,
                          "blind_prior_err_lod": bm - lod(pub),
                          "stated_prior_err_lod": mean(lod(x) for x in d["pos"] + d["neg"]) - lod(pub)})
    pooled = [c for c in cells if c["condition"] == "VNS"]
    models = sorted({c["model"] for c in pooled})
    summ = []
    for m in models + ["ALL"]:
        if m == "ALL":
            by_s = {}
            for c in pooled:
                by_s.setdefault(c["scenario"], []).append(c)
            cs = [{k: mean(x[k] for x in v) for k in v[0] if isinstance(v[0][k], float)} for v in by_s.values()]
        else:
            cs = [c for c in pooled if c["model"] == m]
        if not cs:
            continue
        row = {"model": m, "n_scen": len(cs)}
        for k in ("gap_pos", "gap_neg", "gap_posneg"):
            v = [c[k + "_lod"] for c in cs]
            lo, hi = cf.signflip_ci(v)
            row.update({k + "_lod_mean": mean(v), k + "_lod_lo": lo, k + "_lod_hi": hi,
                        k + "_lod_p": cf.signflip_p(v), k + "_npos": sum(x > 0 for x in v),
                        k + "_pp_med": statistics.median(c[k + "_pp"] for c in cs)})
        row["blind_err_lod_mean"] = mean(c["blind_prior_err_lod"] for c in cs)
        row["blind_err_abs_lod_mean"] = mean(abs(c["blind_prior_err_lod"]) for c in cs)
        row["stated_err_lod_mean"] = mean(c["stated_prior_err_lod"] for c in cs)
        row["stated_err_abs_lod_mean"] = mean(abs(c["stated_prior_err_lod"]) for c in cs)
        tot = abs(row["gap_pos_lod_mean"]) + abs(row["gap_neg_lod_mean"])
        row["share_from_pos"] = abs(row["gap_pos_lod_mean"]) / tot if tot else None
        summ.append(row)
    # refit: blind prior as regressor (model x scenario x condition mean logit; B uses the N blind prior)
    bl_lod = {k: mean(lod(x) for x in v) for k, v in bl.items()}
    fits = []
    for m in models:
        rs = [r for r in main_rows if r["model"] == m and r["condition"] in BLIND_COND_OF]
        cl_b, cl_o, n_drop = {}, {}, 0
        for r in rs:
            k = (m, r["scenario"], BLIND_COND_OF[r["condition"]])
            if k not in bl_lod:
                n_drop += 1
                continue
            y, xl = lod(r["post"]), math.log(r["lr"])
            cl_b.setdefault(r["scenario"], []).append((bl_lod[k], xl, y))
            cl_o.setdefault(r["scenario"], []).append((lod(r["pre"]), xl, y))
        kept = [r for r in rs if (m, r["scenario"], BLIND_COND_OF[r["condition"]]) in bl_lod]
        for variant, cl in (("blind_prior", cl_b), ("own_stated_prior", cl_o), ("published_prior", None)):
            if variant == "published_prior":
                f = cf.fit_cell(kept, n_boot, f"ext1-pub-{m}")
            else:
                f = fit_boot(cl, n_boot, random.Random(f"{SEED}-ext1-{variant}-{m}")) if len(cl) >= 2 else None
            if f is None:
                continue
            fits.append(dict(f, model=m, variant=variant, conditions="VNBS", n_dropped_no_blind=n_drop))
    # macros
    mac["extEOneNTotal"] = str(n_total)
    mac["extEOneNParsed"] = str(n_ok)
    mac["extEOneNTrunc"] = str(sum(length_stopped(r) for r in ext1_final if r["case_id"] in scen_t))
    mac["extEOneParsePct"] = f"{100 * n_ok / n_total:.1f}" if n_total else "n/a"
    for s in summ:
        t = "extEOne" + tag(s["model"])
        for k, w in (("gap_pos", "GapPos"), ("gap_neg", "GapNeg"), ("gap_posneg", "GapPosNeg")):
            mac[t + w + "Lod"] = num(s[k + "_lod_mean"])
            mac[t + w + "Lo"] = num(s[k + "_lod_lo"])
            mac[t + w + "Hi"] = num(s[k + "_lod_hi"])
            mac[t + w + "P"] = cf.fmt_p(s[k + "_lod_p"])
            mac[t + w + "NPos"] = str(s[k + "_npos"])
            mac[t + w + "MedPp"] = num(s[k + "_pp_med"], 1)
        mac[t + "NScen"] = str(s["n_scen"])
        mac[t + "BlindErrLod"] = num(s["blind_err_lod_mean"])
        mac[t + "BlindAbsErrLod"] = num(s["blind_err_abs_lod_mean"])
        mac[t + "StatedErrLod"] = num(s["stated_err_lod_mean"])
        mac[t + "StatedAbsErrLod"] = num(s["stated_err_abs_lod_mean"])
    for f in fits:
        t = "extEOne" + tag(f["model"]) + {"blind_prior": "Blind", "own_stated_prior": "Own",
                                           "published_prior": "Pub"}[f["variant"]]
        for p in "ab":
            mac[t + "Par" + p.upper() + "Pt"] = num(f[p])
            mac[t + "Par" + p.upper() + "Lo"] = num(f.get(p + "_lo"))
            mac[t + "Par" + p.upper() + "Hi"] = num(f.get(p + "_hi"))
    # gap_neg = stated pretest after a negative result minus blind pretest (log-odds); hindsight predicts < 0
    per_model = [s for s in summ if s["model"] != "ALL"]
    mac["extEOneNModelsNegGapExpected"] = str(sum(s["gap_neg_lod_mean"] < 0 for s in per_model))
    mac["extEOneNModelsNegGap"] = str(len(per_model))
    bf = [f for f in fits if f["variant"] == "blind_prior"]
    if bf:
        mac["extEOneBlindParBMin"] = num(min(f["b"] for f in bf))
        mac["extEOneBlindParBMax"] = num(max(f["b"] for f in bf))
        mac["extEOneBlindParACiBelowOneN"] = str(sum(f.get("a_hi") is not None and f["a_hi"] < 1 for f in bf))
        mac["extEOneBlindParBCiAboveOneN"] = str(sum(f.get("b_lo") is not None and f["b_lo"] > 1 for f in bf))
    mac["extEOneRatioUnaffected"] = ("the within-scenario ratio uses only stated posttests and published likelihood "
                                     "ratios, so it does not depend on which prior is used")
    return cells, summ, fits


E1_CELL_FIELDS = ["model", "scenario", "condition", "n_blind", "n_pos", "n_neg", "blind_pct", "pre_pos_pct",
                  "pre_neg_pct", "published_prior_pct", "gap_pos_pp", "gap_neg_pp", "gap_posneg_pp", "gap_pos_lod",
                  "gap_neg_lod", "gap_posneg_lod", "blind_prior_err_lod", "stated_prior_err_lod"]
E1_SUM_FIELDS = ["model", "n_scen"] + [k + s for k in ("gap_pos", "gap_neg", "gap_posneg")
                                       for s in ("_lod_mean", "_lod_lo", "_lod_hi", "_lod_p", "_npos", "_pp_med")] + \
    ["blind_err_lod_mean", "blind_err_abs_lod_mean", "stated_err_lod_mean", "stated_err_abs_lod_mean",
     "share_from_pos"]
FIT_FIELDS = ["model", "variant", "conditions", "n_obs", "n_clusters", "a", "a_lo", "a_hi", "b", "b_lo", "b_hi", "c",
              "n_boot_valid", "n_dropped_no_blind"]


# ---------------------------------------------------------------- E2 believed accuracy
def believed_lrs(sens_pct, spec_pct):
    """(LR+, LR-) from believed sensitivity and specificity in percent; both clipped as in the main analysis."""
    se, sp = clip(sens_pct / 100.0), clip(spec_pct / 100.0)
    return se / (1.0 - sp), (1.0 - se) / sp


def e2_rows(ext2_final, cases):
    rows, n_total = [], 0
    scen_t = scenario_targets(cases)
    for r in ext2_final:
        c = cases[r["case_id"]]
        if is_calibration(c):
            continue
        n_total += 1
        if not parse_ok(r, ["sensitivity", "specificity", "pretest", "posttest"]):
            continue
        pa = r["parsed_answer"]
        lp, ln = believed_lrs(float(pa["sensitivity"]), float(pa["specificity"]))
        t = scen_t[c["scenario"]]
        res = c["result"]
        rows.append({"model": r["model"], "case_id": r["case_id"], "scenario": c["scenario"], "result": res,
                     "seed": r.get("seed"), "sens": float(pa["sensitivity"]), "spec": float(pa["specificity"]),
                     "pre": float(pa["pretest"]) / 100.0, "post": float(pa["posttest"]) / 100.0,
                     "lr_pos_bel": lp, "lr_neg_bel": ln, "lr_bel": lp if res == "pos" else ln,
                     "lr_pos_pub": t["pos"]["lr_point"], "lr_neg_pub": t["neg"]["lr_point"],
                     "lr": c["target"]["lr_point"], "prior": c["target"]["pretest_point"]})
    return rows, n_total


def e2(ext2_final, cases, n_boot, mac):
    rows, n_total = e2_rows(ext2_final, cases)
    for r in rows:
        sgn = 1 if r["result"] == "pos" else -1
        r["steep_excess"] = sgn * (math.log(r["lr_bel"]) - math.log(r["lr"]))   # > 0: believed steeper
        r["implied_log_lr"] = lod(r["post"]) - lod(r["pre"])
        r["coh_err"] = r["implied_log_lr"] - math.log(r["lr_bel"])               # vs own believed LR
        r["log_dor_bel"] = math.log(r["lr_pos_bel"]) - math.log(r["lr_neg_bel"])
        r["log_dor_pub"] = math.log(r["lr_pos_pub"]) - math.log(r["lr_neg_pub"])
    scen = []
    g = {}
    for r in rows:
        g.setdefault((r["model"], r["scenario"]), []).append(r)
    for (m, s), rs in sorted(g.items()):
        pos = [lod(r["post"]) for r in rs if r["result"] == "pos"]
        neg = [lod(r["post"]) for r in rs if r["result"] == "neg"]
        dx_bel = mean(r["log_dor_bel"] for r in rs)
        dx_pub = rs[0]["log_dor_pub"]
        dy = mean(pos) - mean(neg) if pos and neg else None
        scen.append({"model": m, "scenario": s, "n": len(rs), "sens_med": statistics.median(r["sens"] for r in rs),
                     "spec_med": statistics.median(r["spec"] for r in rs),
                     "lr_pos_bel_med": statistics.median(r["lr_pos_bel"] for r in rs),
                     "lr_neg_bel_med": statistics.median(r["lr_neg_bel"] for r in rs),
                     "lr_pos_pub": rs[0]["lr_pos_pub"], "lr_neg_pub": rs[0]["lr_neg_pub"],
                     "log_dor_bel_mean": dx_bel, "log_dor_pub": dx_pub, "log_dor_excess": dx_bel - dx_pub,
                     "log_dor_ratio": dx_bel / dx_pub,
                     "dy": dy, "ratio_own": dy / dx_bel if dy is not None and abs(dx_bel) > 1e-9 else None,
                     "ratio_pub": dy / dx_pub if dy is not None else None})
    models = sorted({r["model"] for r in rows})
    summ, fits = [], []
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        sc = [s for s in scen if s["model"] == m]
        ex = [s["log_dor_excess"] for s in sc]
        lo, hi = cf.signflip_ci(ex) if ex else (None, None)
        cl_own, cl_pub = {}, {}
        for r in rs:
            cl_own.setdefault(r["scenario"], []).append((lod(r["pre"]), math.log(r["lr_bel"]), lod(r["post"])))
            cl_pub.setdefault(r["scenario"], []).append((lod(r["pre"]), math.log(r["lr"]), lod(r["post"])))
        fo = fit_boot(cl_own, n_boot, random.Random(f"{SEED}-ext2-own-{m}")) if len(cl_own) >= 2 else None
        fp = fit_boot(cl_pub, n_boot, random.Random(f"{SEED}-ext2-pub-{m}")) if len(cl_pub) >= 2 else None
        for v, f in (("own_prior_believed_lr", fo), ("own_prior_published_lr", fp)):
            if f is not None:
                fits.append(dict(f, model=m, variant=v, conditions="N"))
        ro = [s["ratio_own"] for s in sc if s["ratio_own"] is not None]
        rp = [s["ratio_pub"] for s in sc if s["ratio_pub"] is not None]
        knowledge = lo is not None and lo > 0
        updating = fo is not None and fo.get("b_lo") is not None and fo["b_lo"] > 1
        verdict = {(True, True): "both", (True, False): "knowledge", (False, True): "updating",
                   (False, False): "neither"}[(knowledge, updating)]
        summ.append({"model": m, "n_total": sum(1 for r in ext2_final if r["model"] == m and
                                                 not is_calibration(cases[r["case_id"]])),
                     "n_ok": len(rs), "n_scen": len(sc),
                     "sens_med": statistics.median(r["sens"] for r in rs),
                     "spec_med": statistics.median(r["spec"] for r in rs),
                     "steep_excess_med": statistics.median(r["steep_excess"] for r in rs),
                     "pct_steeper": 100 * sum(r["steep_excess"] > 0 for r in rs) / len(rs),
                     "log_dor_excess_mean": mean(ex), "log_dor_excess_lo": lo, "log_dor_excess_hi": hi,
                     "log_dor_excess_p": cf.signflip_p(ex) if ex else None,
                     "n_scen_steeper": sum(x > 0 for x in ex),
                     "log_dor_ratio_med": statistics.median(s["log_dor_ratio"] for s in sc),
                     "coh_err_med": statistics.median(r["coh_err"] for r in rs),
                     "abs_coh_err_med": statistics.median(abs(r["coh_err"]) for r in rs),
                     "b_own": fo["b"] if fo else None, "b_own_lo": fo.get("b_lo") if fo else None,
                     "b_own_hi": fo.get("b_hi") if fo else None,
                     "b_pub": fp["b"] if fp else None, "b_pub_lo": fp.get("b_lo") if fp else None,
                     "b_pub_hi": fp.get("b_hi") if fp else None,
                     "ratio_own_med": statistics.median(ro) if ro else None,
                     "ratio_pub_med": statistics.median(rp) if rp else None,
                     "n_ratio_own_gt1": sum(x > 1 for x in ro), "n_ratio": len(ro),
                     "knowledge_error": int(knowledge), "updating_error": int(updating), "verdict": verdict})
    mac["extETwoNTotal"] = str(n_total)
    mac["extETwoNParsed"] = str(len(rows))
    mac["extETwoNTrunc"] = str(sum(length_stopped(r) for r in ext2_final if not is_calibration(cases[r["case_id"]])))
    mac["extETwoParsePct"] = f"{100 * len(rows) / n_total:.1f}" if n_total else "n/a"
    for s in summ:
        t = "extETwo" + tag(s["model"])
        mac[t + "SensMed"] = num(s["sens_med"], 1)
        mac[t + "SpecMed"] = num(s["spec_med"], 1)
        mac[t + "SteepMed"] = num(s["steep_excess_med"])
        mac[t + "PctSteeper"] = num(s["pct_steeper"], 1)
        mac[t + "DorExcess"] = num(s["log_dor_excess_mean"])
        mac[t + "DorExcessLo"] = num(s["log_dor_excess_lo"])
        mac[t + "DorExcessHi"] = num(s["log_dor_excess_hi"])
        mac[t + "DorExcessP"] = cf.fmt_p(s["log_dor_excess_p"])
        mac[t + "NScenSteeper"] = str(s["n_scen_steeper"])
        mac[t + "DorRatioMed"] = num(s["log_dor_ratio_med"])
        mac[t + "NScen"] = str(s["n_scen"])
        mac[t + "CohErrMed"] = num(s["coh_err_med"])
        mac[t + "AbsCohErrMed"] = num(s["abs_coh_err_med"])
        for k, w in (("b_own", "BOwn"), ("b_pub", "BPub")):
            mac[t + w + "Pt"] = num(s[k])
            mac[t + w + "Lo"] = num(s[k + "_lo"])
            mac[t + w + "Hi"] = num(s[k + "_hi"])
        mac[t + "RatioOwnMed"] = num(s["ratio_own_med"])
        mac[t + "RatioPubMed"] = num(s["ratio_pub_med"])
        mac[t + "Verdict"] = s["verdict"]
    rpm = [s for s in summ if s["ratio_pub_med"] is not None]
    if rpm:  # min/max across models of the per-model median ratio vs published LR (replaces hand aliases in main.tex)
        lo_s = min(rpm, key=lambda s: s["ratio_pub_med"])
        hi_s = max(rpm, key=lambda s: s["ratio_pub_med"])
        mac["extETwoPubRatioMedMin"] = num(lo_s["ratio_pub_med"])
        mac["extETwoPubRatioMedMax"] = num(hi_s["ratio_pub_med"])
        assert lo_s["ratio_pub_med"] <= hi_s["ratio_pub_med"]
        assert mac["extETwoPubRatioMedMin"] == mac["extETwo" + tag(lo_s["model"]) + "RatioPubMed"]
        assert mac["extETwoPubRatioMedMax"] == mac["extETwo" + tag(hi_s["model"]) + "RatioPubMed"]
    if summ:
        mac["extETwoNKnowledge"] = str(sum(s["knowledge_error"] for s in summ))
        mac["extETwoNUpdating"] = str(sum(s["updating_error"] for s in summ))
        mac["extETwoNModels"] = str(len(summ))
        bo = [s["b_own"] for s in summ if s["b_own"] is not None]
        if bo:
            mac["extETwoBOwnMin"] = num(min(bo))
            mac["extETwoBOwnMax"] = num(max(bo))
    return scen, summ, fits


E2_SCEN_FIELDS = ["model", "scenario", "n", "sens_med", "spec_med", "lr_pos_bel_med", "lr_neg_bel_med", "lr_pos_pub",
                  "lr_neg_pub", "log_dor_bel_mean", "log_dor_pub", "log_dor_excess", "log_dor_ratio", "dy", "ratio_own", "ratio_pub"]
E2_SUM_FIELDS = ["model", "n_total", "n_ok", "n_scen", "sens_med", "spec_med", "steep_excess_med", "pct_steeper",
                 "log_dor_excess_mean", "log_dor_excess_lo", "log_dor_excess_hi", "log_dor_excess_p", "n_scen_steeper",
                 "log_dor_ratio_med",
                 "coh_err_med", "abs_coh_err_med", "b_own", "b_own_lo", "b_own_hi", "b_pub", "b_pub_lo", "b_pub_hi",
                 "ratio_own_med", "ratio_pub_med", "n_ratio_own_gt1", "n_ratio", "knowledge_error", "updating_error",
                 "verdict"]
E2_FIT_FIELDS = ["model", "variant", "conditions", "n_obs", "n_clusters", "a", "a_lo", "a_hi", "b", "b_lo", "b_hi", "c",
                 "n_boot_valid"]


# ---------------------------------------------------------------- E3 / E4 paired arm contrasts
def arm_value(r, metric):
    e = lod(r["post"]) - lod(r["target"])
    return {"abs": abs(e), "signed": e, "pre": lod(r["pre"]) - lod(r["prior"])}[metric]


def paired_scen(rows_x, rows_y, metric, result=None):
    """{model: {scenario: d}}: per case, seed-paired X minus Y difference averaged over seeds; per scenario the
    mean over its positive and negative case (or only `result`). Same statistic as F6 (cf.scenario_contrasts)."""
    vy = {(r["model"], r["case_id"], r["seed"]): arm_value(r, metric) for r in rows_y
          if result is None or r["result"] == result}
    case_d = {}
    for r in rows_x:
        if result is not None and r["result"] != result:
            continue
        k = (r["model"], r["case_id"], r["seed"])
        if k in vy:
            case_d.setdefault((r["model"], r["scenario"], r["case_id"]), []).append(arm_value(r, metric) - vy[k])
    sd = {}
    for (m, s, _cid), ds in case_d.items():
        sd.setdefault(m, {}).setdefault(s, []).append(mean(ds))
    return {m: {s: mean(v) for s, v in d.items()} for m, d in sd.items()}


def contrast_rows(name, rows_x, rows_y):
    out = []
    for metric in ("abs", "signed", "pre"):
        for res in (None, "pos", "neg"):
            sc = paired_scen(rows_x, rows_y, metric, res)
            allm = {}
            for m in sc:
                for s, v in sc[m].items():
                    allm.setdefault(s, []).append(v)
            for m, d in [(m, sc[m]) for m in sorted(sc)] + ([("ALL", {s: mean(v) for s, v in allm.items()})]
                                                            if allm else []):
                vals = [d[s] for s in sorted(d)]
                lo, hi = cf.signflip_ci(vals) if vals else (None, None)
                out.append({"contrast": name, "metric": metric, "result": res or "all", "model": m,
                            "n_scen": len(vals), "mean": mean(vals), "lo": lo, "hi": hi,
                            "p_signflip": cf.signflip_p(vals) if vals else None,
                            "n_scen_pos": sum(v > 0 for v in vals)})
    return out


def parse_rates(arms, cases):
    """arms: [(arm label, final records)] -> per model x arm parse counts (calibration cases skipped)."""
    out = []
    for arm, final in arms:
        for m in sorted({r["model"] for r in final}):
            rs = [r for r in final if r["model"] == m and not is_calibration(cases[r["case_id"]])]
            ok = sum(parse_ok(r, ["pretest", "posttest"]) for r in rs)
            trunc = sum(1 for r in rs if r.get("completion_tokens") is not None and
                        (r.get("decoding_params") or {}).get("max_tokens") and
                        float(r["completion_tokens"]) >= float(r["decoding_params"]["max_tokens"]))
            out.append({"arm": arm, "model": m, "n_total": len(rs), "n_ok": ok,
                        "pct_ok": 100 * ok / len(rs) if rs else None, "n_length_stopped": trunc})
    return out


def tracking(arm, rows, ref_rows):
    """Does the answer follow the CURRENT result? Two indicators per model x result:
    direction: stated posttest moves from the stated pretest in the direction of the current result;
    nearer_current: logit posttest nearer to the model's own main1 N mean for the shown result than to its
    mean for the opposite result (same scenario). For positive cases the older result is opposite (the trap)."""
    ref = {}
    for r in ref_rows:
        ref.setdefault((r["model"], r["scenario"], r["result"]), []).append(lod(r["post"]))
    ref = {k: mean(v) for k, v in ref.items()}
    out = []
    for m in sorted({r["model"] for r in rows}) + ["ALL"]:
        for res in ("pos", "neg"):
            rs = [r for r in rows if r["result"] == res and (m == "ALL" or r["model"] == m)]
            if not rs:
                continue
            opp = "neg" if res == "pos" else "pos"
            direc = [(r["post"] > r["pre"]) if res == "pos" else (r["post"] < r["pre"]) for r in rs]
            near = [abs(lod(r["post"]) - ref[(r["model"], r["scenario"], res)]) <
                    abs(lod(r["post"]) - ref[(r["model"], r["scenario"], opp)])
                    for r in rs if (r["model"], r["scenario"], res) in ref and (r["model"], r["scenario"], opp) in ref]
            out.append({"arm": arm, "model": m, "result": res, "n": len(rs),
                        "pct_direction_current": 100 * sum(direc) / len(rs),
                        "n_ref": len(near), "pct_nearer_current": 100 * sum(near) / len(near) if near else None})
    return out


def ratio_med(rows, cases, cond):
    cells = cf.ratio_cells(rows, cases, [cond])
    out = {}
    for m in sorted({c["model"] for c in cells}):
        rs = [c["ratio"] for c in cells if c["model"] == m]
        out[m] = statistics.median(rs)
    return out


CTR_FIELDS = ["contrast", "metric", "result", "model", "n_scen", "mean", "lo", "hi", "p_signflip", "n_scen_pos"]
PARSE_FIELDS = ["arm", "model", "n_total", "n_ok", "pct_ok", "n_length_stopped"]
TRACK_FIELDS = ["arm", "model", "result", "n", "pct_direction_current", "n_ref", "pct_nearer_current", "ratio_med",
                "ratio_med_ref"]

CWORD = {"LvsN": "LvsN", "NunVsN": "NUnnamedVsNamed", "LunVsL": "LUnnamedVsNamed", "LunVsN": "LUnnamedVsNNamed"}
MWORD = {"abs": "Abs", "signed": "Signed", "pre": "Pre"}
RWORD = {"all": "All", "pos": "Pos", "neg": "Neg"}


def contrast_macros(prefix, ctr, mac):
    for r in ctr:
        t = prefix + CWORD[r["contrast"]] + MWORD[r["metric"]] + RWORD[r["result"]] + tag(r["model"])
        mac[t + "Mean"] = num(r["mean"])
        mac[t + "Lo"] = num(r["lo"])
        mac[t + "Hi"] = num(r["hi"])
        mac[t + "P"] = cf.fmt_p(r["p_signflip"])
        mac[t + "N"] = str(r["n_scen"])
        mac[t + "NPos"] = str(r["n_scen_pos"])
    for c in sorted({r["contrast"] for r in ctr}):
        ms = [r for r in ctr if r["contrast"] == c and r["model"] != "ALL" and r["metric"] == "abs"
              and r["result"] == "all" and r["lo"] is not None]
        mac[prefix + CWORD[c] + "AbsNExclZero"] = str(sum(r["lo"] > 0 or r["hi"] < 0 for r in ms))
        if ms:  # range across models of the per-model mean change in absolute log-odds error
            lo_r = min(ms, key=lambda r: r["mean"])
            hi_r = max(ms, key=lambda r: r["mean"])
            k = prefix + CWORD[c] + "AbsAll"
            mac[k + "ModelMeanMin"] = num(lo_r["mean"])
            mac[k + "ModelMeanMax"] = num(hi_r["mean"])
            assert lo_r["mean"] <= hi_r["mean"]
            assert mac[k + "ModelMeanMin"] == mac[k + tag(lo_r["model"]) + "Mean"]
            assert mac[k + "ModelMeanMax"] == mac[k + tag(hi_r["model"]) + "Mean"]


def parse_macros(prefix, pr, mac):
    for arm in sorted({p["arm"] for p in pr}):
        ps = [p for p in pr if p["arm"] == arm]
        for p in ps:
            mac[prefix + arm + tag(p["model"]) + "ParsePct"] = num(p["pct_ok"], 1)
            mac[prefix + arm + tag(p["model"]) + "NTrunc"] = str(p["n_length_stopped"])
        tot, ok = sum(p["n_total"] for p in ps), sum(p["n_ok"] for p in ps)
        mac[prefix + arm + "ParsePct"] = f"{100 * ok / tot:.1f}" if tot else "n/a"
        mac[prefix + arm + "NTotal"] = str(tot)
        mac[prefix + arm + "NParsed"] = str(ok)


def track_macros(prefix, tr, mac):
    for t in tr:
        k = prefix + t["arm"] + tag(t["model"]) + RWORD[t["result"]]
        mac[k + "PctDirCurrent"] = num(t["pct_direction_current"], 1)
        mac[k + "PctNearCurrent"] = num(t["pct_nearer_current"], 1)
        if t.get("ratio_med") is not None:
            mac[prefix + t["arm"] + tag(t["model"]) + "RatioMed"] = num(t["ratio_med"])


def add_ratio(tr, rows, ref_rows, cases, cond, ref_cond):
    rm, rr = ratio_med(rows, cases, cond), ratio_med(ref_rows, cases, ref_cond)
    for t in tr:
        if t["result"] == "pos" and t["model"] != "ALL":
            t["ratio_med"], t["ratio_med_ref"] = rm.get(t["model"]), rr.get(t["model"])
    return tr


def e3(ext3_final, main_final, main_rows, cases, mac):
    l_rows = v1_rows(ext3_final, cases)
    n_rows = [r for r in main_rows if r["condition"] == "N"]
    ctr = contrast_rows("LvsN", l_rows, n_rows)
    pr = parse_rates([("L", ext3_final), ("N", [r for r in main_final if r["condition"] == "N"])], cases)
    tr = add_ratio(tracking("L", l_rows, n_rows), l_rows, n_rows, cases, "L", "N") + \
        add_ratio(tracking("N", n_rows, n_rows), n_rows, n_rows, cases, "N", "N")
    contrast_macros("extEThree", ctr, mac)
    parse_macros("extEThree", pr, mac)
    track_macros("extEThree", tr, mac)
    return ctr, pr, tr


def e4(ext4_final, ext3_final, main_final, main_rows, cases, mac):
    u_rows = v1_rows(ext4_final, cases)
    un = [r for r in u_rows if r["condition"] == "N"]
    ul = [r for r in u_rows if r["condition"] == "L"]
    n_rows = [r for r in main_rows if r["condition"] == "N"]
    ctr, arms, tr = [], [], []
    if un:
        ctr += contrast_rows("NunVsN", un, n_rows)
        arms.append(("NUnnamed", [r for r in ext4_final if r["condition"] == "N"]))
        tr += add_ratio(tracking("NUnnamed", un, n_rows), un, n_rows, cases, "N", "N")
    if ul and ext3_final:
        l_rows = v1_rows(ext3_final, cases)
        ctr += contrast_rows("LunVsL", ul, l_rows)
    if ul:
        ctr += contrast_rows("LunVsN", ul, n_rows)   # long chart and no test name together, vs the main1 N note
        arms.append(("LUnnamed", [r for r in ext4_final if r["condition"] == "L"]))
        t = tracking("LUnnamed", ul, n_rows)
        add_ratio(t, ul, n_rows, cases, "L", "N")
        tr += t
    pr = parse_rates(arms, cases)
    contrast_macros("extEFour", ctr, mac)
    parse_macros("extEFour", pr, mac)
    track_macros("extEFour", tr, mac)
    mac["extEFourHaveN"] = "1" if un else "0"
    mac["extEFourHaveLContrast"] = "1" if (ul and ext3_final) else "0"
    return ctr, pr, tr


# ---------------------------------------------------------------- E5 DeepSeek truncation retry (sensitivity)
def stopped_at_limit(r):
    mx = (r.get("decoding_params") or {}).get("max_tokens")
    ct = r.get("completion_tokens")
    return bool(mx) and ct is not None and float(ct) >= float(mx)


def filled_records(main_final, ext5_final):
    """main1 final records with each key that has an ext5 record replaced by the ext5 latest attempt.
    Only main1 keys whose final record is not parse-ok are eligible (the retry spec lists exactly those)."""
    retry = {(r["case_id"], r["condition"], r["model"], r.get("seed")): r for r in ext5_final}
    out, n_rep = [], 0
    for r in main_final:
        k = (r["case_id"], r["condition"], r["model"], r.get("seed"))
        if k in retry:
            assert r.get("parse_status") != "ok", f"ext5 retries a main1 key that parsed: {k}"
            out.append(retry[k])
            n_rep += 1
        else:
            out.append(r)
    return out, n_rep


def headline(rows, cases, n_boot):
    over = {o["model"]: o for o in cf.overestimation(rows)}
    rsum = {s["model"]: s for s in cf.ratio_summary(cf.ratio_cells(rows, cases, cf.FIT_CONDS))}
    out = {}
    for m in sorted(over):
        f = cf.fit_cell([r for r in rows if r["model"] == m and r["condition"] in cf.FIT_CONDS], n_boot, f"pooled-{m}")
        out[m] = {"n_parsed": sum(1 for r in rows if r["model"] == m), "over_med_lod": over[m]["median_signed_lod"],
                  "over_pct_above": over[m]["pct_above_target"], "ratio_med": rsum[m]["median"],
                  "ratio_n_gt1": rsum[m]["n_gt1"], "b_fe": rsum[m]["b_fe"], "b": f["b"], "b_lo": f.get("b_lo"),
                  "b_hi": f.get("b_hi"), "a": f["a"]}
    return out


def e5(ext5_final, main_final, main_rows, cases, n_boot, mac):
    keys = {(r["case_id"], r["condition"], r["model"], r.get("seed")) for r in ext5_final}
    main_keys = {(r["case_id"], r["condition"], r["model"], r.get("seed")): r for r in main_final}
    retry = []
    for r in sorted(ext5_final, key=lambda r: (r["model"], r["case_id"], r["condition"], str(r.get("seed")))):
        k = (r["case_id"], r["condition"], r["model"], r.get("seed"))
        retry.append({"model": r["model"], "case_id": r["case_id"], "condition": r["condition"], "seed": r.get("seed"),
                      "in_main1": int(k in main_keys), "main1_status": main_keys[k].get("parse_status") if k in main_keys else "",
                      "retry_status": r.get("parse_status"), "retry_attempt": r.get("attempt", 1),
                      "retry_completion_tokens": r.get("completion_tokens"),
                      "retry_max_tokens": (r.get("decoding_params") or {}).get("max_tokens"),
                      "retry_stopped_at_limit": int(stopped_at_limit(r)),
                      "retry_finish_reason": r.get("finish_reason"),
                      "pretest": (r.get("parsed_answer") or {}).get("pretest"),
                      "posttest": (r.get("parsed_answer") or {}).get("posttest")})
    assert all(x["in_main1"] for x in retry), "ext5 contains keys that are not main1 keys"
    filled, n_rep = filled_records(main_final, ext5_final)
    assert n_rep == len(keys)
    f_rows = v1_rows(filled, cases)
    pre, post = headline(main_rows, cases, n_boot), headline(f_rows, cases, n_boot)
    hl = []
    for m in sorted(pre):
        for lab, h in (("prespecified", pre[m]), ("filled", post.get(m))):
            if h:
                hl.append(dict(h, model=m, sample=lab))
    n_ok = sum(x["retry_status"] == "ok" for x in retry)
    mac["extEFiveNRetried"] = str(len(retry))
    mac["extEFiveNParsed"] = str(n_ok)
    mac["extEFiveNStillTrunc"] = str(sum(x["retry_stopped_at_limit"] for x in retry))
    mac["extEFiveNCallError"] = str(sum(x["retry_status"] == "call_error" for x in retry))
    ds = [m for m in pre if "deepseek" in m] or sorted(pre)
    for m in ds:
        t = "extEFive" + tag(m)
        for lab, h in (("Pre", pre[m]), ("Fill", post[m])):
            mac[t + lab + "NParsed"] = str(h["n_parsed"])
            mac[t + lab + "OverMed"] = num(h["over_med_lod"])
            mac[t + lab + "OverPct"] = num(h["over_pct_above"], 1)
            mac[t + lab + "RatioMed"] = num(h["ratio_med"])
            mac[t + lab + "BPt"] = num(h["b"])
            mac[t + lab + "BLo"] = num(h["b_lo"])
            mac[t + lab + "BHi"] = num(h["b_hi"])
        mac[t + "DeltaRatioMed"] = num(post[m]["ratio_med"] - pre[m]["ratio_med"])
        mac[t + "DeltaB"] = num(post[m]["b"] - pre[m]["b"])
        mac[t + "DeltaOverMed"] = num(post[m]["over_med_lod"] - pre[m]["over_med_lod"])
    others = [m for m in pre if m not in ds]
    mac["extEFiveOtherModelsUnchanged"] = "yes" if all(
        abs(pre[m]["b"] - post[m]["b"]) < 1e-12 and abs(pre[m]["ratio_med"] - post[m]["ratio_med"]) < 1e-12
        for m in others) else "no"
    return retry, hl


E5_RETRY_FIELDS = ["model", "case_id", "condition", "seed", "in_main1", "main1_status", "retry_status", "retry_attempt",
                   "retry_completion_tokens", "retry_max_tokens", "retry_stopped_at_limit", "retry_finish_reason",
                   "pretest", "posttest"]
E5_HL_FIELDS = ["model", "sample", "n_parsed", "over_med_lod", "over_pct_above", "ratio_med", "ratio_n_gt1", "b_fe", "a",
                "b", "b_lo", "b_hi"]


# ---------------------------------------------------------------- outputs
def write_macros(path, mac, have):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for k, v in mac.items():
        assert k.isalpha() and k.startswith("ext"), f"bad macro name {k}"
        assert v not in ("", None), f"empty macro {k}"
        assert "\u2014" not in str(v)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/ext_analysis.py from results/raw and data/. Do not edit.\n"
                "% R1 extension analyses E1 to E5 (post hoc). \\extHave<E> is 1 if that batch had raw records; macros\n"
                "% of a missing batch are NOT defined, so guard their use with \\ifnum\\extHaveEOne=1 ... \\fi.\n")
        for b in BATCHES:
            f.write(f"\\newcommand{{\\extHave{WORD[b]}}}{{{1 if have[b] else 0}}}\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


def write_table(path, summ, have, models):
    """Compact one-column table (committee round 2, item 10): columns = models (first word of the display name),
    one row per key quantity, all values to two decimals. E3 and E4 show the paired change in |log-odds error|
    pooled over positive and negative results. The E5 row is dropped when filled and pre-specified b are identical
    to two decimals for every model (they are; the E5 values stay in ext_macros.tex and results/derived/ext_e5_headline.csv); otherwise it is kept."""
    cols = "l" + "r" * len(models)
    HDR = {"Claude": "Claude", "DeepSeek": "DS", "Gemini": "Gemini", "GPT-4o": "4o", "GPT-6.1": "6.1"}
    short = [HDR.get(display_name(m).split()[0], display_name(m).split()[0]) for m in models]
    lines = ["% GENERATED by analysis/ext_analysis.py. Do not edit.", "\\footnotesize\\setlength{\\tabcolsep}{1.5pt}",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             "Run & " + " & ".join(short) + " \\\\", "\\midrule"]

    def f2(x):
        if x is None or x == "":
            return ""
        s = f"{float(x):.2f}"
        if float(s) == 0.0:
            s = "0.00"
        return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s

    def row(label, vals):
        lines.append(label + " & " + " & ".join(vals) + " \\\\")
    if have["E1"]:
        s = {x["model"]: x for x in summ["E1"]}
        row("E1 gap, pos.", [f2(s[m]["gap_pos_lod_mean"]) if m in s else "" for m in models])
        row("E1 gap, neg.", [f2(s[m]["gap_neg_lod_mean"]) if m in s else "" for m in models])
        f = {x["model"]: x for x in summ["E1fits"] if x["variant"] == "blind_prior"}
        row("E1 $b$, blind prior", [f2(f[m]["b"]) if m in f else "" for m in models])
    if have["E2"]:
        s = {x["model"]: x for x in summ["E2"]}
        row("E2 DOR ratio", [f2(s[m]["log_dor_ratio_med"]) if m in s else "" for m in models])
        row("E2 $b$, own LR", [f2(s[m]["b_own"]) if m in s else "" for m in models])
    for b, c, lab in (("E3", "LvsN", "E3 long$-$N"), ("E4", "NunVsN", "E4 unnamed, N"),
                      ("E4", "LunVsL", "E4 unnamed, long")):
        if have[b]:
            s = {x["model"]: x for x in summ[b] if x["contrast"] == c and x["metric"] == "abs" and x["result"] == "all"}
            if s:
                row(lab, [f2(s[m]["mean"]) if m in s else "" for m in models])
    if have["E5"]:
        s = {(x["model"], x["sample"]): x for x in summ["E5"]}
        pairs = [(f2(s[(m, "prespecified")]["b"]), f2(s[(m, "filled")]["b"])) for m in models if (m, "filled") in s]
        if any(p != q for p, q in pairs):
            row("E5 $b$ pre / filled",
                [f"{f2(s[(m, 'prespecified')]['b'])} / {f2(s[(m, 'filled')]['b'])}" if (m, "filled") in s else ""
                 for m in models])
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# Compatibility slots expected by paper/main.tex (writer decision; names listed in paper/NUMBERS.md).
SHIM_FILES = {"E1": "ext_pretest_macros.tex", "E2": "ext_believedlr_macros.tex", "E3": "ext_longchart_macros.tex",
              "E5": "ext_dsretry_macros.tex"}


def long_note_tokens(notes_dir=os.path.join(ROOT, "data", "notes", "long")):
    """(min, max) length of the long-chart notes in tokens estimated as chars/4."""
    n = [len(open(p).read()) / 4.0 for p in glob.glob(os.path.join(notes_dir, "*__L.txt"))]
    return (min(n), max(n)) if n else (None, None)


def shim_macros(res, have, cases):
    """{file name: {macro: value}} for the main.tex slots of the present batches."""
    out = {}
    if have["E1"]:
        cells, summ, fits = res["E1"]
        bf = [f for f in fits if f["variant"] == "blind_prior"]
        blind = res["E1blind"]
        m = {"cfPreOnlyMedAll": num(statistics.median(blind), 1)}
        if bf:
            m.update(cfPreOnlyParAMin=num(min(f["a"] for f in bf)), cfPreOnlyParAMax=num(max(f["a"] for f in bf)),
                     cfPreOnlyParBMin=num(min(f["b"] for f in bf)), cfPreOnlyParBMax=num(max(f["b"] for f in bf)))
        out[SHIM_FILES["E1"]] = m
    if have["E2"]:
        _sc, summ, _f = res["E2"]
        ro = [s["ratio_own_med"] for s in summ if s["ratio_own_med"] is not None]
        m = {"cfBelLrDorRatioMin": num(min(s["log_dor_ratio_med"] for s in summ)),
             "cfBelLrDorRatioMax": num(max(s["log_dor_ratio_med"] for s in summ))}
        if ro:
            m.update(cfBelLrRatioMedMin=num(min(ro)), cfBelLrRatioMedMax=num(max(ro)))
        out[SHIM_FILES["E2"]] = m
    if have["E3"]:
        ctr = res["E3"][0]
        lo, hi = long_note_tokens()
        m = {}
        if lo is not None:
            m.update(cfLongTokensMin=f"{lo:,.0f}", cfLongTokensMax=f"{hi:,.0f}")
        for c, name, src in (("LvsN", "cfLongVsNAll", ctr), ("LunVsN", "cfLongUnnamedVsNAll",
                                                              res["E4"][0] if have["E4"] else [])):
            r = [x for x in src if x["contrast"] == c and x["metric"] == "abs" and x["result"] == "all"
                 and x["model"] == "ALL"]
            if r:
                m.update({name + "Mean": num(r[0]["mean"]), name + "Lo": num(r[0]["lo"]), name + "Hi": num(r[0]["hi"])})
        out[SHIM_FILES["E3"]] = m
    if have["E5"]:
        _rt, hl = res["E5"]
        h = {(x["model"], x["sample"]): x for x in hl}
        ds = sorted({x["model"] for x in hl if "deepseek" in x["model"]})
        m = {"cfDsRetryNParsed": str(sum(1 for x in _rt if x["retry_status"] == "ok"))}
        if ds:
            m.update(cfDsRetryParBBefore=num(h[(ds[0], "prespecified")]["b"]),
                     cfDsRetryParBAfter=num(h[(ds[0], "filled")]["b"]))
        out[SHIM_FILES["E5"]] = m
    return out


def write_shim(path, mac):
    for k, v in mac.items():
        assert k.isalpha() and v not in ("", None) and "\u2014" not in str(v), k
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/ext_analysis.py (slot file for paper/main.tex). Do not edit.\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


def remove(path):
    if os.path.exists(path):
        os.remove(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "derived"))
    ap.add_argument("--generated", default=os.path.join(ROOT, "paper", "generated"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--ext4-min-keys", type=int, default=600,
                    help="E4 is analysed only once this many case x condition x model x sample keys have a latest "
                         "attempt (600 = complete ext4_unnamed); a partial run is treated as absent")
    ap.add_argument("--main-run-id", default=None, help="default: the single paper run id in the config (main1)")
    for b in BATCHES:
        ap.add_argument("--" + b.lower().replace("e", "ext"), dest=b, default=DEFAULT_IDS[b])
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    main_id = a.main_run_id or cfg["paper_run_ids"][0]
    assert a.main_run_id or len(cfg["paper_run_ids"]) == 1, "ext analyses assume one paper run id"
    cases = load_cases(a.cases)
    recs, _ = load_raw(a.raw)
    syn = is_synthetic(recs, a.raw)
    for d in (a.out, a.generated):
        guard_output(d, syn)
    ids = present_ids(recs)
    assert main_id in ids, f"main run id {main_id} not in raw"
    main_final = final_of(recs, main_id, cfg)
    main_rows = v1_rows(main_final, cases)
    fin = {b: final_of(recs, getattr(a, b), cfg) if getattr(a, b) in ids else [] for b in BATCHES}
    have = {b: bool(fin[b]) for b in BATCHES}
    if have["E4"] and len(fin["E4"]) < a.ext4_min_keys:
        print(f"ext_analysis: E4 PARTIAL ({len(fin['E4'])} of {a.ext4_min_keys} keys); treated as absent, no E4 outputs")
        fin["E4"], have["E4"] = [], False
    mac, summ, res = {}, {}, {}
    out = lambda n: os.path.join(a.out, n)  # noqa: E731
    if have["E1"]:
        c, s, f = e1(fin["E1"], main_rows, cases, a.n_boot, mac)
        write_csv(out("ext_e1_cells.csv"), c, E1_CELL_FIELDS)
        write_csv(out("ext_e1_summary.csv"), s, E1_SUM_FIELDS)
        write_csv(out("ext_e1_fits.csv"), f, FIT_FIELDS)
        summ["E1"], summ["E1fits"] = s, f
        res["E1"] = (c, s, f)
        bl, _nt, _no = blind_cells(fin["E1"], scenario_targets(cases))
        res["E1blind"] = [100 * p for ps in bl.values() for p in ps]
    if have["E2"]:
        sc, s, f = e2(fin["E2"], cases, a.n_boot, mac)
        write_csv(out("ext_e2_scenarios.csv"), sc, E2_SCEN_FIELDS)
        write_csv(out("ext_e2_summary.csv"), s, E2_SUM_FIELDS)
        write_csv(out("ext_e2_fits.csv"), f, E2_FIT_FIELDS)
        summ["E2"] = s
        res["E2"] = (sc, s, f)
    if have["E3"]:
        ctr, pr, tr = e3(fin["E3"], main_final, main_rows, cases, mac)
        write_csv(out("ext_e3_contrasts.csv"), ctr, CTR_FIELDS)
        write_csv(out("ext_e3_parse.csv"), pr, PARSE_FIELDS)
        write_csv(out("ext_e3_tracking.csv"), tr, TRACK_FIELDS)
        summ["E3"] = ctr
        res["E3"] = (ctr, pr, tr)
    if have["E4"]:
        ctr, pr, tr = e4(fin["E4"], fin["E3"], main_final, main_rows, cases, mac)
        write_csv(out("ext_e4_contrasts.csv"), ctr, CTR_FIELDS)
        write_csv(out("ext_e4_parse.csv"), pr, PARSE_FIELDS)
        write_csv(out("ext_e4_tracking.csv"), tr, TRACK_FIELDS)
        summ["E4"] = ctr
        res["E4"] = (ctr, pr, tr)
    if have["E5"]:
        rt, hl = e5(fin["E5"], main_final, main_rows, cases, a.n_boot, mac)
        write_csv(out("ext_e5_retry.csv"), rt, E5_RETRY_FIELDS)
        write_csv(out("ext_e5_headline.csv"), hl, E5_HL_FIELDS)
        summ["E5"] = hl
        res["E5"] = (rt, hl)
    for b in BATCHES:
        if not have[b]:
            for n in CSV_FILES[b]:
                remove(out(n))
    if all(have[b] for b in ("E1", "E2", "E3", "E4")):
        def tot(keys):
            return sum(int(mac[k]) for k in keys)
        arms3, arms4 = ["extEThreeL"], ["extEFourNUnnamed", "extEFourLUnnamed"]
        mac["extRunsNTotal"] = str(tot(["extEOneNTotal", "extETwoNTotal"] + [x + "NTotal" for x in arms3 + arms4]))
        mac["extRunsNParsed"] = str(tot(["extEOneNParsed", "extETwoNParsed"] + [x + "NParsed" for x in arms3 + arms4]))
        trunc_keys = ["extEOneNTrunc", "extETwoNTrunc"] + [k for k in mac if k.endswith("NTrunc") and any(
            k.startswith(x) for x in arms3 + arms4)]
        mac["extRunsNTruncTotal"] = str(tot(trunc_keys))
        # cross-check against raw records (all models, calibration excluded) and per-run macros
        scen_ids = set(scenario_targets(cases))   # E1 case ids are scenario ids
        analysed = [r for b in ("E1", "E2", "E3", "E4") for r in fin[b]
                    if (r["case_id"] in scen_ids if b == "E1" else not is_calibration(cases[r["case_id"]]))]
        assert int(mac["extRunsNTotal"]) == len(analysed)
        assert int(mac["extRunsNTruncTotal"]) == sum(length_stopped(r) for r in analysed)
        assert int(mac["extRunsNParsed"]) <= int(mac["extRunsNTotal"])
    shims = shim_macros(res, have, cases)
    for b, n in SHIM_FILES.items():
        if n in shims:
            write_shim(os.path.join(a.generated, n), shims[n])
        else:
            remove(os.path.join(a.generated, n))
    mpath, tpath = os.path.join(a.generated, "ext_macros.tex"), os.path.join(a.generated, "table_ext.tex")
    if any(have.values()):
        write_macros(mpath, mac, have)
        write_table(tpath, summ, have, sorted({r["model"] for r in main_rows}))
    else:
        remove(mpath)
        remove(tpath)
    print("ext_analysis: batches present " + (", ".join(f"{b}={getattr(a, b)}" for b in BATCHES if have[b]) or "none")
          + f"; {len(mac)} macros" + ("" if any(have.values()) else "; ext_macros.tex and table_ext.tex not generated"))
    return res, mac, have


if __name__ == "__main__":
    main()
