"""Build paper tables and LaTeX macros from scored.csv, summary_model_condition.csv, wbayes.csv.

Outputs (all asserted):
  <derived>/table_error_by_condition.csv   model x set x result x condition error summary
  <derived>/table_contrasts.csv            planned paired contrasts (protocol Section 6)
  <derived>/table_wbayes.csv               a, b, c with 95% CI: published (main) beside noboundary
  <generated>/results_macros.tex           one \\newcommand per quoted number (letters-only names)
  <generated>/table_error.tex, table_wbayes.tex, table_wbayes_own.tex   booktabs tabulars (main.tex loads booktabs)

Usage: python analysis/make_tables.py [--derived results/derived] [--out-derived results/derived]
                                      [--generated paper/generated] [--n-boot 2000]
Refuses to write into results/derived or paper/generated if any input was synthetic.
"""
import argparse
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, is_calibration, load_cases, display_name, guard_output, guard_pilot, read_run_set, read_csv,  # noqa: E402
                    read_provenance, write_csv)
from score import quartiles  # noqa: E402

SEED = 20261111
CONDS = ["V", "N", "B", "S", "E"]
FIT_CONDS = ["V", "N", "B", "S"]  # protocol Section 5: condition E is excluded from the fits
# planned contrasts: name, X, Y (difference = X minus Y), metrics (protocol Section 6)
CONTRASTS = [("NvsV", "N", "V", ["band_err_pp", "abs_logodds_err"], "H1"),
             ("BvsN", "B", "N", ["band_err_pp", "abs_logodds_err"], "H2"),
             ("SvsN", "S", "N", ["band_err_pp", "abs_logodds_err"], "H3"),
             ("EvsN", "E", "N", ["logodds_err"], "H4")]  # signed shift in log-odds
METRIC_TAG = {"band_err_pp": "Band", "abs_logodds_err": "Lod", "logodds_err": "Shift"}
DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def model_tag(slug):
    """Letters-only CamelCase tag for a model slug, e.g. gpt-4o -> GptFourO."""
    out, tok = [], ""
    for ch in slug + "-":
        if ch.isalpha():
            tok += ch
        else:
            if tok:
                out.append(tok[0].upper() + tok[1:])
                tok = ""
            if ch.isdigit():
                out.append(DIGITS[int(ch)])
    return "".join(out)


def num(x, d=2):
    return "" if x is None or x == "" else f"{float(x):.{d}f}"


def tex_num(x, d=2):
    s = f"{float(x):.{d}f}"
    if float(s) == 0.0 and float(x) != 0.0:  # L3: a tiny nonzero value gets extra decimals, never "-0.0"
        for dd in (d + 1, d + 2, d + 3, d + 4):
            s = f"{float(x):.{dd}f}"
            if float(s) != 0.0:
                break
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def fnum(x):
    return None if x in ("", None) else float(x)


def median_iqr(vals):
    q1, q3 = quartiles(vals)
    return statistics.median(vals), q1, q3


def main_rows(scored):
    """Rows that may enter cross-condition results: everything except calibration cases (A5, V only)."""
    return [r for r in scored if r.get("calibration", "0") != "1"]


# ---------------------------------------------------------------- error table
def error_table(scored, summary):
    groups = {}
    scored = main_rows(scored)  # H1: the A5 calibration cases are not in any error cell
    for r in scored:
        for res in ("all", r["result"]):
            groups.setdefault((r["model"], r["set"], res, r["condition"]), []).append(r)
    rows = []
    for (m, st, res, c), rs in sorted(groups.items()):
        ok = [r for r in rs if r["parse_ok"] == "1"]
        row = {"model": m, "set": st, "result": res, "condition": c, "n_total": len(rs),
               "n_ok": len(ok), "n_parse_fail": len(rs) - len(ok)}
        for src, name in (("band_err_pp", "band_pp"), ("abs_logodds_err", "abs_lod")):
            vals = [float(r[src]) for r in ok]
            if vals:
                row[name + "_med"], row[name + "_q1"], row[name + "_q3"] = median_iqr(vals)
        rows.append(row)
    # cross-check against summary_model_condition.csv for the single-result groups
    sm = {(s["model"], s["condition"], s["set"], s["result"], s["metric"]): s for s in (summary or [])}
    for row in rows:
        if summary is None or row["result"] == "all" or row["n_ok"] == 0:  # summary=None: sensitivity rerun
            continue
        s = sm[(row["model"], row["condition"], row["set"], row["result"], "band_err_pp")]
        assert int(s["n"]) == row["n_ok"] and int(s["n_parse_fail"]) == row["n_parse_fail"], row
        assert abs(float(s["median"]) - row["band_pp_med"]) < 1e-9, (row, s)
    assert sum(r["n_total"] for r in rows if r["result"] == "all") == len(scored)
    # H1 guard: every Set A model x condition cell covers the same cases, so n_total is equal across conditions
    for m in {r["model"] for r in rows}:
        for st in ("A", "B"):
            for res in ("all", "pos", "neg"):
                ns = {r["n_total"] for r in rows if r["model"] == m and r["set"] == st
                      and r["result"] == res and r["condition"] in ("V", "N", "B", "S", "E")}
                # E may differ from the others only in the number of parse failures, never in n_total
                assert len(ns) == 1, (m, st, res, ns)
    for r in rows:
        if r["n_ok"]:
            assert all(math.isfinite(r[k]) for k in ("band_pp_med", "band_pp_q1", "band_pp_q3",
                                                     "abs_lod_med", "abs_lod_q1", "abs_lod_q3")), r
    return rows


ERR_FIELDS = ["model", "set", "result", "condition", "n_total", "n_ok", "n_parse_fail", "band_pp_med",
              "band_pp_q1", "band_pp_q3", "abs_lod_med", "abs_lod_q1", "abs_lod_q3"]


# ---------------------------------------------------------------- contrasts
def paired_case_diffs(scored, model, st, res, X, Y, metric):
    """{case_id: (scenario, mean over seeds of metric_X - metric_Y)} over pairs where both parsed."""
    cell = {}
    for r in scored:
        if r["model"] != model or r["set"] != st or r["parse_ok"] != "1":
            continue
        if res != "all" and r["result"] != res:
            continue
        if r["condition"] in (X, Y):
            cell[(r["case_id"], r["seed"], r["condition"])] = (r["scenario"], float(r[metric]))
    per_case = {}
    for (cid, seed, cond), (scen, v) in cell.items():
        if cond != X or (cid, seed, Y) not in cell:
            continue
        per_case.setdefault(cid, (scen, []))[1].append(v - cell[(cid, seed, Y)][1])
    return {cid: (scen, sum(d) / len(d)) for cid, (scen, d) in per_case.items()}


def contrast_table(scored, n_boot):
    rows = []
    scored = main_rows(scored)
    models = sorted({r["model"] for r in scored})
    for m in models:
        for st in ("A", "B"):
            for res in ("all", "pos", "neg"):
                for name, X, Y, metrics, hyp in CONTRASTS:
                    for metric in metrics:
                        d = paired_case_diffs(scored, m, st, res, X, Y, metric)
                        row = {"model": m, "set": st, "result": res, "contrast": name,
                               "hypothesis": hyp, "metric": metric, "n_cases": len(d)}
                        if d:
                            by = {}
                            for scen, v in d.values():
                                by.setdefault(scen, []).append(v)
                            vals = [v for vs in by.values() for v in vs]
                            row["n_clusters"] = len(by)
                            row["median_diff"] = statistics.median(vals)
                            row["frac_positive"] = sum(v > 0 for v in vals) / len(vals)
                            if len(by) >= 2:
                                rng = random.Random(SEED)
                                ids, boots = list(by), []
                                for _ in range(n_boot):
                                    s = []
                                    for _i in ids:
                                        s.extend(by[ids[rng.randrange(len(ids))]])
                                    boots.append(statistics.median(s))
                                boots.sort()
                                row["ci_lo"] = boots[int(0.025 * (n_boot - 1))]
                                row["ci_hi"] = boots[int(math.ceil(0.975 * (n_boot - 1)))]
                            row["note"] = "few clusters: CIs unreliable" if len(by) < 10 else ""
                        else:
                            row["note"] = "no paired cases"
                        rows.append(row)
    for r in rows:
        if r["n_cases"]:
            assert math.isfinite(r["median_diff"]), r
            if "ci_lo" in r:
                assert r["ci_lo"] <= r["ci_hi"], r
    return rows


CON_FIELDS = ["model", "set", "result", "contrast", "hypothesis", "metric", "n_cases", "n_clusters",
              "median_diff", "ci_lo", "ci_hi", "frac_positive", "note"]


# ---------------------------------------------------------------- wbayes table
def wbayes_table(wb, variant="published"):
    idx = {(r["model"], r["condition"], r["variant"]): r for r in wb}
    rows = []
    for (m, c, v), r in sorted(idx.items()):
        if v != variant or c not in FIT_CONDS:
            continue
        nb = idx.get((m, c, variant + "_noboundary"))
        row = {"model": m, "condition": c, "n_obs": r["n_obs"], "n_clusters": r["n_clusters"],
               "note": r["note"]}
        for p in "abc":
            for suf in ("", "_lo", "_hi"):
                row[f"{p}{suf}"] = fnum(r[f"{p}{suf}"])
                row[f"{p}{suf}_noboundary"] = fnum(nb[f"{p}{suf}"]) if nb else None
        row["n_boundary_answers"] = r["n_boundary_answers"]
        row["boundary_flag"] = int(r["boundary_flag"])
        row["boundary_flag_params"] = r["boundary_flag_params"]
        row["noboundary_note"] = nb["note"] if nb else "missing"
        if row["a"] is not None:
            assert all(math.isfinite(row[p]) for p in "abc"), row
        rows.append(row)
    assert rows, f"no {variant}-variant fits for conditions " + ",".join(FIT_CONDS)
    return rows


WB_FIELDS = ["model", "condition", "n_obs", "n_clusters"] + \
    [f"{p}{s}{n}" for n in ("", "_noboundary") for p in "abc" for s in ("", "_lo", "_hi")] + \
    ["n_boundary_answers", "boundary_flag", "boundary_flag_params", "note", "noboundary_note"]


# ---------------------------------------------------------------- calibration, overestimation, ranking
def calibration_table(scored):
    """A5 (numeric hypothetical, condition V only): per model x result, the stated posttest vs the exact one."""
    rows = []
    cal = [r for r in scored if r.get("calibration") == "1"]
    for m in sorted({r["model"] for r in scored}):
        for res in ("pos", "neg"):
            rs = [r for r in cal if r["model"] == m and r["result"] == res]
            if not rs:
                continue
            ok = [r for r in rs if r["parse_ok"] == "1"]
            row = {"model": m, "result": res, "condition": ",".join(sorted({r["condition"] for r in rs})),
                   "n_total": len(rs), "n_ok": len(ok), "target_post_pp": float(rs[0]["posttest_target_pp"])}
            if ok:
                post = [float(r["posttest_stated"]) for r in ok]
                row["stated_post_med_pp"] = statistics.median(post)
                row["stated_post_min_pp"], row["stated_post_max_pp"] = min(post), max(post)
                row["diff_med_pp"] = row["stated_post_med_pp"] - row["target_post_pp"]
                row["abs_lod_med"] = statistics.median([float(r["abs_logodds_err"]) for r in ok])
            rows.append(row)
    assert rows and all(r["condition"] == "V" for r in rows), "calibration cases must be condition V only"
    return rows


CAL_FIELDS = ["model", "result", "condition", "n_total", "n_ok", "target_post_pp", "stated_post_med_pp",
              "stated_post_min_pp", "stated_post_max_pp", "diff_med_pp", "abs_lod_med"]


def overestimation_table(scored):
    """Positive-test cases (no calibration, all conditions pooled): signed log-odds error and share above."""
    rows = []
    for m in sorted({r["model"] for r in scored}):
        ok = [r for r in main_rows(scored) if r["model"] == m and r["result"] == "pos" and r["parse_ok"] == "1"]
        le = [float(r["logodds_err"]) for r in ok]
        bs = [float(r["band_err_signed_pp"]) for r in ok]
        rows.append({"model": m, "n": len(ok), "median_signed_lod": statistics.median(le),
                     "pct_above_target": 100.0 * sum(x > 0 for x in le) / len(le),
                     "pct_above_band": 100.0 * sum(x > 0 for x in bs) / len(bs)})
    return rows


OVER_FIELDS = ["model", "n", "median_signed_lod", "pct_above_target", "pct_above_band"]


def ranking_table(scored):
    """Per model, median |log-odds error| and median band error over all parsed non-calibration answers."""
    rows = []
    for m in sorted({r["model"] for r in scored}):
        ok = [r for r in main_rows(scored) if r["model"] == m and r["parse_ok"] == "1"]
        rows.append({"model": m, "n": len(ok),
                     "abs_lod_med": statistics.median([float(r["abs_logodds_err"]) for r in ok]),
                     "band_pp_med": statistics.median([float(r["band_err_pp"]) for r in ok])})
    for key, rk in (("abs_lod_med", "rank_lod"), ("band_pp_med", "rank_band")):
        for i, r in enumerate(sorted(rows, key=lambda r: r[key]), 1):
            r[rk] = i
    return sorted(rows, key=lambda r: r["rank_lod"])


RANK_FIELDS = ["model", "n", "abs_lod_med", "rank_lod", "band_pp_med", "rank_band"]


def h4_pooled(scored):
    """H4 (E-N signed log-odds shift): pooled median of paired per-case differences over models, per set,
    and the number of model cells with a positive median shift (result 'all')."""
    out = {}
    models = sorted({r["model"] for r in scored})
    sc = main_rows(scored)
    for st in ("A", "B"):
        pooled, npos, ncases = [], 0, []
        for m in models:
            d = paired_case_diffs(sc, m, st, "all", "E", "N", "logodds_err")
            vals = [v for _, v in d.values()]
            pooled += vals
            ncases.append(len(vals))
            npos += int(statistics.median(vals) > 0)
        out[st] = {"pooled_median": statistics.median(pooled), "n_pairs": len(pooled),
                   "n_cells_positive": npos, "n_cells": len(models),
                   "n_cases_max": max(ncases)}  # M3: each pair is one model-by-case mean of seed-paired differences
        assert len(pooled) <= max(ncases) * len(models)
    return out


# ---------------------------------------------------------------- contrast counts, stated vs published pretest
def contrast_counts(con):
    """Count 'all'-result contrast rows, those whose 95% CI excludes 0, and those excluding 0 against the
    hypothesised direction. H1 to H3 (N-V, B-N, S-N) hypothesise a positive difference (more error); H4 (E-N)
    has no hypothesised sign (protocol), so its exclusions are counted separately and never as 'against'."""
    rows = [r for r in con if r["result"] == "all"]
    out = {"total": len(rows), "excl": 0, "excl_against": 0, "excl_with": 0, "excl_h4": 0}
    for r in rows:
        if r["ci_lo"] == "" or r["ci_hi"] == "":
            continue
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        if lo > 0 or hi < 0:
            out["excl"] += 1
            if r["hypothesis"] == "H4":
                out["excl_h4"] += 1
            elif hi < 0:
                out["excl_against"] += 1
            else:
                out["excl_with"] += 1
    assert out["excl"] == out["excl_against"] + out["excl_with"] + out["excl_h4"]
    return out


STATED_PRIOR_FIELDS = ["scenario", "set", "published_prior_pct", "n_answers", "median_stated_pretest_pct",
                       "q1_pct", "q3_pct", "median_minus_published_pp", "median_exceeds_published"]


def stated_prior_table(scored, cases):
    """Per scenario: median stated pretest (percent), pooled over models, seeds, results and the published-prior
    conditions V N B S (no E, no A5), against the published prior (case file target.pretest_point)."""
    by = {}
    for r in main_rows(scored):
        if r["condition"] not in FIT_CONDS or r["parse_ok"] != "1" or r["pretest_stated"] == "":
            continue
        by.setdefault(r["scenario"], []).append(float(r["pretest_stated"]))
    pub = {}
    for c in cases.values():
        pub.setdefault(c["scenario"], set()).add(c["target"]["pretest_point"])
    rows = []
    for sc in sorted(by):
        assert len(pub[sc]) == 1, f"{sc}: pos and neg cases disagree on the published prior"
        p = next(iter(pub[sc])) * 100.0
        v = by[sc]
        q1, q3 = quartiles(v)
        med = statistics.median(v)
        st = {r["set"] for r in scored if r["scenario"] == sc}
        assert len(st) == 1
        rows.append({"scenario": sc, "set": next(iter(st)), "published_prior_pct": p, "n_answers": len(v),
                     "median_stated_pretest_pct": med, "q1_pct": q1, "q3_pct": q3,
                     "median_minus_published_pp": med - p, "median_exceeds_published": int(med > p)})
    return rows


# ---------------------------------------------------------------- C4 additions; all additive
# H3: the one reply the pre-specified strict parser rejected. The parser is NOT changed; this is a
# sensitivity analysis only. Identifiers of the raw record (results/raw/main1.jsonl, read only) and the
# value a human reader takes from its last two lines are fixed here so the choice is auditable.
UNPARSED_HAND_READ = {
    "run_id": "main1", "timestamp": "2026-11-12T23:24:26+00:00", "case_id": "B1_pe_low_pos",
    "condition": "E", "model": "openai/gpt-4o", "seed": 5,
    "response_tail": "- PRETEST: 15\n- POSTTEST: 25",  # the reply ends with these two bulleted lines
    "pretest": 15.0, "posttest": 25.0,  # the strict parser needs 'PRETEST:' at line start, so it rejects '- PRETEST:'
}


def _find_raw_record(raw_paths, ident):
    import glob
    import json
    hits = []
    for pat in raw_paths:
        for path in sorted(glob.glob(pat)):
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if (r.get("run_id"), r.get("timestamp"), r.get("case_id"), r.get("condition"),
                            r.get("model"), r.get("seed")) == (ident["run_id"], ident["timestamp"], ident["case_id"],
                                                               ident["condition"], ident["model"], ident["seed"]):
                        hits.append(r)
    return hits


def _printed_err(r):
    return tuple(tex_num(r[k], 1 if k.startswith("band") else 2) if r.get(k) not in (None, "") else ""
                 for k in ("band_pp_med", "band_pp_q1", "band_pp_q3", "abs_lod_med", "abs_lod_q1", "abs_lod_q3")) \
        + (r["n_ok"],)


def _printed_con(r):
    d = 1 if r["metric"] == "band_err_pp" else 2
    return tuple(tex_num(r[k], d) if r.get(k) not in (None, "") else "" for k in ("median_diff", "ci_lo", "ci_hi")) \
        + (r["n_cases"],)


def unparsed_sensitivity(scored, cases, raw_paths, n_boot, ident=None):
    """H3. Re-run the error table, the contrasts and the H4 pooled summary with the one unparsed reply read
    by hand (ident), and list every cell whose printed value changes. Returns None when the record is not in
    scored (synthetic test data). Never edits any input."""
    from score import score_record
    ident = ident or UNPARSED_HAND_READ
    key = lambda r: (r["run_id"], r["case_id"], r["model"], r["condition"], str(r["seed"]))  # noqa: E731
    k0 = (ident["run_id"], ident["case_id"], ident["model"], ident["condition"], str(ident["seed"]))
    idx = [i for i, r in enumerate(scored) if key(r) == k0]
    if not idx:
        return None
    assert len(idx) == 1 and scored[idx[0]]["parse_ok"] == "0", "the sensitivity record must be the unparsed one"
    hits = _find_raw_record(raw_paths, ident)
    assert len(hits) == 1, f"expected exactly one raw record, found {len(hits)}"
    rec = hits[0]
    assert rec["response"].rstrip().endswith(ident["response_tail"]), "raw reply no longer ends as documented"
    assert rec["parse_status"] != "ok"
    hand = dict(rec, parse_status="ok", parsed_answer={"pretest": ident["pretest"], "posttest": ident["posttest"]})
    s = score_record(hand, cases[rec["case_id"]])
    new = dict(scored[idx[0]])
    for k in new:
        new[k] = "" if s.get(k) is None else str(s.get(k, ""))
    new["parse_ok"], new["parse_status"], new["failure_kind"] = "1", "ok", ""
    scored2 = list(scored)
    scored2[idx[0]] = new
    err1, err2 = error_table(scored, None), error_table(scored2, None)
    con1, con2 = contrast_table(scored, n_boot), contrast_table(scored2, n_boot)
    h41, h42 = h4_pooled(scored), h4_pooled(scored2)
    changed = []
    for a, b in zip(err1, err2):
        assert (a["model"], a["set"], a["result"], a["condition"]) == (b["model"], b["set"], b["result"], b["condition"])
        if _printed_err(a) != _printed_err(b):
            changed.append({"table": "error", "model": a["model"], "set": a["set"], "result": a["result"],
                            "condition": a["condition"], "contrast": "", "metric": "band_pp_med/abs_lod_med",
                            "strict": "|".join(map(str, _printed_err(a))), "hand_read": "|".join(map(str, _printed_err(b)))})
    for a, b in zip(con1, con2):
        assert (a["model"], a["set"], a["result"], a["contrast"], a["metric"]) == \
               (b["model"], b["set"], b["result"], b["contrast"], b["metric"])
        if a["n_cases"] and _printed_con(a) != _printed_con(b):
            changed.append({"table": "contrast", "model": a["model"], "set": a["set"], "result": a["result"],
                            "condition": "", "contrast": a["contrast"], "metric": a["metric"],
                            "strict": "|".join(map(str, _printed_con(a))), "hand_read": "|".join(map(str, _printed_con(b)))})
    return {"changed": changed, "con_strict": con1, "con_hand": con2, "err_strict": err1, "err_hand": err2,
            "h4_strict": h41, "h4_hand": h42, "model": ident["model"], "set": s["set"], "case_id": ident["case_id"]}


SENS_FIELDS = ["table", "model", "set", "result", "condition", "contrast", "metric", "strict", "hand_read"]


def sensitivity_macros(sens):
    """Macros for the H3 sensitivity. 'Cells' counts: error-table cells (model x set x result x condition) and
    contrast rows whose printed value changes, split into the pooled-result ('all') rows quoted in the paper
    and all strata."""
    out = {}
    ch = sens["changed"]
    out["resUnparsedSensPretest"] = tex_num(UNPARSED_HAND_READ["pretest"], 0)
    out["resUnparsedSensPosttest"] = tex_num(UNPARSED_HAND_READ["posttest"], 0)
    out["resUnparsedSensErrCellsAll"] = sum(1 for c in ch if c["table"] == "error" and c["result"] == "all")
    out["resUnparsedSensErrCellsAllStrata"] = sum(1 for c in ch if c["table"] == "error")
    out["resUnparsedSensConCellsAll"] = sum(1 for c in ch if c["table"] == "contrast" and c["result"] == "all")
    out["resUnparsedSensConCellsAllStrata"] = sum(1 for c in ch if c["table"] == "contrast")
    out["resUnparsedSensCells"] = out["resUnparsedSensErrCellsAll"] + out["resUnparsedSensConCellsAll"]
    m, st = sens["model"], sens["set"]

    def con(rows, contrast, metric):
        r = [x for x in rows if x["model"] == m and x["set"] == st and x["result"] == "all"
             and x["contrast"] == contrast and x["metric"] == metric]
        assert len(r) == 1
        return r[0]
    for tag, rows in (("Strict", sens["con_strict"]), ("Hand", sens["con_hand"])):
        r = con(rows, "EvsN", "logodds_err")
        out[f"resUnparsedSensGptBEN{tag}"] = tex_num(r["median_diff"], 3)
        out[f"resUnparsedSensGptBEN{tag}Lo"] = tex_num(r["ci_lo"], 3)
        out[f"resUnparsedSensGptBEN{tag}Hi"] = tex_num(r["ci_hi"], 3)
    out["resUnparsedSensGptBEN"] = out["resUnparsedSensGptBENHand"]
    for tag, rows in (("Strict", sens["err_strict"]), ("Hand", sens["err_hand"])):
        r = [x for x in rows if x["model"] == m and x["set"] == st and x["result"] == "all" and x["condition"] == "E"]
        assert len(r) == 1
        out[f"resUnparsedSensGptBEBand{tag}"] = tex_num(r[0]["band_pp_med"], 1)
        out[f"resUnparsedSensGptBELod{tag}"] = tex_num(r[0]["abs_lod_med"], 2)
        out[f"resUnparsedSensGptBENOk{tag}"] = r[0]["n_ok"]
    for tag, h in (("Strict", sens["h4_strict"]), ("Hand", sens["h4_hand"])):
        out[f"resUnparsedSensHFourNCellsPositiveSetB{tag}"] = h[st]["n_cells_positive"]
        out[f"resUnparsedSensHFourPooledMedSetB{tag}"] = tex_num(h[st]["pooled_median"], 3)
        out[f"resUnparsedSensHFourNPairsSetB{tag}"] = h[st]["n_pairs"]
    return out


# H4 (C4): response breakdown
def response_breakdown(scored):
    cal = [r for r in scored if r.get("calibration") == "1"]
    main = main_rows(scored)
    parsed = [r for r in main if r["parse_ok"] == "1"]
    out = {"total": len(scored), "cal": len(cal), "cal_parsed": sum(r["parse_ok"] == "1" for r in cal),
           "noncal": len(main), "noncal_parsed": len(parsed), "noncal_unparsed": len(main) - len(parsed)}
    assert out["total"] == out["cal"] + out["noncal"]
    assert out["noncal"] == out["noncal_parsed"] + out["noncal_unparsed"]
    assert out["total"] - out["cal_parsed"] - out["noncal_parsed"] == sum(r["parse_ok"] != "1" for r in scored)
    return out


# H2 (C4): secondary, midpoint-based error (Rodman et al. report MAE, which for ranges appears to use the midpoint)
MID_FIELDS = ["model", "set", "condition", "n", "mid_mae_pp", "mid_median_abs_pp", "band_mae_pp", "band_median_pp",
              "label"]


def midpoint_error_table(scored, cases):
    rows = []
    for m in sorted({r["model"] for r in scored}):
        ok = [r for r in main_rows(scored) if r["model"] == m and r["set"] == "A" and r["condition"] == "V"
              and r["parse_ok"] == "1"]
        mid = []
        for r in ok:
            lo, hi = cases[r["case_id"]]["target"]["posttest_band_primary"]
            mid.append(abs(float(r["posttest_stated"]) - 100.0 * (lo + hi) / 2.0))
        band = [float(r["band_err_pp"]) for r in ok]
        assert ok and len(mid) == len(band)
        rows.append({"model": m, "set": "A", "condition": "V", "n": len(ok), "mid_mae_pp": sum(mid) / len(mid),
                     "mid_median_abs_pp": statistics.median(mid), "band_mae_pp": sum(band) / len(band),
                     "band_median_pp": statistics.median(band),
                     "label": "SECONDARY descriptive; primary metric is distance outside the band"})
    return rows


# M11 (C4): H5 split by test result
H5_FIELDS = ["model", "condition", "n_pos", "n_neg", "n_clusters"] + \
    [f"{p}_{k}{s}" for p in "ab" for k in ("pos", "neg", "diff") for s in ("", "_lo", "_hi")] + ["n_boot_valid", "note"]


def h5_split_fits(scored, cases, wbt, n_boot):
    """Weighted-Bayes fit (published prior, boundary answers kept) separately on positive and negative test
    results, per model x condition V N B S. Scenarios are the bootstrap clusters; each resample is applied to
    both strata so the neg minus pos difference has a paired interval. Cross-checked against wbayes.csv: the
    pooled (pos+neg) refit must reproduce the published-variant a, b."""
    from fit_wbayes import ols, pct
    from common import logit_clipped
    data = {}
    for r in main_rows(scored):
        if r["parse_ok"] != "1" or r["condition"] not in FIT_CONDS:
            continue
        t = cases[r["case_id"]]["target"]
        row = (logit_clipped(t["pretest_point"]), math.log(t["lr_point"]), logit_clipped(float(r["posttest_stated"]) / 100.0))
        data.setdefault((r["model"], r["condition"]), {}).setdefault(r["result"], {}).setdefault(r["scenario"], []).append(row)
    wbi = {(r["model"], r["condition"]): r for r in wbt}
    rows = []
    for (m, c), d in sorted(data.items()):
        pos, neg = d.get("pos", {}), d.get("neg", {})
        ids = sorted(set(pos) | set(neg))
        flat = lambda dd: [x for s in dd.values() for x in s]  # noqa: E731
        allrows = flat(pos) + flat(neg)
        pooled = ols(allrows)
        w = wbi[(m, c)]
        assert pooled is not None and abs(pooled[0] - w["a"]) < 1e-6 and abs(pooled[1] - w["b"]) < 1e-6, (m, c)
        ep, en = ols(flat(pos)), ols(flat(neg))
        row = {"model": m, "condition": c, "n_pos": len(flat(pos)), "n_neg": len(flat(neg)), "n_clusters": len(ids)}
        if ep is None or en is None:
            row["note"] = "stratum rank deficient; no fit"
            rows.append(row)
            continue
        rng = random.Random(SEED)
        boots = []
        for _ in range(n_boot):
            pick = [ids[rng.randrange(len(ids))] for _ in ids]
            bp = ols([x for s in pick for x in pos.get(s, [])])
            bn = ols([x for s in pick for x in neg.get(s, [])])
            if bp is not None and bn is not None:
                boots.append((bp, bn))
        for i, p in enumerate("ab"):
            vals = {"pos": ep[i], "neg": en[i], "diff": en[i] - ep[i]}
            series = {"pos": [b[0][i] for b in boots], "neg": [b[1][i] for b in boots],
                      "diff": [b[1][i] - b[0][i] for b in boots]}
            for k, v in vals.items():
                row[f"{p}_{k}"] = v
                if len(boots) >= 20:
                    sv = sorted(series[k])
                    row[f"{p}_{k}_lo"], row[f"{p}_{k}_hi"] = pct(sv, 0.025), pct(sv, 0.975)
        row["n_boot_valid"] = len(boots)
        row["note"] = "few clusters: CIs unreliable" if len(ids) < 10 else ""
        rows.append(row)
    assert rows
    return rows


def h5_macros(rows):
    out = {}
    fit = [r for r in rows if "a_pos" in r]
    out["hFiveNCells"] = len(fit)
    for p in "ab":
        P = p.upper()
        out[f"hFivePar{P}NegLtPosN"] = sum(r[f"{p}_diff"] < 0 for r in fit)
        out[f"hFivePar{P}DiffCiBelowZeroN"] = sum(r.get(f"{p}_diff_hi") is not None and r[f"{p}_diff_hi"] < 0 for r in fit)
        out[f"hFivePar{P}DiffCiAboveZeroN"] = sum(r.get(f"{p}_diff_lo") is not None and r[f"{p}_diff_lo"] > 0 for r in fit)
        for k in ("pos", "neg"):
            K = k.title()
            out[f"hFivePar{P}{K}LtOneN"] = sum(r[f"{p}_{k}"] < 1 for r in fit)
            out[f"hFivePar{P}{K}CiBelowOneN"] = sum(r.get(f"{p}_{k}_hi") is not None and r[f"{p}_{k}_hi"] < 1 for r in fit)
            out[f"hFivePar{P}{K}CiAboveOneN"] = sum(r.get(f"{p}_{k}_lo") is not None and r[f"{p}_{k}_lo"] > 1 for r in fit)
        for r in fit:
            for k in ("pos", "neg", "diff"):
                base = f"hFivePar{P}{k.title()}{model_tag(r['model'])}{r['condition']}"
                out[base + "Pt"] = tex_num(r[f"{p}_{k}"], 2)
                if r.get(f"{p}_{k}_lo") is not None:
                    out[base + "Lo"] = bound_str(r[f"{p}_{k}_lo"])
                    out[base + "Hi"] = bound_str(r[f"{p}_{k}_hi"])
    return out


# L3 (C4): CI bounds near 1
NEAR_FIELDS = ["table", "model", "condition", "param", "bound", "value", "printed_2dp", "printed_rule", "wholly_above_one",
               "wholly_below_one"]


def near_one_bounds(tables):
    """Every a or b CI bound within 0.005 of 1 (prints as 1.00 at two decimals), with the value that
    the 3-decimal rule prints and whether it counts as 'interval wholly above (below) 1'."""
    rows = []
    for name, tab in tables:
        for r in tab:
            if r["a"] is None:
                continue
            for p in "ab":
                for bnd in ("lo", "hi"):
                    v = r.get(f"{p}_{bnd}")
                    if v is None or abs(float(v) - 1.0) >= 0.005:
                        continue
                    rows.append({"table": name, "model": r["model"], "condition": r["condition"], "param": p,
                                 "bound": bnd, "value": f"{float(v):.6f}", "printed_2dp": tex_num(v, 2),
                                 "printed_rule": bound_str(v), "wholly_above_one": int(bnd == "lo" and float(v) > 1),
                                 "wholly_below_one": int(bnd == "hi" and float(v) < 1)})
    return rows


def near_macros(rows):
    out = {}
    for name in ("Main", "Own"):
        rs = [r for r in rows if r["table"] == name]
        out[f"nearOneBounds{name}N"] = len(rs)
        out[f"nearOneBounds{name}AboveN"] = sum(r["wholly_above_one"] for r in rs)
        out[f"nearOneBounds{name}BelowN"] = sum(r["wholly_below_one"] for r in rs)
    return out


# ---------------------------------------------------------------- macros and tex
# ---------------------------------------------------------------- C4 re-verification additions (N1, N2, N4, N5, N10, M5, M6); additive
# N1: implied likelihood ratio of a model's own (pretest, posttest) answer pair over the published LR.
#   published LR = case target lr_point: the printed point when one is published, the geometric mean
#   (log-space midpoint) of the printed range when only a range is published, the exact value from
#   sensitivity and specificity otherwise (analysis/build_targets.py). Pooled over parsed answers in conditions
#   V N B S, no A5 calibration case, no boundary 0/100 answer (odds undefined, clamped by the scorer).
IMPLIED_LR_FIELDS = ["model", "result", "n_parsed", "n_boundary_excluded", "n_used", "median_ratio",
                     "median_log_ratio", "pct_ratio_above_one"]
HIGH_PRIOR_PCT = 15.0  # M6/N4: a scenario is "high prior" when its published prior is at or above 15%
NEG_SCEN_FIELDS = ["scenario", "set", "published_prior_pct", "high_prior", "n_answers", "median_posttest_pct",
                   "target_point_pct", "band_lo_pct", "band_hi_pct", "median_below_target",
                   "median_below_band_lo", "pct_answers_below_target"]
CLIN_ITEM_FIELDS = ["model", "case_id", "model_post_median", "clin_post_median", "model_post_band_err",
                    "clin_post_band_err", "model_closer", "tie", "model_not_closer"]
EPS = 1e-9


def implied_lr_table(scored, cases):
    """Per model and result (pos, neg): median of implied LR / published LR (N1)."""
    pub = {cid: c["target"]["lr_point"] for cid, c in cases.items()}
    rows = []
    for m in sorted({r["model"] for r in scored}):
        for res in ("pos", "neg"):
            rs = [r for r in main_rows(scored) if r["model"] == m and r["result"] == res
                  and r["condition"] in FIT_CONDS and r["parse_ok"] == "1"]
            used = [r for r in rs if r["boundary_answer"] != "1"]
            ratios = [float(r["implied_lr"]) / pub[r["case_id"]] for r in used]
            assert all(x > 0 and math.isfinite(x) for x in ratios)
            rows.append({"model": m, "result": res, "n_parsed": len(rs), "n_boundary_excluded": len(rs) - len(used),
                         "n_used": len(used), "median_ratio": statistics.median(ratios),
                         "median_log_ratio": statistics.median(math.log(x) for x in ratios),
                         "pct_ratio_above_one": 100.0 * sum(x > 1 for x in ratios) / len(ratios)})
    return rows


def implied_lr_macros(rows):
    out = {}
    for res, R in (("pos", "Pos"), ("neg", "Neg")):
        rr = [r for r in rows if r["result"] == res]
        for r in rr:
            t = model_tag(r["model"])
            out[f"impLr{R}{t}Med"] = tex_num(r["median_ratio"], 2)
            out[f"impLr{R}{t}PctAbove"] = tex_num(r["pct_ratio_above_one"], 0)
            out[f"impLr{R}{t}NUsed"] = r["n_used"]
        out[f"impLr{R}MedMin"] = tex_num(min(r["median_ratio"] for r in rr), 2)
        out[f"impLr{R}MedMax"] = tex_num(max(r["median_ratio"] for r in rr), 2)
        out[f"impLr{R}PctAboveMin"] = tex_num(min(r["pct_ratio_above_one"] for r in rr), 0)
        out[f"impLr{R}PctAboveMax"] = tex_num(max(r["pct_ratio_above_one"] for r in rr), 0)
        out[f"impLr{R}NParsed"] = sum(r["n_parsed"] for r in rr)
        out[f"impLr{R}NBoundaryExcl"] = sum(r["n_boundary_excluded"] for r in rr)
        out[f"impLr{R}NUsed"] = sum(r["n_used"] for r in rr)
    return out


def h5_b_macros(h5):
    """N1: within-result b. Counts duplicate the hFive counts under explicit names; GPT-4o b after positive results."""
    fit = [r for r in h5 if "b_pos" in r]
    out = {"splitBPosCiAboveOneN": sum(r.get("b_pos_lo") is not None and r["b_pos_lo"] > 1 for r in fit),
           "splitBNegCiAboveOneN": sum(r.get("b_neg_lo") is not None and r["b_neg_lo"] > 1 for r in fit),
           "splitNCells": len(fit)}
    g = [r for r in fit if r["model"] == "openai/gpt-4o"]
    if g:
        out["splitBPosGptFourOPtMin"] = tex_num(min(r["b_pos"] for r in g), 2)
        out["splitBPosGptFourOPtMax"] = tex_num(max(r["b_pos"] for r in g), 2)
        out["splitBPosGptFourOCiBelowOneN"] = sum(r["b_pos_hi"] < 1 for r in g)
        out["splitBPosGptFourONCells"] = len(g)
    return out


def neg_scenario_table(scored, cases):
    """M6/N4: per scenario, pooled median posttest answer after the NEGATIVE test (V N B S, parsed, no A5)
    against the exact Bayes target, with the printed/exact band used for scoring."""
    rows = []
    for cid, c in sorted(cases.items()):
        if c["result"] != "neg" or c.get("calibration") or is_calibration(c):
            continue
        v = [float(r["posttest_stated"]) for r in main_rows(scored) if r["case_id"] == cid
             and r["condition"] in FIT_CONDS and r["parse_ok"] == "1"]
        if not v:
            continue
        t = c["target"]
        tgt, (lo, hi) = t["posttest_point"] * 100.0, [x * 100.0 for x in t["posttest_band_primary"]]
        med = statistics.median(v)
        prior = t["pretest_point"] * 100.0
        rows.append({"scenario": c["scenario"], "set": c["set"], "published_prior_pct": prior,
                     "high_prior": int(prior >= HIGH_PRIOR_PCT - EPS), "n_answers": len(v),
                     "median_posttest_pct": med, "target_point_pct": tgt, "band_lo_pct": lo, "band_hi_pct": hi,
                     "median_below_target": int(med < tgt - EPS), "median_below_band_lo": int(med < lo - EPS),
                     "pct_answers_below_target": 100.0 * sum(x < tgt - EPS for x in v) / len(v)})
    return rows


def neg_scenario_macros(rows):
    hp = [r for r in rows if r["high_prior"]]
    out = {"negHighPriorThresholdPct": tex_num(HIGH_PRIOR_PCT, 0), "negNScenarios": len(rows),
           "negNHighPrior": len(hp), "negNHighPriorBelowTarget": sum(r["median_below_target"] for r in hp),
           "negNHighPriorBelowBandLo": sum(r["median_below_band_lo"] for r in hp),
           "negNLowPriorBelowTarget": sum(r["median_below_target"] for r in rows if not r["high_prior"])}
    for r in rows:
        if r["scenario"] == "B3_pe_high":
            out["negPeHighTargetPct"] = tex_num(r["target_point_pct"], 1)
            out["negPeHighMedianPct"] = tex_num(r["median_posttest_pct"], 1)
            out["negPeHighPriorPct"] = tex_num(r["published_prior_pct"], 1)
            out["negPeHighPctBelowTarget"] = tex_num(r["pct_answers_below_target"], 0)
    return out


def uti_pos_macros(scored, cases):
    """M5/N5: urinary positive item. Target and band from the case file; pooled model median posttest."""
    c = cases.get("A4_uti_pos")
    if c is None:
        return {}
    t = c["target"]
    v = [float(r["posttest_stated"]) for r in main_rows(scored) if r["case_id"] == "A4_uti_pos"
         and r["condition"] in FIT_CONDS and r["parse_ok"] == "1"]
    if not v:
        return {}
    # published range printed in Morgan 2021 Table 3 is 0-1% (case file source string); stored low end is 0.1%
    import re
    src = [s["quantity"] for s in c["sources"] if "pretest" in s["quantity"]]
    mt_ = re.match(r"pretest (\d+)-(\d+)%", src[0])
    assert mt_, src
    printed_lo, printed_hi = int(mt_.group(1)), int(mt_.group(2))
    lo_s, hi_s = c["target_inputs"]["pretest"]["range"]
    assert abs(hi_s * 100 - printed_hi) < EPS
    return {"utiPosTargetPct": tex_num(t["posttest_point"] * 100, 2),
            "utiPosBayesBandLoPct": tex_num(t["posttest_band_bayes"][0] * 100, 2),
            "utiPosBayesBandHiPct": tex_num(t["posttest_band_bayes"][1] * 100, 2),
            "utiPosPrintedBandHiPct": tex_num(t["posttest_band_primary"][1] * 100, 1),
            "utiPosPriorPct": tex_num(t["pretest_point"] * 100, 2),
            "utiPosPriorPrintedLoPct": printed_lo, "utiPosPriorPrintedHiPct": printed_hi,
            "utiPosPriorFloorPct": tex_num(lo_s * 100, 1),
            "utiPosMedianPostPct": tex_num(statistics.median(v), 1),
            "utiPosNAnswers": len(v),
            "utiPosPctAboveBayesBand": tex_num(100.0 * sum(x > t["posttest_band_bayes"][1] * 100 + EPS for x in v) / len(v), 0)}


def e_minus_n_macros(con):
    """N10: E minus N (H4) cells, result 'all', whose 95% CI excludes zero (lo > 0 or hi < 0, strict)."""
    rows = [r for r in con if r["result"] == "all" and r["contrast"] == "EvsN"]
    ex = [r for r in rows if r["ci_lo"] != "" and (float(r["ci_lo"]) > 0 or float(r["ci_hi"]) < 0)]
    return {"hFourNCellsCiExclZero": len(ex), "hFourNCellsTotal": len(rows)}


def clinician_item_rows(clin, models):
    """N2: condition V, items A1 to A4 (A5 excluded), model median-answer band error vs clinician median band error."""
    rows = []
    for r in clin:
        if r["condition"] != "V" or r["scenario"] == "A5_hypothetical":
            continue
        me, ce = float(r["model_post_band_err"]), float(r["clin_post_band_err"])
        rows.append({"model": r["model"], "case_id": r["case_id"], "model_post_median": float(r["model_post_median"]),
                     "clin_post_median": float(r["clin_post_median"]), "model_post_band_err": me,
                     "clin_post_band_err": ce, "model_closer": int(me < ce - EPS),
                     "tie": int(abs(me - ce) <= EPS), "model_not_closer": int(me >= ce - EPS)})
    assert {r["model"] for r in rows} <= set(models)
    return sorted(rows, key=lambda r: (r["model"], r["case_id"]))


def clinician_item_macros(rows):
    out = {}
    for m in sorted({r["model"] for r in rows}):
        t = model_tag(m)
        mine = [r for r in rows if r["model"] == m]
        out[f"clinItemV{t}NItems"] = len(mine)
        out[f"clinItemV{t}NNotCloser"] = sum(r["model_not_closer"] for r in mine)
        out[f"clinItemV{t}NTie"] = sum(r["tie"] for r in mine)
        for cid, tag in (("A1_pneumonia_pos", "PneumoniaPos"), ("A4_uti_pos", "UtiPos")):
            for r in mine:
                if r["case_id"] == cid:
                    out[f"clinItemV{tag}{t}ModelMed"] = tex_num(r["model_post_median"], 1)
                    out[f"clinItemV{tag}{t}ModelErr"] = tex_num(r["model_post_band_err"], 1)
                    out[f"clinItemV{tag}{t}ClinErr"] = tex_num(r["clin_post_band_err"], 1)
                    out[f"clinItemV{tag}{t}Closer"] = r["model_closer"]
                    out[f"clinItemV{tag}{t}Tie"] = r["tie"]
    for cid, tag in (("A1_pneumonia_pos", "PneumoniaPos"), ("A4_uti_pos", "UtiPos")):
        rr = [r for r in rows if r["case_id"] == cid]
        if rr:
            out[f"clinItemV{tag}NModelsNotCloser"] = sum(r["model_not_closer"] for r in rr)
            out[f"clinItemV{tag}NModels"] = len(rr)
            out[f"clinItemV{tag}ClinMed"] = tex_num(rr[0]["clin_post_median"], 1)
    return out


def build_macros(err, con, wbt, scored, wbt_own=None, cal=None, over=None, rank=None, h4=None, fb=None,
                 sp=None, extra=None):
    mac = {}

    def put(name, val):
        assert name.isalpha(), f"macro name not letters only: {name}"
        assert name not in mac, f"duplicate macro {name}"
        assert str(val) != "", name
        mac[name] = val

    models = sorted({r["model"] for r in scored})
    tags = {m: model_tag(m) for m in models}
    assert len(set(tags.values())) == len(tags), "model tag collision"
    put("resNScored", len(scored))
    put("resNCollected", len(scored))
    put("resNParsed", sum(1 for r in scored if r["parse_ok"] == "1"))
    put("resNMainScored", len(main_rows(scored)))
    put("resNCalibration", len(scored) - len(main_rows(scored)))
    nfail = sum(1 for r in scored if r["parse_ok"] != "1")
    put("resNParseFail", nfail)
    put("resParseFailPct", tex_num(100.0 * nfail / len(scored), 1))
    put("resNModels", len(models))
    for m in models:
        rs = [r for r in scored if r["model"] == m]
        f = sum(1 for r in rs if r["parse_ok"] != "1")
        put(f"pf{tags[m]}N", len(rs))
        put(f"pf{tags[m]}Fail", f)
        put(f"pf{tags[m]}Pct", tex_num(100.0 * f / len(rs), 1))
    for r in err:
        if r["result"] != "all" or not r["n_ok"]:
            continue
        base = f"Set{r['set']}{tags[r['model']]}{r['condition']}"
        put(f"erb{base}Med", tex_num(r["band_pp_med"], 1))
        put(f"erb{base}Qlo", tex_num(r["band_pp_q1"], 1))
        put(f"erb{base}Qhi", tex_num(r["band_pp_q3"], 1))
        put(f"elo{base}Med", tex_num(r["abs_lod_med"]))
        put(f"elo{base}Qlo", tex_num(r["abs_lod_q1"]))
        put(f"elo{base}Qhi", tex_num(r["abs_lod_q3"]))
        put(f"enf{base}", r["n_parse_fail"])
    resname = {"all": "All", "pos": "Pos", "neg": "Neg"}
    for r in con:
        if r["n_cases"] and "median_diff" in r:
            base = f"ctr{METRIC_TAG[r['metric']]}{r['contrast']}Set{r['set']}{resname[r['result']]}" \
                   f"{tags[r['model']]}"
            put(base + "Med", tex_num(r["median_diff"], 1 if r["metric"] == "band_err_pp" else 2))
            if "ci_lo" in r:
                d = 1 if r["metric"] == "band_err_pp" else 2
                put(base + "Lo", tex_num(r["ci_lo"], d))
                put(base + "Hi", tex_num(r["ci_hi"], d))
            put(base + "N", r["n_cases"])
    variants = {"Main": (wbt, ""), "Nob": (wbt, "_noboundary")}
    if wbt_own:
        variants.update({"Own": (wbt_own, ""), "OwnNob": (wbt_own, "_noboundary")})
    for var, (tab, suf) in variants.items():
        for r in tab:
            if r["a"] is None:
                continue
            for p in "abc":
                base = f"wb{p.upper()}{var}{tags[r['model']]}{r['condition']}"
                if r[p + suf] is None:
                    continue
                put(base + "Pt", tex_num(r[p + suf]))
                if r[f"{p}_lo{suf}"] is not None:
                    put(base + "Lo", tex_num(r[f"{p}_lo{suf}"]))
                    put(base + "Hi", tex_num(r[f"{p}_hi{suf}"]))
    # ---- counts the writer needs (all computed here, never typed)
    cell_sets = {"Main": ("published, boundary answers kept", wbt, ""),
                 "Nob": ("published, boundary answers dropped", wbt, "_noboundary")}
    if wbt_own:
        cell_sets["Own"] = ("own stated prior", wbt_own, "")
    for var, (_, tab, suf) in cell_sets.items():
        cells = [r for r in tab if r["a"] is not None and r["b" + suf] is not None]
        put(f"sumFitCells{var}N", len(cells))
        put(f"sumBGtOne{var}N", sum(float(r["b" + suf]) > 1 for r in cells))
        put(f"sumBCiAboveOne{var}N", sum(r["b_lo" + suf] is not None and float(r["b_lo" + suf]) > 1 for r in cells))
        put(f"sumACiBelowOne{var}N", sum(r["a_hi" + suf] is not None and float(r["a_hi" + suf]) < 1 for r in cells))
        put(f"sumACiContainsOne{var}N", sum(r["a_lo" + suf] is not None and float(r["a_lo" + suf]) <= 1 <= float(r["a_hi" + suf])
                                           for r in cells))
        by_model = {}
        for r in cells:
            if r["a_hi" + suf] is not None and float(r["a_hi" + suf]) < 1:
                by_model.setdefault(r["model"], []).append(r["condition"])
        order = {c: i for i, c in enumerate(FIT_CONDS)}
        put(f"sumACiBelowOne{var}List", "; ".join(
            f"{display_name(m)} ({', '.join(sorted(cs, key=order.get))})" for m, cs in sorted(by_model.items())) or "none")
    for var, tab in (("Main", wbt), ("Own", wbt_own or [])):
        if not tab:
            continue
        cells = [r for r in tab if r["a"] is not None]
        put(f"sumFlagged{var}N", sum(int(r["boundary_flag"]) for r in cells))
        # descriptive companion to the pre-specified flag: point estimate moves by 0.1 or more
        put(f"sumShiftTenth{var}N", sum(1 for r in cells if r["a_noboundary"] is not None and
                                     (abs(r["a"] - r["a_noboundary"]) >= 0.1 or abs(r["b"] - r["b_noboundary"]) >= 0.1)))
        for r in cells:
            put(f"wbFlag{var}{tags[r['model']]}{r['condition']}", int(r["boundary_flag"]))
    if cal:
        for r in cal:
            base = f"cal{r['result'].title()}{tags[r['model']]}"
            put(base + "N", r["n_ok"])
            if r["n_ok"]:
                put(base + "Med", tex_num(r["stated_post_med_pp"], 2))
                put(base + "Lo", tex_num(r["stated_post_min_pp"], 2))
                put(base + "Hi", tex_num(r["stated_post_max_pp"], 2))
                put(base + "Diff", tex_num(r["diff_med_pp"], 2))
        for res in ("pos", "neg"):
            put(f"cal{res.title()}Target", tex_num([r for r in cal if r["result"] == res][0]["target_post_pp"], 2))
    if fb:
        put("pfNEmptyMax", sum(r["n_empty_at_max_tokens"] for r in fb))
        put("pfNOtherUnparsed", sum(r["n_other_unparsed"] for r in fb))
        put("pfNCallError", sum(r["n_call_error"] for r in fb))
        for r in fb:
            put(f"pf{tags[r['model']]}Collected", r["n_collected"])
            put(f"pf{tags[r['model']]}Parsed", r["n_parsed"])
            put(f"pf{tags[r['model']]}EmptyMax", r["n_empty_at_max_tokens"])
            put(f"pf{tags[r['model']]}OtherUnparsed", r["n_other_unparsed"])
    if h4:
        for st, v in h4.items():
            put(f"hFourPooledMedSet{st}", tex_num(v["pooled_median"], 2))
            put(f"hFourNPairsSet{st}", v["n_pairs"])
            put(f"hFourNCellsPositiveSet{st}", v["n_cells_positive"])
            put(f"hFourNCellsSet{st}", v["n_cells"])
            put(f"hFourNCasesPerModelSet{st}", v["n_cases_max"])  # M3: pairs are model-by-case means
    if over:
        for r in over:
            put(f"over{tags[r['model']]}MedSignedLod", tex_num(r["median_signed_lod"], 2))
            put(f"over{tags[r['model']]}PctAboveTarget", tex_num(r["pct_above_target"], 0))
            put(f"over{tags[r['model']]}PctAboveBand", tex_num(r["pct_above_band"], 0))
        for k, name in (("median_signed_lod", "MedSignedLod"), ("pct_above_target", "PctAboveTarget")):
            vs = [r[k] for r in over]
            d = 2 if k == "median_signed_lod" else 0
            put(f"over{name}Min", tex_num(min(vs), d))
            put(f"over{name}Max", tex_num(max(vs), d))
    if rank:
        for r in rank:
            put(f"rank{tags[r['model']]}AbsLodMed", tex_num(r["abs_lod_med"], 2))
            put(f"rank{tags[r['model']]}AbsLodRank", r["rank_lod"])
            put(f"rank{tags[r['model']]}BandMed", tex_num(r["band_pp_med"], 1))
            put(f"rank{tags[r['model']]}BandRank", r["rank_band"])
        put("rankAbsLodOrder", " < ".join(display_name(r["model"]) for r in rank))
    cc = contrast_counts(con)
    put("contrastNTotal", cc["total"])
    put("contrastNExclZero", cc["excl"])
    put("contrastNExclZeroAgainst", cc["excl_against"])
    put("contrastNExclZeroWith", cc["excl_with"])
    put("contrastNExclZeroHFour", cc["excl_h4"])
    if sp:
        put("statedPriorNScen", len(sp))
        put("statedPriorNExceed", sum(r["median_exceeds_published"] for r in sp))
        uti = [r for r in sp if r["scenario"] == "A4_uti"]
        assert len(uti) == 1 and uti[0]["set"] == "A"
        put("statedPriorUtiMedPct", tex_num(uti[0]["median_stated_pretest_pct"], 1))
        put("statedPriorUtiPubPct", tex_num(uti[0]["published_prior_pct"], 2))
    for k, v in sorted((extra or {}).items()):  # C4 additions: only ever added, never replace
        put(k, v)
    put("tabNoteBTargets", "For scenarios B2 and B3 the exact posttest target is a single point, so band error is "
                           "the distance from that point.")
    put("tabNoteAFive", "The numeric calibration case A5 is excluded from every cell and reported separately.")
    return mac


def write_macros(path, mac):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% GENERATED by analysis/make_tables.py from results/derived/. Do not edit.\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


def bound_str(x, d=2):
    """L3: a CI bound that would print as 1.00 gets extra decimals (3, then 4, 5) so that 'interval wholly
    above 1' counts can be reproduced from the printed table. All other values print with d decimals."""
    s = tex_num(x, d)
    if s == "1.00":
        for dd in (3, 4, 5):
            s = tex_num(x, dd)
            if not s.rstrip("0").endswith("1."):  # stop once the string separates the bound from 1
                break
    return s


def ci(pt, lo, hi, d=2):
    if pt is None:
        return "--"
    s = tex_num(pt, d)
    return s + (f" [{bound_str(lo, d)}, {bound_str(hi, d)}]" if lo is not None else "")


def write_error_tex(path, err, models):
    cells = {(r["model"], r["set"], r["condition"]): r for r in err if r["result"] == "all"}
    lines = ["% GENERATED by analysis/make_tables.py. Median band error in pp (Q1 to Q3). A5 excluded.",
             "% Caption note: for B2 and B3 the posttest target is a single point (see macro \\tabNoteBTargets).",
             "\\begin{tabular}{llccccc}", "\\toprule",
             "Model & Set & " + " & ".join(CONDS) + " \\\\", "\\midrule"]
    for m in models:
        for st in ("A", "B"):
            out = []
            for c in CONDS:
                r = cells.get((m, st, c))
                if r is None or not r["n_ok"]:
                    out.append("--")
                else:
                    out.append(f"{tex_num(r['band_pp_med'], 1)} ({tex_num(r['band_pp_q1'], 1)}"
                               f"--{tex_num(r['band_pp_q3'], 1)})")
            lines.append(f"{display_name(m)} & {st} & " + " & ".join(out) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_wbayes_tex(path, wbt, label="Published-prior fit"):
    lines = [f"% GENERATED by analysis/make_tables.py. {label}; 95% scenario-bootstrap CI.",
             "% Last two columns: same fit with clamped 0/100 answers excluded. A dagger after the",
             "% condition marks a cell flagged by the 0/100 sensitivity rule (analysis/README.md).",
             "\\begin{tabular}{llcccc}", "\\toprule",
             "Model & Cond. & $a$ [95\\% CI] & $b$ [95\\% CI] & $a$ (no bdry) & $b$ (no bdry) \\\\",
             "\\midrule"]
    for r in wbt:
        cond = r["condition"] + ("$^\\dagger$" if r.get("boundary_flag") else "")
        lines.append(" & ".join([display_name(r["model"]).replace("_", "\\_"), cond,
                                 ci(r["a"], r["a_lo"], r["a_hi"]), ci(r["b"], r["b_lo"], r["b_hi"]),
                                 ci(r["a_noboundary"], None, None), ci(r["b_noboundary"], None, None)])
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def check_macros_file(path):
    """Every \\newcommand has a non-empty, brace-balanced value and a letters-only name."""
    import re
    n = 0
    for line in open(path):
        if line.startswith("%") or not line.strip():
            continue
        m = re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{(.+)\}\n?", line)
        assert m, f"bad macro line: {line!r}"
        assert m.group(2).count("{") == m.group(2).count("}"), line
        n += 1
    assert n > 0, "no macros written"
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", default=os.path.join(ROOT, "results", "derived"),
                    help="directory holding scored.csv, summary_model_condition.csv, wbayes.csv")
    ap.add_argument("--out-derived", default=None, help="default: same as --derived")
    ap.add_argument("--generated", default=os.path.join(ROOT, "paper", "generated"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw"),
                    help="raw dir, read only, used by the H3 sensitivity (skipped when the record is absent)")
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    a = ap.parse_args(argv)
    out_d = a.out_derived or a.derived
    syn = read_provenance(a.derived)
    guard_output(out_d, syn)
    guard_output(a.generated, syn)
    guard_pilot(a.generated, read_run_set(a.derived))
    scored = read_csv(os.path.join(a.derived, "scored.csv"))
    summary = read_csv(os.path.join(a.derived, "summary_model_condition.csv"))
    wb = read_csv(os.path.join(a.derived, "wbayes.csv"))
    assert scored and summary and wb, "empty input table"
    models = sorted({r["model"] for r in scored})
    err = error_table(scored, summary)
    con = contrast_table(scored, a.n_boot)
    wbt = wbayes_table(wb)
    wbt_own = wbayes_table(wb, "own")
    cal = calibration_table(scored)
    over = overestimation_table(scored)
    rank = ranking_table(scored)
    h4 = h4_pooled(scored)
    fb = [{k: (v if k == "model" else int(v)) for k, v in r.items()}
          for r in read_csv(os.path.join(a.derived, "parse_failure_breakdown.csv"))]
    assert sum(r["n_collected"] for r in fb) == len(scored)
    assert len(con) == len(models) * 2 * 3 * sum(len(c[3]) for c in CONTRASTS)
    write_csv(os.path.join(out_d, "table_error_by_condition.csv"), err, ERR_FIELDS)
    write_csv(os.path.join(out_d, "table_contrasts.csv"), con, CON_FIELDS)
    write_csv(os.path.join(out_d, "table_wbayes.csv"), wbt, WB_FIELDS)
    write_csv(os.path.join(out_d, "table_wbayes_own.csv"), wbt_own, WB_FIELDS)
    write_csv(os.path.join(out_d, "table_calibration.csv"), cal, CAL_FIELDS)
    write_csv(os.path.join(out_d, "table_overestimation.csv"), over, OVER_FIELDS)
    write_csv(os.path.join(out_d, "table_model_ranking.csv"), rank, RANK_FIELDS)
    sp = stated_prior_table(scored, load_cases(a.cases))
    write_csv(os.path.join(out_d, "table_stated_prior.csv"), sp, STATED_PRIOR_FIELDS)
    extra = {}
    cases = load_cases(a.cases)
    rb = response_breakdown(scored)
    for k, name in (("total", "resNResponses"), ("cal", "resNCalibAnswers"), ("cal_parsed", "resNCalibParsed"),
                    ("noncal", "resNNonCalibAnswers"), ("noncal_parsed", "resNNonCalibParsed"),
                    ("noncal_unparsed", "resNNonCalibUnparsed")):
        extra[name] = rb[k]
    extra["resNonCalibUnparsedPct"] = tex_num(100.0 * rb["noncal_unparsed"] / rb["noncal"], 1)
    sens = unparsed_sensitivity(scored, cases, [os.path.join(a.raw, "*.jsonl")], a.n_boot)
    if sens is not None:
        write_csv(os.path.join(out_d, "table_unparsed_sensitivity.csv"), sens["changed"], SENS_FIELDS)
        extra.update(sensitivity_macros(sens))
    mid = midpoint_error_table(scored, cases)
    write_csv(os.path.join(out_d, "table_midpoint_error_setA.csv"), mid, MID_FIELDS)
    for r in mid:
        t = model_tag(r["model"])
        extra[f"secMid{t}N"] = r["n"]
        extra[f"secMid{t}Mae"] = tex_num(r["mid_mae_pp"], 1)
        extra[f"secMid{t}Med"] = tex_num(r["mid_median_abs_pp"], 1)
        extra[f"secBand{t}Mae"] = tex_num(r["band_mae_pp"], 1)
    h5 = h5_split_fits(scored, cases, wbt, a.n_boot)
    write_csv(os.path.join(out_d, "table_wbayes_by_result.csv"), h5, H5_FIELDS)
    extra.update(h5_macros(h5))
    il = implied_lr_table(scored, cases)
    write_csv(os.path.join(out_d, "table_implied_lr.csv"), il, IMPLIED_LR_FIELDS)
    extra.update(implied_lr_macros(il))
    extra.update(h5_b_macros(h5))
    ns = neg_scenario_table(scored, cases)
    write_csv(os.path.join(out_d, "table_neg_scenario_median.csv"), ns, NEG_SCEN_FIELDS)
    extra.update(neg_scenario_macros(ns))
    extra.update(uti_pos_macros(scored, cases))
    extra.update(e_minus_n_macros(con))
    clin_path = os.path.join(a.derived, "clinician_compare.csv")
    if os.path.exists(clin_path):
        ci_rows = clinician_item_rows(read_csv(clin_path), models)
        write_csv(os.path.join(out_d, "table_clinician_item_compare.csv"), ci_rows, CLIN_ITEM_FIELDS)
        extra.update(clinician_item_macros(ci_rows))
    near = near_one_bounds([("Main", wbt), ("Own", wbt_own)])
    write_csv(os.path.join(out_d, "table_near_one_bounds.csv"), near, NEAR_FIELDS)
    extra.update(near_macros(near))
    mac = build_macros(err, con, wbt, scored, wbt_own, cal, over, rank, h4, fb, sp, extra)
    mpath = os.path.join(a.generated, "results_macros.tex")
    write_macros(mpath, mac)
    write_error_tex(os.path.join(a.generated, "table_error.tex"), err, models)
    write_wbayes_tex(os.path.join(a.generated, "table_wbayes.tex"), wbt)
    write_wbayes_tex(os.path.join(a.generated, "table_wbayes_own.tex"), wbt_own,
                     "Own-stated-prior fit (prior = the model's own stated pretest)")
    n = check_macros_file(mpath)
    assert n == len(mac)
    print(f"wrote {len(err)} error rows, {len(con)} contrast rows, {len(wbt)} fit rows, {n} macros"
          f"{' (SYNTHETIC, test dir)' if syn else ''}")
    return mac


if __name__ == "__main__":
    main()
