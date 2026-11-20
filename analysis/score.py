"""Score raw model records against the case targets. See analysis/README.md.

Usage: python analysis/score.py [--raw 'results/raw/*.jsonl'] [--cases data/cases] [--out results/derived]
"""
import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ATTEMPT_FIELDS, ROOT, add_run_args, default_out, guard_output, resolve_run_ids, select_records, is_boundary, is_calibration, is_synthetic, load_cases, load_raw,  # noqa: E402
                    logit_clipped, write_provenance, write_csv)

SCORED_FIELDS = ["run_id", "case_id", "scenario", "set", "result", "model", "condition", "seed",
                 "case_status", "calibration", "parse_status", "failure_kind", "parse_ok", "pretest_stated", "posttest_stated",
                 "band_err_pp", "band_err_signed_pp", "bayes_band_err_pp", "logodds_err",
                 "abs_logodds_err", "pretest_err_pp", "implied_lr", "coherence_err",
                 "boundary_answer", "posttest_target_pp"]
METRICS = ["band_err_pp", "bayes_band_err_pp", "logodds_err", "abs_logodds_err", "pretest_err_pp",
           "coherence_err"]


def band_error(x_pp, band):
    """Signed distance in pp to the band (0 inside, negative below, positive above). band in [0,1]."""
    lo, hi = band[0] * 100.0, band[1] * 100.0
    if x_pp < lo:
        return x_pp - lo
    if x_pp > hi:
        return x_pp - hi
    return 0.0


def failure_kind(rec):
    """Descriptive label for a failed record (never used to re-parse): empty reply that used the whole
    completion budget, a call error, or any other unparsed answer."""
    if rec.get("parse_status") == "call_error":
        return "call_error"
    dp = rec.get("decoding_params")
    mx = dp.get("max_tokens") if isinstance(dp, dict) else None
    ct = rec.get("completion_tokens")
    if not str(rec.get("response") or "").strip() and mx and ct is not None and float(ct) >= float(mx):
        return "empty_at_max_tokens"
    return "other_unparsed"


def score_record(rec, case):
    t = case["target"]
    row = {"run_id": rec.get("run_id"), "case_id": rec["case_id"], "scenario": case["scenario"],
           "set": case["set"], "result": case["result"], "model": rec["model"],
           "condition": rec["condition"], "seed": rec.get("seed"),
           "case_status": case.get("status"), "calibration": int(is_calibration(case)),
           "parse_status": rec.get("parse_status"),
           "posttest_target_pp": t["posttest_point"] * 100.0}
    pa = rec.get("parsed_answer")
    ok = rec.get("parse_status") == "ok" and isinstance(pa, dict) \
        and pa.get("pretest") is not None and pa.get("posttest") is not None
    row["parse_ok"] = int(ok)
    if not ok:  # kept in scored.csv with blank metrics, counted in the summaries
        row["failure_kind"] = failure_kind(rec)
        return row
    pre, post = float(pa["pretest"]), float(pa["posttest"])
    row["pretest_stated"], row["posttest_stated"] = pre, post
    # literal 0/100 (or beyond clip range) answers are clamped to the clip bound; flag them
    row["boundary_answer"] = int(is_boundary(pre / 100.0) or is_boundary(post / 100.0))
    sb = band_error(post, t["posttest_band_primary"])
    row["band_err_signed_pp"] = sb
    row["band_err_pp"] = abs(sb)
    row["bayes_band_err_pp"] = abs(band_error(post, t["posttest_band_bayes"]))
    le = logit_clipped(post / 100.0) - logit_clipped(t["posttest_point"])
    row["logodds_err"] = le
    row["abs_logodds_err"] = abs(le)
    row["pretest_err_pp"] = abs(band_error(pre, t["pretest_range"]))
    implied = logit_clipped(post / 100.0) - logit_clipped(pre / 100.0)  # log implied LR
    row["implied_lr"] = math.exp(implied)
    row["coherence_err"] = abs(implied - math.log(t["lr_point"]))
    return row


def quartiles(vals):
    if len(vals) < 2:
        return (vals[0], vals[0]) if vals else (None, None)
    q = statistics.quantiles(vals, n=4, method="inclusive")
    return q[0], q[2]


def summarize(rows):
    """Cross-condition summary. Calibration cases (A5, condition V only) are excluded;
    they are reported separately in table_calibration.csv by make_tables.py."""
    groups = {}
    for r in rows:
        if r["calibration"]:
            continue
        groups.setdefault((r["model"], r["condition"], r["set"], r["result"]), []).append(r)
    out = []
    for (m, c, s, res), rs in sorted(groups.items()):
        n_total = len(rs)
        n_fail = sum(1 for r in rs if not r["parse_ok"])
        for metric in METRICS:
            vals = [r[metric] for r in rs if r["parse_ok"]]
            q1, q3 = quartiles(vals)
            out.append({"model": m, "condition": c, "set": s, "result": res, "metric": metric,
                        "n": len(vals), "median": statistics.median(vals) if vals else None,
                        "q1": q1, "q3": q3, "n_total": n_total, "n_parse_fail": n_fail})
    return out


def failure_breakdown(rows):
    """Per model: collected, parsed, and failures by kind (call_error, empty_at_max_tokens, other_unparsed)."""
    out = []
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]
        kinds = {k: sum(1 for r in rs if r.get("failure_kind") == k)
                 for k in ("call_error", "empty_at_max_tokens", "other_unparsed")}
        out.append({"model": m, "n_collected": len(rs), "n_parsed": sum(r["parse_ok"] for r in rs),
                    "n_call_error": kinds["call_error"],
                    "n_empty_at_max_tokens": kinds["empty_at_max_tokens"],
                    "n_other_unparsed": kinds["other_unparsed"]})
    for o in out:
        assert o["n_collected"] == o["n_parsed"] + o["n_call_error"] + o["n_empty_at_max_tokens"] + o["n_other_unparsed"], o
    return out


def parse_failure_table(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["condition"]), []).append(r)
    out = []
    for (m, c), rs in sorted(groups.items()):
        fails = [r for r in rs if not r["parse_ok"]]
        reasons = {}
        for r in fails:
            reasons[r["parse_status"]] = reasons.get(r["parse_status"], 0) + 1
        out.append({"model": m, "condition": c, "n_total": len(rs), "n_parse_fail": len(fails),
                    "fail_rate": len(fails) / len(rs),
                    "reasons": ";".join(f"{k}={v}" for k, v in sorted(reasons.items()))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--out", default=None,
                    help="default results/derived (paper) or results/derived/pilot (--set pilot)")
    add_run_args(ap)
    a = ap.parse_args(argv)
    allowed, run_set, cfg = resolve_run_ids(a)
    a.out = default_out(a.out, run_set)
    cases = load_cases(a.cases)
    rawrep = {}
    recs, n_dry = load_raw(a.raw, rawrep)
    syn = is_synthetic(recs, a.raw)  # judged on ALL loaded records, before any filtering
    guard_output(a.out, syn)  # synthetic inputs never reach results/derived
    recs, rep = select_records(recs, allowed, cfg)
    assert recs, f"no records selected for run ids {allowed}; present: see attempts report above"
    unknown = sorted({r["case_id"] for r in recs if r["case_id"] not in cases})
    assert not unknown, f"raw records reference unknown case ids: {unknown}"
    rows = [score_record(r, cases[r["case_id"]]) for r in recs]
    # Key assertions: nothing dropped silently, dry runs never scored.
    assert len(rows) == len(recs)
    assert all(r["parse_ok"] in (0, 1) for r in rows)
    assert all(r["band_err_pp"] is not None for r in rows if r["parse_ok"])
    assert all(r["band_err_pp"] >= 0 and r["bayes_band_err_pp"] >= 0 for r in rows if r["parse_ok"])
    write_csv(os.path.join(a.out, "scored.csv"), rows, SCORED_FIELDS)
    write_provenance(a.out, "score", a.raw, syn, len(recs), run_set, allowed,
                     rawrep["n_malformed"])
    write_csv(os.path.join(a.out, "attempts.csv"), rep["attempts"], ATTEMPT_FIELDS)
    summ = summarize(rows)
    write_csv(os.path.join(a.out, "summary_model_condition.csv"), summ,
              ["model", "condition", "set", "result", "metric", "n", "median", "q1", "q3",
               "n_total", "n_parse_fail"])
    pf = parse_failure_table(rows)
    assert sum(p["n_total"] for p in pf) == len(rows)
    write_csv(os.path.join(a.out, "parse_failures.csv"), pf,
              ["model", "condition", "n_total", "n_parse_fail", "fail_rate", "reasons"])
    fb = failure_breakdown(rows)
    write_csv(os.path.join(a.out, "parse_failure_breakdown.csv"), fb,
              ["model", "n_collected", "n_parsed", "n_call_error", "n_empty_at_max_tokens", "n_other_unparsed"])
    n_fail = sum(1 for r in rows if not r["parse_ok"])
    print(f"scored {len(rows)} records ({n_fail} parse failures kept and flagged; "
          f"{n_dry} dry_run records excluded)")
    print(f"run set {run_set} {allowed}: {rep['n_selected_records']} records selected, "
          f"{rep['n_superseded']} superseded by a later attempt, "
          f"{sum(rep['not_selected'].values())} in known run ids not selected {rep['not_selected']}, "
          f"{sum(rep['unknown'].values())} in UNKNOWN run ids not scored")
    for att in rep["attempts"]:
        if att["n_superseded"] or att["n_final_call_error"]:
            print(f"  attempts {att['run_id']} {att['model']}: {att['n_records']} records, "
                  f"{att['n_keys']} keys, {att['n_superseded']} superseded, "
                  f"{att['n_final_call_error']} keys still call_error")
    return rows


if __name__ == "__main__":
    main()
