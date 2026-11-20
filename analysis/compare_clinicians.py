"""Compare model stated pretest and posttest with published clinician estimates (Set A). Protocol outcome 2.

Clinician values come from data/baselines/morgan2021_clinicians.json (Morgan 2021, Results text p750 for
pooled medians, Table 3 p752 for resident / attending / NP-PA medians). The pooled median is used as the
clinician value; no pooled value is computed by us. Band error of a clinician median uses exactly the
rule in score.band_error (0 inside the band, else distance in pp to the nearest edge).

Modes
  python analysis/compare_clinicians.py --clinicians-only
      Reads ONLY the baseline JSON and data/cases. Writes results/derived/clinician_band_errors.csv and
      paper/generated/clinician_macros.tex (source-derived macros). Touches no raw records.
  python analysis/compare_clinicians.py [--set paper|pilot] [--run-ids a,b] [--raw GLOB] ...
      Also reads raw records (same run-id allow-list, latest-attempt dedup, dry-run drop and synthetic
      firewall as score.py) and writes clinician_compare.csv, clinician_compare_summary.csv and the full
      paper/generated/clinician_macros.tex.

Comparison definitions (per model x condition x Set A case; seeds pooled)
  model_post_median     median stated posttest over parse-ok records
  model_post_band_err   band error of that MEDIAN against posttest_band_primary (like for like with the
                        clinician median); model_median_record_band_err is the median of per-record errors
  diff_post             model_post_band_err minus clinician band error (negative = model closer to the band)
  the same for pretest against pretest_range
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, add_run_args, default_out, guard_output, guard_pilot, is_synthetic,  # noqa: E402
                    load_cases, load_raw, read_csv, resolve_run_ids, select_records, write_csv)
from make_tables import check_macros_file, model_tag, tex_num  # noqa: E402
from score import band_error, score_record  # noqa: E402

BASELINE = os.path.join(ROOT, "data", "baselines", "morgan2021_clinicians.json")
GROUPS = ["pooled", "residents", "attendings", "np_pa"]
CLIN_FIELDS = ["case_id", "scenario", "set", "result", "quantity", "group", "n", "median", "iqr_low",
               "iqr_high", "band_lo_pp", "band_hi_pp", "err_signed_pp", "err_abs_pp", "median_inside_band",
               "source"]
CMP_FIELDS = ["model", "condition", "case_id", "scenario", "result", "n_ok", "n_total",
              "clin_source", "model_post_median", "clin_post_median", "band_lo_pp", "band_hi_pp",
              "model_post_band_err", "model_median_record_band_err", "clin_post_band_err", "diff_post",
              "model_pre_median", "clin_pre_median", "pre_lo_pp", "pre_hi_pp", "model_pre_err",
              "clin_pre_err", "diff_pre"]
SUM_FIELDS = ["model", "condition", "n_cases", "n_post_model_closer", "n_post_tie", "n_post_clin_closer",
              "diff_post_median", "model_post_err_median", "clin_post_err_median", "n_pre_cases",
              "diff_pre_median", "model_pre_err_median", "clin_pre_err_median"]
# Pins: clinician pooled-median band errors (pp) recomputed by hand from Morgan Results text p750 and the
# case-file bands. A change in a case file or the baseline makes the script fail loudly.
PIN_POST = {"A1_pneumonia_pos": 30.0, "A1_pneumonia_neg": 31.0, "A2_breast_pos": 41.0,
            "A2_breast_neg": 4.95, "A3_cardiac_pos": 59.0, "A3_cardiac_neg": 2.5,
            "A4_uti_pos": 71.7, "A4_uti_neg": 4.89, "A5_hypothetical_neg": 5.0}
PIN_PRE = {"A1_pneumonia": 38.0, "A2_breast": 4.7, "A3_cardiac": 5.6, "A4_uti": 19.0}


def load_baseline(path=BASELINE):
    with open(path) as f:
        b = json.load(f)
    assert b["scenarios"], "empty baseline"
    return b


def clinician_values(baseline):
    """Return {(case_id, quantity): {group: dict}} using the baseline's case_ids mapping."""
    out = {}
    for sc in baseline["scenarios"]:
        for key, item in sc["items"].items():
            if item is None:
                continue
            res = item["applies_to_result"]
            targets = ([sc["case_ids"]["pos"], sc["case_ids"]["neg"]] if key == "pretest"
                       else [sc["case_ids"][res]])
            groups = {"pooled": item["pooled"], **item["groups"]}
            for cid in targets:
                out[(cid, "pretest" if key == "pretest" else "posttest")] = {
                    g: groups[g] for g in GROUPS}
    return out


def clinician_rows(baseline, cases):
    vals = clinician_values(baseline)
    rows = []
    for (cid, q), groups in sorted(vals.items()):
        assert cid in cases, f"baseline names unknown case {cid}"
        c = cases[cid]
        t = c["target"]
        band = t["posttest_band_primary"] if q == "posttest" else t["pretest_range"]
        for g in GROUPS:
            v = groups[g]
            eb = band_error(v["median"], band)
            rows.append({"case_id": cid, "scenario": c["scenario"], "set": c["set"], "result": c["result"],
                         "quantity": q, "group": g, "n": v["n"], "median": v["median"],
                         "iqr_low": v["iqr_low"], "iqr_high": v["iqr_high"],
                         "band_lo_pp": band[0] * 100, "band_hi_pp": band[1] * 100,
                         "err_signed_pp": round(eb, 6), "err_abs_pp": round(abs(eb), 6),
                         "median_inside_band": int(eb == 0),
                         "source": ("Results text p750" if g == "pooled" else "Table 3 p752")})
    return rows


def check_clinician_rows(rows):
    assert len(rows) % len(GROUPS) == 0
    by = {(r["case_id"], r["quantity"], r["group"]): r for r in rows}
    for cid, e in PIN_POST.items():
        got = by[(cid, "posttest", "pooled")]["err_abs_pp"]
        assert abs(got - e) < 0.01, f"clinician posttest band error {cid}: {got} != pinned {e}"
    # A5 positive: exact posterior 1.9627%, pooled median 95
    a5 = by[("A5_hypothetical_pos", "posttest", "pooled")]["err_abs_pp"]
    assert abs(a5 - (95 - 1.9627085)) < 1e-4, a5
    for sc, e in PIN_PRE.items():
        got = by[(sc + "_pos", "pretest", "pooled")]["err_abs_pp"]
        assert abs(got - e) < 0.01, f"clinician pretest error {sc}: {got} != pinned {e}"
    for r in rows:
        assert r["err_abs_pp"] >= 0 and not math.isnan(r["err_abs_pp"])
        assert r["iqr_low"] <= r["median"] <= r["iqr_high"], r


def tag_scn(s):
    return "".join(ch for ch in s.split("_", 1)[1].title() if ch.isalpha())


def clinician_macros(rows):
    mac = {}

    def put(k, v):
        assert k.isalpha(), k
        assert k not in mac, k
        mac[k] = v
    for r in rows:
        if r["group"] != "pooled":
            continue
        s = tag_scn(r["scenario"])
        if r["quantity"] == "pretest":
            if r["result"] == "pos":  # pretest is identical for pos and neg cases
                put(f"clinMedPre{s}", tex_num(r["median"], 1))
                put(f"clinErrPre{s}", tex_num(r["err_abs_pp"], 1))
        else:
            R = r["result"].title()
            put(f"clinMedPost{s}{R}", tex_num(r["median"], 1))
            put(f"clinErrPost{s}{R}", tex_num(r["err_abs_pp"], 1))
            put(f"clinInside{s}{R}", str(r["median_inside_band"]))
    return mac


def aggregate(cmp_rows):
    """Per model x condition over Set A cases excluding A5 (A5 is condition V only; kept in the case csv)."""
    out = []
    keys = sorted({(r["model"], r["condition"]) for r in cmp_rows})
    for m, c in keys:
        rs = [r for r in cmp_rows if r["model"] == m and r["condition"] == c
              and not r["scenario"].startswith("A5") and r["diff_post"] != ""]
        if not rs:
            continue
        d = [r["diff_post"] for r in rs]
        pre = [r for r in rs if r["diff_pre"] != ""]
        out.append({"model": m, "condition": c, "n_cases": len(rs),
                    "n_post_model_closer": sum(x < 0 for x in d), "n_post_tie": sum(x == 0 for x in d),
                    "n_post_clin_closer": sum(x > 0 for x in d),
                    "diff_post_median": statistics.median(d),
                    "model_post_err_median": statistics.median([r["model_post_band_err"] for r in rs]),
                    "clin_post_err_median": statistics.median([r["clin_post_band_err"] for r in rs]),
                    "n_pre_cases": len(pre),
                    "diff_pre_median": statistics.median([r["diff_pre"] for r in pre]) if pre else "",
                    "model_pre_err_median": statistics.median([r["model_pre_err"] for r in pre]) if pre else "",
                    "clin_pre_err_median": statistics.median([r["clin_pre_err"] for r in pre]) if pre else ""})
    return out


def compare_rows(final_recs, cases, vals):
    groups = {}
    for rec in final_recs:
        c = cases[rec["case_id"]]
        if c["set"] != "A":
            continue
        groups.setdefault((rec["model"], rec["condition"], rec["case_id"]), []).append(score_record(rec, c))
    rows = []
    for (m, cond, cid), srs in sorted(groups.items()):
        c = cases[cid]
        t = c["target"]
        ok = [r for r in srs if r["parse_ok"]]
        post_clin = vals[(cid, "posttest")]["pooled"]["median"]
        pb = t["posttest_band_primary"]
        row = {"model": m, "condition": cond, "case_id": cid, "scenario": c["scenario"],
               "result": c["result"], "n_ok": len(ok), "n_total": len(srs), "clin_source": "pooled",
               "band_lo_pp": pb[0] * 100, "band_hi_pp": pb[1] * 100, "clin_post_median": post_clin,
               "clin_post_band_err": abs(band_error(post_clin, pb))}
        pre_clin = vals.get((cid, "pretest"))
        if pre_clin:
            pr = t["pretest_range"]
            row.update(pre_lo_pp=pr[0] * 100, pre_hi_pp=pr[1] * 100,
                       clin_pre_median=pre_clin["pooled"]["median"],
                       clin_pre_err=abs(band_error(pre_clin["pooled"]["median"], pr)))
        if ok:  # parse failures stay out of medians (as in score.py); they are counted in n_total
            mp = statistics.median([r["posttest_stated"] for r in ok])
            row.update(model_post_median=mp, model_post_band_err=abs(band_error(mp, pb)),
                       model_median_record_band_err=statistics.median([r["band_err_pp"] for r in ok]),
                       diff_post=abs(band_error(mp, pb)) - row["clin_post_band_err"])
            if pre_clin:
                mq = statistics.median([r["pretest_stated"] for r in ok])
                row.update(model_pre_median=mq, model_pre_err=abs(band_error(mq, pr)),
                           diff_pre=abs(band_error(mq, pr)) - row["clin_pre_err"])
        rows.append(row)
    return rows


def comparison_macros(summ):
    mac = {}
    for r in summ:
        base = f"{model_tag(r['model'])}{r['condition']}"
        for k, v, d in (("DiffPostMed", r["diff_post_median"], 1), ("ModelPostErrMed", r["model_post_err_median"], 1),
                        ("ClinPostErrMed", r["clin_post_err_median"], 1)):
            mac[f"clinSetA{k}{base}"] = tex_num(v, d)
        for k, v in (("NCases", r["n_cases"]), ("NCloser", r["n_post_model_closer"]),
                     ("NTie", r["n_post_tie"]), ("NClinCloser", r["n_post_clin_closer"])):
            mac[f"clinSetA{k}{base}"] = str(v)
        if r["n_pre_cases"]:
            mac[f"clinSetADiffPreMed{base}"] = tex_num(r["diff_pre_median"], 1)
    assert all(k.isalpha() for k in mac)
    return mac


def write_macros(path, mac, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"% GENERATED by analysis/compare_clinicians.py ({header}). Do not edit.\n")
        for k in sorted(mac):
            f.write(f"\\newcommand{{\\{k}}}{{{mac[k]}}}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinicians-only", action="store_true",
                    help="write only the source-derived clinician band errors; reads no raw records")
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--generated", default=None,
                    help="default paper/generated (paper set); pilot runs write no macros unless given")
    add_run_args(ap)
    a = ap.parse_args(argv)
    cases = load_cases(a.cases)
    base = load_baseline(a.baseline)
    clin = clinician_rows(base, cases)
    check_clinician_rows(clin)
    cmac = clinician_macros(clin)
    gen_default = os.path.join(ROOT, "paper", "generated")

    if a.clinicians_only:
        out = a.out or os.path.join(ROOT, "results", "derived")
        gen = a.generated or gen_default
        write_csv(os.path.join(out, "clinician_band_errors.csv"), clin, CLIN_FIELDS)
        write_macros(os.path.join(gen, "clinician_macros.tex"), cmac, "clinicians-only, no raw records")
        check_macros_file(os.path.join(gen, "clinician_macros.tex"))
        print(f"wrote {len(clin)} clinician rows, {len(cmac)} macros (no raw records read)")
        for r in clin:
            if r["group"] == "pooled":
                print(f"  {r['case_id']:22s} {r['quantity']:9s} pooled median {r['median']:>5} "
                      f"band {r['band_lo_pp']:.4g}-{r['band_hi_pp']:.4g}  band error {r['err_abs_pp']:.2f} pp")
        return clin, cmac

    allowed, run_set, cfg = resolve_run_ids(a)
    out = default_out(a.out, run_set)
    gen = a.generated or (None if run_set == "pilot" else gen_default)
    recs, n_dry = load_raw(a.raw)
    syn = is_synthetic(recs, a.raw)  # judged on all loaded records before filtering, as in score.py
    guard_output(out, syn)
    if gen:
        guard_output(gen, syn)
        guard_pilot(gen, run_set)
    recs, rep = select_records(recs, allowed, cfg)
    assert recs, f"no records selected for run ids {allowed}"
    unknown = sorted({r["case_id"] for r in recs if r["case_id"] not in cases})
    assert not unknown, f"raw records reference unknown case ids: {unknown}"
    vals = clinician_values(base)
    rows = compare_rows(recs, cases, vals)
    setA_final = [r for r in recs if cases[r["case_id"]]["set"] == "A"]
    assert rows, "no Set A records among the selected runs"
    assert sum(r["n_total"] for r in rows) == len(setA_final), "records lost in grouping"
    for r in rows:
        assert (r["case_id"], "posttest") in vals, f"no clinician posttest for {r['case_id']}"
        if r["n_ok"]:
            for k in ("model_post_median", "model_post_band_err", "diff_post", "clin_post_band_err"):
                assert r[k] != "" and not math.isnan(r[k]), (k, r)
    summ = aggregate(rows)
    mac = dict(cmac)
    mac.update(comparison_macros(summ))
    write_csv(os.path.join(out, "clinician_band_errors.csv"), clin, CLIN_FIELDS)
    write_csv(os.path.join(out, "clinician_compare.csv"), rows, CMP_FIELDS)
    write_csv(os.path.join(out, "clinician_compare_summary.csv"), summ, SUM_FIELDS)
    if gen:
        mp = os.path.join(gen, "clinician_macros.tex")
        write_macros(mp, mac, "model comparison + clinician values")
        assert check_macros_file(mp) == len(mac)
    n_fail = sum(r["n_total"] - r["n_ok"] for r in rows)
    print(f"{len(rows)} model x condition x case rows ({n_fail} records with parse failures kept in n_total, "
          f"excluded from medians); {len(summ)} summary rows; run set {run_set} {allowed}"
          f"{' (SYNTHETIC, test dir)' if syn else ''}")
    return rows, summ, mac


if __name__ == "__main__":
    main()
