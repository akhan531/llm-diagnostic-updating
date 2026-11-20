"""Round-2 committee analyses. All post hoc except the
pre-specified primary metric (item 8). Reads ONLY results/raw/*.jsonl (same allow-list and latest-attempt
dedup as score.py) and data/. Never writes to results/raw/. Writes:
  results/derived/r2_floor_shares.csv        item 2: share of answers at the floor and at the ceiling
  results/derived/r2_ratio_robust.csv        item 2: over-reaction ratio r under floor/ceiling handling
  results/derived/r2_counts.csv              item 3: one consistent set of parse counts
  results/derived/r2_threshold_alt.csv       item 3: 35% threshold share vs primary and alternative PE targets
  results/derived/r2_scenario_level.csv      item 5: scenario-level counts of r > 1
  results/derived/r2_holm.csv                item 5: raw and Holm-adjusted p over the post hoc family
  results/derived/r2_update_only.csv         item 6: r on the update only (own stated pretest, blind pretest)
  results/derived/r2_primary_metric.csv      item 8: pre-specified primary metric by model and condition
  paper/generated/round2_macros.tex          \\rt... macros (letters only)
  paper/generated/table_primary.tex          item 8 compact table
The rules are written up in. Table 4 (table_ext.tex, item 10) is rebuilt by
analysis/ext_analysis.py.

Usage: python analysis/round2_analysis.py [--raw GLOB] [--cases DIR] [--out DIR] [--generated DIR] [--n-boot 50]
"""
import argparse
import math
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import committee_fixes as cf  # noqa: E402
import committee_fixes2 as cf2  # noqa: E402
import ext_analysis as ext  # noqa: E402
from common import (ROOT, add_run_args, display_name, guard_output, guard_pilot, is_calibration,  # noqa: E402
                    load_cases, load_raw, logit_clipped, resolve_run_ids, select_records, write_csv)
from make_tables import model_tag, tex_num  # noqa: E402
from score import score_record  # noqa: E402

CONDS = cf.FIT_CONDS            # V, N, B, S: the conditions pooled in r
ORIGINAL_SCENARIOS = frozenset({"A1_pneumonia", "A2_breast", "A3_cardiac", "A4_uti", "A5_hypothetical", "B1_pe_low",
                                "B2_pe_mod", "B3_pe_high", "B4_aas", "B5_appy_ct", "B6_appy_us"})  # by id, not by glob
SUB1 = 0.01                     # scenarios whose NEGATIVE target is below 1% are "floor scenarios"


def tag(m):
    return "All" if m == "ALL" else model_tag(m)


def holm(ps):
    """Holm step-down adjusted p-values (same order as input)."""
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    adj = [None] * n
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (n - rank) * ps[i]))
        adj[i] = run
    return adj


def clipped_logit(p, lo, hi):
    return logit_clipped(min(max(p, lo), hi))


# ---------------------------------------------------------------- Tobit-style cell mean
def _log_phi(z):
    """log of the standard normal CDF, stable in both tails."""
    if z > -30:
        return math.log(max(0.5 * math.erfc(-z / math.sqrt(2)), 1e-300))
    return -0.5 * z * z - math.log(-z) - 0.5 * math.log(2 * math.pi)


def _golden(f, a, b, it=60):
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(it):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def tobit_mean(ps, lo=0.01, hi=0.99):
    """Mean of the latent log-odds under a normal model with answers <= lo and >= hi treated as censored
    (left or right) at logit(lo), logit(hi). Maximum likelihood in the convex parametrisation
    (delta = mu/sigma, gamma = 1/sigma) by coordinate descent with golden-section steps. If every answer is
    censored on one side, the latent mean is not identified; the bound itself is returned, which is the
    value least favourable to a large separation. Returns (mu, n_censored)."""
    L, H = math.log(lo / (1 - lo)), math.log(hi / (1 - hi))
    obs, nl, nr = [], 0, 0
    for p in ps:
        if p <= lo:
            nl += 1
        elif p >= hi:
            nr += 1
        else:
            obs.append(math.log(p / (1 - p)))
    nc = nl + nr
    if nc == 0:
        return statistics.mean(obs), 0
    if not obs:
        if nl and not nr:
            return L, nc
        if nr and not nl:
            return H, nc
        return (nl * L + nr * H) / nc, nc

    def nll(delta, gamma):
        v = sum(-math.log(gamma) + 0.5 * (gamma * y - delta) ** 2 for y in obs)
        v += -nl * _log_phi(gamma * L - delta) if nl else 0.0
        v += -nr * _log_phi(delta - gamma * H) if nr else 0.0
        return v

    delta, gamma = 0.0, 0.3
    for _ in range(40):
        delta = _golden(lambda d: nll(d, gamma), -40.0, 40.0)
        gamma = _golden(lambda g: nll(delta, g), 1e-3, 30.0)
    return delta / gamma, nc


# ---------------------------------------------------------------- ratio cells with pluggable per-cell estimator
def lr_table(cases):
    lr = {}
    for c in cases.values():
        if not is_calibration(c):
            lr.setdefault(c["scenario"], {})[c["result"]] = c["target"]
    return lr


def ratio_cells_by(rows, cases, cell_fn, scenarios=None):
    """One row per model x scenario (conditions V, N, B, S pooled): r = (est pos - est neg) / (log LR+ - log LR-),
    where est is cell_fn(list of rows) -> number (and optionally a censored count)."""
    lr = lr_table(cases)
    g = {}
    for r in rows:
        if r["condition"] in CONDS and (scenarios is None or r["scenario"] in scenarios):
            g.setdefault((r["model"], r["scenario"]), {"pos": [], "neg": []})[r["result"]].append(r)
    out = []
    for (m, s), d in sorted(g.items()):
        if not d["pos"] or not d["neg"]:
            continue
        dx = math.log(lr[s]["pos"]["lr_point"]) - math.log(lr[s]["neg"]["lr_point"])
        ep, en = cell_fn(d["pos"]), cell_fn(d["neg"])
        nc = (ep[1] + en[1]) if isinstance(ep, tuple) else 0
        ep = ep[0] if isinstance(ep, tuple) else ep
        en = en[0] if isinstance(en, tuple) else en
        out.append({"model": m, "scenario": s, "dy": ep - en, "log_dor": dx, "ratio": (ep - en) / dx, "n_censored": nc})
    return out


def summarize_ratio(cells, models):
    out = []
    for m in models + ["ALL"]:
        cs = [c for c in cells if m == "ALL" or c["model"] == m]
        if m == "ALL":
            # scenario level: the median across models of r, one value per scenario
            by = {}
            for c in cs:
                by.setdefault(c["scenario"], []).append(c["ratio"])
            vals = [statistics.median(v) for _, v in sorted(by.items())]
        else:
            vals = [c["ratio"] for c in cs]
        lo, hi, cov = cf.median_ci(vals)
        ngt = sum(v > 1 for v in vals)
        out.append({"model": m, "n_scen": len(vals), "median": statistics.median(vals), "med_lo": lo, "med_hi": hi,
                    "n_gt1": ngt, "sign_p": cf.sign_test_p(ngt, sum(v != 1 for v in vals)),
                    "n_cells_gt1": sum(c["ratio"] > 1 for c in cs), "n_cells": len(cs),
                    "n_censored": sum(c["n_censored"] for c in cs)})
    return out


def mean_logit(rs, lo, hi):
    return statistics.mean(clipped_logit(r["post"], lo, hi) for r in rs)


VARIANTS = {   # name: (cell estimator factory, scenario filter flag, label)
    "primary": ("answers clipped to [0.001%, 99.999%] (the pipeline rule)", "mean", (1e-5, 1 - 1e-5), False),
    "cliphalf": ("answers clipped to [0.5%, 99.5%]", "mean", (0.005, 0.995), False),
    "clipone": ("answers clipped to [1%, 99%]", "mean", (0.01, 0.99), False),
    "median": ("median log-odds per cell (rank-based, unchanged by clipping unless the median itself is censored)",
               "median", (1e-5, 1 - 1e-5), False),
    "tobit": ("Tobit-style: answers <= 1% or >= 99% censored at 1% and 99%", "tobit", (0.01, 0.99), False),
    "exclsubone": ("scenarios whose negative target is below 1% excluded", "mean", (1e-5, 1 - 1e-5), True),
    "exclsubone_clipone": ("sub-1% scenarios excluded and answers clipped to [1%, 99%]", "mean", (0.01, 0.99), True),
}
VWORD = {"primary": "Primary", "cliphalf": "ClipHalf", "clipone": "ClipOne", "median": "Median", "tobit": "Tobit",
         "exclsubone": "ExclSubOne", "exclsubone_clipone": "ExclSubOneClipOne"}


def make_cell_fn(kind, clip):
    lo, hi = clip
    if kind == "mean":
        return lambda rs: mean_logit(rs, lo, hi)
    if kind == "median":
        return lambda rs: statistics.median(clipped_logit(r["post"], lo, hi) for r in rs)
    return lambda rs: tobit_mean([r["post"] for r in rs], lo, hi)


# ---------------------------------------------------------------- item 2: floor and ceiling shares
def floor_shares(rows, models):
    out = []
    for m in models + ["ALL"]:
        rs = [r for r in rows if m == "ALL" or r["model"] == m]
        neg = [r for r in rs if r["result"] == "neg"]
        pos = [r for r in rs if r["result"] == "pos"]
        n = len(rs)
        row = {"model": m, "n": n, "n_neg": len(neg), "n_pos": len(pos),
               "n_zero": sum(r["post"] <= 0 for r in rs),
               "n_le_half": sum(r["post"] <= 0.005 for r in rs),
               "n_le_one": sum(r["post"] <= 0.01 for r in rs),
               "n_ge_99": sum(r["post"] >= 0.99 for r in rs),
               "n_hundred": sum(r["post"] >= 1.0 for r in rs),
               "n_neg_le_one": sum(r["post"] <= 0.01 for r in neg),
               "n_pos_ge_99": sum(r["post"] >= 0.99 for r in pos)}
        for k in ("zero", "le_half", "le_one", "ge_99", "hundred"):
            row["pct_" + k] = 100 * row["n_" + k] / n
        row["pct_neg_le_one"] = 100 * row["n_neg_le_one"] / len(neg)
        row["pct_pos_ge_99"] = 100 * row["n_pos_ge_99"] / len(pos)
        out.append(row)
    return out


# ---------------------------------------------------------------- item 3: counts
def count_set(recs, cases, cfg, ext_ids):
    """Parse counts from raw, latest attempt per key, calibration excluded (except where stated)."""
    main_id = cfg["paper_run_ids"][0]
    main_final = ext.final_of(recs, main_id, cfg)
    rows = []

    def add(group, arm, final, fields, scen_only=False):
        scen = ext.scenario_targets(cases)
        for r in final:
            if scen_only:   # E1 case ids are scenario ids; calibration (A5) is not a scenario
                if r["case_id"] not in scen:
                    continue
            elif is_calibration(cases[r["case_id"]]):
                continue
            ok = ext.parse_ok(r, fields)
            rows.append({"group": group, "arm": arm, "model": r["model"], "ok": ok,
                         "trunc": (not ok) and ext.length_stopped(r)})
    add("main", "main1", main_final, ["pretest", "posttest"])
    n_cal = sum(is_calibration(cases[r["case_id"]]) for r in main_final)
    spec = [("E1", ext_ids["E1"], ["pretest"], True), ("E2", ext_ids["E2"], ["sensitivity", "specificity", "pretest",
                                                                             "posttest"], False),
            ("E3", ext_ids["E3"], ["pretest", "posttest"], False), ("E4", ext_ids["E4"], ["pretest", "posttest"], False)]
    for arm, rid, fields, so in spec:
        add("ext", arm, ext.final_of(recs, rid, cfg), fields, so)
    return rows, len(main_final), n_cal


# ---------------------------------------------------------------- item 6: update-only r
def update_cells(rows, cases, lo, hi, blind=None):
    """r on the update only. Per answer: logit(posttest) - logit(pretest), where the pretest is the model's own
    stated pretest (blind=None) or the model's blind pretest for that scenario and condition (blind dict,
    logit-mean over samples). Cell estimate = mean of updates; r = (pos - neg) / (log LR+ - log LR-)."""
    lr = lr_table(cases)
    g = {}
    for r in rows:
        if r["condition"] not in CONDS:
            continue
        if blind is None:
            u = clipped_logit(r["post"], lo, hi) - clipped_logit(r["pre"], lo, hi)
        else:
            bk = (r["model"], r["scenario"], ext.BLIND_COND_OF[r["condition"]])
            if bk not in blind:
                continue
            u = clipped_logit(r["post"], lo, hi) - blind[bk]
        g.setdefault((r["model"], r["scenario"]), {"pos": [], "neg": []})[r["result"]].append(u)
    out = []
    for (m, s), d in sorted(g.items()):
        if not d["pos"] or not d["neg"]:
            continue
        dx = math.log(lr[s]["pos"]["lr_point"]) - math.log(lr[s]["neg"]["lr_point"])
        dy = statistics.mean(d["pos"]) - statistics.mean(d["neg"])
        out.append({"model": m, "scenario": s, "dy": dy, "log_dor": dx, "ratio": dy / dx, "n_censored": 0})
    return out


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "derived"))
    ap.add_argument("--generated", default=os.path.join(ROOT, "paper", "generated"))
    ap.add_argument("--n-boot", type=int, default=50, help="bootstrap size for the fits inside the reused ext "
                    "analysis; only p-values and counts are used here, so this is small")
    add_run_args(ap)
    a = ap.parse_args(argv)
    allowed, run_set, cfg = resolve_run_ids(a)
    cases, final, rows, syn = cf.load_rows(a.raw, a.cases, allowed, cfg)
    # restrict explicitly to the original scenario ids (new cases B7, B8 are analysed as an extension)
    cases = {k: v for k, v in cases.items() if v["scenario"] in ORIGINAL_SCENARIOS}
    rows = [r for r in rows if r["scenario"] in ORIGINAL_SCENARIOS]
    final = [r for r in final if r["case_id"] in cases]
    assert len({v["scenario"] for v in cases.values()} - {"A5_hypothetical"}) == 10 or a.run_ids
    for d in (a.out, a.generated):
        guard_output(d, syn)
    guard_pilot(a.generated, run_set)
    models = cf.models_of(rows)
    mac = {}
    recs, _ = load_raw(a.raw)

    # ------------------------------------------------ item 2: floor and ceiling
    fl = floor_shares(rows, models)
    for f in fl:
        t = "rtFloor" + tag(f["model"])
        mac[t + "N"] = str(f["n"])
        mac[t + "PctZero"] = f"{f['pct_zero']:.1f}"
        mac[t + "PctAtOrBelowHalf"] = f"{f['pct_le_half']:.1f}"
        mac[t + "PctAtOrBelowOne"] = f"{f['pct_le_one']:.1f}"
        mac[t + "PctNegAtOrBelowOne"] = f"{f['pct_neg_le_one']:.1f}"
        mac[t + "PctAtOrAboveNinetyNine"] = f"{f['pct_ge_99']:.1f}"
        mac[t + "PctPosAtOrAboveNinetyNine"] = f"{f['pct_pos_ge_99']:.1f}"
        mac[t + "PctHundred"] = f"{f['pct_hundred']:.1f}"
        mac[t + "NZero"] = str(f["n_zero"])
        mac[t + "NHundred"] = str(f["n_hundred"])
    ms = [f for f in fl if f["model"] != "ALL"]
    mac["rtFloorPctAtOrBelowOneMin"] = f"{min(f['pct_le_one'] for f in ms):.1f}"
    mac["rtFloorPctAtOrBelowOneMax"] = f"{max(f['pct_le_one'] for f in ms):.1f}"
    write_csv(os.path.join(a.out, "r2_floor_shares.csv"), fl,
              ["model", "n", "n_neg", "n_pos", "n_zero", "n_le_half", "n_le_one", "n_ge_99", "n_hundred", "n_neg_le_one",
               "n_pos_ge_99", "pct_zero", "pct_le_half", "pct_le_one", "pct_ge_99", "pct_hundred", "pct_neg_le_one",
               "pct_pos_ge_99"])

    lr = lr_table(cases)
    in_data = {r["scenario"] for r in rows}   # scenarios with answers in the analysed run (new case files without data are ignored)
    floor_scen = {s for s, d in lr.items() if s in in_data and d["neg"]["posttest_point"] < SUB1}
    keep_scen = in_data - floor_scen
    mac["rtRobustNFloorScenarios"] = str(len(floor_scen))
    mac["rtRobustNKeptScenarios"] = str(len(keep_scen))
    rob, cells_all = [], {}
    for vname, (label, kind, clip, excl) in VARIANTS.items():
        cells = ratio_cells_by(rows, cases, make_cell_fn(kind, clip), keep_scen if excl else None)
        cells_all[vname] = cells
        summ = summarize_ratio(cells, models)
        for s in summ:
            rob.append(dict(s, variant=vname, label=label))
            t = "rtRobust" + VWORD[vname] + tag(s["model"])
            mac[t + "Med"] = tex_num(s["median"])
            mac[t + "MedLo"] = tex_num(s["med_lo"]) if s["med_lo"] is not None else "n/a"
            mac[t + "MedHi"] = tex_num(s["med_hi"]) if s["med_hi"] is not None else "n/a"
            mac[t + "NGtOne"] = str(s["n_gt1"])
            mac[t + "NScen"] = str(s["n_scen"])
        pm = [s for s in summ if s["model"] != "ALL"]
        t = "rtRobust" + VWORD[vname]
        mac[t + "MedMin"] = tex_num(min(s["median"] for s in pm))
        mac[t + "MedMax"] = tex_num(max(s["median"] for s in pm))
        mac[t + "NModelsMedGtOne"] = str(sum(s["median"] > 1 for s in pm))
        mac[t + "NGtOneMin"] = str(min(s["n_gt1"] for s in pm))
        mac[t + "NGtOneMax"] = str(max(s["n_gt1"] for s in pm))
        mac[t + "NCellsGtOne"] = str(sum(s["n_cells_gt1"] for s in pm))
        mac[t + "NCells"] = str(sum(s["n_cells"] for s in pm))
        if kind == "tobit":
            mac[t + "NCensoredAnswersInCells"] = str(sum(s["n_censored"] for s in pm))
    # the primary variant must reproduce the pre-existing F1 ratio exactly
    f1_cells = cf.ratio_cells(rows, cases, CONDS)
    key = lambda c: (c["model"], c["scenario"])  # noqa: E731
    base = {key(c): c["ratio"] for c in f1_cells}
    for c in cells_all["primary"]:
        assert abs(base[key(c)] - c["ratio"]) < 1e-9, key(c)
    assert len(base) == len(cells_all["primary"])
    write_csv(os.path.join(a.out, "r2_ratio_robust.csv"), rob,
              ["variant", "label", "model", "n_scen", "median", "med_lo", "med_hi", "n_gt1", "sign_p", "n_cells_gt1",
               "n_cells", "n_censored"])

    # ------------------------------------------------ item 5: scenario-level counts
    sl = []
    prim = summarize_ratio(cells_all["primary"], models)
    for s in prim:
        sl.append({"model": s["model"], "level": "model median r per scenario" if s["model"] != "ALL"
                   else "median over models of r, per scenario", "n_scen": s["n_scen"], "n_gt1": s["n_gt1"],
                   "sign_p": s["sign_p"], "median": s["median"]})
        t = "rtScen" + tag(s["model"])
        mac[t + "NGtOne"] = str(s["n_gt1"])
        mac[t + "NScen"] = str(s["n_scen"])
        mac[t + "SignP"] = cf.fmt_p(s["sign_p"])
        mac[t + "Med"] = tex_num(s["median"])
    by = {}
    for c in cells_all["primary"]:
        by.setdefault(c["scenario"], []).append(c["ratio"])
    scen_med = {s: statistics.median(v) for s, v in by.items()}
    le1 = sorted(s for s, v in scen_med.items() if v <= 1)
    mac["rtScenAllLeOneList"] = ", ".join(x.replace("_", "\\_") for x in le1) if le1 else "none"
    mac["rtScenAllNLeOne"] = str(len(le1))
    mac["rtScenAllMinModelsGtOne"] = str(min(sum(1 for c in cells_all["primary"] if c["scenario"] == s and c["ratio"] > 1)
                                             for s in scen_med))
    mac["rtScenPairsNGtOne"] = str(sum(c["ratio"] > 1 for c in cells_all["primary"]))
    mac["rtScenPairsN"] = str(len(cells_all["primary"]))
    write_csv(os.path.join(a.out, "r2_scenario_level.csv"), sl, ["model", "level", "n_scen", "n_gt1", "sign_p", "median"])

    # ------------------------------------------------ item 6: update-only r
    recs_ids = {r.get("run_id") for r in recs}
    e1_id = ext.DEFAULT_IDS["E1"]
    upd = []
    for vname, (lo, hi) in (("own", (1e-5, 1 - 1e-5)), ("ownhalf", (0.005, 0.995))):
        cells = update_cells(rows, cases, lo, hi)
        summ = summarize_ratio(cells, models)
        for s in summ:
            upd.append(dict(s, variant=vname))
            t = "rtUpd" + ("" if vname == "own" else "ClipHalf") + tag(s["model"])
            mac[t + "Med"] = tex_num(s["median"])
            mac[t + "MedLo"] = tex_num(s["med_lo"]) if s["med_lo"] is not None else "n/a"
            mac[t + "MedHi"] = tex_num(s["med_hi"]) if s["med_hi"] is not None else "n/a"
            mac[t + "NGtOne"] = str(s["n_gt1"])
            mac[t + "NScen"] = str(s["n_scen"])
        pm = [s for s in summ if s["model"] != "ALL"]
        t = "rtUpd" + ("" if vname == "own" else "ClipHalf")
        mac[t + "MedMin"] = tex_num(min(s["median"] for s in pm))
        mac[t + "MedMax"] = tex_num(max(s["median"] for s in pm))
        mac[t + "NModelsMedGtOne"] = str(sum(s["median"] > 1 for s in pm))
        mac[t + "NModelsLoAboveOne"] = str(sum(s["med_lo"] is not None and s["med_lo"] > 1 for s in pm))
        if vname == "own":
            upd_own = summ
    have_blind = e1_id in recs_ids
    mac["rtBlindHave"] = "1" if have_blind else "0"
    if have_blind:
        e1_final = ext.final_of(recs, e1_id, cfg)
        bl, _nt, _no = ext.blind_cells(e1_final, ext.scenario_targets(cases))
        blind = {k: statistics.mean(logit_clipped(x) for x in v) for k, v in bl.items()}
        cells_b = update_cells(rows, cases, 1e-5, 1 - 1e-5, blind=blind)
        summ_b = summarize_ratio(cells_b, models)
        # same cells, no pretest subtracted: with a blind pretest shared by the positive and negative version
        # the update difference equals the posttest difference up to unequal answer counts per condition
        sub = {(c["model"], c["scenario"]) for c in cells_b}
        cells_same = [c for c in cells_all["primary"] if (c["model"], c["scenario"]) in sub]
        for s in summ_b:
            upd.append(dict(s, variant="blind"))
            t = "rtBlind" + tag(s["model"])
            mac[t + "Med"] = tex_num(s["median"])
            mac[t + "MedLo"] = tex_num(s["med_lo"]) if s["med_lo"] is not None else "n/a"
            mac[t + "MedHi"] = tex_num(s["med_hi"]) if s["med_hi"] is not None else "n/a"
            mac[t + "NGtOne"] = str(s["n_gt1"])
            mac[t + "NScen"] = str(s["n_scen"])
        pm = [s for s in summ_b if s["model"] != "ALL"]
        mac["rtBlindMedMin"] = tex_num(min(s["median"] for s in pm))
        mac["rtBlindMedMax"] = tex_num(max(s["median"] for s in pm))
        mac["rtBlindNModelsMedGtOne"] = str(sum(s["median"] > 1 for s in pm))
        mac["rtBlindNModelsLoAboveOne"] = str(sum(s["med_lo"] is not None and s["med_lo"] > 1 for s in pm))
        by_c = {(c["model"], c["scenario"]): c["ratio"] for c in cells_same}
        mac["rtBlindMaxAbsDiffFromPlainR"] = tex_num(max(abs(by_c[(c["model"], c["scenario"])] - c["ratio"])
                                                         for c in cells_b))
        mac["rtBlindNCells"] = str(len(cells_b))
    write_csv(os.path.join(a.out, "r2_update_only.csv"), upd,
              ["variant", "model", "n_scen", "median", "med_lo", "med_hi", "n_gt1", "sign_p", "n_cells_gt1", "n_cells"])

    # ------------------------------------------------ item 8: pre-specified primary metric
    prim_rows = []
    for m in models + ["ALL"]:
        for sset, metric in (("A", "band_err_pp"), ("B", "abs_logodds_err")):
            for cond in ["V", "N", "B", "S", "E"]:
                vs = [r[metric] for r in rows if (m == "ALL" or r["model"] == m) and r["set"] == sset
                      and r["condition"] == cond]
                q = statistics.quantiles(vs, n=4, method="inclusive") if len(vs) > 1 else [vs[0]] * 3
                prim_rows.append({"model": m, "set": sset, "metric": metric, "condition": cond, "n": len(vs),
                                  "median": statistics.median(vs), "q1": q[0], "q3": q[2]})
    for o in prim_rows:
        d = 1 if o["set"] == "A" else 2
        mac["rtPrim" + o["set"] + tag(o["model"]) + o["condition"] + "Med"] = tex_num(o["median"], d)
        mac["rtPrim" + o["set"] + tag(o["model"]) + o["condition"] + "N"] = str(o["n"])
    for m in models + ["ALL"]:
        for sset in ("A", "B"):
            g = {o["condition"]: o["median"] for o in prim_rows if o["model"] == m and o["set"] == sset}
            d = 1 if sset == "A" else 2
            for c1, c0 in (("N", "V"), ("B", "N"), ("S", "N"), ("E", "N")):
                mac["rtPrim" + sset + tag(m) + c1 + "Minus" + c0] = tex_num(g[c1] - g[c0], d)
    for sset in ("A", "B"):
        pm = [o for o in prim_rows if o["set"] == sset and o["model"] != "ALL"]
        d = 1 if sset == "A" else 2
        for c1, c0 in (("N", "V"), ("B", "N"), ("S", "N")):
            diffs = [next(x for x in pm if x["model"] == m and x["condition"] == c1)["median"] -
                     next(x for x in pm if x["model"] == m and x["condition"] == c0)["median"] for m in models]
            mac["rtPrim" + sset + c1 + "Minus" + c0 + "Min"] = tex_num(min(diffs), d)
            mac["rtPrim" + sset + c1 + "Minus" + c0 + "Max"] = tex_num(max(diffs), d)
    write_csv(os.path.join(a.out, "r2_primary_metric.csv"), prim_rows,
              ["model", "set", "metric", "condition", "n", "median", "q1", "q3"])
    lines = ["% GENERATED by analysis/round2_analysis.py. Median pre-specified primary metric: Set A band error in",
             "% percentage points (distance outside the target band), Set B absolute log-odds error. A5 excluded.",
             "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
             " & \\multicolumn{5}{c}{Set A, band error (pp)} & \\multicolumn{5}{c}{Set B, $|$log-odds error$|$} \\\\",
             "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
             "Model & V & N & B & S & E & V & N & B & S & E \\\\", "\\midrule"]
    for m in models + ["ALL"]:
        if m == "ALL":
            lines.append("\\midrule")
        nm = "All models" if m == "ALL" else display_name(m).split()[0]
        cellsA = [tex_num(next(o for o in prim_rows if o["model"] == m and o["set"] == "A" and o["condition"] == c)["median"], 1)
                  for c in "VNBSE"]
        cellsB = [tex_num(next(o for o in prim_rows if o["model"] == m and o["set"] == "B" and o["condition"] == c)["median"], 2)
                  for c in "VNBSE"]
        lines.append(nm + " & " + " & ".join(cellsA + cellsB) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    cf2.write_tex(os.path.join(a.generated, "table_primary.tex"), lines)
    with open(os.path.join(a.generated, "table_primary.tex")) as f:
        txt = f.read().replace("committee_fixes2.py", "round2_analysis.py", 1)
    with open(os.path.join(a.generated, "table_primary.tex"), "w") as f:
        f.write(txt)

    # ------------------------------------------------ items 3 and 5 need the extension analyses
    ext_ids = dict(ext.DEFAULT_IDS)
    with tempfile.TemporaryDirectory() as tmp:
        eres, emac, ehave = ext.main(["--raw", a.raw, "--cases", a.cases, "--out", tmp, "--generated", tmp,
                                      "--n-boot", str(a.n_boot), "--config", a.config])

    # ------------------------------------------------ item 3: counts
    crow, n_main_final, n_cal = count_set(recs, cases, cfg, ext_ids)
    cnt = []
    for grp in ("main", "ext"):
        for arm in sorted({r["arm"] for r in crow if r["group"] == grp}):
            for m in models + ["ALL"]:
                rs = [r for r in crow if r["arm"] == arm and (m == "ALL" or r["model"] == m)]
                cnt.append({"group": grp, "arm": arm, "model": m, "n_total": len(rs), "n_parsed": sum(r["ok"] for r in rs),
                            "n_unparsed": sum(not r["ok"] for r in rs), "n_unparsed_at_token_limit": sum(r["trunc"] for r in rs)})
    write_csv(os.path.join(a.out, "r2_counts.csv"), cnt,
              ["group", "arm", "model", "n_total", "n_parsed", "n_unparsed", "n_unparsed_at_token_limit"])

    def tot(grp, m="ALL"):
        rs = [r for r in crow if r["group"] == grp and (m == "ALL" or r["model"] == m)]
        return len(rs), sum(r["ok"] for r in rs), sum(r["trunc"] for r in rs)
    mt, mp, mtr = tot("main")
    et, ep, etr = tot("ext")
    mac["rtCntMainCollected"] = str(n_main_final)
    mac["rtCntMainCalibration"] = str(n_cal)
    mac["rtCntMainAnalysable"] = str(mt)
    mac["rtCntMainParsed"] = str(mp)
    mac["rtCntMainUnparsed"] = str(mt - mp)
    mac["rtCntMainUnparsedAtLimit"] = str(mtr)
    mac["rtCntMainUnparsedOther"] = str(mt - mp - mtr)
    mac["rtCntMainParsedPct"] = f"{100 * mp / mt:.1f}"
    for m in models:
        t_, p_, tr_ = tot("main", m)
        mac["rtCntMainUnparsed" + tag(m)] = str(t_ - p_)
        mac["rtCntMainUnparsedAtLimit" + tag(m)] = str(tr_)
    mac["rtCntExtAnalysable"] = str(et)
    mac["rtCntExtParsed"] = str(ep)
    mac["rtCntExtUnparsed"] = str(et - ep)
    mac["rtCntExtUnparsedAtLimit"] = str(etr)
    mac["rtCntExtUnparsedOther"] = str(et - ep - etr)
    mac["rtCntExtParsedPct"] = f"{100 * ep / et:.1f}"
    mac["rtCntAllAnalysable"] = str(mt + et)
    mac["rtCntAllParsed"] = str(mp + ep)
    mac["rtCntAllUnparsed"] = str(mt + et - mp - ep)
    mac["rtCntAllUnparsedAtLimit"] = str(mtr + etr)
    mac["rtCntAllUnparsedOther"] = str(mt + et - mp - ep - mtr - etr)
    mac["rtCntAllParsedPct"] = f"{100 * (mp + ep) / (mt + et):.1f}"
    if ehave.get("E5"):
        rt = eres["E5"][0]
        mac["rtCntRetryN"] = str(len(rt))
        mac["rtCntRetryParsed"] = str(sum(1 for r in rt if r["retry_status"] == "ok"))
    # cross-checks against the older macros so a silent change fails loudly
    assert str(ep) == emac["extRunsNParsed"] and str(et) == emac["extRunsNTotal"], (ep, et)
    assert str(etr) == emac["extRunsNTruncTotal"]
    assert mp == len(rows), (mp, len(rows))
    assert mt - mp == sum(1 for r in final if not is_calibration(cases[r["case_id"]]) and
                          score_record(r, cases[r["case_id"]])["parse_ok"] == 0)

    # ------------------------------------------------ item 3: threshold share, primary and alternative targets
    thr_rows = []
    pe_thr = cf2.threshold_specs(cases)
    spec = next(s for s in pe_thr if s["id"] == "pe_test_kline")
    thr = spec["value"]
    b2n = [r for r in rows if r["case_id"] == "B2_pe_mod_neg" and r["condition"] in CONDS]
    assert b2n
    alt = cf2.alt_specs(cases, cf2.load_json(cf2.ALT))
    targets = {"primary": cases["B2_pe_mod_neg"]["target"]["posttest_point"],
               "pioped_lr_point": alt["lr_ci_band"]["B2_pe_mod_neg"]["target"],
               "pioped_lr_band_lo": alt["lr_ci_band"]["B2_pe_mod_neg"]["band"][0],
               "pioped_observed_npv": alt["observed"]["B2_pe_mod_neg"]["target"]}
    share_below_thr = 100 * sum(r["post"] < thr for r in b2n) / len(b2n)
    mac["rtThrBTwoNegN"] = str(len(b2n))
    mac["rtThrBTwoNegThresholdPct"] = f"{100 * thr:g}"
    mac["rtThrBTwoNegBelowThresholdPct"] = f"{share_below_thr:.0f}"
    mac["rtThrBTwoNegBelowThresholdN"] = str(sum(r["post"] < thr for r in b2n))
    for m in models:
        mr = [r for r in b2n if r["model"] == m]
        mac["rtThrBTwoNeg" + tag(m) + "BelowThresholdPct"] = f"{100 * sum(r['post'] < thr for r in mr) / len(mr):.0f}"
    for k, t in targets.items():
        w = "".join(x.capitalize() for x in k.split("_"))
        below = 100 * sum(r["post"] < t for r in b2n) / len(b2n)
        thr_rows.append({"target": k, "target_pct": 100 * t, "threshold_pct": 100 * thr, "target_above_threshold": int(t > thr),
                         "n": len(b2n), "pct_answers_below_threshold": share_below_thr, "pct_answers_below_target": below})
        mac["rtThrBTwoNeg" + w + "TargetPct"] = tex_num(100 * t, 2)
        mac["rtThrBTwoNeg" + w + "AboveThreshold"] = "yes" if t > thr else "no"
        mac["rtThrBTwoNeg" + w + "BelowTargetPct"] = f"{below:.0f}"
    # cross-check against the older macro (same cases and conditions)
    with tempfile.TemporaryDirectory() as tmp:
        tmac = {}
        cf2.f8(rows, cases, tmac, tmp)
    assert tmac["cfThrCaseBTwoPeModNegPeTestKlineAllOtherPct"] == mac["rtThrBTwoNegBelowThresholdPct"], \
        (tmac["cfThrCaseBTwoPeModNegPeTestKlineAllOtherPct"], mac["rtThrBTwoNegBelowThresholdPct"])
    write_csv(os.path.join(a.out, "r2_threshold_alt.csv"), thr_rows,
              ["target", "target_pct", "threshold_pct", "target_above_threshold", "n", "pct_answers_below_threshold",
               "pct_answers_below_target"])

    # ------------------------------------------------ item 5: Holm over the post hoc family
    fam = []   # (family, test, model, p)

    def add(family, test, m, p):
        assert p is not None, (family, test, m)
        fam.append({"family": family, "test": test, "model": m, "p": p})
    for s in prim:
        add("Ratio" if s["model"] != "ALL" else "Pooled", "r_gt_1_sign_test", s["model"], s["sign_p"])
    hc, hs = cf.hindsight(rows)
    for s in hs:
        add("Hind" if s["model"] != "ALL" else "Pooled", "hindsight_pre_pos_gt_neg", s["model"], s["sign_p"])
    for s in eres["E1"][1]:
        for k, nm in (("gap_pos", "BlindPos"), ("gap_neg", "BlindNeg")):
            add(nm if s["model"] != "ALL" else "Pooled", "blind_" + k, s["model"], s[k + "_lod_p"])
    for s in eres["E2"][1]:
        add("Dor", "believed_log_dor_excess", s["model"], s["log_dor_excess_p"])
    for x in eres["E3"][0]:
        if x["metric"] == "abs" and x["result"] == "all":
            add("Long" if x["model"] != "ALL" else "Pooled", "long_vs_short_abs", x["model"], x["p_signflip"])
    for x in eres["E4"][0]:
        if x["metric"] == "abs" and x["result"] == "all" and x["contrast"] in ("NunVsN", "LunVsL"):
            add("Unnamed" if x["model"] != "ALL" else "Pooled", x["contrast"] + "_abs", x["model"], x["p_signflip"])
    permodel = [f for f in fam if f["family"] != "Pooled"]
    pooled = [f for f in fam if f["family"] == "Pooled"]
    for f, ad in zip(permodel, holm([f["p"] for f in permodel])):
        f["p_holm_family"] = ad
    for f, ad in zip(pooled, holm([f["p"] for f in pooled])):
        f["p_holm_family"] = ad
    for name in sorted({f["family"] for f in fam}):
        sub = [f for f in fam if f["family"] == name]
        for f, ad in zip(sub, holm([f["p"] for f in sub])):
            f["p_holm_within"] = ad
    for f in fam:
        f["raw_lt_05"] = int(f["p"] < 0.05)
        f["holm_family_lt_05"] = int(f["p_holm_family"] < 0.05)
        f["holm_within_lt_05"] = int(f["p_holm_within"] < 0.05)
    write_csv(os.path.join(a.out, "r2_holm.csv"), fam,
              ["family", "test", "model", "p", "p_holm_family", "p_holm_within", "raw_lt_05", "holm_family_lt_05",
               "holm_within_lt_05"])
    mac["rtHolmNTests"] = str(len(permodel))
    mac["rtHolmNRaw"] = str(sum(f["raw_lt_05"] for f in permodel))
    mac["rtHolmNSurviveFamily"] = str(sum(f["holm_family_lt_05"] for f in permodel))
    mac["rtHolmNSurviveWithin"] = str(sum(f["holm_within_lt_05"] for f in permodel))
    mac["rtHolmMinRawP"] = cf.fmt_p(min(f["p"] for f in permodel))
    mac["rtHolmPooledNTests"] = str(len(pooled))
    mac["rtHolmPooledNRaw"] = str(sum(f["raw_lt_05"] for f in pooled))
    mac["rtHolmPooledNSurvive"] = str(sum(f["holm_family_lt_05"] for f in pooled))
    for name in sorted({f["family"] for f in permodel}):
        sub = [f for f in permodel if f["family"] == name]
        mac[f"rtHolm{name}NTests"] = str(len(sub))
        mac[f"rtHolm{name}NRaw"] = str(sum(f["raw_lt_05"] for f in sub))
        mac[f"rtHolm{name}NSurviveWithin"] = str(sum(f["holm_within_lt_05"] for f in sub))
        mac[f"rtHolm{name}NSurviveFamily"] = str(sum(f["holm_family_lt_05"] for f in sub))
        mac[f"rtHolm{name}MinRawP"] = cf.fmt_p(min(f["p"] for f in sub))
        mac[f"rtHolm{name}MinAdjWithin"] = cf.fmt_p(min(f["p_holm_within"] for f in sub))
        mac[f"rtHolm{name}MinAdjFamily"] = cf.fmt_p(min(f["p_holm_family"] for f in sub))
    for f in pooled:
        nm = {"r_gt_1_sign_test": "Ratio", "hindsight_pre_pos_gt_neg": "Hind", "blind_gap_pos": "BlindPos",
              "blind_gap_neg": "BlindNeg", "long_vs_short_abs": "Long", "NunVsN_abs": "UnnamedN",
              "LunVsL_abs": "UnnamedL"}[f["test"]]
        mac[f"rtHolmPooled{nm}RawP"] = cf.fmt_p(f["p"])
        mac[f"rtHolmPooled{nm}AdjP"] = cf.fmt_p(f["p_holm_family"])

    # ------------------------------------------------ write macros
    path = os.path.join(a.generated, "round2_macros.tex")
    cf.write_macros(path, mac)
    with open(path) as f:
        txt = f.read().replace("committee_fixes.py", "round2_analysis.py", 1) \
.replace("prompted by", "prompted by")
    with open(path, "w") as f:
        f.write(txt)
    print(f"round2_analysis: {len(rows)} analysed rows, {len(mac)} macros")
    return {"mac": mac, "rob": rob, "floor": fl, "upd": upd, "holm": fam, "prim": prim_rows, "cnt": cnt,
            "thr": thr_rows, "scen": sl}, mac


if __name__ == "__main__":
    main()
