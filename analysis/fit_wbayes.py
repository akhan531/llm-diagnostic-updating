"""Weighted-Bayes fit: logit(post) = a*logit(prior) + b*log(LR) + c, per model x condition.

Variants: "published" (prior = case pretest_point) and "own" (prior = model's stated pretest),
each also as "<variant>_noboundary" (rows with clamped 0/100 answers excluded).
Bootstrap: 2000 resamples of scenarios (clusters) with replacement, percentile 95% CI, fixed seed.
Usage: python analysis/fit_wbayes.py [--raw ...] [--cases ...] [--out ...] [--n-boot 2000]
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ROOT, add_run_args, check_unclipped, default_out, resolve_run_ids, select_records, guard_output, is_boundary, is_synthetic, load_cases,  # noqa: E402
                    load_raw, logit_clipped, write_csv, write_provenance)

SEED = 20261111
FIT_EXCLUDED_CONDITIONS = {"E"}  # protocol Section 6: E has no defined prior; never fitted
EXCLUDED = {}  # (model, condition, variant_noboundary) -> rows dropped; reset in collect
FIELDS = ["model", "condition", "variant", "n_obs", "n_clusters", "a", "a_lo", "a_hi", "b", "b_lo",
          "b_hi", "c", "c_lo", "c_hi", "n_boot_valid", "n_boot", "n_boundary_answers", "boundary_flag", "boundary_flag_params",
          "note"]


def solve3(A, y):
    """Solve A x = y for 3x3 A by Gaussian elimination with partial pivoting; None if singular."""
    M = [row[:] + [y[i]] for i, row in enumerate(A)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(M[r][i]))
        if abs(M[p][i]) < 1e-10:
            return None
        M[i], M[p] = M[p], M[i]
        for r in range(i + 1, 3):
            f = M[r][i] / M[i][i]
            for k in range(i, 4):
                M[r][k] -= f * M[i][k]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (M[i][3] - sum(M[i][k] * x[k] for k in range(i + 1, 3))) / M[i][i]
    return x


def ols(rows):
    """rows: list of (x_prior, x_lr, y). Returns [a, b, c] or None if rank deficient."""
    if len(rows) < 3:
        return None
    A = [[0.0] * 3 for _ in range(3)]
    y = [0.0] * 3
    for xp, xl, yy in rows:
        v = (xp, xl, 1.0)
        for i in range(3):
            y[i] += v[i] * yy
            for j in range(3):
                A[i][j] += v[i] * v[j]
    return solve3(A, y)


def pct(sorted_vals, q):
    k = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def fit_boot(clusters, n_boot, rng):
    """clusters: {cluster_id: [(xp, xl, y), ...]}. Returns dict of estimates and CIs."""
    allrows = [r for rs in clusters.values() for r in rs]
    est = ols(allrows)
    if est is None:
        return None
    ids = list(clusters)
    boots = []
    for _ in range(n_boot):
        rs = []
        for _i in ids:
            rs.extend(clusters[ids[rng.randrange(len(ids))]])
        e = ols(rs)
        if e is not None:
            boots.append(e)
    out = {"n_obs": len(allrows), "n_clusters": len(ids), "n_boot_valid": len(boots),
           "n_boot": n_boot}
    for i, name in enumerate("abc"):
        out[name] = est[i]
        if len(boots) >= 20:
            s = sorted(b[i] for b in boots)
            out[name + "_lo"], out[name + "_hi"] = pct(s, 0.025), pct(s, 0.975)
    return out




def add_entry(data, nb, model, condition, scenario, xp_pub, xp_own, xl, y, post_pct, pre_pct):
    """Add one parsed answer to all four variants.

    "published"/"own" keep every row. The *_noboundary variants drop rows whose stated posttest
    (and, for "own", stated pretest) is a clamped boundary answer. Keys are always created so an
    emptied variant still appears in the output with a note. nb counts clamped rows kept in
    "published"/"own" and excluded rows (nb_excl) in the noboundary variants.
    """
    post_b = is_boundary(post_pct / 100.0)
    pre_b = is_boundary(pre_pct / 100.0)
    for variant, xp, clamped in (("published", xp_pub, post_b), ("own", xp_own, post_b or pre_b)):
        key = (model, condition, variant)
        nkey = (model, condition, variant + "_noboundary")
        data.setdefault(key, {}).setdefault(scenario, []).append((xp, xl, y))
        data.setdefault(nkey, {})
        nb[key] = nb.get(key, 0) + int(clamped)
        nb.setdefault(nkey, 0)
        if clamped:
            EXCLUDED[nkey] = EXCLUDED.get(nkey, 0) + 1
        else:
            data[nkey].setdefault(scenario, []).append((xp, xl, y))


def collect(recs, cases):
    """Group ok records into {(model, condition, variant): {scenario: [(xp, xl, y)]}}."""
    from score import score_record
    EXCLUDED.clear()
    data = {}
    nb = {}  # (model, condition, variant) -> number of fit rows with a clamped model answer
    for r in recs:
        case = cases[r["case_id"]]
        if case.get("exclude_from_fit"):
            continue  # e.g. A5 hypothetical: exact LR- is 0, placeholder LR must not enter the fit
        if r["condition"] in FIT_EXCLUDED_CONDITIONS:
            continue
        s = score_record(r, case)
        if not s["parse_ok"]:
            continue
        check_unclipped(case["target"]["pretest_point"], f"{case['case_id']} pretest_point")
        check_unclipped(case["target"]["posttest_point"], f"{case['case_id']} posttest_point")
        y = logit_clipped(s["posttest_stated"] / 100.0)
        xl = math.log(case["target"]["lr_point"])
        add_entry(data, nb, r["model"], r["condition"], case["scenario"],
                  logit_clipped(case["target"]["pretest_point"]),
                  logit_clipped(s["pretest_stated"] / 100.0), xl, y,
                  s["posttest_stated"], s["pretest_stated"])
    return data, nb


def self_check():
    """Assert the estimator and bootstrap on synthetic data (runs on every invocation)."""
    rng = random.Random(1)
    # Perfect Bayesian: a=b=1, c=0 exactly.
    cl = {}
    for s in range(8):
        pri = rng.uniform(-2.5, 2.5)
        for lr in (math.log(rng.uniform(4, 20)), math.log(rng.uniform(0.05, 0.4))):
            cl.setdefault(s, []).append((pri, lr, pri + lr))
    r = fit_boot(cl, 50, random.Random(SEED))
    assert abs(r["a"] - 1) < 1e-6 and abs(r["b"] - 1) < 1e-6 and abs(r["c"]) < 1e-6, r
    # a=0.6, b=0.5 plus noise: truth must fall inside the bootstrap CI.
    cl = {}
    for s in range(40):
        pri = rng.uniform(-2.5, 2.5)
        for lr in (math.log(rng.uniform(4, 20)), math.log(rng.uniform(0.05, 0.4))):
            cl.setdefault(s, []).append((pri, lr, 0.6 * pri + 0.5 * lr + rng.gauss(0, 0.2)))
    r = fit_boot(cl, 500, random.Random(SEED))
    assert r["a_lo"] <= 0.6 <= r["a_hi"] and r["b_lo"] <= 0.5 <= r["b_hi"], r
    assert r["c_lo"] <= 0.0 <= r["c_hi"], r


def _num(x):
    return None if x in (None, "") else float(x)


def boundary_flags(rows):
    """Pre-specified 0/100 sensitivity flag (protocol Section 4; see README).

    For each model x condition x variant pair (main, main_noboundary) with at least one boundary answer:
    flag the pair, on BOTH rows, if for a or b the no-boundary point estimate lies outside the main 95% CI,
    or the main point estimate lies outside the no-boundary 95% CI, or the no-boundary fit does not exist.
    boundary_flag_params lists the parameters that triggered it (e.g. "a;b"). Fills the two columns in place."""
    idx = {(r["model"], r["condition"], r["variant"]): r for r in rows}
    for (m, c, v), r in idx.items():
        r.setdefault("boundary_flag", 0)
        r.setdefault("boundary_flag_params", "")
        if v.endswith("_noboundary"):
            continue
        nb = idx.get((m, c, v + "_noboundary"))
        if nb is None or not int(r.get("n_boundary_answers") or 0) or r.get("a") in (None, ""):
            continue
        hit = []
        if nb.get("a") in (None, ""):
            hit.append("nofit")
        else:
            for p in "ab":
                mp, np_ = _num(r[p]), _num(nb[p])
                mlo, mhi, nlo, nhi = (_num(r.get(p + "_lo")), _num(r.get(p + "_hi")),
                                      _num(nb.get(p + "_lo")), _num(nb.get(p + "_hi")))
                if (mlo is not None and not mlo <= np_ <= mhi) or (nlo is not None and not nlo <= mp <= nhi):
                    hit.append(p)
        for row in (r, nb):
            row["boundary_flag"] = int(bool(hit))
            row["boundary_flag_params"] = ";".join(hit)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "*.jsonl"))
    ap.add_argument("--cases", default=os.path.join(ROOT, "data", "cases"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    add_run_args(ap)
    a = ap.parse_args(argv)
    allowed, run_set, cfg = resolve_run_ids(a)
    a.out = default_out(a.out, run_set)
    self_check()
    cases = load_cases(a.cases)
    rawrep = {}
    recs, _ = load_raw(a.raw, rawrep)
    syn = is_synthetic(recs, a.raw)
    guard_output(a.out, syn)
    recs, rep = select_records(recs, allowed, cfg)
    assert recs, f"no records selected for run ids {allowed}"
    print(f"run set {run_set} {allowed}: {rep['n_final']} latest-attempt records "
          f"({rep['n_superseded']} superseded), condition E excluded from fits")
    assert all(r["case_id"] in cases for r in recs), "raw records reference unknown case ids"
    data, nb = collect(recs, cases)
    rows = []
    for key in sorted(data):
        m, c, v = key
        rng = random.Random(SEED)
        res = fit_boot(data[key], a.n_boot, rng)
        row = {"model": m, "condition": c, "variant": v, "n_boundary_answers": nb.get(key, 0)}
        notes = []
        if v.endswith("_noboundary"):
            notes.append(f"{EXCLUDED.get(key, 0)} boundary rows excluded (sensitivity)")
        if res is None:
            notes.append("rank deficient or fewer than 3 observations; no fit")
            row["note"] = "; ".join(notes)
        else:
            row.update(res)
            if res["n_clusters"] < 10:
                notes.append("few clusters: CIs unreliable")
            if res["n_boot_valid"] < res["n_boot"]:
                notes.append(f"{res['n_boot'] - res['n_boot_valid']} degenerate resamples dropped")
            row["note"] = "; ".join(notes)
            assert all(math.isfinite(res[k]) for k in "abc")
        rows.append(row)
    boundary_flags(rows)
    write_csv(os.path.join(a.out, "wbayes.csv"), rows, FIELDS)
    write_provenance(a.out, "wbayes", a.raw, syn, len(recs), run_set, allowed,
                     rawrep["n_malformed"])
    print(f"wrote {len(rows)} fits")
    return rows


if __name__ == "__main__":
    main()
