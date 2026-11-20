"""Recompute every Bayesian target in data/cases/*.json from the stored inputs and assert a match.

Usage:
    python analysis/build_targets.py            # verify all case files, print the table
    python analysis/build_targets.py --write    # (used by analysis/make_cases.py) recompute and store

All arithmetic goes through analysis/bayes.py. The stored `target_inputs` block says how each number
is obtained; nothing in `target` is typed by hand.
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bayes import expit, logit, posterior  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-9


def _value(spec):
    """A scalar from {"num":..,"den":..} or {"value":..}."""
    if "num" in spec:
        return spec["num"] / spec["den"]
    return spec["value"]


def compute(ti, result):
    """Return the `target` dict implied by inputs `ti` for result 'pos' or 'neg'."""
    pre = ti["pretest"]
    if pre["rule"] == "logodds_midpoint":
        lo, hi = pre["range"]
        pre_point = expit((logit(lo) + logit(hi)) / 2)
        pre_range = [lo, hi]
    elif pre["rule"] == "point":
        pre_point = _value(pre)
        pre_range = [pre_point, pre_point]
    else:
        raise ValueError(pre["rule"])

    lr = ti["lr"][result]
    rule = lr["rule"]
    exact_zero = False
    if rule == "geomean_range":
        lo, hi = lr["range"]
        lr_point, lr_range = math.sqrt(lo * hi), [lo, hi]
    elif rule == "published_point":
        lr_point = lr["point"]
        lr_range = lr.get("range", [lr_point, lr_point])
    elif rule == "from_sens_spec":
        sens, spec = ti["sens"], ti["spec"]
        exact = (sens / (1 - spec)) if result == "pos" else ((1 - sens) / spec)
        lr_point, lr_range = exact, [exact, exact]
        if exact == 0:
            exact_zero = True
    else:
        raise ValueError(rule)

    if exact_zero:
        # LR- = 0: posterior is exactly 0 (logit undefined). Documented fit value kept separately.
        post_point = 0.0
        lr_point = lr["fit_value"]
        lr_range = [lr["fit_value"]] * 2
        bayes_band = [0.0, 0.0]
        primary = [0.0, 0.0]
    else:
        post_point = posterior(pre_point, [lr_point])
        corners = [posterior(p, [l]) for p in pre_range for l in lr_range]
        bayes_band = [min(corners), max(corners)]
        bp = ti["band_primary"]
        if bp["rule"] == "printed":
            primary = [x / 100.0 for x in bp["pct"][result]]
        elif bp["rule"] == "bayes_lr_range_at_point_prior":
            c = [posterior(pre_point, [l]) for l in lr_range]
            primary = [min(c), max(c)]
            bayes_band = list(primary)
        elif bp["rule"] == "degenerate":
            primary = [post_point, post_point]
            bayes_band = [post_point, post_point]
        else:
            raise ValueError(bp["rule"])
    out = {"pretest_point": pre_point, "pretest_range": pre_range, "lr_point": lr_point,
           "lr_range": lr_range, "posttest_point": post_point,
           "posttest_band_primary": primary, "posttest_band_bayes": bayes_band}
    if exact_zero:
        out["lr_exact"] = 0.0
    return out


def sens_prior_targets(ti, result):
    """posttest at each alternative prior (sensitivity analysis), same LR point as the main target."""
    t = compute(ti, result)
    return [{"label": s["label"], "pretest": s["pretest"],
             "posttest_point": posterior(s["pretest"], [t["lr_point"]])}
            for s in ti.get("sensitivity_priors_in", [])]


def _close(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return abs(a - b) <= TOL


def main(argv):
    write = "--write" in argv
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "cases", "*.json"))):
        with open(path) as f:
            c = json.load(f)
        res = c["result"]
        assert res in ("pos", "neg"), f"{path}: result must be pos or neg"
        assert c["case_id"].endswith("_" + res) and c["case_id"].startswith(c["scenario"])
        new = compute(c["target_inputs"], res)
        if write:
            c["target"] = new
            if c["target_inputs"].get("sensitivity_priors_in"):
                c["sensitivity_priors"] = sens_prior_targets(c["target_inputs"], res)
            with open(path, "w") as f:
                json.dump(c, f, indent=2, ensure_ascii=False)
                f.write("\n")
        else:
            for k, v in new.items():
                assert k in c["target"], f"{path}: stored target lacks {k}"
                assert _close(v, c["target"][k]), f"{path}: {k} stored {c['target'][k]} != recomputed {v}"
            for k in c["target"]:
                assert k in new, f"{path}: stored target has unexpected key {k}"
            for s, n in zip(c.get("sensitivity_priors", []), sens_prior_targets(c["target_inputs"], res)):
                assert _close(s["posttest_point"], n["posttest_point"]), f"{path}: sensitivity prior mismatch"
        # structural checks
        t = new
        for k in ("posttest_band_primary", "posttest_band_bayes"):
            assert 0 <= t[k][0] <= t[k][1] <= 1, f"{path}: bad {k}"
        assert t["lr_point"] > 0, f"{path}: lr_point must be > 0"
        assert c["sources"] and all({"quantity", "citation_key", "id", "location", "status"} <= set(s)
                                    for s in c["sources"]), f"{path}: incomplete sources"
        for st in c.get("sensitivity_targets", []):
            assert abs(st["value"] - st["num"] / st["den"]) <= TOL and st["ci"][0] <= st["value"] <= st["ci"][1], \
                f"{path}: bad sensitivity_target {st['label']}"
            if "npv" in st:  # converted NPV: must be stored as P(disease) = 1 - NPV with CI [1 - hi, 1 - lo]
                assert abs(st["value"] - (1 - st["npv"])) <= TOL and _close(st["ci"], [1 - st["npv_ci"][1], 1 - st["npv_ci"][0]]), \
                    f"{path}: sensitivity_target {st['label']} must store 1 - NPV"
            else:
                assert "NPV" not in st["label"], f"{path}: NPV-labelled sensitivity target must be converted to P(disease)"
        if c["scenario"] == "B4_aas":  # ADD-RS=1 stratum counts must be consistent with the primary prior (96/1071)
            ti = c["target_inputs"]["pretest"]
            st = c["sensitivity_targets"]
            assert len(st) == 1, f"{path}: B4 needs exactly one stratum sensitivity target per polarity"
            # observed counts from Nazerian p. 253-255: positives 441 (94 AAS), negatives 630 (2 AAS)
            exp = (94, 441) if c["result"] == "pos" else (2, 630)
            assert (st[0]["num"], st[0]["den"]) == exp, f"{path}: B4 stratum counts {st[0]['num']}/{st[0]['den']} != {exp}"
            assert 94 + 2 == ti["num"] and 441 + 630 == ti["den"], f"{path}: stratum counts do not sum to the ADD-RS=1 prior"
        assert c["status"] in ("VERIFIED", "PROVISIONAL")
        if c["status"] == "PROVISIONAL":
            assert c.get("unverified"), f"{path}: PROVISIONAL needs an unverified list"
        for k in ("diagnosis", "setting", "patient_summary", "test_name", "pretest_source",
                  "set", "scenario"):
            assert c.get(k), f"{path}: missing {k}"
        rows.append((c["case_id"], t, c))
    pct = lambda x: f"{100 * x:.4g}"
    hdr = f"{'case_id':20s} {'prior%':>8s} {'LR':>8s} {'post_pt%':>9s} {'primary band %':>18s} {'bayes band %':>18s} fit"
    print(hdr)
    for cid, t, c in rows:
        print(f"{cid:20s} {pct(t['pretest_point']):>8s} {t['lr_point']:8.4g} {pct(t['posttest_point']):>9s} "
              f"{pct(t['posttest_band_primary'][0]) + '-' + pct(t['posttest_band_primary'][1]):>18s} "
              f"{pct(t['posttest_band_bayes'][0]) + '-' + pct(t['posttest_band_bayes'][1]):>18s} "
              f"{'EXCL' if c.get('exclude_from_fit') else ''}")
    print(f"{len(rows)} case files; all targets match recomputation to {TOL}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
