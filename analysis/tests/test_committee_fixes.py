"""Tests for analysis/committee_fixes.py (committee review F items)."""
import csv
import math
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import committee_fixes as cf  # noqa: E402
from common import ROOT, logit_clipped  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "synthetic.jsonl")


def bayes(p, lr):
    o = p / (1 - p) * lr
    return o / (1 + o)


def row(model, scen, res, cond, prior, lr, post, pre=None, seed=1, target=None):
    return {"model": model, "scenario": scen, "result": res, "condition": cond, "prior": prior, "lr": lr,
            "post": post, "pre": prior if pre is None else pre, "seed": seed,
            "target": bayes(prior, lr) if target is None else target, "case_id": f"{scen}_{res}"}


def fake_cases(spec):
    """spec: {scenario: (prior, lr_pos, lr_neg)} -> cases dict like load_cases."""
    out = {}
    for s, (p, lp, ln) in spec.items():
        for res, lr in (("pos", lp), ("neg", ln)):
            out[f"{s}_{res}"] = {"case_id": f"{s}_{res}", "scenario": s, "result": res, "set": "A",
                                 "target": {"pretest_point": p, "lr_point": lr, "posttest_point": bayes(p, lr)}}
    return out


SPEC = {f"S{i}": (0.05 + 0.04 * i, 2.0 + i, 1.0 / (2.0 + 0.5 * i)) for i in range(10)}


def bayes_rows(b=1.0, model="M", conds=("V", "N")):
    rows = []
    for s, (p, lp, ln) in SPEC.items():
        for res, lr in (("pos", lp), ("neg", ln)):
            for c in conds:
                lo = logit_clipped(p) + b * math.log(lr)
                rows.append(row(model, s, res, c, p, lr, 1 / (1 + math.exp(-lo))))
    return rows


class TestExactTests(unittest.TestCase):
    def test_sign_test(self):
        self.assertAlmostEqual(cf.sign_test_p(10, 10), 2 / 1024)
        self.assertAlmostEqual(cf.sign_test_p(9, 10), 22 / 1024)
        self.assertAlmostEqual(cf.sign_test_p(5, 10), 1.0)
        self.assertAlmostEqual(cf.sign_test_p(0, 10), 2 / 1024)

    def test_median_ci_n10(self):
        lo, hi, cov = cf.median_ci(list(range(10)))
        self.assertEqual((lo, hi), (1, 8))
        self.assertAlmostEqual(cov, 1 - 22 / 1024)

    def test_signflip(self):
        d = [1.0] * 10
        self.assertAlmostEqual(cf.signflip_p(d), 2 / 1024)
        lo, hi = cf.signflip_ci([0.1 * i - 0.45 for i in range(10)])
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)
        self.assertGreater(cf.signflip_p([0.1 * i - 0.45 for i in range(10)], lo + 1e-3), 0.05)
        self.assertLessEqual(cf.signflip_p([0.1 * i - 0.45 for i in range(10)], lo - 1e-2), 0.05)


class TestF1Ratio(unittest.TestCase):
    def test_exact_bayes_ratio_is_one(self):
        cells = cf.ratio_cells(bayes_rows(1.0), fake_cases(SPEC), ["V", "N"])
        self.assertEqual(len(cells), 10)
        for c in cells:
            self.assertAlmostEqual(c["ratio"], 1.0, places=9)
        s = cf.ratio_summary(cells)[0]
        self.assertAlmostEqual(s["b_fe"], 1.0, places=9)
        self.assertAlmostEqual(s["median"], 1.0, places=9)

    def test_overupdating_recovered(self):
        cells = cf.ratio_cells(bayes_rows(1.5), fake_cases(SPEC), ["V", "N"])
        s = cf.ratio_summary(cells)[0]
        self.assertAlmostEqual(s["b_fe"], 1.5, places=9)
        self.assertAlmostEqual(s["loso_min"], 1.5, places=9)
        self.assertEqual(s["n_gt1"], 10)
        self.assertAlmostEqual(s["sign_p"], 2 / 1024)

    def test_all_row_is_scenario_level(self):
        rows = bayes_rows(1.0)
        for r in rows:
            r["pre"] = 0.30 if r["result"] == "pos" else 0.20
        # a second model with the same data must not inflate the count: ALL has one value per scenario
        rows += [dict(r, model="M2") for r in rows]
        _, summ = cf.hindsight(rows, conds=["V", "N"])
        a = [x for x in summ if x["model"] == "ALL"][0]
        self.assertEqual(a["n_cells"], 10)
        self.assertEqual(a["n_gap_pos"], 10)
        self.assertAlmostEqual(a["sign_p"], 2 / 1024)
        self.assertAlmostEqual(a["gap_pp_med"], 10.0)

    def test_contrast_macros_two_decimals(self):
        self.assertEqual(cf.num2(0.0001), "0.00")
        self.assertEqual(cf.num2(-0.0001), "0.00")
        self.assertEqual(cf.num2(0.004), "0.00")
        self.assertEqual(cf.num2(-1.234), cf.tex_num(-1.234, 2))

    def test_prior_shift_cancels(self):
        """A scenario-level shift in the stated answer (wrong prior) must not move r."""
        rows = bayes_rows(1.0)
        for r in rows:
            lo = logit_clipped(r["post"]) + 0.7
            r["post"] = 1 / (1 + math.exp(-lo))
        for c in cf.ratio_cells(rows, fake_cases(SPEC), ["V", "N"]):
            self.assertAlmostEqual(c["ratio"], 1.0, places=9)


class TestF2Hindsight(unittest.TestCase):
    def test_gap(self):
        rows = bayes_rows(1.0)
        for r in rows:
            r["pre"] = 0.30 if r["result"] == "pos" else 0.20
        cells, summ = cf.hindsight(rows, conds=["V", "N"])
        s = [x for x in summ if x["model"] == "M"][0]
        self.assertAlmostEqual(s["gap_pp_med"], 10.0)
        self.assertEqual(s["n_gap_pos"], 10)
        self.assertEqual(s["n_cond_cells_pos"], 20)
        self.assertAlmostEqual(s["sign_p"], 2 / 1024)


class TestF9Decomposition(unittest.TestCase):
    def test_components_add_up_and_isolate_prior(self):
        rows = bayes_rows(1.0)
        for r in rows:  # stated pretest wrong by +1 log-odds, update exact from that wrong prior
            r["pre"] = 1 / (1 + math.exp(-(logit_clipped(r["prior"]) + 1.0)))
            r["post"] = 1 / (1 + math.exp(-(logit_clipped(r["pre"]) + math.log(r["lr"]))))
        dec = [d for d in cf.decomposition(rows, conds=["V", "N"]) if d["model"] == "M"]
        for d in dec:
            self.assertAlmostEqual(d["prior_mean"], 1.0, places=6)
            self.assertAlmostEqual(d["update_mean"], 0.0, places=6)
            self.assertAlmostEqual(d["total_mean"], d["prior_mean"] + d["update_mean"], places=9)


class TestF6Contrasts(unittest.TestCase):
    def test_constant_worsening(self):
        rows = bayes_rows(1.0, conds=("V", "N"))
        for r in rows:
            if r["condition"] == "N":  # N answers 0.5 log-odds above target: |err| rises by 0.5
                r["post"] = 1 / (1 + math.exp(-(logit_clipped(r["target"]) + 0.5)))
        sc = cf.scenario_contrasts(rows, "N", "V", "abs")["M"]
        self.assertEqual(len(sc), 10)
        for v in sc.values():
            self.assertAlmostEqual(v, 0.5, places=9)
        self.assertAlmostEqual(cf.signflip_p(list(sc.values())), 2 / 1024)


class TestF11RunSettings(unittest.TestCase):
    def test_truncation_count(self):
        cases = fake_cases({"S0": (0.1, 2.0, 0.5)})
        recs = [{"model": "M", "case_id": "S0_pos", "condition": "N", "parse_status": "ok", "completion_tokens": 50,
                 "decoding_params": {"temperature": 1.0, "max_tokens": 100}},
                {"model": "M", "case_id": "S0_neg", "condition": "N", "parse_status": "unparseable_missing_field",
                 "completion_tokens": 100, "decoding_params": {"temperature": 1.0, "max_tokens": 100}}]
        rs, tr = cf.run_settings(recs, recs, cases)
        self.assertEqual(rs[0]["n_length_stopped_final"], 1)
        self.assertEqual(rs[0]["n_length_stopped_unparsed_final"], 1)
        self.assertEqual(rs[0]["provider"], "not recorded")
        self.assertEqual(rs[0]["reasoning_settings"], "none recorded")
        self.assertEqual(tr, [{"model": "M", "scenario": "S0", "condition": "N", "n_length_stopped": 1}])


class TestF7Floor(unittest.TestCase):
    def test_fclip(self):
        self.assertEqual(cf.fclip(0.0001), 0.005)
        self.assertEqual(cf.fclip(0.9999), 0.995)
        self.assertEqual(cf.fclip(0.3), 0.3)


class TestOnRealData(unittest.TestCase):
    """Runs the script on the real raw records into a temp dir (no paper or derived files touched)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.res, cls.mac = cf.main(["--out", os.path.join(cls.tmp, "d"), "--generated", os.path.join(cls.tmp, "g"),
                                    "--figures", os.path.join(cls.tmp, "f"), "--n-boot", "50", "--no-figures"])

    def test_overestimation_matches_make_tables(self):
        p = os.path.join(ROOT, "results", "derived", "table_overestimation.csv")
        if not os.path.exists(p):
            self.skipTest("derived table absent")
        with open(p) as fh:
            ref = {r["model"]: r for r in csv.DictReader(fh)}
        for o in self.res["over"]:
            if o["sample"] == "all":
                self.assertAlmostEqual(o["median_signed_lod"], float(ref[o["model"]]["median_signed_lod"]), places=9)
                self.assertAlmostEqual(o["pct_above_target"], float(ref[o["model"]]["pct_above_target"]), places=9)

    def test_clinician_counts_match_compare_clinicians(self):
        p = os.path.join(ROOT, "results", "derived", "clinician_compare_summary.csv")
        if not os.path.exists(p):
            self.skipTest("derived table absent")
        with open(p) as fh:
            ref = {(r["model"], r["condition"]): r for r in csv.DictReader(fh)}
        for c in self.res["clin"]:
            if c["sample"] == "all":
                r = ref[(c["model"], c["condition"])]
                self.assertEqual(c["n_model_closer"], int(r["n_post_model_closer"]))
                self.assertEqual(c["n_tie"], int(r["n_post_tie"]))

    def test_ratio_shape(self):
        pooled = [s for s in self.res["ratio"] if s["condition"] == "VNBS" and s["model"] != "ALL"]
        self.assertEqual(len(pooled), 5)
        for s in pooled:
            self.assertEqual(s["n_scen"], 10)
            self.assertLessEqual(s["loso_min"], s["b_fe"])
            self.assertGreaterEqual(s["loso_max"], s["b_fe"])

    def test_no_uti_sets_are_nine_scenarios(self):
        for f in self.res["fits"]:
            if f["analysis"].endswith("no_A4"):
                self.assertEqual(f["n_clusters"], 9)

    def test_macro_names_letters_only(self):
        for k, v in self.mac.items():
            self.assertTrue(k.isalpha() and k.startswith("cf"), k)
            self.assertNotEqual(v, "")

    def test_clinician_midpoint_set_a_only(self):
        clin = [m for m in self.res["mid"] if m["who"] == "clinicians_pooled_median"]
        self.assertEqual({m["sample"]: m["n_cases"] for m in clin}, {"all": 8, "no_A4": 6})

    def test_contrasts_ten_clusters(self):
        for c in self.res["ctr"]:
            self.assertEqual(c["n_scen"], 10)
            self.assertLessEqual(c["lo"], c["mean"])
            self.assertGreaterEqual(c["hi"], c["mean"])

    def test_fit_rows_reproduce_prespecified_fit(self):
        """Per model x condition, OLS on cf.fit_rows equals the pre-specified published fit in wbayes.csv."""
        from fit_wbayes import ols
        from common import load_config
        p = os.path.join(ROOT, "results", "derived", "wbayes.csv")
        if not os.path.exists(p):
            self.skipTest("wbayes.csv absent")
        cfg = load_config()
        cases, final, rows, syn = cf.load_rows(os.path.join(ROOT, "results", "raw", "*.jsonl"),
                                               os.path.join(ROOT, "data", "cases"), cfg["paper_run_ids"], cfg)
        with open(p) as fh:
            wb = [r for r in csv.DictReader(fh) if r["variant"] == "published"]
        self.assertTrue(wb)
        for w in wb:
            sel = [r for r in rows if r["model"] == w["model"] and r["condition"] == w["condition"]]
            est = ols([x for xs in cf.fit_rows(sel).values() for x in xs])
            self.assertAlmostEqual(est[0], float(w["a"]), places=6)
            self.assertAlmostEqual(est[1], float(w["b"]), places=6)

    def test_decomposition_rows(self):
        self.assertEqual(len(self.res["dec"]), 12)

    def test_real_raw_untouched(self):
        # the script must never write into results/raw
        self.assertFalse(any(n.startswith("cf_") for n in os.listdir(os.path.join(ROOT, "results", "raw"))))


class TestFirewall(unittest.TestCase):
    def test_synthetic_refused_for_real_dirs(self):
        with self.assertRaises(AssertionError):
            cf.main(["--raw", FIXTURE, "--run-ids", "SYNTHETIC", "--n-boot", "5", "--no-figures"])


if __name__ == "__main__":
    unittest.main()
