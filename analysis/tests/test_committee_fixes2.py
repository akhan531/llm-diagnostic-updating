"""Tests for analysis/committee_fixes2.py (F4 alternative targets, F5 H4 by group, F8 thresholds)."""
import math
import os
import statistics
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import committee_fixes as cf  # noqa: E402
import committee_fixes2 as cf2  # noqa: E402
from common import ROOT, load_cases  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "synthetic.jsonl")
CASES = load_cases(os.path.join(ROOT, "data", "cases"))


class TestF4Specs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = cf2.alt_specs(CASES, cf2.load_json(cf2.ALT))

    def test_pioped_bands_match_note(self):
        #: B2 pos band 72.0-84.9, B2 neg 2.46-4.44, B3 neg 7.24-12.59
        b = self.s["lr_ci_band"]
        self.assertAlmostEqual(100 * b["B2_pe_mod_pos"]["band"][0], 72.0, delta=0.06)
        self.assertAlmostEqual(100 * b["B2_pe_mod_pos"]["band"][1], 84.9, delta=0.06)
        self.assertAlmostEqual(100 * b["B2_pe_mod_neg"]["band"][0], 2.46, delta=0.006)
        self.assertAlmostEqual(100 * b["B3_pe_high_neg"]["band"][1], 12.59, delta=0.006)
        self.assertEqual(b["B2_pe_mod_pos"]["lr"], 29.0)  # steep end of the CI
        self.assertEqual(b["B2_pe_mod_neg"]["lr"], 0.13)

    def test_observed_lr_reproduces_observed_value(self):
        for cid, o in self.s["observed"].items():
            self.assertAlmostEqual(cf2.bayes(CASES[cid]["target"]["pretest_point"], o["lr"]), o["target"], places=12)
        self.assertAlmostEqual(self.s["observed"]["B3_pe_high_neg"]["target"], 0.4)

    def test_pickhardt_targets(self):
        p = self.s["pickhardt"]
        self.assertAlmostEqual(100 * p["B5_appy_ct_pos"]["target"], 94.0, delta=0.05)
        self.assertAlmostEqual(100 * p["B5_appy_ct_neg"]["target"], 0.46, delta=0.005)
        self.assertAlmostEqual(self.s["pickhardt_recomputed"]["B5_appy_ct_pos"]["lr"], 50.3)
        self.assertEqual(set(self.s["all_alt"]), set(self.s["observed"]) | set(p))
        self.assertEqual(self.s["primary"], {})

    def test_signed_err_band(self):
        r = {"post": 0.5, "band": (0.4, 0.6), "target": 0.45}
        self.assertEqual(cf2.signed_err(r), 0.0)
        r["post"] = 0.7
        self.assertAlmostEqual(cf2.signed_err(r), math.log(0.7 / 0.3) - math.log(0.6 / 0.4))
        r["band"] = None
        self.assertAlmostEqual(cf2.signed_err(r), math.log(0.7 / 0.3) - math.log(0.45 / 0.55))


class TestF5Groups(unittest.TestCase):
    def test_classification(self):
        g = cf2.e_groups()
        self.assertEqual(sorted(s for s, v in g.items() if not v["any_component"]), ["A1_pneumonia", "B8_septic"])
        self.assertEqual(sorted(s for s, v in g.items() if v["score_item"]),
                         ["B1_pe_low", "B2_pe_mod", "B3_pe_high", "B4_aas", "B5_appy_ct", "B6_appy_us", "B7_strep"])
        self.assertTrue(g["A3_cardiac"]["any_component"])  # vignette definition decides A3's group
        self.assertFalse(g["A3_cardiac"]["score_item"])

    def test_perm_diff(self):
        self.assertAlmostEqual(cf2.perm_diff_p([1, 2, 3, 4], [True, True, False, False]), 2 / 6)
        self.assertAlmostEqual(cf2.perm_diff_p([1, 1, 1, 1], [True, False, True, False]), 1.0)


class TestF8Thresholds(unittest.TestCase):
    def test_specs(self):
        sp = {s["id"]: s for s in cf2.threshold_specs(CASES)}
        self.assertEqual(sp["pe_test_kline"]["value"], 0.018)
        self.assertEqual(sp["pe_test_lessler"]["value"], 0.014)
        self.assertEqual(sp["pneumonia_test"]["value"], 0.095)
        self.assertEqual(sp["pneumonia_treat"]["value"], 0.431)
        self.assertEqual(sp["appy_upper_ct"]["case_ids"], ["B5_appy_ct_neg", "B5_appy_ct_pos"])
        self.assertTrue(all(s["primary"] == (s["id"] in cf2.PRIMARY) for s in sp.values()))
        self.assertFalse(any("uti" in i or "hypothetical" in i for i in sp))

    def test_fragile_and_side(self):
        self.assertTrue(cf2.is_fragile(0.0221, 0.018))
        self.assertFalse(cf2.is_fragile(0.0335, 0.018))
        self.assertTrue(cf2.is_fragile(0.012, 0.018))
        self.assertEqual(cf2.side(0.018, 0.018), "above")

    def test_discordance_counts(self):
        sp = [{"id": "t", "label": "t", "value": 0.1, "case_ids": ["X_neg", "X_pos"], "primary": True}]
        rows = [{"model": "M", "case_id": "X_neg", "condition": "V", "target": 0.05, "post": p} for p in (0.02, 0.2)]
        rows += [{"model": "M", "case_id": "X_pos", "condition": "N", "target": 0.5, "post": p} for p in (0.05, 0.6, 0.7)]
        per, cs = cf2.discordance(rows, sp)
        p = [x for x in per if x["model"] == "M" and x["cond_group"] == "VNBS" and not x["excl_fragile"]][0]
        self.assertEqual((p["n_false_alarm"], p["n_target_below"]), (1, 2))
        self.assertEqual((p["n_false_reassurance"], p["n_target_above"]), (1, 3))
        self.assertAlmostEqual(p["discordance_rate"], 40.0)


class TestOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.res, cls.mac = cf2.main(["--out", os.path.join(cls.tmp, "d"), "--generated", os.path.join(cls.tmp, "g"),
                                     "--n-boot", "200"])
        cls.cfres, cls.cfmac = cf.main(["--out", os.path.join(cls.tmp, "d1"), "--generated", os.path.join(cls.tmp, "g1"),
                                        "--n-boot", "200", "--no-figures"])

    def test_primary_variant_reproduces_first_pass(self):
        for o in self.res["alt"]:
            if o["variant"] != "primary" or o["model"] == "ALL":
                continue
            t = cf.model_tag(o["model"])
            self.assertEqual(cf2.tex_num(o["ratio_med"]), self.cfmac["cfRatio" + t + "Med"])
            self.assertEqual(cf2.tex_num(o["b"]), self.cfmac["cfPool" + t + "ParBPt"])
        over = {o["model"]: o for o in self.cfres["over"] if o["sample"] == "all"}
        for o in self.res["alt"]:
            if o["variant"] == "primary" and o["model"] != "ALL":
                self.assertAlmostEqual(o["over_med_lod"], over[o["model"]]["median_signed_lod"], places=12)

    def test_b3_primary_matches_make_tables(self):
        b = [x for x in self.res["b3"] if x["variant"] == "primary"][0]
        self.assertAlmostEqual(b["target_pct"], 9.69, delta=0.01)
        self.assertEqual(b["median_post_pct"], 5.0)

    def test_h4_groups_recombine(self):
        allc = [r for r in self.cfres["ctr"] if r["contrast"] == "EvsN" and r["model"] == "ALL"][0]
        for split in ("any_component", "score_item"):
            g = [o for o in self.res["h4"] if o["split"] == split and o["model"] == "ALL"]
            self.assertEqual(sum(o["n_scen"] for o in g), 10)
            self.assertAlmostEqual(sum(o["mean"] * o["n_scen"] for o in g) / 10, allc["mean"], places=12)

    def test_threshold_rows(self):
        p = [x for x in self.res["thr"] if x["threshold"] == "pe_test_kline" and x["model"] == "ALL"
             and x["cond_group"] == "VNBS" and not x["excl_fragile"]][0]
        self.assertEqual(p["n_target_below"] + p["n_target_above"], p["n"])
        self.assertEqual(p["fragile_cases"], "B1_pe_low_pos")
        a1 = [x for x in self.res["thr"] if x["threshold"] == "pneumonia_test" and x["model"] == "ALL"
              and x["cond_group"] == "VNBS" and not x["excl_fragile"]][0]
        self.assertEqual(a1["fragile_cases"], "A1_pneumonia_neg")

    def test_macros_letters_only(self):
        for k, v in self.mac.items():
            self.assertTrue(k.isalpha(), k)
            self.assertNotIn("\u2014", v)

    def test_real_raw_untouched(self):
        raw = os.path.join(ROOT, "results", "raw")
        self.assertTrue(all(not f.startswith("cf2") for f in os.listdir(raw)))


class TestFirewall(unittest.TestCase):
    def test_synthetic_refused_for_real_dirs(self):
        with self.assertRaises(AssertionError):
            cf2.main(["--raw", FIXTURE, "--run-ids", "SYNTHETIC", "--n-boot", "5"])


if __name__ == "__main__":
    unittest.main()
