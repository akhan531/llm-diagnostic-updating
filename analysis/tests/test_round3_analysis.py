"""Tests for analysis/round3_analysis.py (round-3 simulated review). Asserts the key numbers on the real raw data, so a
silent change in the data or the rules fails loudly, and the invariants: no existing macro redefined, letters-only
names, no em dash, no resizebox, the pre-specified contrasts reproduce results/derived/table_contrasts.csv."""
import csv
import glob
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import round3_analysis as r3  # noqa: E402
from common import ROOT  # noqa: E402


def pick(rows, **kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


class TestRound3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.res, cls.mac = r3.main(["--out", cls.tmp, "--generated", cls.tmp, "--n-boot", "2000"])

    def test_item1_blind_prior(self):
        b = {(s["sample"], s["model"]): s for s in self.res["blind"]}
        a = b[("all", "ALL")]
        self.assertEqual((a["n_scen"], a["n_above"]), (10, 9))
        self.assertAlmostEqual(a["med_pp"], 4.77, places=2)
        self.assertAlmostEqual(a["med_lod"], 0.77, places=2)
        self.assertAlmostEqual(a["signflip_p"], 0.0039, places=3)
        n = b[("no_A4", "ALL")]
        self.assertEqual((n["n_scen"], n["n_above"]), (9, 8))
        self.assertAlmostEqual(n["mean_lod"], 0.70, places=2)
        counts = [b[("all", m)]["n_above"] for m in self.res["models"]]
        self.assertEqual(counts, [7, 8, 9, 10, 9])

    def test_item1_range(self):
        rs = {(s["sample"], s["model"]): s for s in self.res["range"]}
        a = rs[("all", "ALL")]
        self.assertEqual((a["n_range"], a["n_point_only"], a["n_above"], a["n_inside"], a["n_below"]), (9, 1, 8, 1, 0))
        pos = {c["scenario"]: c["kind"] for c in self.res["range_cells"]}
        self.assertEqual(pos["B4_aas"], "point_only")
        self.assertEqual(pos["A1_pneumonia"], "morgan_range")
        self.assertEqual(pos["B2_pe_mod"], "source_ci")
        self.assertEqual(rs[("all", "openai/gpt-4o")]["n_above"], 9)
        for s in rs.values():
            self.assertAlmostEqual(s["pct_ans_above"] + s["pct_ans_inside"] + s["pct_ans_below"], 100.0, places=6)

    def test_item2_decomposition(self):
        d = {(x["sample"], x["model"], x["result"]): x for x in self.res["decomp"]}
        self.assertAlmostEqual(d[("all", "ALL", "pos")]["prior_mean"], 1.40, places=2)      # reproduces F9
        self.assertAlmostEqual(d[("no_A4", "ALL", "pos")]["prior_mean"], 1.06, places=2)
        self.assertAlmostEqual(d[("no_A4", "ALL", "pos")]["update_mean"], 0.67, places=2)
        self.assertAlmostEqual(d[("no_A4", "ALL", "neg")]["prior_mean"], 0.68, places=2)
        self.assertAlmostEqual(d[("no_A4", "ALL", "neg")]["update_mean"], -0.51, places=2)
        self.assertEqual(d[("no_A4", "ALL", "pos")]["n"], 750)
        sh = {(x["model"], x["result"]): x for x in self.res["share"]}
        self.assertAlmostEqual(sh[("ALL", "pos")]["a4_share_of_prior_mean"], 0.31, places=2)
        self.assertAlmostEqual(sh[("ALL", "neg")]["a4_share_of_prior_mean"], 0.40, places=2)
        self.assertAlmostEqual(sh[("ALL", "both")]["a4_share_of_prior_mean"], 0.35, places=2)

    def test_item3_belief(self):
        sh = {s["model"]: s for s in self.res["bshare"]}
        a = sh["ALL"]
        self.assertAlmostEqual(a["r_main_n"], 1.51, places=2)
        self.assertAlmostEqual(a["r_ext2_published"], 1.23, places=2)
        self.assertAlmostEqual(a["r_ext2_believed"], 1.13, places=2)
        self.assertAlmostEqual(a["share_from_believed_lr"] + a["share_from_different_answers"], 1.0, places=9)
        self.assertAlmostEqual(a["share_from_believed_lr"], 0.26, places=2)
        d = pick(self.res["rdiff"], model="ALL")
        self.assertEqual((d["n_scen"], d["n_pos"]), (10, 0))
        self.assertAlmostEqual(d["mean_diff"], -0.21, places=2)
        f = {(x["arm"], x["variant"], x["model"]): x for x in self.res["fits"]}
        self.assertAlmostEqual(f[("ext2_N", "own_pretest_published_lr", "ALL")]["b"], 1.11, places=2)
        self.assertAlmostEqual(f[("ext2_N", "own_pretest_believed_lr", "ALL")]["b"], 1.03, places=2)
        self.assertAlmostEqual(f[("main1_N", "own_pretest_published_lr", "ALL")]["b"], 1.27, places=2)
        # the existing E2 b against the believed LR is reproduced (same estimator, same data)
        self.assertAlmostEqual(f[("ext2_N", "own_pretest_believed_lr", "openai/gpt-6.1-sol")]["b"], 0.99935, places=4)
        self.assertEqual(self.res["n_e2"], 299)

    def test_item4_without_a4(self):
        n = {(h["headline"], h["model"]): h for h in self.res["noa4"]}
        h = n[("hindsight_pos_minus_neg_stated_pretest", "ALL")]
        self.assertEqual((h["n_scen"], h["n_pos"]), (9, 9))
        self.assertAlmostEqual(h["estimate"], 6.89, places=2)
        self.assertAlmostEqual(h["p"], 2 / 2 ** 9, places=6)
        r = n[("ratio_r_median", "ALL")]
        self.assertEqual((r["n_scen"], r["n_pos"]), (9, 7))
        self.assertAlmostEqual(r["estimate"], 1.48, places=2)
        self.assertLess(r["lo"], 1.0)
        c = n[("contrast_NvsV_H1", "ALL")]
        self.assertAlmostEqual(c["estimate"], 0.12, places=2)
        self.assertLess(c["lo"], 0.0)
        self.assertGreater(c["hi"], 0.0)
        e = n[("contrast_EvsN_H4", "ALL")]
        self.assertAlmostEqual(e["estimate"], 0.14, places=2)
        self.assertLess(e["lo"], 0.0)

    def test_item5_prespec_matches_existing_table(self):
        with open(os.path.join(ROOT, "results", "derived", "table_contrasts.csv")) as f:
            old = {(r["model"], r["set"], r["result"], r["contrast"], r["metric"]): r for r in csv.DictReader(f)}
        self.assertEqual(len(self.res["prespec"]), 5 * 2 * 3 * 2)   # 2 metrics per set, both stored
        for c in self.res["prespec"]:
            o = old[(c["model"], c["set"], "all", c["contrast"], c["metric"])]
            self.assertAlmostEqual(c["median_diff"], float(o["median_diff"]), places=9)
            self.assertAlmostEqual(c["ci_lo"], float(o["ci_lo"]), places=9)
            self.assertAlmostEqual(c["ci_hi"], float(o["ci_hi"]), places=9)
        with open(os.path.join(self.tmp, "table_prespec.tex")) as f:
            t = f.read()
        self.assertNotIn("resizebox", t)
        self.assertIn("\\footnotesize", t)
        self.assertEqual(t.count("H1: N"), 5)

    def test_macros_new_letters_only_no_em_dash(self):
        existing = set()
        for p in glob.glob(os.path.join(ROOT, "paper", "generated", "*.tex")):
            if os.path.basename(p) != "round3_macros.tex":
                with open(p) as f:
                    existing |= set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read()))
        for k, v in self.mac.items():
            self.assertTrue(k.isalpha() and k.startswith("rthree"), k)
            self.assertNotIn(k, existing, k)
            self.assertNotIn("\u2014", v)
        for fn in ("round3_macros.tex", "table_prespec.tex"):
            with open(os.path.join(self.tmp, fn)) as f:
                self.assertNotIn("\u2014", f.read())
        self.assertEqual(self.mac["rthreeBlindAllNAbove"], "9")
        self.assertEqual(self.mac["rthreeBlindNoUtiAllNAbove"], "8")

    def test_committed_outputs_match_regeneration(self):
        for fn in ("round3_macros.tex", "table_prespec.tex"):
            with open(os.path.join(self.tmp, fn)) as f:
                new = f.read()
            with open(os.path.join(ROOT, "paper", "generated", fn)) as f:
                self.assertEqual(new, f.read(), fn)


if __name__ == "__main__":
    unittest.main()
