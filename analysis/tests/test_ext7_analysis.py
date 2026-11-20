"""Tests for analysis/ext7_analysis.py (12 scenarios extension to). Real-data assertions on the key numbers
plus the invariant that no pre-existing generated macro file is redefined."""
import glob
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ext7_analysis as e7  # noqa: E402
from common import ROOT  # noqa: E402


def macro_names(path):
    with open(path) as f:
        return set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read()))


class TestExt7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.B, cls.mac, cls.vr = e7.main(["--out", cls.tmp, "--generated", cls.tmp])

    def test_counts(self):
        self.assertEqual(self.mac["extSevenNTotal"], "390")
        self.assertEqual(self.mac["extSevenNUnparsed"], "2")
        self.assertEqual(self.mac["extSevenNCallError"], "1")
        self.assertEqual(self.mac["extSevenNAnalysedRows"], "298")

    def test_cluster_counts(self):
        for m in self.B["models"] + ["ALL"]:
            self.assertEqual(e7.pick(self.B["heads"]["Ten"]["r_sum"], m)["n_scen"], 10)
            self.assertEqual(e7.pick(self.B["heads"]["Twelve"]["r_sum"], m)["n_scen"], 12)

    def test_ten_arm_reproduces_headlines(self):
        h = self.B["heads"]["Ten"]
        self.assertEqual(self.mac["extSevenTenRModelsGtOne"], "5")
        self.assertEqual(self.mac["extSevenTenAllRNGtOne"], "8")
        self.assertEqual(self.mac["extSevenTenHolmPooledHindAdjP"], "0.014")
        self.assertAlmostEqual(e7.pick(h["r_sum"], "ALL")["median"], 1.475, places=3)

    def test_twelve_arm_values(self):
        self.assertEqual(self.mac["extSevenTwelveRModelsGtOne"], "5")
        self.assertEqual(self.mac["extSevenTwelveAllRNGtOne"], "10")
        self.assertEqual(self.mac["extSevenTwelveAllRSignP"], "0.039")
        self.assertEqual(self.mac["extSevenTwelveHolmPooledRatioAdjP"], "0.193")
        self.assertEqual(self.mac["extSevenTwelveHolmPooledHindAdjP"], "0.003")
        self.assertEqual(self.mac["extSevenTwelveStatedNExceed"], "10")
        self.assertEqual(self.mac["extSevenTwelveStatedNScen"], "12")
        self.assertEqual(self.mac["extSevenTwelveHolmHindNSurviveWithin"], "5")
        self.assertEqual(self.mac["extSevenTwelveHolmRatioNSurviveWithin"], "0")

    def test_seed_matched_close_to_all_seed(self):
        self.assertLess(abs(float(self.mac["extSevenSeedMatchedDeltaR"].replace("\\ensuremath{-}", "-"))), 0.05)

    def test_new_scenarios_below_published_prior(self):
        self.assertEqual(self.mac["extSevenStrepAllPreExceeds"], "no")
        self.assertEqual(self.mac["extSevenSepticAllPreExceeds"], "no")

    def test_macro_names_new_and_letters_only(self):
        gen = os.path.join(ROOT, "paper", "generated")
        mine = macro_names(os.path.join(self.tmp, "ext7_macros.tex"))
        self.assertTrue(all(m.startswith("extSeven") and m.isalpha() for m in mine))
        for p in glob.glob(os.path.join(gen, "*.tex")):
            if os.path.basename(p) in ("ext7_macros.tex", "table_ext7_rows.tex"):
                continue
            self.assertFalse(mine & macro_names(p), p)

    def test_no_em_dash_in_outputs(self):
        for p in glob.glob(os.path.join(self.tmp, "*")):
            with open(p) as f:
                self.assertNotIn("\u2014", f.read(), p)

    def test_table_rows(self):
        with open(os.path.join(self.tmp, "table_ext7_rows.tex")) as f:
            rows = [x for x in f.read().splitlines() if x.startswith("E7")]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r.count("&") == 5 and r.endswith("\\\\") for r in rows))


if __name__ == "__main__":
    unittest.main()
