"""Tests for analysis/ext6_analysis.py (I4 prompt paraphrases and). Real-data assertions on the key
numbers plus the invariant that no pre-existing generated macro file is redefined."""
import glob
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ext6_analysis as e6  # noqa: E402
from common import ROOT  # noqa: E402


def macro_names(path):
    with open(path) as f:
        return set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read()))


class TestExt6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.res, cls.mac = e6.main(["--out", cls.tmp, "--generated", cls.tmp])

    def get(self, arm, m="ALL"):
        return e6.get(self.res["summ"], arm, m)

    def test_counts(self):
        self.assertEqual(self.mac["extSixNTotalPara"], "600")
        self.assertEqual(self.mac["extSixNUnparsedTotal"], "3")
        self.assertEqual(self.get("ParaOne")["n_parsed"], 299)
        self.assertEqual(self.get("ParaTwo")["n_parsed"], 298)
        self.assertEqual(self.get("Main")["n_total"], 300)      # seeds 1 to 3 only
        self.assertEqual(self.get("Main")["n_parsed"], 297)

    def test_unparsed_are_deepseek_only(self):
        bad = [p for p in self.res["parse"] if p["arm"] != "Main"]
        self.assertEqual(len(bad), 3)
        self.assertTrue(all("deepseek" in p["model"] for p in bad))

    def test_no_b7_b8(self):
        self.assertFalse(any(c.startswith(("B7_", "B8_")) for c in self.res["cases"]))
        self.assertEqual(len(self.res["cases"]), 20)

    def test_headline_values(self):
        self.assertAlmostEqual(self.get("Main")["mean_band_err_pp"], 10.632, places=2)
        self.assertAlmostEqual(self.get("ParaOne")["mean_band_err_pp"], 10.546, places=2)
        self.assertAlmostEqual(self.get("ParaTwo")["mean_band_err_pp"], 10.645, places=2)
        for arm in ("Main", "ParaOne", "ParaTwo"):
            self.assertEqual(self.mac[f"extSix{arm}RModelsGtOne"], "5")
            self.assertGreater(self.get(arm)["r_med"], 1.0)
        self.assertEqual(self.mac["extSixMaterial"], "no")
        self.assertEqual(self.mac["extSixRCrossesOne"], "no")

    def test_macro_names_new_and_letters_only(self):
        gen = os.path.join(ROOT, "paper", "generated")
        mine = macro_names(os.path.join(self.tmp, "ext6_macros.tex"))
        self.assertTrue(all(m.startswith("extSix") and m.isalpha() for m in mine))
        for p in glob.glob(os.path.join(gen, "*.tex")):
            if os.path.basename(p) in ("ext6_macros.tex", "table_ext6_rows.tex"):
                continue
            self.assertFalse(mine & macro_names(p), p)

    def test_no_em_dash_in_outputs(self):
        for n in ("ext6_macros.tex", "table_ext6_rows.tex"):
            with open(os.path.join(self.tmp, n)) as f:
                self.assertNotIn("\u2014", f.read())

    def test_table_rows(self):
        with open(os.path.join(self.tmp, "table_ext6_rows.tex")) as f:
            rows = [x for x in f.read().splitlines() if x.startswith("E6 para")]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r.count("&") == 5 and r.endswith("\\\\") for r in rows))


if __name__ == "__main__":
    unittest.main()
