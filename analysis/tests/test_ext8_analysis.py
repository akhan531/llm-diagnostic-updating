"""Tests for analysis/ext8_analysis.py (extra family qwen/qwen4-max, robustness row). Real-data assertions on the
key numbers plus the invariants: macro names new and letters only, no existing generated macro redefined, no em dash,
Qwen never in the roster."""
import glob
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ext8_analysis as e8  # noqa: E402
from common import ROOT  # noqa: E402


def macro_names(path):
    with open(path) as f:
        return set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read()))


class TestExt8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.res, cls.mac = e8.main(["--out", cls.tmp, "--generated", cls.tmp])

    def test_parse_counts(self):
        m = self.mac
        self.assertEqual((m["extEightQwenNRecords"], m["extEightQwenNKeys"]), ("187", "186"))
        self.assertEqual((m["extEightQwenNParsed"], m["extEightQwenNUnparsed"]), ("186", "0"))
        self.assertEqual((m["extEightQwenNRetried"], m["extEightQwenNFinalCallError"]), ("1", "0"))
        self.assertEqual((m["extEightQwenNAnalysed"], m["extEightQwenNCalibration"]), ("180", "6"))

    def test_signed_e_minus_n_qwen(self):
        m = self.mac
        self.assertEqual((m["extEightCtrEvsNSignedQwenMean"], m["extEightCtrEvsNSignedQwenLo"],
                          m["extEightCtrEvsNSignedQwenHi"]), ("0.20", "0.06", "0.34"))
        self.assertEqual((m["extEightCtrEvsNSignedQwenP"], m["extEightCtrEvsNSignedQwenNPos"],
                          m["extEightCtrEvsNSignedQwenNScen"]), ("0.014", "8", "10"))
        self.assertEqual(m["extEightCtrEvsNQwenMean"], "0.17")   # absolute-error version unchanged

    def test_roster_excludes_qwen_and_has_five_models(self):
        self.assertEqual(len(self.res["models"]), 5)
        self.assertNotIn(e8.QWEN, self.res["models"])

    def test_error_cells(self):
        self.assertEqual(self.mac["extEightQwenVN"], "60")
        self.assertEqual(self.mac["extEightQwenVAbsLod"], "1.26")
        self.assertEqual(self.mac["extEightQwenNAbsLod"], "1.33")
        self.assertEqual(self.mac["extEightQwenEAbsLod"], "1.49")

    def test_contrasts(self):
        self.assertEqual(self.mac["extEightCtrNvsVQwenMean"], "0.07")
        self.assertEqual(self.mac["extEightCtrNvsVQwenP"], "0.701")
        self.assertEqual(self.mac["extEightCtrEvsNQwenMean"], "0.17")
        self.assertEqual(self.mac["extEightCtrEvsNQwenNPos"], "8")
        self.assertEqual(self.mac["extEightCtrNvsVQwenNScen"], "10")

    def test_ratio_hindsight_pretest(self):
        self.assertEqual(self.mac["extEightRVNQwenMed"], "1.34")
        self.assertEqual(self.mac["extEightRVNQwenNGtOne"], "8")
        self.assertEqual(self.mac["extEightHindNQwenNPos"], "10")
        self.assertEqual(self.mac["extEightHindNQwenNScen"], "10")
        self.assertEqual(self.mac["extEightPreVNQwenNExceed"], "8")
        self.assertEqual(self.mac["extEightPreNQwenNExceed"], "8")
        self.assertEqual(self.mac["extEightRVNRosterModelsGtOne"], "5")

    def test_macro_names_new_and_letters_only(self):
        gen = os.path.join(ROOT, "paper", "generated")
        mine = macro_names(os.path.join(self.tmp, "ext8_macros.tex"))
        self.assertTrue(all(m.startswith("extEight") and m.isalpha() for m in mine))
        for p in glob.glob(os.path.join(gen, "*.tex")):
            if os.path.basename(p) in ("ext8_macros.tex", "table_ext8_rows.tex"):
                continue
            self.assertFalse(mine & macro_names(p), p)

    def test_no_em_dash_in_outputs(self):
        for p in glob.glob(os.path.join(self.tmp, "*")):
            with open(p) as f:
                self.assertNotIn("\u2014", f.read(), p)

    def test_table_row(self):
        with open(os.path.join(self.tmp, "table_ext8_rows.tex")) as f:
            rows = [x for x in f.read().splitlines() if x.startswith("E8 Qwen")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].count("&"), 5)
        self.assertTrue(rows[0].endswith("\\\\") and "--" in rows[0])

    def test_committed_outputs_match_regeneration(self):
        """The macro file in paper/generated equals a fresh run (a silent change fails here)."""
        committed = os.path.join(ROOT, "paper", "generated", "ext8_macros.tex")
        if os.path.exists(committed):
            with open(committed) as a, open(os.path.join(self.tmp, "ext8_macros.tex")) as b:
                self.assertEqual(a.read(), b.read())


class TestRetryCounts(unittest.TestCase):
    def test_not_resent_macro(self):
        with open(os.path.join(ROOT, "paper", "generated", "retry_macros.tex")) as f:
            t = f.read()
        self.assertIn("\\newcommand{\\nNotResentNoReply}{2}", t)
        self.assertIn("\\newcommand{\\nRetriedNoReplyRunExtEight}{1}", t)


if __name__ == "__main__":
    unittest.main()
