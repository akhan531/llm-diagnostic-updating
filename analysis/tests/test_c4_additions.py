"""C4 additions: response breakdown, H3 sensitivity, midpoint error, H5 split, near-one bounds, M3.
Read-only against results/derived and results/raw; nothing is written to the repo."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import make_tables as mt  # noqa: E402

DERIVED = os.path.join(common.ROOT, "results", "derived")
RAW = os.path.join(common.ROOT, "results", "raw", "*.jsonl")
CASES = common.load_cases(os.path.join(common.ROOT, "data", "cases"))
SCORED = common.read_csv(os.path.join(DERIVED, "scored.csv"))
MACROS = os.path.join(common.ROOT, "paper", "generated", "results_macros.tex")


def macros():
    import re
    out = {}
    for line in open(MACROS):
        m = re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{(.+)\}\n?", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


class ResponseBreakdown(unittest.TestCase):
    def test_counts_add_up(self):
        rb = mt.response_breakdown(SCORED)
        self.assertEqual((rb["total"], rb["cal"], rb["noncal"], rb["noncal_parsed"], rb["noncal_unparsed"]),
                         (2142, 42, 2100, 2078, 22))
        self.assertEqual(rb["cal_parsed"], 42)
        self.assertEqual(rb["total"], rb["cal"] + rb["noncal"])
        self.assertEqual(rb["noncal"], rb["noncal_parsed"] + rb["noncal_unparsed"])

    def test_macros_match(self):
        m = macros()
        self.assertEqual([m[k] for k in ("resNResponses", "resNCalibAnswers", "resNNonCalibAnswers",
                                         "resNNonCalibParsed", "resNNonCalibUnparsed")],
                         ["2142", "42", "2100", "2078", "22"])
        self.assertEqual(int(m["resNParseFail"]), int(m["resNNonCalibUnparsed"]))  # all 22 failures are non-calibration


class UnparsedSensitivity(unittest.TestCase):
    def setUp(self):
        self.sens = mt.unparsed_sensitivity(SCORED, CASES, [RAW], 2000)

    def test_cells_change(self):
        s = self.sens
        self.assertIsNotNone(s)
        mac = mt.sensitivity_macros(s)
        self.assertEqual(mac["resUnparsedSensGptBENStrict"], "0.019")
        self.assertEqual(mac["resUnparsedSensGptBENHand"], "\\ensuremath{-}0.010")
        self.assertEqual(mac["resUnparsedSensGptBEBandStrict"], "7.7")
        self.assertEqual(mac["resUnparsedSensGptBEBandHand"], "8.2")
        self.assertEqual((mac["resUnparsedSensHFourNCellsPositiveSetBStrict"],
                          mac["resUnparsedSensHFourNCellsPositiveSetBHand"]), (5, 4))
        # every changed cell is GPT-4o, Set B, and involves condition E
        for c in s["changed"]:
            self.assertEqual((c["model"], c["set"]), ("openai/gpt-4o", "B"))
            self.assertTrue(c["condition"] == "E" or c["contrast"] == "EvsN", c)
        self.assertEqual(mac["resUnparsedSensCells"], 2)

    def test_matches_written_macros(self):
        m, mac = macros(), mt.sensitivity_macros(self.sens)
        for k, v in mac.items():
            self.assertEqual(str(m[k]), str(v), k)

    def test_wrong_identifier_or_tail_fails(self):
        bad = dict(mt.UNPARSED_HAND_READ, response_tail="- PRETEST: 99\n- POSTTEST: 99")
        with self.assertRaises(AssertionError):
            mt.unparsed_sensitivity(SCORED, CASES, [RAW], 50, bad)
        bad = dict(mt.UNPARSED_HAND_READ, timestamp="2000-01-01T00:00:00+00:00")
        with self.assertRaises(AssertionError):
            mt.unparsed_sensitivity(SCORED, CASES, [RAW], 50, bad)

    def test_absent_record_returns_none(self):
        self.assertIsNone(mt.unparsed_sensitivity(SCORED, CASES, [RAW], 50, dict(mt.UNPARSED_HAND_READ, seed=999)))


class H4Pairs(unittest.TestCase):
    def test_pairs_are_model_by_case(self):
        h4 = mt.h4_pooled(SCORED)
        self.assertEqual((h4["A"]["n_pairs"], h4["B"]["n_pairs"]), (40, 60))
        self.assertEqual((h4["A"]["n_cases_max"] * h4["A"]["n_cells"], h4["B"]["n_cases_max"] * h4["B"]["n_cells"]), (40, 60))
        m = macros()
        self.assertEqual((m["hFourNCasesPerModelSetA"], m["hFourNCasesPerModelSetB"]), ("8", "12"))


class MidpointError(unittest.TestCase):
    def test_toy(self):
        cases = {"c": {"target": {"posttest_band_primary": [0.10, 0.20]}}}
        rows = [{"model": "m", "set": "A", "condition": "V", "calibration": "0", "parse_ok": "1", "case_id": "c",
                 "posttest_stated": v, "band_err_pp": b} for v, b in (("20", "0"), ("25", "5"), ("10", "0"))]
        r = mt.midpoint_error_table(rows, cases)[0]
        self.assertAlmostEqual(r["mid_mae_pp"], (5 + 10 + 5) / 3)
        self.assertAlmostEqual(r["band_mae_pp"], 5 / 3)
        self.assertIn("SECONDARY", r["label"])

    def test_real_not_below_band_error(self):
        for r in common.read_csv(os.path.join(DERIVED, "table_midpoint_error_setA.csv")):
            self.assertGreaterEqual(float(r["mid_mae_pp"]) + 1e-9, float(r["band_mae_pp"]))
            self.assertGreater(int(r["n"]), 0)


class H5Split(unittest.TestCase):
    def test_split_fits(self):
        wbt = mt.wbayes_table(common.read_csv(os.path.join(DERIVED, "wbayes.csv")))
        rows = mt.h5_split_fits(SCORED, CASES, wbt, 100)  # asserts the pooled refit equals wbayes.csv
        self.assertEqual(len(rows), 20)
        self.assertTrue(all("a_pos" in r for r in rows))
        mac = mt.h5_macros(rows)
        self.assertEqual(mac["hFiveNCells"], 20)

    def test_written_counts(self):
        m = macros()
        self.assertEqual(m["hFiveNCells"], "20")
        for k in ("hFiveParANegLtPosN", "hFiveParBNegLtPosN", "hFiveParBDiffCiBelowZeroN"):
            self.assertTrue(0 <= int(m[k]) <= 20, k)
        rows = common.read_csv(os.path.join(DERIVED, "table_wbayes_by_result.csv"))
        self.assertEqual(len(rows), 20)
        self.assertEqual(int(m["hFiveParBNegLtPosN"]), sum(float(r["b_diff"]) < 0 for r in rows))
        for r in rows:
            self.assertLessEqual(float(r["b_diff_lo"]), float(r["b_diff_hi"]))


class NearOne(unittest.TestCase):
    def test_bound_str_rule(self):
        self.assertEqual(mt.bound_str(1.0013), "1.001")
        self.assertEqual(mt.bound_str(0.998), "0.998")
        self.assertEqual(mt.bound_str(1.2), "1.20")
        self.assertEqual(mt.bound_str(1.0001), "1.0001")  # never prints 1.00 or 1.000 for a bound that is not 1

    def test_counts_reproducible_from_printed_tables(self):
        import re
        m = macros()
        for name, f, key in (("Main", "table_wbayes.tex", "Main"), ("Own", "table_wbayes_own.tex", "Own")):
            text = open(os.path.join(common.ROOT, "paper", "generated", f)).read()
            above = 0
            for line in text.splitlines():
                cells = line.split("&")
                if len(cells) == 6:
                    mb = re.search(r"\[([\d.]+), ([\d.]+)\]", cells[3])
                    if mb and float(mb.group(1)) > 1:
                        above += 1
            self.assertEqual(above, int(m[f"sumBCiAboveOne{key}N"]), name)  # printed bounds reproduce the count
        self.assertEqual((m["nearOneBoundsMainN"], m["nearOneBoundsOwnN"], m["nearOneBoundsOwnAboveN"]), ("1", "4", "2"))


if __name__ == "__main__":
    unittest.main()
