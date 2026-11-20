"""C4 re-verification additions (N1, N2, N4, N5, N10, M5, M6). Read-only against results/derived and data/cases;
recomputes the key numbers independently of make_tables.py and checks the committed macros."""
import math
import os
import re
import statistics
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import make_tables as mt  # noqa: E402

DERIVED = os.path.join(common.ROOT, "results", "derived")
CASES = common.load_cases(os.path.join(common.ROOT, "data", "cases"))
SCORED = common.read_csv(os.path.join(DERIVED, "scored.csv"))
CLIN = common.read_csv(os.path.join(DERIVED, "clinician_compare.csv"))
MACROS = os.path.join(common.ROOT, "paper", "generated", "results_macros.tex")
GPT4O, GEMINI, SOL, CLAUDE, DEEPSEEK = ("openai/gpt-4o", "google/gemini-3.8-flash", "openai/gpt-6.1-sol",
                                        "anthropic/claude-sonnet-5.5", "deepseek/deepseek-v4-pro")


def macros():
    out = {}
    for line in open(MACROS):
        m = re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{(.+)\}\n?", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


class ImpliedLr(unittest.TestCase):
    def test_independent_recompute(self):
        rows = {(r["model"], r["result"]): r for r in mt.implied_lr_table(SCORED, CASES)}
        for res in ("pos", "neg"):
            for m in (GPT4O, CLAUDE, GEMINI, SOL, DEEPSEEK):
                v = []
                for r in SCORED:
                    if (r["model"], r["result"]) != (m, res) or r["calibration"] == "1" or r["condition"] == "E" \
                            or r["parse_ok"] != "1" or r["boundary_answer"] == "1":
                        continue
                    pre, post = float(r["pretest_stated"]) / 100, float(r["posttest_stated"]) / 100
                    imp = (post / (1 - post)) / (pre / (1 - pre))
                    lr = CASES[r["case_id"]]["target"]["lr_point"]
                    v.append(imp / lr)
                self.assertAlmostEqual(rows[(m, res)]["median_ratio"], statistics.median(v), places=9)
                self.assertEqual(rows[(m, res)]["n_used"], len(v))

    def test_published_lr_rule(self):
        # range-only LRs use the geometric mean; printed points are used as printed
        self.assertAlmostEqual(CASES["A2_breast_pos"]["target"]["lr_point"], math.sqrt(13.0 * 33.0), places=9)
        self.assertEqual(CASES["A4_uti_pos"]["target"]["lr_point"], 9.0)
        self.assertEqual(CASES["B1_pe_low_pos"]["target"]["lr_point"], 1.68)

    def test_macros(self):
        m = macros()
        self.assertEqual((m["impLrPosMedMin"], m["impLrPosMedMax"]), ("1.33", "2.02"))
        self.assertEqual((m["impLrNegMedMin"], m["impLrNegMedMax"]), ("0.53", "0.73"))
        self.assertEqual((m["impLrPosNBoundaryExcl"], m["impLrNegNBoundaryExcl"]), ("9", "25"))
        self.assertEqual(int(m["impLrPosNParsed"]), int(m["impLrPosNBoundaryExcl"]) + int(m["impLrPosNUsed"]))
        self.assertEqual(int(m["impLrNegNParsed"]), int(m["impLrNegNBoundaryExcl"]) + int(m["impLrNegNUsed"]))
        self.assertEqual((m["impLrPosPctAboveMin"], m["impLrPosPctAboveMax"]), ("68", "75"))
        self.assertEqual(m["impLrPosOpenaiGptFourOMed"], "2.02")

    def test_split_counts(self):
        m = macros()
        self.assertEqual((m["splitBPosCiAboveOneN"], m["splitBNegCiAboveOneN"], m["splitNCells"]), ("0", "0", "20"))
        self.assertEqual(m["splitBPosCiAboveOneN"], m["hFiveParBPosCiAboveOneN"])
        self.assertEqual(m["splitBNegCiAboveOneN"], m["hFiveParBNegCiAboveOneN"])
        self.assertEqual((m["splitBPosGptFourOPtMin"], m["splitBPosGptFourOPtMax"]), ("0.43", "0.83"))
        self.assertEqual(m["splitBPosGptFourOCiBelowOneN"], "0")  # GPT-4o b after positive results: no cell wholly below 1


class NegativeScenarios(unittest.TestCase):
    def test_counts_and_example(self):
        rows = mt.neg_scenario_table(SCORED, CASES)
        self.assertEqual(len(rows), 10)  # A5 excluded
        hp = [r["scenario"] for r in rows if r["high_prior"]]
        self.assertEqual(sorted(hp), ["A1_pneumonia", "B2_pe_mod", "B3_pe_high", "B5_appy_ct", "B6_appy_us"])
        for r in rows:  # high-prior rule: published prior >= 15%
            self.assertEqual(r["high_prior"], int(r["published_prior_pct"] >= 15.0))
        below = sorted(r["scenario"] for r in rows if r["high_prior"] and r["median_below_target"])
        self.assertEqual(below, ["A1_pneumonia", "B2_pe_mod", "B3_pe_high", "B5_appy_ct"])  # B6 above the target
        m = macros()
        self.assertEqual((m["negNHighPrior"], m["negNHighPriorBelowTarget"], m["negNHighPriorBelowBandLo"]), ("5", "4", "3"))
        self.assertEqual((m["negPeHighTargetPct"], m["negPeHighMedianPct"]), ("9.7", "5.0"))
        v = [float(r["posttest_stated"]) for r in SCORED if r["case_id"] == "B3_pe_high_neg"
             and r["condition"] != "E" and r["parse_ok"] == "1"]
        self.assertEqual(statistics.median(v), 5.0)


class UtiPositive(unittest.TestCase):
    def test_macros(self):
        m = macros()
        self.assertEqual((m["utiPosTargetPct"], m["utiPosBayesBandLoPct"], m["utiPosBayesBandHiPct"]),
                         ("2.78", "0.89", "8.33"))
        self.assertEqual((m["utiPosPriorPct"], m["utiPosPriorPrintedLoPct"], m["utiPosPriorPrintedHiPct"],
                          m["utiPosPriorFloorPct"]), ("0.32", "0", "1", "0.1"))
        self.assertEqual(m["utiPosMedianPostPct"], "80.0")
        v = [float(r["posttest_stated"]) for r in SCORED if r["case_id"] == "A4_uti_pos"
             and r["condition"] != "E" and r["parse_ok"] == "1"]
        self.assertEqual(len(v), int(m["utiPosNAnswers"]))
        self.assertEqual(statistics.median(v), 80.0)


class ContrastAndClinician(unittest.TestCase):
    def test_e_minus_n(self):
        m = macros()
        self.assertEqual((m["hFourNCellsCiExclZero"], m["hFourNCellsTotal"]), ("4", "10"))
        self.assertEqual(m["hFourNCellsCiExclZero"], m["contrastNExclZeroHFour"])

    def test_clinician_items(self):
        m = macros()
        rows = mt.clinician_item_rows(CLIN, [GPT4O, GEMINI, SOL, CLAUDE, DEEPSEEK])
        self.assertEqual(len(rows), 40)  # 5 models x 8 items, A5 excluded
        by = {(r["model"], r["case_id"]): r for r in rows}
        for mod in (GEMINI, SOL):  # ties with the clinician median on the pneumonia positive item
            self.assertEqual((by[(mod, "A1_pneumonia_pos")]["tie"], by[(mod, "A1_pneumonia_pos")]["model_closer"]), (1, 0))
        self.assertEqual(by[(DEEPSEEK, "A1_pneumonia_pos")]["model_closer"], 0)
        self.assertEqual(by[(DEEPSEEK, "A1_pneumonia_pos")]["tie"], 0)
        self.assertEqual(by[(CLAUDE, "A4_uti_pos")]["tie"], 1)
        self.assertEqual((m["clinItemVPneumoniaPosNModelsNotCloser"], m["clinItemVUtiPosNModelsNotCloser"]), ("3", "3"))
        self.assertEqual(m["clinItemVDeepseekDeepseekVFourProNNotCloser"], "2")
        for tag in ("AnthropicClaudeSonnetFiveFive", "GoogleGeminiThreeEightFlash", "OpenaiGptFourO",
                    "OpenaiGptSixOneSol"):
            self.assertEqual(m[f"clinItemV{tag}NNotCloser"], "1")
        self.assertEqual(m["clinItemVPneumoniaPosClinMed"], "95.0")
        # consistency with the existing summary: closer + not closer = items
        for mod in (GPT4O, GEMINI, SOL, CLAUDE, DEEPSEEK):
            mine = [r for r in rows if r["model"] == mod]
            self.assertEqual(sum(r["model_closer"] for r in mine) + sum(r["model_not_closer"] for r in mine), 8)


if __name__ == "__main__":
    unittest.main()
