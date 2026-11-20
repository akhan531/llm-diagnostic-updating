"""Tests for analysis/round2_analysis.py (committee round 2) and the compact table_ext (item 10).
Asserts the key numbers on the real raw data, so a silent change in the data or the rules fails loudly."""
import math
import os
import random
import statistics
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import committee_fixes as cf  # noqa: E402
import ext_analysis as ext  # noqa: E402
import round2_analysis as r2  # noqa: E402
from common import ROOT, load_cases  # noqa: E402

CASES = load_cases(os.path.join(ROOT, "data", "cases"))


class TestHelpers(unittest.TestCase):
    def test_holm(self):
        self.assertEqual(r2.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(r2.holm([0.5]), [0.5])
        self.assertEqual(r2.holm([0.4, 0.4]), [0.8, 0.8])
        self.assertTrue(all(a <= 1.0 for a in r2.holm([0.9, 0.95])))

    def test_tobit_no_censoring_is_mean(self):
        ps = [0.2, 0.4, 0.6]
        mu, nc = r2.tobit_mean(ps)
        self.assertEqual(nc, 0)
        self.assertAlmostEqual(mu, statistics.mean(math.log(p / (1 - p)) for p in ps))

    def test_tobit_all_censored_returns_bound(self):
        mu, nc = r2.tobit_mean([0.001, 0.002, 0.0])
        self.assertEqual(nc, 3)
        self.assertAlmostEqual(mu, math.log(0.01 / 0.99))
        mu, _ = r2.tobit_mean([0.995, 1.0])
        self.assertAlmostEqual(mu, math.log(0.99 / 0.01))

    def test_tobit_recovers_latent_mean(self):
        rng = random.Random(7)
        true_mu, sd = -4.0, 1.5
        ys = [rng.gauss(true_mu, sd) for _ in range(4000)]
        ps = [1 / (1 + math.exp(-y)) for y in ys]
        naive = statistics.mean(math.log(min(max(p, 0.01), 0.99) / (1 - min(max(p, 0.01), 0.99))) for p in ps)
        mu, nc = r2.tobit_mean(ps)
        self.assertGreater(nc, 500)
        self.assertLess(abs(mu - true_mu), 0.15)
        self.assertGreater(abs(naive - true_mu), 0.3)   # clipping alone is biased; Tobit is not


class TestRound2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.res, cls.mac = r2.main(["--out", cls.tmp, "--generated", cls.tmp, "--n-boot", "20"])
        with open(os.path.join(cls.tmp, "table_primary.tex")) as f:
            cls.table = f.read()

    def m(self, k):
        return self.mac[k]

    # item 3: counts
    def test_counts(self):
        m = self.m
        self.assertEqual((m("rtCntMainCollected"), m("rtCntMainCalibration"), m("rtCntMainAnalysable")), ("2142", "42", "2100"))
        self.assertEqual((m("rtCntMainParsed"), m("rtCntMainUnparsed")), ("2078", "22"))
        self.assertEqual((m("rtCntMainUnparsedAtLimit"), m("rtCntMainUnparsedOther")), ("21", "1"))
        self.assertEqual(m("rtCntMainUnparsedDeepseekDeepseekVFourPro"), "21")
        self.assertEqual(m("rtCntMainUnparsedOpenaiGptFourO"), "1")
        self.assertEqual((m("rtCntExtAnalysable"), m("rtCntExtParsed"), m("rtCntExtUnparsed")), ("1650", "1646", "4"))
        self.assertEqual((m("rtCntExtUnparsedAtLimit"), m("rtCntExtUnparsedOther")), ("2", "2"))
        self.assertEqual((m("rtCntAllAnalysable"), m("rtCntAllParsed"), m("rtCntAllUnparsed")), ("3750", "3724", "26"))
        self.assertEqual((m("rtCntRetryN"), m("rtCntRetryParsed")), ("21", "21"))
        for g in ("Main", "Ext", "All"):
            self.assertEqual(int(m(f"rtCnt{g}Analysable")) - int(m(f"rtCnt{g}Parsed")), int(m(f"rtCnt{g}Unparsed")))
            self.assertEqual(int(m(f"rtCnt{g}UnparsedAtLimit")) + int(m(f"rtCnt{g}UnparsedOther")), int(m(f"rtCnt{g}Unparsed")))

    # item 2: floor and ceiling
    def test_floor_shares(self):
        m = self.m
        self.assertEqual(m("rtFloorAllN"), "2078")
        self.assertEqual((m("rtFloorAllNZero"), m("rtFloorAllNHundred")), ("30", "12"))
        self.assertEqual(m("rtFloorAllPctAtOrBelowOne"), "22.4")   # the 22% quoted by the committee
        self.assertEqual(m("rtFloorAllPctNegAtOrBelowOne"), "44.8")
        self.assertEqual(m("rtFloorAllPctAtOrAboveNinetyNine"), "4.2")
        for r in self.res["floor"]:
            self.assertLessEqual(r["n_zero"], r["n_le_half"])
            self.assertLessEqual(r["n_le_half"], r["n_le_one"])
            self.assertLessEqual(r["n_hundred"], r["n_ge_99"])
        self.assertEqual(sum(r["n_le_one"] for r in self.res["floor"] if r["model"] != "ALL"),
                         next(r["n_le_one"] for r in self.res["floor"] if r["model"] == "ALL"))

    def test_primary_ratio_matches_f1(self):
        rob = [r for r in self.res["rob"] if r["variant"] == "primary" and r["model"] != "ALL"]
        self.assertEqual(len(rob), 5)
        self.assertEqual(sorted(round(r["median"], 2) for r in rob), [1.34, 1.44, 1.46, 1.52, 1.83])
        self.assertEqual(self.m("rtRobustNFloorScenarios"), "4")
        self.assertEqual(self.m("rtRobustNKeptScenarios"), "6")

    def test_floor_robustness_headline(self):
        # r > 1 for every model median under every floor handling; scenario-level count falls from 8 to 6 to 8 of 10
        for v in r2.VWORD.values():
            self.assertEqual(self.m(f"rtRobust{v}NModelsMedGtOne"), "5", v)
        self.assertEqual(self.m("rtRobustClipOneNGtOneMin"), "6")
        self.assertEqual(self.m("rtRobustClipOneNGtOneMax"), "8")
        self.assertEqual(self.m("rtRobustTobitNGtOneMin"), "6")
        # excluding the four scenarios whose negative target is below 1%: every model r > 1 in all 6 remaining
        self.assertEqual((self.m("rtRobustExclSubOneNGtOneMin"), self.m("rtRobustExclSubOneNGtOneMax")), ("6", "6"))
        for r in self.res["rob"]:
            if r["variant"] == "exclsubone" and r["model"] != "ALL":
                self.assertGreater(r["med_lo"], 1.0)
        # but with only 10 clusters the interval includes 1 for all five models under 1% clipping
        for r in self.res["rob"]:
            if r["variant"] == "clipone" and r["model"] != "ALL":
                self.assertLess(r["med_lo"], 1.0)

    # item 5: scenario level and Holm
    def test_scenario_level(self):
        m = self.m
        self.assertEqual((m("rtScenAllNGtOne"), m("rtScenAllNScen"), m("rtScenAllNLeOne")), ("8", "10", "2"))
        self.assertEqual(m("rtScenAllLeOneList"), "A2\\_breast, B4\\_aas")
        self.assertEqual((m("rtScenPairsNGtOne"), m("rtScenPairsN")), ("41", "50"))
        self.assertEqual(m("rtScenOpenaiGptFourONGtOne"), "9")

    def test_holm_family(self):
        m = self.m
        self.assertEqual((m("rtHolmNTests"), m("rtHolmNRaw")), ("40", "15"))
        self.assertEqual(m("rtHolmNSurviveFamily"), "0")
        self.assertEqual(m("rtHolmRatioNSurviveWithin"), "0")
        self.assertEqual(m("rtHolmRatioNRaw"), "1")
        self.assertEqual(m("rtHolmHindNSurviveWithin"), "5")
        self.assertEqual(m("rtHolmBlindPosNSurviveWithin"), "5")
        self.assertEqual(m("rtHolmBlindNegNSurviveWithin"), "0")
        self.assertEqual((m("rtHolmPooledNTests"), m("rtHolmPooledNSurvive")), ("7", "2"))
        for f in self.res["holm"]:
            self.assertGreaterEqual(f["p_holm_family"], f["p"] - 1e-12)
            self.assertGreaterEqual(f["p_holm_within"], f["p"] - 1e-12)
            self.assertLessEqual(f["p_holm_family"], 1.0)
        # Holm adjusted values are monotone in the raw p within a set
        pm = sorted((f for f in self.res["holm"] if f["family"] != "Pooled"), key=lambda f: f["p"])
        adj = [f["p_holm_family"] for f in pm]
        self.assertEqual(adj, sorted(adj))

    # item 6: update-only
    def test_update_only(self):
        m = self.m
        self.assertEqual(m("rtUpdNModelsMedGtOne"), "5")
        self.assertEqual((m("rtUpdMedMin"), m("rtUpdMedMax")), ("1.30", "1.65"))
        self.assertEqual(m("rtUpdNModelsLoAboveOne"), "1")
        self.assertEqual(m("rtBlindHave"), "1")
        self.assertEqual(m("rtBlindNCells"), "50")
        self.assertEqual((m("rtBlindMedMin"), m("rtBlindMedMax")), ("1.34", "1.83"))
        # a shared blind pretest cancels in the positive minus negative update: r is unchanged up to answer counts
        self.assertLess(float(m("rtBlindMaxAbsDiffFromPlainR")), 0.05)
        # the own-pretest update lowers r because the stated pretest is higher after a positive result
        own = {r["model"]: r["median"] for r in self.res["upd"] if r["variant"] == "own"}
        plain = {r["model"]: r["median"] for r in self.res["rob"] if r["variant"] == "primary"}
        for k in own:
            self.assertLessEqual(own[k], plain[k] + 1e-9)

    # item 8: primary metric
    def test_primary_metric(self):
        m = self.m
        self.assertEqual((m("rtPrimAAllVN"), m("rtPrimAAllNN")), ("166", "167"))
        self.assertEqual((m("rtPrimBAllVN"), m("rtPrimBAllNN")), ("247", "249"))
        self.assertEqual((m("rtPrimAAllVMed"), m("rtPrimAAllNMed")), ("0.9", "0.9"))
        self.assertEqual((m("rtPrimBAllVMed"), m("rtPrimBAllNMed")), ("1.23", "1.39"))
        self.assertEqual(len(self.res["prim"]), 6 * 2 * 5)
        self.assertEqual(self.table.count("\\\\"), 2 + 5 + 1)
        self.assertIn("All models", self.table)
        self.assertNotIn("—", self.table)

    # item 3: threshold share
    def test_threshold_share(self):
        m = self.m
        self.assertEqual((m("rtThrBTwoNegBelowThresholdPct"), m("rtThrBTwoNegBelowThresholdN"), m("rtThrBTwoNegN")),
                         ("35", "29", "82"))
        self.assertEqual(m("rtThrBTwoNegPrimaryTargetPct"), "3.35")
        self.assertEqual(m("rtThrBTwoNegThresholdPct"), "1.8")
        for k in ("Primary", "PiopedLrPoint", "PiopedLrBandLo", "PiopedObservedNpv"):
            self.assertEqual(m(f"rtThrBTwoNeg{k}AboveThreshold"), "yes")
        self.assertEqual(m("rtThrBTwoNegPiopedObservedNpvTargetPct"), "11.03")

    def test_macro_hygiene(self):
        for k, v in self.mac.items():
            self.assertTrue(k.isalpha(), k)
            self.assertTrue(k.startswith("rt"), k)
            self.assertNotIn("—", v)
            self.assertNotEqual(v, "")

    def test_raw_untouched(self):
        raw = os.path.join(ROOT, "results", "raw")
        self.assertTrue(all(not f.startswith("r2") for f in os.listdir(raw)))


class TestTableExt(unittest.TestCase):
    """Item 10: compact Table 4."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        ext.main(["--out", cls.tmp, "--generated", cls.tmp, "--n-boot", "20"])
        with open(os.path.join(cls.tmp, "table_ext.tex")) as f:
            cls.t = f.read()

    def test_compact(self):
        self.assertNotIn("E5", self.t)                       # redundant row dropped (filled == prespecified)
        self.assertNotIn("—", self.t)
        rows = [ln for ln in self.t.splitlines() if ln.endswith("\\\\") and "&" in ln]
        self.assertEqual(len(rows), 1 + 8)                   # header + 8 quantities
        self.assertIn("E2 DOR ratio", self.t)   # a ratio (believed / published log DOR, defined in the caption)
        self.assertNotIn("Believed minus published", self.t)
        for ln in rows[1:]:
            cells = [c.strip().rstrip("\\").strip() for c in ln.split("&")[1:]]
            for c in cells:
                num = c.replace("\\ensuremath{-}", "")
                self.assertEqual(len(num.split(".")[1]), 2, ln)   # consistent decimals

    def test_e5_identical(self):
        import csv
        # the dropped row is justified: prespecified and filled b agree to two decimals for every model
        with open(os.path.join(self.tmp, "ext_e5_headline.csv")) as fh:
            rows = list(csv.DictReader(fh))
        by = {}
        for r in rows:
            by.setdefault(r["model"], {})[r["sample"]] = float(r["b"])
        for m, d in by.items():
            self.assertEqual(f"{d['prespecified']:.2f}", f"{d['filled']:.2f}", m)


if __name__ == "__main__":
    unittest.main()
