"""Tests for analysis/ext_analysis.py (R1 extension analyses E1 to E5) on SYNTHETIC fixtures only.

Fixtures: fixtures/synthetic.jsonl (main run, run id SYNTHETIC) and fixtures/ext_synthetic/SYNTH_EXT{1..5}.jsonl
(made by fixtures/make_ext_fixture.py; designed truths are listed there). Nothing is written to results/derived or
paper/generated (synthetic firewall); one test asserts the refusal.
"""
import contextlib
import filecmp
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import committee_fixes as cf  # noqa: E402
import committee_fixes2 as cf2  # noqa: E402
import ext_analysis as ea  # noqa: E402
import fit_wbayes  # noqa: E402
import make_tables  # noqa: E402
import score  # noqa: E402
from common import ROOT, load_cases, load_config, select_records  # noqa: E402

FIX = os.path.join(HERE, "fixtures")
MAIN = os.path.join(FIX, "synthetic.jsonl")
EXT = {b: os.path.join(FIX, "ext_synthetic", f"SYNTH_EXT{b[1]}.jsonl") for b in ea.BATCHES}
CASES = os.path.join(ROOT, "data", "cases")
IDS = ["--main-run-id", "SYNTHETIC"] + [x for b in ea.BATCHES for x in ("--" + b.lower().replace("e", "ext"),
                                                                        f"SYNTH_EXT{b[1]}")] + ["--ext4-min-keys", "0"]


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def run_ext(tmp, batches, n_boot=30):
    raw = os.path.join(tmp, "raw")
    os.makedirs(raw, exist_ok=True)
    shutil.copy(MAIN, raw)
    for b in batches:
        shutil.copy(EXT[b], raw)
    out, gen = os.path.join(tmp, "out"), os.path.join(tmp, "gen")
    res, mac, have = quiet(ea.main, ["--raw", os.path.join(raw, "*.jsonl"), "--cases", CASES, "--out", out,
                                     "--generated", gen, "--n-boot", str(n_boot)] + IDS)
    return res, mac, have, out, gen


class TestAllBatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.res, cls.mac, cls.have, cls.out, cls.gen = run_ext(cls.tmp, ea.BATCHES)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_all_present_and_files(self):
        self.assertTrue(all(self.have.values()))
        for b in ea.BATCHES:
            for n in ea.CSV_FILES[b]:
                self.assertTrue(os.path.exists(os.path.join(self.out, n)), n)
        self.assertTrue(os.path.exists(os.path.join(self.gen, "ext_macros.tex")))
        self.assertTrue(os.path.exists(os.path.join(self.gen, "table_ext.tex")))
        txt = open(os.path.join(self.gen, "ext_macros.tex")).read()
        self.assertIn("\\newcommand{\\extHaveEFive}{1}", txt)
        self.assertNotIn("\u2014", txt + open(os.path.join(self.gen, "table_ext.tex")).read())

    def test_main_tex_slot_files(self):
        for n, names in (("ext_pretest_macros.tex", ["cfPreOnlyMedAll", "cfPreOnlyParAMin", "cfPreOnlyParBMax"]),
                         ("ext_believedlr_macros.tex", ["cfBelLrDorRatioMin", "cfBelLrDorRatioMax", "cfBelLrRatioMedMin",
                                                        "cfBelLrRatioMedMax"]),
                         ("ext_longchart_macros.tex", ["cfLongTokensMin", "cfLongTokensMax", "cfLongVsNAllMean",
                                                       "cfLongVsNAllLo", "cfLongVsNAllHi", "cfLongUnnamedVsNAllMean"]),
                         ("ext_dsretry_macros.tex", ["cfDsRetryNParsed"])):  # ParB* need a DeepSeek model (not in fixtures)
            with open(os.path.join(self.gen, n)) as f:
                txt = f.read()
            for k in names:
                self.assertIn("\\newcommand{\\%s}" % k, txt, (n, k))
        # SYNTH-conservative believes LR+ x 2 and LR- / 2: its log DOR ratio is above 1
        s = {x["model"]: x for x in self.res["E2"][1]}
        self.assertGreater(s["SYNTH-conservative"]["log_dor_ratio_med"], 1.0)
        self.assertAlmostEqual(s["SYNTH-calibrated"]["log_dor_ratio_med"], 1.0, delta=1e-4)

    def test_e1_blind_prior(self):
        cells, summ, fits = self.res["E1"]
        s = {x["model"]: x for x in summ}
        self.assertAlmostEqual(s["SYNTH-calibrated"]["blind_err_lod_mean"], 0.0, delta=1e-3)
        self.assertEqual(self.mac["extEOneNTotal"], "270")
        self.assertEqual(self.mac["extEOneNParsed"], "269")
        c = [x for x in cells if x["condition"] == "VNS"][0]
        self.assertAlmostEqual(c["gap_pos_lod"] - c["gap_neg_lod"], c["gap_posneg_lod"], places=12)
        self.assertEqual({f["variant"] for f in fits}, {"blind_prior", "own_stated_prior", "published_prior"})

    def test_extension_totals_are_sums_of_per_run_macros(self):
        m = self.mac
        runs = ["extEOne", "extETwo", "extEThreeL", "extEFourNUnnamed", "extEFourLUnnamed"]
        self.assertEqual(int(m["extRunsNTotal"]), sum(int(m[r + "NTotal"]) for r in runs))
        self.assertEqual(int(m["extRunsNParsed"]), sum(int(m[r + "NParsed"]) for r in runs))
        self.assertLessEqual(int(m["extRunsNParsed"]), int(m["extRunsNTotal"]))
        self.assertGreaterEqual(int(m["extRunsNTruncTotal"]), 0)

    def test_e1_neg_gap_count_is_a_subset_of_models(self):
        self.assertLessEqual(int(self.mac["extEOneNModelsNegGapExpected"]), int(self.mac["extEOneNModelsNegGap"]))

    def test_e1_ratio_does_not_use_a_prior(self):
        # the within-scenario ratio is unchanged when every prior (stated or published) is replaced
        cases = load_cases(CASES)
        rows = ea.v1_rows(json_records(MAIN), cases)
        r1 = cf.ratio_cells(rows, cases, cf.FIT_CONDS)
        rows2 = [dict(r, pre=0.5, prior=0.5) for r in rows]
        r2 = cf.ratio_cells(rows2, cases, cf.FIT_CONDS)
        self.assertEqual([x["ratio"] for x in r1], [x["ratio"] for x in r2])

    def test_e2_knowledge_vs_updating(self):
        _sc, summ, _f = self.res["E2"]
        s = {x["model"]: x for x in summ}
        self.assertAlmostEqual(s["SYNTH-calibrated"]["log_dor_excess_mean"], 0.0, delta=1e-4)
        self.assertAlmostEqual(s["SYNTH-calibrated"]["b_own"], 1.0, delta=1e-3)
        self.assertAlmostEqual(s["SYNTH-conservative"]["log_dor_excess_mean"], 2 * math.log(2), delta=1e-4)
        self.assertAlmostEqual(s["SYNTH-conservative"]["b_own"], 1.0, delta=1e-3)
        self.assertAlmostEqual(s["SYNTH-noisy"]["b_own"], 1.6, delta=1e-3)
        self.assertEqual(s["SYNTH-calibrated"]["verdict"], "neither")
        self.assertEqual(s["SYNTH-conservative"]["verdict"], "knowledge")
        self.assertEqual(s["SYNTH-noisy"]["verdict"], "updating")
        self.assertAlmostEqual(s["SYNTH-conservative"]["abs_coh_err_med"], 0.0, delta=1e-3)

    def test_believed_lr_clip(self):
        lp, ln = ea.believed_lrs(90, 100)
        self.assertTrue(math.isfinite(lp) and lp > 1e4)
        self.assertAlmostEqual(ln, 0.1 / (1 - 1e-5))

    def test_e3_tracking_and_contrasts(self):
        ctr, pr, tr = self.res["E3"]
        t = {(x["arm"], x["model"], x["result"]): x for x in tr}
        self.assertEqual(t[("L", "SYNTH-noisy", "pos")]["pct_direction_current"], 0.0)  # follows the older result
        self.assertEqual(t[("L", "SYNTH-calibrated", "pos")]["pct_direction_current"], 100.0)
        self.assertEqual(t[("N", "SYNTH-calibrated", "pos")]["ratio_med"], t[("N", "SYNTH-calibrated", "pos")]["ratio_med_ref"])
        noisy = [x for x in ctr if x["model"] == "SYNTH-noisy" and x["metric"] == "abs"]
        d = {x["result"]: x for x in noisy}
        self.assertGreater(d["pos"]["mean"], 1.0)  # the trap raises error only on positive cases
        self.assertEqual(d["pos"]["n_scen"], 10)
        p = {(x["arm"], x["model"]): x for x in pr}
        self.assertEqual(p[("L", "SYNTH-noisy")]["n_total"], 60)

    def test_contrast_matches_f6_statistic(self):
        # paired_scen with result=None reproduces committee_fixes.scenario_contrasts for a main1 contrast
        cases = load_cases(CASES)
        rows = ea.v1_rows(json_records(MAIN), cases)
        ref = cf.scenario_contrasts(rows, "N", "V", "abs")
        got = ea.paired_scen([r for r in rows if r["condition"] == "N"], [r for r in rows if r["condition"] == "V"], "abs")
        for m in ref:
            for s in ref[m]:
                self.assertAlmostEqual(ref[m][s], got[m][s], places=12)

    def test_e4_arms(self):
        ctr, pr, tr = self.res["E4"]
        self.assertEqual({x["contrast"] for x in ctr}, {"NunVsN", "LunVsL", "LunVsN"})
        self.assertEqual({x["arm"] for x in pr}, {"NUnnamed", "LUnnamed"})

    def test_e5_retry_sensitivity(self):
        retry, hl = self.res["E5"]
        self.assertEqual(len(retry), 29)
        self.assertEqual(self.mac["extEFiveNParsed"], "28")
        self.assertEqual(self.mac["extEFiveNStillTrunc"], "1")
        h = {(x["model"], x["sample"]): x for x in hl}
        self.assertEqual(h[("SYNTH-calibrated", "prespecified")]["b"], h[("SYNTH-calibrated", "filled")]["b"])
        self.assertEqual(h[("SYNTH-noisy", "filled")]["n_parsed"] - h[("SYNTH-noisy", "prespecified")]["n_parsed"],
                         sum(1 for x in retry if x["model"] == "SYNTH-noisy" and x["retry_status"] == "ok"))

    def test_e5_prespecified_reproduces_committee_fixes(self):
        res, _mac = quiet(cf.main, ["--raw", MAIN, "--run-ids", "SYNTHETIC", "--n-boot", "30", "--no-figures",
                                    "--figures", os.path.join(self.tmp, "fig"), "--out", os.path.join(self.tmp, "cf"), "--generated", os.path.join(self.tmp, "cfg")])
        retry, hl = self.res["E5"]
        pre = {x["model"]: x for x in hl if x["sample"] == "prespecified"}
        for f in res["extra_fits"]:
            if f["analysis"] == "wbayes_published_pooled_VNBS":
                self.assertAlmostEqual(pre[f["model"]]["b"], f["b"], places=12)
                self.assertAlmostEqual(pre[f["model"]]["b_lo"], f["b_lo"], places=12)
        for r in res["ratio"]:
            if r["model"] == "ALL" or r.get("condition") not in (None, "", "VNBS"):
                continue
            self.assertAlmostEqual(pre[r["model"]]["ratio_med"], r["median"], places=12)


def json_records(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


class TestMissingBatches(unittest.TestCase):
    def test_partial_and_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            res, mac, have, out, gen = run_ext(tmp, ["E1", "E3"])
            self.assertEqual([b for b in ea.BATCHES if have[b]], ["E1", "E3"])
            self.assertFalse(os.path.exists(os.path.join(out, "ext_e2_summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(gen, "ext_pretest_macros.tex")))
            self.assertFalse(os.path.exists(os.path.join(gen, "ext_believedlr_macros.tex")))
            self.assertFalse(os.path.exists(os.path.join(gen, "ext_dsretry_macros.tex")))
            txt = open(os.path.join(gen, "ext_macros.tex")).read()
            self.assertIn("\\newcommand{\\extHaveETwo}{0}", txt)
            self.assertFalse(any(k.startswith("extETwo") or k.startswith("extEFive") for k in mac))
            # E4 without E3: the L-unnamed vs L-named contrast is skipped, N-unnamed still runs
        with tempfile.TemporaryDirectory() as tmp:
            res, mac, have, out, gen = run_ext(tmp, ["E4"])
            self.assertEqual(mac["extEFourHaveLContrast"], "0")
            self.assertEqual({x["contrast"] for x in res["E4"][0]}, {"NunVsN", "LunVsN"})
        with tempfile.TemporaryDirectory() as tmp:
            run_ext(tmp, ["E1"])
            os.remove(os.path.join(tmp, "raw", os.path.basename(EXT["E1"])))
            quiet(ea.main, ["--raw", os.path.join(tmp, "raw", "*.jsonl"), "--cases", CASES, "--out",
                            os.path.join(tmp, "out"), "--generated", os.path.join(tmp, "gen"), "--n-boot", "5"] + IDS)
            self.assertFalse(os.path.exists(os.path.join(tmp, "gen", "ext_macros.tex")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "gen", "table_ext.tex")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "gen", "ext_pretest_macros.tex")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "out", "ext_e1_cells.csv")))


class TestE4CompletenessGate(unittest.TestCase):
    """A partial ext4 run (real default 600 keys) must be treated as absent: no E4 outputs."""
    def test_partial_e4_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "raw")
            os.makedirs(raw)
            for f in (MAIN, EXT["E3"], EXT["E4"]):
                shutil.copy(f, raw)
            ids = IDS[:-2]      # drop the --ext4-min-keys 0 override: default 600 applies
            res, mac, have = quiet(ea.main, ["--raw", os.path.join(raw, "*.jsonl"), "--cases", CASES, "--out",
                                             os.path.join(tmp, "out"), "--generated", os.path.join(tmp, "gen"),
                                             "--n-boot", "5"] + ids)
            self.assertFalse(have["E4"])
            self.assertTrue(have["E3"])
            self.assertFalse(any(k.startswith("extEFour") for k in mac))
            self.assertFalse(os.path.exists(os.path.join(tmp, "out", "ext_e4_contrasts.csv")))
            self.assertNotIn("cfLongUnnamedVsNAllMean",
                             open(os.path.join(tmp, "gen", "ext_longchart_macros.tex")).read())


class TestFirewall(unittest.TestCase):
    def test_refuses_real_dirs(self):
        real = os.path.join(ROOT, "results", "derived")
        before = sorted(os.listdir(real))
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "raw")
            os.makedirs(raw)
            shutil.copy(MAIN, raw)
            shutil.copy(EXT["E1"], raw)
            with self.assertRaises(AssertionError):
                quiet(ea.main, ["--raw", os.path.join(raw, "*.jsonl"), "--cases", CASES, "--n-boot", "5"] + IDS)
        self.assertEqual(before, sorted(os.listdir(real)))


class TestMain1Unchanged(unittest.TestCase):
    """The pre-specified chain and the committee scripts give byte-identical outputs with and without ext records
    in the raw directory (ext records are other run ids; E5 is a sensitivity, never merged into main1)."""

    def chain(self, raw_glob, out):
        d, g = os.path.join(out, "d"), os.path.join(out, "g")
        run = ["--run-ids", "SYNTHETIC", "--raw", raw_glob, "--cases", CASES]
        quiet(score.main, run + ["--out", d])
        quiet(fit_wbayes.main, run + ["--out", d, "--n-boot", "20"])
        quiet(make_tables.main, ["--derived", d, "--generated", g, "--n-boot", "20"])
        quiet(cf.main, run + ["--out", d, "--generated", g, "--figures", os.path.join(out, "f"), "--n-boot", "25",
                             "--no-figures"])
        quiet(cf2.main, run + ["--out", d, "--generated", g, "--n-boot", "25"])
        return d, g

    def test_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "raw")
            os.makedirs(raw)
            shutil.copy(MAIN, raw)
            glob_ = os.path.join(raw, "*.jsonl")
            a = self.chain(glob_, os.path.join(tmp, "without"))
            for b in ea.BATCHES:
                shutil.copy(EXT[b], raw)
            b_ = self.chain(glob_, os.path.join(tmp, "with"))
            for x, y in zip(a, b_):
                names = sorted(os.listdir(x))
                self.assertEqual(names, sorted(os.listdir(y)))
                self.assertGreater(len(names), 5)
                _m, mism, errs = filecmp.cmpfiles(x, y, names, shallow=False)
                self.assertEqual((mism, errs), ([], []))

    def test_real_config_knows_ext_ids(self):
        cfg = load_config()
        self.assertEqual(cfg["paper_run_ids"], ["main1"])
        for rid in ea.DEFAULT_IDS.values():
            self.assertIn(rid, cfg["ext_run_ids"])
        main_recs = [dict(r, run_id="main1") for r in json_records(MAIN)]
        ext_recs = [dict(r, run_id=ea.DEFAULT_IDS[b]) for b in ea.BATCHES for r in json_records(EXT[b])]
        f1, rep1 = select_records(main_recs, ["main1"], cfg, quiet=True)
        f2, rep2 = select_records(main_recs + ext_recs, ["main1"], cfg, quiet=True)
        self.assertEqual(f1, f2)
        self.assertEqual(rep1["attempts"], rep2["attempts"])
        self.assertEqual(rep2["unknown"], {})
        self.assertEqual(sum(rep2["not_selected"].values()), len(ext_recs))


if __name__ == "__main__":
    unittest.main()
