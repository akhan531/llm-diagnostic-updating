"""End-to-end: synthetic fixture -> score -> fit -> tables -> figures, in a temp dir only."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import fit_wbayes  # noqa: E402
try:
    import make_figures  # noqa: E402
except SystemExit:  # make_figures exits when matplotlib is missing (L2)
    make_figures = None
import make_tables  # noqa: E402
import score  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "synthetic.jsonl")
CASES = os.path.join(common.ROOT, "data", "cases")


def listing(d):
    return sorted(os.listdir(d)) if os.path.isdir(d) else None


class MakeOutputsTest(unittest.TestCase):
    def test_fixture_is_marked_synthetic(self):
        recs = [json.loads(line) for line in open(FIXTURE) if line.strip()]
        self.assertGreater(len(recs), 100)
        self.assertTrue(all(r["synthetic"] is True and r["model"].startswith("SYNTH-") for r in recs))

    def test_end_to_end_in_temp_dir(self):
        self._end_to_end(make_figures is not None)  # tables always; figures only when matplotlib exists

    @unittest.skipIf(make_figures is None, "matplotlib not installed")
    def test_figures_written(self):
        self._end_to_end(True)

    def _end_to_end(self, figures):
        with tempfile.TemporaryDirectory() as tmp:
            d, gen, fig = (os.path.join(tmp, x) for x in ("derived", "generated", "figures"))
            score.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES, "--out", d])
            fit_wbayes.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES, "--out", d, "--n-boot", "100"])
            mac = make_tables.main(["--derived", d, "--generated", gen, "--n-boot", "100"])
            if figures:
                make_figures.main(["--derived", d, "--out", fig])
            for f in ("table_error_by_condition.csv", "table_contrasts.csv", "table_wbayes.csv"):
                self.assertTrue(os.path.getsize(os.path.join(d, f)) > 0, f)
            for f in ("results_macros.tex", "table_error.tex", "table_wbayes.tex"):
                self.assertTrue(os.path.getsize(os.path.join(gen, f)) > 0, f)
            for f in (("fig_stated_vs_target.pdf", "fig_stated_vs_target.png", "fig_wbayes.pdf",
                       "fig_wbayes.png") if figures else ()):
                self.assertTrue(os.path.getsize(os.path.join(fig, f)) > 0, f)
            n = make_tables.check_macros_file(os.path.join(gen, "results_macros.tex"))
            self.assertEqual(n, len(mac))
            self.assertTrue(all(k.isalpha() for k in mac))
            self.assertIn("wbAMainSYNTHCalibratedVPt", mac)
            # the calibrated synthetic model is near a=b=1 in V; the conservative one is not
            self.assertGreater(float(mac["wbBMainSYNTHCalibratedVPt"]), 0.85)
            rows = common.read_csv(os.path.join(d, "table_wbayes.csv"))
            self.assertTrue(all(r["condition"] != "E" for r in rows))
            self.assertIn("boundary_flag", rows[0])
            # H1: A5 is in no error cell or contrast, and Set A n_total is equal across V, N, B, S, E
            err = common.read_csv(os.path.join(d, "table_error_by_condition.csv"))
            for m in {r["model"] for r in err}:
                ns = {r["n_total"] for r in err if r["model"] == m and r["set"] == "A" and r["result"] == "all"}
                self.assertEqual(len(ns), 1, (m, ns))
            scored = common.read_csv(os.path.join(d, "scored.csv"))
            n_a5 = sum(1 for r in scored if r["scenario"].startswith("A5"))
            self.assertGreater(n_a5, 0)
            self.assertEqual(sum(int(r["n_total"]) for r in err if r["result"] == "all"), len(scored) - n_a5)
            summ = common.read_csv(os.path.join(d, "summary_model_condition.csv"))
            self.assertEqual(sum(int(r["n_total"]) for r in summ if r["metric"] == "band_err_pp"
                                 and r["result"] in ("pos", "neg")), len(scored) - n_a5)
            cal = common.read_csv(os.path.join(d, "table_calibration.csv"))
            self.assertTrue(cal and all(r["condition"] == "V" for r in cal))
            self.assertIn("table_wbayes_own.tex", os.listdir(gen))
            # L3: no negative zero in any macro; L5: display names in the tables
            text = open(os.path.join(gen, "results_macros.tex")).read()
            self.assertNotRegex(text, r"\\ensuremath\{-\}0\.0+\}")
            self.assertNotIn("{-0.0", text)
            self.assertIn("sumFitCellsOwnN", mac)
            self.assertIn("pfNEmptyMax", mac)
            self.assertIn("tabNoteBTargets", mac)
            # contrast counts and stated-vs-published pretest
            con = common.read_csv(os.path.join(d, "table_contrasts.csv"))
            allr = [r for r in con if r["result"] == "all"]
            self.assertEqual(int(mac["contrastNTotal"]), len(allr))
            ex = [r for r in allr if float(r["ci_lo"]) > 0 or float(r["ci_hi"]) < 0]
            self.assertEqual(int(mac["contrastNExclZero"]), len(ex))
            ag = [r for r in ex if r["hypothesis"] != "H4" and float(r["ci_hi"]) < 0]
            self.assertEqual(int(mac["contrastNExclZeroAgainst"]), len(ag))
            self.assertLessEqual(int(mac["contrastNExclZeroAgainst"]), int(mac["contrastNExclZero"]))
            sp = common.read_csv(os.path.join(d, "table_stated_prior.csv"))
            self.assertEqual(int(mac["statedPriorNScen"]), len(sp))
            self.assertEqual(int(mac["statedPriorNExceed"]),
                             sum(float(r["median_stated_pretest_pct"]) > float(r["published_prior_pct"]) for r in sp))
            self.assertTrue(all(0 < float(r["published_prior_pct"]) < 100 for r in sp))
            # L1: provenance stores a repo-relative raw path
            import json as _j
            prov = _j.load(open(os.path.join(d, "provenance_score.json")))
            self.assertFalse(os.path.isabs(prov["raw_glob"]), prov["raw_glob"])

    def test_tex_num_no_negative_zero(self):
        self.assertEqual(make_tables.tex_num(-0.004, 1), "\\ensuremath{-}0.004")
        self.assertEqual(make_tables.tex_num(-0.0, 1), "0.0")
        self.assertEqual(make_tables.tex_num(-1.26, 1), "\\ensuremath{-}1.3")

    def test_display_names(self):
        self.assertEqual(common.display_name("openai/gpt-4o"), "GPT-4o")
        self.assertEqual(common.display_name("openai/gpt-6.1-sol"), "GPT-6.1 Sol")
        self.assertEqual(common.display_name("anthropic/claude-sonnet-5.5"), "Claude Sonnet 5.5")

    def test_model_tag_letters_only(self):
        self.assertEqual(make_tables.model_tag("gpt-4o"), "GptFourO")
        self.assertEqual(make_tables.model_tag("openai/gpt-6.1-sol"), "OpenaiGptSixOneSol")

    def test_synthetic_inputs_refused_for_real_dirs(self):
        before = {d: listing(d) for d in common.REAL_OUT_DIRS}
        with self.assertRaises(AssertionError):
            score.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES])  # default out = results/derived
        with self.assertRaises(AssertionError):
            fit_wbayes.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES, "--n-boot", "10"])
        with tempfile.TemporaryDirectory() as tmp:
            score.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES, "--out", tmp])
            fit_wbayes.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES, "--out", tmp, "--n-boot", "10"])
            with self.assertRaises(AssertionError):
                make_tables.main(["--derived", tmp, "--n-boot", "10"])  # default generated dir
            if make_figures is not None:
                with self.assertRaises(AssertionError):
                    make_figures.main(["--derived", tmp])  # default paper/figures
        self.assertEqual(before, {d: listing(d) for d in common.REAL_OUT_DIRS})

    def test_missing_provenance_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):
                make_tables.main(["--derived", tmp, "--generated", os.path.join(tmp, "g")])


if __name__ == "__main__":
    unittest.main()
