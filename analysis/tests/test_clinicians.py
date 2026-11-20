"""Tests for analysis/compare_clinicians.py and the Morgan 2021 clinician baseline. Temp dirs only."""
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import compare_clinicians as cc  # noqa: E402
import score  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "synthetic.jsonl")
CASES = os.path.join(common.ROOT, "data", "cases")
SOURCE_TXT = os.path.join(common.ROOT, "data", "sources", "morgan2021.txt")
ORDER = ["A1_pneumonia", "A2_breast", "A3_cardiac", "A4_uti"]
NUM = r"(\d+(?:\.\d+)?)"


def fmt(x):
    return f"{x:g}"


def source_text():
    return open(SOURCE_TXT, encoding="utf-8").read()


def narrative():
    """Right-column Results text of pdf page 4 (journal p750), whitespace and hyphenation normalised."""
    page = source_text().split("\f")[3]
    lines = page.splitlines()
    # two-column layout: the pneumonia sentence starts at the bottom of the left column and continues at
    # the top of the right column, whose lines start at character 90
    i = next(k for k, ln in enumerate(lines) if "For pneumonia" in ln)
    left = " ".join(ln[:90] for ln in lines[i:i + 2])
    right = " ".join(ln[90:] for ln in lines)
    right = right[re.search(r"90%;\s+evidence", right).start():]  # continuation of the left-column sentence
    s = re.sub(r"\s+", " ", left + " " + right)
    s = re.sub(r"(?<=[a-z])- (?=[a-z])", "", s)
    return re.sub(r"%- ", "%-", s)


@unittest.skipUnless(os.path.exists(SOURCE_TXT), "source full text not distributed (copyright); place Morgan et al. 2021 text at data/sources/morgan2021.txt to run")
class BaselineAgainstSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = cc.load_baseline()
        cls.sc = {s["scenario"]: s for s in cls.base["scenarios"]}

    def test_structure(self):
        self.assertEqual(list(self.sc), ORDER + ["A5_hypothetical"])
        for s in self.base["scenarios"]:
            for cid in s["case_ids"].values():
                self.assertTrue(os.path.exists(os.path.join(CASES, cid + ".json")), cid)
        self.assertEqual(self.base["status"], "VERIFIED")

    def test_pooled_medians_equal_results_text(self):
        text = narrative()
        pat = rf"{NUM}% \(IQR, {NUM}%-{NUM}%; evidence range, ([^;]+?); P < \.001\)"
        found = re.findall(pat, text)
        self.assertEqual(len(found), 12, found)  # 4 scenarios x (pretest, positive, negative)
        i = 0
        for sid in ORDER:
            for key in ("pretest", "post_pos", "post_neg"):
                med, q1, q3, ev = found[i]
                i += 1
                p = self.sc[sid]["items"][key]["pooled"]
                self.assertEqual((float(med), float(q1), float(q3)),
                                 (p["median"], p["iqr_low"], p["iqr_high"]), (sid, key))
                self.assertEqual(ev.replace("%", ""), self.sc[sid]["items"][key]["evidence_range_printed_pct"],
                                 (sid, key))

    def test_hypothetical_pooled_equals_results_text(self):
        text = narrative()
        h = self.sc["A5_hypothetical"]["items"]
        self.assertIn("95% (IQR, 95%-100%)", text)
        self.assertIn("5% (IQR, 0%-5%)", text)
        self.assertEqual((h["post_pos"]["pooled"]["median"], h["post_pos"]["pooled"]["iqr_low"],
                          h["post_pos"]["pooled"]["iqr_high"]), (95, 95, 100))
        self.assertEqual((h["post_neg"]["pooled"]["median"], h["post_neg"]["pooled"]["iqr_low"],
                          h["post_neg"]["pooled"]["iqr_high"]), (5, 0, 5))
        self.assertIsNone(self.sc["A5_hypothetical"]["items"]["pretest"])

    def test_table3_group_cells_equal_source(self):
        page = source_text().split("\f")[5]  # journal p752
        start = page.index("Table 3.")
        end = page.index("Table 4.")
        cells = re.findall(rf"{NUM} \({NUM}-{NUM}\)", page[start:end])
        self.assertEqual(len(cells), 14 * 3)
        i = 0
        for sid in ORDER + ["A5_hypothetical"]:
            for key in ("pretest", "post_pos", "post_neg"):
                item = self.sc[sid]["items"][key]
                if item is None:
                    continue
                for g in ("residents", "attendings", "np_pa"):
                    got = tuple(float(x) for x in cells[i])
                    i += 1
                    v = item["groups"][g]
                    self.assertEqual(got, (v["median"], v["iqr_low"], v["iqr_high"]), (sid, key, g))
        self.assertEqual(i, 14 * 3)

    def test_group_sizes(self):
        g = self.sc["A1_pneumonia"]["items"]["pretest"]["groups"]
        self.assertEqual((g["residents"]["n"], g["attendings"]["n"], g["np_pa"]["n"]), (290, 202, 61))
        self.assertIn("n = 290", source_text().replace("\n", " "))


class ClinicianBandErrors(unittest.TestCase):
    def test_rows_and_pins(self):
        cases = common.load_cases(CASES)
        rows = cc.clinician_rows(cc.load_baseline(), cases)
        cc.check_clinician_rows(rows)
        self.assertEqual(len(rows), (10 + 8) * 4)  # 10 posttest cases (A1-A5 pos/neg) + 8 pretest (A1-A4)
        by = {(r["case_id"], r["quantity"], r["group"]): r for r in rows}
        self.assertEqual(by[("A1_pneumonia_pos", "posttest", "pooled")]["err_abs_pp"], 30.0)
        self.assertEqual(by[("A3_cardiac_neg", "posttest", "pooled")]["err_abs_pp"], 2.5)
        self.assertEqual(by[("A1_pneumonia_neg", "posttest", "residents")]["median"], 60)
        self.assertNotIn(("A5_hypothetical_pos", "pretest", "pooled"), by)

    def test_same_rule_as_score(self):
        self.assertEqual(score.band_error(95, [0.46, 0.65]), 30.0)
        self.assertEqual(score.band_error(50, [0.46, 0.65]), 0.0)

    def test_clinicians_only_reads_no_raw_and_writes_to_given_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, g = os.path.join(tmp, "derived"), os.path.join(tmp, "generated")
            rows, mac = cc.main(["--clinicians-only", "--out", d, "--generated", g])
            self.assertEqual(len(common.read_csv(os.path.join(d, "clinician_band_errors.csv"))), len(rows))
            self.assertEqual(sorted(os.listdir(d)), ["clinician_band_errors.csv"])
            self.assertEqual(mac["clinErrPostPneumoniaPos"], "30.0")
            self.assertEqual(mac["clinMedPrePneumonia"], "80.0")
            self.assertTrue(all(k.isalpha() for k in mac))


class FullModeOnFixture(unittest.TestCase):
    def test_compare_on_synthetic_fixture_in_temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, g = os.path.join(tmp, "derived"), os.path.join(tmp, "generated")
            rows, summ, mac = cc.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES,
                                       "--out", d, "--generated", g])
            for f in ("clinician_compare.csv", "clinician_compare_summary.csv", "clinician_band_errors.csv"):
                self.assertGreater(os.path.getsize(os.path.join(d, f)), 0, f)
            self.assertTrue(rows and summ)
            self.assertTrue(all(r["case_id"].startswith("A") for r in rows))
            for r in rows:
                if r["n_ok"]:
                    self.assertAlmostEqual(r["diff_post"], r["model_post_band_err"] - r["clin_post_band_err"])
                    self.assertEqual(r["clin_source"], "pooled")
                if r["scenario"].startswith("A5"):
                    self.assertNotIn("clin_pre_median", r)
            self.assertTrue(any(k.startswith("clinSetADiffPostMedSYNTH") for k in mac))
            n = cc.check_macros_file(os.path.join(g, "clinician_macros.tex"))
            self.assertEqual(n, len(mac))
            # condition-V pretests exist for A1-A4, so pretest comparison is populated
            self.assertTrue(any(r["diff_pre_median"] != "" for r in summ))

    def test_synthetic_refused_for_real_dirs(self):
        before = {x: sorted(os.listdir(x)) if os.path.isdir(x) else None for x in common.REAL_OUT_DIRS}
        with self.assertRaises(AssertionError):
            cc.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES])
        after = {x: sorted(os.listdir(x)) if os.path.isdir(x) else None for x in common.REAL_OUT_DIRS}
        self.assertEqual(before, after)

    def test_pilot_set_never_writes_paper_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):
                cc.main(["--run-ids", "SYNTHETIC", "--raw", FIXTURE, "--cases", CASES, "--out", tmp,
                         "--generated", os.path.join(common.ROOT, "paper", "generated")])

    def test_unselected_run_id_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):
                cc.main(["--run-ids", "NOPE", "--raw", FIXTURE, "--cases", CASES, "--out", tmp,
                         "--generated", os.path.join(tmp, "g")])


if __name__ == "__main__":
    unittest.main()
