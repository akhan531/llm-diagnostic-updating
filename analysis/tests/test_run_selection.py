"""Run-id allow-list, unknown run id reporting, latest-attempt dedup, E excluded from fits (M6)."""
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common  # noqa: E402
import fit_wbayes  # noqa: E402
import score  # noqa: E402
from test_pipeline import make_case, make_rec  # noqa: E402

CFG = {"paper_run_ids": ["main"], "pilot_run_ids": ["pilot"], "excluded_run_ids": ["old"]}


def rec(case, run, model="m", cond="N", seed=1, attempt=None, status="ok", post=None):
    t = case["target"]
    r = make_rec(case, model, cond, seed, t["pretest_point"] * 100,
                 (post if post is not None else t["posttest_point"] * 100), status=status)
    r["run_id"] = run
    if attempt is not None:
        r["attempt"] = attempt
    return r


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.cdir, self.rdir, self.out = (os.path.join(d, x) for x in ("cases", "raw", "out"))
        os.makedirs(self.cdir)
        os.makedirs(self.rdir)
        rng = random.Random(5)
        self.cases = []
        for i in range(6):
            pre = rng.uniform(0.05, 0.6)
            for res, lr in (("pos", rng.uniform(4, 15)), ("neg", rng.uniform(0.08, 0.4))):
                self.cases.append(make_case(f"c{i}{res}", f"s{i}", "A", res, pre, lr))
        for c in self.cases:
            with open(os.path.join(self.cdir, c["case_id"] + ".json"), "w") as f:
                json.dump(c, f)
        self.cfgpath = os.path.join(d, "config.json")
        with open(self.cfgpath, "w") as f:
            json.dump(CFG, f)
        self.raw = os.path.join(self.rdir, "*.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, recs):
        with open(os.path.join(self.rdir, "r.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def run_score(self, *extra):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = score.main(["--raw", self.raw, "--cases", self.cdir, "--out", self.out,
                               "--config", self.cfgpath, *extra])
        return rows, buf.getvalue()


class RunSelectionTest(Base):
    def test_allow_list_and_unknown(self):
        c = self.cases[0]
        self.write([rec(c, "main"), rec(c, "main", seed=2), rec(c, "pilot"), rec(c, "old"),
                    rec(c, "mystery"), rec(c, "mystery", seed=2), rec(c, "mystery", seed=3)])
        rows, text = self.run_score()  # default set = paper
        self.assertEqual({r["run_id"] for r in rows}, {"main"})
        self.assertEqual(len(rows), 2)
        self.assertIn("3 in UNKNOWN run ids not scored", text)
        self.assertIn("WARNING: 3 records with UNKNOWN run ids", text)
        self.assertIn("2 in known run ids not selected", text)
        rows, _ = self.run_score("--set", "pilot")
        self.assertEqual({r["run_id"] for r in rows}, {"pilot"})
        rows, _ = self.run_score("--run-ids", "main,old")
        self.assertEqual(len(rows), 3)

    def test_pilot_default_out_is_separate(self):
        a = type("A", (), {"run_ids": None, "run_set": "pilot", "config": self.cfgpath})()
        ids, label, _ = common.resolve_run_ids(a)
        self.assertEqual((ids, label), (["pilot"], "pilot"))
        self.assertEqual(common.default_out(None, "pilot"),
                         os.path.join(common.ROOT, "results", "derived", "pilot"))
        self.assertEqual(common.default_out(None, "paper"), os.path.join(common.ROOT, "results", "derived"))

    def test_nothing_selected_fails_loudly(self):
        self.write([rec(self.cases[0], "mystery")])
        with self.assertRaises(AssertionError):
            self.run_score()

    def test_real_config_is_consistent(self):
        cfg = common.load_config()
        self.assertEqual(cfg["paper_run_ids"], ["main1"])
        self.assertNotIn("pilot1", cfg["paper_run_ids"] + cfg["pilot_run_ids"])


class N4N6Test(Base):
    def test_explicit_pilot_id_is_pilot_set(self):
        a = type("A", (), {"run_ids": "pilot", "run_set": "paper", "config": self.cfgpath})()
        ids, label, _ = common.resolve_run_ids(a)
        self.assertEqual((ids, label), (["pilot"], "pilot"))
        self.assertTrue(common.default_out(None, label).endswith(os.path.join("derived", "pilot")))
        with self.assertRaises(AssertionError):
            common.guard_pilot(os.path.join(common.ROOT, "paper", "generated"), label)

    def test_explicit_non_pilot_stays_explicit(self):
        a = type("A", (), {"run_ids": "main", "run_set": "paper", "config": self.cfgpath})()
        self.assertEqual(common.resolve_run_ids(a)[1], "explicit")

    def test_mixing_pilot_and_paper_ids_refused(self):
        a = type("A", (), {"run_ids": "main,pilot", "run_set": "paper", "config": self.cfgpath})()
        with self.assertRaises(AssertionError):
            common.resolve_run_ids(a)

    def test_truncated_line_skipped_counted_file_untouched(self):
        c = self.cases[0]
        self.write([rec(c, "main"), rec(c, "main", seed=2)])
        path = os.path.join(self.rdir, "r.jsonl")
        with open(path, "a") as f:
            f.write('{"run_id": "main", "case_id": "c0')  # partial last line, no newline
        with open(path, "rb") as f:
            before = f.read()
        rows, text = self.run_score()
        self.assertEqual(len(rows), 2)
        self.assertIn("skipping malformed raw line", text)
        with open(os.path.join(self.out, "provenance_score.json")) as f:
            self.assertEqual(json.load(f)["n_malformed_raw_lines"], 1)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_load_raw_report_zero_when_clean(self):
        self.write([rec(self.cases[0], "main")])
        rep = {}
        with contextlib.redirect_stdout(io.StringIO()):
            common.load_raw(self.raw, rep)
        self.assertEqual(rep["n_malformed"], 0)


class AttemptDedupTest(Base):
    def test_error_then_ok_counted_once(self):
        c = self.cases[0]
        self.write([rec(c, "main", status="call_error", attempt=1), rec(c, "main", attempt=2),
                    rec(c, "main", seed=2)])  # seed 2: single record, attempt missing = 1
        rows, text = self.run_score()
        self.assertEqual(len(rows), 2)
        s1 = [r for r in rows if r["seed"] == 1][0]
        self.assertEqual(s1["parse_ok"], 1)
        att = common.read_csv(os.path.join(self.out, "attempts.csv"))
        self.assertEqual(len(att), 1)
        self.assertEqual((att[0]["n_records"], att[0]["n_keys"], att[0]["n_superseded"],
                          att[0]["n_final_call_error"], att[0]["max_attempt"]), ("3", "2", "1", "0", "2"))
        self.assertIn("1 superseded", text)
        # metrics count the ok attempt once
        summ = common.read_csv(os.path.join(self.out, "summary_model_condition.csv"))
        self.assertTrue(all(r["n_total"] == "2" for r in summ if r["result"] == c["result"]))

    def test_highest_attempt_wins_regardless_of_file_order(self):
        c = self.cases[0]
        self.write([rec(c, "main", attempt=2), rec(c, "main", status="call_error", attempt=1)])
        rows, _ = self.run_score()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["parse_ok"], 1)

    def test_unresolved_error_stays_flagged(self):
        c = self.cases[0]
        self.write([rec(c, "main", status="call_error", attempt=1),
                    rec(c, "main", status="call_error", attempt=2)])
        rows, _ = self.run_score()
        self.assertEqual((len(rows), rows[0]["parse_ok"]), (1, 0))
        att = common.read_csv(os.path.join(self.out, "attempts.csv"))
        self.assertEqual((att[0]["n_superseded"], att[0]["n_final_call_error"]), ("1", "1"))

    def test_same_key_different_run_not_merged(self):
        c = self.cases[0]
        self.write([rec(c, "main"), rec(c, "old")])
        rows, _ = self.run_score("--run-ids", "main,old")
        self.assertEqual(len(rows), 2)

    def test_raw_untouched(self):
        c = self.cases[0]
        self.write([rec(c, "main", status="call_error", attempt=1), rec(c, "main", attempt=2)])
        p = os.path.join(self.rdir, "r.jsonl")
        before = open(p).read()
        self.run_score()
        self.assertEqual(before, open(p).read())


class EExcludedTest(Base):
    def test_e_not_fitted(self):
        recs = []
        for c in self.cases:
            t = c["target"]
            for cond in ("N", "E"):
                recs.append(rec(c, "main", cond=cond))
        self.write(recs)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fit_wbayes.main(["--raw", self.raw, "--cases", self.cdir, "--out", self.out,
                             "--config", self.cfgpath, "--n-boot", "50"])
        fits = common.read_csv(os.path.join(self.out, "wbayes.csv"))
        self.assertTrue(fits)
        self.assertEqual({r["condition"] for r in fits}, {"N"})
        # E records are still scored (reported), just not fitted
        rows, _ = self.run_score()
        self.assertEqual({r["condition"] for r in rows}, {"N", "E"})

    def test_fit_uses_latest_attempt(self):
        recs = []
        for c in self.cases:
            recs.append(rec(c, "main", status="call_error", attempt=1))
            recs.append(rec(c, "main", attempt=2))
        self.write(recs)
        with contextlib.redirect_stdout(io.StringIO()):
            fit_wbayes.main(["--raw", self.raw, "--cases", self.cdir, "--out", self.out,
                             "--config", self.cfgpath, "--n-boot", "50"])
        fits = common.read_csv(os.path.join(self.out, "wbayes.csv"))
        pub = [r for r in fits if r["variant"] == "published"][0]
        self.assertEqual(pub["n_obs"], str(len(self.cases)))
        self.assertAlmostEqual(float(pub["a"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
