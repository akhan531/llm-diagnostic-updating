"""End-to-end tests on synthetic fixtures built in a temp dir (never touches results/raw/)."""
import json
import math
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import fit_wbayes  # noqa: E402
import score  # noqa: E402


def make_case(cid, scen, st, result, pre, lr):
    post = 1 / (1 + (1 - pre) / pre / lr)
    return {"case_id": cid, "scenario": scen, "set": st, "result": result, "diagnosis": "d",
            "test_name": "t", "status": "VERIFIED",
            "target": {"pretest_point": pre, "pretest_range": [pre - 0.05, pre + 0.05],
                       "lr_point": lr, "lr_range": [lr * 0.8, lr * 1.2], "posttest_point": post,
                       "posttest_band_primary": [post - 0.1, post + 0.1],
                       "posttest_band_bayes": [post - 0.05, post + 0.05]}}


def make_rec(case, model, cond, seed, pre_pp, post_pp, status="ok", dry=False):
    ok = status == "ok"
    return {"run_id": "t", "case_id": case["case_id"], "condition": cond, "model": model,
            "seed": seed, "parse_status": status, "dry_run": dry,
            "parsed_answer": {"pretest": pre_pp, "posttest": post_pp} if ok else None}


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.cdir, self.rdir, self.out = (os.path.join(d, x) for x in ("cases", "raw", "out"))
        os.makedirs(self.cdir)
        os.makedirs(self.rdir)
        rng = random.Random(3)
        self.cases = []
        for i in range(8):
            pre = rng.uniform(0.05, 0.6)
            for res, lr in (("pos", rng.uniform(4, 15)), ("neg", rng.uniform(0.08, 0.4))):
                self.cases.append(make_case(f"c{i}{res}", f"s{i}", "A", res, pre, lr))
        for c in self.cases:
            with open(os.path.join(self.cdir, c["case_id"] + ".json"), "w") as f:
                json.dump(c, f)
        recs = []
        for c in self.cases:
            t = c["target"]
            # model "perfect": states the published prior and the exact posterior
            recs.append(make_rec(c, "perfect", "N", 1, t["pretest_point"] * 100,
                                 t["posttest_point"] * 100))
        c0 = self.cases[0]
        recs.append(make_rec(c0, "perfect", "N", 7, None, None, status="unparseable_missing_field"))
        recs.append(make_rec(c0, "perfect", "N", 8, 10, 10, dry=True))
        # a known-error record: posterior 30 pp above the primary band's upper edge
        t0 = c0["target"]
        recs.append(make_rec(c0, "biased", "B", 1, 20, (t0["posttest_band_primary"][1] + 0.3) * 100))
        with open(os.path.join(self.rdir, "t.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        self.raw = os.path.join(self.rdir, "*.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def test_score_end_to_end(self):
        rows = score.main(["--run-ids", "t", "--raw", self.raw, "--cases", self.cdir, "--out", self.out])
        self.assertEqual(len(rows), len(self.cases) + 2)  # dry_run excluded, failure kept
        fails = [r for r in rows if not r["parse_ok"]]
        self.assertEqual(len(fails), 1)
        perfect = [r for r in rows if r["model"] == "perfect" and r["parse_ok"]]
        for r in perfect:
            self.assertEqual(r["band_err_pp"], 0)
            self.assertAlmostEqual(r["logodds_err"], 0, places=6)
            self.assertAlmostEqual(r["coherence_err"], 0, places=6)
        b = [r for r in rows if r["model"] == "biased"][0]
        self.assertAlmostEqual(b["band_err_pp"], 30, places=6)
        self.assertAlmostEqual(b["bayes_band_err_pp"], 35, places=6)
        self.assertGreater(b["logodds_err"], 0)
        # summary and parse failure tables
        summ = common.read_csv(os.path.join(self.out, "summary_model_condition.csv"))
        s = [r for r in summ if r["model"] == "perfect" and r["result"] == "pos"
             and r["metric"] == "band_err_pp"][0]
        self.assertEqual(s["n_parse_fail"], "1")
        self.assertEqual(s["n"], str(int(s["n_total"]) - 1))
        pf = common.read_csv(os.path.join(self.out, "parse_failures.csv"))
        p = [r for r in pf if r["model"] == "perfect"][0]
        self.assertEqual((p["n_total"], p["n_parse_fail"]), ("17", "1"))
        self.assertIn("unparseable_missing_field=1", p["reasons"])

    def test_fit_perfect_bayesian(self):
        # only the "perfect" model has enough data; "biased" has 1 obs -> no fit, reported as such
        fit_wbayes.main(["--run-ids", "t", "--raw", self.raw, "--cases", self.cdir, "--out", self.out,
                         "--n-boot", "100"])
        fits = common.read_csv(os.path.join(self.out, "wbayes.csv"))
        pub = [r for r in fits if r["model"] == "perfect" and r["variant"] == "published"][0]
        own = [r for r in fits if r["model"] == "perfect" and r["variant"] == "own"][0]
        for r in (pub, own):
            self.assertAlmostEqual(float(r["a"]), 1, delta=1e-6)
            self.assertAlmostEqual(float(r["b"]), 1, delta=1e-6)
            self.assertAlmostEqual(float(r["c"]), 0, delta=1e-6)
        bi = [r for r in fits if r["model"] == "biased"]
        self.assertTrue(all(r["a"] == "" and "no fit" in r["note"] for r in bi))

    def test_estimator_recovers_known_parameters(self):
        fit_wbayes.self_check()

    def test_missing_target_field_fails_loudly(self):
        bad = dict(self.cases[0]["target"])
        del bad["lr_point"]
        c = dict(self.cases[0], case_id="bad", target=bad)
        with open(os.path.join(self.cdir, "bad.json"), "w") as f:
            json.dump(c, f)
        with self.assertRaises(AssertionError):
            common.load_cases(self.cdir)


if __name__ == "__main__":
    unittest.main()
