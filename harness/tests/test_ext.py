"""Tests for the committee-extension harness features (prompt versions v2 to v4, collapse-polarity,
keys-file reruns, finish_reason, gate --expected). Dry run only; no network."""
import json, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run as H  # noqa: E402


class Ext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "raw")
        self.ledger = os.path.join(self.tmp.name, "none.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def recs(self, rid):
        return [json.loads(l) for l in open(os.path.join(self.out, rid + ".jsonl"))]

    def base(self, rid, cases, conds, notes_dir, extra=()):
        return ["--run-id", rid, "--cases", *cases, "--conditions", *conds, "--models", "fake/a", "fake/b",
                "--seeds", "1", "2", "--notes-dir", notes_dir, "--out-dir", self.out, "--ledger", self.ledger,
                "--dry-run", *extra]

    # ---- parsers
    def test_parse_v2(self):
        self.assertEqual(H.parse_fields("x\nPRETEST: 12", "v2_pretest_only"), ({"pretest": 12.0}, "ok"))
        self.assertEqual(H.parse_fields("POSTTEST: 12", "v2_pretest_only")[1], "unparseable_missing_field")
        self.assertEqual(H.parse_fields("PRETEST: 101", "v2_pretest_only")[1], "out_of_range")

    def test_parse_v3(self):
        t = "think\nSENSITIVITY: 85\nSPECIFICITY: 70.5\nPRETEST: 20\nPOSTTEST: 40"
        self.assertEqual(H.parse_fields(t, "v3_believed_accuracy"),
                         ({"sensitivity": 85.0, "specificity": 70.5, "pretest": 20.0, "posttest": 40.0}, "ok"))
        self.assertEqual(H.parse_fields("PRETEST: 20\nPOSTTEST: 40", "v3_believed_accuracy")[1],
                         "unparseable_missing_field")
        self.assertEqual(H.parse_fields(t.replace("85", "185"), "v3_believed_accuracy")[1], "out_of_range")
        # last match wins, as in v1
        self.assertEqual(H.parse_fields("SENSITIVITY: 1\n" + t, "v3_believed_accuracy")[0]["sensitivity"], 85.0)

    def test_parse_v1_and_v4_same_as_before(self):
        for v in ("v1", "v4_unnamed_test", "v5_para1", "v6_para2"):
            self.assertEqual(H.parse_fields("PRETEST: 12.5\nPOSTTEST: 3", v), ({"pretest": 12.5, "posttest": 3.0}, "ok"))
            self.assertEqual(H.parse_fields("PRETEST: 12", v)[1], "unparseable_missing_field")

    def test_paraphrase_prompts_same_quantities(self):
        case = {"diagnosis": "DIAG_X", "test_name": "TEST_Y"}
        base = H.build_prompt(case, "NOTE_Z", "v1")
        for v in ("v5_para1", "v6_para2"):
            p = H.build_prompt(case, "NOTE_Z", v)
            self.assertNotEqual(p, base)
            self.assertIn("NOTE_Z", p)
            self.assertIn("DIAG_X", p)
            self.assertIn("TEST_Y", p)
            self.assertTrue(p.count("PRETEST: <number>") == 1 and p.count("POSTTEST: <number>") == 1)
            for w in ("bayes", "likelihood", "odds", "base rate", "sensitivity"):
                self.assertNotIn(w, p.lower())
            # answer text parses exactly as v1 does
            txt = "reasoning...\nPRETEST: 20\nPOSTTEST: 61.5%"
            self.assertEqual(H.parse_fields(txt, v), H.parse_answer(txt))
            self.assertEqual(H.parse_fields("PRETEST: 20", v)[1], "unparseable_missing_field")
            self.assertEqual(H.parse_fields("PRETEST: 20\nPOSTTEST: 120", v)[1], "out_of_range")

    def test_prompts_exist_and_format(self):
        case = {"diagnosis": "DIAG_X", "test_name": "TEST_Y"}
        for v in H.PROMPT_FIELDS:
            p = H.build_prompt(case, "NOTE_Z", v)
            self.assertIn("NOTE_Z", p)
            self.assertIn("DIAG_X", p)
            for f in H.PROMPT_FIELDS[v]:
                self.assertIn(f + ": <number>", p)
            if v == "v4_unnamed_test":
                self.assertNotIn("TEST_Y", p)
            else:
                self.assertIn("TEST_Y", p)
        self.assertNotIn("POSTTEST", H.build_prompt(case, "n", "v2_pretest_only"))

    # ---- collapse-polarity (pretest-only notes)
    def _pretest_fixture(self):
        d = os.path.join(self.tmp.name, "notes")
        os.makedirs(d)
        cases = []
        for pol in ("pos", "neg"):
            cf = os.path.join(self.tmp.name, f"X1_demo_{pol}.json")
            json.dump({"case_id": f"X1_demo_{pol}", "diagnosis": "demo", "test_name": "t"}, open(cf, "w"))
            cases.append(cf)
        open(os.path.join(d, "X1_demo__N.txt"), "w").write("SYNTHETIC NOTE\nredacted")
        return cases, d

    def test_collapse_polarity_one_key_per_scenario(self):
        cases, d = self._pretest_fixture()
        H.main(self.base("p1", cases, ["N"], d, ["--collapse-polarity", "--prompt-version", "v2_pretest_only"]))
        r = self.recs("p1")
        self.assertEqual(len(r), 4)    # 1 scenario x 2 models x 2 seeds
        self.assertEqual({x["case_id"] for x in r}, {"X1_demo"})
        self.assertTrue(all(x["prompt_version"] == "v2_pretest_only" for x in r))
        self.assertTrue(all(x["collapse_polarity"] for x in r))
        self.assertEqual(r[0]["parsed_answer"], {"pretest": 30.0})
        self.assertNotIn("POSTTEST:", r[0]["prompt"])
        H.main(self.base("p1", cases, ["N"], d, ["--collapse-polarity", "--prompt-version", "v2_pretest_only"]))
        self.assertEqual(len(self.recs("p1")), 4)   # resumable

    def test_collapse_rejects_bad_ids(self):
        with self.assertRaises(SystemExit):
            H.scenario_of("A5_hypothetical")

    def test_v3_dry_run_records(self):
        fx = os.path.join(HERE, "fixture")
        H.main(self.base("b1", [os.path.join(fx, "case_fixture.json")], ["N"], fx,
                         ["--prompt-version", "v3_believed_accuracy"]))
        r = self.recs("b1")
        self.assertEqual(r[0]["parse_status"], "ok")
        self.assertEqual(set(r[0]["parsed_answer"]), {"sensitivity", "specificity", "pretest", "posttest"})
        self.assertEqual(r[0]["finish_reason"], "stop")

    # ---- keys-file reruns
    def test_keys_file_runs_only_listed_keys(self):
        fx = os.path.join(HERE, "fixture")
        cid = json.load(open(os.path.join(fx, "case_fixture.json")))["case_id"]
        kf = os.path.join(self.tmp.name, "keys.jsonl")
        with open(kf, "w") as f:
            f.write(json.dumps({"case_id": cid, "condition": "N", "model": "fake/b", "seed": 2}) + "\n")
        H.main(self.base("r1", [os.path.join(fx, "case_fixture.json")], ["N", "E"], fx,
                         ["--keys-file", kf, "--retry-of", "main1", "--max-tokens", "16000"]))
        r = self.recs("r1")
        self.assertEqual([(x["model"], x["seed"]) for x in r], [("fake/b", 2)])
        self.assertEqual(r[0]["retry_of_run_id"], "main1")
        self.assertEqual(r[0]["decoding_params"]["max_tokens"], 16000)

    def test_keys_file_refuses_unmatched(self):
        fx = os.path.join(HERE, "fixture")
        kf = os.path.join(self.tmp.name, "keys.jsonl")
        with open(kf, "w") as f:
            f.write(json.dumps({"case_id": "nope", "condition": "N", "model": "fake/b", "seed": 2}) + "\n")
        with self.assertRaises(SystemExit):
            H.main(self.base("r2", [os.path.join(fx, "case_fixture.json")], ["N"], fx, ["--keys-file", kf]))

    # ---- gate with a non-default size
    def test_gate_expected(self):
        p = os.path.join(self.tmp.name, "g.jsonl")
        with open(p, "w") as f:
            for i, m in enumerate(["m1", "m2", "m1"]):
                f.write(json.dumps({"case_id": f"c{i}", "condition": "N", "model": m, "seed": 1,
                                    "parse_status": "ok", "dry_run": False}) + "\n")
        self.assertTrue(H.gate_pilot(p, ["m1", "m2"], expected=3, min_ok=3)[0])
        self.assertFalse(H.gate_pilot(p, ["m1", "m2"])[0])   # default expects 15

    # ---- retry spec is read-only on raw and finds truncations
    def test_retry_spec(self):
        import make_retry_spec as M
        self.assertTrue(M.truncated({"parse_status": "unparseable_missing_field", "completion_tokens": 4000,
                                     "decoding_params": {"max_tokens": 4000}}))
        self.assertFalse(M.truncated({"parse_status": "ok", "completion_tokens": 3999,
                                      "decoding_params": {"max_tokens": 4000}}))
        self.assertTrue(M.truncated({"parse_status": "ok", "finish_reason": "length", "completion_tokens": 10}))
        self.assertFalse(M.truncated({"parse_status": "call_error", "completion_tokens": None}))


class ExtNotes(unittest.TestCase):
    def test_extension_notes_pass_checks(self):
        sys.path.insert(0, os.path.join(ROOT, "data", "notes"))
        import check_ext
        fails = []
        n = check_ext.check_pretest(fails)
        self.assertEqual(n, 48)   # 40 original + 8 for B7/B8
        self.assertEqual(fails, [])
        n, sizes = check_ext.check_long(fails)
        self.assertEqual(n, 24)   # 20 original + 4 for B7/B8
        self.assertEqual(fails, [])
        self.assertTrue(all(2000 <= x <= 4000 for x in sizes))

    def test_redaction_removes_only_result_sentence(self):
        sys.path.insert(0, os.path.join(ROOT, "data", "notes"))
        import render_ext
        t, rm = render_ext.remove_result_sentence("A: x.\nData: WBC 9. CT: bad thing. Plan: y.\nZ\n", "bad thing")
        self.assertEqual(rm, "CT: bad thing.")
        self.assertEqual(t, "A: x.\nData: WBC 9. Plan: y.\nZ\n")
        t, _ = render_ext.remove_result_sentence("Results:\nlabs ok.\nCXR: bad.\n\nA/P: z\n", "bad")
        self.assertEqual(t, "Results:\nlabs ok.\n\nA/P: z\n")


class Ext8Script(unittest.TestCase):
    def test_ext8_dry_run_single_model_186_keys(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, DRY_RUN="1", RAW_DIR=d)
            sh = os.path.join(ROOT, "scripts", "run_ext.sh")
            p = subprocess.run(["bash", sh, "pilot", "ext8_family"], env=env, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(p.returncode, 0, p.stderr)
            m = subprocess.run(["bash", sh, "main", "ext8_family"], env=env, capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(m.returncode, 0, m.stderr)
            recs = [json.loads(l) for l in open(os.path.join(d, "ext8_family.jsonl"))]
            self.assertEqual(len(recs), 186)
            self.assertEqual({r["model"] for r in recs}, {"qwen/qwen4-max"})
            self.assertEqual({r["condition"] for r in recs}, {"V", "N", "E"})
            self.assertFalse(any(r["case_id"].startswith(("B7_", "B8_")) for r in recs))


if __name__ == "__main__":
    unittest.main()
