import json, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import run as H


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "raw")
        self.ledger = os.path.join(self.tmp.name, "none.csv")  # missing ledger = $0

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, seeds):
        return ["--run-id", "t1", "--cases", os.path.join(HERE, "fixture", "case_fixture.json"),
                "--conditions", "N", "--models", "fake/model-a", "fake/model-b", "--seeds", *map(str, seeds),
                "--notes-dir", os.path.join(HERE, "fixture"), "--out-dir", self.out,
                "--ledger", self.ledger, "--dry-run"]

    def recs(self):
        return [json.loads(l) for l in open(os.path.join(self.out, "t1.jsonl"))]

    def test_dry_run_resume_and_flagging(self):
        H.main(self.args([1, 7]))
        r = self.recs()
        self.assertEqual(len(r), 4)
        for k in ["run_id", "timestamp", "case_id", "condition", "model", "decoding_params", "seed", "prompt",
                  "response", "parsed_answer", "parse_status", "prompt_tokens", "completion_tokens", "cost_usd",
                  "prompt_version"]:
            self.assertIn(k, r[0])
        ok = [x for x in r if x["seed"] == 1][0]
        self.assertEqual(ok["parsed_answer"], {"pretest": 30.0, "posttest": 60.0})
        bad = [x for x in r if x["seed"] == 7][0]
        self.assertTrue(bad["flagged"])
        self.assertEqual(bad["parse_status"], "unparseable_missing_field")
        H.main(self.args([1, 7]))  # resume: nothing new, bad one not retried
        self.assertEqual(len(self.recs()), 4)
        H.main(self.args([1, 7, 2]))  # only the new seed is added
        self.assertEqual(len(self.recs()), 6)

    def test_parser(self):
        self.assertEqual(H.parse_answer("PRETEST: 12.5\nPOSTTEST: 3")[1], "ok")
        self.assertEqual(H.parse_answer("PRETEST: 12")[1], "unparseable_missing_field")
        self.assertEqual(H.parse_answer("PRETEST: 120\nPOSTTEST: 3")[1], "out_of_range")

    def test_budget_check(self):
        with self.assertRaises(SystemExit):
            H.check_budget(4.0, 1.5)
        H.check_budget(1.0, 1.0)
        H.check_budget(30.0, 1.5, logged_estimate=20.0)
        with self.assertRaises(SystemExit):
            H.check_budget(30.0, 25.0, logged_estimate=20.0)  # logged + ledger > $40 cap

    def test_abort_threshold(self):
        self.assertAlmostEqual(H.abort_threshold(10.0), 12.5)   # 25% is lower than $5
        self.assertAlmostEqual(H.abort_threshold(40.0), 45.0)   # $5 is lower than 25%

    def test_auto_abort_records_reason(self):
        def costly(model, prompt, params):
            return {"text": "PRETEST: 10\nPOSTTEST: 20", "prompt_tokens": 1, "completion_tokens": 1,
                    "cost": 0.5, "error": None}
        import types
        ns = types.SimpleNamespace(run_id="t1", cases=[os.path.join(HERE, "fixture", "case_fixture.json")],
                                   conditions=["N"], models=["fake/model-a"], seeds=[1, 2, 3, 4, 5, 6],
                                   temperature=1.0, max_tokens=100, notes_dir=os.path.join(HERE, "fixture"),
                                   out_dir=self.out, ledger=self.ledger, dry_run=True, logged_estimate=1.0)
        with self.assertRaises(SystemExit) as cm:
            H.run(ns, responder=costly)
        self.assertEqual(cm.exception.code, H.EXIT_COST_ABORT)  # abort exits nonzero
        # threshold 1.25: costs 0.5 each, abort after the third call (running 1.5 > 1.25)
        self.assertEqual(len(self.recs()), 3)
        self.assertIn("ABORTED", open(os.path.join(self.out, "t1.abort.txt")).read())

    def test_estimate_uses_prices(self):
        est = H.estimate_batch([("m", "x" * 400)], 100, lambda m: (0.001, 0.01))
        self.assertAlmostEqual(est, 100 * 0.001 + 100 * 0.01)

    def test_plan_cost(self):
        import estimate_cost as E
        per, tot = E.plan_cost({"a": (0.001, 0.01), "b": (0.0, 0.02)}, 10, 100, {"a": 50, "b": 25})
        self.assertAlmostEqual(per["a"], 10 * (0.1 + 0.5))
        self.assertAlmostEqual(per["b"], 10 * 0.5)
        self.assertAlmostEqual(tot, 11.0)

    def test_ledger_total(self):
        p = os.path.join(self.tmp.name, "l.csv")
        open(p, "w").write("timestamp_utc,usage_usd,key_limit,note\na,1.5,50,x\nb,2.25,50,y\n")
        self.assertEqual(H.ledger_total(p), 2.25)
        self.assertEqual(H.ledger_total(self.ledger), 0.0)

    # ---- tests ----
    def ns(self, **kw):
        import types
        d = dict(run_id="t1", cases=[os.path.join(HERE, "fixture", "case_fixture.json")], conditions=["N"],
                 models=["fake/model-a"], seeds=[1], temperature=1.0, max_tokens=100,
                 notes_dir=os.path.join(HERE, "fixture"), out_dir=self.out, ledger=self.ledger, dry_run=True,
                 logged_estimate=None, run_estimate=None, retry_errors=False)
        d.update(kw)
        return types.SimpleNamespace(**d)

    @staticmethod
    def err(model, prompt, params):
        return {"text": "", "prompt_tokens": None, "completion_tokens": None, "cost": None,
                "error": "HTTPError: 429"}

    @staticmethod
    def good(cost=0.0, pt=10, ct=20):
        return lambda model, prompt, params: {"text": "PRETEST: 10\nPOSTTEST: 20", "prompt_tokens": pt,
                                               "completion_tokens": ct, "cost": cost, "error": None}

    def test_error_only_pilot_fails_gate(self):
        path = os.path.join(self.out, "pilot.jsonl")
        os.makedirs(self.out)
        with open(path, "w") as f:
            for i in range(15):
                f.write(json.dumps({"case_id": f"c{i % 3}", "condition": "B", "model": f"m{i % 5}", "seed": 1,
                                    "parse_status": "call_error", "dry_run": False}) + "\n")
        ok, msg = H.gate_pilot(path, [f"m{i}" for i in range(5)])
        self.assertFalse(ok)
        self.assertIn("0 of 15", msg)

    def test_gate_pass_and_thresholds(self):
        path = os.path.join(self.out, "pilot.jsonl")
        os.makedirs(self.out)
        def write(n_ok, dry=False):
            with open(path, "w") as f:
                for i in range(15):
                    f.write(json.dumps({"case_id": f"c{i % 3}", "condition": "B", "model": f"m{i % 5}", "seed": 1,
                                        "parse_status": "ok" if i < n_ok else "unparseable_missing_field",
                                        "dry_run": dry}) + "\n")
        models = [f"m{i}" for i in range(5)]
        write(12); self.assertTrue(H.gate_pilot(path, models)[0])
        write(11); self.assertFalse(H.gate_pilot(path, models)[0])
        write(15, dry=True)
        self.assertFalse(H.gate_pilot(path, models)[0])            # dry records fail a real gate
        self.assertTrue(H.gate_pilot(path, models, allow_dry=True)[0])  # DRY_RUN pipeline only
        write(12); self.assertFalse(H.gate_pilot(path, models + ["missing"])[0])

    def test_consecutive_error_stop(self):
        a = self.ns(seeds=list(range(1, 9)))
        old, H.MODEL_ERROR_STREAK = H.MODEL_ERROR_STREAK, 99   # isolate the global streak rule
        try:
            with self.assertRaises(SystemExit) as cm:
                H.run(a, responder=self.err)
        finally:
            H.MODEL_ERROR_STREAK = old
        self.assertEqual(cm.exception.code, H.EXIT_ERROR_ABORT)
        self.assertEqual(len(self.recs()), H.MAX_CONSECUTIVE_ERRORS)
        self.assertIn("consecutive", open(os.path.join(self.out, "t1.abort.txt")).read())

    def test_retry_appends_attempt2_without_duplicating_ok(self):
        H.run(self.ns(seeds=[1]), responder=self.good())          # seed 1 ok
        H.run(self.ns(seeds=[2]), responder=self.err)               # seed 2 call_error
        before = open(os.path.join(self.out, "t1.jsonl")).read()
        H.run(self.ns(seeds=[1, 2]), responder=self.good())         # no flag: nothing runs
        self.assertEqual(open(os.path.join(self.out, "t1.jsonl")).read(), before)
        H.run(self.ns(seeds=[1, 2], retry_errors=True), responder=self.good())
        r = self.recs()
        self.assertEqual(len(r), 3)
        self.assertTrue(open(os.path.join(self.out, "t1.jsonl")).read().startswith(before))  # append only
        self.assertEqual([(x["seed"], x["attempt"], x["parse_status"]) for x in r],
                         [(1, 1, "ok"), (2, 1, "call_error"), (2, 2, "ok")])
        H.run(self.ns(seeds=[1, 2], retry_errors=True), responder=self.good())  # now all done
        self.assertEqual(len(self.recs()), 3)

    def test_missing_cost_guard(self):
        pr = (0.001, 0.01)
        c, src = H.guard_cost({"cost": None, "error": None, "prompt_tokens": 100, "completion_tokens": 50},
                              "x" * 400, 4000, pr)
        self.assertEqual(src, "estimated_from_tokens")
        self.assertAlmostEqual(c, 100 * 0.001 + 50 * 0.01)
        c, src = H.guard_cost({"cost": None, "error": None, "prompt_tokens": None, "completion_tokens": None},
                              "x" * 400, 4000, pr)
        self.assertEqual(src, "estimated_per_call")
        self.assertAlmostEqual(c, 100 * 0.001 + 4000 * 0.01)
        self.assertEqual(H.guard_cost({"cost": 0.3, "error": None, "prompt_tokens": 1, "completion_tokens": 1},
                                      "", 1, pr), (0.3, "reported"))

    def test_missing_cost_triggers_abort_not_zero(self):
        # responder reports no cost; price is zero in dry mode, so patch prices via estimate_cost.price
        import estimate_cost
        orig = estimate_cost.price
        estimate_cost.price = lambda m: (0.0, 0.1)
        try:
            a = self.ns(dry_run=False, logged_estimate=1.0, seeds=[1, 2, 3, 4, 5, 6])
            with self.assertRaises(SystemExit) as cm:
                H.run(a, responder=self.good(cost=None, ct=5), usage_fn=lambda: 0.0)
        finally:
            estimate_cost.price = orig
        self.assertEqual(cm.exception.code, H.EXIT_COST_ABORT)
        r = self.recs()
        self.assertTrue(all(x["cost_usd"] is None and x["cost_source"] == "estimated_from_tokens" for x in r))
        self.assertAlmostEqual(r[0]["cost_guard_usd"], 0.5)   # 5 tokens x 0.1
        self.assertEqual(len(r), 3)  # 0.5 each, threshold 1.25, abort before the 4th call

    def test_abort_margin_persists_across_invocations(self):
        a = self.ns(logged_estimate=1.0, seeds=[1, 2, 3, 4, 5, 6], run_estimate=1.0)
        with self.assertRaises(SystemExit):
            H.run(a, responder=self.good(cost=0.5))
        self.assertEqual(len(self.recs()), 3)
        with self.assertRaises(SystemExit) as cm:  # a rerun gets no fresh allowance
            H.run(a, responder=self.good(cost=0.5))
        self.assertEqual(cm.exception.code, H.EXIT_COST_ABORT)
        self.assertEqual(len(self.recs()), 3)
        self.assertAlmostEqual(H.run_cost_so_far(os.path.join(self.out, "t1.jsonl")), 1.5)

    def test_truncated_trailing_line_tolerated(self):
        H.run(self.ns(seeds=[1]), responder=self.good())
        path = os.path.join(self.out, "t1.jsonl")
        with open(path, "a") as f:
            f.write('{"run_id": "t1", "case')  # crash mid-write
        H.run(self.ns(seeds=[1, 2]), responder=self.good())
        self.assertEqual(len(H.read_records(path)), 2)

    def test_estimate_main_from_pilot_and_cap(self):
        path = os.path.join(self.out, "pilot.jsonl")
        os.makedirs(self.out)
        with open(path, "w") as f:
            f.write(json.dumps({"case_id": "c", "condition": "B", "model": "m", "seed": 1, "parse_status": "ok",
                                "prompt_tokens": 400, "completion_tokens": 1000}) + "\n")
        total, _ = H.estimate_main(path, {"m": 10}, lambda m: (0.0001, 0.0001), spend_usd=1.0)
        self.assertAlmostEqual(total, 10 * (0.04 + 0.1))
        with self.assertRaises(SystemExit):
            H.estimate_main(path, {"m": 10}, lambda m: (0.001, 0.01), spend_usd=1.0)  # $104 > cap

    # ---- N1, N2, N3, N5 (C2 re-verification) ----
    def with_price(self, pin, pout):
        import estimate_cost
        orig = estimate_cost.price
        estimate_cost.price = lambda m: (pin, pout)
        self.addCleanup(setattr, estimate_cost, "price", orig)

    def test_n1_per_model_error_streak(self):
        with self.assertRaises(SystemExit) as cm:
            H.run(self.ns(seeds=list(range(1, 9))), responder=self.err)
        self.assertEqual(cm.exception.code, H.EXIT_ERROR_ABORT)
        self.assertEqual(len(self.recs()), H.MODEL_ERROR_STREAK)
        self.assertIn("for fake/model-a", open(os.path.join(self.out, "t1.abort.txt")).read())

    def test_n1_error_rate_stop_without_streaks(self):
        def flaky(model, prompt, params):   # errors on 2 of every 3 seeds: no streak longer than 2
            return self.err(model, prompt, params) if params["seed"] % 3 else self.good()(model, prompt, params)
        with self.assertRaises(SystemExit) as cm:
            H.run(self.ns(seeds=list(range(1, 31))), responder=flaky)
        self.assertEqual(cm.exception.code, H.EXIT_ERROR_ABORT)
        self.assertEqual(len(self.recs()), H.ERROR_RATE_MIN_CALLS)
        self.assertIn("error rate", open(os.path.join(self.out, "t1.abort.txt")).read())

    def test_n1_call_errors_are_charged(self):
        self.with_price(0.0, 0.01)   # per-call estimate: 100 max_tokens x 0.01 = 1.0 plus input
        a = self.ns(dry_run=False, logged_estimate=1.0, run_estimate=1.0, seeds=[1, 2, 3, 4])
        flaky = lambda m, p, pa: self.err(m, p, pa) if pa["seed"] % 2 else self.good(cost=0.0)(m, p, pa)
        with self.assertRaises(SystemExit) as cm:
            H.run(a, responder=flaky, usage_fn=lambda: 0.0)
        self.assertEqual(cm.exception.code, H.EXIT_COST_ABORT)   # threshold 1.25: second error charged 1.0 each
        r = self.recs()
        self.assertEqual(r[0]["cost_source"], "estimated_call_error")
        self.assertAlmostEqual(r[0]["cost_guard_usd"], 1.0)
        self.assertIsNone(r[0]["cost_usd"])

    def test_n1_usage_check_failure_is_fatal(self):
        self.with_price(0.0, 0.0)
        def bad():
            raise OSError("net")
        a = self.ns(dry_run=False, logged_estimate=5.0, seeds=[1, 2])
        with self.assertRaises(SystemExit) as cm:
            H.run(a, responder=self.good(), usage_fn=bad)
        self.assertEqual(cm.exception.code, H.EXIT_COST_ABORT)
        self.assertFalse(os.path.exists(os.path.join(self.out, "t1.jsonl")) and self.recs())  # no call made
        # periodic: two failures in a row stop the run, a single failure does not
        old, H.USAGE_CHECK_EVERY = H.USAGE_CHECK_EVERY, 2
        self.addCleanup(setattr, H, "USAGE_CHECK_EVERY", old)
        seq = iter([0.0, "fail", "fail"])
        def flaky():
            v = next(seq)
            if v == "fail":
                raise OSError("net")
            return v
        with self.assertRaises(SystemExit) as cm:
            H.run(self.ns(dry_run=False, run_id="t2", logged_estimate=5.0, seeds=list(range(1, 12))),
                  responder=self.good(), usage_fn=flaky)
        self.assertEqual(cm.exception.code, H.EXIT_COST_ABORT)
        self.assertEqual(len(H.read_records(os.path.join(self.out, "t2.jsonl"))), 4)  # checks at 2 and 4

    def test_n2_estimate_includes_unparseable_and_truncated(self):
        path = os.path.join(self.out, "pilot.jsonl")
        os.makedirs(self.out)
        rows = [("ok", 1000), ("unparseable_missing_field", 4000), ("out_of_range", 4000), ("call_error", None)]
        with open(path, "w") as f:
            for i, (st, ct) in enumerate(rows):
                f.write(json.dumps({"case_id": f"c{i}", "condition": "B", "model": "m", "seed": 1,
                                    "parse_status": st, "prompt_tokens": 100 if ct else None,
                                    "completion_tokens": ct}) + "\n")
        pr = lambda m: (0.0, 0.001)
        total, _ = H.estimate_main(path, {"m": 10}, pr)
        self.assertAlmostEqual(total, 10 * 3000 * 0.001)          # mean of 1000, 4000, 4000 (ok-only gives 10)
        self.assertAlmostEqual(H.estimate_main(path, {"m": 10}, pr, stat="max")[0], 10 * 4000 * 0.001)

    def test_n3_resume_compares_remaining_cost(self):
        path = os.path.join(self.out, "pilot.jsonl")
        main = os.path.join(self.out, "main1.jsonl")
        os.makedirs(self.out)
        with open(path, "w") as f:
            f.write(json.dumps({"case_id": "c", "condition": "B", "model": "m", "seed": 1, "parse_status": "ok",
                                "prompt_tokens": 0, "completion_tokens": 1000}) + "\n")
        pr = lambda m: (0.0, 0.001)    # full estimate 27.0 for 27 calls x $1
        calls = {"m": 27}
        with self.assertRaises(SystemExit):
            H.estimate_main(path, calls, pr, spend_usd=18.0)                       # fresh: 27 + 18 > 40
        with open(main, "w") as f:
            f.write(json.dumps({"cost_guard_usd": 15.0}) + "\n")                    # 15 already spent on main1
        total, _ = H.estimate_main(path, calls, pr, spend_usd=18.0, main_raw=main)   # remaining 12 + 18 <= 40
        self.assertAlmostEqual(total, 27.0)                                         # full figure still returned
        with self.assertRaises(SystemExit):
            H.estimate_main(path, calls, pr, spend_usd=29.0, main_raw=main)         # remaining 12 + 29 > 40
        total, _ = H.estimate_main(path, calls, pr, spend_usd=1.0, main_raw=main, floor=30.0)
        self.assertAlmostEqual(total, 30.0)                                         # floor only raises
        self.assertAlmostEqual(H.estimate_main(path, calls, pr, floor=1.0)[0], 27.0)

    def test_l3_error_guard_charges_not_subtracted_as_done(self):
        path = os.path.join(self.out, "pilot.jsonl")
        main = os.path.join(self.out, "main1.jsonl")
        os.makedirs(self.out)
        with open(path, "w") as f:
            f.write(json.dumps({"case_id": "c", "condition": "B", "model": "m", "seed": 1, "parse_status": "ok",
                                "prompt_tokens": 0, "completion_tokens": 1000}) + "\n")
        with open(main, "w") as f:
            f.write(json.dumps({"cost_guard_usd": 5.0, "cost_source": "reported"}) + "\n")
            f.write(json.dumps({"cost_guard_usd": 6.0, "cost_source": "estimated_call_error"}) + "\n")
        self.assertAlmostEqual(H.run_cost_so_far(main), 11.0)                 # abort guard stays conservative
        self.assertAlmostEqual(H.run_cost_so_far(main, exclude_error_estimates=True), 5.0)
        pr = lambda m: (0.0, 0.001)
        # remaining = 27 - 5 = 22 (not 16); 22 + 19 > 40 refused, whereas 16 + 19 would have passed
        with self.assertRaises(SystemExit):
            H.estimate_main(path, {"m": 27}, pr, spend_usd=19.0, main_raw=main)

    def test_n3_chunk_start_check_uses_unfinished_share(self):
        H.run(self.ns(seeds=[1, 2, 3, 4]), responder=self.good())    # 4 of 8 keys done (dry run, zero prices)
        led = os.path.join(self.tmp.name, "l.csv")
        open(led, "w").write("timestamp_utc,usage_usd,key_limit,note\na,30,50,x\n")
        a = self.ns(seeds=[1, 2, 3, 4], models=["fake/model-a", "fake/model-b"], logged_estimate=15.0, ledger=led)
        H.run(a, responder=self.good())   # 15 x (4/8) + 30 <= 40 passes; the full 15 + 30 would be refused
        with self.assertRaises(SystemExit):
            H.run(self.ns(run_id="t9", seeds=[1], logged_estimate=15.0, ledger=led), responder=self.good())

    def test_n5_worst_case_call_must_fit_under_cap(self):
        self.with_price(0.0, 0.01)   # worst call = 100 tokens x 0.01 = $1.00
        led = os.path.join(self.tmp.name, "l.csv")
        open(led, "w").write("timestamp_utc,usage_usd,key_limit,note\na,38.0,50,x\n")
        a = self.ns(dry_run=False, ledger=led, seeds=list(range(1, 8)), logged_estimate=1.0, run_estimate=100.0)
        with self.assertRaises(SystemExit) as cm:
            H.run(a, responder=self.good(cost=0.5), usage_fn=lambda: 38.0)
        self.assertEqual(cm.exception.code, H.EXIT_COST_ABORT)
        # spend 38 + 0.5 per call; the call is refused once spend + $1.00 > 40, i.e. after 3 calls (39.5 + 1.0 > 40)
        self.assertEqual(len(self.recs()), 3)
        total = 38.0 + sum(x["cost_guard_usd"] for x in self.recs())
        self.assertLessEqual(total, 40.0)


if __name__ == "__main__":
    unittest.main()
