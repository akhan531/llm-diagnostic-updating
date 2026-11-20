#!/usr/bin/env python3
"""Model-calling harness. See harness/README.md. Stdlib only."""
import argparse, csv, datetime, json, os, re, sys, urllib.request, urllib.error

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import spend  # auth_headers lives here so local and cloud paths behave the same
import estimate_cost

URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "v1"
BATCH_LIMIT_USD = 5.0
PLANNING_CAP_USD = 40.0
MAX_CONSECUTIVE_ERRORS = 5   # K in stop with exit 4 after this many call_error records in a row
USAGE_CHECK_EVERY = 50       # live key-usage check (scripts/spend.py fetch) every N calls in real runs
MODEL_ERROR_STREAK = 3       # N1: stop when one model has this many call_error records in a row
ERROR_RATE_MIN_CALLS, ERROR_RATE_MAX = 10, 0.30   # N1: stop when errors/calls > 30% after >= 10 calls in an invocation
USAGE_FAIL_MAX = 2           # N1: a failing live usage check is fatal at the start, or after this many in a row
EXIT_COST_ABORT, EXIT_ERROR_ABORT = 3, 4
GATE_MIN_OK, GATE_EXPECTED = 12, 15
PRETEST_RE = re.compile(r"^\s*PRETEST:\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$", re.M)
POSTTEST_RE = re.compile(r"^\s*POSTTEST:\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$", re.M)
# Prompt versions and the fields each must return (committee extensions to C5 in
#). Each field is parsed from a line "NAME: <number>" (last match wins),
# must be present and in [0, 100]. Parsed keys are the lowercase field names.
PROMPT_FIELDS = {
    "v1": ["PRETEST", "POSTTEST"],
    "v2_pretest_only": ["PRETEST"],
    "v3_believed_accuracy": ["SENSITIVITY", "SPECIFICITY", "PRETEST", "POSTTEST"],
    "v4_unnamed_test": ["PRETEST", "POSTTEST"],
    "v5_para1": ["PRETEST", "POSTTEST"],   # ext6 prompt robustness: reworded v1, same quantities and format
    "v6_para2": ["PRETEST", "POSTTEST"],
}


def field_re(name):
    # identical to PRETEST_RE / POSTTEST_RE (same strictness as v1, so parse rates are comparable)
    return re.compile(r"^\s*%s:\s*([0-9]+(?:\.[0-9]+)?)\s*%%?\s*$" % name, re.M)


def load_prompt(version=PROMPT_VERSION):
    with open(os.path.join(ROOT, "harness", "prompts", f"{version}.txt")) as f:
        return f.read()


def build_prompt(case, note, version=PROMPT_VERSION):
    return load_prompt(version).format(note=note.strip(), diagnosis=case["diagnosis"],
                                       test_name=case["test_name"])


def parse_fields(text, version):
    """Generic parser for any version in PROMPT_FIELDS. Returns (parsed dict or None, status)."""
    if version == "v1":
        return parse_answer(text)
    vals = {}
    for name in PROMPT_FIELDS[version]:
        hits = field_re(name).findall(text or "")
        if not hits:
            return None, "unparseable_missing_field"
        vals[name.lower()] = float(hits[-1])
    if not all(0 <= v <= 100 for v in vals.values()):
        return vals, "out_of_range"
    return vals, "ok"


def scenario_of(case_id):
    """A1_pneumonia_pos -> A1_pneumonia (used by --collapse-polarity)."""
    base, _, pol = case_id.rpartition("_")
    if pol not in ("pos", "neg"):
        raise SystemExit(f"--collapse-polarity: case id {case_id} does not end in _pos or _neg")
    return base


def read_keys_file(path):
    """JSON lines with case_id, condition, model, seed. Returns a set of keys."""
    keys = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                keys.add((d["case_id"], d["condition"], d["model"], int(d["seed"])))
    return keys


def parse_answer(text):
    """Return (parsed dict or None, status). Uses the LAST match of each field."""
    pre, post = PRETEST_RE.findall(text or ""), POSTTEST_RE.findall(text or "")
    if not pre or not post:
        return None, "unparseable_missing_field"
    p, q = float(pre[-1]), float(post[-1])
    if not (0 <= p <= 100 and 0 <= q <= 100):
        return {"pretest": p, "posttest": q}, "out_of_range"
    return {"pretest": p, "posttest": q}, "ok"


def key_of(rec):
    return (rec["case_id"], rec["condition"], rec["model"], rec["seed"])


def read_records(path):
    """All parseable records in file order. A truncated last line (crash mid-write) is skipped with a
    warning; raw files are never edited."""
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"WARNING: {path} line {i} is not valid JSON; skipped (raw file left untouched).",
                          file=sys.stderr)
    return recs


def key_states(path):
    """Return (done, errored). done: keys with at least one record that has a returned response (ok,
    unparseable_*, out_of_range). errored: {key: highest attempt} for keys whose records are all call_error.
    Records without an `attempt` field are attempt 1."""
    done, errs = set(), {}
    for r in read_records(path):
        k = key_of(r)
        if r.get("parse_status") == "call_error":
            errs[k] = max(errs.get(k, 0), r.get("attempt", 1))
        else:
            done.add(k)
    for k in done:
        errs.pop(k, None)
    return done, errs


def completed_keys(path):
    """Keys with a returned response. call_error records are NOT done (they are retried only with
    --retry-errors, as a new appended record; nothing is ever overwritten)."""
    return key_states(path)[0]


def run_cost_so_far(path, exclude_error_estimates=False):
    """Guard cost already recorded in a run's raw file (persists across invocations of the same run id).
    exclude_error_estimates=True drops records whose cost is only the worst-case call_error guard charge
    (cost_source "estimated_call_error"); used by estimate_main so unrun keys are not counted as done (L3).
    The abort guard (run) keeps the default, which counts them, so the guard stays conservative."""
    return sum(r.get("cost_guard_usd", r.get("cost_usd")) or 0.0 for r in read_records(path)
               if not (exclude_error_estimates and r.get("cost_source") == "estimated_call_error"))


def guard_cost(r, prompt, max_tokens, prices):
    """Cost used by the abort guard. Reported cost if present; otherwise a conservative estimate (tokens x
    list price, missing counts filled by chars/4 for input and max_tokens for output). A call_error is charged
    the per-call estimate (chars/4 input, max_tokens output) because a provider may bill a timeout (N1).
    Returns (usd, cost_source)."""
    if r["cost"] is not None:
        return float(r["cost"]), "reported"
    pin, pout = prices
    if r["error"]:
        return len(prompt) / 4 * pin + max_tokens * pout, "estimated_call_error"
    pt, ct = r["prompt_tokens"], r["completion_tokens"]
    src = "estimated_from_tokens" if pt is not None and ct is not None else "estimated_per_call"
    pt = pt if pt is not None else len(prompt) / 4
    ct = ct if ct is not None else max_tokens
    return pt * pin + ct * pout, src


def ledger_total(path=None):
    """Cumulative key usage = max usage_usd in the ledger (the column is cumulative). Missing = 0."""
    path = path or os.path.join(ROOT, "results", "spend_ledger.csv")
    if not os.path.exists(path):
        return 0.0
    with open(path, newline="") as f:
        vals = [float(r["usage_usd"]) for r in csv.DictReader(f) if r.get("usage_usd")]
    return max(vals, default=0.0)


def estimate_batch(jobs, max_tokens, price_fn=estimate_cost.price):
    """jobs: list of (model, prompt). Input tokens ~ chars/4; output assumed = max_tokens (upper bound)."""
    total, cache = 0.0, {}
    for model, prompt in jobs:
        if model not in cache:
            cache[model] = price_fn(model)
        pin, pout = cache[model]
        total += (len(prompt) / 4) * pin + max_tokens * pout
    return total


def check_budget(estimate, ledger_usd, logged_estimate=None, limit=BATCH_LIMIT_USD, cap=PLANNING_CAP_USD):
    """Without a logged estimate, a batch (estimate + ledger) is held to the small pilot limit.
    With --logged-estimate (the figure), the batch is allowed if
    logged estimate + ledger stays under the planning cap; overruns are caught by abort_threshold."""
    if logged_estimate is None:
        if estimate + ledger_usd > limit:
            raise SystemExit(f"REFUSED: estimate ${estimate:.4f} + ledger ${ledger_usd:.4f} exceeds "
                             f"${limit:.2f}; pass --logged-estimate USD.")
    elif logged_estimate + ledger_usd > cap:
        raise SystemExit(f"REFUSED: logged estimate ${logged_estimate:.2f} + ledger ${ledger_usd:.2f} "
                         f"exceeds the ${cap:.2f} planning cap.")


def abort_threshold(logged_estimate):
    """Abort when running cost exceeds the logged estimate by 25% or $5, whichever is lower."""
    return logged_estimate + min(0.25 * logged_estimate, 5.0)


def fake_responder(model, prompt, params):
    """Deterministic fake. Seeds ending in 7 return unparseable text to exercise failure flagging."""
    if params["seed"] % 10 == 7:
        return {"text": "I cannot give numbers.", "prompt_tokens": len(prompt) // 4,
                "completion_tokens": 5, "cost": 0.0, "error": None, "finish_reason": "stop"}
    extra = "SENSITIVITY: 80\nSPECIFICITY: 90\n" if "SENSITIVITY:" in prompt else ""
    return {"text": "Reasoning...\n" + extra + "PRETEST: 30\nPOSTTEST: 60", "prompt_tokens": len(prompt) // 4,
            "completion_tokens": 12, "cost": 0.0, "error": None, "finish_reason": "stop"}


def real_responder(model, prompt, params):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": params["temperature"], "max_tokens": params["max_tokens"],
            "seed": params["seed"], "usage": {"include": True}}
    headers = {"Content-Type": "application/json", **spend.auth_headers()}
    req = urllib.request.Request(URL, json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        u = d.get("usage") or {}
        ch = d["choices"][0]
        return {"text": ch["message"]["content"] or "",
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "cost": u.get("cost"), "error": None, "finish_reason": ch.get("finish_reason"),
                "native_finish_reason": ch.get("native_finish_reason"), "provider": d.get("provider")}
    except Exception as e:  # kept and flagged, never retried here
        return {"text": "", "prompt_tokens": None, "completion_tokens": None, "cost": None,
                "error": f"{type(e).__name__}: {e}"}


def note_path(notes_dir, case_id, condition):
    return os.path.join(notes_dir, f"{case_id}__{condition}.txt")


def _abort(args, msg, code):
    print(msg, file=sys.stderr)
    with open(os.path.join(args.out_dir, f"{args.run_id}.abort.txt"), "a") as af:
        af.write(msg + "\n")
    raise SystemExit(code)


def run(args, responder=None, usage_fn=None):
    responder = responder or (fake_responder if args.dry_run else real_responder)
    if usage_fn is None and not args.dry_run:
        usage_fn = lambda: spend.fetch()[0]
    retry_errors = getattr(args, "retry_errors", False)
    run_estimate = getattr(args, "run_estimate", None)
    version = getattr(args, "prompt_version", None) or PROMPT_VERSION
    if version not in PROMPT_FIELDS:
        raise SystemExit(f"unknown --prompt-version {version}; known: {', '.join(PROMPT_FIELDS)}")
    collapse = getattr(args, "collapse_polarity", False)
    only_keys = read_keys_file(args.keys_file) if getattr(args, "keys_file", None) else None
    if run_estimate is None:
        run_estimate = args.logged_estimate
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.run_id}.jsonl")
    done, errs = key_states(out)
    jobs, skipped_errs, n_keys, seen_ids = [], 0, 0, set()
    for cf in sorted(args.cases):
        case = json.load(open(cf))
        cid = scenario_of(case["case_id"]) if collapse else case["case_id"]
        if cid in seen_ids:   # collapse: pos and neg share one redacted note and one key
            continue
        seen_ids.add(cid)
        for cond in args.conditions:
            if only_keys is not None and not any(k[0] == cid and k[1] == cond for k in only_keys):
                continue   # keys-file runs only open the notes they need (E and A5 notes do not all exist)
            note = open(note_path(args.notes_dir, cid, cond)).read()
            prompt = build_prompt(case, note, version)
            for model in args.models:
                for seed in args.seeds:
                    k = (cid, cond, model, seed)
                    if only_keys is not None and k not in only_keys:
                        continue
                    n_keys += 1
                    if k in done:
                        continue
                    if k in errs:
                        if retry_errors:
                            jobs.append((*k, prompt, errs[k] + 1))
                        else:
                            skipped_errs += 1
                    else:
                        jobs.append((*k, prompt, 1))
    price_fn = (lambda m: (0.0, 0.0)) if args.dry_run else estimate_cost.price
    est = estimate_batch([(j[2], j[4]) for j in jobs], args.max_tokens, price_fn)
    ledger = ledger_total(args.ledger)
    if only_keys is not None and n_keys != len(only_keys):
        raise SystemExit(f"REFUSED: --keys-file lists {len(only_keys)} keys but only {n_keys} match the given "
                         f"cases/conditions/models/seeds.")
    print(f"{len(jobs)} calls to run ({len(done)} already done; {skipped_errs} errored keys left alone, "
          f"use --retry-errors). Estimated ${est:.4f}; ledger ${ledger:.4f}.")
    # N3: on a resume only the unfinished share of the chunk's logged estimate is still to be spent
    remaining_logged = (args.logged_estimate * len(jobs) / n_keys if args.logged_estimate is not None and n_keys
                        else args.logged_estimate)
    check_budget(est, ledger, remaining_logged)
    prices = {m: price_fn(m) for m in {j[2] for j in jobs}}
    params_base = {"temperature": args.temperature, "max_tokens": args.max_tokens}
    prior = run_cost_so_far(out)   # cost already in this run id's raw file (margin persists across invocations)
    limit = abort_threshold(run_estimate) if run_estimate is not None else None
    cap_left = PLANNING_CAP_USD - ledger if args.logged_estimate is not None else None
    usage_start = None
    latest_usage = None
    if usage_fn is not None and jobs:
        try:
            usage_start = usage_fn()
        except Exception as e:   # N1: fail closed; a paid run without a working usage check does not start
            _abort(args, f"ABORTED run {args.run_id}: live usage check failed at start ({e}); no calls made.",
                   EXIT_COST_ABORT)
        latest_usage = usage_start
    base_spend = max(ledger, usage_start or 0.0)
    n_bad, running, consec, n_done = 0, prior, 0, 0
    n_err, model_streak, usage_fails = 0, {}, 0
    # newline guard: a crash can leave a truncated last line; start a fresh line without editing it
    needs_nl = False
    if os.path.exists(out) and os.path.getsize(out) > 0:
        with open(out, "rb") as fb:
            fb.seek(-1, os.SEEK_END)
            needs_nl = fb.read(1) != b"\n"
    with open(out, "a") as f:
        if needs_nl:
            f.write("\n")
        for case_id, cond, model, seed, prompt, attempt in jobs:
            if limit is not None and running > limit:
                _abort(args, f"ABORTED run {args.run_id}: run cost ${running:.4f} (incl. ${prior:.4f} from earlier "
                       f"invocations) exceeded abort threshold ${limit:.4f} (run estimate ${run_estimate:.4f} + "
                       f"min(25%, $5)). Next job not run: {case_id} {cond} {model} seed {seed}.", EXIT_COST_ABORT)
            if cap_left is not None and running - prior > cap_left:
                _abort(args, f"ABORTED run {args.run_id}: this invocation spent ${running - prior:.4f}, above the "
                       f"remaining planning cap ${cap_left:.4f}.", EXIT_COST_ABORT)
            if usage_fn is not None and n_done and n_done % USAGE_CHECK_EVERY == 0:
                try:
                    u = usage_fn()
                except Exception as e:
                    usage_fails += 1
                    print(f"WARNING: live usage check failed ({e}) [{usage_fails}/{USAGE_FAIL_MAX}].", file=sys.stderr)
                    if usage_fails >= USAGE_FAIL_MAX:
                        _abort(args, f"ABORTED run {args.run_id}: {usage_fails} consecutive live usage check "
                               f"failures ({e}).", EXIT_COST_ABORT)
                else:
                    usage_fails, latest_usage = 0, u
                    if u >= PLANNING_CAP_USD or (usage_start is not None and limit is not None
                                                 and u - usage_start > limit - prior):
                        _abort(args, f"ABORTED run {args.run_id}: live key usage ${u:.4f} (at start "
                               f"{usage_start}) breaches the planning cap or run allowance.", EXIT_COST_ABORT)
            # N5: one worst-case call (input chars/4 + max_tokens output at list price) must still fit under the cap
            pin_w, pout_w = prices.get(model, (0.0, 0.0))
            worst_call = len(prompt) / 4 * pin_w + args.max_tokens * pout_w
            spend_now = max(base_spend + (running - prior), latest_usage or 0.0)
            if spend_now + worst_call > PLANNING_CAP_USD:
                _abort(args, f"ABORTED run {args.run_id}: spend ${spend_now:.4f} + worst-case call ${worst_call:.4f} "
                       f"would exceed the ${PLANNING_CAP_USD:.2f} planning cap. Next job not run: {case_id} {cond} "
                       f"{model} seed {seed}.", EXIT_COST_ABORT)
            params = {**params_base, "seed": seed}
            r = responder(model, prompt, params)
            if r["error"]:
                parsed, status = None, "call_error"
            else:
                parsed, status = parse_fields(r["text"], version)
            gc, src = guard_cost(r, prompt, args.max_tokens, prices.get(model, (0.0, 0.0)))
            n_bad += status != "ok"
            running += gc
            n_done += 1
            rec = {"run_id": args.run_id,
                   "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                   "case_id": case_id, "condition": cond, "model": model, "decoding_params": params_base,
                   "seed": seed, "prompt_version": version, "dry_run": bool(args.dry_run),
                   "prompt": prompt, "response": r["text"], "parsed_answer": parsed,
                   "parse_status": status, "flagged": status != "ok", "error": r["error"],
                   "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                   "cost_usd": r["cost"], "attempt": attempt, "cost_source": src,
                   "cost_guard_usd": round(gc, 8),
                   "finish_reason": r.get("finish_reason"), "native_finish_reason": r.get("native_finish_reason"),
                   "provider": r.get("provider"), "collapse_polarity": bool(collapse),
                   "notes_dir": os.path.relpath(os.path.abspath(args.notes_dir), ROOT)}
            if getattr(args, "retry_of", None):
                rec["retry_of_run_id"] = args.retry_of
            f.write(json.dumps(rec) + "\n")
            f.flush()
            is_err = status == "call_error"
            consec = consec + 1 if is_err else 0
            model_streak[model] = model_streak.get(model, 0) + 1 if is_err else 0
            n_err += is_err
            why = None
            if consec >= MAX_CONSECUTIVE_ERRORS:
                why = f"{consec} consecutive call_error records"
            elif model_streak[model] >= MODEL_ERROR_STREAK:
                why = f"{model_streak[model]} consecutive call_error records for {model}"
            elif n_done >= ERROR_RATE_MIN_CALLS and n_err / n_done > ERROR_RATE_MAX:
                why = f"call error rate {n_err}/{n_done} exceeds {ERROR_RATE_MAX:.0%}"
            if why:
                _abort(args, f"ABORTED run {args.run_id}: {why} (last: {r['error']}). Fix the cause, then rerun "
                       f"with --retry-errors.", EXIT_ERROR_ABORT)
    print(f"Wrote {n_done} records to {out} ({n_bad} flagged).")
    return out


def latest_by_key(recs):
    d = {}
    for r in recs:
        k = key_of(r)
        if k not in d or r.get("attempt", 1) >= d[k].get("attempt", 1):
            d[k] = r
    return d


def gate_pilot(path, models, allow_dry=False, expected=None, min_ok=None):
    """Main gate. Returns (ok, message). Uses the latest attempt per key; needs >= GATE_MIN_OK of
    GATE_EXPECTED records parse_status ok and every model with >= 1 ok record. dry_run records fail the gate
    unless allow_dry (DRY_RUN pipeline only)."""
    recs = latest_by_key(read_records(path))
    if not allow_dry and any(r.get("dry_run") for r in recs.values()):
        return False, "pilot contains dry_run records"
    oks = [r for r in recs.values() if r["parse_status"] == "ok"]
    missing = [m for m in models if not any(r["model"] == m for r in oks)]
    expected = GATE_EXPECTED if expected is None else expected
    min_ok = GATE_MIN_OK if min_ok is None else min_ok
    if len(recs) < expected:
        return False, f"pilot has {len(recs)} keys, expected {expected}"
    if len(oks) < min_ok:
        return False, f"only {len(oks)} of {len(recs)} pilot records parsed ok (need {min_ok})"
    if missing:
        return False, f"no ok pilot record for: {', '.join(missing)}"
    return True, f"pilot gate passed: {len(oks)} of {len(recs)} ok, all {len(models)} models present"


def estimate_main(path, calls_by_model, price_fn=estimate_cost.price, spend_usd=None, main_raw=None,
                  floor=None, stat="mean"):
    """Recompute the main estimate from pilot measurements (N2): prompt_tokens and completion_tokens per model
    over the latest attempt of every key that RETURNED a reply (ok, unparseable, out_of_range, truncated; any
    status except call_error) with non-null tokens. Replies that burned the whole token budget and came back
    unparseable are exactly the costly ones, so they are included. stat="mean" (default) is the planning
    estimate; "max" is a stress figure. `floor` lets the operator only raise the total.
    Returns (total_usd, per_model dict), the FULL-run figure. N3: the cap check compares the REMAINING cost
    (total minus guard cost already recorded in `main_raw`) plus current spend (which already includes that
    recorded cost) against the planning cap, so an interrupted main run can resume.
    Raises SystemExit if remaining + spend exceeds the cap."""
    recs = [r for r in latest_by_key(read_records(path)).values() if r.get("parse_status") != "call_error"]
    pick = (lambda xs: sum(xs) / len(xs)) if stat == "mean" else max
    per = {}
    for m, n in calls_by_model.items():
        rs = [r for r in recs if r["model"] == m and r.get("completion_tokens") is not None
              and r.get("prompt_tokens") is not None]
        if not rs:
            raise SystemExit(f"REFUSED: no measured tokens for {m} in {path}")
        pt = pick([r["prompt_tokens"] for r in rs])
        ct = pick([r["completion_tokens"] for r in rs])
        pin, pout = price_fn(m)
        per[m] = n * (pt * pin + ct * pout)
    total = sum(per.values())
    if floor is not None and floor > total:
        per = {m: v * floor / total for m, v in per.items()} if total else per
        total = floor
    if spend_usd is not None:
        done = run_cost_so_far(main_raw, exclude_error_estimates=True) if main_raw else 0.0
        remaining = max(0.0, total - done)
        if remaining + spend_usd > PLANNING_CAP_USD:
            raise SystemExit(f"REFUSED: remaining estimate ${remaining:.2f} (full ${total:.2f} less ${done:.2f} "
                             f"already recorded for the run) + spend ${spend_usd:.2f} exceeds the "
                             f"${PLANNING_CAP_USD:.2f} planning cap.")
    return total, per


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "gate":
        ap = argparse.ArgumentParser(prog="run.py gate")
        ap.add_argument("--pilot", required=True)
        ap.add_argument("--models", nargs="+", required=True)
        ap.add_argument("--allow-dry", action="store_true", help="DRY_RUN pipeline only")
        ap.add_argument("--expected", type=int, default=None, help=f"pilot keys expected (default {GATE_EXPECTED})")
        ap.add_argument("--min-ok", type=int, default=None, help=f"ok records needed (default {GATE_MIN_OK})")
        a = ap.parse_args(argv[1:])
        ok, msg = gate_pilot(a.pilot, a.models, a.allow_dry, a.expected, a.min_ok)
        print(msg)
        raise SystemExit(0 if ok else 1)
    if argv and argv[0] == "estimate-main":
        ap = argparse.ArgumentParser(prog="run.py estimate-main")
        ap.add_argument("--pilot", required=True)
        ap.add_argument("--calls", nargs="+", required=True, help="model=ncalls ...")
        ap.add_argument("--main-raw", default=None, help="main run raw file; its recorded cost is subtracted")
        ap.add_argument("--floor", type=float, default=None, help="operator estimate; can only raise the total")
        ap.add_argument("--spend", type=float, default=None, help="current spend; default live from spend.fetch()")
        ap.add_argument("--per-model-out", default=None, help="write per-model full-run estimates as JSON here (L2)")
        a = ap.parse_args(argv[1:])
        calls = {kv.rsplit("=", 1)[0]: int(kv.rsplit("=", 1)[1]) for kv in a.calls}
        spent = a.spend if a.spend is not None else spend.fetch()[0]
        total, per = estimate_main(a.pilot, calls, spend_usd=spent, main_raw=a.main_raw, floor=a.floor)
        try:
            print(f"  stress (max tokens per model): ${estimate_main(a.pilot, calls, stat='max')[0]:.2f}",
                  file=sys.stderr)
        except SystemExit:
            pass
        for m, v in per.items():
            print(f"  {m}: ${v:.2f}", file=sys.stderr)
        if a.per_model_out:
            with open(a.per_model_out, "w") as pf:
                json.dump(per, pf)
        print(f"  current spend ${spent:.2f}", file=sys.stderr)
        print(f"{total:.2f}")
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cases", nargs="+", required=True, help="case JSON files")
    ap.add_argument("--conditions", nargs="+", required=True, help="e.g. V N B S E (E: A1-A4, B1-B6 only)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--notes-dir", default=os.path.join(ROOT, "data", "notes"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "raw"))
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--logged-estimate", type=float, default=None,
                    help="batch cost estimate; enables the auto guard")
    ap.add_argument("--run-estimate", type=float, default=None,
                    help="total estimate for the whole run id (all chunks); abort margin is set from it and "
                         "counts cost already in the raw file. Default: --logged-estimate")
    ap.add_argument("--prompt-version", default=PROMPT_VERSION, choices=sorted(PROMPT_FIELDS),
                    help="prompt file harness/prompts/<version>.txt; also selects the parser")
    ap.add_argument("--collapse-polarity", action="store_true",
                    help="case id = scenario (pos/neg share one note, e.g. data/notes/pretest/<scenario>__N.txt)")
    ap.add_argument("--keys-file", default=None,
                    help="JSON lines {case_id, condition, model, seed}: run only these keys (e.g. a retry spec)")
    ap.add_argument("--retry-of", default=None, help="run id these keys re-run (recorded as retry_of_run_id)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run keys whose only records are call_error; appends a new record with attempt+1")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
