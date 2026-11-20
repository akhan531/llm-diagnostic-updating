#!/usr/bin/env python3
"""P4: build the rerun spec for replies truncated at the token budget in a finished run. Read-only on raw.

Usage: python scripts/make_retry_spec.py [--raw results/raw/main1.jsonl] [--model deepseek/deepseek-v4-pro]
                                         [--out results/retry_specs/ext5_dsretry_keys.jsonl]

A reply counts as truncated when its latest attempt returned text (any status except call_error) and EITHER
finish_reason is "length" (recorded by the harness from Nov 13 on) OR, for records without finish_reason (main1
was written before the field existed), completion_tokens >= the max_tokens in its decoding_params. For main1 the
second rule is the only one available; it is equivalent because the provider stops exactly at max_tokens.
Checks: every selected key's prompt, rebuilt from the current note and prompt v1, equals the recorded prompt
(so the rerun asks the identical question); the spec lists each key once. Exit 1 on any mismatch.
The spec is written OUTSIDE results/raw. The rerun appends NEW records under a NEW run id (ext5_dsretry);
main1.jsonl is never edited.
"""
import argparse, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import run as H  # noqa: E402


def truncated(r):
    if r.get("parse_status") == "call_error":
        return False
    if r.get("finish_reason") is not None:
        return r["finish_reason"] == "length"
    mt = (r.get("decoding_params") or {}).get("max_tokens")
    ct = r.get("completion_tokens")
    return mt is not None and ct is not None and ct >= mt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "raw", "main1.jsonl"))
    ap.add_argument("--model", default="deepseek/deepseek-v4-pro")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "retry_specs", "ext5_dsretry_keys.jsonl"))
    ap.add_argument("--notes-dir", default=os.path.join(ROOT, "data", "notes"))
    a = ap.parse_args(argv)
    latest = H.latest_by_key(H.read_records(a.raw))
    sel = sorted((r for r in latest.values() if r["model"] == a.model and truncated(r)), key=H.key_of)
    bad = []
    for r in sel:
        case = json.load(open(os.path.join(ROOT, "data", "cases", r["case_id"] + ".json")))
        note = open(H.note_path(a.notes_dir, r["case_id"], r["condition"])).read()
        if H.build_prompt(case, note, r.get("prompt_version", "v1")) != r["prompt"]:
            bad.append(H.key_of(r))
    if bad:
        print("FAIL: rebuilt prompt differs from the recorded prompt for: %s" % bad)
        sys.exit(1)
    others = [r for r in latest.values() if r["model"] != a.model and truncated(r)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in sel:
            f.write(json.dumps({"case_id": r["case_id"], "condition": r["condition"], "model": r["model"],
                                "seed": r["seed"], "orig_run_id": r["run_id"], "orig_status": r["parse_status"],
                                "orig_completion_tokens": r["completion_tokens"],
                                "orig_max_tokens": r["decoding_params"]["max_tokens"]}) + "\n")
    by_cond = {}
    for r in sel:
        by_cond[r["condition"]] = by_cond.get(r["condition"], 0) + 1
    print("%d truncated %s replies in %s (by condition %s); prompts verified identical; spec: %s"
          % (len(sel), a.model, os.path.relpath(a.raw, ROOT), dict(sorted(by_cond.items())),
             os.path.relpath(a.out, ROOT)))
    print("truncated replies from other models: %d" % len(others))
    return sel


if __name__ == "__main__":
    main()
