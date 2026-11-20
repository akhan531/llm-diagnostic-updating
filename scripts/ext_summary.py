#!/usr/bin/env python3
"""Per-model summary of one raw run file (read-only): records, parse status, truncation, tokens, cost.
Usage: python scripts/ext_summary.py --raw results/raw/<run_id>.jsonl"""
import argparse, collections, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import run as H  # noqa: E402


def truncated(r):
    if r.get("finish_reason") is not None:
        return r["finish_reason"] == "length"
    mt, ct = (r.get("decoding_params") or {}).get("max_tokens"), r.get("completion_tokens")
    return r.get("parse_status") != "call_error" and mt is not None and ct is not None and ct >= mt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    a = ap.parse_args(argv)
    recs = list(H.latest_by_key(H.read_records(a.raw)).values())
    if not recs:
        print(f"{a.raw}: no records"); return
    by = collections.defaultdict(list)
    for r in recs:
        by[r["model"]].append(r)
    print(f"{os.path.basename(a.raw)}: {len(recs)} keys (latest attempt), prompt versions "
          f"{sorted({r.get('prompt_version') for r in recs})}, dry_run {any(r.get('dry_run') for r in recs)}")
    tot = 0.0
    for m, rs in sorted(by.items()):
        st = collections.Counter(r["parse_status"] for r in rs)
        tr = sum(truncated(r) for r in rs)
        cts = [r["completion_tokens"] for r in rs if r.get("completion_tokens") is not None]
        pts = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens") is not None]
        cost = sum(r.get("cost_guard_usd") or 0.0 for r in rs)
        tot += cost
        print(f"  {m:30s} n={len(rs):4d} ok={st.get('ok', 0):4d} other={dict((k, v) for k, v in st.items() if k != 'ok')} "
              f"truncated={tr} in~{(sum(pts) / len(pts)) if pts else 0:.0f} out~{(sum(cts) / len(cts)) if cts else 0:.0f} "
              f"cost=${cost:.4f}")
    print(f"  total guard cost ${tot:.4f}")


if __name__ == "__main__":
    main()
