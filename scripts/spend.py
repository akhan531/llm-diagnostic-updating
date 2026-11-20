#!/usr/bin/env python3
"""Report actual OpenRouter spend for the key in OPENROUTER_API_KEY.

Usage: python scripts/spend.py [--log NOTE]
With --log, appends a row to results/spend_ledger.csv.
Exits 1 if spend cannot be read or the ledger cannot be written (fatal for the pipeline).
Exits 2 if spend is at or above the planning cap (default $40, env SPEND_CAP_USD).
"""
import csv, datetime, json, os, sys, urllib.request

CAP = float(os.environ.get("SPEND_CAP_USD", "40"))
LEDGER = os.environ.get("SPEND_LEDGER") or os.path.join(os.path.dirname(__file__), "..", "results", "spend_ledger.csv")


def auth_headers():
    """In cloud sessions the key is a network secret: a network proxy adds the Authorization header
    to requests for openrouter.ai, and the key is never visible here. Locally, set OPENROUTER_API_KEY."""
    key = os.environ.get("OPENROUTER_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def fetch():
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers=auth_headers())
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r).get("data", {})
    return float(data.get("usage") or 0.0), data.get("limit")


def main():
    # Any failure to read spend is fatal (exit 1): callers (run_pipeline.sh) must not continue unmetered.
    # Exit 2 means at or above the planning cap; also fatal for the pipeline.
    try:
        usage, limit = fetch()
    except Exception as e:
        print(f"SPEND CHECK FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OpenRouter spend: ${usage:.4f} | key limit: {limit} | planning cap: ${CAP:.2f}")
    if "--log" in sys.argv:
        note = sys.argv[sys.argv.index("--log") + 1] if len(sys.argv) > sys.argv.index("--log") + 1 else ""
        new = not os.path.exists(LEDGER)
        try:
            with open(LEDGER, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp_utc", "usage_usd", "key_limit", "note"])
                w.writerow([datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                            f"{usage:.4f}", limit, note])
        except OSError as e:
            print(f"LEDGER WRITE FAILED: {e}", file=sys.stderr)
            sys.exit(1)
    if usage >= CAP:
        print("AT OR ABOVE PLANNING CAP: stop paid runs (planning cap reached; flag [REVIEW]).")
        sys.exit(2)


if __name__ == "__main__":
    main()
