#!/usr/bin/env python3
"""Cost estimate for the committee-extension batches (ext1 to ext5), with scripts/estimate_cost.py logic.

Usage: python scripts/estimate_ext.py [--offline] [--markdown]

For every batch and model: calls x (input tokens x input price + output tokens x output price), via
estimate_cost.plan_cost. Inputs:
- Input tokens: the real prompts the batch will send (prompt file + note, chars/4, mean over the batch's notes),
  scaled per model by that model's measured tokenizer ratio in main1 (reported prompt_tokens / (chars/4));
  Claude's tokenizer counts about 1.7x chars/4 on these prompts.
- Output tokens: per-model mean completion tokens over every returned main1 reply (planning figure); the stress
  figure uses the per-model 95th percentile and 1.5x the mean for the long chart (longer input, plausibly
  longer reasoning). ext5 output uses the 16,000-token ceiling for every call (worst case).
- Prices: live from the OpenRouter models endpoint (estimate_cost.price); with --offline, or if the endpoint is
  unreachable, the list prices recorded in results/c2_cost_estimate.md on 2026-11-11 (stated in the output).
Spent so far is the latest ledger value (results/spend_ledger.csv). The planning cap is $40.
"""
import argparse, glob, json, os, statistics, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import estimate_cost  # noqa: E402
import run as H  # noqa: E402

MODELS = ["openai/gpt-6.1-sol", "anthropic/claude-sonnet-5.5", "google/gemini-3.8-flash",
          "deepseek/deepseek-v4-pro", "openai/gpt-4o"]
C2_PRICES = {  # USD per token, results/c2_cost_estimate.md (2026-11-11)
    "openai/gpt-6.1-sol": (2.00e-6, 10.00e-6), "anthropic/claude-sonnet-5.5": (2.00e-6, 10.00e-6),
    "google/gemini-3.8-flash": (0.825e-6, 4.125e-6), "deepseek/deepseek-v4-pro": (0.955e-6, 1.911e-6),
    "openai/gpt-4o": (2.50e-6, 10.00e-6)}
SEEDS = 3
EXTRA_USD = 9.0
CAP = H.PLANNING_CAP_USD


def cases(non_a5=True):
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "cases", "*.json")))
    return [f for f in fs if not (non_a5 and "A5_" in os.path.basename(f))]


def prompt_chars(version, notes):
    case = {"diagnosis": "x" * 25, "test_name": "y" * 25}
    return statistics.mean(len(H.build_prompt(case, open(n).read(), version)) for n in notes)


def batch_defs():
    nd = os.path.join(ROOT, "data", "notes")
    scen = sorted({os.path.basename(f)[:-5].rsplit("_", 1)[0] for f in cases()})
    ids = [os.path.basename(f)[:-5] for f in cases()]
    long_notes = [os.path.join(nd, "long", f"{c}__L.txt") for c in ids]
    have_long = all(os.path.exists(p) for p in long_notes)
    pre = [os.path.join(nd, "pretest", f"{s}__{c}.txt") for s in scen for c in ("V", "N", "S")]
    n_notes = [os.path.join(nd, f"{c}__N.txt") for c in ids]
    d = {
        "ext1_pretest": dict(desc="pretest-only, redacted notes, V N S x 10 scenarios, prompt v2",
                             parts=[("v2_pretest_only", pre, len(pre))]),
        "ext2_belief": dict(desc="believed sensitivity/specificity + pretest/posttest, N x 20 cases, prompt v3",
                            parts=[("v3_believed_accuracy", n_notes, len(n_notes))]),
        "ext6_para1": dict(desc="paraphrased v1 prompt (v5_para1), N x 20 cases",
                           parts=[("v5_para1", n_notes, len(n_notes))]),
        "ext6_para2": dict(desc="paraphrased v1 prompt (v6_para2), N x 20 cases",
                           parts=[("v6_para2", n_notes, len(n_notes))]),
        "ext3_long": dict(desc="long chart L x 20 cases, prompt v1 (test named)",
                          parts=[("v1", long_notes if have_long else None, len(ids))], long=True),
        "ext4_unnamed": dict(desc="test not named, prompt v4, N x 20 and L x 20",
                             parts=[("v4_unnamed_test", n_notes, len(ids)),
                                    ("v4_unnamed_test", long_notes if have_long else None, len(ids))], long=True),
    }
    return d, have_long


def main1_stats():
    recs = [r for r in H.latest_by_key(H.read_records(os.path.join(ROOT, "results", "raw", "main1.jsonl"))).values()
            if r.get("parse_status") != "call_error" and r.get("completion_tokens") is not None]
    out = {}
    for m in MODELS:
        rs = [r for r in recs if r["model"] == m]
        cts = sorted(r["completion_tokens"] for r in rs)
        ratio = statistics.mean(r["prompt_tokens"] / (len(r["prompt"]) / 4) for r in rs)
        out[m] = dict(ct_mean=statistics.mean(cts), ct_p95=cts[int(0.95 * (len(cts) - 1))], tok_ratio=ratio,
                      cost_per_call=statistics.mean(r["cost_usd"] for r in rs if r.get("cost_usd") is not None))
    return out


def prices(offline):
    if offline:
        return C2_PRICES, "C2 list prices (results/c2_cost_estimate.md, 2026-11-11); live endpoint not used"
    try:
        return {m: estimate_cost.price(m) for m in MODELS}, "live OpenRouter list prices"
    except BaseException as e:   # price() raises SystemExit or URLError
        return C2_PRICES, f"C2 list prices (2026-11-11); live endpoint unreachable ({type(e).__name__})"


def estimate(offline=False, long_fallback_tokens=3000):
    defs, have_long = batch_defs()
    st = main1_stats()
    pr, price_src = prices(offline)
    rows = []
    for name, d in defs.items():
        plan_total, stress_total, calls_total, per_model = 0.0, 0.0, 0, {m: 0.0 for m in MODELS}
        for version, notes, n_notes in d["parts"]:
            if notes is None:
                chars = long_fallback_tokens * 4 + len(H.load_prompt(version))
            else:
                chars = prompt_chars(version, notes)
            calls = n_notes * SEEDS
            is_long = notes is not None and "long" in (notes[0] if notes else "") or notes is None
            for m in MODELS:
                inp = chars / 4 * st[m]["tok_ratio"]
                per, tot = estimate_cost.plan_cost({m: pr[m]}, calls, inp, {m: st[m]["ct_mean"]})
                per_s, tot_s = estimate_cost.plan_cost(
                    {m: pr[m]}, calls, inp,
                    {m: max(st[m]["ct_p95"], 1.5 * st[m]["ct_mean"] if is_long else st[m]["ct_mean"])})
                plan_total += tot
                stress_total += tot_s
                per_model[m] += tot
            calls_total += calls * len(MODELS)
        rows.append(dict(batch=name, desc=d["desc"], calls=calls_total, calls_per_model=calls_total // len(MODELS),
                         plan=plan_total, stress=stress_total, per_model=per_model))
    # ext5: 21 DeepSeek keys at the 16,000-token ceiling (worst case) and at the main1 mean
    keys = [json.loads(l) for l in open(os.path.join(ROOT, "results", "retry_specs", "ext5_dsretry_keys.jsonl"))]
    m = "deepseek/deepseek-v4-pro"
    recs = {H.key_of(r): r for r in H.read_records(os.path.join(ROOT, "results", "raw", "main1.jsonl"))}
    inp = statistics.mean(recs[(k["case_id"], k["condition"], k["model"], k["seed"])]["prompt_tokens"] for k in keys)
    _, worst = estimate_cost.plan_cost({m: pr[m]}, len(keys), inp, {m: 16000})
    _, plan5 = estimate_cost.plan_cost({m: pr[m]}, len(keys), inp, {m: 8000})
    rows.append(dict(batch="ext5_dsretry", desc="21 truncated DeepSeek main1 replies, max_tokens 16000",
                     calls=len(keys), calls_per_model=len(keys), plan=plan5, stress=worst,
                     per_model={m: plan5}))
    pilots = {r["batch"]: (15 if r["batch"] != "ext5_dsretry" else 0) for r in rows}
    return rows, st, price_src, have_long, pilots


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--json", default=None, help="write the per-batch figures here")
    a = ap.parse_args(argv)
    rows, st, src, have_long, pilots = estimate(a.offline)
    spent = H.ledger_total()
    print(f"Prices: {src}. Long-chart notes present: {have_long}. Ledger spend: ${spent:.4f}.")
    print("Per-model main1 measurements: " + "; ".join(
        f"{m.split('/')[1]} out mean {v['ct_mean']:.0f} p95 {v['ct_p95']} tok-ratio {v['tok_ratio']:.2f}"
        for m, v in st.items()))
    tot_p = tot_s = 0.0
    tot_c = 0
    for r in rows:
        pil = r["plan"] / r["calls"] * pilots[r["batch"]] if r["calls"] else 0.0
        r["pilot_plan"] = pil
        print(f"{r['batch']:14s} calls {r['calls']:4d} (+{pilots[r['batch']]:2d} pilot)  plan ${r['plan']:.2f}  "
              f"stress ${r['stress']:.2f}  pilot ${pil:.2f}  | {r['desc']}")
        print("    " + ", ".join(f"{m.split('/')[1]} ${v:.2f}" for m, v in r["per_model"].items()))
        tot_p += r["plan"] + pil
        tot_s += r["stress"] + pil
        tot_c += r["calls"] + pilots[r["batch"]]
    print(f"TOTAL calls {tot_c}  plan ${tot_p:.2f}  stress ${tot_s:.2f}  "
          f"(budget for extras ${EXTRA_USD:.2f}; spend after plan ${spent + tot_p:.2f}, after stress "
          f"${spent + tot_s:.2f}; cap ${CAP:.2f})")
    if a.json:
        json.dump(dict(rows=rows, price_source=src, spent=spent, total_plan=tot_p, total_stress=tot_s,
                       total_calls=tot_c), open(a.json, "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
