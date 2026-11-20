#!/usr/bin/env python3
"""Estimate the cost of a batch before running it.

Usage:
  python scripts/estimate_cost.py --model openai/gpt-x --calls 600 --in-tokens 900 --out-tokens 400
Prices are fetched live from the OpenRouter models endpoint (USD per token).
Reasoning models bill hidden reasoning tokens as output: set --out-tokens to include them
(measure from a pilot, do not guess).
"""
import argparse, json, urllib.request


def price(model):
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        models = json.load(r)["data"]
    for m in models:
        if m["id"] == model:
            p = m["pricing"]
            return float(p["prompt"]), float(p["completion"])
    raise SystemExit(f"model {model} not found on OpenRouter")


def plan_cost(models, calls_per_model, in_tokens, out_tokens_by_model):
    """models: {slug: (price_in_per_token, price_out_per_token)}. Returns ({slug: usd}, total usd).
    out_tokens_by_model must include hidden reasoning tokens for reasoning models."""
    per = {m: calls_per_model * (in_tokens * pi + out_tokens_by_model[m] * po) for m, (pi, po) in models.items()}
    return per, sum(per.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calls", type=int, required=True)
    ap.add_argument("--in-tokens", type=int, required=True)
    ap.add_argument("--out-tokens", type=int, required=True)
    a = ap.parse_args()
    if min(a.calls, a.in_tokens, a.out_tokens) <= 0:
        raise SystemExit("--calls, --in-tokens and --out-tokens must be positive")
    pin, pout = price(a.model)
    cost = a.calls * (a.in_tokens * pin + a.out_tokens * pout)
    print(f"{a.model}: {a.calls} calls x ({a.in_tokens} in, {a.out_tokens} out) = ${cost:.2f}")


if __name__ == "__main__":
    main()
