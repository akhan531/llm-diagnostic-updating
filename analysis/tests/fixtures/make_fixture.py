"""Generate analysis/tests/fixtures/synthetic.jsonl. SYNTHETIC DATA ONLY, never real model output.

Every record has "synthetic": true; the analysis scripts refuse to write such inputs into
results/derived/ or paper/generated/. dry_run is false only so the records are scored in the
end-to-end test. Run once: python analysis/tests/fixtures/make_fixture.py
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common import ROOT, load_cases, logit_clipped  # noqa: E402

# model -> (a, b, c, noise sd in log-odds, parse-fail prob, boundary prob)
MODELS = {"SYNTH-calibrated": (0.95, 0.95, 0.0, 0.3, 0.0, 0.0),
          "SYNTH-conservative": (0.6, 0.5, 0.2, 0.5, 0.03, 0.02),
          "SYNTH-noisy": (0.8, 0.7, -0.1, 0.9, 0.05, 0.0)}
# extra noise by condition (notes are harder than vignettes by construction of the fixture)
COND_NOISE = {"V": 0.0, "N": 0.3, "B": 0.6, "S": 0.4, "E": 0.3}


def main():
    rng = random.Random(424242)
    cases = load_cases(os.path.join(ROOT, "data", "cases"))
    recs = []
    for m, (a, b, c, sd, pf, pb) in MODELS.items():
        for cid, case in sorted(cases.items()):
            for cond in ("V", "N", "B", "S", "E"):
                if case["scenario"].startswith("A5") and cond != "V":
                    continue
                if cond == "E" and case["set"] == "A" and case["scenario"].startswith("A5"):
                    continue
                for seed in (1, 2, 3):
                    t = case["target"]
                    rec = {"run_id": "SYNTHETIC", "case_id": cid, "condition": cond, "model": m,
                           "seed": seed, "dry_run": False, "synthetic": True,
                           "parse_status": "ok", "parsed_answer": None}
                    if rng.random() < pf:
                        rec["parse_status"] = "unparseable_missing_field"
                        recs.append(rec)
                        continue
                    pre = min(max(t["pretest_point"] * math.exp(rng.gauss(0, 0.3)), 1e-4), 0.99)
                    lp = logit_clipped(t["pretest_point"])
                    y = a * lp + b * math.log(t["lr_point"]) + c + rng.gauss(0, sd + COND_NOISE[cond])
                    post = 100 / (1 + math.exp(-y))
                    if rng.random() < pb:
                        post = 0.0 if y < 0 else 100.0
                    rec["parsed_answer"] = {"pretest": round(pre * 100, 4), "posttest": round(post, 4)}
                    recs.append(rec)
    with open(os.path.join(HERE, "synthetic.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} synthetic records")


if __name__ == "__main__":
    main()
