"""Generate analysis/tests/fixtures/ext_synthetic/SYNTH_EXT{1..5}.jsonl for tests of analysis/ext_analysis.py.

SYNTHETIC DATA ONLY, never real model output. Every record has "synthetic": true and lives under a path with a
`fixtures` component, so the analysis scripts refuse to write anything derived from it into results/derived or
paper/generated. Run once: python analysis/tests/fixtures/make_ext_fixture.py

Designed truths (asserted in tests/test_ext_analysis.py):
- E1: SYNTH-calibrated's blind pretest equals the published prior (rounded to 1e-4 pp).
- E2: SYNTH-calibrated believes the published LRs and updates exactly on them (believed log DOR excess 0, b_own 1);
  SYNTH-conservative believes LR+ x 2 and LR- / 2 (log DOR excess 2 ln 2) and updates exactly on its belief
  (b_own 1): a pure knowledge error. SYNTH-noisy believes the published LRs but updates with weight 1.6 on them.
- E3/E4: SYNTH-noisy follows the OLDER negative result on positive L cases (posttest below its pretest).
- E5: every main SYNTHETIC key that did not parse is retried; one retry stays empty at max_tokens.
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from common import ROOT, load_cases  # noqa: E402

OUT = os.path.join(HERE, "ext_synthetic")
MODELS = ["SYNTH-calibrated", "SYNTH-conservative", "SYNTH-noisy"]


def logit(p):
    return math.log(p / (1 - p))


def expit(x):
    return 1 / (1 + math.exp(-x))


def rec(run_id, cid, cond, m, seed, pa, status="ok", **kw):
    r = {"run_id": run_id, "case_id": cid, "condition": cond, "model": m, "seed": seed, "dry_run": False,
         "synthetic": True, "parse_status": status, "parsed_answer": pa}
    r.update(kw)
    return r


def write(name, recs):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main():
    rng = random.Random(515151)
    cases = load_cases(os.path.join(ROOT, "data", "cases"))
    real = {cid: c for cid, c in cases.items() if not c.get("exclude_from_fit")}
    scen = {}
    for c in real.values():
        scen.setdefault(c["scenario"], {})[c["result"]] = c
    # E1: blind pretest, case id = scenario
    e1 = []
    for s, d in sorted(scen.items()):
        prior = d["pos"]["target"]["pretest_point"]
        for cond in ("V", "N", "S"):
            for m in MODELS:
                for seed in (1, 2, 3):
                    if m == "SYNTH-calibrated":
                        p = prior
                    else:
                        p = expit(logit(prior) + rng.gauss(-0.2, 0.4))
                    if m == "SYNTH-noisy" and seed == 3 and cond == "S" and s.startswith("A1"):
                        e1.append(rec("SYNTH_EXT1", s, cond, m, seed, None, "unparseable_missing_field"))
                        continue
                    e1.append(rec("SYNTH_EXT1", s, cond, m, seed, {"pretest": round(100 * p, 4)}))
    write("SYNTH_EXT1", e1)
    # E2: believed sensitivity/specificity on N
    e2 = []
    for s, d in sorted(scen.items()):
        lp, ln = d["pos"]["target"]["lr_point"], d["neg"]["target"]["lr_point"]
        for res in ("pos", "neg"):
            c = d[res]
            prior = c["target"]["pretest_point"]
            for m in MODELS:
                k = 2.0 if m == "SYNTH-conservative" else 1.0
                bp, bn = lp * k, ln / k
                spec = (bp - 1) / (bp - bn)
                sens = bp * (1 - spec)
                sens_pct, spec_pct = round(100 * sens, 6), round(100 * spec, 6)
                se, sp = sens_pct / 100, spec_pct / 100
                blr = se / (1 - sp) if res == "pos" else (1 - se) / sp
                for seed in (1, 2, 3):
                    pre = round(100 * expit(logit(prior) + rng.gauss(0, 0.3)), 6)
                    w = 1.6 if m == "SYNTH-noisy" else 1.0
                    lr_used = c["target"]["lr_point"] if m == "SYNTH-noisy" else blr
                    post = round(100 * expit(logit(pre / 100) + w * math.log(lr_used)), 6)
                    e2.append(rec("SYNTH_EXT2", c["case_id"], "N", m, seed,
                                  {"sensitivity": sens_pct, "specificity": spec_pct, "pretest": pre, "posttest": post}))
    write("SYNTH_EXT2", e2)

    # E3 and E4: v1-style answers on L (and N for E4)
    def answers(run_id, conds, noise, follow_old):
        out = []
        for cid, c in sorted(real.items()):
            t = c["target"]
            for cond in conds:
                for m in MODELS:
                    for seed in (1, 2, 3):
                        pre = expit(logit(t["pretest_point"]) + rng.gauss(0, 0.3))
                        lr = t["lr_point"]
                        if follow_old and m == "SYNTH-noisy" and c["result"] == "pos" and cond == "L":
                            lr = scen[c["scenario"]]["neg"]["target"]["lr_point"]
                        post = expit(logit(pre) + 1.2 * math.log(lr) + rng.gauss(0, noise))
                        out.append(rec(run_id, cid, cond, m, seed,
                                       {"pretest": round(100 * pre, 4), "posttest": round(100 * post, 4)},
                                       decoding_params={"temperature": 1.0, "max_tokens": 8000},
                                       completion_tokens=500))
        return out
    write("SYNTH_EXT3", answers("SYNTH_EXT3", ["L"], 0.6, True))
    write("SYNTH_EXT4", answers("SYNTH_EXT4", ["N", "L"], 0.7, True))
    # E5: retry every main SYNTHETIC key that did not parse
    main = [json.loads(line) for line in open(os.path.join(HERE, "synthetic.jsonl"))]
    e5, first = [], True
    for r in main:
        if r["parse_status"] == "ok":
            continue
        t = cases[r["case_id"]]["target"]
        kw = {"decoding_params": {"temperature": 1.0, "max_tokens": 16000}, "retry_of_run_id": "SYNTHETIC"}
        if first:
            e5.append(rec("SYNTH_EXT5", r["case_id"], r["condition"], r["model"], r["seed"], None,
                          "unparseable_missing_field", completion_tokens=16000, response="", **kw))
            first = False
            continue
        pre = expit(logit(t["pretest_point"]) + rng.gauss(0, 0.3))
        post = expit(logit(pre) + 1.5 * math.log(t["lr_point"]))
        e5.append(rec("SYNTH_EXT5", r["case_id"], r["condition"], r["model"], r["seed"],
                      {"pretest": round(100 * pre, 4), "posttest": round(100 * post, 4)}, completion_tokens=3000, **kw))
    write("SYNTH_EXT5", e5)
    print(f"wrote {len(e1)}, {len(e2)}, E3, E4, {len(e5)} records to {OUT}")


if __name__ == "__main__":
    main()
