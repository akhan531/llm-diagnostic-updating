"""Shared helpers for the analysis scripts. Stdlib only (numpy/pandas/scipy are not installed)."""
import csv
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Probabilities are clipped to [CLIP_LO, CLIP_HI] before any logit. 1e-5 (not 1e-3) because some
# exact targets are below 0.1% (e.g. 0.027%); a 1e-3 clip biases an exact-Bayes fit to a=0.905, b=0.940.
CLIP_LO, CLIP_HI = 1e-5, 1 - 1e-5


def clip(p):
    return min(max(p, CLIP_LO), CLIP_HI)


def is_boundary(p):
    """True if p is outside the open clip range, i.e. a literal 0/1 (or beyond) that gets clamped."""
    return p < CLIP_LO or p > CLIP_HI


def check_unclipped(p, what="input"):
    """Assert a prior/target is strictly inside the clip range so clipping cannot bias it."""
    assert CLIP_LO < p < CLIP_HI, f"{what}={p} would be clipped (range {CLIP_LO}..{CLIP_HI})"


def logit_clipped(p):
    p = clip(p)
    return math.log(p / (1.0 - p))


# JSON files in data/cases/ that are not case files (explicit list, so a malformed case file still fails loudly).
# E_removed_classification.json: F5 classification of removed E negatives (data/notes/classify_e.py).
NON_CASE_FILES = {"E_removed_classification.json"}


def load_cases(cases_dir):
    """Return {case_id: case dict}. Fails loudly if a required field is missing."""
    cases = {}
    for path in sorted(glob.glob(os.path.join(cases_dir, "*.json"))):
        if os.path.basename(path) in NON_CASE_FILES:  # metadata beside the cases, named explicitly
            continue
        with open(path) as f:
            c = json.load(f)
        for k in ("case_id", "scenario", "set", "result", "target"):
            assert k in c, f"{path}: missing field {k}"
        t = c["target"]
        for k in ("pretest_point", "pretest_range", "lr_point", "lr_range", "posttest_point",
                  "posttest_band_primary", "posttest_band_bayes"):
            assert k in t, f"{path}: missing target.{k}"
        assert c["case_id"] not in cases, f"duplicate case_id {c['case_id']}"
        cases[c["case_id"]] = c
    assert cases, f"no case files in {cases_dir}"
    return cases


def load_raw(raw_glob, report=None):
    """Return (kept records, n_dry_run_excluded). Dry-run records are dropped and counted.

    A line that is not valid JSON (e.g. a partially written last line of a run in progress) is
    skipped with a warning, never fixed in the file. If `report` is a dict it gets
    report["n_malformed"] (count of skipped lines)."""
    kept, n_dry, n_bad = [], 0, 0
    files = sorted(glob.glob(raw_glob))
    assert files, f"no raw files match {raw_glob}"
    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    assert isinstance(r, dict)
                except (ValueError, AssertionError):
                    n_bad += 1
                    print(f"WARNING: skipping malformed raw line in {path} (not valid JSON, "
                          f"{len(line)} chars); file left untouched.")
                    continue
                if r.get("dry_run"):
                    n_dry += 1
                    continue
                kept.append(r)
    if report is not None:
        report["n_malformed"] = n_bad
    return kept, n_dry


CONFIG_PATH = os.path.join(ROOT, "analysis", "config.json")


def load_config(path=None):
    with open(path or CONFIG_PATH) as f:
        cfg = json.load(f)
    for k in ("paper_run_ids", "pilot_run_ids", "excluded_run_ids"):
        assert isinstance(cfg.get(k), list), f"config missing list {k}"
    assert not (set(cfg["paper_run_ids"]) & set(cfg["pilot_run_ids"])), "paper and pilot run ids overlap"
    assert not (set(cfg["paper_run_ids"] + cfg["pilot_run_ids"]) & set(cfg["excluded_run_ids"])), \
        "excluded run id also allow-listed"
    ext = set(cfg.get("ext_run_ids", []))
    assert not (ext & set(cfg["paper_run_ids"] + cfg["pilot_run_ids"] + cfg["excluded_run_ids"])), \
        "ext run id also in paper/pilot/excluded lists"
    return cfg


def add_run_args(ap):
    ap.add_argument("--set", dest="run_set", choices=["paper", "pilot"], default="paper",
                    help="allow-list from analysis/config.json: paper (default) or pilot")
    ap.add_argument("--run-ids", default=None,
                    help="comma-separated run ids; overrides --set (used by tests on fixtures)")
    ap.add_argument("--config", default=CONFIG_PATH)


def resolve_run_ids(args):
    """Return (allowed run ids, set label, config)."""
    cfg = load_config(args.config)
    if args.run_ids:
        ids = [x for x in args.run_ids.split(",") if x]
        pil = [x for x in ids if x in cfg["pilot_run_ids"]]
        if pil:
            # N4: any pilot id makes the selection pilot-scored (pilot output dir, paper guard applies);
            # mixing pilot and paper ids would double count, so it is refused.
            assert not any(x in cfg["paper_run_ids"] for x in ids), \
                f"REFUSING to mix pilot run ids {pil} with paper run ids in --run-ids {ids}"
            return ids, "pilot", cfg
        return ids, "explicit", cfg
    return list(cfg[f"{args.run_set}_run_ids"]), args.run_set, cfg


def default_out(args_out, run_set):
    if args_out:
        return args_out
    d = os.path.join(ROOT, "results", "derived")
    return os.path.join(d, "pilot") if run_set == "pilot" else d


def select_records(recs, allowed, cfg, quiet=False):
    """Allow-list filter then latest-attempt dedup.

    Returns (final records, report dict). Records whose run_id is neither allowed, nor listed in the
    config (paper/pilot/excluded) are UNKNOWN: counted, printed loudly, never scored. Records of known
    but not selected run ids are counted as not selected. Dedup key is (run_id, case_id, condition,
    model, seed); the record with the highest `attempt` (missing = 1; ties go to the later line) is
    kept. Raw records are never modified.
    """
    known = set(cfg["paper_run_ids"]) | set(cfg["pilot_run_ids"]) | set(cfg["excluded_run_ids"]) | \
        set(cfg.get("ext_run_ids", []))  # R1 extension runs: known, read only by ext_analysis.py
    allowed = set(allowed)
    unknown, notsel, sel = {}, {}, []
    for r in recs:
        rid = r.get("run_id")
        if rid in allowed:
            sel.append(r)
        elif rid in known:
            notsel[rid] = notsel.get(rid, 0) + 1
        else:
            unknown[rid] = unknown.get(rid, 0) + 1
    if unknown and not quiet:
        print(f"WARNING: {sum(unknown.values())} records with UNKNOWN run ids {sorted(unknown, key=str)} "
              "found in raw; NOT scored. Add them to analysis/config.json if intended.")
    latest, order = {}, []
    for r in sel:
        k = (r.get("run_id"), r["case_id"], r["condition"], r["model"], r.get("seed"))
        if k not in latest:
            order.append(k)
            latest[k] = r
        elif r.get("attempt", 1) >= latest[k].get("attempt", 1):
            latest[k] = r
    final = [latest[k] for k in order]
    att = {}
    for r in sel:
        a = att.setdefault((r.get("run_id"), r["model"]),
                           {"run_id": r.get("run_id"), "model": r["model"], "n_records": 0,
                            "n_keys": 0, "n_superseded": 0, "n_final_call_error": 0,
                            "max_attempt": 1})
        a["n_records"] += 1
        a["max_attempt"] = max(a["max_attempt"], r.get("attempt", 1))
    for r in final:
        a = att[(r.get("run_id"), r["model"])]
        a["n_keys"] += 1
        a["n_final_call_error"] += int(r.get("parse_status") == "call_error")
    for a in att.values():
        a["n_superseded"] = a["n_records"] - a["n_keys"]
    rep = {"unknown": unknown, "not_selected": notsel, "n_selected_records": len(sel),
           "n_final": len(final), "n_superseded": len(sel) - len(final),
           "attempts": sorted(att.values(), key=lambda a: (str(a["run_id"]), a["model"]))}
    assert rep["n_final"] + rep["n_superseded"] == rep["n_selected_records"]
    return final, rep


ATTEMPT_FIELDS = ["run_id", "model", "n_records", "n_keys", "n_superseded", "n_final_call_error",
                  "max_attempt"]


def guard_pilot(out_dir, run_set):
    """Pilot-scored data may never feed the paper directories."""
    if run_set == "pilot":
        for d in (os.path.join(ROOT, "paper", "generated"), os.path.join(ROOT, "paper", "figures")):
            p, dd = os.path.realpath(out_dir), os.path.realpath(d)
            assert not (p == dd or p.startswith(dd + os.sep)), \
                f"REFUSING to write pilot-scored output into {out_dir}"


def repo_relative(path):
    """Repo-relative form of a path (L1), so provenance does not depend on the checkout location."""
    ap = os.path.abspath(path)
    if ap == ROOT or ap.startswith(ROOT + os.sep):
        return os.path.relpath(ap, ROOT)
    return path


def display_name(slug, config=None):
    """Short display name for a model slug, from the single mapping in analysis/config.json."""
    names = load_config(CONFIG_PATH if config is None else config).get("display_names", {})
    return names.get(slug, slug)


def is_calibration(case):
    """A5-type cases: numeric calibration check, condition V only; never in cross-condition results."""
    return bool(case.get("exclude_from_fit"))


def write_provenance(out_dir, name, raw_glob, synthetic, n_records, run_set=None, run_ids=None,
                     n_malformed=0):
    os.makedirs(out_dir, exist_ok=True)
    raw_glob = repo_relative(raw_glob)
    with open(os.path.join(out_dir, f"provenance_{name}.json"), "w") as f:
        json.dump({"raw_glob": raw_glob, "synthetic": bool(synthetic), "n_records": n_records,
                   "run_set": run_set, "run_ids": run_ids,
                   "n_malformed_raw_lines": n_malformed}, f)


def read_run_set(derived_dir):
    """Run set recorded by score.py/fit_wbayes.py ('paper', 'pilot' or 'explicit'); must agree."""
    sets = set()
    for n in ("score", "wbayes"):
        with open(os.path.join(derived_dir, f"provenance_{n}.json")) as f:
            sets.add(json.load(f).get("run_set"))
    assert len(sets) == 1, f"score and wbayes provenance disagree on run set: {sets}"
    return sets.pop()


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---- Synthetic-data firewall -------------------------------------------------------------
# Synthetic fixtures must never reach the paper. Records carry "synthetic": true, and anything read
# from a path containing "fixtures" counts as synthetic. score.py and fit_wbayes.py write
# provenance_*.json next to their outputs; make_tables.py and make_figures.py read them. Every writer
# calls guard_output() first, which refuses to write into a real output directory if any input is
# synthetic (or if provenance is missing).
REAL_OUT_DIRS = [os.path.join(ROOT, "results", "derived"), os.path.join(ROOT, "paper", "generated"),
                 os.path.join(ROOT, "paper", "figures")]


def is_real_out_dir(path):
    p = os.path.realpath(path)
    return any(p == d or p.startswith(d + os.sep) for d in map(os.path.realpath, REAL_OUT_DIRS))


def is_synthetic(recs, raw_glob):
    path_flag = "fixtures" in raw_glob.replace("\\", "/").split("/") or \
        any("fixtures" in p.replace("\\", "/").split("/") for p in glob.glob(raw_glob))
    return path_flag or any(r.get("synthetic") for r in recs)


def guard_output(out_dir, synthetic):
    """Refuse (AssertionError) to write synthetic-derived outputs into a real output directory."""
    if is_real_out_dir(out_dir):
        assert not synthetic, (f"REFUSING to write synthetic-derived output into {out_dir}: "
                               "inputs came from fixture/synthetic records")


def read_provenance(derived_dir, names=("score", "wbayes")):
    """True if any upstream input was synthetic. Missing provenance is treated as an error."""
    syn = False
    for n in names:
        p = os.path.join(derived_dir, f"provenance_{n}.json")
        assert os.path.exists(p), f"missing {p}: rerun analysis/score.py and fit_wbayes.py"
        with open(p) as f:
            syn = syn or bool(json.load(f)["synthetic"])
    return syn
