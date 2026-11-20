#!/usr/bin/env python3
"""Extension notes for the committee fixes (P1 pretest-only, P3 long chart). Stdlib only, deterministic.

Usage:
  python data/notes/render_ext.py            # writes data/notes/pretest/ and data/notes/long/
  python data/notes/check_ext.py             # leak and fact checks (exit 1 on failure)

P1 (pretest-only): for each of the 10 scenarios with a facts file (A1-A4, B1-B6; A5 is the numeric
calibration item and is excluded) and conditions V, N, B, S, the rendered pos and neg notes from render.py are
redacted INDEPENDENTLY by removing the sentence that carries the result wording (from the previous ". " or the
start of the line to the next ". " or the end of the line; a line left empty is dropped). check_ext.py asserts
that the two redacted versions are byte-identical (so nothing that differs by result survives) and that no
result wording of either polarity in any condition remains. Output: pretest/<scenario>__<cond>.txt.
The harness reads them with --collapse-polarity (case id = scenario).

P3 (long chart, condition L): the N note of each of the 20 cases is embedded unchanged as the
"current encounter" in a synthetic chart built from fixed blocks: six older visit notes, an unrelated lab panel,
a problem and medication list, and an older result of the same test (negative-type finding, so the opposite
sign for every positive case) dated more than a year earlier.
Every block carries an explicit date; the current encounter is dated INDEX_DATE and is placed mid-document.
Output: long/<case_id>__L.txt. Distractor blocks are text in data/notes/long_spec.py, written to be
probability-neutral (no leak words, no new findings about the target diagnosis beyond the older opposite-sign
result, which is the manipulation).
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402

PRE_DIR = os.path.join(HERE, "pretest")
LONG_DIR = os.path.join(HERE, "long")
PRE_CONDS = ["V", "N", "B", "S"]


def scenarios():
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "facts", "*.json"))):
        out.append(render.load_facts(p))
    return out


def rendered(facts, table, result, pol, cond):
    return render.render_one(facts, table, result, pol, cond)


def remove_result_sentence(text, surface):
    """Remove the sentence holding `surface` (case-sensitive first match). Returns (new_text, removed_span)."""
    idx = text.find(surface)
    if idx < 0:
        raise ValueError("result surface not found: %r" % surface)
    ls = text.rfind("\n", 0, idx) + 1
    le = text.find("\n", idx)
    le = len(text) if le < 0 else le
    line = text[ls:le]
    i = idx - ls
    b = line.rfind(". ", 0, i)
    start = 0 if b < 0 else b + 2
    m = re.compile(r"\.(\s|$)").search(line, i + len(surface) - 1 if surface.endswith(".") else i + len(surface))
    end = len(line) if m is None else m.start() + 1
    removed = line[start:end]
    new_line = (line[:start] + line[end:]).rstrip()
    new_line = re.sub(r"  +", " ", new_line)
    if new_line.strip() == "":
        new_text = text[:ls] + text[le + 1:] if le < len(text) else text[:ls]
    else:
        new_text = text[:ls] + new_line + text[le:]
    new_text = re.sub(r"\n{3,}", "\n\n", new_text).rstrip("\n") + "\n"
    return new_text, removed


def redact(facts, table, result, pol, cond):
    text = rendered(facts, table, result, pol, cond)
    return remove_result_sentence(text, result[pol][cond][0])


def write_pretest():
    os.makedirs(PRE_DIR, exist_ok=True)
    written = []
    for facts, table, result in scenarios():
        sc = facts["scenario"]
        for cond in PRE_CONDS:
            t_pos, _ = redact(facts, table, result, "pos", cond)
            t_neg, _ = redact(facts, table, result, "neg", cond)
            if t_pos != t_neg:   # fatal here too; check_ext.py repeats it
                raise SystemExit("%s/%s: redacted pos and neg differ" % (sc, cond))
            fn = os.path.join(PRE_DIR, "%s__%s.txt" % (sc, cond))
            with open(fn, "w", encoding="utf-8") as f:
                f.write(t_pos)
            written.append(fn)
    return written


# ---------------------------------------------------------------- P3 long chart
INDEX_DATE = "2026-03-14"
def _first(table, fid, cond="N"):
    return (table.get(fid, {}).get(cond) or [None])[0]


def _days(a, b):
    import datetime
    return (datetime.date.fromisoformat(b) - datetime.date.fromisoformat(a)).days


def long_note(facts, table, result, pol):
    """Chart extract for condition L. Block order (export order, not date order): header, problem and medication
    list, visit 0, OLDER RESULT, visit 1, extra visit 0, CURRENT ENCOUNTER (the N note, unchanged), unrelated
    labs, visits 2-3, extra visit 1, footer. All text other than the N note comes from long_spec.py."""
    from long_spec import LONG_SPEC, COMMON, VISITS, LABS, EXTRA_VISITS
    sc = facts["scenario"]
    spec = LONG_SPEC[sc]
    n_note = rendered(facts, table, result, pol, "N")
    body = n_note.split("\n", 1)[1].strip("\n")  # drop the per-note header; the chart has one header
    age_sex = _first(table, "age_sex")
    age, sex = int(age_sex.split("-")[0]), age_sex.split()[-1]
    pron = {"woman": ("She", "her"), "man": ("He", "his")}[sex]

    def fill(text, date):
        then = age - _days(date, INDEX_DATE) // 365
        return text.format(Pronoun=pron[0], possessive=pron[1], Age_sex="%d-year-old %s" % (then, sex))

    pmh = _first(table, "f_pmh") or _first(table, "htn") or "no chronic problems on file"
    meds = _first(table, "f_med") or "none"
    allergy = _first(table, "f_allergy") or "no known drug allergies"
    old = spec["old_result"]
    parts = [render_header(sc)]
    parts.append(block("PROBLEM LIST AND MEDICATIONS (reconciled %s)" % VISITS[0][0],
                       "Problems: %s.\nMedications: %s.\nAllergies: %s." % (cap(pmh), cap(meds), cap(allergy))))
    v = VISITS[0]
    parts.append(block("%s | %s" % (v[0], v[1]), fill(v[2], v[0])))
    parts.append(block("%s | %s" % (old["date"], old["label"]), "%s: %s.\n%s" % (old["test"], old["text"], old["comment"])))
    v = VISITS[1]
    parts.append(block("%s | %s" % (v[0], v[1]), fill(v[2], v[0])))
    v = EXTRA_VISITS[0]
    parts.append(block("%s | %s" % (v[0], v[1]), fill(v[2], v[0])))
    parts.append(block("%s | CURRENT ENCOUNTER" % INDEX_DATE, body))
    parts.append(block("%s | LABORATORY RESULTS" % LABS[0], LABS[1]))
    for v in VISITS[2:] + EXTRA_VISITS[1:]:
        parts.append(block("%s | %s" % (v[0], v[1]), fill(v[2], v[0])))
    parts.append(block("CHART FOOTER", COMMON["footer"]))
    return "\n\n".join(parts).rstrip("\n") + "\n"


def cap(t):
    return t[0].upper() + t[1:]


def render_header(sc):
    return ("SYNTHETIC NOTE FOR RESEARCH. NOT A REAL PATIENT.\n\n"
            "SYNTHETIC ELECTRONIC HEALTH RECORD EXTRACT. Entries are in the order exported, not by date. "
            "Each entry is dated. Today is %s." % INDEX_DATE)


def block(title, text):
    return "===== %s =====\n%s" % (title, text.strip("\n"))


def write_long():
    os.makedirs(LONG_DIR, exist_ok=True)
    written = []
    for facts, table, result in scenarios():
        for pol in render.POLS:
            fn = os.path.join(LONG_DIR, "%s_%s__L.txt" % (facts["scenario"], pol))
            with open(fn, "w", encoding="utf-8") as f:
                f.write(long_note(facts, table, result, pol))
            written.append(fn)
    return written


def main(argv):
    what = argv or ["pretest", "long"]
    n = 0
    if "pretest" in what:
        n += len(write_pretest())
    if "long" in what:
        n += len(write_long())
    print("wrote %d extension notes" % n)


if __name__ == "__main__":
    main(sys.argv[1:])
