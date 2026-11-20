#!/usr/bin/env python3
"""Checks for the extension notes written by render_ext.py. Stdlib only. Exit 1 on any failure.

Pretest-only notes (data/notes/pretest/<scenario>__<cond>.txt), per scenario and condition:
  1. the file equals a fresh render (no hand edits);
  2. the pos and neg notes, redacted independently, are byte-identical (nothing that varies with the result
     survives redaction);
  3. no result wording of either polarity in ANY condition appears (case-insensitive), and no result verb
     or outcome word from OUTCOME_WORDS appears;
  4. every core and filler fact for the condition is still present (redaction removed only the result);
  5. header present, no em or en dash, no leak words (check_notes.LEAK);
  6. the removed span contains the result wording and no other fact surface.
Long-chart notes (data/notes/long/<case_id>__L.txt): see check_long() below.
"""
import glob, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render, render_ext, check_notes  # noqa: E402

HEADER = check_notes.HEADER
# Words that report an outcome of a test. Checked on the redacted notes only. Context words that occur
# legitimately in the unredacted remainder are listed in OUTCOME_ALLOW per scenario (none needed so far).
OUTCOME_WORDS = ["resulted", "findings:", "final:", "culture final", "ucx final", "shows", "showed", "revealed",
                 "filling defect", "airspace opacity", "consolidation", "cfu", "no growth", "e coli",
                 "escherichia", "ischemic st", "recalled", "abnormal", "noncompressible", "dilated appendix",
                 "normal appendix", "sonographic", "ng/ml feu", "fat stranding", "addendum"]


# Facts that are part of the test report itself (same for both results) and are removed with the result line.
# A3: the treadmill protocol, duration and heart rate reached are reported inside the stress-test result line.
REPORT_FACTS = {"A3_cardiac": {"f_proto"}}
# Outcome-word hits that are not about the test of interest.
OUTCOME_ALLOW = {("A4_uti", "V"): ["a urine dipstick shows trace blood"]}


def check_pretest(fails):
    n = 0
    for facts, table, result in render_ext.scenarios():
        sc = facts["scenario"]
        all_surfaces = [s for p in render.POLS for c in result[p] for s in result[p][c]]
        for cond in render_ext.PRE_CONDS:
            tag = "pretest %s/%s" % (sc, cond)
            fn = os.path.join(render_ext.PRE_DIR, "%s__%s.txt" % (sc, cond))
            if not os.path.exists(fn):
                fails.append(tag + ": missing"); continue
            text = open(fn, encoding="utf-8").read()
            n += 1
            t_pos, rm_pos = render_ext.redact(facts, table, result, "pos", cond)
            t_neg, rm_neg = render_ext.redact(facts, table, result, "neg", cond)
            if text != t_pos:
                fails.append(tag + ": file differs from a fresh render")
            if t_pos != t_neg:
                fails.append(tag + ": redacted pos and neg differ")
            low = text.lower()
            for s in all_surfaces:
                if s.lower() in low:
                    fails.append("%s: result wording present: %r" % (tag, s))
            low_o = low
            for a in OUTCOME_ALLOW.get((sc, cond), []):
                low_o = low_o.replace(a, "")
            for w in OUTCOME_WORDS:
                if w in low_o:
                    fails.append("%s: outcome word %r present" % (tag, w))
            for w in check_notes.LEAK:
                if w in low:
                    fails.append("%s: leak word %r" % (tag, w))
            if text.splitlines()[0] != HEADER:
                fails.append(tag + ": header")
            if chr(0x2014) in text or chr(0x2013) in text:
                fails.append(tag + ": em/en dash")
            report = REPORT_FACTS.get(sc, set())
            for fid, surf in table.items():
                if fid in report:
                    continue
                if cond in surf and not any(s.lower() in low for s in surf[cond]):
                    fails.append("%s: fact %s missing after redaction" % (tag, fid))
            for pol, rm in (("pos", rm_pos), ("neg", rm_neg)):
                if result[pol][cond][0] not in rm:
                    fails.append("%s: removed span (%s) lacks the result" % (tag, pol))
                rest = rm.replace(result[pol][cond][0], "")
                for fid, surf in table.items():
                    if fid in report:
                        continue
                    for s in surf.get(cond, []):
                        if len(s) > 3 and s in rest:
                            fails.append("%s: removed span (%s) also holds fact %s" % (tag, pol, fid))
    return n


def check_long(fails):
    """Long chart L, per case: (a) the N note body (everything after its header line) appears verbatim once;
    (b) every core and filler fact of N is present; (c) the current result wording appears exactly once and
    lies inside the CURRENT ENCOUNTER block dated INDEX_DATE; (d) the older result (negative-type finding, so opposite sign
    for positive cases) appears exactly once, in a block dated more than 365 days before INDEX_DATE,
    and the current polarity's wording appears nowhere else; no result wording of either polarity and no
    "positive"/"negative" appear in the distractor text; (e) every block has a date header and only the current encounter carries INDEX_DATE;
    (f) the current encounter starts between 25% and 75% of the document (mid-document);
    (g) length 2,000 to 4,000 tokens by the chars/4 rule; (h) header, no dashes, no leak words outside the
    embedded N note (the N note already passes check_notes.py)."""
    import datetime
    from long_spec import LONG_SPEC
    n, sizes = 0, []
    idx_date = datetime.date.fromisoformat(render_ext.INDEX_DATE)
    for facts, table, result in render_ext.scenarios():
        sc = facts["scenario"]
        for pol in render.POLS:
            other = "neg" if pol == "pos" else "pos"
            tag = "long %s_%s" % (sc, pol)
            fn = os.path.join(render_ext.LONG_DIR, "%s_%s__L.txt" % (sc, pol))
            if not os.path.exists(fn):
                fails.append(tag + ": missing"); continue
            text = open(fn, encoding="utf-8").read()
            n += 1
            if text != render_ext.long_note(facts, table, result, pol):
                fails.append(tag + ": file differs from a fresh render")
            n_note = render.render_one(facts, table, result, pol, "N")
            body = n_note.split("\n", 1)[1].strip("\n")
            if text.count(body) != 1:
                fails.append(tag + ": N body not present exactly once")
            low = text.lower()
            for fid, surf in table.items():
                if "N" in surf and not any(s.lower() in low for s in surf["N"]):
                    fails.append("%s: fact %s missing" % (tag, fid))
            blocks = re.split(r"(?m)^===== (.+?) =====\n", text)
            heads = blocks[1::2]
            bodies = blocks[2::2]
            cur = [i for i, h in enumerate(heads) if h.startswith(render_ext.INDEX_DATE)]
            if len(cur) != 1 or "CURRENT ENCOUNTER" not in heads[cur[0]]:
                fails.append(tag + ": exactly one CURRENT ENCOUNTER block dated INDEX_DATE required"); continue
            ci = cur[0]
            cur_res = result[pol]["N"][0]
            if bodies[ci].count(cur_res) != 1 or text.count(cur_res) != 1:
                fails.append(tag + ": current result not exactly once inside the current encounter")
            old_res = LONG_SPEC[sc]["old_result"]["text"]
            old_hits = [i for i, b in enumerate(bodies) if old_res in b]
            if text.count(old_res) != 1 or len(old_hits) != 1:
                fails.append(tag + ": older opposite-sign result not exactly once")
            else:
                d = heads[old_hits[0]].split(" | ")[0]
                try:
                    od = datetime.date.fromisoformat(d)
                    if (idx_date - od).days <= 365:
                        fails.append(tag + ": older result dated within a year")
                except ValueError:
                    fails.append(tag + ": older result block undated")
            for s in result[pol]["N"]:
                if low.count(s.lower()) != 1:
                    fails.append("%s: current-polarity wording %r count != 1" % (tag, s))
            for i, h in enumerate(heads):
                if h == "CHART FOOTER" or h.startswith("PROBLEM LIST"):
                    if h.startswith("PROBLEM LIST") and not re.search(r"\d{4}-\d{2}-\d{2}", h):
                        fails.append(tag + ": med list undated")
                    continue
                m = re.match(r"(\d{4}-\d{2}-\d{2}) \| ", h)
                if not m:
                    fails.append("%s: block without date: %r" % (tag, h)); continue
                if i != ci and m.group(1) >= render_ext.INDEX_DATE:
                    fails.append("%s: non-current block dated on/after index: %r" % (tag, h))
            pos_frac = text.find("CURRENT ENCOUNTER") / len(text)
            if not 0.25 <= pos_frac <= 0.75:
                fails.append("%s: current encounter at %.2f of the document, not mid" % (tag, pos_frac))
            toks = len(text) / 4
            sizes.append(toks)
            if not 2000 <= toks <= 4000:
                fails.append("%s: %d tokens (chars/4), outside 2000 to 4000" % (tag, toks))
            if text.splitlines()[0] != HEADER:
                fails.append(tag + ": header")
            if chr(0x2014) in text or chr(0x2013) in text:
                fails.append(tag + ": em/en dash")
            outside = text.replace(body, "").lower()
            for s_ in [x for p_ in render.POLS for c_ in result[p_] for x in result[p_][c_]] + ["negative", "positive"]:
                if s_.lower() in outside:
                    fails.append("%s: result wording or sign word %r in distractor text" % (tag, s_))
            for w in check_notes.LEAK:
                if w in outside:
                    fails.append("%s: leak word %r in distractor text" % (tag, w))
    return n, sizes


def main(argv):
    fails = []
    what = argv or ["pretest", "long"]
    if "pretest" in what:
        n = check_pretest(fails)
        print("pretest-only notes checked: %d" % n)
    if "long" in what:
        n, sizes = check_long(fails)
        if sizes:
            print("long-chart notes checked: %d (chars/4 tokens min %d, max %d)" % (n, min(sizes), max(sizes)))
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1:])
