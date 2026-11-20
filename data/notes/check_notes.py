#!/usr/bin/env python3
"""Invariance and leakage checks for rendered notes. Stdlib only. Exit 1 on any failure."""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render

HEADER = "SYNTHETIC NOTE FOR RESEARCH. NOT A REAL PATIENT."
# Words that would interpret the result or state a probability.
LEAK = ["probab", "likelihood", "pretest", "posttest", "diagnos", "consistent with",
        "positive culture", "negative culture", "confirm", "rule out", "ruled out", "% chance",
        "wells", "alvarado", "add-rs", "add rs", "risk score", "low risk", "high risk", "intermediate risk"]


def spec_removed(sc):
    return set(render.E_SPEC[sc]["remove"])


def words(t):
    return len(t.split())


def main():
    fails, rows = [], []
    for p in sorted(glob.glob(os.path.join(render.HERE, "facts", "*.json"))):
        facts, table, result = render.load_facts(p)
        sc = facts["scenario"]
        # Wells / ADD-RS: items must sum to the stated total; score and category must never appear in a note
        if "wells_items" in facts:
            tot = sum(i["points"] for i in facts["wells_items"] if i["present"])
            cat = "low" if tot < 2 else ("moderate" if tot <= 6 else "high")
            if tot != facts["wells_total"] or not facts["wells_category"].startswith(cat):
                fails.append("%s: wells_total/category inconsistent with items" % sc)
        if "alvarado_items" in facts:
            if sum(i["points"] for i in facts["alvarado_items"] if i["present"]) != facts["alvarado_total"]:
                fails.append("%s: alvarado_total inconsistent with items" % sc)
        if "add_rs_items" in facts:
            if sum(1 for i in facts["add_rs_items"] if i["present"]) != facts["add_rs_total"]:
                fails.append("%s: add_rs_total inconsistent with items" % sc)
        if sc in render.E_SPEC:
            spec = render.E_SPEC[sc]
            n_t, e_t = render.e_template(facts)
            removed = set(spec["remove"])
            added = set(spec["add"])
            if render.placeholders(n_t) - render.placeholders(e_t) != removed:
                fails.append("%s: E drops %s but E_SPEC.remove is %s" % (
                    sc, sorted(render.placeholders(n_t) - render.placeholders(e_t)), sorted(removed)))
            if render.placeholders(e_t) - render.placeholders(n_t) != added:
                fails.append("%s: E adds placeholders %s but E_SPEC.add is %s" % (
                    sc, sorted(render.placeholders(e_t) - render.placeholders(n_t)), sorted(added)))
            if not removed <= set(table):
                fails.append("%s: E_SPEC.remove names unknown facts" % sc)
        for pol in render.POLS:
            other = "neg" if pol == "pos" else "pos"
            wc = {}
            for cond in render.CONDS:
                fn = os.path.join(render.HERE, "%s_%s__%s.txt" % (sc, pol, cond))
                tag = "%s_%s/%s" % (sc, pol, cond)
                if not os.path.exists(fn):
                    fails.append(tag + ": file missing"); continue
                text = open(fn, encoding="utf-8").read()
                low = text.lower()
                wc[cond] = words(text)
                if text.splitlines()[0] != HEADER:
                    fails.append(tag + ": header")
                if chr(0x2014) in text or chr(0x2013) in text:
                    fails.append(tag + ": em/en dash")
                for w in LEAK:
                    if w in low:
                        fails.append("%s: leak word %r" % (tag, w))
                for fid, surf in table.items():
                    if cond in surf and not any(s.lower() in low for s in surf[cond]):
                        fails.append("%s: fact %s missing" % (tag, fid))
                if not any(s.lower() in low for s in result[pol][cond]):
                    fails.append(tag + ": result wording missing")
                if any(s.lower() in low for s in result[other][cond]):
                    fails.append(tag + ": opposite-polarity wording present")
                # result appears exactly once
                n = sum(low.count(s.lower()) for s in result[pol][cond][:1])
                if n != 1:
                    fails.append("%s: result wording count %d" % (tag, n))
            # Condition E: N minus the recorded negatives; same positive facts and the same result as N
            if sc in render.E_SPEC:
                tag = "%s_%s/E" % (sc, pol)
                fn = os.path.join(render.HERE, "%s_%s__E.txt" % (sc, pol))
                if not os.path.exists(fn):
                    fails.append(tag + ": file missing")
                else:
                    text = open(fn, encoding="utf-8").read()
                    low = text.lower()
                    ntext = open(os.path.join(render.HERE, "%s_%s__N.txt" % (sc, pol)), encoding="utf-8").read().lower()
                    wc["E"] = words(text)
                    if text.splitlines()[0] != HEADER:
                        fails.append(tag + ": header")
                    if chr(0x2014) in text or chr(0x2013) in text:
                        fails.append(tag + ": em/en dash")
                    for w in LEAK:
                        if w in low:
                            fails.append("%s: leak word %r" % (tag, w))
                    for fid, surf in table.items():
                        if "E" in surf and not any(x.lower() in low for x in surf["E"]):
                            fails.append("%s: fact %s missing" % (tag, fid))
                        if "E" not in surf and fid in spec_removed(sc):
                            for x in surf.get("N", []):
                                if x.lower() in low:
                                    fails.append("%s: removed negative %s still present: %r" % (tag, fid, x))
                    for x in render.E_SPEC[sc]["template_removed"]:
                        if x.lower() in low:
                            fails.append("%s: removed template text %r still present" % (tag, x))
                    if low.count(result[pol]["N"][0].lower()) != 1:
                        fails.append(tag + ": result wording count != 1")
                    if any(x.lower() in low for x in result[other]["N"]):
                        fails.append(tag + ": opposite-polarity wording present")
                    if wc["E"] >= wc["N"]:
                        fails.append(tag + ": E not shorter than N")
                    # no new content: every sentence fragment of E occurs in N (except E-only replacement facts)
                    added = [t.lower() for t in render.E_SPEC[sc]["add"].values()]
                    for frag in re.split(r"[.;:\n]\s*", low):
                        frag = frag.strip(" .")
                        if frag and frag not in ntext and not any(a in frag for a in added):
                            fails.append("%s: E fragment not in N: %r" % (tag, frag[:60]))
            # V carries no filler; N, B, S carry the same filler set
            if "N" in wc and "B" in wc:
                r = abs(wc["B"] - wc["N"]) / wc["N"]
                if r > 0.10:
                    fails.append("%s_%s: |B-N|/N = %.3f > 0.10" % (sc, pol, r))
                if "S" in wc and wc["S"] >= wc["N"]:
                    fails.append("%s_%s: S not shorter than N" % (sc, pol))
            rows.append((sc, pol, wc))
    # A5 (condition V only, no facts file): header and dash checks; exempt from the leak check (it states prevalence, sensitivity, specificity by design)
    n_a5 = 0
    for fn in sorted(glob.glob(os.path.join(render.HERE, "A5_*__V.txt"))):
        text = open(fn, encoding="utf-8").read()
        n_a5 += 1
        if text.splitlines()[0] != HEADER:
            fails.append(os.path.basename(fn) + ": header")
        if chr(0x2014) in text or chr(0x2013) in text:
            fails.append(os.path.basename(fn) + ": em/en dash")
    print("%-16s %-4s %5s %5s %5s %5s %5s  |B-N|/N" % ("scenario", "res", "V", "N", "B", "S", "E"))
    for sc, pol, wc in rows:
        r = abs(wc["B"] - wc["N"]) / wc["N"]
        print("%-16s %-4s %5d %5d %5d %5d %5d  %.3f" % (sc, pol, wc["V"], wc["N"], wc["B"], wc["S"], wc.get("E", 0), r))
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    n_e = sum(1 for _, _, wc in rows if "E" in wc)
    assert n_a5 == 2, "expected exactly 2 A5 V notes (static, hand-written, tracked in git), found %d" % n_a5
    print("PASS: %d notes checked (%d V/N/B/S + %d E + %d A5 V)" % (len(rows) * 4 + n_e + n_a5, len(rows) * 4, n_e, n_a5))


if __name__ == "__main__":
    main()
