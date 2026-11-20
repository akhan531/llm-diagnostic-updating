#!/usr/bin/env python3
"""Deterministic note renderer. Stdlib only.

Usage: python data/notes/render.py [facts_file ...]   (default: all data/notes/facts/*.json)
Reads facts/<scenario>.json and templates/<scenario>__<cond>.txt, writes
<scenario>_<pos|neg>__<cond>.txt next to this script. No randomness, no network.
Condition E (omitted pertinent negatives) is derived mechanically from the N template: E_SPEC below lists,
per scenario, the facts removed, text edits applied to the N template (each must match exactly once) and any
E-only replacement fact. E notes are written for the 12 scenarios in E_SPEC (A1-A4, B1-B8), not A5.
A5 V notes (A5_hypothetical_{pos,neg}__V.txt) are static hand-written files (numeric calibration question, no facts file); render.py does not touch them and check_notes.py asserts both exist.
Placeholders: {fact_id} first surface for the condition; {fact_id#k} k-th surface;
{^fact_id} same with first letter capitalised; {RESULT} result wording for the polarity.
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ["V", "N", "B", "S"]   # conditions rendered from their own template
ECOND = "E"                      # derived from N via E_SPEC
POLS = ["pos", "neg"]

ROS = "ROS 'Denies' list of negatives"
E_SPEC = {
    "A1_pneumonia": dict(
        remove={"f_ros_neg": ROS + " (headache, abdominal pain, nausea, vomiting, diarrhea, rash, dysuria, leg swelling)",
                "f_exam_rest": "negative exam line (abdomen soft and nontender, no leg edema, no rash)"},
        add={},
        edits=[(" Denies {f_ros_neg}.", ""), ("\n{f_exam_rest}", "")],
        template_removed=[],
        note="The Morgan vignette has no pertinent negatives; only the ROS and exam negatives this study added are removed, so E is the weakest manipulation here."),
    "A2_breast": dict(
        remove={"no_rf": "vignette negatives: no family history of breast or ovarian cancer, no prior breast biopsy, no breast lump, pain, nipple discharge, or skin changes",
                "f_hpi": "'She feels well and has no concerns today'",
                "f_cbe": "negative breast exam (symmetric, no masses, skin changes, nipple discharge, or axillary adenopathy)",
                "f_ros_neg": ROS + " (headache, cough, abdominal pain, dysuria, rash); the whole ROS line is dropped"},
        add={},
        edits=[(" {^f_hpi}. She has {no_rf}.", ""), ("ROS: Denies {f_ros_neg}.\n\n", ""),
               ("Breasts: {^f_cbe}\n", ""), ("{age_sex} at {visit}, asymptomatic.", "{age_sex} at {visit}.")],
        template_removed=["asymptomatic"],
        note="The A/P word 'asymptomatic' is also dropped."),
    "A3_cardiac": dict(
        remove={"no_rf": "vignette negatives: no hypertension, diabetes, hyperlipidemia, tobacco use, or family history of premature coronary disease",
                "exam_norm": "normal exam (no murmur, rub, or gallop; lungs clear; chest wall nontender; abdomen soft; no leg edema)",
                "f_ros_neg": ROS + " (headache today, abdominal pain, diarrhea, dysuria, rash)"},
        add={},
        edits=[(" She has {no_rf}.", ""), (" Denies {f_ros_neg}.", ""), ("\n{^exam_norm}", "")],
        template_removed=[],
        note="Normal vitals and the normal ECG are kept (values and a result, not narrative negatives)."),
    "A4_uti": dict(
        remove={"no_sx": "vignette negative: no pain or difficulty with urination",
                "f_ros_neg": ROS + " (chest pain, cough, rash)"},
        add={},
        edits=[("He has noted {odor} and reports {no_sx}.", "He has noted {odor}."), (" Denies {f_ros_neg}.", "")],
        template_removed=[],
        note="Dipstick (trace blood) and the result are kept."),
    "B1_pe_low": dict(
        remove={"no_hemoptysis": "no hemoptysis", "no_cancer": "no history of cancer", "no_surg": "no recent surgery or immobilization",
                "no_vte": "no prior DVT or PE", "no_dvt": "no unilateral leg swelling or calf tenderness (extremity exam line)",
                "f_ros_neg": ROS + " (abdominal pain, nausea, vomiting, diarrhea, rash, headache, dysuria)"},
        add={},
        edits=[(" She has {no_hemoptysis}.", ""), ("PMH: {^f_pmh}; {no_cancer}; {no_vte}.", "PMH: {^f_pmh}."),
               ("Surgical hx: {^no_surg}.\n", ""), (" Denies {f_ros_neg}.", ""), ("Extremities: {^no_dvt}.\n", "")],
        template_removed=[],
        note="All six Wells-scored negatives are removed. The clinician-assessment line (Wells item 2) is kept."),
    "B2_pe_mod": dict(
        remove={"no_hemoptysis": "no hemoptysis", "no_cancer": "no history of cancer", "no_surg": "no recent surgery or immobilization",
                "no_vte": "no prior DVT or PE", "no_dvt": "no leg swelling or calf tenderness (extremity exam line)",
                "f_ros_neg": ROS + " (headache, abdominal pain, nausea, vomiting, diarrhea, dysuria, rash)"},
        add={},
        edits=[(" He has {no_hemoptysis}.", ""), ("PMH: {^f_pmh}; {no_cancer}; {no_vte}.", "PMH: {^f_pmh}."),
               ("Surgical hx: {^no_surg}.\n", ""), (" Denies {f_ros_neg}.", ""), ("Extremities: {^no_dvt}.\n", "")],
        template_removed=[],
        note="All Wells-scored negatives are removed. The clinician-assessment line is kept."),
    "B3_pe_high": dict(
        remove={"no_hemoptysis": "no hemoptysis", "no_cancer": "no history of cancer", "no_vte": "no prior DVT or PE",
                "f_ros_neg": ROS + " (headache, abdominal pain, nausea, vomiting, diarrhea, dysuria, rash)"},
        add={},
        edits=[(" She has {no_hemoptysis}.", ""), ("PMH: {^f_pmh}; {no_cancer}; {no_vte}.", "PMH: {^f_pmh}."),
               (" Denies {f_ros_neg}.", "")],
        template_removed=[],
        note="Positive Wells items (DVT signs, immobilization, HR) and the clinician assessment are kept."),
    "B4_aas": dict(
        remove={"no_known": "no known aortic disease, Marfan syndrome, family history of aortic dissection, or recent aortic surgery or instrumentation",
                "no_neuro": "no focal neurologic deficit (neuro exam line)", "no_murmur": "no new murmur",
                "bp_arms": "bilateral arm BPs and symmetric pulses (documents no pulse or BP differential)",
                "f_ros_neg": ROS + " (headache, fever, cough, abdominal pain, vomiting, diarrhea, dysuria)"},
        add={"bp_one": "BP 150/88"},
        edits=[(" He has {no_known}.", ""), ("Vitals: {bp_arms}.", "Vitals: {bp_one}."), ("; {no_murmur}.", "."),
               ("Neuro: {^no_neuro}.\n", ""), (" Denies {f_ros_neg}.", "")],
        template_removed=[],
        note="Single-arm BP is kept (one value, no comparison), so the note no longer documents the absence of a pulse or BP differential."),
}
for _sc, _ref in (("B5_appy_ct", "RLQ"), ("B6_appy_us", "RLQ")):
    E_SPEC[_sc] = dict(
        remove={"no_anorexia": "no anorexia (Alvarado item documented absent)",
                "no_rebound": "no rebound tenderness (Alvarado item documented absent)",
                "f_ros_neg": ROS + " (chest pain, cough, dysuria, hematuria, rash, headache)",
                "f_pmh": "'no significant medical history and no prior abdominal surgery' (PMH line dropped; no prior abdominal surgery is a pertinent negative, since a prior appendectomy would exclude the diagnosis)",
                "f_ua": "urinalysis without blood, protein, or pyuria"},
        add={},
        edits=[(" but {no_anorexia}", ""), ("; {no_rebound}", ""), (" Denies {f_ros_neg}.", ""), ("PMH: {^f_pmh}.\n", ""), (" {f_ua}", "")],
        template_removed=[],
        note="Equivocal presentation (Alvarado 3): the Alvarado negatives (no anorexia, no rebound), the pertinent negative 'no prior abdominal surgery', and the unrelated negatives (ROS list, urinalysis without blood, protein, or pyuria) are removed; "
             "pain localized to the RLQ since onset, the temperature, and the WBC are values and are kept, so the note still documents no migration, T 37.1 and WBC 9.4.")

E_SPEC["B7_strep"] = dict(
    remove={"no_exudate": "no tonsillar exudate (HEENT exam negative)",
            "no_uri": "no cough or rhinorrhea (HPI negative)",
            "f_ros_neg": ROS + " (chest pain, abdominal pain, vomiting, diarrhea, dysuria, leg swelling)"},
    add={},
    edits=[("; {no_exudate}.", "."), (" She has {no_uri}.", ""), (" Denies {f_ros_neg}.", "")],
    template_removed=[],
    note="Adult sore throat presentation; the prior is the unstratified prevalence among tested adults, not a score stratum. Absence of exudate and absence of cough are modified Centor items; the tender anterior cervical nodes (the other present item), the vitals and the exam findings are kept.")
E_SPEC["B8_septic"] = dict(
    remove={"no_trauma": "no recent trauma",
            "no_risk": "no prosthetic joint, no prior joint surgery or injection, no injection drug use, and no history of gout",
            "f_ros_neg": ROS + " (headache, cough, chest pain, abdominal pain, diarrhea, dysuria, rash)"},
    add={},
    edits=[(" He has {no_trauma}.", ""), (" He has {no_risk}.", ""), (" Denies {f_ros_neg}.", "")],
    template_removed=[],
    note="Acute native-joint monoarthritis; the prior is the ED monoarthritis prevalence (Jeng 1997 via Carpenter 2011). 'No other joint pain or swelling' (the monoarticular definition), the effusion, the vitals and the labs are kept.")


def expand(surfaces):
    """Expand shorthand keys (NB, NBS) into per-condition lists."""
    out = {}
    for key, vals in surfaces.items():
        for c in key:
            if c not in CONDS + [ECOND]:
                raise ValueError("bad surface key %r" % key)
            out[c] = list(vals)
    return out


def load_facts(path):
    with open(path, encoding="utf-8") as f:
        facts = json.load(f)
    table = {}
    for fact in facts["core_facts"] + facts["filler"]:
        table[fact["id"]] = expand(fact["surfaces"])
    result = {p: expand(facts["result"][p]) for p in POLS}
    spec = E_SPEC.get(facts["scenario"])
    if spec:  # derive E: every N fact except the removed ones, same result wording as N
        for fid, surf in table.items():
            if fid not in spec["remove"] and "N" in surf:
                surf[ECOND] = list(surf["N"])
        for fid, text in spec["add"].items():
            table[fid] = {ECOND: [text]}
        for p in POLS:
            result[p][ECOND] = list(result[p]["N"])
    return facts, table, result


def e_template(facts):
    """Return (n_template_text, e_template_text) for a scenario, applying E_SPEC edits to the N template."""
    spec = E_SPEC[facts["scenario"]]
    with open(os.path.join(HERE, "templates", "%s__N.txt" % facts["scenario"]), encoding="utf-8") as f:
        n_text = f.read()
    e_text = n_text
    for old, new in spec["edits"]:
        assert e_text.count(old) == 1, "%s: E edit must match once: %r" % (facts["scenario"], old)
        e_text = e_text.replace(old, new)
    return n_text, e_text


def placeholders(text):
    return set(re.findall(r"\{\^?([A-Za-z_0-9]+)(?:#\d+)?\}", text))


def render_one(facts, table, result, pol, cond):
    tpath = os.path.join(HERE, "templates", "%s__%s.txt" % (facts["scenario"], cond))
    if cond == ECOND:
        text = e_template(facts)[1]
    else:
        with open(tpath, encoding="utf-8") as f:
            text = f.read()

    def sub(m):
        cap, name, idx = m.group(1), m.group(2), int(m.group(3) or 0)
        if name == "RESULT":
            s = result[pol][cond][0]
        else:
            s = table[name][cond][idx]
        return s[0].upper() + s[1:] if cap else s

    out = re.sub(r"\{(\^?)([A-Za-z_0-9]+)(?:#(\d+))?\}", sub, text)
    assert "{" not in out and "}" not in out, "unresolved placeholder in " + tpath
    return out.rstrip("\n") + "\n"


def write_removed_md():
    lines = ["# Condition E: omitted pertinent negatives", "",
             "Generated by `render.py` from `E_SPEC`; do not edit by hand. E is the N note with the items below removed,",
             "everything else (all positive facts, vitals values, results, the clinician assessment, the result wording)",
             "identical to N, for both result polarities. A5 is not rendered in E.", ""]
    for sc in sorted(E_SPEC):
        spec = E_SPEC[sc]
        lines.append("## " + sc)
        for fid, desc in spec["remove"].items():
            lines.append("- `%s`: %s" % (fid, desc))
        for fid, text in spec["add"].items():
            lines.append("- replaced by `%s` (E only): \"%s\"" % (fid, text))
        for t in spec["template_removed"]:
            lines.append("- template text removed: \"%s\"" % t)
        lines.append("- N-template edits: " + "; ".join("%r -> %r" % e for e in spec["edits"]))
        lines.append("- note: " + spec["note"])
        lines.append("")
    with open(os.path.join(HERE, "E_removed_negatives.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv):
    paths = argv or sorted(glob.glob(os.path.join(HERE, "facts", "*.json")))
    written = []
    for p in paths:
        facts, table, result = load_facts(p)
        for pol in POLS:
            for cond in CONDS:
                note = render_one(facts, table, result, pol, cond)
                fn = os.path.join(HERE, "%s_%s__%s.txt" % (facts["scenario"], pol, cond))
                with open(fn, "w", encoding="utf-8") as f:
                    f.write(note)
                written.append(fn)
            if facts["scenario"] in E_SPEC:
                note = render_one(facts, table, result, pol, ECOND)
                fn = os.path.join(HERE, "%s_%s__%s.txt" % (facts["scenario"], pol, ECOND))
                with open(fn, "w", encoding="utf-8") as f:
                    f.write(note)
                written.append(fn)
    if not argv:
        write_removed_md()
    print("wrote %d notes" % len(written))
    return written


if __name__ == "__main__":
    main(sys.argv[1:])
