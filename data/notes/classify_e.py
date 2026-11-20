#!/usr/bin/env python3
"""F5 (committee #6): classify every pertinent negative removed in condition E.

Usage: python data/notes/classify_e.py   -> writes data/cases/meta/E_removed_classification.json

For each scenario in render.E_SPEC and each removed fact (plus template text removed), records the exact
N-note wording removed, whether it is a component of the score or definition behind the case's prior
(score_component), which score or definition, and the basis:
- "score item": an item of the clinical score that sets the prior stratum (Wells for B1-B3, ADD-RS for B4,
  Alvarado for B5/B6; the score items are listed in data/notes/facts/<scenario>.json).
- "vignette definition": a negative stated in the Morgan 2021 source vignette, so it is part of the patient
  definition the Set A evidence-based pretest range refers to (A2, A3, A4).
- "none": a negative this study added (ROS lists, unrelated exam and lab negatives) that is in no score or
  source definition.
Fields also carry `other_relevance` for items that are not score components but bear on the target in
another way (stated, not counted). The classification is a judgment by the case builder and is
asserted to cover every removed item in E_SPEC exactly once.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import render  # noqa: E402

N = "none"
SI, VD = "score item", "vignette definition"
ROSN = (False, None, N, None)
# scenario -> fact id -> (score_component, which score or definition, basis, other_relevance)
CLASS = {
    "A1_pneumonia": {
        "f_ros_neg": ROSN,
        "f_exam_rest": (False, None, N, None),
    },
    "A2_breast": {
        "no_rf": (True, "Morgan 2021 vignette: asymptomatic woman at average risk (no family history, no prior "
                  "biopsy, no breast symptoms) presenting for screening", VD,
                  "family history and prior biopsy are also breast-cancer risk-model inputs"),
        "f_hpi": (True, "Morgan 2021 vignette: asymptomatic screening visit", VD,
                  "study wording that restates the vignette's 'no symptoms'"),
        "f_cbe": (False, None, N, "exam line added by this study (not in the Morgan vignette); a palpable mass "
                  "would turn screening into a diagnostic workup, so it bears on the setting"),
        "f_ros_neg": ROSN,
        "TEMPLATE:asymptomatic": (True, "Morgan 2021 vignette: asymptomatic screening visit", VD, None),
    },
    "A3_cardiac": {
        "no_rf": (True, "Morgan 2021 vignette: premenopausal woman with no cardiac risk factors", VD,
                  "coronary risk factors are not inputs of age, sex and symptom-type pretest tables"),
        "exam_norm": (False, None, N, "normal exam added by this study; argues against alternative causes"),
        "f_ros_neg": ROSN,
    },
    "A4_uti": {
        "no_sx": (True, "definition of urinary tract infection versus asymptomatic bacteriuria (dysuria or "
                  "urinary symptoms required); Morgan 2021 vignette states no pain or difficulty with urination", VD,
                  "removing it changes what a positive culture means, not only the probability"),
        "f_ros_neg": ROSN,
    },
    "B1_pe_low": {
        "no_hemoptysis": (True, "Wells PE: hemoptysis", SI, None),
        "no_cancer": (True, "Wells PE: malignancy", SI, None),
        "no_surg": (True, "Wells PE: immobilization or surgery in the previous 4 weeks", SI, None),
        "no_vte": (True, "Wells PE: previous PE or DVT", SI, None),
        "no_dvt": (True, "Wells PE: clinical signs and symptoms of DVT", SI, None),
        "f_ros_neg": ROSN,
    },
    "B2_pe_mod": {
        "no_hemoptysis": (True, "Wells PE: hemoptysis", SI, None),
        "no_cancer": (True, "Wells PE: malignancy", SI, None),
        "no_surg": (True, "Wells PE: immobilization or surgery in the previous 4 weeks", SI, None),
        "no_vte": (True, "Wells PE: previous PE or DVT", SI, None),
        "no_dvt": (True, "Wells PE: clinical signs and symptoms of DVT", SI, None),
        "f_ros_neg": ROSN,
    },
    "B3_pe_high": {
        "no_hemoptysis": (True, "Wells PE: hemoptysis", SI, None),
        "no_cancer": (True, "Wells PE: malignancy", SI, None),
        "no_vte": (True, "Wells PE: previous PE or DVT", SI, None),
        "f_ros_neg": ROSN,
    },
    "B4_aas": {
        "no_known": (True, "ADD-RS: high-risk conditions", SI, None),
        "no_neuro": (True, "ADD-RS: high-risk examination features (focal neurologic deficit)", SI, None),
        "no_murmur": (True, "ADD-RS: high-risk examination features (new aortic regurgitation murmur)", SI, None),
        "bp_arms": (True, "ADD-RS: high-risk examination features (pulse deficit or systolic BP differential)", SI,
                    "replaced in E by a single-arm BP, so the differential is undocumented, not denied"),
        "f_ros_neg": ROSN,
    },
}
for _sc in ("B5_appy_ct", "B6_appy_us"):
    CLASS[_sc] = {
        "no_anorexia": (True, "Alvarado: anorexia", SI, None),
        "no_rebound": (True, "Alvarado: rebound tenderness", SI, None),
        "f_ros_neg": ROSN,
        "f_pmh": (False, None, N, "no prior abdominal surgery is a precondition (a prior appendectomy would exclude "
                  "the diagnosis), not an Alvarado item"),
        "f_ua": (False, None, N, "urinalysis without blood or pyuria argues against ureteral stone or urinary "
                 "infection as alternatives; not an Alvarado item"),
    }

CLASS["B7_strep"] = {
    "no_exudate": (True, "modified Centor: tonsillar swelling or exudate (absence documented)", SI,
                   "the prior is the unstratified prevalence among tested adults, not a Centor stratum"),
    "no_uri": (True, "modified Centor: absence of cough (also rhinorrhea, a viral feature)", SI,
               "the prior is the unstratified prevalence among tested adults, not a Centor stratum"),
    "f_ros_neg": ROSN,
}
CLASS["B8_septic"] = {
    "no_trauma": (False, None, N, "trauma is an alternative explanation for an acute effusion (hemarthrosis, internal derangement)"),
    "no_risk": (False, None, N, "prosthetic joint, joint injection or surgery, injection drug use and prior gout are "
                "recognized risk factors or alternative explanations; no score sets the prior (a single ED cohort prevalence)"),
    "f_ros_neg": ROSN,
}


def build():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "facts", "*.json"))):
        facts, table, result = render.load_facts(p)
        sc = facts["scenario"]
        if sc not in render.E_SPEC:
            continue
        spec = render.E_SPEC[sc]
        cls = CLASS[sc]
        want = set(spec["remove"]) | {"TEMPLATE:" + t for t in spec["template_removed"]}
        if set(cls) != want:
            raise SystemExit("%s: classification covers %s but E removes %s" % (sc, sorted(cls), sorted(want)))
        for fid in list(spec["remove"]) + ["TEMPLATE:" + t for t in spec["template_removed"]]:
            sc_comp, which, basis, other = cls[fid]
            if fid.startswith("TEMPLATE:"):
                text, desc = fid.split(":", 1)[1], "template text removed from the A/P line"
            else:
                text, desc = table[fid]["N"][0], spec["remove"][fid]
            for pol in ("pos", "neg"):
                rows.append({"case_id": "%s_%s" % (sc, pol), "scenario": sc, "fact_id": fid,
                             "removed_text_N": text, "description": desc, "score_component": sc_comp,
                             "score": which, "basis": basis, "other_relevance": other})
    return rows


def main():
    rows = build()
    out = os.path.join(ROOT, "data", "cases", "meta", "E_removed_classification.json")  # subfolder: data/cases/*.json is globbed as case files
    summ = {}
    for r in rows:
        if r["case_id"].endswith("_pos"):
            s = summ.setdefault(r["scenario"], {"score_component": 0, "not_score_component": 0})
            s["score_component" if r["score_component"] else "not_score_component"] += 1
    doc = {"_generated_by": "data/notes/classify_e.py",
           "_definition": "score_component = the removed negative is an item of the clinical score that sets the "
                          "prior stratum (Wells, ADD-RS, Alvarado) or part of the source-vignette patient definition "
                          "behind the Set A pretest range (basis 'vignette definition'). One row per case (pos and "
                          "neg share the removed items).",
           "summary_per_scenario": summ, "rows": rows}
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote %d rows to %s" % (len(rows), os.path.relpath(out, ROOT)))
    for sc, s in sorted(summ.items()):
        print("  %-14s score/definition items %d, other %d" % (sc, s["score_component"], s["not_score_component"]))


if __name__ == "__main__":
    main()
