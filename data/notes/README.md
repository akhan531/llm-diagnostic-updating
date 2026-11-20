# Synthetic notes

All notes are synthetic. Line 1 of every file is `SYNTHETIC NOTE FOR RESEARCH. NOT A REAL PATIENT.`

## Id scheme
- Scenario: `<set><n>_<name>` (for example `A1_pneumonia`, `A4_uti`). Numbers follow the Morgan 2021 vignette order (A=pneumonia 1, UTI 4).
- Case id: `<scenario>_<pos|neg>`, naming the result variant shown. Same prior and LR family per scenario.
- Note file: `data/notes/<case_id>__<condition>.txt`, condition in V, N, B, S, E (the harness convention). E exists for A1 to A4 and B1 to B6 only (20 files); A5 is V only. The two A5 V files (`A5_hypothetical_{pos,neg}__V.txt`) are static, hand-written, tracked in git and not produced by `render.py`; `check_notes.py` asserts exactly 2 exist (do not delete them in a clean re-render).
- Case stub: `data/cases/<case_id>.json`.

## Source of truth
- `facts/<scenario>.json`: `core_facts` (from the source vignette, nothing added), `filler` (probability-neutral, each with a one-line justification), and `result` (pos and neg wording). Each fact lists acceptable surface strings per condition. First string is rendered; any counts in the check.
- `templates/<scenario>__<cond>.txt`: hand-written prose with placeholders `{fact}`, `{fact#k}`, `{^fact}` (capitalise), `{RESULT}`.
- `render.py` (stdlib, deterministic) writes the 8 notes per scenario (10 with E) and `E_removed_negatives.md`. `check_notes.py` verifies them. Edit facts or templates, never the .txt files.

## Conditions
- V: the vignette wording plus one result sentence. No filler.
- N: realistic note (CC, HPI, PMH, meds, allergies, social, ROS with unrelated negatives, exam, Results, A/P). Result in the Results section. A/P lists a neutral ddx and plan and does not interpret the result.
- B: identical sentences to N. The result is removed from Results and placed mid-paragraph in the HPI. Length within 10% of N.
- S: same facts in dense shorthand (shorter than N).
- E: the N note with the documented pertinent negatives removed, everything else identical, same result wording. Derived mechanically from the N template by `E_SPEC` in `render.py`; the removed items are listed in `E_removed_negatives.md` (generated). The Bayes target for E is the case target only if "not documented" is read as "absent", so the primary E outcome is the paired N to E shift.

## Invariants (enforced by check_notes.py)
1. Header line present. No em or en dash.
2. Every core fact and every filler fact appears in each condition where it has a surface (core in V N B S; filler in N B S).
3. Result wording for the stated polarity appears exactly once; the opposite polarity's wording never appears.
4. |words(B) - words(N)| / words(N) <= 0.10; S shorter than N.
5. No leak words (probability, likelihood, pretest, posttest, diagnos*, confirm, rule out, "consistent with", positive/negative culture).
6. Filler is identical across N, B, S of a scenario. No priors, LRs, or targets appear in any note.

## Fidelity caveats (recorded for review)
- Core facts are the Morgan Figure-legend vignettes, which condense the survey item (verbatim eAppendix wording not verified).
- A1 "vital signs otherwise normal" is rendered as T 37.2, BP 118/72, RR 16, SpO2 98% on room air (clinic temperature normal; the fever is the home history). The A1 plan names no drug or antibiotic; ibuprofen is the only antipyretic in HPI and medication list. A4 gives BP, HR, SpO2 but no temperature, so no fever information is added.
- ROS negatives are limited to systems unrelated to the target diagnosis, so that no extra pertinent negative shifts the probability.
- Visit types ("same-day", "scheduled", "follow-up") are not in the Morgan vignettes and were removed; settings are "primary care outpatient clinic" (p748). A4 shows the culture as a day-3 addendum ("Addendum, day 3: urine culture final: ...") in N and S, and as "the urine culture sent at this visit was finalized on day 3; result: ..." inside the HPI in B (a culture cannot be final at the visit where it is collected). The A4 differential lists "urinary tract infection", never "bacteriuria", so it matches the target label. A4 dipstick is stated as "trace blood" only, as in the vignette.

## Scenarios added 2026-11-11 (A2, A3, B1 to B6)
- A2_breast and A3_cardiac use the Morgan vignettes B and C. Their "no risk factors or symptoms" and "normal vital signs and examination" are elaborated in N, B, S into explicit pertinent negatives and vitals (V keeps the vignette wording). A2 pos is a plain positive (abnormal) screening mammogram with recall for additional imaging (no BI-RADS category or mass description, since Morgan p750 and p753 give an undifferentiated positive), neg is a negative screening mammogram. A3 pos is a positive exercise stress test with ischemic ST changes (no invented mm criteria, Morgan p750), neg is no ischemic ST changes (same protocol line in both).
- Set B vignettes (B1 to B6) are written for this study; there is no source vignette. Clinical facts are chosen to match the published population (Wells category, ADD-RS, suspected appendicitis referred for imaging); numbers and sources are in the case files and.
- Wells decision: no note states the Wells score, category, or the word "Wells" (check_notes.py rejects "wells", "add-rs", "risk score", "low/high/intermediate risk"). Instead each facts file carries `wells_items` (item, points, present), `wells_total`, `wells_category`, so a clinician can recompute the score from the note facts. check_notes.py verifies items sum to the total and match the category. B1 = 0 (low), B2 = 4.5 (PE leading consideration 3 plus HR 108 1.5; fever, hypoxia, abnormal breath sounds and troponin were neutralised so no unscored finding shifts the prior), B3 = 9.0 (DVT signs 3, PE leading 3, HR 112 1.5, immobilization 1.5). Item-by-item scores are in each facts file (`wells_note`) and in the B1 to B3 case-file notes.
- The Wells "alternative diagnosis less likely than PE" item is a clinician judgment, so each PE note has a one line "Clinical assessment" in the A/P and the V text states it. This is the only qualitative likelihood statement in the notes and is a deliberate fact, not filler. It is identical across result polarities and conditions.
- B4_aas: `add_rs_items` lists the three ADD-RS categories. ADD-RS = 1 (abrupt severe pain only; no high-risk condition, no pulse or BP differential, no focal deficit, no new murmur). Hypertension is present but is not an ADD-RS item. The primary prior is the ADD-RS = 1 stratum prevalence 96/1071 (Nazerian p. 253-254); the D-dimer LRs are cohort-wide. The clinician line ("acute aortic syndrome is among the leading considerations") and "diaphoretic" are not ADD-RS items; they are deliberate realism facts, identical across conditions and polarities. Both D-dimer results state the 500 ng/mL cutoff.
- PE and AAS D-dimer results state ng/mL FEU and the 500 ng/mL FEU cutoff in both polarities (ADvISED p251). B1 says "ELISA D-dimer" (Stein 2004 ELISA row). B2 and B3 positive CTPA is generic ("filling defect in a pulmonary artery"), with no segment level or clot-burden detail.
- B5_appy_ct and B6_appy_us share identical clinical facts and fillers, written as an equivocal CT-referral presentation so the 23.5% prior (all adults referred for CT) is plausible: RLQ pain localized there since onset (no migration), nausea, no anorexia, T 37.1, RLQ tenderness, no rebound, WBC 9.4; Alvarado 3, items in `alvarado_items` (never stated in a note). Only the imaging line differs. US negative reads "no sonographic findings of appendicitis" (identical across V, N, B, S, E), matching the pooled Terasawa 2004 LR- 0.19 definition and avoiding the stronger visualized-normal-appendix claim; the case file records this. The pain-character filler (worse with movement and coughing) was removed from B5/B6 because it is an unscored peritoneal-irritation finding; 'no prior abdominal surgery' is treated as a pertinent negative (removed in E).
- Plans are result-neutral: "Disposition per results and reassessment", monitoring, and analgesia only. No anticoagulation, surgery, or antibiotics are mentioned. No note carries a "Results discussed with patient" line (it was dropped in all polarities because, after a decisive positive result, a plan without treatment read as the clinician discounting the result).
- B1 to B3 carry "CXR: no acute cardiopulmonary process" in the Results (filler, not a Wells item). B1 age is 52 (PERC-positive by age, see `perc_note`). B3 immobilization is bed-bound for 10 days after an acute lumbar disc herniation (both calves exposed; no cast). B3 SpO2 92% and RR 22 are unscored findings. A2 states the mammogram was performed last week with the report in chart; A3 states the exercise test was performed 2 days ago with the report in chart (all conditions); A4 plan says the urine culture was sent at the visit; A1 and A2 plans are polarity-neutral ("per chest radiograph result", "per today's and recent studies"). A3 describes atypical chest pain (substernal pressure, sometimes exertional, not reliably relieved by rest: 2 of 3 anginal features). In condition B the result is introduced with "; findings:" or "; result:" so the same result string works for both polarities.
- Notes are built from facts and templates by render.py; B templates are N with the Results line removed and one sentence inserted mid-HPI.
