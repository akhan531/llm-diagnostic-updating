"""Write data/cases/<case_id>.json for every scenario (A1-A5, B1-B6), then run build_targets --write.

Inputs and sources live here; every target number is computed by build_targets.compute() via bayes.py.
Existing keys in a case file (case_id, diagnosis, test_name, ...) are preserved, new fields are added.
Usage: python analysis/make_cases.py && python analysis/build_targets.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_targets  # noqa: E402

ROOT = build_targets.ROOT
IND = ("Each case applies a single likelihood ratio (the LR for the result shown) to a single pretest "
       "probability, so no conditional-independence assumption is needed. If findings were ever combined, "
       "the LRs would be multiplied assuming conditional independence given disease status.")
MORGAN = "morgan2021"
M_ID = "PMID 33818595; DOI 10.1001/jamainternmed.2021.0269; PMC8022260"


def ms(q, loc):  # Morgan source entry, verified against PMC XML
    return {"quantity": q, "citation_key": MORGAN, "id": M_ID, "location": loc, "status": "VERIFIED"}


def src(q, key, ident, loc, status="ABSTRACT ONLY"):
    return {"quantity": q, "citation_key": key, "id": ident, "location": loc, "status": status}


A_RES = {
    "A1_pneumonia": ("a right lower lobe airspace opacity", "no focal consolidation"),
    "A2_breast": ("a positive (abnormal) screening mammogram, recalled for additional imaging", "a negative screening mammogram"),
    "A3_cardiac": ("a positive exercise stress test (ischemic ST changes)", "a negative exercise stress test (no ischemic ST changes)"),
    "A4_uti": ("greater than 100,000 CFU/mL of Escherichia coli", "no growth"),
}

SCEN = {}

# ---- Set A (Morgan 2021, Tables 3 and 4, verified) ----
SCEN["A1_pneumonia"] = dict(
    set="A", diagnosis="Community-acquired pneumonia", test_name="chest radiograph",
    setting="primary care outpatient clinic (Morgan p748; no visit type is stated in the source)",
    patient_summary="Previously healthy 35-year-old woman who smokes tobacco, 5 days of fatigue, productive cough, "
                    "worsening shortness of breath, temperatures to 38.9 C, decreased breath sounds in the lower right "
                    "field, HR 105, other vital signs normal (Morgan 2021 Figure legend, vignette A).",
    pretest_source="Morgan 2021 Table 3, pneumonia pretest evidence range 25-42%; point = log-odds midpoint.",
    inputs=dict(pretest=dict(rule="logodds_midpoint", range=[0.25, 0.42]),
                lr=dict(pos=dict(rule="published_point", point=2.57), neg=dict(rule="published_point", point=0.33)),
                band_primary=dict(rule="printed", pct=dict(pos=[46, 65], neg=[10, 19]))),
    sources=[ms("pretest 25-42%", "Table 3, Pneumonia: pretest"),
             ms("LR+ 2.57, LR- 0.33", "Table 4, Chest radiography, positive/negative"),
             ms("posttest 46-65% (pos), 10-19% (neg)", "Table 3, Pneumonia: after positive/negative")],
    status="VERIFIED", unverified=[])
SCEN["A2_breast"] = dict(
    set="A", diagnosis="Breast cancer", test_name="screening mammography",
    setting="primary care outpatient clinic, annual visit (annual visit is in the vignette)",
    patient_summary="45-year-old woman at an annual visit, no specific risk factors or symptoms for breast cancer "
                    "(Morgan 2021 Figure legend, vignette B).",
    pretest_source="Morgan 2021 Table 3, breast cancer pretest evidence range 0.2-0.3%; point = log-odds midpoint.",
    inputs=dict(pretest=dict(rule="logodds_midpoint", range=[0.002, 0.003]),
                lr=dict(pos=dict(rule="geomean_range", range=[13.0, 33.0]),
                        neg=dict(rule="geomean_range", range=[0.05, 0.24])),
                band_primary=dict(rule="printed", pct=dict(pos=[3, 9], neg=[0, 0.05]))),
    sources=[ms("pretest 0.2-0.3%", "Table 3, Breast cancer: pretest"),
             ms("LR+ 13.00-33.00, LR- 0.05-0.24", "Table 4, Mammography, positive/negative"),
             ms("posttest 3-9% (pos); '<0.05' (neg) stored as [0, 0.05]%", "Table 3, Breast cancer: after positive/negative")],
    status="VERIFIED", unverified=[],
    notes="Printed '<0.05' post-negative is not reproduced by Bayes on the printed ranges (Bayes band 0.010-0.072%).")
SCEN["A3_cardiac"] = dict(
    set="A", diagnosis="Cardiac ischemia", test_name="exercise stress test",
    setting="primary care outpatient clinic (Morgan p748)",
    patient_summary="43-year-old premenopausal woman with atypical chest pain and normal ECG, no risk factors, normal "
                    "vital signs and examination (Morgan 2021 Figure legend, vignette C).",
    pretest_source="Morgan 2021 Table 3, cardiac ischemia pretest evidence range 1-4.4%; point = log-odds midpoint.",
    inputs=dict(pretest=dict(rule="logodds_midpoint", range=[0.01, 0.044]),
                lr=dict(pos=dict(rule="geomean_range", range=[2.03, 2.65]),
                        neg=dict(rule="geomean_range", range=[0.51, 0.56])),
                band_primary=dict(rule="printed", pct=dict(pos=[2, 11], neg=[0.43, 2.5]))),
    sources=[ms("pretest 1-4.4%", "Table 3, Cardiac ischemia: pretest"),
             ms("LR+ 2.03-2.65, LR- 0.51-0.56", "Table 4, Exercise stress test, positive/negative"),
             ms("posttest 2-11% (pos), 0.43-2.5% (neg)", "Table 3, Cardiac ischemia: after positive/negative")],
    status="VERIFIED", unverified=[],
    notes="Printed post-negative lower bound 0.43 vs Bayes 0.51.")
SCEN["A4_uti"] = dict(
    set="A", diagnosis="Urinary tract infection", test_name="urine culture",
    setting="primary care outpatient clinic (Morgan p748)",
    patient_summary="65-year-old man seen for osteoarthritis, foul-smelling urine, no pain or difficulty with urination, "
                    "urine dipstick with trace blood (asymptomatic bacteriuria per Morgan 2021 Results; Figure legend, vignette D).",
    pretest_source="Morgan 2021 Table 3, UTI pretest evidence range 0-1%; lower end 0 replaced by 0.1%; "
                   "point = log-odds midpoint of 0.1% and 1%. The 0.1% replacement is by analogy with Morgan's 0% to 0.1% response rule (p749); "
                   "Morgan applies that rule to practitioner responses, not to evidence-range endpoints. "
                   "The 0.1% lower edge is a convention, independent of the analysis clipping bounds (1e-5).",
    inputs=dict(pretest=dict(rule="logodds_midpoint", range=[0.001, 0.01]),
                lr=dict(pos=dict(rule="published_point", point=9.0), neg=dict(rule="published_point", point=0.11)),
                band_primary=dict(rule="printed", pct=dict(pos=[0, 8.3], neg=[0, 0.11]))),
    sources=[ms("pretest 0-1% (stored as 0.1-1%)", "Table 3, UTI: pretest"),
             ms("LR+ 9.00, LR- 0.11", "Table 4, Urine culture, positive/negative"),
             ms("posttest 0-8.3% (pos), 0-0.11% (neg)", "Table 3, UTI: after positive/negative")],
    status="VERIFIED", unverified=[],
    notes="Zero endpoints handled. The printed bands keep their 0 lower bound. The target is 'urinary tract infection', the question "
          "Morgan's respondents answered (abstract p747, Results p750, Tables 3 and 4 p752); the vignette itself describes asymptomatic bacteriuria "
          "(Results p750). The word bacteriuria is deliberately kept out of the diagnosis string shown to models.")

def wilson(k, n, z=1.959964):
    """Wilson 95% interval for k/n (computed here, not published)."""
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(mid - half, 4), round(mid + half, 4)]


# ---- Set A5 hypothetical (Morgan 2021 Results text) ----
SCEN["A5_hypothetical"] = dict(
    set="A", diagnosis="Hypothetical disease (prevalence 1 in 1000)", test_name="hypothetical test",
    setting="numeric question, no clinical setting (condition V only)",
    patient_summary="A person tested with a test for a disease of prevalence 1 in 1000, sensitivity 100%, specificity 95% "
                    "(Morgan 2021 Results text, hypothetical question).",
    pretest_source="Morgan 2021 Results text, hypothetical question: prevalence 1 out of 1000.",
    inputs=dict(pretest=dict(rule="point", num=1, den=1000), sens=1.0, spec=0.95,
                lr=dict(pos=dict(rule="from_sens_spec"), neg=dict(rule="from_sens_spec", fit_value=0.001)),
                band_primary=dict(rule="degenerate")),
    sources=[ms("prevalence 1/1000, sensitivity 100%, specificity 95%", "Results text, hypothetical question"),
             ms("correct answers 2% (pos) and 0% (neg)", "Table 3, Hypothetical rows; Results text")],
    status="VERIFIED", unverified=[],
    exclude_from_fit=True,
    notes="Calibration check, condition V only. Posterior after positive = 0.001/(0.001+0.999*0.05) = 1.963% (Morgan prints 2). "
          "LR- = 0 exactly, so the posterior after a negative is exactly 0 and logit is undefined; lr_point 0.001 is a documented "
          "placeholder (a convention, independent of the analysis clip, which is 1e-5) so score.py can run, and exclude_from_fit is true. "
          "Primary band is degenerate at the exact posterior rather than at the rounded printed 2%.")

# ---- Set B (verified against full-text PDFs in data/sources/, 2026-11-11) ----
IND_B = "Single LR per case; no combination."


def vs(q, key, ident, loc):  # verified source entry
    return src(q, key, ident, loc, status="VERIFIED")


WELLS_ID = "PMID 11453709; DOI 10.7326/0003-4819-135-2-200107170-00010"
STEIN04_ID = "PMID 15096330; DOI 10.7326/0003-4819-140-8-200404200-00005"
PIOPED_ID = "PMID 16738268; DOI 10.1056/NEJMoa052367"
NAZ_ID = "PMID 29030346; DOI 10.1161/CIRCULATIONAHA.117.029457"
TERA_ID = "PMID 15466771; DOI 10.7326/0003-4819-141-7-200410050-00011"
PICK_ID = "PMID 21690593; DOI 10.7326/0003-4819-154-12-201106210-00006"
RUD_ID = "PMID 31743429; DOI 10.1002/14651858.CD009977.pub2; PMC6953397"
# PIOPED II CTA 2x2 among conclusive scans (Table 4, p. 2324): TP 150, FN 31, FP 25, TN 567.
CTA_SENS, CTA_SPEC = 150 / 181, 567 / 592
PIOPED_SRC = vs("CTA 2x2 (conclusive scans): 150 TP, 31 FN, 25 FP, 567 TN, so sensitivity 150/181 (83%), specificity 567/592 (96%); "
                "LRs computed from these counts (not published)", "stein2006pioped2", PIOPED_ID, "p. 2324, Table 4")
SCEN["B1_pe_low"] = dict(
    set="B", diagnosis="Pulmonary embolism", test_name="D-dimer",
    setting="emergency department",
    patient_summary="52-year-old adult ED patient with pleuritic chest pain and dyspnea, low Wells category (synthetic).",
    pretest_source="Wells 2001 Results, p. 102 (also Abstract p. 98, Figure 2 p. 103): low-probability group, 7 of 527 (1.3%, CI 0.5-2.7%) had PE.",
    inputs=dict(pretest=dict(rule="point", num=7, den=527),
                lr=dict(pos=dict(rule="published_point", point=1.68, range=[1.44, 1.95]),
                        neg=dict(rule="published_point", point=0.13, range=[0.03, 0.58])),
                band_primary=dict(rule="bayes_lr_range_at_point_prior")),
    sources=[vs("pretest 7/527 = 1.3% (CI 0.5-2.7)", "wells2001pe", WELLS_ID, "p. 102, Results text (Abstract p. 98; group sizes in Figure 2, p. 103)"),
             vs("LR+ 1.68 (1.44-1.95), LR- 0.13 (0.03-0.58), ELISA, PE studies, Tier 1 analysis", "stein2004ddimer", STEIN04_ID,
                "p. 593, Table 1 (Pulmonary Embolism, ELISA row)")],
    status="VERIFIED", unverified=[],
    notes="Positive-result variant is weakly informative. Assay mismatch: the prior is from a cohort that used the SimpliRED whole-blood "
          "agglutination assay (Wells 2001 p. 97 and throughout), while the LRs are for the ELISA class (Stein 2004 Tier 1). For the "
          "whole-blood assay class Stein 2004 Table 1 (p. 593) gives PE LR+ 2.93 (1.89-4.52), LR- 0.31 (0.18-0.51); Tier 2 ELISA PE is "
          "LR+ 1.97, LR- 0.08 (0.01-0.43). The Wells cohort's own implied LR- is about 0.34 to 0.51: 2 of the 7 low-probability PE had a negative D-dimer (diagnosed by high-probability V/Q scan at presentation, p. 102 and Figure 2 footnote, p. 103) "
          "and 1 more of the 437 D-dimer-negative low-probability patients had PE in follow-up (Abstract p. 98; low-group NPV printed as 99.5%, p. 102), so 2 to 3 of 7 PE were D-dimer negative (approximate; LR- = (k/7)/((437-k)/520) for k = 2, 3). "
          "Primary band = Bayes over the Stein Tier 1 ELISA LR CIs at the Wells prior. "
          "Wells score as written in the note (items from data/notes/facts/B1_pe_low.json): DVT signs 0; PE leading consideration 0 (viral pleurisy judged more likely); "
          "HR 88, 0; immobilization or surgery 0; prior PE or DVT 0; hemoptysis 0; malignancy 0. Total 0 = low (<2). "
          "Age 52 is not a Wells item; it keeps the patient PERC-positive (age >= 50) so a D-dimer is a guideline-plausible order (L3).")
SCEN["B2_pe_mod"] = dict(
    set="B", diagnosis="Pulmonary embolism", test_name="CT pulmonary angiography",
    setting="emergency department",
    patient_summary="Adult ED patient with dyspnea and chest pain, moderate Wells category (synthetic).",
    pretest_source="Wells 2001 Results, p. 102: moderate-probability group, 55 of 339 (16.2%, CI 12.5-20.6%) had PE.",
    inputs=dict(pretest=dict(rule="point", num=55, den=339), sens=CTA_SENS, spec=CTA_SPEC,
                lr=dict(pos=dict(rule="from_sens_spec"), neg=dict(rule="from_sens_spec")),
                band_primary=dict(rule="degenerate")),
    sources=[vs("pretest 55/339 = 16.2% (CI 12.5-20.6)", "wells2001pe", WELLS_ID, "p. 102, Results text (Abstract p. 98)"),
             PIOPED_SRC,
             vs("observed CTA PPV 92% (93/101; CI 84-96) and NPV 89% (121/136; CI 82-93) in the intermediate (Wells moderate) stratum; sensitivity target only",
                "stein2006pioped2", PIOPED_ID, "p. 2325, Table 5 (text p. 2323)")],
    status="VERIFIED", unverified=[],
    sensitivity_targets_in=dict(
        pos=[dict(label="PIOPED II observed CTA PPV, intermediate clinical probability", num=93, den=101, ci=[0.84, 0.96], location="p. 2325, Table 5")],
        neg=[dict(label="PIOPED II observed CTA NPV, intermediate clinical probability (stored as P(PE) = 1 - NPV)", num=121, den=136, ci=[0.82, 0.93], location="p. 2325, Table 5", complement=True)]),
    notes="Prior (Wells 2001 ED cohort) and LRs (PIOPED II conclusive scans) come from different cohorts. PIOPED II's observed PPV/NPV "
          "in the intermediate stratum are stored as sensitivity_targets, not as the primary target. LRs apply to conclusive scans only "
          "(51 of 824 CTAs inconclusive). Wells 2001 and PIOPED II both use the Wells score tiers (<2 low, 2-6 moderate, >6 high). "
          "Wells score as written in the note: DVT signs 0; PE leading consideration with pneumonia and pleurisy less likely 3; HR 108, 1.5; immobilization or surgery 0; "
          "prior PE or DVT 0; hemoptysis 0; malignancy 0. Total 4.5 = moderate. Unscored findings neutral (T 37.0, SpO2 96% RA, clear lungs, ECG sinus tachycardia only, no troponin).")
SCEN["B3_pe_high"] = dict(
    set="B", diagnosis="Pulmonary embolism", test_name="CT pulmonary angiography",
    setting="emergency department",
    patient_summary="Adult ED patient with dyspnea, tachycardia and unilateral leg swelling, high Wells category (synthetic).",
    pretest_source="Wells 2001 Abstract p. 98: high-probability group (n=64), 37.5% had PE (24 of 64). The Results text p. 102 prints "
                   "'24 of 64 (40.6%)', an internal inconsistency in the paper; 24/64 = 37.5%. Small n.",
    inputs=dict(pretest=dict(rule="point", num=24, den=64), sens=CTA_SENS, spec=CTA_SPEC,
                lr=dict(pos=dict(rule="from_sens_spec"), neg=dict(rule="from_sens_spec")),
                band_primary=dict(rule="degenerate"), sensitivity_priors_in=[
                    dict(label="Wells 2001 Results text p. 102 printed percentage 40.6% (CI 28.7-53.7)", pretest=0.406)]),
    sources=[vs("pretest 24/64 = 37.5% (Abstract); Results text prints 40.6% (CI 28.7-53.7) for '24 of 64', inconsistent; the count 24 and "
                "Figure 2 flow are consistent with 37.5%", "wells2001pe", WELLS_ID, "p. 98 Abstract; p. 102 Results text; p. 103 Figure 2"),
             PIOPED_SRC,
             vs("observed CTA PPV 96% (22/23; CI 78-99) and NPV 60% (9/15; CI 32-83) in the high-probability stratum; sensitivity target only",
                "stein2006pioped2", PIOPED_ID, "p. 2325, Table 5 (text p. 2323)")],
    status="VERIFIED", unverified=[],
    sensitivity_targets_in=dict(
        pos=[dict(label="PIOPED II observed CTA PPV, high clinical probability", num=22, den=23, ci=[0.78, 0.99], location="p. 2325, Table 5")],
        neg=[dict(label="PIOPED II observed CTA NPV, high clinical probability (stored as P(PE) = 1 - NPV)", num=9, den=15, ci=[0.32, 0.83], location="p. 2325, Table 5", complement=True)]),
    notes="n=64 so the prior is imprecise (Wells CI 28.7-53.7 around the printed 40.6%). Prior and LRs from different cohorts. "
          "PIOPED II high-stratum values rest on 23 and 15 patients. "
          "Wells score as written in the note: DVT signs 3 (left calf swelling, deep venous tenderness, 3 cm asymmetry); PE leading consideration, no alternative more likely 3; "
          "HR 112, 1.5; immobilization (bed-bound 10 days from an acute lumbar disc herniation; both calves exposed for the measurement) 1.5; prior PE or DVT 0; hemoptysis 0; malignancy 0. Total 9.0 = high (>6). "
          "Unscored findings (SpO2 92%, RR 22) are not Wells items; they are plausible within the high-probability stratum and are identical across conditions and result polarities (L5).")
SCEN["B4_aas"] = dict(
    set="B", diagnosis="Acute aortic syndrome", test_name="D-dimer",
    setting="emergency department",
    patient_summary="Adult ED patient with acute chest and back pain and suspected aortic syndrome, ADD-RS 1 (synthetic).",
    pretest_source="Nazerian 2018 (ADvISED) p. 253-254: ADD-RS=1 stratum, 1071 of 1850 patients (p. 253), AAS adjudicated in 96 of them (p. 254), "
                   "96/1071 = 9.0% (printed 9%, p. 254). The note is written to ADD-RS = 1, so the stratum prevalence is the primary prior.",
    inputs=dict(pretest=dict(rule="point", num=96, den=1071),
                lr=dict(pos=dict(rule="published_point", point=2.69, range=[2.51, 2.88]),
                        neg=dict(rule="published_point", point=0.05, range=[0.03, 0.1])),
                band_primary=dict(rule="bayes_lr_range_at_point_prior"),
                sensitivity_priors_in=[dict(label="Whole-cohort prevalence 241/1850 = 13% (Nazerian p. 253), the earlier primary prior", pretest=241 / 1850),
                                       dict(label="ADD-RS 0 stratum prevalence 2.7% (Nazerian p. 254)", pretest=0.027),
                                       dict(label="ADD-RS >1 stratum prevalence 39% (Nazerian p. 254)", pretest=0.39)]),
    sensitivity_targets_in=dict(
        pos=[dict(label="Observed P(AAS | D-dimer positive) within ADD-RS = 1: 441 positives, 96 - 2 = 94 AAS (counts derived from published numbers; CI computed, Wilson)",
                  num=94, den=441, ci=wilson(94, 441), location="p. 253 (441 D-dimer positive with ADD-RS=1); p. 254 (96 AAS in ADD-RS=1); p. 255 (3 AAS in ADD-RS<=1 D-dimer negative, 1 in ADD-RS=0)")],
        neg=[dict(label="Observed P(AAS | D-dimer negative) within ADD-RS = 1: 924 - 294 = 630 negatives, 3 - 1 = 2 AAS (counts derived from published numbers; CI computed, Wilson)",
                  num=2, den=630, ci=wilson(2, 630), location="p. 253 (1071 and 441 with ADD-RS=1); p. 255 (294 ADD-RS=0 D-dimer negative with 1 AAS; 924 ADD-RS<=1 D-dimer negative with 3 AAS)")]),
    sources=[vs("ADD-RS=1 stratum: 1071 of 1850 patients; 96 AAS in this stratum; prevalence 96/1071 = 9.0% (printed 9%)", "nazerian2018advised", NAZ_ID,
                "p. 253, Results (1071 patients with ADD-RS=1); p. 254, Results, ADD-RS Classification (96 AAS in ADD-RS=1; prevalence 9%)"),
             vs("D-dimer (cutoff 500 ng/mL) sensitivity 96.7% (93.6-98.6), specificity 64% (61.6-66.4), LR+ 2.69 (2.51-2.88), LR- 0.05 (0.03-0.1); "
                "8 AAS with negative D-dimer; cohort-wide values (all ADD-RS strata)", "nazerian2018advised", NAZ_ID, "p. 255, Results (D-Dimer section); Table 2 lists the 8"),
             vs("whole-cohort prevalence 241/1850 (13%); prevalence by ADD-RS 2.7% (0), 9% (1), 39% (>1); used as sensitivity priors", "nazerian2018advised", NAZ_ID,
                "p. 253, Results (13%); p. 254, Results (ADD-RS Classification)")],
    status="VERIFIED", unverified=[],
    notes="Primary prior is the ADD-RS = 1 stratum prevalence 96/1071, matching the note (ADD-RS 1: abrupt severe pain only). The D-dimer LRs are "
          "cohort-wide (computed over all ADD-RS strata, p. 255), so applying them to one stratum assumes the LRs are constant across strata. "
          "Primary band = Bayes over the published LR CIs at the 9.0% prior. The whole-cohort 13% prior, ADD-RS 0 and ADD-RS >1 prevalences are "
          "sensitivity priors; the ADD-RS plus D-dimer strategies (Table 3, p. 256) are not used for targets. "
          "The note's clinician line (acute aortic syndrome among the leading considerations) and 'diaphoretic' are not ADD-RS items; they are deliberate "
          "realism facts, identical across conditions and result polarities (L6). "
          "Stratum check: within ADD-RS = 1 the observed AAS rate is 2/630 = 0.32% after a negative D-dimer and 94/441 = 21.3% after a positive one "
          "(counts derived from p. 253-255, stored as sensitivity_targets), implying stratum LR- about 0.032 and LR+ about 2.75, both inside the primary bands, "
          "so the cohort-wide LRs are consistent with the stratum data. These observed values are not the primary target.")
APPY_PRIORS = [dict(label="Rud 2019 Cochrane median prevalence (71 populations)", pretest=0.43)]
APPY_SRC = [vs("prior 23.5% = 675/2871 consecutive adults referred for MDCT (95% CI 22.0-25.1)", "pickhardt2011mdct", PICK_ID,
               "p. 791, Results; p. 792, Table 2 (Prevalence row); Abstract p. 789"),
            src("sensitivity-analysis prior 0.43 (median prevalence, 71 populations)", "rud2019cochrane", RUD_ID,
                "Abstract, Main results (published abstract text; main-text tables not seen)", status="VERIFIED (abstract text only)")]
SCEN["B5_appy_ct"] = dict(
    set="B", diagnosis="Acute appendicitis", test_name="abdominal CT",
    setting="emergency department",
    patient_summary="Adult ED patient with equivocal right lower quadrant pain (no migration, no anorexia, afebrile, normal WBC; Alvarado 3) referred for CT for suspected appendicitis (synthetic).",
    pretest_source="Pickhardt 2011 p. 791 and Table 2 p. 792: 675/2871 = 23.5% of adults imaged by MDCT.",
    inputs=dict(pretest=dict(rule="point", num=675, den=2871),
                lr=dict(pos=dict(rule="published_point", point=13.3, range=[9.9, 17.9]),
                        neg=dict(rule="published_point", point=0.09, range=[0.07, 0.12])),
                band_primary=dict(rule="bayes_lr_range_at_point_prior"),
                sensitivity_priors_in=APPY_PRIORS + [dict(label="Terasawa 2004 CT studies pooled prevalence 533/1182, computed from Appendix Table 1 counts", pretest=533 / 1182)]),
    sources=APPY_SRC + [vs("CT pooled sensitivity 0.94 (0.91-0.95), specificity 0.95 (0.93-0.96), LR+ 13.3 (9.9-17.9), LR- 0.09 (0.07-0.12), 12 CT studies",
                           "terasawa2004appy", TERA_ID, "p. 541, Results (Abstract p. 537; Appendix Table 1 p. W-100 gives per-study counts)"),
                        vs("Pickhardt's own CT LR+ 51.3 (38.1-69.0), LR- 0.015 (0.008-0.028), context only", "pickhardt2011mdct", PICK_ID, "p. 792, Table 2")],
    status="VERIFIED", unverified=[],
    notes="Prior (single-center referral cohort) and LRs (pooled meta-analysis) are from different sources. Terasawa's included CT studies had a "
          "pooled prevalence of about 45% (computed from Appendix Table 1 counts, p. W-100), far above 23.5%; both appear as sensitivity priors. "
          "Pickhardt's own LRs are far more extreme than the pooled LRs; we keep the pooled ones. "
          + "Alvarado score as written in the note (items in data/notes/facts/B5_appy_ct.json): migration 0, anorexia 0, nausea 1, RLQ tenderness 2, rebound 0, "
          "T 37.1 0, WBC 9.4 0, left shift not reported 0; total 3. The presentation is deliberately equivocal so the 23.5% prior (all adults referred for CT) is a plausible pretest probability for the patient on the page. "
          "The Alvarado score is never stated in a note.")
SCEN["B6_appy_us"] = dict(
    set="B", diagnosis="Acute appendicitis", test_name="abdominal ultrasound",
    setting="emergency department",
    patient_summary="Adult ED patient with equivocal right lower quadrant pain (no migration, no anorexia, afebrile, normal WBC; Alvarado 3) undergoing ultrasound for suspected appendicitis (synthetic).",
    pretest_source="Pickhardt 2011 p. 791 and Table 2 p. 792: 675/2871 = 23.5% of adults imaged by MDCT (applied here to an ultrasound-evaluated adult, a cross-modality assumption).",
    inputs=dict(pretest=dict(rule="point", num=675, den=2871),
                lr=dict(pos=dict(rule="published_point", point=5.8, range=[3.5, 9.5]),
                        neg=dict(rule="published_point", point=0.19, range=[0.13, 0.27])),
                band_primary=dict(rule="bayes_lr_range_at_point_prior"),
                sensitivity_priors_in=APPY_PRIORS + [dict(label="Terasawa 2004 US studies pooled prevalence 751/1516, computed from Appendix Table 1 counts", pretest=751 / 1516)]),
    sources=APPY_SRC + [vs("US pooled sensitivity 0.86 (0.83-0.88), specificity 0.81 (0.78-0.84), LR+ 5.8 (3.5-9.5), LR- 0.19 (0.13-0.27), 14 US studies",
                           "terasawa2004appy", TERA_ID, "p. 542-543, Results (LRs on p. 543; Abstract p. 537; Appendix Table 1 p. W-100)")],
    status="VERIFIED", unverified=[],
    notes="The 23.5% prior comes from a CT-referred cohort and is applied to an ultrasound-evaluated patient (cross-modality assumption). "
          "Terasawa's ultrasound studies had pooled prevalence of about 50% (computed, Appendix Table 1). "
          + "Alvarado score as written in the note (items in data/notes/facts/B6_appy_us.json): migration 0, anorexia 0, nausea 1, RLQ tenderness 2, rebound 0, "
          "T 37.1 0, WBC 9.4 0, left shift not reported 0; total 3. The presentation is deliberately equivocal so the 23.5% prior (all adults referred for CT) is a plausible pretest probability for the patient on the page. "
          "The Alvarado score is never stated in a note; facts identical to B5. "
          "Negative-result wording: the note says \"no sonographic findings of appendicitis\", the pooled-study definition of a negative scan. "
          "Terasawa 2004 (p. 541 and Appendix Table 1 footnote) defines a positive scan but not how a nonvisualized appendix was classified, so the pooled LR- 0.19 is applied to "
          "a generic negative ultrasound, not to a visualized, compressible normal appendix (which would justify a lower posttest probability than the 5.5% target).")

# ---- Set B additions, 2026-11-17. Sources read in PMC full text. ----
STEWART_ID = "PMID 25369170; DOI 10.1371/journal.pone.0111727; PMC4219770"
CARP_ID = "PMID 21843213; DOI 10.1111/j.1553-2712.2011.01121.x; PMC3229263"
JENG_ID = "PMID 9375540; DOI 10.1016/s0735-6757(97)90173-x"


def vq(q, key, ident, loc, quote):  # verified source entry with a verbatim quote
    d = vs(q, key, ident, loc)
    d["quote"] = quote
    return d


_S_SENS, _S_SPEC = 0.91, 0.93
_S_LRP, _S_LRN = _S_SENS / (1 - _S_SPEC), (1 - _S_SENS) / _S_SPEC
_S_CORNER_POS = [0.87 / (1 - 0.92), 0.94 / (1 - 0.95)]   # sens/spec CI corners (computed, not published)
_S_CORNER_NEG = [(1 - 0.94) / 0.95, (1 - 0.87) / 0.92]
_S_PRIOR = 259 / 1216
SCEN["B7_strep"] = dict(
    set="B", diagnosis="Group A streptococcal pharyngitis", test_name="rapid antigen detection test",
    setting="primary care outpatient clinic",
    patient_summary="Adult clinic patient with 2 days of sore throat, tender anterior cervical nodes, no tonsillar exudate, no cough or rhinorrhea, afebrile (modified Centor 2); throat swab sent for a rapid antigen detection test (synthetic).",
    pretest_source="Stewart 2014 PLoS ONE, Results, 'Adults - Immunochromatographic Methods': prevalence of group A streptococcal infection in the 6 adult strata 21.3% (259/1,216 patients); the same 1,216 patients give the test sensitivity and specificity (Table 1, Immunochromatographic, Adults).",
    inputs=dict(pretest=dict(rule="point", num=259, den=1216), sens=_S_SENS, spec=_S_SPEC,
                lr=dict(pos=dict(rule="from_sens_spec"), neg=dict(rule="from_sens_spec")),
                band_primary=dict(rule="degenerate")),
    sources=[vq("pretest 259/1,216 = 21.3% (range of the 6 strata 16.1% to 25.7%)", "stewart2014rapid", STEWART_ID,
                "Results, 'Adults - Immunochromatographic Methods' (prevalence sentence); Table 1, Adults, Immunochromatographic, 'Number of patients' 1,216 and 'Number of strata' 6",
                "The prevalence of GAS infection in the 6 adults strata was 21.3% (259/1,216 patients) (range 16.1% to 25.7%)."),
             vq("pooled sensitivity 91% (95% CI 87-94), specificity 93% (95% CI 92-95), immunochromatographic methods, adults (1,216 patients, 6 strata); LRs computed from these, not published",
                "stewart2014rapid", STEWART_ID, "Table 1 (Immunochromatographic, Adults: Sensitivity, Specificity rows); Abstract and Results text",
                "In the adult population (1,216 patients), the pooled sensitivity was 91% (95% CI, 87 to 94%) and the pooled specificity was 93% (95% CI, 92 to 95%)")],
    status="VERIFIED", unverified=[],
    notes="LR arithmetic (computed from published sensitivity and specificity, not published LRs): LR+ = sens/(1 - spec) = 0.91/0.07 = %.3f; LR- = (1 - sens)/spec = 0.09/0.93 = %.4f. "
          "Prior and test come from the same 6 adult strata and the same 1,216 patients and the same culture reference standard, the cleanest prior and LR match among the Set B cases. "
          "Computed CI-corner sensitivity band (not the primary band, which is degenerate as for B2 and B3): LR+ %.1f to %.1f, LR- %.3f to %.3f. "
          "Caveats: pooled across devices (I2 61%% sensitivity, 72%% specificity); a throat culture reference standard cannot separate infection from carriage; "
          "the prior is the prevalence among adults who were tested, not Centor-stratified, so the note is written as an equivocal presentation (modified Centor 2: tender anterior nodes and no cough present; no exudate; afebrile). "
          "The clinician line (group A streptococcal pharyngitis among the considerations) is a deliberate realism fact, identical across conditions and polarities. "
          "Centor items are in data/notes/facts/B7_strep.json and are never stated in a note."
          % (_S_LRP, _S_LRN, _S_CORNER_POS[0], _S_CORNER_POS[1], _S_CORNER_NEG[0], _S_CORNER_NEG[1]))
SCEN["B8_septic"] = dict(
    set="B", diagnosis="Septic arthritis", test_name="synovial fluid white blood cell count",
    setting="emergency department",
    patient_summary="Adult ED patient with 3 days of acute right knee monoarthritis (warm, swollen, effusion, no trauma, no prosthesis or other risk factor), arthrocentesis performed (synthetic).",
    pretest_source="Carpenter 2011 Acad Emerg Med, Prevalence subsection: best-estimate prevalence of nongonococcal septic arthritis in ED patients with monoarticular arthritis 27% (95% CI 17% to 38%), from one ED-based prospective study (Jeng 1997, Taiwan; reference 47; 20 of 75 = 26.7% by the Jeng abstract). Point prior 0.27 as published.",
    inputs=dict(pretest=dict(rule="point", value=0.27),
                lr=dict(pos=dict(rule="published_point", point=4.7, range=[2.5, 8.5]),
                        neg=dict(rule="published_point", point=0.52, range=[0.38, 0.72])),
                band_primary=dict(rule="bayes_lr_range_at_point_prior"),
                sensitivity_priors_in=[dict(label="Jeng 1997 abstract counts 20/75 = 26.7% (Carpenter's 27% unrounded; Clopper-Pearson 17.1% to 38.1% reproduces Carpenter's CI)", pretest=20 / 75)]),
    alt_lr_in=dict(pos=dict(label="Carpenter 2011 interval LR for synovial WBC 50 to 100 x 10^9/L (the note's 68 x 10^9/L lies in this bin)", lr=3.59),
                   neg=dict(label="Carpenter 2011 interval LR for synovial WBC 0 to 25 x 10^9/L (the note's 12 x 10^9/L lies in this bin)", lr=0.33),
                   location="Abstract (interval LRs 0.33, 1.06, 3.59, infinity for 0-25, 25-50, 50-100, >100 x 10^9/L); no CIs published; interval LRs from four trials",
                   quote="The interval LR for synovial white blood cell (sWBC) counts of 0 x 10^9-25 x 10^9/L was 0.33; for 25 x 10^9-50 x 10^9/L, 1.06; for 50 x 10^9-100 x 10^9/L, 3.59; and exceeding 100 x 10^9/L, infinity."),
    sources=[vq("pretest 27% (95% CI 17% to 38%), nongonococcal septic arthritis in ED patients with monoarticular arthritis, from one 1997 Taiwanese ED study", "carpenter2011septic", CARP_ID,
                "Prevalence subsection (also Abstract)",
                "Based on one ED-based prospective study published in 1997 and conducted in Taiwan, the best-estimate prevalence of nongonococcal septic arthritis in ED patients with monoarticular arthritis is 27% (95% confidence interval [CI] = 17% to 38%)."),
             vq("underlying study: 20 culture-proven bacterial arthritis among 75 patients with arthritis disorders presenting to an emergency service (20/75 = 26.7%); population label (monoarticular, nongonococcal) is Carpenter's, Jeng full text not open access",
                "jeng1997tnf", JENG_ID, "Abstract (PubMed)",
                "Synovial fluid samples were obtained from 75 patients with arthritis disorders who presented to an emergency service ... Twenty patients with culture-proven bacterial arthritis"),
             vq("synovial WBC >50 x 10^9/L: sensitivity 56 (49-63), specificity 90 (88-92), +LR 4.7 (2.5-8.5), -LR 0.52 (0.38-0.72), summary estimates; the published LRs are used, not LRs recomputed from the sensitivity and specificity (0.56/0.10 = 5.6 and 0.44/0.90 = 0.49 differ because the summaries were pooled separately)",
                "carpenter2011septic", CARP_ID, "Table 4, row '>50 x 10^9/L', Summary estimate (the Results text misprints the +LR upper bound as 85; Table 4 and the Discussion print 8.5)",
                "Summary estimate / 56 (49-63) / 90 (88-92) / 4.7 (2.5-8.5) / 0.52 (0.38-0.72)"),
             vq("interval LRs 0.33, 1.06, 3.59, infinity (alternative target field)", "carpenter2011septic", CARP_ID, "Abstract",
                "The interval LR for synovial white blood cell (sWBC) counts of 0 x 10^9-25 x 10^9/L was 0.33; for 25 x 10^9-50 x 10^9/L, 1.06; for 50 x 10^9-100 x 10^9/L, 3.59; and exceeding 100 x 10^9/L, infinity."),
             vq("authors' own worked example from the 27% prior", "carpenter2011septic", CARP_ID, "Discussion, synovial WBC paragraph",
                "if one starts with a pretest probability for septic arthritis of 27%, and the sWBC count is >50 x 10^9 but <100 x 10^9/L, then the posttest probability would increase to 64%.")],
    status="VERIFIED", unverified=[],
    notes="Primary target uses the published dichotomous LRs at synovial WBC >50 x 10^9/L (one positive and one negative result per scenario). The note's values (68 and 12 x 10^9/L) lie in the interval-LR bins 50 to 100 and 0 to 25, so the "
          "interval-LR targets in alternative_lr_targets apply to the same notes; they are not used for primary scoring. "
          "Prior and LRs come from different studies: the prior is one 1997 Taiwanese ED cohort (n = 75, 20 cases; the 'monoarticular, nongonococcal' label is Carpenter's); the LRs are pooled from heterogeneous cohorts "
          "(rheumatology, mixed settings, varied cutoffs and reference standards; I2 53% to 70%). Native joints only (prosthetic joints have lower counts). "
          "The target replicates the source authors' own worked example (27% prior, 64% posttest). "
          "Note design: no prosthesis, no immunosuppression, no injection drug use, T 37.8 C, serum WBC 10.6, so no unscored finding moves the prior. "
          "The clinician line (septic arthritis among the considerations) is a deliberate realism fact, identical across conditions and polarities.")

RESULT_TEXT = {  # what the result shown is, for the spec (not used in rendered notes by this script)
    "A5_hypothetical": ("a positive test result", "a negative test result"),
    "B1_pe_low": ("ELISA D-dimer above the assay cutoff (500 ng/mL FEU)", "ELISA D-dimer below the assay cutoff (500 ng/mL FEU)"),
    "B2_pe_mod": ("a filling defect in a pulmonary artery on CTPA", "no filling defect in the pulmonary arteries on CTPA"),
    "B3_pe_high": ("a filling defect in a pulmonary artery on CTPA", "no filling defect in the pulmonary arteries on CTPA"),
    "B4_aas": ("D-dimer at or above 500 ng/mL FEU", "D-dimer below 500 ng/mL FEU"),
    "B5_appy_ct": ("a dilated, fluid-filled appendix with periappendiceal stranding on CT", "a normal appendix on CT"),
    "B6_appy_us": ("a noncompressible, dilated appendix on ultrasound", "no sonographic findings of appendicitis"),
}
RESULT_TEXT.update(A_RES)
RESULT_TEXT["B7_strep"] = ("rapid antigen detection test positive for group A streptococcal antigen", "no group A streptococcal antigen detected on rapid antigen detection test")
RESULT_TEXT["B8_septic"] = ("synovial fluid WBC 68 x 10^9/L (68,000 cells/uL)", "synovial fluid WBC 12 x 10^9/L (12,000 cells/uL)")


def sens_target(t):
    """Every stored sensitivity target is P(disease). A published NPV is converted: value = 1 - NPV, CI = [1 - hi, 1 - lo];
    num/den then count the diseased patients among the test-negative (den - NPV numerator), and the NPV itself is kept in npv/npv_ci."""
    t = dict(t)
    if t.pop("complement", False):
        npv, ci = t["num"] / t["den"], t["ci"]
        t.update(num=t["den"] - t["num"], npv=npv, npv_ci=list(ci), ci=[1 - ci[1], 1 - ci[0]],
                 quantity="P(disease) = 1 - NPV")
    t["value"] = t["num"] / t["den"]
    return t


def main():
    d = os.path.join(ROOT, "data", "cases")
    for scen, s in SCEN.items():
        for res in ("pos", "neg"):
            cid = f"{scen}_{res}"
            path = os.path.join(d, cid + ".json")
            old = json.load(open(path)) if os.path.exists(path) else {}
            c = dict(old)
            for k in ("case_id", "test_name"):  # never changed once present (diagnosis wording may be corrected)
                if k in old:
                    assert old[k] == (cid if k == "case_id" else s[k]), f"{cid}: existing {k} differs: {old[k]!r}"
            c.update(case_id=cid, scenario=scen, set=s["set"], result=res, diagnosis=s["diagnosis"],
                     test_name=s["test_name"], setting=s["setting"], patient_summary=s["patient_summary"],
                     pretest_source=s["pretest_source"],
                     result_shown=RESULT_TEXT[scen][0 if res == "pos" else 1],
                     status=s["status"], unverified=s["unverified"], sources=s["sources"],
                     conditional_independence=IND,
                     target_inputs=s["inputs"], target={})
            if s.get("notes"):
                c["notes"] = s["notes"]
            c.pop("sensitivity_targets", None)
            if s.get("sensitivity_targets_in"):
                c["sensitivity_targets"] = [sens_target(t) for t in s["sensitivity_targets_in"][res]]
            c.pop("alternative_lr_targets", None)
            if s.get("alt_lr_in"):  # alternative LR targets (e.g. interval LRs), computed with bayes.posterior at the primary prior
                a = s["alt_lr_in"]
                pre = s["inputs"]["pretest"]
                p0 = pre["num"] / pre["den"] if "num" in pre else pre["value"]
                c["alternative_lr_targets"] = [dict(
                    label=a[res]["label"], lr_point=a[res]["lr"], pretest_point=p0,
                    posttest_point=build_targets.posterior(p0, [a[res]["lr"]]),
                    citation_key="carpenter2011septic", id=CARP_ID, location=a["location"], quote=a["quote"],
                    status="VERIFIED", use="alternative target only; not used for primary scoring")]
            if s.get("exclude_from_fit"):
                c["exclude_from_fit"] = True
            if scen.startswith("A5"):
                c["conditions_run"] = ["V"]
            with open(path, "w") as f:
                json.dump(c, f, indent=2, ensure_ascii=False)
                f.write("\n")
    build_targets.main(["--write"])


if __name__ == "__main__":
    main()
