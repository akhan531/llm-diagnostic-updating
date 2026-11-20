"""Distractor chart text for the long-chart condition L (P3). Synthetic; no real patient.

Used by render_ext.long_note(). Each scenario gets the same four generic prior visits (pronouns and age filled
from the case's N facts), one unrelated laboratory panel, a problem and medication list built from the N note's
own PMH and medication facts, and one older result of the SAME test, dated more than a year before the index
encounter. The older result is always the NEGATIVE-type finding worded differently from the current negative
wording, so it is the opposite sign for every positive case and the same sign for every negative case
(explains why an older POSITIVE result cannot be used without contradicting core facts such as
"no prior DVT or PE"). Older-result contexts were chosen not to change the current prior (no prior episode of
the target condition, no new risk factor).
Rules for this text: no leak words (check_notes.LEAK), no current result wording of either polarity, never the
word "negative", every block dated, nothing about the target condition other than the older result.
"""

VISITS = [
    ("2025-10-02", "OUTPATIENT NOTE, DERMATOLOGY",
     "Reason for visit: {Pronoun} reports a rough, raised spot on the upper back present for about a year, "
     "occasionally itchy, not bleeding, not changing in color or size per {possessive} partner.\n"
     "History: No personal or family history of melanoma. Uses sunscreen inconsistently. No new medications.\n"
     "Medications and allergies: reviewed, unchanged from the reconciled list.\n"
     "Exam: Skin type II. Upper back with a 9 mm tan, waxy, stuck-on papule with a verrucous surface. Several "
     "similar 3 to 5 mm lesions on the trunk. Scattered benign-appearing nevi, symmetric, uniform color, under "
     "6 mm. No ulceration. No lymphadenopathy in the cervical or axillary chains.\n"
     "Dermoscopy: milia-like cysts and comedo-like openings; no pigment network irregularity.\n"
     "Assessment: Seborrheic keratosis, upper back, irritated. Multiple benign nevi.\n"
     "Plan: Liquid nitrogen cryotherapy to the irritated lesion, two freeze-thaw cycles, tolerated well. "
     "Wound care handout given. Daily broad-spectrum sunscreen SPF 30 or higher. Self skin checks monthly. "
     "Return in 12 months for a full skin check or sooner for any changing lesion."),
    ("2024-11-19", "OUTPATIENT NOTE, ACUTE VISIT",
     "CC: Right ear pain for 3 days.\n"
     "HPI: {Age_sex} reports right ear pain and itching for 3 days after a week of daily swimming at a "
     "community pool. Mild muffled hearing on the right. No fever, no discharge from the left ear, no dizziness, "
     "no recent upper respiratory symptoms.\n"
     "Medications and allergies: reviewed, unchanged from the reconciled list.\n"
     "Exam: T 36.7 C, HR 74, BP 122/76. Right external canal erythematous and edematous with scant white debris; "
     "pain on tragal pressure and with traction of the pinna. Tympanic membrane partly obscured, the visible "
     "portion intact and pearly. Left ear canal and tympanic membrane normal. No preauricular or cervical "
     "lymphadenopathy. Oropharynx clear.\n"
     "Assessment: Acute otitis externa, right.\n"
     "Plan: Canal cleaned gently under direct vision. Ciprofloxacin and dexamethasone otic suspension, 4 drops "
     "in the right ear twice daily for 7 days. Keep the ear dry; earplugs for showering; no swimming for 10 "
     "days. Acetaminophen as needed for pain. Return if pain worsens, fever develops, or hearing does not "
     "improve within 1 week of finishing drops."),
    ("2024-04-08", "OUTPATIENT NOTE, ACUTE VISIT",
     "CC: Left elbow pain for 6 weeks.\n"
     "HPI: {Age_sex} reports lateral left elbow pain for 6 weeks, worse with gripping, carrying grocery bags, "
     "and using a computer mouse; started after a weekend of painting a fence. No trauma, no numbness or "
     "tingling, no night pain, no swelling, no other joint symptoms.\n"
     "Medications and allergies: reviewed, unchanged from the reconciled list.\n"
     "Exam: Left elbow with full range of motion, no effusion, no warmth. Point tenderness over the lateral "
     "epicondyle and the common extensor origin. Pain with resisted wrist extension and resisted middle finger "
     "extension. Grip strength mildly reduced on the left by pain. Sensation intact in the radial, median, and "
     "ulnar distributions. Right elbow normal.\n"
     "Assessment: Lateral epicondylalgia (tennis elbow), left.\n"
     "Plan: Activity modification, counterforce strap, eccentric wrist extensor exercises (handout given), ice "
     "after activity. Topical diclofenac gel up to four times daily for 2 weeks. Physical therapy referral if "
     "not improving in 6 weeks. Discussed that most cases improve within several months."),
    ("2023-09-21", "OUTPATIENT NOTE, ACUTE VISIT",
     "CC: Itchy rash on both hands for 2 weeks.\n"
     "HPI: {Age_sex} reports an itchy, dry rash on the backs of both hands for 2 weeks after starting to "
     "wear a new brand of disposable gloves for household cleaning. Improves on weekends. No blisters, no "
     "rash elsewhere, no new soaps, no fever.\n"
     "Medications and allergies: reviewed, unchanged from the reconciled list.\n"
     "Exam: Dorsal hands and wrists with symmetric erythematous, mildly lichenified plaques with fine scale and "
     "excoriations, sharply limited at the glove line. Palms and interdigital spaces spared. Nails normal. No "
     "vesicles, no pustules, no honey-colored crust. No rash on the face, trunk, or feet.\n"
     "Assessment: Contact dermatitis of the hands, likely related to the new gloves.\n"
     "Plan: Stop the new gloves; use cotton liners under vinyl gloves. Triamcinolone 0.1% ointment twice daily "
     "for 2 weeks, then as needed. Fragrance-free emollient after each hand washing. Patch testing referral if "
     "the rash recurs after avoidance. Return if blistering, pain, or spreading."),
]

EXTRA_VISITS = [   # VISIT_BEFORE goes just before the current encounter, VISIT_AFTER near the end
    ("2025-06-11", "OUTPATIENT NOTE, ACUTE VISIT",
     "CC: Left ear fullness for 2 weeks.\n"
     "HPI: {Age_sex} reports a blocked feeling and decreased hearing in the left ear for 2 weeks, worse after "
     "showering. Uses cotton swabs most days. No ear pain, no drainage, no fever, no ringing, no vertigo, no "
     "recent air travel or upper respiratory symptoms.\n"
     "Medications and allergies: reviewed, unchanged from the reconciled list.\n"
     "Exam: T 36.6 C, HR 70, BP 120/74. Left external canal occluded by firm, dark cerumen; tympanic membrane not "
     "visible. Right canal with a small amount of soft cerumen, tympanic membrane intact with a normal light "
     "reflex. Weber lateralizes to the left; Rinne air conduction greater than bone conduction on the right. "
     "No mastoid tenderness. Nasal mucosa and oropharynx unremarkable.\n"
     "Procedure: Left ear irrigation with body-temperature water after 15 minutes of carbamide peroxide drops; "
     "a large cerumen plug was removed. Post-procedure the left canal is clear, mildly erythematous, and the "
     "tympanic membrane is intact and pearly. Patient reports hearing back to baseline.\n"
     "Assessment: Cerumen impaction, left, resolved after irrigation.\n"
     "Plan: Stop cotton swab use. Carbamide peroxide drops once weekly as needed. Return for recurrent "
     "symptoms, ear pain, or drainage."),
    ("2023-03-02", "OUTPATIENT NOTE, ACUTE VISIT",
     "CC: Right lower tooth pain for 4 days.\n"
     "HPI: {Age_sex} reports sharp pain in a right lower back tooth when chewing and with cold drinks for 4 "
     "days, after biting down on an olive pit. No facial swelling, no fever, no trismus, no drainage, no "
     "difficulty swallowing. Has not seen a dentist in about 2 years.\n"
     "Medications and allergies: reviewed, unchanged from the reconciled list.\n"
     "Exam: T 36.8 C, HR 72, BP 124/78. Right mandibular first molar with a visible fracture line on the "
     "occlusal surface and pain on biting a cotton roll; no gingival swelling, no fluctuance, no sinus tract, "
     "no tooth mobility. No submandibular or cervical lymphadenopathy. Floor of mouth soft. Face symmetric.\n"
     "Assessment: Cracked tooth, right mandibular first molar, without signs of abscess.\n"
     "Plan: Urgent dental referral placed (dental clinic called; appointment in 3 days). Soft diet, chew on the "
     "left side, avoid very cold or hot drinks. Acetaminophen as needed for pain. Return or go to the emergency "
     "department for facial swelling, fever, trouble opening the mouth, or trouble swallowing."),
]

LABS = ("2024-11-19",
        "Drawn fasting with routine health maintenance labs (unrelated panel).\n"
        "Lipid panel: total cholesterol 182 mg/dL, LDL 104 mg/dL, HDL 58 mg/dL, triglycerides 98 mg/dL.\n"
        "Hemoglobin A1c 5.3%.\n"
        "TSH 1.9 mIU/L.\n"
        "Comprehensive metabolic panel: sodium 139, potassium 4.2, chloride 103, bicarbonate 25, BUN 14, "
        "creatinine 0.9, glucose 88, calcium 9.4, albumin 4.3, total protein 7.0, AST 21, ALT 19, alkaline "
        "phosphatase 72, total bilirubin 0.6.\n"
        "Vitamin D, 25-hydroxy: 31 ng/mL.\n"
        "Urinalysis not sent. CBC not sent at this draw (last CBC on file 2023-03-02 at an outside dental "
        "office pre-procedure screen: hemoglobin 13.6 g/dL, platelets 241 x 10^9/L, white cell count within the "
        "laboratory reference range).\n"
        "Reviewed with the patient by portal message; no changes to care.")

OLD_DATE = "2024-07-15"   # 607 days before the index date 2026-03-14

LONG_SPEC = {
    "A1_pneumonia": dict(old_result=dict(
        date=OLD_DATE, label="IMAGING REPORT",
        test="Chest radiograph (PA and lateral), obtained for a pre-employment health screen",
        text="lungs clear, no infiltrate, effusion, or pneumothorax; heart size normal",
        comment="No prior films for comparison. No follow-up imaging recommended.")),
    "A2_breast": dict(old_result=dict(
        date=OLD_DATE, label="IMAGING REPORT",
        test="Screening mammogram (bilateral CC and MLO views)",
        text="BI-RADS 1, scattered fibroglandular density, no suspicious mass, calcification, or architectural distortion",
        comment="Routine screening interval recommended.")),
    "A3_cardiac": dict(old_result=dict(
        date=OLD_DATE, label="CARDIOLOGY TEST REPORT",
        test="Exercise treadmill test (Bruce protocol), obtained before starting a supervised running program",
        text="no ST segment deviation during exercise or recovery; normal blood pressure and heart rate response; no arrhythmia",
        comment="Report reviewed with the patient.")),
    "A4_uti": dict(old_result=dict(
        date=OLD_DATE, label="MICROBIOLOGY REPORT",
        test="Urine culture, sent with a routine urinalysis at a health maintenance visit",
        text="less than 10,000 CFU/mL of mixed skin flora",
        comment="Reported per laboratory protocol for mixed flora; no susceptibility testing performed.")),
    "B1_pe_low": dict(old_result=dict(
        date=OLD_DATE, label="LABORATORY REPORT (URGENT CARE)",
        test="ELISA D-dimer, drawn at an urgent care visit for chest wall soreness after moving furniture",
        text="240 ng/mL FEU (laboratory cutoff 500 ng/mL FEU)",
        comment="Chest wall soreness resolved within a week with rest and acetaminophen.")),
    "B2_pe_mod": dict(old_result=dict(
        date=OLD_DATE, label="IMAGING REPORT",
        test="CT pulmonary angiogram with IV contrast, obtained for chest discomfort after a viral illness",
        text="pulmonary arteries opacify normally to the subsegmental level without an intraluminal defect; lungs clear",
        comment="Symptoms resolved over the following week.")),
    "B3_pe_high": dict(old_result=dict(
        date=OLD_DATE, label="IMAGING REPORT",
        test="CT pulmonary angiogram with IV contrast, obtained for chest discomfort after a viral illness",
        text="pulmonary arteries opacify normally to the subsegmental level without an intraluminal defect; lungs clear",
        comment="Symptoms resolved over the following week.")),
    "B4_aas": dict(old_result=dict(
        date=OLD_DATE, label="LABORATORY REPORT (URGENT CARE)",
        test="D-dimer, drawn at an urgent care visit for chest wall soreness after moving furniture",
        text="270 ng/mL FEU (laboratory cutoff 500 ng/mL FEU)",
        comment="Chest wall soreness resolved within a week with rest and acetaminophen.")),
    "B5_appy_ct": dict(old_result=dict(
        date=OLD_DATE, label="IMAGING REPORT",
        test="CT abdomen and pelvis with IV contrast, obtained after a low-speed motor vehicle collision",
        text="no acute traumatic injury; appendix normal in caliber (4 mm) and air-filled; no free fluid",
        comment="Discharged from the emergency department the same day; no injuries.")),
    "B6_appy_us": dict(old_result=dict(
        date=OLD_DATE, label="IMAGING REPORT",
        test="Right lower quadrant and groin ultrasound (graded compression), obtained for a small right groin lump",
        text="1 cm oval lymph node with a fatty hilum in the right groin; appendix visualized, compressible, 4 mm",
        comment="Groin lump resolved on recheck at 6 weeks.")),
    "B7_strep": dict(old_result=dict(
        date=OLD_DATE, label="MICROBIOLOGY REPORT",
        test="Rapid antigen detection test for group A Streptococcus (throat swab), sent at a visit for a sore throat after a cold",
        text="group A streptococcal antigen not detected; internal control valid",
        comment="Sore throat resolved within a week with supportive care.")),
    "B8_septic": dict(old_result=dict(
        date=OLD_DATE, label="LABORATORY REPORT",
        test="Synovial fluid cell count, left knee aspiration after a sports injury",
        text="3 x 10^9/L (3,000 cells/uL), blood-tinged fluid, no organisms on Gram stain",
        comment="Left knee effusion resolved with rest and physical therapy.")),
}

COMMON = {"footer": "End of extract. Entries not shown: scanned outside records (none on file), patient "
                    "messages about scheduling, immunization history (influenza vaccine yearly, tetanus "
                    "booster 2021)."}
