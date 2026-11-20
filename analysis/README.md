# Analysis pipeline

Stdlib only for scoring, fitting and tables (numpy, scipy, pandas are not installed). `make_figures.py`
needs matplotlib (`python -m pip install matplotlib`; if the import fails with a PIL error,
`python -m pip install -U pillow`). Run from the repo root:

    python analysis/score.py        # scored.csv, summary_model_condition.csv, parse_failures.csv
    python analysis/fit_wbayes.py   # wbayes.csv (runs a self-check first)
    python analysis/make_tables.py  # table_*.csv, paper/generated/{results_macros,table_error,table_wbayes}.tex
    python analysis/compare_clinicians.py  # clinician_compare*.csv, paper/generated/clinician_macros.tex
    python analysis/make_figures.py # paper/figures/fig_stated_vs_target.*, fig_wbayes.*
    # order matters: score, fit_wbayes, make_tables, compare_clinicians, make_figures
    python -m unittest discover -s analysis/tests -v

Both scripts take `--raw GLOB` (default `results/raw/*.jsonl`), `--cases DIR` (default `data/cases`),
`--out DIR` (default `results/derived`). They read only raw records and case files and never write to
`results/raw/`.

Display names (`display_names` in `analysis/config.json`, one mapping, used by tables and figures), repo-relative
provenance paths (L1), and `parse_failure_breakdown.csv` (per model: collected, parsed, call errors, empty replies that
used the whole token budget, other unparsed; labels only, the parser is unchanged) are also produced.
`tex_num` never prints negative zero (L3); a tiny nonzero value gets extra decimals.

## Inputs
**Raw records** (see `harness/README.md`): uses `case_id, condition, model, seed, run_id, dry_run,
parse_status, parsed_answer = {pretest, posttest}` with answers in percent (0 to 100).

**Case files** `data/cases/<case_id>.json`: `case_id, scenario, set ("A"/"B"), result ("pos"/"neg"),
status` (optional, copied to scored.csv as `case_status`), and `target` with `pretest_point`,
`pretest_range [lo,hi]`, `lr_point`, `lr_range`, `posttest_point`, `posttest_band_primary [lo,hi]`,
`posttest_band_bayes [lo,hi]`. All target probabilities are in [0,1]. A missing field fails loudly.
Raw records naming a case with no file also fail loudly. `lr_point` must be > 0 (LR for the observed
result, i.e. LR+ for pos, LR- for neg). Bootstrap clusters are the `scenario` field.

## Run-id allow-list, attempts, condition E (C2 M6)
`analysis/config.json` lists `paper_run_ids` (["main1"]), `pilot_run_ids` (["pilot1b"]) and
`excluded_run_ids` (["pilot1"]). `score.py` and `fit_wbayes.py` take `--set paper|pilot` (default
paper), `--run-ids a,b` (explicit override, used by tests on fixtures) and `--config PATH`.
- Only the selected run ids are scored. Records of other known run ids are counted as "not selected".
  Records whose run id appears nowhere in the config are UNKNOWN: a WARNING with the count is printed,
  they are never scored, and the fix is to add the id to the config on purpose. If nothing is selected
  the script fails loudly.
- `--set pilot` defaults its output to `results/derived/pilot/` so pilot cells never mix with paper
  results. `make_tables.py` and `make_figures.py` read the run set from the provenance files and refuse
  to write pilot-scored data into `paper/generated` or `paper/figures` (`common.guard_pilot`).
- Latest-attempt dedup: for each key (run_id, case_id, condition, model, seed) only the record with the
  highest `attempt` is scored (a missing `attempt` means 1; ties go to the later line). Superseded
  records stay in `results/raw/` untouched. `attempts.csv` (written by `score.py` beside scored.csv) has per
  run_id x model: `n_records`, `n_keys`, `n_superseded`, `n_final_call_error` (keys whose latest attempt
  is still a call error, kept as parse failures), `max_attempt`. The same numbers are printed.
  `fit_wbayes.py` dedups the same way.
- Condition E is excluded from every weighted-Bayes fit (protocol Section 6; no defined prior):
  `fit_wbayes.py` skips E records (`FIT_EXCLUDED_CONDITIONS`), so `wbayes.csv` has no E rows. E is still
  scored, tabulated and contrasted (E-N, H4).
- Order of operations: drop `dry_run`, filter by run id, dedup by attempt, then score. The synthetic
  firewall is judged on all loaded records before filtering.

## Calibration case A5
A5 (numeric hypothetical, condition V only, `exclude_from_fit: true` in the case file) is scored and kept in
`scored.csv` (column `calibration` = 1, `posttest_target_pp`), but it never enters a cross-condition result:
`summary_model_condition.csv`, `table_error_by_condition.csv`, `table_contrasts.csv`, Figure 1, the fits and the
clinician comparison all skip it. Its own line is `table_calibration.csv` (per model x result: stated posttest
median, min, max, difference from the exact target in pp) and the `\cal...` macros. Counts of collected and
parsed responses (`\resNCollected`, `\resNParsed`) still include A5.

## Exclusion rules
- `dry_run: true` records are dropped before anything else; the count is printed, never scored.
- A record is "parse ok" only if `parse_status == "ok"` and both fields are present. All others
  (`call_error`, `unparseable_missing_field`, `out_of_range`) are KEPT in scored.csv with blank
  metrics, counted in `n_total` and `n_parse_fail` in every summary row, and tabulated by reason in
  `parse_failures.csv`. They are excluded from metric medians and from the fits. Nothing else is
  dropped. Out-of-range answers are not clipped back in; they count as failures.
- Every scored seed is a separate observation (no averaging across seeds).

## Clipping
Before any logit, probabilities are clipped to [1e-5, 1-1e-5] (stated 0 or 100 percent would
otherwise give infinite log-odds). Applied to stated pretest, stated posttest, and `posttest_point`
and `pretest_point` in the fits. Band errors use unclipped percentages.

## Metrics (scored.csv)
- `band_err_pp`: 0 if the stated posttest lies inside `posttest_band_primary`, else distance in
  percentage points to the nearest edge (Rodman 2023 style). `band_err_signed_pp` is the same with
  sign (negative below the band, positive above).
- `bayes_band_err_pp`: same against `posttest_band_bayes` (robustness check).
- `logodds_err`: logit(stated posttest) minus logit(`posttest_point`), signed; `abs_logodds_err`.
- `pretest_err_pp`: distance in pp from stated pretest to `pretest_range` (0 inside).
- `implied_lr`: exp(logit(post) minus logit(pre)) using the model's OWN stated pretest and posttest.
- `coherence_err`: |log(implied_lr) minus log(`lr_point`)|.
Summary (`summary_model_condition.csv`): long format, one row per model x condition x set x result x
metric with n (parse ok), median, q1, q3 (inclusive quartiles), n_total, n_parse_fail. Set is added
to the requested grouping because Set A and Set B bands have different provenance.

## Weighted-Bayes fit (wbayes.csv)
OLS of logit(post) on a*logit(prior) + b*log(`lr_point`) + c, pooled over cases (pos and neg) and
seeds within model x condition. Variants: `published` (prior = `pretest_point`) and `own` (prior =
stated pretest). Bootstrap: 2000 resamples of scenarios with replacement (all rows of a scenario move
together), seed 20261111, percentile 95% CI. Resamples with a rank-deficient design are dropped and
counted (`n_boot_valid`). Rows with fewer than 3 observations or rank-deficient data get no fit and a
note. The `note` column flags fewer than 10 clusters ("CIs unreliable"). With few scenarios, bootstrap
CIs are narrow-biased; the paper must say so.
Self-check on every run: perfect Bayesian data gives a=b=1, c=0 within 1e-6; data generated with
a=0.6, b=0.5 plus noise has truth inside the CI.
Caveat: under the `published` variant with only one prior per scenario, `a` is identified only from
between-scenario variation.
- Boundary sensitivity: every fit is repeated as `published_noboundary` and `own_noboundary`,
  dropping rows whose stated posttest is a clamped boundary answer (0 or 100, per `is_boundary` in
  common.py), and for `own_noboundary` also rows whose stated pretest is one. Same columns;
  `n_boundary_answers` is 0 by construction and `note` says how many rows were excluded. On data with
  no boundary answers the variants agree exactly with the main ones. If a main and a noboundary fit
  disagree, the main fit is flagged in the paper.
- 0/100 sensitivity flag (M2, rule set in, no criterion existed in the protocol beyond "if they disagree"):
  for each model x condition x variant (published, own) with at least one boundary answer, `boundary_flag` = 1
  (on both the main and the noboundary row, with `boundary_flag_params` naming a and/or b) if, for a or b, the
  no-boundary point estimate lies outside the main 95% CI, or the main point estimate lies outside the no-boundary
  95% CI, or the no-boundary fit does not exist. Cells without boundary answers are never flagged. A descriptive
  companion, `sumShiftTenth*N`, counts cells where a or b moves by 0.1 or more. The rule was fixed after the data
  were seen; it is reported as such. Implemented in `fit_wbayes.boundary_flags`, tested in `test_wbayes_boundary.py`. Unit test: `analysis/tests/test_wbayes_boundary.py`.
- Cases with `"exclude_from_fit": true` (A5 hypothetical) are scored but skipped by fit_wbayes.py.

## Tables, macros, figures (raw results to paper)
One command after real data lands: `scripts/run_pipeline.sh` ends with score, fit, tables, figures.
By hand: the four commands above, then `cd paper` and rebuild. `main.tex` should
`\input{generated/results_macros}` and `\input{generated/table_error}` / `table_wbayes` inside its own
`table` floats (the generated files are bare `tabular`s using booktabs, which main.tex loads; wrap them in
`\resizebox{\columnwidth}{!}{...}` if too wide).

`make_tables.py` (flags `--derived`, `--out-derived`, `--generated`, `--n-boot`) writes:
- `table_error_by_condition.csv`: model x set x result (pos, neg, all) x condition: n_total, n_ok,
  n_parse_fail, median and Q1/Q3 of band error (pp) and of |log-odds error|. Cross-checked against
  `summary_model_condition.csv` (n, n_parse_fail, median must agree). A5 (calibration case, `exclude_from_fit`) is NOT in any error cell, so Set A n_total is equal across V, N, B, S, E.
- `table_contrasts.csv`: the planned contrasts of protocol Section 6, no others: N-V (H1), B-N (H2),
  S-N (H3) on band error and |log-odds error|; E-N signed log-odds shift (H4). Paired by case and seed
  (pairs where both parsed), averaged over seeds within case; statistic = median over cases of the paired
  difference; 95% percentile CI from a scenario-clustered bootstrap (seed 20261111). Per model, set
  (A, B) and result (all, pos, neg). `frac_positive` = fraction of cases with a positive difference.
  `note` flags fewer than 10 clusters (Set A has 4 scenarios without A5, so Set A CIs are unreliable).
- `table_wbayes_own.csv`, `table_wbayes_own.tex`: the same layout for the own-stated-prior fit (M4).
- `table_calibration.csv`, `table_overestimation.csv`, `table_model_ranking.csv`: A5 line; positive-test overestimation by model; per-model median error ranking.
- `table_wbayes.csv`: published-prior fit (a, b, c, 95% CI) beside the `published_noboundary` fit
  for conditions V, N, B, S. Condition E is omitted by protocol (no defined prior), and `wbayes.csv` has no E rows. The `own` variant stays in `wbayes.csv` only.
- `paper/generated/results_macros.tex`: one `\newcommand` per number, names letters only. Scheme (model tag
  = CamelCase of the slug with digits spelled out, e.g. `gpt-4o` is `GptFourO`; condition = V N B S E):
  `wb{A|B|C}{Main|Nob}{Model}{Cond}{Pt|Lo|Hi}` fit coefficients (Main = published prior, Nob = no-boundary);
  `erb|elo` + `Set{A|B}{Model}{Cond}{Med|Qlo|Qhi}` median band error (pp) and |log-odds error|;
  `enfSet..` parse failures in that cell; `ctr{Band|Lod|Shift}{NvsV|BvsN|SvsN|EvsN}Set{A|B}{All|Pos|Neg}{Model}{Med|Lo|Hi|N}`
  contrasts; `pf{Model}{N|Fail|Pct}` and `resNScored`, `resNParseFail`, `resParseFailPct`, `resNModels`.
  Negative numbers are `\ensuremath{-}x`. The script asserts names are unique and every value non-empty.

`make_figures.py`: Figure 1 `fig_stated_vs_target` (stated vs exact posttest log-odds, one panel per
model, color and marker by condition, dashed y = x); Figure 2 `fig_wbayes` (published-prior a and b with
95% CIs, dashed line at 1). Okabe-Ito colors plus distinct markers, so it reads in grayscale. PDF and PNG.

## Synthetic data firewall
`analysis/tests/fixtures/synthetic.jsonl` (made by `fixtures/make_fixture.py`, model names `SYNTH-*`, every
record `"synthetic": true`, `dry_run` false only so the records are scored in the test) exercises the whole
chain in a temp dir (`tests/test_make_outputs.py`). Safeguards: `score.py` and `fit_wbayes.py` mark an input
synthetic if any record has `synthetic: true` or its path has a `fixtures` component, write
`provenance_score.json` / `provenance_wbayes.json` beside their outputs, and call `common.guard_output`,
which raises AssertionError before writing into `results/derived`, `paper/generated` or `paper/figures`.
`make_tables.py` and `make_figures.py` read the provenance files (missing means failure) and apply the
same guard. Tests assert the real directories are unchanged after a refused run.

## Clinician comparison (protocol outcome 2)
Baseline: `data/baselines/morgan2021_clinicians.json`, copied from Morgan 2021 (Results text p750 for the
pooled medians and IQRs, n=553; Table 3 p752 for residents n=290, attendings n=202, NP/PA n=61), with the
printed evidence ranges. Every cell is checked against `data/sources/morgan2021.txt` by
`tests/test_clinicians.py`. Pooled values are only those printed; none is computed by us. The hypothetical
question (A5) has no pretest estimate. Status VERIFIED; the survey eAppendices were not reachable and are
not used.
- `compare_clinicians.py --clinicians-only` reads only the baseline and `data/cases`, no raw records. It
  writes `results/derived/clinician_band_errors.csv` (pooled and each group, posttest against
  `posttest_band_primary`, pretest against `pretest_range`, same rule as `score.band_error`) and
  `paper/generated/clinician_macros.tex` (`clinMedPre*`, `clinErrPre*`, `clinMedPost*`, `clinErrPost*`,
  `clinInside*`; suffix scenario then Pos/Neg). Hard-coded pins fail the script if a band or value changes.
- Without that flag it takes the same `--set/--run-ids/--raw/--cases/--out/--config` flags and the same
  filtering as `score.py` (dry runs dropped, run-id allow-list, latest attempt, synthetic firewall, pilot
  guard; a pilot set writes no macros unless `--generated` is given). It writes
  `clinician_compare.csv` (model x condition x Set A case), `clinician_compare_summary.csv` (model x
  condition over A1 to A4; A5 is condition V only and stays in the case file) and the full
  `clinician_macros.tex` (`clinSetA{DiffPostMed|ModelPostErrMed|ClinPostErrMed|NCases|NCloser|NTie|NClinCloser|DiffPreMed}<Model><Cond>`).
- Definitions: model value = median stated answer over parse-ok records (seeds pooled); model band error =
  band error of that median (like for like with the clinician median); the median of per-record errors is
  kept as `model_median_record_band_err`. `diff = model minus clinician`, negative means the model is closer
  to the band. Parse failures stay out of medians and are counted in `n_total`.
- Caveats for the paper: clinicians answered pretest, then post-positive, then post-negative in a survey
  of primary-care vignettes; our models get one note per call. The clinician values are medians of
  population estimates, so the comparison is a comparison of medians and not of individuals. Four Set A
  scenarios give small, clustered samples.

## Committee-review analyses (post hoc, 2026-11-13)
    python analysis/committee_fixes.py   # after score.py; reads results/raw and data only
Writes `results/derived/cf_*.csv`, `paper/generated/committee_macros.tex` (`\cf...`), `table_hindsight.tex`,
`table_decomp.tex`, `table_runsettings.tex` and `paper/figures/fig_ratio_vs_dor.{pdf,png}`. Covers F1 (within-scenario
update ratio), F2 (hindsight in the stated pretest), F3 (everything without A4), F6 (paired log-odds contrasts with
exact scenario-level sign-flip tests, sets pooled; b pooled per model), F7 (0.5% floor), F9 (prior vs update error),
F10 (clinician midpoint error), F11 (run settings and length-stopped replies). Same synthetic firewall and pilot
guard as the other scripts. Tests: tests/test_committee_fixes.py.
Macro list: paper/NUMBERS.md, section "Committee-review analyses".

    python analysis/committee_fixes2.py  # second pass: F4, F5, F8
Writes `results/derived/cf2_*.csv`, `paper/generated/committee_macros2.tex` (`\cfAlt..`, `\cfHFour..`, `\cfThr..`),
`table_alt_targets.tex`, `table_thresholds.tex`. F4 alternative targets (PIOPED II LR CI bands and observed stratum
PPV/NPV for B2/B3, Pickhardt 2011 for B5), F5 H4 by removed-item type, F8 decision-threshold discordance. Inputs
beyond the cases: data/baselines/alt_targets.json, decision_thresholds.json, data/cases/meta/E_removed_classification.json.
Tests: tests/test_committee_fixes2.py.

## R1 extension analyses E1 to E5 (post hoc, 2026-11-13)
One command, after any subset of the paid batches has landed in results/raw (no other step needed; score.py is not required):

    python3 analysis/ext_analysis.py

Reads results/raw/*.jsonl and data/ only. Run ids: main1 (selected exactly as score.py selects it) and the five
`ext_run_ids` in analysis/config.json (ext1_pretest, ext2_belief, ext3_long, ext4_unnamed, ext5_dsretry). The ext ids
(and their `_pilot` ids) are listed there only so the allow-list knows them: score.py, fit_wbayes.py, make_tables.py,
compare_clinicians.py and committee_fixes*.py still select `paper_run_ids` (main1) only, so every pre-specified and
committee output is byte-identical with or without ext files present (tested). Writes, for each batch that has records:
- E1 `ext_e1_{cells,summary,fits}.csv`: blind pretest (result-redacted note, V N S; B is identical to N after redaction
  and uses the N blind prior) vs main1 stated pretest after the positive and after the negative version, same
  condition: gaps in pp and log-odds per model x scenario, exact sign-flip tests over scenarios; weighted-Bayes refit
  with the blind prior as regressor beside the own-stated-prior and published-prior fits on the same rows; blind and
  stated prior error vs the published prior. The within-scenario ratio r uses no prior, so it is unaffected.
- E2 `ext_e2_{scenarios,summary,fits}.csv`: believed sensitivity/specificity -> believed LR+ = se/(1-sp), LR- =
  (1-se)/sp (clipped); believed vs published log DOR (scenario-level sign-flip test); b refit against the
  model's own believed log LR (b_own) and against the published LR on the same v3 answers; coherence of the posttest
  with the model's own prior and believed LR; verdict per model: knowledge error (believed DOR steeper, interval > 0),
  updating error (b_own interval > 1), both or neither.
- E3 `ext_e3_{contrasts,parse,tracking}.csv`: long chart L (ext3, prompt v1) minus main1 N, seed-paired, absolute and
  signed log-odds error and logit stated pretest, all / positive / negative cases, exact scenario sign-flip test and
  test-inversion interval (as F6); parse rates and length-stopped counts; whether answers follow the current result
  (direction of the update, and nearer to the model's own N mean for the shown result than for the opposite one).
- E4 `ext_e4_{contrasts,parse,tracking}.csv`: prompt without the test name: N-unnamed minus main1 N-named,
  L-unnamed minus ext3 L-named (needs E3), and L-unnamed minus main1 N-named.
- E5 `ext_e5_{retry,headline}.csv`: the 21 DeepSeek retries (status, still stopped at 16,000) and the headline numbers
  (overestimation after positives, median within-scenario ratio, pooled published-prior b) pre-specified (main1 as is)
  and with the retries filling their main1 keys (sensitivity).
- `paper/generated/ext_macros.tex` (`\ext...`; `\extHave{EOne..EFive}` = 1/0; macros of a missing batch are not
  defined), `table_ext.tex`, and the four slot files main.tex reads (`ext_pretest_macros.tex`, `ext_believedlr_macros.tex`,
  `ext_longchart_macros.tex`, `ext_dsretry_macros.tex`; written only when their batch exists, removed otherwise, so
  `\IfFileExists` shows the fallback). If no batch is present nothing is generated and stale files are removed.
Tests: tests/test_ext_analysis.py on fixtures/ext_synthetic (made by fixtures/make_ext_fixture.py; synthetic firewall
applies).
