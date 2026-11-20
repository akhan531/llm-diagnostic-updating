# Inflated Priors and Hindsight in LLM Diagnostic Updating from Synthetic Clinical Notes

Code, materials and model outputs for the paper of the same name by Ali Khan (Rutgers University),
submitted to W3PHIAI-27, the 11th International Workshop on Health Intelligence at AAAI-27.

## The study

A clinician (or a clinical agent) who reads a test result in a chart should revise the probability of
disease by Bayes' rule. We built 22 cases (11 scenarios, each with a positive and a negative result) whose
exact posttest targets come from published pretest probabilities and likelihood ratios, and rendered each
case as a clean vignette and as synthetic clinical notes in four formats:

| Condition | What changes |
|---|---|
| V | Clean vignette, as in Rodman et al. (2023) and Morgan et al. (2021) |
| N | Realistic synthetic clinical note |
| B | Same note with the test result relocated into the history |
| S | Same note in standard clinical shorthand |
| E | Same note with pertinent negatives omitted |

Five LLMs (Claude Sonnet 5.5, GPT-6.1 Sol, Gemini 3.8 Flash, DeepSeek V4 Pro, GPT-4o) were asked for the
pretest and posttest probability, three samples per case and condition, through OpenRouter. Follow-up runs
add a result-redacted pretest prompt, believed test accuracy, a long-chart condition, an unnamed-test
prompt, two prompt paraphrases, two added scenarios (strep pharyngitis, septic arthritis) and a sixth model
family (Qwen). Errors are scored against the exact Bayesian target, and a weighted-Bayes model
`logit(posterior) = a * logit(prior) + b * log(LR) + c` is fitted per model.

All notes are synthetic. No patient data were used.

## Repository layout

| Path | Contents |
|---|---|
| `data/cases/` | One JSON file per case: pretest probability, likelihood ratios, exact targets, and the source table or page for every number |
| `data/notes/` | Every rendered note and vignette given to the models (`<case>__<condition>.txt`), plus the note facts, templates, renderers and checks |
| `data/baselines/` | Published clinician estimates (Morgan et al. 2021), alternative targets and decision thresholds |
| `harness/` | Model-calling code (`run.py`) and the exact prompts (`harness/prompts/`) |
| `scripts/` | Cost estimates and the one-command pipelines used for the paid runs |
| `results/raw/` | Every model call as JSONL: prompt, full response, parsed answer, parse status, tokens, cost. Failed and unparseable calls are kept and flagged |
| `results/derived/` | Tables produced by the analysis scripts from the raw records |
| `analysis/` | Scoring, weighted-Bayes fits, statistics, tables and figures, with tests |
| `paper/generated/`, `paper/figures/` | The number macros, tables and figures used in the paper, as written by the analysis scripts |

Each subfolder has its own README with details.

## Reproducing the numbers in the paper

Python 3 standard library only, except `make_figures.py`, which needs matplotlib. Tested with Python 3.13.
From the repository root:

```bash
python3 analysis/score.py
python3 analysis/fit_wbayes.py
python3 analysis/make_tables.py
python3 analysis/compare_clinicians.py
python3 analysis/make_figures.py
python3 analysis/committee_fixes.py
python3 analysis/committee_fixes2.py
python3 analysis/ext_analysis.py
python3 analysis/round2_analysis.py
python3 analysis/round3_analysis.py
python3 analysis/ext6_analysis.py
python3 analysis/ext7_analysis.py
python3 analysis/ext8_analysis.py
python3 analysis/retry_counts.py

python3 -m unittest discover -s analysis/tests
python3 -m unittest discover -s harness/tests
```

The scripts read only `results/raw/` and `data/`, never modify the raw records, and rewrite
`results/derived/`, `paper/generated/` and `paper/figures/`. Every number printed in the paper is a macro in
`paper/generated/`. Re-running the pipeline reproduces those macros exactly; some unrounded values in
`results/derived/` can differ in the last digits across Python versions.

Analyses other than the pre-specified hypotheses H1 to H4 (the `committee_fixes*`, `ext*` and `round*`
scripts) are post hoc. The files are named after the review round that prompted them.

## Re-running the model calls

The calls go through OpenRouter and cost money (the whole study cost about $18). Set `OPENROUTER_API_KEY` and
see `harness/README.md` and `scripts/run_pipeline.sh`. Use `DRY_RUN=1` with a scratch `RAW_DIR` to test the
pipeline without any network calls. Models change over time, so new calls will not reproduce the stored
responses exactly. Analyses should be run on the stored records in `results/raw/`.

## Sources

Pretest probabilities and likelihood ratios come from published studies. Each case file cites the study,
its DOI or PMID, and the exact table or page. The full texts of the sources are not included because of
copyright. Four source checks in `analysis/tests/test_clinicians.py` are skipped unless the text of
Morgan et al. (2021) is placed at `data/sources/morgan2021.txt`.

## AI use

AI agents helped execute the experiments and draft the manuscript. The author provided the full research
direction and reviewed all work.

## License

Code is released under the MIT License (`LICENSE`). The cases, notes and results are released under
CC BY 4.0 (`LICENSE-DATA`).

## Citation

Khan, A. Inflated Priors and Hindsight in LLM Diagnostic Updating from Synthetic Clinical Notes.
Submitted to W3PHIAI-27 (AAAI-27 workshop), 2026.
