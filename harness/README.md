# Harness

Stdlib only (the `requests` package is not installed). Entry point: `harness/run.py`.

## Inputs
- Case JSON (from `data/cases/`): needs `case_id`, `diagnosis`, `test_name`. Priors and LRs in the
  case file are never put in the prompt.
- Rendered note: `data/notes/<case_id>__<condition>.txt` (condition: V, N, B, S, E; E exists for A1 to A4 and B1 to B6).
- Prompt: `harness/prompts/v1.txt` (version recorded in each record as `prompt_version`).

## Usage
    python harness/run.py --run-id pilot1 --cases data/cases/a1.json --conditions N \
        --models some/model --seeds 1 --temperature 1.0 [--dry-run] [--logged-estimate USD]
    python -m unittest discover -s harness/tests -v

Output: appended to `results/raw/<run_id>.jsonl` (override with `--out-dir`). One line per call with
run_id, timestamp, case_id, condition, model, decoding_params, seed, prompt_version, prompt, response,
parsed_answer, parse_status, flagged, error, prompt_tokens, completion_tokens, cost_usd, dry_run, plus the
fields:
- `attempt` (int): 1 for a first call, n+1 for a `--retry-errors` retry of a key whose records were all
  `call_error`. Records written before this field existed have no `attempt` and count as attempt 1.
  Analysis uses the latest attempt per (run_id, case, condition, model, seed) and reports all attempts.
- `cost_source`: `reported` (provider usage.cost), `estimated_from_tokens` (cost missing; tokens x list
  price), `estimated_per_call` (cost and tokens missing; chars/4 input and max_tokens output x list price),
  or `estimated_call_error` (no response returned; charged the per-call estimate, chars/4 input and `max_tokens`
  output at list price, because a provider may bill a timeout).
- `cost_guard_usd`: the value the abort guard used for this call. `cost_usd` stays exactly what the provider
  reported (null if missing); the guard value is never written back into it.

## Behaviour
- Resumable: key is (case_id, condition, model, seed). A key is done once any record has a returned response
  (`ok`, `unparseable_missing_field`, `out_of_range`); those are flagged, never retried, never deleted.
  `call_error` records are never deleted or overwritten but do NOT count as done. A plain rerun leaves errored
  keys alone and says so; `--retry-errors` re-runs them and appends a new record with `attempt` + 1.
  A truncated last line (crash mid-write) is skipped with a warning, the next append starts on a new line,
  and the file is never edited.
- Stop on errors (exit 4, reason written to `<out-dir>/<run_id>.abort.txt`), any of: 5 `call_error` records in a
  row (`MAX_CONSECUTIVE_ERRORS`); 3 in a row for one model (`MODEL_ERROR_STREAK`); or, after at least 10 calls in
  the invocation, more than 30% of calls erroring (`ERROR_RATE_MAX`), which catches errors that alternate between
  models. Each call error also counts at the per-call estimate toward the cost abort threshold. Fix the cause, then
  rerun with `--retry-errors`.
- Live usage check (real runs): done at the start (when there are calls to make) and every 50 calls. A failure at
  the start aborts with exit 3 before any call; two failures in a row later abort too.
- Per-call cap check (real runs): before each call, the harness refuses (exit 3) if max(ledger, live usage at start)
  plus spend in this invocation plus one worst-case call (chars/4 input and `--max-tokens` output at list price)
  would exceed $40.
- Resume: the start-of-batch check uses the unfinished share of `--logged-estimate` (calls still to run divided by
  the chunk's keys), so a partly finished chunk is not refused on its full estimate.
- Budget guard. Without `--logged-estimate`, estimate (live prices, chars/4 input, `--max-tokens` output, an
  upper bound) plus the latest `usage_usd` in `results/spend_ledger.csv` must stay under $5 (pilots). With
  `--logged-estimate USD` (per chunk), the batch is refused unless it plus the ledger is under the $40 planning cap
  (start-of-batch check only). `run_pipeline.sh main` sets the run estimate from pilot1b measured tokens. The abort margin is
  per run id, not per invocation: `--run-estimate USD` (the whole-run estimate, default `--logged-estimate`)
  sets the threshold (run estimate + min(25% of it, $5); the $5 term binds for estimates above $20), and the cost already in `<run_id>.jsonl` (sum of
  `cost_guard_usd`) counts against it, so reruns get no fresh allowance. The guard also aborts if one
  invocation spends more than $40 minus the ledger, and, in real runs, checks live key usage
  (`scripts/spend.py` fetch) at the start and every 50 calls (abort at usage >= $40 or run allowance).
  Missing cost on a returned call is replaced by a conservative estimate (`cost_source`), never $0.
  Any abort writes the reason to `<run_id>.abort.txt` and exits nonzero (3 cost, 4 consecutive errors);
  records already written stay. Run `scripts/spend.py` before and after each batch (`--log` appends to the ledger).
- Subcommands: `run.py gate --pilot FILE --models ... [--allow-dry]` (main gate: all 15 pilot keys present,
  >= 12 `ok` on the latest attempt, every model >= 1 `ok`, no `dry_run` records unless `--allow-dry`; exit 1 if
  failed) and `run.py estimate-main --pilot FILE --calls model=n ... [--main-raw FILE] [--floor USD]` (recomputes the main
  estimate from the MEAN prompt and completion tokens per model at live prices, over the latest attempt of every
  pilot key that returned a reply: ok, unparseable, out_of_range or truncated; only `call_error` is excluded. Mean,
  not a maximum: with 3 pilot replies per model a percentile is just the maximum, and one outlier times 306 calls
  would inflate the estimate; a stress figure at maximum tokens is printed to stderr. `--floor` can only raise the
  total. The cap check is REMAINING cost (total less the cost recorded in `--main-raw`) plus current spend against
  $40, so an interrupted main resumes; refuses otherwise; prints the full-run total).
- Auth: `scripts.spend.auth_headers()`; no key means no header (cloud proxy injects it).
- `--dry-run`: fake responder, zero prices, no network. Seeds ending in 7 yield an unparseable reply.
  Dry-run records are marked `dry_run: true`; do not write them into `results/raw/` for real runs.

## Running locally
One command per stage, wrapping the key check, spend logging, cost estimate, harness (config) and
analysis: `scripts/run_pipeline.sh`.

- Prerequisites: `python3` (stdlib only for the harness), `OPENROUTER_API_KEY` exported in your shell
  (in a cloud run the proxy injects it instead). The key is never printed or stored.
- Pilot (pilot1b, 3 cases x 1 sample x 5 models = 15 calls, condition B, max tokens 4000): `scripts/run_pipeline.sh pilot`.
  Expected cost about $0.50 (at most about $0.6). The main gate needs >= 12 of 15 parsed ok and every model >= 1 ok.
- Main (scenario B, 2142 calls, run id `main1`): `scripts/run_pipeline.sh main`. Refuses unless the pilot1b
  gate passes (see Subcommands), recomputes the estimate from pilot1b measured tokens and prints it, and then
  requires `CONFIRM_MAIN=<that estimate>`. `LOGGED_ESTIMATE` (must be a number) can only raise it. It refuses if an earlier
  `main1.abort.txt` exists unless `ACK_ABORT=1`. It runs four resumable chunks under one run id, an abort stops
  the pipeline (nonzero exit), `spend.py --log` failures and exit 2 (cap) are fatal, a lock prevents concurrent
  runs, and a spend row is logged on exit. After an outage, rerun with `RETRY_ERRORS=1` (appends attempt 2
  records). Requires bash >= 4.4. Serial 2142 calls can take several hours; it is resumable per key.
- Pilot errors: fix the cause and `RETRY_ERRORS=1 scripts/run_pipeline.sh pilot` (appends; no new run id needed).
- If the pilot fails (unparseable or out_of_range replies count as done and are not retried, so the gate cannot
  pass on the same run id): decide the change (raise `MAX_TOKENS` in `scripts/run_pipeline.sh` if replies were
  truncated at 4000, edit the prompt and `PROMPT_VERSION` in `harness/run.py` and `harness/prompts/`, or change the
  model roster in `ALL_MODELS`), log it in, then run `PILOT_RUN_ID=pilot1c
  scripts/run_pipeline.sh pilot` (15 fresh keys; the old file stays untouched) and `PILOT_RUN_ID=pilot1c
  scripts/run_pipeline.sh main` (the main gate, estimate and analysis read that id). Add `pilot1c` to the pilot
  run ids in `analysis/config.json` so it is scored as a pilot (never as a paper run), and keep `MAX_TOKENS`
  identical between pilot and main.
- Lock: `<repo>/.pipeline.lock` (gitignored, outside `results/raw`) allows one pipeline at a time, repo-wide: a DRY_RUN in a scratch `RAW_DIR` is also refused while a real run (e.g. main1) holds it.
- Dry run (no network, no spend, no ledger write; a dry pilot passes the gate only under DRY_RUN): `DRY_RUN=1 RAW_DIR=/scratch/dir scripts/run_pipeline.sh pilot`
  then `... main`. `RAW_DIR` must be outside `results/raw`.
- Commit results: `git add results/raw results/derived results/spend_ledger.csv && git commit -m "Pilot/main run results"`.
