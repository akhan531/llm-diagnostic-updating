#!/usr/bin/env bash
# One-command paid pipeline.
#   scripts/run_pipeline.sh pilot   # pilot1b: 3 cases x 1 sample x 5 models = 15 calls (about $0.50)
#   scripts/run_pipeline.sh main    # scenario B main run, 2142 calls (about $21.50); needs pilot1b first
# Key: OPENROUTER_API_KEY locally, or proxy injection in cloud runs (scripts/spend.py auth_headers).
# Never edits or deletes anything in results/raw/ (the harness only appends; reruns resume).
#
# Dry run (no network, no spend, no ledger writes):
#   DRY_RUN=1 RAW_DIR=/some/scratch/dir scripts/run_pipeline.sh pilot
# RAW_DIR defaults to results/raw; dry runs REFUSE to use results/raw.
# Main computes its estimate from pilot1b measured tokens and refuses unless CONFIRM_MAIN=<that estimate>
# is set. LOGGED_ESTIMATE (a number) may only raise
# it. The cap check compares the REMAINING cost (estimate less main1 cost already in raw) plus spend with $40, so
# an interrupted main resumes. Dry runs use a nominal 21.50.
# Other env: RETRY_ERRORS=1 (append retries for call_error keys), ACK_ABORT=1 (acknowledge an earlier
# abort file for main1), PILOT_RUN_ID (default pilot1b; see harness/README.md "If the pilot fails"), PYTHON. Failures of spend.py (including exit 2 at the cap) are fatal.
set -euo pipefail

if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; then
  echo "REFUSED: bash >= 4.4 required (found $BASH_VERSION). On macOS install a newer bash." >&2; exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
RAW_DIR="${RAW_DIR:-$ROOT/results/raw}"
LOGGED_ESTIMATE="${LOGGED_ESTIMATE:-}"
PILOT_RUN_ID="${PILOT_RUN_ID:-pilot1b}"
case "$PILOT_RUN_ID" in *[!A-Za-z0-9_]*|"") echo "REFUSED: PILOT_RUN_ID must be alphanumeric/underscore." >&2; exit 1 ;; esac
if [ -n "$LOGGED_ESTIMATE" ] && ! [[ "$LOGGED_ESTIMATE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "REFUSED: LOGGED_ESTIMATE='$LOGGED_ESTIMATE' is not a number (e.g. 21.50)." >&2; exit 1
fi
RETRY_ERRORS="${RETRY_ERRORS:-0}"
ACK_ABORT="${ACK_ABORT:-0}"
MAX_TOKENS=4000
TEMP=1.0

EXPENSIVE=(openai/gpt-6.1-sol anthropic/claude-sonnet-5.5)
CHEAP=(google/gemini-3.8-flash deepseek/deepseek-v4-pro openai/gpt-4o)
ALL_MODELS=("${EXPENSIVE[@]}" "${CHEAP[@]}")
PILOT_CASES=(data/cases/A2_breast_pos.json data/cases/A3_cardiac_neg.json data/cases/B2_pe_mod_neg.json)

usage() { echo "usage: $0 {pilot|main}   (env: DRY_RUN=1 RAW_DIR=dir LOGGED_ESTIMATE=usd PYTHON=python3)" >&2; exit 64; }
[ $# -eq 1 ] || usage
CMD="$1"
case "$CMD" in pilot|main) ;; *) usage ;; esac

mkdir -p "$RAW_DIR"
RAW_DIR="$(cd "$RAW_DIR" && pwd)"
if [ "$DRY_RUN" = "1" ] && [ "$RAW_DIR" = "$ROOT/results/raw" ]; then
  echo "REFUSED: DRY_RUN=1 must use RAW_DIR outside results/raw." >&2; exit 1
fi
DRYFLAG=(); [ "$DRY_RUN" = "1" ] && DRYFLAG=(--dry-run)
RETRYFLAG=(); [ "$RETRY_ERRORS" = "1" ] && RETRYFLAG=(--retry-errors)
# One pipeline per repo at a time (the lock is repo-wide, so a DRY_RUN in a scratch RAW_DIR is also refused
# while another pipeline, e.g. main1, holds it; two concurrent runs would duplicate calls and spend).
if command -v flock >/dev/null 2>&1; then
  exec 9>"$ROOT/.pipeline.lock"   # outside results/raw so it is never committed (gitignored)
  flock -n 9 || { echo "REFUSED: another run_pipeline.sh holds $ROOT/.pipeline.lock" >&2; exit 1; }
else
  echo "WARNING: flock not found; concurrent runs are not prevented." >&2
fi
DERIVED="$ROOT/results/derived"
[ "$DRY_RUN" = "1" ] && DERIVED="$RAW_DIR/derived"

echo "== run_pipeline: $CMD (dry_run=$DRY_RUN, raw dir=$RAW_DIR)"

# Main gate: the pilot ($PILOT_RUN_ID, default pilot1b) needs >= 12 of 15 records parsed ok, every model >= 1 ok (latest attempt per key).
# Pilot errors are fixed with RETRY_ERRORS=1 (appends attempt 2 records; nothing in raw is edited or deleted).
if [ "$CMD" = "main" ]; then
  GATE_FLAGS=(); [ "$DRY_RUN" = "1" ] && GATE_FLAGS=(--allow-dry)   # dry pilot counts as parsed in DRY_RUN only
  if [ ! -s "$RAW_DIR/${PILOT_RUN_ID}.jsonl" ] || ! "$PY" harness/run.py gate --pilot "$RAW_DIR/${PILOT_RUN_ID}.jsonl" \
       --models "${ALL_MODELS[@]}" ${GATE_FLAGS[@]+"${GATE_FLAGS[@]}"}; then
    echo "REFUSED: pilot $PILOT_RUN_ID missing or failed the gate. Call errors: RETRY_ERRORS=1 $0 pilot. Unparseable replies: see harness/README.md \"If the pilot fails\" (new PILOT_RUN_ID)." >&2
    exit 1
  fi
  if [ -s "$RAW_DIR/main1.abort.txt" ] && [ "$ACK_ABORT" != "1" ]; then
    echo "REFUSED: $RAW_DIR/main1.abort.txt exists (an earlier abort). Read it, fix the cause, then ACK_ABORT=1." >&2
    exit 1
  fi
fi

# 1. Key check and spend before.
if [ "$DRY_RUN" != "1" ]; then
  if ! "$PY" scripts/spend.py; then
    echo "KEY CHECK FAILED: scripts/spend.py did not succeed (401 means no key; set OPENROUTER_API_KEY, or" >&2
    echo "exit 2 means spend is at the planning cap). No calls made." >&2
    exit 1
  fi
  "$PY" scripts/spend.py --log "before $CMD (run_pipeline.sh)" || { echo "LEDGER LOG FAILED: stopping." >&2; exit 1; }
  # End-of-run ledger row even on failure or abort (L4); a failing log here only warns, the exit code is kept.
  on_exit() { local rc=$?; trap - EXIT; "$PY" scripts/spend.py --log "after $CMD exit=$rc (run_pipeline.sh)" || echo "WARNING: end-of-run ledger log failed" >&2; exit $rc; }
  trap on_exit EXIT
else
  echo "(dry run: skipping spend check and ledger)"
fi

# 2. Cost estimate (live prices; measured mean input 461 tokens; assumed output 2000 reasoning, 300 gpt-4o).
estimate() {  # $1 calls per expensive/cheap model for this stage
  local ecalls="$1" ccalls="$2" m out calls
  for m in "${ALL_MODELS[@]}"; do
    out=2000; [ "$m" = "openai/gpt-4o" ] && out=300
    calls="$ccalls"; for e in "${EXPENSIVE[@]}"; do [ "$m" = "$e" ] && calls="$ecalls"; done
    if [ "$DRY_RUN" = "1" ]; then echo "(dry) $m: $calls calls"; else
      "$PY" scripts/estimate_cost.py --model "$m" --calls "$calls" --in-tokens 461 --out-tokens "$out" || echo "(estimate failed for $m)"
    fi
  done
}
if [ "$CMD" = "pilot" ]; then
  echo "-- Estimate (pilot1b, 3 calls per model; the harness also prints its own upper bound):"
  estimate 3 3
else
  # recompute from pilot1b measured tokens per model; refuses if estimate + current spend > $40.
  PER_MODEL_JSON="$(mktemp)"   # removed after the chunks (no new EXIT trap: on_exit above owns it)
  CALLS_ARGS=(); for m in "${ALL_MODELS[@]}"; do
    n=510; for e in "${EXPENSIVE[@]}"; do [ "$m" = "$e" ] && n=306; done; CALLS_ARGS+=("$m=$n"); done
  if [ "$DRY_RUN" = "1" ]; then
    MEASURED="${LOGGED_ESTIMATE:-21.50}"; echo "(dry run: nominal main estimate $MEASURED, no live prices)"
  else
    # Mean tokens over every returned pilot reply (N2); cap check on remaining cost + spend (N3); the operator's
    # LOGGED_ESTIMATE can only raise the figure (--floor). Prints the full-run estimate.
    FLOOR_ARGS=(); [ -n "$LOGGED_ESTIMATE" ] && FLOOR_ARGS=(--floor "$LOGGED_ESTIMATE")
    MEASURED="$("$PY" harness/run.py estimate-main --pilot "$RAW_DIR/${PILOT_RUN_ID}.jsonl" \
      --main-raw "$RAW_DIR/main1.jsonl" --calls "${CALLS_ARGS[@]}" ${FLOOR_ARGS[@]+"${FLOOR_ARGS[@]}"} \
      --per-model-out "$PER_MODEL_JSON")"
  fi
  LOGGED_ESTIMATE="$MEASURED"
  echo "-- Main estimate from $PILOT_RUN_ID measured tokens: \$$LOGGED_ESTIMATE (2142 calls)"
  if [ "$DRY_RUN" != "1" ]; then
    if [ "${CONFIRM_MAIN:-}" != "$LOGGED_ESTIMATE" ]; then
      echo "REFUSED: confirm the estimate first: rerun with CONFIRM_MAIN=$LOGGED_ESTIMATE" >&2
      exit 1
    fi
  fi
fi

# 3. Harness runs.
if [ "$CMD" = "pilot" ]; then
  # No --logged-estimate: the harness $5 pilot limit applies (results/c2_cost_estimate.md, pilot1b plan).
  "$PY" harness/run.py --run-id "$PILOT_RUN_ID" --cases "${PILOT_CASES[@]}" --conditions B \
    --models "${ALL_MODELS[@]}" --seeds 1 --temperature "$TEMP" --max-tokens "$MAX_TOKENS" \
    --out-dir "$RAW_DIR" ${DRYFLAG[@]+"${DRYFLAG[@]}"} ${RETRYFLAG[@]+"${RETRYFLAG[@]}"}
else
  RUN_ID=main1
  # Chunks: harness takes one models/seeds/conditions set per call, and notes exist for V on all 22 cases
  # but N, B, S, E only on the 20 non-A5 cases. Same run id, so reruns resume. Expensive tier: seeds 1-3;
  # cheap tier: seeds 1-5 (2142 calls). Each chunk's logged estimate is its share of the total
  # (tier share of the per-model estimate-main figures, by notes/102; falls back to the ratio 12.8/21.5 vs
  # 8.7/21.5 only in a dry run, which has no live prices), and spend is logged between chunks so
  # the harness ledger guard sees fresh spend.
  ALL_CASES=(data/cases/*.json)
  NON_A5=(); for c in "${ALL_CASES[@]}"; do case "$c" in *A5_*) ;; *) NON_A5+=("$c") ;; esac; done
  chunk() {  # name cases_var conditions tier(exp|cheap) nnotes
    local name="$1" cases_var="$2" conds="$3" tier="$4" notes="$5" seeds models share est
    local -n cases_ref="$cases_var"
    if [ "$tier" = exp ]; then models=("${EXPENSIVE[@]}"); seeds=(1 2 3); else models=("${CHEAP[@]}"); seeds=(1 2 3 4 5); fi
    est="$("$PY" - "$LOGGED_ESTIMATE" "$tier" "$notes" "${PER_MODEL_JSON:-}" "${EXPENSIVE[*]}" <<'PYE'
import sys, json
e, tier, n, path, exp = float(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5].split()
frac = 12.8 / 21.5 if tier == "exp" else 8.7 / 21.5   # fallback (dry run)
try:
    per = json.load(open(path))
    tot = sum(per.values())
    if tot > 0:
        frac = sum(v for m, v in per.items() if (m in exp) == (tier == "exp")) / tot
except (OSError, ValueError):
    pass
print(round(e * frac * n / 102, 4))
PYE
)"
    echo "-- chunk $name: logged estimate \$$est"
    # shellcheck disable=SC2086
    "$PY" harness/run.py --run-id "$RUN_ID" --cases "${cases_ref[@]}" --conditions $conds \
      --models "${models[@]}" --seeds "${seeds[@]}" --temperature "$TEMP" --max-tokens "$MAX_TOKENS" \
      --out-dir "$RAW_DIR" --logged-estimate "$est" --run-estimate "$LOGGED_ESTIMATE" \
      ${DRYFLAG[@]+"${DRYFLAG[@]}"} ${RETRYFLAG[@]+"${RETRYFLAG[@]}"}
    # Fatal if the ledger cannot be written or spend is at the cap (exit 2): set -e stops the pipeline.
    if [ "$DRY_RUN" != "1" ]; then "$PY" scripts/spend.py --log "after chunk $name (run_pipeline.sh)"; fi
  }
  chunk V_expensive ALL_CASES "V" exp 22
  chunk V_cheap ALL_CASES "V" cheap 22
  chunk NBSE_expensive NON_A5 "N B S E" exp 80
  chunk NBSE_cheap NON_A5 "N B S E" cheap 80
  rm -f "$PER_MODEL_JSON"
fi

# 4. Spend after.
if [ "$DRY_RUN" != "1" ]; then
  "$PY" scripts/spend.py --log "after $CMD (run_pipeline.sh)"
fi

# 5. Analysis.
echo "-- analysis"
if [ "$DRY_RUN" = "1" ]; then
  echo "(dry run: dry_run records are excluded from scoring by design; running score.py to confirm it only reads $RAW_DIR)"
  "$PY" analysis/score.py --raw "$RAW_DIR/*.jsonl" --out "$DERIVED" || echo "(score.py found nothing to score, expected for a dry run)"
elif [ "$CMD" = "pilot" ]; then
  # Pilot records are scored separately (results/derived/pilot/) and never reach paper outputs.
  "$PY" analysis/score.py --set pilot
  "$PY" analysis/fit_wbayes.py --set pilot
else
  "$PY" analysis/score.py --set pilot
  "$PY" analysis/score.py
  "$PY" analysis/fit_wbayes.py
  "$PY" analysis/make_tables.py
  "$PY" analysis/compare_clinicians.py
  "$PY" analysis/make_figures.py
fi
echo "== done. Commit with: git add results/raw results/derived paper/generated paper/figures results/spend_ledger.csv && git commit"
