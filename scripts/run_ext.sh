#!/usr/bin/env bash
# One command per stage for the committee-extension batches (P1 to P4, results/ext_cost_estimate.md).
#   scripts/run_ext.sh pilot BATCH     # 3 cases x 1 sample x 5 models = 15 calls (run id BATCH_pilot)
#   scripts/run_ext.sh main  BATCH     # main run (run id BATCH); needs the pilot gate and CONFIRM_EXT=<estimate>
# BATCH (ext6_para1/ext6_para2: paraphrased v1 prompts on N, 20 cases x 3 seeds; one run id each because keys omit prompt version): ext1_pretest | ext2_belief | ext3_long | ext4_unnamed | ext5_dsretry (no pilot; same prompt and model as main1)
# Env: DRY_RUN=1 RAW_DIR=/scratch (no network, no spend), RETRY_ERRORS=1, CONFIRM_EXT=<estimate>, LOGGED_ESTIMATE=<usd>
#      (can only raise the estimate), EXT4_REDUCED=1 (ext4 on L only), ACK_ABORT=1, MAX_TOKENS (default 8000),
#      PILOT_RUN_ID (default BATCH_pilot; set a new id, e.g. ext2_belief_pilot2, to redo a failed pilot; main reads the
#      same variable), PYTHON.
# Never edits results/raw/ (the harness only appends; reruns resume per key). Failures of spend.py are fatal.
set -euo pipefail
if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; then
  echo "REFUSED: bash >= 4.4 required." >&2; exit 1
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
RAW_DIR="${RAW_DIR:-$ROOT/results/raw}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
ACK_ABORT="${ACK_ABORT:-0}"
LOGGED_ESTIMATE="${LOGGED_ESTIMATE:-}"
EXT4_REDUCED="${EXT4_REDUCED:-0}"
MAX_TOKENS="${MAX_TOKENS:-8000}"     # (main1 used 4000 and lost 21 DeepSeek replies to truncation)
TEMP=1.0
PILOT_CEILING=1.00                   # logged estimate for a 15-call pilot (planned <= $0.15); the ledger is above
                                     # the harness's $5 no-estimate limit, so pilots pass an explicit estimate
MODELS=(openai/gpt-6.1-sol anthropic/claude-sonnet-5.5 google/gemini-3.8-flash deepseek/deepseek-v4-pro openai/gpt-4o)
PILOT_CASES=(data/cases/A2_breast_pos.json data/cases/A3_cardiac_neg.json data/cases/B2_pe_mod_neg.json)
EXT7_CASES=(data/cases/B7_strep_pos.json data/cases/B7_strep_neg.json data/cases/B8_septic_pos.json data/cases/B8_septic_neg.json)
# B7/B8 (added Nov 17) are excluded from the original-set batches; they run under their own run id.
NON_A5=(); for c in data/cases/*.json; do case "$c" in *A5_*|*B7_*|*B8_*) ;; *) NON_A5+=("$c") ;; esac; done
ALL_CASES=(); for c in data/cases/*.json; do case "$c" in *B7_*|*B8_*) ;; *) ALL_CASES+=("$c") ;; esac; done

usage() { echo "usage: $0 {pilot|main} {ext1_pretest|ext2_belief|ext3_long|ext4_unnamed|ext5_dsretry|ext6_para1|ext6_para2|ext7_newcases|ext8_family}" >&2; exit 64; }
[ $# -eq 2 ] || usage
CMD="$1"; BATCH="$2"
case "$CMD" in pilot|main) ;; *) usage ;; esac
if [ -n "$LOGGED_ESTIMATE" ] && ! [[ "$LOGGED_ESTIMATE" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  echo "REFUSED: LOGGED_ESTIMATE is not a number." >&2; exit 1
fi

# Batch table: prompt version | conditions (chunks separated by ';', each "COND@notes_dir") | extra flags | pilot condition chunk
case "$BATCH" in
  ext1_pretest) VERSION=v2_pretest_only; CHUNKS="V N S@data/notes/pretest"
                EXTRA=(--collapse-polarity); PILOT_CHUNK="N@data/notes/pretest"; KEYS_PER_MODEL=$((10*3*3)) ;;
  ext2_belief)  VERSION=v3_believed_accuracy; CHUNKS="N@data/notes"; EXTRA=(); PILOT_CHUNK="N@data/notes"
                KEYS_PER_MODEL=$((20*3)) ;;
  ext3_long)    VERSION=v1; CHUNKS="L@data/notes/long"; EXTRA=(); PILOT_CHUNK="L@data/notes/long"; KEYS_PER_MODEL=$((20*3)) ;;
  ext4_unnamed) VERSION=v4_unnamed_test; CHUNKS="N@data/notes;L@data/notes/long"; EXTRA=()
                PILOT_CHUNK="L@data/notes/long"; KEYS_PER_MODEL=$((40*3))
                if [ "$EXT4_REDUCED" = "1" ]; then CHUNKS="L@data/notes/long"; KEYS_PER_MODEL=$((20*3)); fi ;;
  ext6_para1)   VERSION=v5_para1; CHUNKS="N@data/notes"; EXTRA=(); PILOT_CHUNK="N@data/notes"; KEYS_PER_MODEL=$((20*3)) ;;
  ext6_para2)   VERSION=v6_para2; CHUNKS="N@data/notes"; EXTRA=(); PILOT_CHUNK="N@data/notes"; KEYS_PER_MODEL=$((20*3)) ;;
  # B7/B8 only. Chunk form COND@notes_dir[@prompt_version[@collapse]]; the blind pretest chunk follows ext1_pretest.
  ext7_newcases) VERSION=v1; CHUNKS="V B N S E@data/notes;V N S@data/notes/pretest@v2_pretest_only@collapse"; EXTRA=()
                PILOT_CHUNK="N@data/notes"; KEYS_PER_MODEL=$((4*5*3 + 2*3*3)) ;;
  # one extra model family (single-model override of MODELS). V on the 22 original cases, N and E on the 20 non-A5 cases,
  # exactly as main1 ran them; 22*3 + 40*3 = 186 keys. The pilot is 3 cases x 1 seed on N, so the gate expects 3 keys, >= 2 ok.
  ext8_family)  VERSION=v1; CHUNKS="V@data/notes;N E@data/notes"; EXTRA=(); PILOT_CHUNK="N@data/notes"
                MODELS=("${EXT8_MODEL:-qwen/qwen4-max}"); GATE_ARGS=(--expected 3 --min-ok 2); KEYS_PER_MODEL=$((22*3 + 40*3)) ;;
  ext5_dsretry) VERSION=v1; CHUNKS="V B N S E@data/notes"; MAX_TOKENS=16000
                EXTRA=(--keys-file results/retry_specs/ext5_dsretry_keys.jsonl --retry-of main1)
                PILOT_CHUNK=""; KEYS_PER_MODEL=0 ;;
  *) usage ;;
esac
if [ "$BATCH" = "ext5_dsretry" ] && [ "$CMD" = "pilot" ]; then
  echo "REFUSED: ext5_dsretry has no pilot (same prompt and model as main1). Run: $0 main ext5_dsretry" >&2; exit 1
fi

mkdir -p "$RAW_DIR"; RAW_DIR="$(cd "$RAW_DIR" && pwd)"
if [ "$DRY_RUN" = "1" ] && [ "$RAW_DIR" = "$ROOT/results/raw" ]; then
  echo "REFUSED: DRY_RUN=1 must use RAW_DIR outside results/raw." >&2; exit 1
fi
GATE_ARGS=(${GATE_ARGS[@]+"${GATE_ARGS[@]}"})
DRYFLAG=(); [ "$DRY_RUN" = "1" ] && DRYFLAG=(--dry-run)
RETRYFLAG=(); [ "$RETRY_ERRORS" = "1" ] && RETRYFLAG=(--retry-errors)
if command -v flock >/dev/null 2>&1; then
  exec 9>"$ROOT/.pipeline.lock"
  flock -n 9 || { echo "REFUSED: another pipeline holds $ROOT/.pipeline.lock" >&2; exit 1; }
fi

# Notes must be current and pass their checks before any call.
"$PY" data/notes/check_notes.py >/dev/null || { echo "REFUSED: data/notes/check_notes.py failed." >&2; exit 1; }
case "$BATCH" in
  ext1_pretest) "$PY" data/notes/check_ext.py pretest >/dev/null || { echo "REFUSED: check_ext.py pretest failed." >&2; exit 1; } ;;
  ext7_newcases) "$PY" data/notes/check_ext.py pretest >/dev/null || { echo "REFUSED: check_ext.py pretest failed." >&2; exit 1; } ;;
  ext3_long|ext4_unnamed) "$PY" data/notes/check_ext.py long >/dev/null || { echo "REFUSED: check_ext.py long failed." >&2; exit 1; } ;;
esac

PILOT_RUN_ID="${PILOT_RUN_ID:-${BATCH}_pilot}"
case "$PILOT_RUN_ID" in *[!A-Za-z0-9_]*|"") echo "REFUSED: PILOT_RUN_ID must be alphanumeric/underscore." >&2; exit 1 ;; esac
RUN_ID="$BATCH"; [ "$CMD" = "pilot" ] && RUN_ID="$PILOT_RUN_ID"
echo "== run_ext: $CMD $BATCH (run id $RUN_ID, prompt $VERSION, max tokens $MAX_TOKENS, dry_run=$DRY_RUN, raw $RAW_DIR)"

if [ "$CMD" = "main" ] && [ "$BATCH" != "ext5_dsretry" ]; then
  GF=(); [ "$DRY_RUN" = "1" ] && GF=(--allow-dry)
  if [ ! -s "$RAW_DIR/${PILOT_RUN_ID}.jsonl" ] || ! "$PY" harness/run.py gate --pilot "$RAW_DIR/${PILOT_RUN_ID}.jsonl" \
       --models "${MODELS[@]}" ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} ${GF[@]+"${GF[@]}"}; then
    echo "REFUSED: pilot $PILOT_RUN_ID missing or failed the gate (>= 12 of 15 ok, every model >= 1 ok). See." >&2
    exit 1
  fi
fi
if [ "$CMD" = "main" ] && [ -s "$RAW_DIR/${RUN_ID}.abort.txt" ] && [ "$ACK_ABORT" != "1" ]; then
  echo "REFUSED: $RAW_DIR/${RUN_ID}.abort.txt exists. Read it, fix the cause, then ACK_ABORT=1." >&2; exit 1
fi

if [ "$DRY_RUN" != "1" ]; then
  "$PY" scripts/spend.py || { echo "KEY CHECK FAILED (401 = no key; exit 2 = at the cap). No calls made." >&2; exit 1; }
  "$PY" scripts/spend.py --log "before $CMD $BATCH (run_ext.sh)" || { echo "LEDGER LOG FAILED." >&2; exit 1; }
  on_exit() { local rc=$?; trap - EXIT; "$PY" scripts/spend.py --log "after $CMD $BATCH exit=$rc (run_ext.sh)" || echo "WARNING: ledger log failed" >&2; exit $rc; }
  trap on_exit EXIT
fi

# Estimate.
if [ "$CMD" = "pilot" ]; then
  EST="$PILOT_CEILING"
elif [ "$BATCH" = "ext5_dsretry" ]; then
  EST="${LOGGED_ESTIMATE:-0.70}"    # worst case 21 x 16,000 output tokens at list price = $0.65 (ext_cost_estimate.md)
else
  CALLS_ARGS=(); for m in "${MODELS[@]}"; do CALLS_ARGS+=("$m=$KEYS_PER_MODEL"); done
  if [ "$DRY_RUN" = "1" ]; then EST="${LOGGED_ESTIMATE:-5.00}"; else
    FLOOR=(); [ -n "$LOGGED_ESTIMATE" ] && FLOOR=(--floor "$LOGGED_ESTIMATE")
    EST="$("$PY" harness/run.py estimate-main --pilot "$RAW_DIR/${PILOT_RUN_ID}.jsonl" --main-raw "$RAW_DIR/${RUN_ID}.jsonl" \
          --calls "${CALLS_ARGS[@]}" ${FLOOR[@]+"${FLOOR[@]}"})"
  fi
fi
echo "-- estimate for $RUN_ID: \$$EST"
if [ "$CMD" = "main" ] && [ "$DRY_RUN" != "1" ]; then
  if [ "${CONFIRM_EXT:-}" != "$EST" ]; then
    echo "REFUSED: confirm the estimate first: rerun with CONFIRM_EXT=$EST" >&2
    exit 1
  fi
fi

# Harness.
IFS=';' read -r -a CH <<< "$([ "$CMD" = "pilot" ] && echo "$PILOT_CHUNK" || echo "$CHUNKS")"
NCH=${#CH[@]}
for chunk in "${CH[@]}"; do
  IFS='@' read -r conds notes cver ccol <<< "$chunk"
  cver="${cver:-$VERSION}"; cextra=(${EXTRA[@]+"${EXTRA[@]}"}); [ "${ccol:-}" = "collapse" ] && cextra+=(--collapse-polarity)
  if [ "$CMD" = "pilot" ] && [ "$BATCH" = "ext7_newcases" ]; then cases=("${EXT7_CASES[@]:0:3}"); seeds=(1); models=("${MODELS[@]}")
  elif [ "$CMD" = "pilot" ]; then cases=("${PILOT_CASES[@]}"); seeds=(1); models=("${MODELS[@]}")
  elif [ "$BATCH" = "ext8_family" ]; then
    if [ "$conds" = "V" ]; then cases=("${ALL_CASES[@]}"); else cases=("${NON_A5[@]}"); fi; seeds=(1 2 3); models=("${MODELS[@]}")
  elif [ "$BATCH" = "ext7_newcases" ]; then cases=("${EXT7_CASES[@]}"); seeds=(1 2 3); models=("${MODELS[@]}")
  elif [ "$BATCH" = "ext5_dsretry" ]; then cases=("${ALL_CASES[@]}"); seeds=(1 2 3 4 5); models=(deepseek/deepseek-v4-pro)
  else cases=("${NON_A5[@]}"); seeds=(1 2 3); models=("${MODELS[@]}"); fi
  chunk_est="$("$PY" -c "print(round($EST / $NCH, 4))")"
  echo "-- chunk $conds from $notes (logged estimate \$$chunk_est of run estimate \$$EST)"
  # shellcheck disable=SC2086
  "$PY" harness/run.py --run-id "$RUN_ID" --prompt-version "$cver" --cases "${cases[@]}" --conditions $conds \
    --models "${models[@]}" --seeds "${seeds[@]}" --temperature "$TEMP" --max-tokens "$MAX_TOKENS" \
    --notes-dir "$notes" --out-dir "$RAW_DIR" --logged-estimate "$chunk_est" --run-estimate "$EST" \
    ${cextra[@]+"${cextra[@]}"} ${DRYFLAG[@]+"${DRYFLAG[@]}"} ${RETRYFLAG[@]+"${RETRYFLAG[@]}"}
  if [ "$DRY_RUN" != "1" ]; then "$PY" scripts/spend.py --log "after $RUN_ID chunk $conds (run_ext.sh)"; fi
done

"$PY" scripts/ext_summary.py --raw "$RAW_DIR/${RUN_ID}.jsonl"
if [ "$CMD" = "pilot" ]; then
  GF=(); [ "$DRY_RUN" = "1" ] && GF=(--allow-dry)
  "$PY" harness/run.py gate --pilot "$RAW_DIR/${RUN_ID}.jsonl" --models "${MODELS[@]}" ${GATE_ARGS[@]+"${GATE_ARGS[@]}"} ${GF[@]+"${GF[@]}"} || {
    echo "PILOT GATE FAILED for $RUN_ID. Do not run main. See (failure steps)." >&2; exit 1; }
fi
echo "== done: $RUN_ID. Commit: git add results/raw/${RUN_ID}.jsonl results/spend_ledger.csv && git commit"
