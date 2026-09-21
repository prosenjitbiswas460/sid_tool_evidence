#!/usr/bin/env bash
# End-to-end: SID shortlist → LLM vs random shortlist → LLM.
#
# Design
#   Universe U of size M (GT + fillers). LLM can only rank L=20.
#   Arms:
#     rand20_hm  — random 20 from U → EAA force H,M (no SID merge)
#     sid20_hm   — SID top-20 from U → EAA force H,M (no SID merge)
#     sid20_hmt  — SID top-20 → EAA force H,M,S_tool + merge/fallback
#   Miss (GT not in shortlist) ⇒ HR/NDCG = 0 under standard eval.
#
# Usage (Ollama serving the model):
#   DATASET=amazon bash scripts/run_sid_e2e_shortlist_eval.sh
#   DATASET=yelp   bash scripts/run_sid_e2e_shortlist_eval.sh
#   DATASET=goodreads bash scripts/run_sid_e2e_shortlist_eval.sh
#
# Env:
#   DATASET        amazon|yelp|goodreads (default amazon)
#   DATA_DIR ARTIFACT_DIR OUTPUT_ROOT RESULTS_ROOT
#   OLLAMA_MODEL   default qwen2.5:7b-instruct
#   UNIVERSE_SIZES default "100 200 500"
#   MAX_TASKS      default 200
#   FORCE_BUILD=1  rebuild shortlist tasks
#   FORCE_RERUN=1  redo LLM evals even if predictions exist
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATASET="${DATASET:-amazon}"
DATA_DIR="${DATA_DIR:-$ROOT/websocietysimulator/processed_data}"
if [[ -z "${ARTIFACT_DIR:-}" ]]; then
  if [[ "$DATASET" == "yelp" ]]; then
    ARTIFACT_DIR="$ROOT/websocietysimulator/artifacts/yelp_sid_v2_fixed"
  else
    ARTIFACT_DIR="$ROOT/websocietysimulator/artifacts/${DATASET}_sid_v2"
  fi
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/sid_e2e_shortlist}"
TASK_ROOT="${OUTPUT_ROOT}/${DATASET}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
UNIVERSE_SIZES="${UNIVERSE_SIZES:-100 200 500}"
MAX_TASKS="${MAX_TASKS:-400}"

MODEL_TAG="${OLLAMA_MODEL//[:\/]/_}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/sid_e2e_shortlist/${DATASET}/${MODEL_TAG}}"

echo "== SID e2e shortlist eval =="
echo "  dataset=$DATASET  model=$OLLAMA_MODEL  M=($UNIVERSE_SIZES)"
echo "  tasks=$TASK_ROOT"
echo "  results=$RESULTS_ROOT"

need_build=0
if [[ "${FORCE_BUILD:-0}" == "1" ]]; then
  need_build=1
else
  for M in $UNIVERSE_SIZES; do
    [[ -f "$TASK_ROOT/m${M}/meta.json" ]] || need_build=1
  done
fi

# Backward compat: Amazon tasks previously lived at outputs/sid_e2e_shortlist/m{M}
if [[ "$need_build" == "1" && "$DATASET" == "amazon" ]]; then
  legacy_ok=1
  for M in $UNIVERSE_SIZES; do
    [[ -f "$OUTPUT_ROOT/m${M}/meta.json" ]] || legacy_ok=0
  done
  if [[ "$legacy_ok" == "1" && "${FORCE_BUILD:-0}" != "1" ]]; then
    echo "== Using legacy Amazon shortlists under $OUTPUT_ROOT/m* =="
    TASK_ROOT="$OUTPUT_ROOT"
    need_build=0
  fi
fi

if [[ "$need_build" == "1" ]]; then
  echo "== Building shortlist task dirs for $DATASET =="
  OMP_NUM_THREADS=1 python scripts/build_sid_e2e_shortlist_tasks.py \
    --data_dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --artifact_dir "$ARTIFACT_DIR" \
    --output_root "$OUTPUT_ROOT" \
    --universe_sizes $UNIVERSE_SIZES \
    --max_tasks "$MAX_TASKS"
  TASK_ROOT="${OUTPUT_ROOT}/${DATASET}"
else
  echo "== Reusing existing shortlist tasks under $TASK_ROOT =="
fi

run_arm () {
  local M="$1" ARM="$2" FORCE="$3" NOMERGE="$4"
  local MANIFEST="$TASK_ROOT/m${M}/arms/${ARM}/manifest.json"
  local OUT_TAG="sid_e2e_${DATASET}_m${M}_${ARM}_${MODEL_TAG}"
  local PRED="$ROOT/outputs/${OUT_TAG}_predictions.json"

  # Legacy Amazon pred name (no dataset infix)
  local LEGACY_PRED=""
  if [[ "$DATASET" == "amazon" ]]; then
    LEGACY_PRED="$ROOT/outputs/sid_e2e_m${M}_${ARM}_${MODEL_TAG}_predictions.json"
  fi

  mkdir -p "$RESULTS_ROOT/m${M}" "$ROOT/outputs"

  if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: missing $MANIFEST — run build step first" >&2
    exit 1
  fi

  if [[ -f "$PRED" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "[skip] $PRED exists (FORCE_RERUN=1 to redo)"
    return 0
  fi
  if [[ -n "$LEGACY_PRED" && -f "$LEGACY_PRED" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "[skip] legacy $LEGACY_PRED exists (FORCE_RERUN=1 to redo)"
    return 0
  fi

  echo "== Eval dataset=$DATASET M=$M arm=$ARM modules=$FORCE nomerge=$NOMERGE model=$OLLAMA_MODEL =="
  local EXTRA=()
  if [[ "$NOMERGE" == "1" ]]; then
    EXTRA+=(--eaa_no_sid_merge)
  fi

  python scripts/run_condition_eval.py \
    --agent eaa \
    --manifest "$MANIFEST" \
    --condition classic \
    --data_dir "$DATA_DIR" \
    --artifact_dir "$ARTIFACT_DIR" \
    --llm_class OllamaLLM \
    --ollama_model "$OLLAMA_MODEL" \
    --eaa_force_modules "$FORCE" \
    "${EXTRA[@]}" \
    --predictions_dir "$ROOT/outputs" \
    --predictions_output "$PRED" \
    --results_dir "$RESULTS_ROOT/m${M}" \
    --output "$RESULTS_ROOT/m${M}/${ARM}_eval.json" \
    2>&1 | tee "$RESULTS_ROOT/m${M}/${ARM}_run.log"
}

for M in $UNIVERSE_SIZES; do
  run_arm "$M" rand20_hm "H,M" 1
  run_arm "$M" sid20_hm  "H,M" 1
  run_arm "$M" sid20_hmt "H,M,S_tool" 0
done

echo "== Aggregate + bootstrap significance =="
OMP_NUM_THREADS=1 python scripts/analyze_sid_e2e_shortlist.py \
  --dataset "$DATASET" \
  --output_root "$OUTPUT_ROOT" \
  --results_root "$RESULTS_ROOT" \
  --predictions_dir "$ROOT/outputs" \
  --ollama_model "$OLLAMA_MODEL" \
  --universe_sizes $UNIVERSE_SIZES

echo ""
echo "Primary compelling contrast: sid20_hm vs rand20_hm (same LLM; only shortlist differs)."
echo "Look for HELPS* on HR@5 / NDCG@5 in $RESULTS_ROOT/pairwise_bootstrap.csv"
