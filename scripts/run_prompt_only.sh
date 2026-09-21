#!/usr/bin/env bash
# HMT with SID scores in the prompt only (no merge / fallback).
#
#   MODEL=<ollama-tag> PREFIX=<run-id> HM_PREFIX=<hm-run-id> \
#     DATASETS=amazon bash scripts/run_prompt_only.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-websocietysimulator/processed_data}"
MODEL="${MODEL:?set MODEL to an Ollama tag}"
PREFIX="${PREFIX:?set PREFIX}"
HM_PREFIX="${HM_PREFIX:-eaa}"
read -r -a DATASETS <<< "${DATASETS:-amazon}"
CONDS=(classic)
if [[ -n "${CONDS_OVERRIDE:-}" ]]; then
  read -r -a CONDS <<< "$CONDS_OVERRIDE"
fi

artifact_for() {
  case "$1" in
    amazon)    echo "websocietysimulator/artifacts/amazon_sid_v2" ;;
    goodreads) echo "websocietysimulator/artifacts/goodreads_sid_v2" ;;
    yelp)      echo "websocietysimulator/artifacts/yelp_sid_v2_fixed" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

for ds in "${DATASETS[@]}"; do
  art="$(artifact_for "$ds")"
  man="manifests/${ds}_track2.json"
  [[ -f "$man" ]] || { echo "MISSING $man" >&2; exit 1; }
  for c in "${CONDS[@]}"; do
    out="outputs/${PREFIX}_oracle_hmt_${ds}_track2_${c}_predictions.json"
    metrics="results/${PREFIX}_oracle_hmt_${ds}_track2_${c}.json"
    if [[ -f "$out" ]]; then
      echo "skip exists: $out"
      continue
    fi
    echo "[start] $out"
    python scripts/run_condition_eval.py \
      --data_dir "$DATA" \
      --manifest "$man" \
      --condition "$c" \
      --agent eaa \
      --artifact_dir "$art" \
      --llm_class OllamaLLM \
      --ollama_model "$MODEL" \
      --simulator_device gpu \
      --eaa_force_modules "H,M,S_tool" \
      --eaa_no_sid_merge \
      --output "$metrics" \
      --predictions_output "$out" \
      --predictions_dir outputs
    echo "[done] $out"
  done
done

echo "Compare against HM with:"
echo "  python scripts/analyze_hmt_significance.py \\"
echo "    --predictions_dirs outputs --datasets ${DATASETS[*]} \\"
echo "    --pairs ${PREFIX}_oracle_hmt,${HM_PREFIX}_oracle_hm \\"
echo "    --output_dir results"
