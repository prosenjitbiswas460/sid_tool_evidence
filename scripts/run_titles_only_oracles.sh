#!/usr/bin/env bash
# HM / HMT with user history as item titles (no review text).
#
#   MODEL=<ollama-tag> PREFIX=<run-id> bash scripts/run_titles_only_oracles.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-websocietysimulator/processed_data}"
MODEL="${MODEL:?set MODEL to an Ollama tag}"
PREFIX="${PREFIX:-eaa_nr}"
read -r -a DATASETS <<< "${DATASETS:-amazon goodreads yelp}"
CONDS=(classic cold_start_user cold_start_item evolving_interest)

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
    for role in hm hmt; do
      if [[ "$role" == hm ]]; then mods="H,M"; else mods="H,M,S_tool"; fi
      out="outputs/${PREFIX}_oracle_${role}_${ds}_track2_${c}_predictions.json"
      metrics="results/${PREFIX}_oracle_${role}_${ds}_track2_${c}.json"
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
        --evidence_no_reviews \
        --eaa_force_modules "$mods" \
        --output "$metrics" \
        --predictions_output "$out" \
        --predictions_dir outputs
    done
  done
done
