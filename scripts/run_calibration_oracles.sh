#!/usr/bin/env bash
# Matched-calibration HM / HMT (and optional adaptive) runs.
#
#   OLLAMA_MODEL=<tag> bash scripts/run_calibration_oracles.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-websocietysimulator/processed_data}"
MODEL="${OLLAMA_MODEL:?set OLLAMA_MODEL}"
RUN_SIGNALS="${RUN_SIGNALS:-1}"
CONDS=(classic cold_start_user cold_start_item evolving_interest)

run_one() {
  local ds="$1" art="$2" man="$3" c="$4"
  local tag="$5"
  shift 5
  local extra=("$@")
  local out="outputs/${tag}_${ds}_track2_${c}_predictions.json"
  local metrics="results/${tag}_${ds}_track2_${c}.json"
  if [[ -f "$out" ]]; then
    echo "skip exists: $out"
    return 0
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
    --output "$metrics" \
    --predictions_output "$out" \
    --predictions_dir outputs \
    "${extra[@]}"
  echo "[done] $out"
}

run_ds() {
  local ds="$1" art="$2"
  local man="manifests/calibration_matched/${ds}_track2_calibration_matched.json"
  if [[ ! -f "$man" ]]; then
    echo "MISSING manifest: $man" >&2
    return 1
  fi
  local c
  for c in "${CONDS[@]}"; do
    run_one "$ds" "$art" "$man" "$c" "eaa_calm_oracle_hm"  --eaa_force_modules "H,M"
    run_one "$ds" "$art" "$man" "$c" "eaa_calm_oracle_hmt" --eaa_force_modules "H,M,S_tool"
    if [[ "$RUN_SIGNALS" == "1" ]]; then
      run_one "$ds" "$art" "$man" "$c" "eaa_calm" --eaa_policy quality
    fi
  done
}

echo "[info] calibration oracles at $(date)  RUN_SIGNALS=$RUN_SIGNALS"
run_ds amazon    websocietysimulator/artifacts/amazon_sid_v2
run_ds goodreads websocietysimulator/artifacts/goodreads_sid_v2
run_ds yelp      websocietysimulator/artifacts/yelp_sid_v2_fixed
echo "[done] $(date)"
