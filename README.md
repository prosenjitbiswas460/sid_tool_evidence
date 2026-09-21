# SID Tool Evidence

Evaluate when semantic-ID (SID) scores help an LLM recommender: as extra
evidence on a fixed candidate list (HM vs HMT), or as a shortlister before
ranking.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

Processed catalogs go in `websocietysimulator/processed_data/{item,review,user}.json`:

```bash
python data_process.py --input_dir "$RAW_DIR" --output_dir websocietysimulator/processed_data
```

## SID artifacts and task manifests

```bash
python scripts/build_sid_v2.py \
  --item_json websocietysimulator/processed_data/item.json \
  --review_json websocietysimulator/processed_data/review.json \
  --output_dir websocietysimulator/artifacts/amazon_sid_v2 \
  --source amazon

for ds in amazon goodreads yelp; do
  python scripts/build_condition_manifest.py \
    --data_dir websocietysimulator/processed_data --dataset "$ds" --track 2
done
python scripts/patch_evolving_interest_last_n_manifest.py \
  --manifest manifests/goodreads_track2.json
```

Yelp codebooks should be trained with `--kmeans_init --ema --revive_dead`.

## Ranking (HM / HMT)

```bash
MODEL=<ollama-tag> PREFIX=<run-id> DATASETS="amazon goodreads yelp" \
  bash scripts/run_hm_hmt.sh

python scripts/analyze_hmt_significance.py \
  --predictions_dirs outputs --datasets amazon goodreads yelp \
  --backbones 7b:<run-id> \
  --output_dir results
```

HM uses `--eaa_force_modules H,M`. HMT uses `H,M,S_tool`.
Prompt-only HMT: `bash scripts/run_prompt_only.sh`.
Title-only history: `bash scripts/run_titles_only_oracles.sh`.

## Shortlisting and plots

```bash
DATASET=amazon bash scripts/run_sid_e2e_shortlist_eval.sh
python scripts/generate_figures.py
```

Plots are written to `docs/figures/`.
