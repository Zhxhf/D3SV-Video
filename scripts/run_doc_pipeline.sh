#!/usr/bin/env bash
set -e

source scripts/env_lzhy03.sh
source configs.sh

IN_JSON=${1:-$MLVU_ROOT/splits/part6_chunks/chunk_000.json}
DOCS_JSON=${2:-}
LIMIT=${3:-50}
DATASET=${4:-mlvu}

echo "IN_JSON=$IN_JSON"
echo "DOCS_JSON=$DOCS_JSON"
echo "LIMIT=$LIMIT"
echo "DATASET=$DATASET"

mkdir -p outputs/doc_pipeline logs

python datasets/load_mcq.py \
  --input "$IN_JSON" \
  --dataset "$DATASET" \
  --output outputs/doc_pipeline/input_norm.json \
  --limit "$LIMIT"

for METHOD in direct_doc llovi_lite drvideo_lite videorag_lite sv_videorag
do
  echo "===== run $METHOD ====="
  python sv_videorag/run_doc_methods.py \
    --input outputs/doc_pipeline/input_norm.json \
    --method "$METHOD" \
    --docs_json "$DOCS_JSON" \
    --output outputs/doc_pipeline/$METHOD/output.json \
    --topk 4 \
    --max_rounds 3 \
    --threshold 0.62 \
    --margin 0.06
done

echo "===== compare ====="
python analysis/compare_outputs.py \
  --files \
  outputs/doc_pipeline/direct_doc/output.json \
  outputs/doc_pipeline/llovi_lite/output.json \
  outputs/doc_pipeline/drvideo_lite/output.json \
  outputs/doc_pipeline/videorag_lite/output.json \
  outputs/doc_pipeline/sv_videorag/output.json \
  --output outputs/doc_pipeline/summary.json
