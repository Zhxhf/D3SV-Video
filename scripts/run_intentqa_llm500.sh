#!/usr/bin/env bash
set -e

source scripts/env_lzhy03.sh
source configs.sh

ANN=/home/ubuntu/videomind/VideoMind/datasets/intentqa/videomind_format/valid.covered.csv
DOCS=/home/ubuntu/videomind/VideoMind/datasets/intentqa/videomind_format/llava1.5_fps1.json

mkdir -p outputs/intentqa_llm500 logs

echo "===== normalize IntentQA ====="
python datasets/load_mcq.py \
  --input "$ANN" \
  --dataset intentqa \
  --output outputs/intentqa_llm500/input_norm.json \
  --limit 500

for METHOD in llovi_lite videorag_lite sv_videorag
do
  echo "===== run $METHOD with Qwen2-VL text answerer ====="
  python sv_videorag/run_llm_methods.py \
    --input outputs/intentqa_llm500/input_norm.json \
    --docs_json "$DOCS" \
    --method "$METHOD" \
    --output outputs/intentqa_llm500/$METHOD/output.json \
    --model_path "$MODEL_PATH" \
    --topk 4 \
    --max_rounds 3 \
    --threshold 0.62 \
    --margin 0.06
done

echo "===== compare ====="
python analysis/compare_outputs.py \
  --files \
  outputs/intentqa_llm500/llovi_lite/output.json \
  outputs/intentqa_llm500/videorag_lite/output.json \
  outputs/intentqa_llm500/sv_videorag/output.json \
  --output outputs/intentqa_llm500/summary.json
