#!/usr/bin/env bash
set -e

source scripts/env_lzhy03.sh
source configs.sh

# 先找一个 MLVU split 文件做流程测试
IN_JSON=$MLVU_ROOT/splits/part6_chunks/chunk_000.json

if [ ! -f "$IN_JSON" ]; then
  echo "Cannot find $IN_JSON"
  echo "Try:"
  find "$MLVU_ROOT" -name "*.json" | head -20
  exit 1
fi

echo "===== normalize dataset ====="
python datasets/load_mcq.py \
  --input "$IN_JSON" \
  --dataset mlvu \
  --output outputs/smoke/mlvu_part6_chunk000_norm.json \
  --limit 30

echo "===== run direct ====="
python baselines/simple_rag_baselines.py \
  --input outputs/smoke/mlvu_part6_chunk000_norm.json \
  --method direct \
  --output outputs/smoke/direct/output.json

echo "===== run llovi_lite ====="
python baselines/simple_rag_baselines.py \
  --input outputs/smoke/mlvu_part6_chunk000_norm.json \
  --method llovi_lite \
  --output outputs/smoke/llovi_lite/output.json \
  --topk 4

echo "===== run drvideo_lite ====="
python baselines/simple_rag_baselines.py \
  --input outputs/smoke/mlvu_part6_chunk000_norm.json \
  --method drvideo_lite \
  --output outputs/smoke/drvideo_lite/output.json \
  --topk 4

echo "===== run videorag_lite ====="
python baselines/simple_rag_baselines.py \
  --input outputs/smoke/mlvu_part6_chunk000_norm.json \
  --method videorag_lite \
  --output outputs/smoke/videorag_lite/output.json \
  --topk 4

echo "===== run sv_videorag ====="
python sv_videorag/run_sv_videorag.py \
  --input outputs/smoke/mlvu_part6_chunk000_norm.json \
  --output outputs/smoke/sv_videorag/output.json \
  --topk 4 \
  --max_rounds 3 \
  --threshold 0.62 \
  --margin 0.06

echo "===== compare ====="
python analysis/compare_outputs.py \
  --files \
  outputs/smoke/direct/output.json \
  outputs/smoke/llovi_lite/output.json \
  outputs/smoke/drvideo_lite/output.json \
  outputs/smoke/videorag_lite/output.json \
  outputs/smoke/sv_videorag/output.json \
  --output outputs/smoke/summary.json
