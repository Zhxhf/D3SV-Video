#!/usr/bin/env bash
set -e

source scripts/env_lzhy03.sh
source configs.sh

DOCS=/home/ubuntu/videomind/VideoMind/datasets/intentqa/videomind_format/llava1.5_fps1.json
INPUT=outputs/intentqa_llm500/input_norm.json

if [ ! -f "$INPUT" ]; then
  echo "Missing $INPUT"
  echo "Please run intentqa_llm500 first."
  exit 1
fi

mkdir -p outputs/intentqa_llm500_sweep logs analysis

echo "===== SV-A: topk4 threshold0.68 margin0.04 ====="
python sv_videorag/run_llm_methods.py \
  --input "$INPUT" \
  --docs_json "$DOCS" \
  --method sv_videorag \
  --output outputs/intentqa_llm500_sweep/sv_t068_m004_k4/output.json \
  --model_path "$MODEL_PATH" \
  --topk 4 \
  --max_rounds 3 \
  --threshold 0.68 \
  --margin 0.04

echo "===== SV-B: topk5 threshold0.68 margin0.04 ====="
python sv_videorag/run_llm_methods.py \
  --input "$INPUT" \
  --docs_json "$DOCS" \
  --method sv_videorag \
  --output outputs/intentqa_llm500_sweep/sv_t068_m004_k5/output.json \
  --model_path "$MODEL_PATH" \
  --topk 5 \
  --max_rounds 3 \
  --threshold 0.68 \
  --margin 0.04

echo "===== compare all ====="
python analysis/compare_outputs.py \
  --files \
  outputs/intentqa_llm500/llovi_lite/output.json \
  outputs/intentqa_llm500/videorag_lite/output.json \
  outputs/intentqa_llm500/sv_videorag/output.json \
  outputs/intentqa_llm500_sweep/sv_t068_m004_k4/output.json \
  outputs/intentqa_llm500_sweep/sv_t068_m004_k5/output.json \
  --output outputs/intentqa_llm500_sweep/summary.json

echo "===== pair compare with Video-RAG-lite ====="
python analysis/compare_pair.py \
  --base outputs/intentqa_llm500/videorag_lite/output.json \
  --ours outputs/intentqa_llm500_sweep/sv_t068_m004_k4/output.json \
  --output analysis/intentqa_llm500_videorag_vs_sv_t068_m004_k4.json

python analysis/compare_pair.py \
  --base outputs/intentqa_llm500/videorag_lite/output.json \
  --ours outputs/intentqa_llm500_sweep/sv_t068_m004_k5/output.json \
  --output analysis/intentqa_llm500_videorag_vs_sv_t068_m004_k5.json

echo "===== pair compare with LLoVi-lite ====="
python analysis/compare_pair.py \
  --base outputs/intentqa_llm500/llovi_lite/output.json \
  --ours outputs/intentqa_llm500_sweep/sv_t068_m004_k4/output.json \
  --output analysis/intentqa_llm500_llovi_vs_sv_t068_m004_k4.json

python analysis/compare_pair.py \
  --base outputs/intentqa_llm500/llovi_lite/output.json \
  --ours outputs/intentqa_llm500_sweep/sv_t068_m004_k5/output.json \
  --output analysis/intentqa_llm500_llovi_vs_sv_t068_m004_k5.json

echo "===== short summaries ====="
python - <<'PY'
import json
for p in [
    "outputs/intentqa_llm500_sweep/summary.json",
    "analysis/intentqa_llm500_videorag_vs_sv_t068_m004_k4.json",
    "analysis/intentqa_llm500_videorag_vs_sv_t068_m004_k5.json",
    "analysis/intentqa_llm500_llovi_vs_sv_t068_m004_k4.json",
    "analysis/intentqa_llm500_llovi_vs_sv_t068_m004_k5.json",
]:
    print("\n###", p)
    x=json.load(open(p,"r",encoding="utf-8"))
    if isinstance(x, list):
        for r in x:
            print(r)
    else:
        print(x["summary"])
PY
