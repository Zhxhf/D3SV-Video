#!/usr/bin/env bash

# ===== conda =====
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate uav

# ===== use GPU 1 by default =====
export CUDA_VISIBLE_DEVICES=1

# ===== project root =====
export LZHY03_ROOT=/home/ubuntu/videomind/VideoMind/LZHY03
export PYTHONPATH=$LZHY03_ROOT:$PYTHONPATH

# ===== reuse existing local models =====
export QWEN2VL_MODEL=/home/ubuntu/videomind/VideoMind/model_zoo/Qwen2-VL-2B-Instruct
export VIDEOMIND_MODEL=/home/ubuntu/videomind/VideoMind/model_zoo/VideoMind-2B

# ===== offline mode if local model is used =====
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ===== memory control =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== LZHY03 env loaded ====="
echo "python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LZHY03_ROOT=$LZHY03_ROOT"
echo "QWEN2VL_MODEL=$QWEN2VL_MODEL"
echo "VIDEOMIND_MODEL=$VIDEOMIND_MODEL"
