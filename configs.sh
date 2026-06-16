# ===== common paths =====
export ROOT=/home/ubuntu/videomind/VideoMind/LZHY03

# ===== datasets =====
export MLVU_ROOT=/home/ubuntu/videomind/VideoMind/datasets/mlvu_dev
export VIDEOMME_ROOT=/home/ubuntu/videomind/VideoMind/datasets/video_mme
export INTENTQA_ROOT=/home/ubuntu/videomind/VideoMind/datasets/intentqa

# ===== outputs =====
export OUT_ROOT=$ROOT/outputs
export LOG_ROOT=$ROOT/logs

# ===== model =====
export MODEL_PATH=/home/ubuntu/videomind/VideoMind/model_zoo/Qwen2-VL-2B-Instruct

# ===== default experiment setting =====
export MAX_FRAMES=16
export CLIP_SECONDS=8
export TOPK=4
export MAX_ROUNDS=3
export EVIDENCE_THRESHOLD=0.62
export ACCEPT_MARGIN=0.06
