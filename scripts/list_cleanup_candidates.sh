#!/usr/bin/env bash
set -e

echo "===== disk ====="
df -h /

echo
echo "===== biggest dirs under /home/ubuntu/videomind, depth 3 ====="
du -h --max-depth=3 /home/ubuntu/videomind 2>/dev/null | sort -hr | head -80

echo
echo "===== model_zoo ====="
du -sh /home/ubuntu/videomind/VideoMind/model_zoo/* 2>/dev/null | sort -hr

echo
echo "===== possible cache dirs ====="
for p in \
  /home/ubuntu/.cache \
  /home/ubuntu/.cache/huggingface \
  /home/ubuntu/.cache/pip \
  /home/ubuntu/miniconda3/pkgs \
  /home/ubuntu/videomind/VideoMind/datasets/mlvu_dev/.cache \
  /home/ubuntu/videomind/VideoMind/datasets/video_mme_hf_cache
do
  if [ -e "$p" ]; then
    du -sh "$p"
  fi
done

echo
echo "===== large files over 500M ====="
find /home/ubuntu/videomind -type f -size +500M 2>/dev/null | while read f; do
  du -h "$f"
done | sort -hr | head -100
