#!/usr/bin/env bash
set -e

echo "===== disk ====="
df -h /home/ubuntu/videomind/VideoMind/LZHY03

echo
echo "===== model paths ====="
for p in \
  /home/ubuntu/videomind/VideoMind/model_zoo/Qwen2-VL-2B-Instruct \
  /home/ubuntu/videomind/VideoMind/model_zoo/VideoMind-2B
do
  if [ -e "$p" ]; then
    du -sh "$p"
  else
    echo "MISSING: $p"
  fi
done

echo
echo "===== possible dataset paths ====="
for p in \
  /home/ubuntu/videomind/VideoMind/VideoMind-xinG3/data/mlvu_dev \
  /home/ubuntu/videomind/VideoMind/datasets/video_mme \
  /home/ubuntu/videomind/VideoMind/datasets/intentqa \
  /home/ubuntu/videomind/VideoMind/VideoMind-xinG3/data
do
  if [ -e "$p" ]; then
    du -sh "$p"
  else
    echo "MISSING: $p"
  fi
done

echo
echo "===== LZHY03 third_party ====="
du -sh /home/ubuntu/videomind/VideoMind/LZHY03/third_party/*
