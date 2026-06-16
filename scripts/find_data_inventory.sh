#!/usr/bin/env bash
set -e

ROOT=/home/ubuntu/videomind/VideoMind

echo "===== disk ====="
df -h "$ROOT"

echo
echo "===== LZHY03 size ====="
du -sh /home/ubuntu/videomind/VideoMind/LZHY03

echo
echo "===== candidate dataset dirs ====="
find "$ROOT" -maxdepth 6 -type d 2>/dev/null | \
grep -Ei "mlvu|video.?mme|intent|next|msvd|egoschema|activity|dataset|data" | \
sort | head -200

echo
echo "===== video file directories: mp4/avi/mkv count by folder ====="
find "$ROOT" -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.webm" \) 2>/dev/null | \
sed 's#/[^/]*$##' | sort | uniq -c | sort -nr | head -100

echo
echo "===== annotation files: json/jsonl/csv/tsv ====="
find "$ROOT" -type f \( -iname "*.json" -o -iname "*.jsonl" -o -iname "*.csv" -o -iname "*.tsv" \) 2>/dev/null | \
grep -Ei "mlvu|video.?mme|intent|next|msvd|qa|anno|valid|test|train|dev|eval" | \
head -300

echo
echo "===== old output json files ====="
find "$ROOT" -type f \( -name "output.json" -o -name "*compare*.json" -o -name "*summary*.json" \) 2>/dev/null | \
grep -Ei "mlvu|video.?mme|intent|msvd|scvm|baseline|outputs|analysis" | \
head -300

echo
echo "===== largest dirs under VideoMind, depth 2 ====="
du -h --max-depth=2 "$ROOT" 2>/dev/null | sort -hr | head -60
