#!/usr/bin/env bash
set -e

echo "===== disk ====="
df -h .

echo "===== project size ====="
du -sh . || true
du -sh third_party/* 2>/dev/null || true

echo "===== python ====="
which python
python --version

echo "===== torch / cuda ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
