#!/usr/bin/env bash
# 冒烟:统一(E5)= 27B 融合 + 后验头。验证 tap/head/NLL 管线通、param_nll 下降、能存能评。
exec > "$HOME/unified_smoke.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u training/train_unified.py --config training/configs/unified_qwen36_27b.yaml \
  --max-samples 64 --epochs 1 --load-4bit true --output-dir output/runs/unified_smoke \
  || { echo "SMOKE_TRAIN_FAIL"; exit 1; }
echo "===== EVAL ====="
python -u evaluation/evaluate_unified.py --adapter output/runs/unified_smoke \
  --test output/training_data/e5/test.jsonl \
  --image-root output/spectrograms_viridis --strain-root output/strain_arrays --max-samples 48
echo "UNIFIED_SMOKE_DONE=$?"
