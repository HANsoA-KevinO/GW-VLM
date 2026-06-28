#!/usr/bin/env bash
# 3B 统一冒烟(bf16,快、稳)——隔离 nan 是否模型无关。
exec > "$HOME/u3b.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u training/train_unified.py --config training/configs/unified_qwen25vl_3b.yaml \
  --max-samples 96 --epochs 1 --output-dir output/runs/unified_q25_3b_smoke || { echo "U3B_TRAIN_FAIL"; exit 1; }
echo "===== EVAL ====="
python -u evaluation/evaluate_unified.py --adapter output/runs/unified_q25_3b_smoke \
  --test output/training_data/e5/test.jsonl \
  --image-root output/spectrograms_viridis --strain-root output/strain_arrays --max-samples 60
echo "U3B_SMOKE_DONE=$?"
