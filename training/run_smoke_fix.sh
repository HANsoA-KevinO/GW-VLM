#!/usr/bin/env bash
# 修复后验证:小规模 train + eval,确认图像 pooler_output 修复后两端都通。
exec > "$HOME/smoke_fix.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u training/train_fusion.py --config training/configs/fusion_qwen2.5vl_3b.yaml \
  --use-image true --use-strain true --max-samples 32 --epochs 1 --output-dir output/runs/smoke_fix
echo "===== EVAL ====="
python -u evaluation/evaluate_fusion.py --adapter output/runs/smoke_fix \
  --image-root output/spectrograms_viridis --strain-root output/strain_arrays --max-samples 32
echo "SMOKEFIX_DONE=$?"
