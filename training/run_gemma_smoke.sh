#!/usr/bin/env bash
# 冒烟:Gemma 路径 patch+自注意力编码器,32 样本 1 epoch,train+eval 跑通即可。
exec > "$HOME/gemma_smoke.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u training/train_fusion.py --config training/configs/fusion_gemma4_e4b.yaml \
  --use-image true --use-strain true --strain-patch 256 --max-samples 32 --epochs 1 \
  --output-dir output/runs/gemma_smoke || { echo "SMOKE_TRAIN_FAIL"; exit 1; }
echo "===== EVAL ====="
python -u evaluation/evaluate_fusion.py --adapter output/runs/gemma_smoke \
  --image-root output/spectrograms_viridis --strain-root output/strain_arrays --max-samples 32
echo "GEMMA_SMOKE_DONE=$?"
