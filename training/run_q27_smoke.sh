#!/usr/bin/env bash
# 冒烟:Qwen3.6-27B + patch+自注意力编码器,32 样本 1 epoch,验证融合代码在 27B 上兼容
# (重点验证 qwen 路径的 get_image_features.pooler_output + get_rope_index 在 27B 不报错)。
exec > "$HOME/q27_smoke.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u training/train_fusion.py --config training/configs/fusion_qwen36_27b.yaml \
  --use-image true --use-strain true --strain-patch 256 --max-samples 32 --epochs 1 \
  --output-dir output/runs/q27_smoke || { echo "SMOKE_TRAIN_FAIL"; exit 1; }
echo "===== EVAL ====="
python -u evaluation/evaluate_fusion.py --adapter output/runs/q27_smoke \
  --image-root output/spectrograms_viridis --strain-root output/strain_arrays --max-samples 32
echo "Q27_SMOKE_DONE=$?"
