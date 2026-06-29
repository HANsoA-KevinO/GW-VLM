#!/usr/bin/env bash
# new-E2(真实-only 对照):统一模型(检测+后验头)+ 经典基线,只在 90 真实事件(e5_real)上训。
# 对照 E2-old(bin) 看"形态+应变"的贡献;对照 E4 看"注入数据"的贡献。
# 用法: bash training/run_e2.sh [epochs]   (默认 2)
exec > "$HOME/e2.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
EP=${1:-2}

echo "===== 1) 统一模型(27B bf16+梯度检查点,检测+后验头,真实-only)EP=$EP $(date +%T) ====="
python -u training/train_unified.py --config training/configs/unified_qwen36_27b_e2real.yaml \
  --epochs "$EP" --output-dir output/runs/e2_unified_real || { echo "UNIFIED_TRAIN_FAIL"; exit 1; }
echo "===== EVAL 统一(真实测试集)$(date +%T) ====="
python -u evaluation/evaluate_unified.py --adapter output/runs/e2_unified_real \
  --test output/training_data/e5_real/test.jsonl \
  --image-root output/spectrograms_viridis --strain-root output/strain_arrays || echo "UNIFIED_EVAL_FAIL"

echo "===== 2) 经典基线(strain→参数,无VLM,真实-only)$(date +%T) ====="
python -u training/train_baseline_param.py --data-dir output/training_data/e5_real \
  --strain-root output/strain_arrays --output-dir output/runs/e2_baseline_real --epochs 80 \
  || { echo "BASELINE_TRAIN_FAIL"; exit 1; }
python -u evaluation/evaluate_baseline_param.py --adapter output/runs/e2_baseline_real \
  --test output/training_data/e5_real/test.jsonl --strain-root output/strain_arrays || echo "BASELINE_EVAL_FAIL"

echo "E2_DONE $(date +%T)"
