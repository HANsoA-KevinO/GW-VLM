#!/usr/bin/env bash
# 正式 E4:统一模型(检测+后验头)+ 经典基线,在 真实+注入(e5)上训 → 三方对照。
# 同时覆盖 E3(检测对注入的召回,看 unified 的检测 ROC-AUC vs 纯真实的 0.96)。
# 用法:bash training/run_e4.sh [epochs]   (默认 2;想快出先 1)
exec > "$HOME/e4.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
EP=${1:-2}

echo "===== 1) 统一模型(27B bf16+梯度检查点,检测+后验头)EP=$EP $(date +%T) ====="
python -u training/train_unified.py --config training/configs/unified_qwen36_27b.yaml \
  --epochs "$EP" --output-dir output/runs/e4_unified || { echo "UNIFIED_TRAIN_FAIL"; exit 1; }
echo "===== EVAL 统一 $(date +%T) ====="
python -u evaluation/evaluate_unified.py --adapter output/runs/e4_unified \
  --test output/training_data/e5/test.jsonl \
  --image-root output/spectrograms_viridis --strain-root output/strain_arrays || echo "UNIFIED_EVAL_FAIL"

echo "===== 2) 经典基线(strain→参数,无VLM)$(date +%T) ====="
python -u training/train_baseline_param.py --data-dir output/training_data/e5 \
  --strain-root output/strain_arrays --output-dir output/runs/e4_baseline --epochs 80 \
  || { echo "BASELINE_TRAIN_FAIL"; exit 1; }
python -u evaluation/evaluate_baseline_param.py --adapter output/runs/e4_baseline \
  --test output/training_data/e5/test.jsonl --strain-root output/strain_arrays || echo "BASELINE_EVAL_FAIL"

echo "E4_DONE $(date +%T)"
