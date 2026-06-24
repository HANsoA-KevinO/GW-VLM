#!/usr/bin/env bash
# 评估 E2(多任务)最终 adapter。自重定向日志。
exec > "$HOME/eval_e2.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u evaluation/evaluate_e2.py \
  --adapter output/runs/e2_qwen36_27b_viridis \
  --image-root output/spectrograms_viridis
echo "EVAL_E2_DONE=$?"
