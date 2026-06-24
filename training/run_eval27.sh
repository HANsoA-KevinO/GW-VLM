#!/usr/bin/env bash
# 评估 Qwen3.6-27B checkpoint-600(2ep, viridis) —— 离线,概率版评估。
exec > "$HOME/eval27.log" 2>&1   # 脚本自重定向日志,不依赖启动命令
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u evaluation/evaluate_prob.py \
  --adapter output/runs/e1_qwen36_27b_viridis_3ep/checkpoint-600 \
  --no-4bit --image-root output/spectrograms_viridis --no-unsloth
echo "EVAL27_DONE=$?"
