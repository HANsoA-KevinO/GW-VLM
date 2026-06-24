#!/usr/bin/env bash
# 思考诊断:带思考生成、保存 <think> 原文。自重定向日志。
exec > "$HOME/diag_think.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u evaluation/diag_e2_thinking.py \
  --adapter output/runs/e2_qwen36_27b_viridis \
  --image-root output/spectrograms_viridis \
  --out "$HOME/think_diag.jsonl"
echo "DIAG_DONE=$?"
