#!/usr/bin/env bash
# 启动 Qwen3.6-27B E2(多任务:检测+参数bin)训练。自重定向日志,不依赖启动命令。
exec > "$HOME/full_e2.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/GW-VLM" || exit 1
exec bash training/run_spark.sh train training/configs/e2_qwen36_27b.yaml \
  --epochs 3 \
  --image-root /home/kevin/GW-VLM/output/spectrograms_viridis \
  --output-dir output/runs/e2_qwen36_27b_viridis
