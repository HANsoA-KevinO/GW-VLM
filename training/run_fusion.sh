#!/usr/bin/env bash
# 融合训练启动器。自重定向日志(不依赖启动命令的重定向)。参数透传给 train_fusion.py。
exec > "$HOME/fusion_run.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python -u training/train_fusion.py --config training/configs/fusion_qwen2.5vl_3b.yaml "$@"
echo "FUSION_EXIT=$?"
