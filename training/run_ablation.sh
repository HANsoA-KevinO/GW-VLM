#!/usr/bin/env bash
# 串行跑融合消融 A/B/C(训练+评估),自重定向日志。回答"加了应变还要不要图"。
exec > "$HOME/ablation.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
EP=1
run() {   # $1=name $2=use_image $3=use_strain
  echo "===== TRAIN $1 (img=$2 strain=$3) $(date +%T) ====="
  python -u training/train_fusion.py --config training/configs/fusion_qwen2.5vl_3b.yaml \
    --use-image "$2" --use-strain "$3" --epochs $EP --output-dir output/runs/fusion_$1 || { echo "TRAIN_$1_FAIL"; return; }
  echo "===== EVAL $1 $(date +%T) ====="
  python -u evaluation/evaluate_fusion.py --adapter output/runs/fusion_$1 \
    --image-root output/spectrograms_viridis --strain-root output/strain_arrays || echo "EVAL_$1_FAIL"
}
run C true false
run B true true
run A false true
echo "ABLATION_DONE $(date +%T)"
