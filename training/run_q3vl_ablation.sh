#!/usr/bin/env bash
# 第 2 轮融合消融 —— Qwen3-VL-8B + patch/自注意力应变编码器,两种粒度(32/64)。
# C 仅图 / B-32 图+应变(32) / B-64 / A-32 仅应变(32) / A-64。自重定向日志。
exec > "$HOME/q3vl_ablation.log" 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
CFG=training/configs/fusion_qwen3vl_8b.yaml
EP=2
run() {   # $1=name $2=use_image $3=use_strain $4=patch_size
  echo "===== TRAIN $1 (img=$2 strain=$3 patch=$4) $(date +%T) ====="
  python -u training/train_fusion.py --config $CFG \
    --use-image "$2" --use-strain "$3" --strain-patch "$4" --epochs $EP \
    --output-dir output/runs/fusion_q3vl_$1 || { echo "TRAIN_$1_FAIL"; return; }
  echo "===== EVAL $1 $(date +%T) ====="
  python -u evaluation/evaluate_fusion.py --adapter output/runs/fusion_q3vl_$1 \
    --image-root output/spectrograms_viridis --strain-root output/strain_arrays || echo "EVAL_$1_FAIL"
}
run C   true  false 256
run B32 true  true  256
run B64 true  true  128
run A32 false true  256
run A64 false true  128
echo "===== Q3VL 消融对照 ====="
for n in C B32 B64 A32 A64; do
  f=output/runs/fusion_q3vl_$n/fusion_eval.json
  [ -f "$f" ] && python -c "import json;d=json.load(open('$f'));print('  $n | img=%s strain=%s | ROC-AUC=%.4f PR-AUC=%.4f | R@0.5=%s R@FPR5%%=%s'%(d['use_image'],d['use_strain'],d['roc_auc'],d['pr_auc'],d['operating_points']['default_0.5']['recall'],d['operating_points']['fpr<=0.05']['recall']))" || echo "  $n 无结果"
done
echo "Q3VL_ABLATION_DONE $(date +%T)"
