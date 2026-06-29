#!/usr/bin/env bash
# 噪声大扩充 + 注入扩量 + 重建 e5(全本地)。日志 output/noise_expand.log。
#   03b 拉干净O3噪声池(neg/injbg 互斥)→ 04 注入(含injbg背景)→ 05 渲染注入
#   → 05b 渲染负样本(数=注入数,配平)→ 06/07/08 重建 e5/e1 → 守卫(val/test零合成)。
# 用法: nohup bash data_pipeline/scripts/run_noise_expand.sh &
set -u
cd "$HOME/code/GW-VLM" || exit 1
exec > output/noise_expand.log 2>&1
FILT='Warning|IERS|astropy|finals|urlopen|pkg-config|NO_PKGCONFIG|WARNING'

echo "===== 0) 清理旧噪声渲染(raw_noise 保留,03b skip复用)$(date +%T) ====="
rm -f output/spectrograms_viridis/noise_*.png output/strain_arrays/noise_*.npy

echo "===== 1) 03b 拉噪声池 injbg90 + neg200 $(date +%T) ====="
.venv-render/bin/python data_pipeline/scripts/03b_fetch_noise_pool.py --n-injbg 90 --n-neg 200 \
  2>&1 | grep -vE "$FILT" || { echo "03b_FAIL"; exit 1; }
NSEG=$(wc -l < output/noise_pool_manifest.jsonl); echo "噪声段(H1+L1齐全) $NSEG"

echo "===== 2) 04 注入(.venv-inject,真实host+injbg背景)$(date +%T) ====="
.venv-inject/bin/python data_pipeline/scripts/04_inject_signals.py --n 2400 \
  2>&1 | grep -vE "$FILT" | tail -12 || { echo "04_FAIL"; exit 1; }
E=$(wc -l < output/injections_manifest.jsonl); echo "注入事件 $E"

echo "===== 3) 05 渲染注入(图+应变+sidecar)$(date +%T) ====="
.venv-inject/bin/python data_pipeline/scripts/05_render_injections.py \
  2>&1 | grep -vE "$FILT" | tail -6 || { echo "05_FAIL"; exit 1; }

NEG_TARGET=$((E * 2))   # 配平:负样本数 ≈ 注入样本数(每注入×2探测器)
echo "===== 4) 05b 渲染负样本 目标 $NEG_TARGET(配平注入)$(date +%T) ====="
.venv-render/bin/python data_pipeline/scripts/05b_render_noise_negatives.py --target "$NEG_TARGET" \
  2>&1 | grep -vE "$FILT" | tail -6 || { echo "05b_FAIL"; exit 1; }

echo "===== 5) 06 重建数据集(viridis,真实+注入+噪声neg)$(date +%T) ====="
.venv-render/bin/python data_pipeline/scripts/06_build_dataset.py --dir output/spectrograms_viridis \
  2>&1 | tail -16 || { echo "06_FAIL"; exit 1; }

echo "===== 6) 07 切分(注入/噪声neg 只进train,val/test全真实)$(date +%T) ====="
.venv-render/bin/python data_pipeline/scripts/07_split_by_event.py 2>&1 | tail -22 || { echo "07_FAIL"; exit 1; }

echo "===== 7) 08 导出 e5(unified)+ e1(detection_only)$(date +%T) ====="
.venv-render/bin/python data_pipeline/scripts/08_export_training_format.py --schema unified \
  --image-base output/spectrograms_viridis 2>&1 | tail -3 || { echo "08U_FAIL"; exit 1; }
.venv-render/bin/python data_pipeline/scripts/08_export_training_format.py --schema detection_only \
  --image-base output/spectrograms_viridis 2>&1 | tail -3 || { echo "08E1_FAIL"; exit 1; }

echo "===== 8) 守卫 + 平衡 $(date +%T) ====="
for sp in val test; do
  inj=$(grep -c '"inject' output/dataset_${sp}.jsonl 2>/dev/null || echo 0)
  noi=$(grep -c 'noise_neg' output/dataset_${sp}.jsonl 2>/dev/null || echo 0)
  echo "  dataset_${sp}: inject=$inj noise_neg=$noi (都应为0)"
done
echo "  train 检测平衡:"; grep -oE '"detection": "(YES|NO)"' output/dataset_train.jsonl | sort | uniq -c
echo "  e5 行数:"; for sp in train val test; do echo "    $sp: $(wc -l < output/training_data/e5/${sp}.jsonl)"; done
echo "NOISE_EXPAND_DONE $(date +%T)"
