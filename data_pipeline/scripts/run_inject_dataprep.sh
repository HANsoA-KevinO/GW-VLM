#!/usr/bin/env bash
# 本地数据准备链:等 04 注入生成完 → 05 渲染 → 06(viridis,含注入)→ 07(注入只进train)→ 08 e5+e1。
# 全自动,产物在 output/training_data/{e5,e1}/。日志 output/dataprep.log。
exec > "$HOME/code/GW-VLM/output/dataprep.log" 2>&1
cd "$HOME/code/GW-VLM" || exit 1
echo "=== 等 04 注入生成完 $(date +%T) ==="
while pgrep -f 04_inject_signals >/dev/null; do sleep 30; done
echo "04 完成. manifest=$(wc -l < output/injections_manifest.jsonl) $(date +%T)"

echo "=== 05 渲染注入(图+应变+sidecar)$(date +%T) ==="
.venv-inject/bin/python data_pipeline/scripts/05_render_injections.py 2>&1 | grep -vE "pkg-config|NO_PKGCONFIG|IERS|astropy|finals2000A|urlopen|WARNING" | tail -6 || { echo "05_FAIL"; exit 1; }

echo "=== 06 重建数据集(viridis,含真实+注入)$(date +%T) ==="
.venv-render/bin/python data_pipeline/scripts/06_build_dataset.py --dir output/spectrograms_viridis 2>&1 | tail -12 || { echo "06_FAIL"; exit 1; }

echo "=== 07 切分(注入只进 train)$(date +%T) ==="
.venv-render/bin/python data_pipeline/scripts/07_split_by_event.py 2>&1 | tail -20 || { echo "07_FAIL"; exit 1; }

echo "=== 08 导出 e5(unified)+ e1(detection_only)$(date +%T) ==="
.venv-render/bin/python data_pipeline/scripts/08_export_training_format.py --schema unified \
  --image-base output/spectrograms_viridis 2>&1 | tail -3 || { echo "08_UNIFIED_FAIL"; exit 1; }
.venv-render/bin/python data_pipeline/scripts/08_export_training_format.py --schema detection_only \
  --image-base output/spectrograms_viridis 2>&1 | tail -3 || { echo "08_E1_FAIL"; exit 1; }

echo "=== 守卫:val/test 必须零注入 ==="
for sp in val test; do
  n=$(grep -c '"inject' output/dataset_${sp}.jsonl 2>/dev/null || echo 0)
  echo "  dataset_${sp}: inject=$n (应为0)"
done
echo "=== e5 行数 ==="; for sp in train val test; do echo "  e5/$sp: $(wc -l < output/training_data/e5/${sp}.jsonl)"; done
echo "DATAPREP_DONE $(date +%T)"
