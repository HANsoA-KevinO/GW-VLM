#!/usr/bin/env bash
# GW-VLM Spark 运行包装：自动 source 环境，跑训练或评估。
# 用法（建议配 setsid 脱离 ssh 会话）：
#   setsid nohup bash training/run_spark.sh train <config.yaml> [额外参数] > ~/log 2>&1 </dev/null &
#   setsid nohup bash training/run_spark.sh eval  <adapter_dir>  [额外参数] > ~/log 2>&1 </dev/null &
set +e
source "$HOME/GW-VLM/training/spark_env.sh"
cd "$HOME/GW-VLM" || exit 1
IMG_ROOT="$HOME/GW-VLM/output/spectrograms"
mode="$1"; target="$2"; shift 2
case "$mode" in
  train)
    python -u training/train_vlm.py --config "$target" --image-root "$IMG_ROOT" "$@" ;;
  eval)
    python -u evaluation/evaluate.py --adapter "$target" \
      --test output/training_data/e1/test.jsonl --image-root "$IMG_ROOT" "$@" ;;
  *) echo "用法: run_spark.sh train|eval <target> [args]"; exit 2 ;;
esac
echo "RUN_EXIT=$?"
