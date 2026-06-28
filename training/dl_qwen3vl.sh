#!/usr/bin/env bash
# 下载 Qwen3-VL-8B-Instruct(禁 xet + hf_transfer,用稳的标准下载器,断点续传)。
export https_proxy=http://192.168.1.6:7897 http_proxy=http://192.168.1.6:7897
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-VL-8B-Instruct", max_workers=2)
print("SNAPSHOT_AT", p)
PY
echo "DL_DONE=$?"
