#!/usr/bin/env bash
# 装 27B 快路径所需的 causal_conv1d(CUDA 编译)。fla 已装好,这里补 causal_conv1d。
# 自动按本机 GPU 算力设编译架构(GB10 Blackwell = 12.1),走 Mac 代理。
export https_proxy=http://192.168.1.6:7897 http_proxy=http://192.168.1.6:7897
cd "$HOME/GW-VLM" || exit 1
source .venv/bin/activate

echo "=== 现状 ==="
python -c "import fla; print('fla', fla.__version__)" 2>&1 | tail -1
ARCH=$(python -c "import torch;c=torch.cuda.get_device_capability();print(f'{c[0]}.{c[1]}')" 2>/dev/null)
echo "GPU 算力 = $ARCH"
export TORCH_CUDA_ARCH_LIST="$ARCH"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

echo "=== 检查 Python.h(缺则提示装 python3.12-dev)==="
python -c "import sysconfig,os;h=os.path.join(sysconfig.get_path('include'),'Python.h');print('Python.h:', 'OK' if os.path.exists(h) else 'MISSING -> 先 sudo apt install -y python3.12-dev', h)"

echo "=== 装 ninja(加速编译)==="
pip install -q ninja 2>&1 | tail -2

echo "=== 编译安装 causal-conv1d(可能几分钟)==="
pip install -v causal-conv1d 2>&1 | tail -40
echo "CC_PIP_EXIT=${PIPESTATUS[0]}"

echo "=== 验证 import ==="
python -c "import causal_conv1d; print('CC_IMPORT_OK', causal_conv1d.__version__)" 2>&1 | tail -3
echo "ACCEL_DONE"
