# GW-VLM 在 DGX Spark(GB10/aarch64)上的运行环境 —— 主机 venv 路线（非 Docker）。
# 用法：  source training/spark_env.sh   然后跑 train_vlm.py / evaluate.py
#
# 背景：DGX Spark 无直连外网，靠局域网 clash 代理(端口 7897)；GB10=sm_121 需 cu130 的
# torch；Gemma4 处理器需 torchvision；Triton 编译需 Python.h；torch2.12+Unsloth 对
# gemma4 的 torch.compile 有问题需走 eager。下列开关把这些一次性配好。

# —— 代理（clash 局域网 IP，会随 DHCP 变；失效时改这里）——
export https_proxy=http://192.168.31.42:7897
export http_proxy=http://192.168.31.42:7897
export no_proxy=localhost,127.0.0.1,::1,*.local,spark-91b6.local

# —— HuggingFace 下载（Xet/hf_transfer 的 rust 客户端不走代理，改经典 HTTPS）——
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# —— Unsloth ——
export UNSLOTH_DISABLE_STATISTICS=1   # 遥测在受限网络会 snapshot_download 卡 120s 误报 HF down
export UNSLOTH_COMPILE_DISABLE=1      # torch2.12+Unsloth 对 gemma4 的 dynamo fullgraph 编译会失败，走 eager

# —— DGX Spark 大 bf16 模型必需 ——
# GB10 统一内存上 device_map 会把权重"offload 到 CPU"(其实同一块物理内存,无害),
# accelerate 会误判成分布式 + 多设备而拒训。设此变量跳过该检查（官方逃生门）。
export ACCELERATE_BYPASS_DEVICE_MAP=true

# —— Triton 编译需要 Python.h（免 sudo：apt-get download python3.12-dev libpython3.12-dev
#     到 ~/pydev 解压，CPATH 指过去）——
export CPATH=$HOME/pydev/root/usr/include/python3.12:$HOME/pydev/root/usr/include

# —— 激活 venv ——
source $HOME/GW-VLM/.venv/bin/activate
