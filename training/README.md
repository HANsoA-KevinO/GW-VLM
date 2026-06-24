# training/ — Unsloth LoRA 视觉微调（DGX Spark 实测流程）

对 08 导出的训练数据做 LoRA 微调，Qwen3-VL / Qwen3.6 / Gemma 4 共用一套脚本。
训练机：**DGX Spark**（`super-cortant` = `kevin@spark-91b6.local`，GB10/sm_121，128GB，Ubuntu 24.04）。

> 实测走的是**主机 venv**（不是 Docker 容器）——因为 GB10 的 cu130 torch、Gemma4 处理器、
> Triton 编译这些都能在主机 venv 里跑通，避免了 docker 守护进程代理需要 sudo 的问题。

## 文件
- `train_vlm.py` — 训练主程序（Unsloth `FastVisionModel` + `SFTTrainer`，eager 模式）。
- `configs/e1_*.yaml` — 各模型 E1（纯检测）配置（gemma4_e4b / gemma4_31b / qwen3vl_8b / qwen36_27b）。
- `spark_env.sh` — **Spark 运行环境一键 source**（代理 / HF / Unsloth / Python.h CPATH，见下）。
- `run_spark.sh` — detached 运行包装（train/eval），配 setsid 脱离 ssh 会话。
- `requirements_train.txt` — 依赖清单（torch/torchvision 见下，需 cu130）。

## 一次性环境置备（已在 Spark 上完成，记录备查）

```bash
# 1) venv（系统 python3.12）
cd ~/GW-VLM && python3.12 -m venv .venv && source .venv/bin/activate

# 2) PyTorch + torchvision（GB10=sm_121 必须 cu130；download.pytorch.org 有 aarch64 轮子）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
#   验证：python -c "import torch;x=torch.randn(8,8,device='cuda');print((x@x).sum())"

# 3) Unsloth 栈（按 NVIDIA DGX Spark 组合；--no-deps 装 unsloth 以免动 cu130 torch）
pip install numpy pillow pyyaml scikit-learn matplotlib accelerate sentencepiece \
    hf_transfer "datasets==4.3.0" "trl==0.26.1" transformers peft
pip install --no-deps unsloth unsloth_zoo bitsandbytes

# 4) Python.h（Triton 编译需要；免 sudo：下 .deb 解压，用 CPATH 指过去）
mkdir -p ~/pydev && cd ~/pydev
PX='-o Acquire::http::Proxy=http://<代理IP>:7897 -o Acquire::https::Proxy=http://<代理IP>:7897'
apt-get download $PX python3.12-dev libpython3.12-dev
for d in *.deb; do dpkg -x "$d" ~/pydev/root; done   # 头文件落到 ~/pydev/root/usr/include
```

> 踩过的坑（都已写进 `spark_env.sh` 的开关 / `train_vlm.py` 的 setdefault）：
> - **Xet 下载卡死** → `HF_HUB_DISABLE_XET=1`（其 rust 客户端不走代理）。
> - **Unsloth 遥测 120s 超时误报 HF down** → `UNSLOTH_DISABLE_STATISTICS=1`。
> - **Gemma4 处理器 ModuleNotFoundError** → 其实是 `torchvision` 没装（image_processing_gemma4 依赖它）。
> - **Triton gcc 编译失败** → 缺 `Python.h` → CPATH 指向解压的头文件。
> - **dynamo fullgraph 编译失败** → `UNSLOTH_COMPILE_DISABLE=1` 走 eager。
> - **大 bf16 device_map 在 UMA 上 offload→accelerate 拒训** → `ACCELERATE_BYPASS_DEVICE_MAP=true`（已写进 train_vlm.py）。
> - **代理掉了也能训** → 模型已缓存,训练/评估加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 即可离线跑。

## 跑训练 / 评估

```bash
# 数据已在本机用 08 导出并 rsync 到 Spark；模型从 unsloth/ 镜像下（免门）。
# 全量 E1（E4B 调试模型）—— detached，日志到 ~/full_e4b.log
cd ~/GW-VLM
setsid nohup bash training/run_spark.sh train training/configs/e1_gemma4_e4b.yaml --epochs 3 \
    > ~/full_e4b.log 2>&1 </dev/null &

# 看进度（tqdm 用 \r，看真实进度要转一下）
tail -c 2000 ~/full_e4b.log | tr '\r' '\n' | tail

# 评估（4bit 基座 + adapter）
setsid nohup bash training/run_spark.sh eval output/runs/e1_gemma4_e4b \
    > ~/eval_e4b.log 2>&1 </dev/null &

# 主模型（验证通过后）
... run_spark.sh train training/configs/e1_gemma4_31b.yaml      # Gemma 4 31B（bf16 LoRA）
... run_spark.sh train training/configs/e1_qwen36_27b.yaml      # Qwen3.6-27B（bf16 LoRA）
```

产物：LoRA adapter → `output/runs/e1_<model>/`。

## 评估(在 `evaluation/`)
```bash
# 贪心评估(accuracy/F1/混淆矩阵) —— 仅参考,会低估 recall
python evaluation/evaluate.py --adapter output/runs/<run> --no-4bit --image-tokens 560

# 概率版评估(主用):ROC-AUC/PR-AUC + 阈值扫描 + FAR 工作点 + 每样本分数
python evaluation/evaluate_prob.py --adapter output/runs/<run> --no-4bit --image-tokens 560

# 损失曲线 / SNR 诊断(detection efficiency vs SNR)
python evaluation/plot_loss.py output/runs/<run>
python evaluation/plot_snr.py --per-sample output/runs/<run>/per_sample.jsonl --dataset output/dataset_test.jsonl --threshold 0.18
```
> 评估的 `--image-tokens` / `--no-4bit` 必须和该 adapter 训练时一致(分辨率/精度匹配)。

## ✅ 当前最佳配方(主模型都用这套)
`bf16 LoRA, r=32, α=32, lr=2e-4, dropout=0, 有效batch=8(b2×ga4), image_tokens=560`
→ 31B 实测 **ROC-AUC 0.922**(FPR≤5% 下 recall 75%)。见 `configs/e1_gemma4_31b_v2.yaml`。

## 备注
- 任意 YAML 项可被同名 CLI 覆盖（`--epochs/--batch-size/--learning-rate/--max-samples/--image-tokens/...`）。
- **代理 IP 会变**（clash DHCP）：失效时改 `spark_env.sh` 顶部 IP;或训练/评估直接 `HF_HUB_OFFLINE=1` 离线跑。
- **分辨率**:`image_tokens` 控 Gemma4 token 预算(560≈1135px 吃满 1024 源);默认 512px 偏低。
- E4B 4bit / 31B·27B bf16(`load_in_4bit:false`)。E2:`08 --schema multitask` → 改 config `data_dir`。
- **运维经验**:① 启动命令常"无输出"但其实已执行 → 只读重连核对、别盲杀;② ControlMaster 复用 socket 损坏会全 255 → 删 socket 文件即恢复;③ 长任务用"本地循环 + 每轮短连接轮询"监视。
- 📋 完整实验记录/结果/发现 → [`docs/04_e1_experiments_and_findings.md`](../docs/04_e1_experiments_and_findings.md)。
