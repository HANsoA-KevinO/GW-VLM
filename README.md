# GW-VLM

用预训练 VLM（Qwen3-VL / Qwen3.6 / Gemma 4）微调做引力波检测 + 粗粒度参数估计。原 GW-TF 项目（LLaMA-3-8B + K-Means 量化丢失 99.8% 信息）的完全重写版。

**新接手请先读 [`CLAUDE.md`](CLAUDE.md)** —— 完整项目上下文、关键决策、当前进度、下一步、迁移指南都在那里。

## 快速开始（新机器再生数据）

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r data_pipeline/requirements.txt
python data_pipeline/scripts/01_download_strain.py --full
python data_pipeline/scripts/02_generate_spectrograms.py --full
python data_pipeline/scripts/06_build_dataset.py
python data_pipeline/scripts/07_split_by_event.py
```
确定性再生（seed=42 + events.csv 固定，H1+L1 排除 V1），产出 **train 2394 / val 306 / test 270**（共 2970，正/负各 1485）。

## 训练（已搭建）

```bash
# 1. 导出训练格式（E1 纯检测；--schema multitask 出 E2）
python data_pipeline/scripts/08_export_training_format.py --schema detection_only
# 2. 在 DGX Spark 上微调（Unsloth，详见 training/README.md）
python training/train_vlm.py --config training/configs/e1_gemma4_e4b.yaml \
    --image-root output/spectrograms
# 3. 评估
python evaluation/evaluate.py --adapter <out_dir> --test output/training_data/e1/test.jsonl \
    --image-root output/spectrograms
```

## 当前阶段

数据 pipeline ✅ + 训练基建 ✅ + **E1（纯检测）已在 DGX Spark 跑完多模型**(Gemma4 E4B / 31B 基线 / 31B v2 / 31B viridis / Qwen3.6-27B)。

**E1 最佳结果**:🥇 **Qwen3.6-27B(原生多模态,viridis,2ep)ROC-AUC 0.940**,默认 0.5 阈值 recall 0.837(次优 Gemma4 31B v2 = 0.922)。彩色(viridis≈灰度)/ 分辨率(>560)/ 堆 epoch(3ep≤2ep)均无法突破 ~0.92–0.94 天花板(已实验证伪),瓶颈是低 SNR 弱信号;评估按 **ROC-AUC + FAR 工作点**(非贪心 accuracy)。

> 📋 **这两天的完整实验记录/结果/发现/环境踩坑 → [`docs/04_e1_experiments_and_findings.md`](docs/04_e1_experiments_and_findings.md)**;接手必读 [`CLAUDE.md`](CLAUDE.md)。

## 目录

```
data_pipeline/   数据生成 + 08 训练格式导出 + 02(支持 --cmap/--outdir)（含 events.csv）
training/        Unsloth LoRA 微调脚本 + 各模型 E1 配置 + spark_env.sh + run_spark.sh
evaluation/      指标/混淆矩阵 + 贪心评估 + 概率评估(ROC/PR/阈值) + 损失曲线 + SNR 诊断
docs/            01 选型 / 02 研究方案(权威) / 03 Gemma4 调查 / 04 E1 实验记录与发现
output/          strain/spectrogram/dataset + training_data + runs/(adapter+图表)（大文件 .gitignore）
CLAUDE.md        迁移交接文档（必读）
```
