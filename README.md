# GW-VLM

用预训练 VLM（Qwen3-VL / Qwen3.6）微调做引力波检测 + 粗粒度参数估计。原 GW-TF 项目（LLaMA-3-8B + K-Means 量化丢失 99.8% 信息）的完全重写版。

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
确定性再生（seed=42 + events.csv 固定），产出 train 2884 / val 366 / test 330。

## 当前阶段

数据 pipeline ✅ 完成 → **下一步：训练首次 E1**（详见 CLAUDE.md §7）。

## 目录

```
data_pipeline/   数据生成（含 events.csv 自包含）
docs/            研究方案（02_research_design.md 为权威）
output/          生成的 strain/spectrogram/dataset（大文件 .gitignore）
CLAUDE.md        迁移交接文档（必读）
```
