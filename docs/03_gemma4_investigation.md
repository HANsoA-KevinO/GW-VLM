# Gemma 4 调查报告 —— 纳入 GW-VLM 模型对比

**日期**：2026-06-15
**目的**：评估 Google Gemma 4（开源原生多模态）能否纳入本项目作为 Qwen 之外的第二个 VLM 对比族，重点回答**能不能做（视觉）微调**，并确定选用变体。

---

## 0. 结论速览（TL;DR）

- **能微调，且一线支持**：Gemma 4 全系开放权重（Apache 2.0），支持图像输入；**Unsloth day-zero 支持 vision/text LoRA+QLoRA**，HF Transformers + TRL 亦可。
- **选型**：
  - **主对比 = Gemma 4 31B Dense**（`google/gemma-4-31B-it`）——与 Qwen3.6-27B 规模对位做"公平对决"，DGX Spark 128GB bf16 LoRA 充裕。
  - **调试 = Gemma 4 E4B**（`google/gemma-4-E4B-it`）——vision LoRA 确认可用、显存小（~17GB）、跑通管线最快。
  - **后续候选 = Gemma 4 12B "Unified"**（编码器无关），见 §3。
- **科学价值**：把论文叙事从"Qwen 单族"扩成**三方对比**——Qwen3-VL（vision-encoder/DeepStack 拼接式）vs Qwen3.6（原生多模态）vs **Gemma 4（另一独立开源原生多模态族）**，强化"原生多模态做 GW 检测"主线、提升结论普适性。

---

## 1. Gemma 4 是什么

Google 2026 年发布的开源多模态模型族（"byte for byte, the most capable open models"）。文本 + **图像**输入（部分小模型含音频），文本输出。开放权重，含 base 与 instruction-tuned（`-it`）两版，Apache 2.0 商用友好。

**图像理解能力**：目标检测、文档/PDF 解析、图表理解、OCR（多语种）、手写识别、UI/屏幕理解，并可按帧分析视频。对我们"读 Q-transform 时频图判断 chirp 形态"的任务是强匹配的视觉先验。

**上下文**：E2B/E4B 128K，12B/26B-A4B/31B 256K。多语种 140+。内置 reasoning（thinking）模式——**本项目用 non-thinking**（只输出 JSON，不要思维链）。

---

## 2. 变体全表

| 变体 | 参数 | 架构 | 上下文 | 模态 | HF model ID（base / -it） |
|---|---|---|---|---|---|
| **E2B** | 2.3B 有效 / 5.1B | Dense | 128K | 文/图/音 | `google/gemma-4-E2B` · `-E2B-it` |
| **E4B** | 4.5B 有效 / 8B | Dense | 128K | 文/图/音 | `google/gemma-4-E4B` · `-E4B-it` |
| **12B Unified** | ~12B | **Dense，编码器无关** | 256K | 文/图/音(/视频) | `google/gemma-4-12B` · `-12B-it` |
| **26B-A4B** | 4B 激活 / 26B 总 | MoE | 256K | 文/图/视频 | `google/gemma-4-26B-A4B` · `-26B-A4B-it` |
| **31B** | 31B | Dense | 256K | 文/图/视频 | `google/gemma-4-31B` · `-31B-it` |

- "E" = effective（用 Per-Layer Embeddings 提效）；E2B/E4B 面向边缘/浏览器。
- 全系支持图像输入；图像 token 预算可配（70/140/280/560/1120）。
- MoE（26B-A4B）显存按**总参数**算（26B 全驻留），省的是算力不是显存——与 Qwen MoE 同理。

---

## 3. 12B "Unified"：编码器无关（为何对本项目特别有意思）

传统 VLM（含 Qwen3-VL）= 独立 vision encoder（如 ViT/CLIP）→ projector → LLM。Gemma 4 12B **去掉独立视觉/音频编码器**，把"原始图像 patch / 原始音频波形**直接投影**进 LLM embedding 空间"（一个轻量模块：单次矩阵乘 + 位置编码 + 归一化）。

这正是本项目 [`docs/01`](01_model_and_toolchain_investigation.md) §2 论证的 **early-fusion / 原生多模态**范式的极致形态——视觉 token 与文本 token 从 layer 0 在同一注意力里互看，跨模态推理更深，且省显存（12B 在 16GB 上可跑，benchmark 接近 26B）。

→ **论文上很有料**：可对比"编码器无关(12B) vs 有视觉编码器(31B/Qwen3-VL)"在 chirp 形态识别上的差异。本轮先以 31B 做规模对位主对比，12B 列为后续消融候选。

---

## 4. 能不能微调？——能（核心问题）

| 框架 | Gemma 4 支持 | 视觉微调 | 备注 |
|---|---|---|---|
| **Unsloth**（选用） | ✅ day-zero 全系 | ✅ `FastVisionModel`，可选只调 vision / language / attn / MLP 层 | 比 FA2 快 ~1.5×、省 ~60% 显存，精度无损 |
| **HF Transformers + TRL** | ✅ | ✅ | 编码器无关使"图/音/文共享同一权重，LoRA 一次性更新整条多模态回路" |
| LLaMA-Factory / MS-Swift | 部分 | — | 回退选项 |

**LoRA/QLoRA 显存（Unsloth 实测/文档）**：

| 变体 | 任务 | 显存 | 在 DGX Spark (128GB) |
|---|---|---|---|
| E2B | LoRA | 8–10GB | ✅ |
| **E4B** | LoRA(文) / QLoRA | 17GB / 10GB | ✅（调试用）|
| 26B-A4B | LoRA(16-bit) | >40GB | ✅ |
| **31B** | QLoRA | 22GB（→ bf16 LoRA 也充裕）| ✅（主对比用）|

**关键配置（Unsloth）**：
- 视觉微调 `FastVisionModel.from_pretrained(...)` → `get_peft_model(finetune_vision_layers=True, finetune_language_layers=True, r=32, lora_alpha=32)`。
- 数据格式：`messages` 会话，user content = `[{"type":"image",...},{"type":"text",...}]`，**图像必须在文本之前**（Gemma 强制；Qwen 兼容）→ 本项目 `08_export_training_format.py` 已按此统一导出，Qwen/Gemma 共用一份 jsonl。
- chat 模板：`get_chat_template(tokenizer, "gemma-4")`（或 thinking 变体）。
- 已知坑：E2B/E4B 正常 loss 偏高(13–15)、视觉 3–5 属正常；生成时 `use_cache=True`；31B/26B-A4B 早期有 `num_kv_shared_layers=0` 的 IndexError，Unsloth 已修。

---

## 5. 在本项目中的落位

| 角色 | Qwen 族 | Gemma 4 族（新增） |
|---|---|---|
| 调试 / 跑通链路 | Qwen3-VL-8B | **Gemma 4 E4B** |
| 主对比（Dense） | Qwen3.6-27B | **Gemma 4 31B** |
| MoE 对照（可选） | Qwen3.6-35B-A3B | （26B-A4B，暂不纳入）|
| 编码器无关（后续候选）| — | Gemma 4 12B Unified |

- 训练机：**DGX Spark**（`spark-91b6.local`），bf16 LoRA。
- 实验矩阵不变（E1–E4），只是每路在两个模型族上各跑一份用于横向对比。
- 本轮先用 **E4B 跑通 E1 端到端**，再扩 31B / Qwen3.6-27B。

---

## 6. 风险 / 待核验

- 31B 的**视觉微调**在 Unsloth 当前版本的成熟度以 Spark dry-run 为准；若受限则回退 HF TRL（E4B 路径已确认无虞，足以先验证链路）。
- thinking 模式：分类任务禁用，只训最终 JSON。
- Gemma 4 与 Qwen 的 1024×1024 图像 token 数不同 → `max_seq_length` 以各自 processor 实测为准（预算 ≤2048 应足够）。

---

## 7. 实测结果(2026-06，E1 纯检测)

确认 **Gemma 4 可在 DGX Spark 上 LoRA 微调**(Unsloth `FastVisionModel`,bf16/4bit 均可)。E1 实测:

| 变体 | 精度 | ROC-AUC | 备注 |
|---|---|---|---|
| **31B(调参版)** | bf16 LoRA r32 | **0.922** | 当前最佳;FPR≤5% 下 recall 75% |
| E4B | 4bit QLoRA r32 | 0.904 | 小模型很能打、成本低(~2.5h) |
| 31B(基线 r8/lr5e-5) | bf16 LoRA | 0.858 | 配方太保守 → 欠拟合(反面教材) |

实测要点:
- **必须装 torchvision**(Gemma4 图像处理器依赖),否则处理器 import 失败。
- Gemma4 输入分辨率按 **token 预算**(70/140/280/560/1120),Unsloth 默认兜底 512px 偏低;用 `image_tokens=560`(≈1135px)吃满 1024 源。
- bf16 大模型在 GB10 统一内存上需 `ACCELERATE_BYPASS_DEVICE_MAP=true`;torch2.12+gemma4 的 dynamo 编译要关(`UNSLOTH_COMPILE_DISABLE=1`,走 eager)。
- 完整记录见 [`04_e1_experiments_and_findings.md`](04_e1_experiments_and_findings.md)。

## 附：参考链接

- HF 博客 Welcome Gemma 4：https://huggingface.co/blog/gemma4
- Gemma 4 core 文档：https://ai.google.dev/gemma/docs/core
- 12B 编码器无关：https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
- Unsloth Gemma 4 微调指南：https://unsloth.ai/docs/models/gemma-4/train
- Unsloth 视觉微调：https://unsloth.ai/docs/basics/vision-fine-tuning
- HF 模型卡：`google/gemma-4-31B-it`、`google/gemma-4-E4B-it`、`google/gemma-4-12B-it`
