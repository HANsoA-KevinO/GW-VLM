# GW-TF Research Track 2：模型选型与工具链调查报告

**版本**：v1.0
**日期**：2026-05-04
**作者**：Yu Tangxuan
**目的**：为基于 foundation-model-era 多模态大模型的引力波检测研究项目，确定模型选型、训练硬件分工和微调工具链；本文档作为后续实验方案设计与代码实现的依据

---

## 0. 文档导航

本文档分六部分：

1. **行业现状**：VLM 在科学信号 / spectrogram 任务上的微调实践
2. **架构概念**：原生多模态（Native Multimodal）vs 拼接式 VLM 的本质区别
3. **候选模型**：Qwen 多模态系列时间线与命名澄清
4. **硬件方案**：DGX Spark + RTX 5090 双机协同的可行性
5. **核心模型对比**：Qwen3.6-27B (Dense) vs Qwen3.6-35B-A3B (MoE) 的研究适配度
6. **微调工具链成熟度**：Unsloth / LLaMA-Factory 等支持现状与已知坑点

文末附 **决策摘要** 和 **下一步工作清单**。

---

## 1. 行业现状：VLM 在科学信号上的微调实践

### 1.1 直接相关的研究证据

经 2026-05 联网调查，发现以下**与本研究高度相关**的工作：

- **VLM 直接吃 spectrogram 图像分类已被验证可行**
  - Vision Language Models Are Few-Shot Audio Spectrogram Classifiers (arXiv:2411.12058) — GPT-4o 在 ESC-10 环境声分类数据集上达到 59% 准确率，**超过商业音频专用模型 Gemini-1.5**
  - 验证了"把信号当图像处理"这一思路的工程可行性

- **GW 领域已有 ViT 做 spectrogram 分类的先例**
  - SEMD (arXiv:2508.19311) — 用 DeiT-Tiny + Squeeze-and-Excitation 做强透镜引力波识别
  - 输入是 CQT 时频谱图对，把 GW 检测重新表述为"二分类图像比对问题"
  - **任务方向是 lensing 识别，不是 detection；用的是 ViT，不是预训练 VLM**

- **VLM 微调的工程链已经标准化**
  - LoRA 微调 Qwen2.5-VL 在 5K-50K 样本上只需 \$100-\$5000 算力
  - BioCLIP + LoRA 在生物图像上达 94% 准确率（不需要数据增强或过采样）

### 1.2 本研究填补的空白

| 已有工作 | 本研究的空白填补 |
|---|---|
| VLM 做音频 spectrogram 分类 (2411.12058) | 没人在 **GW spectrogram** 上做 VLM 微调 |
| ViT 做 GW lensing 分类 (SEMD) | 用的是 ViT，不是预训练 VLM；任务是 lensing，不是 **detection** |
| LLM 文本 token 化做 GW detection (Li 2026) | 文本 token 化丢 99.8% 信息，**VLM 不会丢** |
| Foundation 模型做 GW PE (Dingo-T1, GraviBERT) | 都是参数估计任务，**不是 detection**；都是 strain time-series，**不是 spectrogram 图像** |

**研究 niche**：用预训练**原生多模态 / VLM** 做 **GW detection** 的 **CQT 图像微调**——目前是真空地带。

---

## 2. 架构概念：原生多模态 vs 拼接式 VLM

### 2.1 本质区别

二者**不是同一类东西**，是两种**根本不同的设计哲学**：

| 维度 | 传统 VLM（拼接式 / Late Fusion） | 原生多模态（Native Multimodal / Early Fusion） |
|---|---|---|
| **架构来源** | 预训练 vision encoder（如 CLIP-ViT）+ projector + 预训练 LLM **缝合** | 从训练第 1 步就用统一架构同时见所有模态 |
| **数据流** | 图像 → vision encoder → projector → LLM 文本空间 | 图像直接 token 化 → 与文本 token 拼成单序列 → 一个 transformer 处理 |
| **跨模态交互** | 视觉信息已被压缩成"语言友好"形式才进入 LLM，深层视觉细节有损失 | 文本 token 和视觉 token 在 attention 层**直接互看**，跨模态推理更深 |
| **代表模型** | LLaVA、Qwen-VL（早期）、InternVL、CLIP+LLM 系列 | **GPT-4o**、**Gemini 1.5/2**、**Llama 4**、**Chameleon**、**Qwen3.6**、SenseNova NEO |
| **训练成本** | 视觉端可冻结，便宜 | 必须从头联训，贵几个数量级 |
| **Scaling 行为** | 性能上限受 projector 瓶颈限制 | 早融合需要更少参数即可达到相同 loss（参考 arXiv:2504.07951） |

### 2.2 典型原生多模态架构

- **Chameleon (Meta, 2024)**：用 VQ-VAE 把图像量化成 token，与文本 token 一起塞进同一 transformer，从零联合预训练
- **Llama 4 (Meta, 2025)**：早融合 + MoE + 增强版 MetaCLIP，视觉/文本 token 从 layer 0 拼成同一序列
- **Qwen3.6 (Alibaba, 2026-04)**：原生多模态 + Hybrid attention（DeltaNet + Gated Attention），是本研究**主要候选**

### 2.3 关键澄清

> **Native multimodal** 和 **Dense** 是**正交概念**，可以同时成立。
>
> Qwen3.6-27B = "原生多模态 dense 模型" = 从零联合训练所有模态 + 所有参数都激活。

---

## 3. 候选模型：Qwen 多模态系列时间线

### 3.1 命名时间线（截至 2026-05）

| 模型系列 | 发布时间 | 类型 | 状态 |
|---|---|---|---|
| Qwen2.5-VL (7B/72B) | 2024 | 拼接式 VLM | 第一代主流 |
| Qwen3-VL (2B/4B/8B/32B + 30B-A3B/235B-A22B) | 2025-10 起 | **深度集成的 VLM**（DeepStack 等） | 主力开源版本 |
| Qwen3.5 / Qwen3.5-VL | 2026 初 | 改进版 | 工具链支持成熟 |
| **Qwen3.6-Plus** | **2026-03-31** | **多模态** | 旗舰，闭源 API |
| **Qwen3.6-35B-A3B** | **2026-04-16** | **原生多模态 MoE**（35B 总 / 3B 激活） | **Apache 2.0 开源** |
| **Qwen3.6-Max-Preview** | **2026-04-20** | 旗舰 | 闭源 API |
| **Qwen3.6-27B** | **2026-04-22** | **原生多模态 Dense + Vision Encoder** | **Apache 2.0 开源** |

### 3.2 关键判断

- **Qwen3-VL** 严格说还是"深度集成的拼接式 VLM"（DeepStack 把多层 ViT 特征喂进 LLM）
- **Qwen3.6 才是真正的 "natively multimodal"**（呼应 Llama 4 / Chameleon 那一代范式）
- **本研究主力候选是 Qwen3.6 系列**（27B Dense 和 35B-A3B MoE 两个变体）

---

## 4. 硬件方案：DGX Spark + RTX 5090 双机协同

### 4.1 DGX Spark 关键规格

| 参数 | 规格 |
|---|---|
| 芯片 | GB10 Grace Blackwell Superchip（CPU + GPU 单封装） |
| CPU | 20 核 ARM（10× Cortex-X925 + 10× Cortex-A725） |
| GPU 架构 | Blackwell，第 5 代 Tensor Cores |
| **统一内存** | **128GB LPDDR5x**（CPU/GPU 共享） |
| 内存带宽 | 273 GB/s |
| AI 算力 | 1 PetaFLOP @ FP4 |
| CPU↔GPU 互联 | NVLink-C2C @ 900 GB/s |
| TDP | 300W |
| 官方支持 | 微调最大 **70B 参数**模型；推理最大 **200B 参数**模型 |
| 价格 | $3,000-4,699 |

### 4.2 DGX Spark vs RTX 5090 对照

| 维度 | RTX 5090 (32GB) | DGX Spark (128GB) | 谁赢 |
|---|---|---|---|
| 显存容量 | 32GB GDDR7 | **128GB LPDDR5x** | **DGX Spark 4×** |
| 显存带宽 | **1792 GB/s** | 273 GB/s | **5090 6.5×** |
| AI FLOPS (FP4) | ~3.4 PetaFLOPS | 1 PetaFLOP | **5090 3.4×** |
| 单 token 推理速度 | **快**（带宽吃饱） | 慢（带宽瓶颈） | **5090** |
| 模型规模上限 | ~13B QLoRA 极限 | **70B bf16 LoRA**（无需量化） | **DGX Spark** |
| 训练精度 | QLoRA 4-bit | **bf16 全精度** | **DGX Spark** |
| 功耗 | 575W | 300W | DGX Spark |

### 4.3 双机分工

| 机器 | 角色 | 任务 |
|---|---|---|
| **RTX 5090 (32GB)** | **快迭代机 + 推理评估机** | (a) 小模型快速调试 pipeline（Qwen3-VL-8B QLoRA 跑通流程）；(b) 训练完成后高速推理评估 |
| **DGX Spark (128GB)** | **主训练机** | 跑 Qwen3.6-27B / 35B-A3B 的 **bf16 LoRA** 微调；每个 step 慢但**无量化精度损失** |

### 4.4 DGX Spark 在本研究中的关键价值

> **DGX Spark 是这两个候选模型唯一能做 bf16 LoRA 的设备。**
>
> - 5090 上必须 QLoRA 4-bit，但 Qwen 官方明确**不推荐** Qwen3.5/3.6 用 QLoRA（量化敏感）
> - DGX Spark 上 bf16 全精度无障碍，**梯度质量更高 → 研究结论更可靠**
> - 已有人在 DGX Spark + Qwen3.5-35B-A3B + Unsloth + bf16 LoRA 配置下跑通（[NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/bf16-lora-fine-tuning-of-qwen3-5-35b-a3b-on-dgx-spark-no-quantization-required/363268) + [GitHub: kreuzhofer/dgx-spark-unsloth-qwen3.5-training](https://github.com/kreuzhofer/dgx-spark-unsloth-qwen3.5-training)）

---

## 5. 核心模型对比：Qwen3.6-27B (Dense) vs Qwen3.6-35B-A3B (MoE)

### 5.1 概念澄清

#### "Dense" 是什么意思
- 每次前向传播时**所有参数都参与计算**
- Qwen3.6-27B：27B 参数全部激活
- 简单、稳定、可预测；训练和微调相对容易

#### "MoE (A3B)" 是什么意思
- A3B = **A**ctivated **3B** = 激活只有 3B 参数
- Qwen3.6-35B-A3B：256 个 expert，每 token 激活其中 9 个（约 3B 参数）
- **router（路由器）** 动态决定哪些 expert 参与
- 推理快（只算 3B），但训练复杂（router 路由 + 负载均衡 + 专家利用率都要管）

类比：Dense = 全能员工；MoE = 256 人团队 + 调度员

### 5.2 详细对比表

| 维度 | **Qwen3.6-27B (Dense)** | **Qwen3.6-35B-A3B (MoE)** |
|---|---|---|
| 总参数 | 27B | 35B |
| **激活参数** | **27B（全部）** | **3B（9/256 个 expert）** |
| 架构 | 64 层 hybrid（Gated DeltaNet × 3 + Gated Attention × 1）× 16 块 | 10 层 × (DeltaNet→MoE × 3 + Attention→MoE × 1) |
| 训练稳定性 | **稳定、简单** | 需要 router z-loss 等技巧 |
| 微调难度 | **简单**，标准 LoRA 即可 | 复杂，router 易过拟合（Unsloth 默认禁用 router 微调） |
| 推理速度 | 慢 | **快 3-4×**（只算 3B） |
| 综合质量（Qwen 官方基准） | **73 分（综合平均）** | 67 分 |
| 多模态基准 MMMU | **82.9** | 81.7 |
| 显存占用（推理） | 全 27B 都要驻留 | 仍需 35B 都驻留（**MoE 不省显存**） |
| **bf16 LoRA 显存** | ~75-80GB（估计） | **74GB（实证数据）** |

### 5.3 常见误区澄清

#### 误区 1：MoE 显存更省？
**错。** MoE 在显存占用上**和总参数挂钩，不是和激活参数挂钩**。
- Qwen3.6-35B-A3B 推理时需要把所有 35B 权重都加载到显存里（router 不知道下一秒要用哪个 expert）
- MoE 省的是**算力**和**带宽消耗**，不是**显存**

#### 误区 2：MoE 训练比 Dense 快？
**部分对。**
- 预训练时 MoE 比 Dense 更 compute-efficient
- **但微调时 MoE 更难**：研究界普遍发现 MoE 微调容易过拟合，generalization 差
- HuggingFace 官方原话："MoEs have historically struggled to generalize during fine-tuning, leading to overfitting"

#### 误区 3：参数多的肯定更强？
**不一定。** 在同代 Qwen3.6 中，**Dense 27B (73 分) > MoE 35B-A3B (67 分)**。激活参数才是能力上限的主要决定因素。

### 5.4 对本研究的适配度评估

| 我们的需求 | Qwen3.6-27B (Dense) | Qwen3.6-35B-A3B (MoE) |
|---|---|---|
| 训练稳定性（避免 router 不收敛） | ✅ 没这问题 | ⚠️ 需要监控 router 健康 |
| 多种子复现实验（论文需要） | ✅ 简单 | ⚠️ 不同种子可能 router 不同行为 |
| 微调过拟合风险 | ✅ 标准风险 | ⚠️ 文献明确 MoE 更易过拟合 |
| 多模态质量（视觉基准） | ✅ MMMU 82.9 | ✅ MMMU 81.7（差不多） |
| DGX Spark 128GB bf16 LoRA 可行性 | ✅ 充裕 | ✅ 充裕 |
| 训练速度 | 慢 | 快 3-4× |
| 推理速度（评估时） | 慢 | 快 3-4× |
| 论文可解释性 | ✅ 注意力可视化等标准方法 | ⚠️ 还要解释 expert 路由模式 |

### 5.5 决策

**主要实验对象：Qwen3.6-27B (Dense)**

理由：
1. 任务本质太简单（二分类）不需要 MoE 的容量分配机制
2. 微调稳定性是研究质量的命脉
3. 27B 综合质量更高
4. 可发表性更好（dense 模型审稿人不需要解释 router 行为）
5. DGX Spark 显存完全够

**对照实验对象：Qwen3.6-35B-A3B (MoE)**

理由：
1. 在论文里加"我们也对比了 MoE，但 dense 表现更好"作为消融对照
2. 这一对照本身可以成为研究贡献："**dense 模型在小样本特定领域微调场景下显著优于 MoE**，与 MoE 在大规模预训练上的优势形成对比"
3. 为 AI-for-science 中的模型选型提供实证证据

---

## 6. 微调工具链成熟度评估

### 6.1 工具支持矩阵

| 工具 | 27B Dense 支持 | 35B-A3B MoE 支持 | 多模态支持 | 成熟度 |
|---|---|---|---|---|
| **Unsloth**（推荐） | ✅ 官方文档 unsloth.ai/docs/models/qwen3.6 | ✅ 官方 GGUF 已上传 | ✅ 含视觉微调 | ⭐⭐⭐ 最成熟 |
| **LLaMA-Factory** | ✅ 通用 LoRA 配置可用 | ✅ MoE 通用支持 | ✅ multimodal SFT 已支持 | ⭐⭐ 已熟悉，但 Qwen3.6 专用配置示例较少 |
| **HuggingFace Transformers + TRL** | ✅ 标准 PEFT 路径 | ✅ 标准路径 | ✅ TRL 有 VLM cookbook | ⭐⭐ 灵活但要自己拼 |
| **MS-Swift** | ✅ Qwen3-VL Best Practices 文档完善 | ✅ | ✅ | ⭐⭐ 阿里官方背书 |

### 6.2 工具链关键事实

> Unsloth 官方原话："To train Qwen3.6, you can refer to the previous Qwen3.5 fine-tuning guide" —— **Qwen3.5 → Qwen3.6 配置兼容**，3.5 的微调脚本基本可以直接移植。

### 6.3 已跑通的实证案例

#### 案例 A：DGX Spark + Qwen3.5-35B-A3B + Unsloth + bf16 LoRA
- 来源：NVIDIA Developer Forums 官方帖子 + GitHub kreuzhofer 项目
- 配置：**bf16，无需量化**，单台 DGX Spark
- **完全对应本研究计划的硬件 + 模型 + 精度组合**

#### 案例 B：Qwen3-VL-30B-A3B MoE + LoRA
- 来源：Medium 文章
- 提供完整 LoRA target module 配置和坑点列表

#### 案例 C：Qwen3.5 MoE vs Dense 对比微调
- 来源：Medium "Fine-Tuning Qwen3.5: MoE vs Dense"
- 直接对应本研究的对比实验

### 6.4 关键技术配置

#### LoRA 配置（Dense）
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
r = 16-64                  # Rank
lora_alpha = r             # Unsloth 推荐：alpha = r（不是 2*r）
lora_dropout = 0.05-0.1
use_gradient_checkpointing = "unsloth"   # 而不是 True，省更多 VRAM
```

#### LoRA 配置（MoE 额外注意）
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",
                  "gate_up_proj"]    # MoE 专属：fused MoE projection
# 不微调 router 层（Unsloth 默认禁用，避免破坏 expert 路由）
```

#### 量化精度选择
**官方明确建议**：
> "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5/3.6 models, no matter MoE or dense, due to higher than normal quantization differences. Instead, bf16 setups are recommended."

**对应本研究**：DGX Spark 128GB unified memory 让我们**不需要量化**，正好规避这一限制。

#### 序列长度限制
- **Qwen3.6-35B-A3B 在单卡上**：safe ceiling = **2048 tokens**（4096 会 OOM）
- **27B Dense 在单卡上**：估计也类似 2048
- **本研究够用**：CQT 谱图 vision token ~256-1280 + 简短文本指令 + "0/1" 输出，总 sequence length 远低于 2048

### 6.5 显存需求

| 模型 | bf16 LoRA VRAM | DGX Spark (128GB) | 5090 (32GB) |
|---|---|---|---|
| **Qwen3.6-27B (Dense)** | ~75-80GB（估计） | ✅ 充裕 | ❌ 不够 |
| **Qwen3.6-35B-A3B (MoE)** | **74GB（实证）** | ✅ 充裕 | ❌ 不够 |

### 6.6 已知踩坑清单

#### MoE 通用坑
1. **Sequence length 别贪心**：单 H100 80GB 上 4096 都会 OOM；用 2048
2. **Router 别动**：Unsloth 默认禁用 router 微调，**不要打开**
3. **必须设置环境变量**：`UNSLOTH_COMPILE_DISABLE=1`（解决混合精度错误）
4. **Dataloader 单进程**：`dataloader_num_workers=0`、`dataset_num_proc=1`（防多进程死锁）

#### 多模态通用坑
1. **使用 FastModel 不是 FastLanguageModel**：Qwen3.5+ 多模态版本 Unsloth 用 FastModel
2. **Tokenizer 要从 processor 取**：`processor.tokenizer`，而不是直接 `tokenizer`
3. **Vision dependencies**：torchvision + Pillow 必装

#### Qwen3.6 特定坑
1. **Hybrid thinking 模式**：Qwen3.6 有"thinking" 和 "non-thinking" 两种推理模式，**本研究用 non-thinking**（输出"0"或"1"，不需要思考链）
2. **Preserve Thinking**：Qwen3.6 新特性，对单轮分类无影响

### 6.7 数据格式

#### Qwen3-VL 多模态 JSONL 格式（推荐）
```json
{
  "messages": [
    {"role": "system", "content": "You classify gravitational wave signals."},
    {"role": "user", "content": [
      {"type": "image", "image": "samples/000001_h1l1v1.png"},
      {"type": "text", "text": "Does this CQT spectrogram contain a gravitational wave signal? Answer 0 or 1."}
    ]},
    {"role": "assistant", "content": "1"}
  ]
}
```

#### LLaMA-Factory ShareGPT 多模态变体（备选）
```json
{
  "messages": [
    {"role": "user", "content": "<image>Does this CQT spectrogram contain a GW signal?"},
    {"role": "assistant", "content": "1"}
  ],
  "images": ["samples/000001_h1l1v1.png"]
}
```

---

## 7. 决策摘要

### 7.1 模型选型

| 角色 | 模型 | 理由 |
|---|---|---|
| **主要实验对象** | **Qwen3.6-27B (Dense)** | 训练稳定 + 综合质量更高 + 微调简单 + DGX Spark 内存够 |
| **对照实验对象** | **Qwen3.6-35B-A3B (MoE)** | 形成 dense vs MoE 对比，本身是有发表价值的方法学贡献 |
| **Pipeline 调试** | **Qwen3-VL-8B** | 5090 上 QLoRA 快速跑通数据流，作为开发调试用 |

### 7.2 硬件分工

| 机器 | 角色 |
|---|---|
| **DGX Spark (128GB)** | 主训练机：bf16 LoRA 微调 27B / 35B-A3B |
| **RTX 5090 (32GB)** | 调试机 + 评估推理机：8B 调 pipeline；训完后高速推理 |
| **Mac (本机)** | 代码开发、数据预处理、配置文件管理 |

### 7.3 工具链选择

| 任务 | 工具 |
|---|---|
| 主微调框架 | **Unsloth**（最成熟、官方推荐、DGX Spark 适配） |
| 备选 | LLaMA-Factory（已熟悉，但 Qwen3.6 配置示例少） |
| 数据格式 | Qwen3-VL 多模态 JSONL（标准格式） |
| 训练精度 | **bf16 LoRA**（DGX Spark 唯一能做的，且官方推荐） |

### 7.4 已确认事项

✅ 工具链对 Qwen3.6 的支持已经成熟（Unsloth 有官方 GGUF + 文档 + Colab）
✅ DGX Spark + Qwen3.5-35B-A3B + Unsloth + bf16 LoRA 已有人完整跑通
✅ Qwen3.5 → Qwen3.6 微调配置兼容
✅ MoE vs Dense 对比微调有 prior art 可参考
✅ 多模态 JSONL 数据格式成熟、有大量教程
✅ DGX Spark 128GB 显存对 bf16 LoRA 充裕
✅ 本研究填补的 niche（VLM + GW spectrogram + detection）是真空地带

### 7.5 仍需在 Stage 0 确认的点

⚠️ Qwen3.6-27B 多模态微调的具体显存数据（估计 75-80GB，需 dry-run 确认）
⚠️ Qwen3.6 hybrid thinking 模式在多模态分类任务的最佳处理方式（建议禁用 thinking）
⚠️ Qwen3.6 Unsloth 视觉微调的最新支持状态（27B 是 2026-04-22 才发布）

---

## 8. 实验方案总览（草案）

### 8.1 Stage 划分

| 阶段 | 模型 | 机器 | 训练精度 | 预期时长 | 目标 |
|---|---|---|---|---|---|
| **Stage 0** | Qwen3-VL-8B | 5090 | QLoRA 4-bit | 数小时 | 跑通完整 pipeline、调试数据格式 |
| **Stage 1** | **Qwen3.6-27B (Dense)** | **DGX Spark** | **bf16 LoRA** | 1-3 天 | **主实验**，作为 baseline |
| **Stage 2** | **Qwen3.6-35B-A3B (MoE)** | **DGX Spark** | **bf16 LoRA** | 1-3 天 | **对照实验**，dense vs MoE |
| **Stage 3** | Stage 1/2 模型变体 | DGX Spark + 5090 | 混合 | 1-2 周 | 消融（CQT 格式、图像组合方式、prompt 模板等） |
| **Stage 4** | Stage 1/2 训完模型 | 5090 | bf16 推理 | 数小时 | 大规模评估（按 MLGWSC 标准：FAR=1/月 灵敏距离） |

### 8.2 论文叙事框架

> "我们对比了 Qwen3.6 系列的 dense (27B) 和 MoE (35B-A3B) 两种架构在 GW spectrogram 检测任务上的表现。**dense 模型在我们的小样本特定领域微调场景下显著优于 MoE**，这与 MoE 在大规模预训练上的优势形成对比，为 AI-for-science 中的模型选型提供了实证证据。"

---

## 9. 下一步工作清单

本文档完成后，下一步工作如下：

### 9.1 立即开始（数据处理与训练数据设计）

- [ ] **数据 pipeline 重新设计**：原项目的 CQT + K-Means 流水线必须废弃，重新设计直接保留 CQT 谱图为图像的 pipeline
- [ ] **CQT 谱图打包方式**：3 个探测器（H1/L1/V1）→ 3 张图 vs RGB 通道合成 vs 拼接图 vs 其他
- [ ] **图像分辨率与 token 数权衡**：Qwen3-VL token 范围 256-1280，与 CQT 时间-频率分辨率的对应关系
- [ ] **数据增强策略**：旋转/翻转对 spectrogram 不适用，需要专用增强（噪声扰动、时间平移等）
- [ ] **类别平衡处理**：原项目 52% 全预测正类暴露的不平衡问题
- [ ] **Train/Val/Test 划分**：尤其要避免数据泄漏（同一物理事件的不同时间窗）
- [ ] **Prompt 模板设计**：system + user prompt 怎么写最有利于二分类
- [ ] **MLGWSC 评估对齐**：FAR=1/月 灵敏距离的计算方式

### 9.2 短期（Stage 0 阶段确认）

- [ ] 在 5090 上跑通 Qwen3-VL-8B + Unsloth + LoRA 的完整 pipeline
- [ ] 验证 Qwen3.6 thinking/non-thinking 模式选择
- [ ] DGX Spark 上 dry-run Qwen3.6-27B，确认显存占用

### 9.3 中期（Stage 1-2 主实验）

- [ ] DGX Spark 上跑 Qwen3.6-27B bf16 LoRA 主实验
- [ ] 多种子（≥3）训练，报告均值±方差
- [ ] DGX Spark 上跑 Qwen3.6-35B-A3B bf16 LoRA 对照
- [ ] 5090 高速推理评估

### 9.4 长期（Stage 3-4 消融与评估）

- [ ] 消融实验完成
- [ ] MLGWSC 标准评估
- [ ] 论文撰写

---

## 附录：参考资源链接

### 模型文档
- Qwen3.6-27B HF: https://huggingface.co/Qwen/Qwen3.6-27B
- Qwen3.6-35B-A3B HF: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.6 Unsloth 文档: https://unsloth.ai/docs/models/qwen3.6
- Qwen3.5 Unsloth 微调指南（Qwen3.6 兼容）: https://unsloth.ai/docs/models/qwen3.5/fine-tune

### 实证案例
- DGX Spark + Qwen3.5-35B-A3B 实证（NVIDIA 论坛）: https://forums.developer.nvidia.com/t/bf16-lora-fine-tuning-of-qwen3-5-35b-a3b-on-dgx-spark-no-quantization-required/363268
- kreuzhofer/dgx-spark-unsloth-qwen3.5-training: https://github.com/kreuzhofer/dgx-spark-unsloth-qwen3.5-training
- Fine-Tuning Qwen3.5 MoE vs Dense（Medium）: https://medium.com/@ishaafsalman/qwen3-5-fine-tuning-in-2026-moe-vs-dense-b2d17de73a9e

### 硬件文档
- NVIDIA DGX Spark 官方页: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- DGX Spark User Guide: https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf
- Unsloth on DGX Spark: https://build.nvidia.com/spark/unsloth
- Fine-tuning LLMs with DGX Spark and Unsloth: https://docs.unsloth.ai/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth

### 相关研究
- Vision Language Models Are Few-Shot Audio Spectrogram Classifiers: https://arxiv.org/abs/2411.12058
- SEMD（GW lensing 用 ViT）: https://arxiv.org/abs/2508.19311
- Li 2026（LLM-GW，原项目论文）: https://arxiv.org/abs/2512.04031
- Chameleon（原生多模态架构代表）: https://arxiv.org/abs/2405.09818
- Scaling Laws for Native Multimodal Models: https://arxiv.org/pdf/2504.07951

### 工具链
- Unsloth GitHub: https://github.com/unslothai/unsloth
- LLaMA-Factory: https://github.com/hiyouga/LlamaFactory
- Qwen3-VL Best Practices (MS-Swift): https://swift.readthedocs.io/en/latest/BestPractices/Qwen3-VL-Best-Practice.html

---

**文档结束。**
**下一份文档**：`02_data_pipeline_design.md` —— 实验数据处理与训练数据设计方案
