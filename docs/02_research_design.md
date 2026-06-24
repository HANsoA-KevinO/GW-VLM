# GW-VLM 研究实验方案

**版本**：v0.1（草稿）
**日期**：2026-05-06
**作者**：Yu Tangxuan
**目的**：定义 GW-VLM 项目的完整实验设计——数据、模型、训练、评估、对照——作为后续代码实现与论文撰写的依据

---

## 0. 文档结构

| 章节 | 内容 |
|---|---|
| 1 | 研究目标与科学问题 |
| 2 | 数据集设计（训练 / 测试） |
| 3 | 模型与训练阶梯 |
| 4 | 输入 / 输出格式（prompt + schema） |
| 5 | 4 路消融实验矩阵 |
| 6 | 评估方法论与 baseline 对照 |
| 7 | 评估指标 |
| 8 | 工程实现与时间表 |
| 9 | 待商榷细节（v0.2 解决） |

---

## 1. 研究目标与科学问题

### 主问题
**预训练 VLM 通过对 Q-transform 时频图的微调，能否在 GW 检测与粗粒度参数估计上达到与 T2 (CNN) 路径相当甚至更好的性能？**

### 子问题（对应 protocol v1.1 的 RQ）
- **RQ8 (Pipeline Readiness)**：VLM 在 readiness × modernity 矩阵中处于哪个格子？
- **RQ9 (Modernity Penetration)**：T4-vision (我们) 与 T2/T3 早期方法在 GW 任务上的强弱对比？
- **本研究新增**：纯检测 vs 多任务（检测+参数）的能力代价权衡；真实数据 vs 注入扩充的边际增益

### 评估双轨
- **轨 A — 性能基线对照**：与 MLGWSC-1 baseline (T2 CNN) 及 LVK matched-filter SNR (T0) 直接可比
- **轨 B — 鲁棒性纵向对比**：在 4 类压力测试场景下评估各 baseline 与本研究方法

> **注（2026-06-02 决策）**：原计划的 T4-text baseline（Li 2026, arXiv:2512.04031）已**全面弃用**。原因：经审查发现其方法论存在严重缺陷——正样本通过纯复制过采样（同一样本复制 37 份），且在过采样后才做随机 train/test 切分，导致测试集与训练集存在像素级相同样本（数据泄露），其报告的 97.4% accuracy 不反映真实泛化。本研究不再引用、不作 baseline、不作设计依据。

---

## 2. 数据集设计

### 2.1 数据来源

| 数据来源 | 用途 | 工程获取 |
|---|---|---|
| GWOSC 90 个 confident GW 事件（events.csv 中 3 个缺 chirp_mass 字段被跳过）| 主训练 / 主测试 | `pycbc.catalog.Merger().strain()` / `gwpy.fetch_open_data` |
| O3 真实噪声段 | OFF-source 负样本 + 注入背景 | GWOSC bulk download |
| MLGWSC-1 DS4（real O3a 噪声 + IMRPhenom 注入）| 额外测试集 + T2 CNN baseline；自行用 `generate_data.py -d 4` 生成 | github.com/gwastro/ml-mock-data-challenge-1 |
| GravitySpy 标注 | Glitch overlap 测试集 | Zenodo 10.5281/zenodo.5118835 |
| GWOSC O4a/O4b 已确认事件 | OOD 测试集 | GWOSC eventapi |

### 2.2 训练集构造（4 路实验共享）

**多探测器样本组合策略**：每个 (事件, 探测器) 组合作为独立训练样本（方案 A）。同一事件的 H1 和 L1 spectrogram 是两个**独立训练样本**，**不做多通道堆叠或多图输入合成**。

理由：
1. **与社区标准对齐**：Gabbard 2018、MLGWSC-1、Dingo 等主流 GW-ML 工作均采用单探测器 / 单时序流独立输入。
2. **保持单探测器部署灵活性**：真实场景下经常出现单探测器事件（如 GW190425 仅 L1 检测到），模型必须支持任何探测器子集输入。
3. **数据量保留**：方案 A 样本数为多通道方案的 2 倍，对小数据训练（90 事件）极其重要。
4. **泄露已通过事件 ID 切分防范**：同事件多个 (探测器, jitter) 样本绑定在同一 train/val/test 切分。

**多探测器一致性可在 inference 时后处理**：同事件 H1 和 L1 都预测 YES 时提升联合置信度，但这是 deployment-time aggregation，不是 model-time joint training。

---

**正样本（chirp）**：93 真实事件，按事件 ID 切分 80/10/10（同一事件绑定，防泄露）：
- 训练 80% ≈ 72 事件
- 验证 10% ≈ 9 事件
- 测试 10% ≈ 9 事件

每个训练事件 **× 2 探测器 (H1/L1)** × jitter 增广 (**9 个 ±1s POS 偏移**) ≈ **~1500 张正样本图**

**负样本组成（三类）**：

| 类别 | 数量 | 来源 | 标签 |
|---|---|---|---|
| OFF-source 纯噪声 | ~1500 | 训练事件 strain 中合并前 100s（BNS 600s） × 2 探测器 × **9 个 ±20s NEG 偏移** | `detection: NO` |
| Glitch hard negative | **~500** | GravitySpy (Zenodo 10.5281/zenodo.5118835) 22 类标注分层采样 GPS 时刻 | `detection: NO`（不做 glitch_type 多分类） |
| 独立 O3 干净段（可选） | 0–500 | GWOSC bulk download + DQ flag 筛 SCIENCE+CAT1，避开所有 trigger ±60s | `detection: NO` |

**类别比例**：正 1500 / OFF-source 1500 / glitch 500 ≈ **1 : 1 : 0.33**（正:负 = 1:1.33）

**关于训练比例 vs 真实 base rate**：真实部署时 GW 信号 vs 噪声 base rate ≈ 1:600,000，但训练时按真实比例会让模型学会"永远说 NO"。业界标准（MLGWSC-1 / Gabbard 2018）训练用 1:1 平衡。**评估时用 FAR (False Alarm Rate) 而非 accuracy**——FAR 不依赖测试集类比例。**部署时调输出阈值**（如 confidence > 0.99 才报 YES）使实际 FAR 满足 1/month 标准。

**E3/E4 额外注入**：~1k 注入样本，按 events.csv 物理分布采样源参数 + O3 真实噪声背景 + SNR ∈ [4, 20]（低 SNR 过采样）

### 2.3 数据切分原则

| 原则 | 实施 |
|---|---|
| **按事件 ID 切，不按窗口切** | 同事件多窗口绝不跨集合，避免泄露 |
| **固定随机种子** | `seed=42` 用于事件 ID 切分，论文可复现 |
| **测试集独立** | MLGWSC、GravitySpy、O4 各自构成独立测试集，与训练分离 |

### 2.4 输入图像规格（沿用 Stage 0 PoC，与 GWOSC quickview 流程对齐）

| 参数 | 值 | 备注 |
|---|---|---|
| 窗口长度 | **4 秒**（中心对齐合并时刻） | BBH chirp 占图 5–50% 宽度，肉眼可见；与 GWOSC quickview 显示窗口（2s）量级一致 |
| 采样率 | 4096 Hz | GWOSC 标准 |
| Bandpass | 通过 q_transform `frange` 隐式约束 | 不再显式调 `bandpass()`，对齐 GWOSC 官方 |
| Whitening | q_transform 内置自动（默认 `whiten=True`） | 不再手动 `whiten()`，对齐 GWOSC 官方 |
| PSD 估计基底 | 整段 strain（含合并前 ~256s + 显示窗口） | 由 q_transform 内部完成 |
| Q-transform | Q ∈ [4, 64], frange ∈ [20, 512] Hz, `outseg=(c-2, c+2)` | 唯一显式调用 |
| 输出图像 | 1024×1024 灰度 PNG，固定 vmin=0 / vmax=25.5 | 避免 per-image normalization 视觉欺骗；对齐 LIGO 论文 colorbar |
| **通道（探测器选择）** | **每探测器单独一张（H1/L1 两张，V1 排除）** | 训练仅用 LIGO 双探测器 |
| **Jitter 增广** | POS 9 个 ±1s（间隔 0.25s）+ NEG 9 个 ±20s（间隔 5s）| 每事件每探测器 18 张图，扩样本量约 1.8 倍 |

**V1 排除理由**：灵敏度比 LIGO 低 2.5-3 倍（O3 BNS 距离 45-51 Mpc vs LIGO 108-135 Mpc），多数事件 V1 SNR < 5（chirp 视觉不可见），且本研究输出不含 sky position / polarization。V1 数据文件保留磁盘备用。与 MLGWSC-1 / Gabbard 2018 / Dingo 等社区标准对齐。

**与 GWOSC 官方 quickview 一致性**：训练集图像生成流程与 `gwosc-tutorial/quickview` 仓库一致，确保审稿人可直接用 GWOSC 网站事件页图像与我们处理结果肉眼比对。

---

## 3. 模型与训练阶梯

### 3.1 三阶段硬件 / 模型映射

| Stage | 硬件 | 模型 | 训练精度 | 用途 |
|---|---|---|---|---|
| Stage 0 ✅ | Mac | 无 | 无 | 数据 PoC（已完成） |
| **调试** | DGX Spark (`spark-91b6.local`) | Qwen3-VL-8B / **Gemma 4 E4B** | bf16/QLoRA | 跑通链路、4 路快速筛选 |
| **Stage 2** | DGX Spark (128GB) | Qwen3.6-27B Dense / **Gemma 4 31B Dense** | bf16 LoRA | 主表数据（最佳 1-2 路精度版）|
| **Stage 3 (可选)** | DGX Spark (128GB) | Qwen3.6-35B-A3B (MoE) | bf16 LoRA | T4 内部 Dense vs MoE 对照 |

> **第二模型族 Gemma 4（2026-06-15 纳入）**：Google 开源原生多模态（Apache 2.0）。主对比 **Gemma 4 31B Dense**（对位 Qwen3.6-27B），调试用 **E4B**。论文叙事扩为三方对比：Qwen3-VL（vision-encoder/DeepStack）vs Qwen3.6（原生多模态）vs Gemma 4（另一开源原生多模态族，12B 变体为"编码器无关"统一架构）。微调细节见 `docs/03_gemma4_investigation.md`。框架统一用 **Unsloth**。

### 3.2 关键技术选择

- **bf16 LoRA**（不是 QLoRA 4-bit）：Qwen 官方推荐；保持参数学习精度
- **LoRA rank**：8（VLM 微调常用值，小数据集足够）
- **LoRA alpha**：16
- **Dropout**：0.1
- **Optimizer**：AdamW
- **微调框架**：Unsloth（DGX Spark 已支持）+ TRL backend

### 3.3 训练超参（待 Stage 1 调优后定稿）

| 参数 | 默认值 | 备注 |
|---|---|---|
| Batch size | 4 (8B) / 1 (27B) | 视显存调整 |
| Gradient accumulation | 8 / 16 | 等效 batch=32 |
| Learning rate | 2e-4 (8B) / 5e-5 (27B) | 经验值 |
| Epochs | 3-5 | 早停于验证集 loss |
| Max sequence length | 2048 | 一张 1024×1024 图 ≈ 1024 visual tokens + JSON ~50 tokens |

---

## 4. 输入 / 输出格式

### 4.1 Prompt 模板（4 路实验共用，固定不变）

**System prompt（固定）**：
```
You are an expert gravitational-wave data analyst. Given a Q-transform time-frequency spectrogram from LIGO/Virgo strain data, determine whether it contains a gravitational wave signal. If yes, estimate the source parameters in discrete bins.

Output strictly as a JSON object with fields:
- "detection": "YES" or "NO"
- "chirp_mass_bin": predefined bin label (only if YES, else "N/A")
- "distance_bin": predefined bin label (only if YES, else "N/A")
- "chi_eff_bin": predefined bin label (only if YES, else "N/A")

Output only the JSON, no other text.
```

**User message**（仅图像，无文字）：
```
[image: 1024×1024 spectrogram PNG]
```

**Assistant target（YES 例）**：
```json
{"detection": "YES", "chirp_mass_bin": "25-30", "distance_bin": "400-500", "chi_eff_bin": "0.0-0.2"}
```

**Assistant target（NO 例）**：
```json
{"detection": "NO", "chirp_mass_bin": "N/A", "distance_bin": "N/A", "chi_eff_bin": "N/A"}
```

**E1/E3（纯检测）schema 简化**：
```json
{"detection": "YES"}
```

> **注**：方案不包含 sky position (RA/Dec) 输出。理由：每探测器单独成图且都按合并时刻对齐到窗口中心，多探测器时延信息被人为消除，模型从单图无法重建 sky direction。GW-Whisper / Dingo-T1 等 T4 sentinel papers 也未做 sky output。

### 4.2 参数 bin 定义

| 参数 | 范围 | bin 切分 | bin 数 |
|---|---|---|---|
| chirp_mass (M☉) | 1–80 | [1-2.5][2.5-5][5-15][15-25][25-40][40-60][60+] | 7 |
| distance (Mpc) | 40–9000 | [<200][200-400][400-800][800-1600][1600-3200][3200+] | 6 |
| chi_eff | -1.0 ~ +1.0 | [<-0.2][-0.2~0.0][0.0~0.2][0.2~0.4][0.4+] | 5 |

bin 边界设计原则：
- chirp_mass：对数 + 物理意义（BNS / NSBH / stellar BBH / IMBH 边界）
- distance：对数（近距事件密度高，需更细 bin）
- chi_eff：均匀（分布对称，中心区域居多）

---

## 5. 四路消融实验矩阵

### 5.1 实验定义

| 实验编号 | 训练数据 | 输出 schema | 目的 |
|---|---|---|---|
| **E1（主实验）** | 90 真实 + 真实 OFF-source | 纯检测 (A) | 主路径 / 纯真实数据训练能力上限 |
| **E2** | 90 真实 + 真实 OFF-source | 检测 + 参数 bin (B) | 多任务能力验证 |
| **E3** | 90 真实 + ~1k 注入 + OFF-source | 纯检测 (A) | 注入对 detection 的边际增益 |
| **E4** | 90 真实 + ~1k 注入 + OFF-source | 检测 + 参数 bin (B) | 注入对参数 bin 的边际增益 |

### 5.2 比较对组

| 对比 | 回答的问题 |
|---|---|
| E1 vs E2 | 多任务是否拖累检测？ |
| E1 vs E3 | 注入对 detection 是否帮助？（纯真实数据 vs 真实+注入的边际增益）|
| E2 vs E4 | 注入对参数估计是否帮助？ |
| E1 ~ E4 vs T0/T2 | readiness × modernity 矩阵填格 |

### 5.3 Stage 1 → Stage 2 升级原则
- Stage 1 跑全 4 路（在 5090 上 8B 快速筛选，~2 天）
- 根据 Stage 1 结果**选最佳 1-2 路升到 Stage 2 27B**（DGX Spark）
- 选择标准：Clean baseline + Low SNR 综合表现最好的 2 路

---

## 6. 评估方法论与 baseline 对照

### 6.1 Baseline 层级

| 层级 | 名称 | 实现方式 | 工程量 |
|---|---|---|---|
| **T0** | LVK matched-filter SNR | 直接读 events.csv 的 `network_matched_filter_snr` 列 | 0 |
| **T2** | MLGWSC-1 baseline CNN | 仓库**不附预训练权重**；用 `examples/example_torch.py` 参考 CNN 或外部 AResGW (passalis/gw-detection-deep-learning) 在 MLGWSC-1 DS4 上跑出 baseline | 1 周 |
| **T4-vision (本研究)** | Qwen3-VL/3.6 + spectrogram | 4 路 E1-E4 主路径 | 主线 |

> T4-text baseline 已弃用（见 §1 注）。

### 6.2 四类鲁棒性测试场景

| 场景 | 测试集 | 目的 | 工程量 |
|---|---|---|---|
| **1. Clean baseline** | 90 真实事件测试切分 + 同等 OFF-source | 平地基础表现 | 0（含训练流程） |
| **2. MLGWSC-1 DS4 注入** | 自行用 `generate_data.py -d 4` 生成（real O3a 噪声 + IMRPhenom 注入，需 94 GB 噪声底）| 检测灵敏度 / sensitive distance | ~1 周（含数据生成 + 适配）|
| **3. Glitch overlap** | GravitySpy 标注 + 注入混合 | 抗干扰能力 | 1.5 周 |
| **4. OOD events** | GWOSC O4a/O4b 已确认事件 | 时序泛化 | 3 天 |

### 6.3 完整对照表（论文核心数据）

每个 baseline × 每个场景，加上本研究 4 路实验：

```
                        Clean    MLGWSC-DS4   Glitch   OOD
T0 (LVK SNR)            …        …            …        …
T2 (MLGWSC CNN)         …        …            …        …
E1 (本: 真实+det)       …        …            …        …
E2 (本: 真实+det+param) …        …            …        …
E3 (本: 注入+det)       …        …            …        …
E4 (本: 注入+det+param) …        …            …        …
```

---

## 7. 评估指标

### 7.1 检测指标（所有实验必报）

| 指标 | 定义 | 与谁可比 |
|---|---|---|
| Accuracy | (TP+TN)/total | 标准 |
| Recall (per class) | TP/(TP+FN) per class | 标准 |
| Precision | TP/(TP+FP) | 标准 |
| F1 | 2*P*R/(P+R) | 标准 |
| ROC AUC | ROC 曲线下面积 | 标准（阈值无关） |
| PR AUC | PR 曲线下面积 | 适合不平衡场景 |
| Confusion matrix | 4-cell 表 | 标准 |

### 7.2 物理评测指标（MLGWSC-1 DS4 场景必报）

| 指标 | 定义 |
|---|---|
| Sensitive distance @ FAR=1/month | MLGWSC-1 标准（`evaluate.py` 计算）|
| Detection rate vs injected distance curve | MLGWSC-1 按 chirp_distance 采样，无固定 SNR；按距离统计召回 |

### 7.3 参数估计指标（E2/E4 必报）

| 指标 | 定义 |
|---|---|
| Top-1 bin accuracy (per parameter) | 预测 bin == 真实 bin 比例 |
| Adjacent bin accuracy | 预测 bin 在真实 bin ±1 内比例（容错） |
| Confusion matrix per parameter | chirp 7×7、distance 6×6、chi_eff 5×5 |

### 7.4 Calibration（论文亮点）

| 指标 | 定义 |
|---|---|
| Expected Calibration Error (ECE) | 预测置信度与真实准确率的偏差 |
| Reliability diagram | 置信度 bin × accuracy 折线图 |

### 7.5 推理速度（readiness 评估）

| 指标 | 定义 |
|---|---|
| Latency per sample (ms) | 单样本前向时间 |
| Throughput (samples/sec) | batch=32 时的吞吐 |

---

## 8. 工程实现与时间表

### 8.1 时间表（按周）

| 周 | 任务 | 输出 |
|---|---|---|
| W1 ✅ | Stage 1 数据 pipeline（90 事件全量 H1+L1 + OFF-source，jitter 9/9）| 训练数据完整（2394 train / 306 val / 270 test，排除 V1 后口径）|
| W2 | 5090 环境 + Qwen3-VL-8B 微调 pipeline | E1 跑通 |
| W3 | 跑完 E1-E4 4 路 × 内置 clean test | Stage 1 主表 |
| W4 | MLGWSC-1 DS4 生成 + 适配 + baseline CNN | 场景 2 数据 + T2 baseline |
| W5 | GravitySpy 数据 + Glitch overlap 测试集 | 场景 3 数据 |
| W6 | O4 OOD 测试集 | 场景 4 数据 |
| W7 | DGX Spark 环境部署 + 27B 主实验启动 | Stage 2 启动 |
| W8 | Stage 2 完成 + 数据汇总 + 论文初稿大纲 | 论文骨架 |
| W9-10 | 论文撰写 + 图表生成 | 论文 v1 |

**总计：~10 周（含撰写）**

### 8.2 项目目录扩展

```
GW-VLM/
├── README.md
├── data_pipeline/                    （已建）
│   ├── scripts/
│   │   ├── 01_download_strain.py     ✅
│   │   ├── 02_generate_spectrograms.py  ✅
│   │   ├── 03_make_montage.py        ✅
│   │   ├── 04_inject_signals.py      ⏳ Stage 1
│   │   ├── 05_build_dataset.py       ⏳ Stage 1（生成 JSONL）
│   │   └── 06_split_by_event.py      ⏳ Stage 1
│   ├── config.py
│   └── requirements.txt
├── training/                         ⏳ Stage 1
│   ├── train_qwen3vl_lora.py
│   ├── configs/
│   │   ├── e1_real_detection.yaml
│   │   ├── e2_real_multitask.yaml
│   │   ├── e3_injection_detection.yaml
│   │   └── e4_injection_multitask.yaml
│   └── scripts/
├── evaluation/                       ⏳ Stage 1
│   ├── metrics.py
│   ├── eval_clean.py
│   ├── eval_lowsnr.py
│   ├── eval_glitch.py
│   ├── eval_ood.py
│   └── baselines/
│       ├── mlgwsc_cnn.py
│       ├── li2026_llm.py
│       └── lvk_mf_snr.py
├── docs/
│   ├── 00_poc_findings.md
│   └── 02_research_design.md         ← 当前文档
└── output/
```

### 8.3 实验日志与复现性

- 每次训练用 W&B 记录 loss / metric 曲线
- 模型 checkpoint 命名：`{model}_{exp_id}_{date}.safetensors`
- 数据集版本号写入 dataset card（events.csv hash + 注入 seed）
- 论文复现包：requirements 锁定版本 + bash 脚本一键跑全 4 路

---

## 9. 待商榷细节（v0.2 解决）

下列项在 Stage 1 启动前需进一步定稿：

| 项 | 问题 | 计划 |
|---|---|---|
| 数据增强 | 时间抖动 ±0.5s 之外要不要加频率扰动 / SNR 缩放？ | Stage 1 跑通后视情况补 |
| 训练-验证早停 | 用什么指标早停？ | val accuracy（与论文主指标对齐） |
| Glitch 注入精确度 | 用 BayesWave 重建 vs 直接叠加？ | 直接叠加，BayesWave 留 future work |
| MLGWSC baseline 数据兼容 | MLGWSC dataset 与我们的 image preprocessing 是否一致？ | Stage 1 W4 解决 |

---

## 10. 决策追溯（已收敛事项）

| 决策 | 选择 | 日期 |
|---|---|---|
| 评估目标双轨 | 性能基线 + 鲁棒性纵向（RQ8/9） | 2026-05-06 |
| Baseline | T0 (LVK SNR) + T2 (MLGWSC CNN) + T4-vision (本) | 2026-05-06 |
| 4 类鲁棒性场景 | 全做（Clean/MLGWSC-DS4/Glitch/OOD） | 2026-05-06 |
| MLGWSC-1 用法 | DS4 作额外测试集 + T2 CNN baseline | 2026-05-06 |
| 训练数据 | 主实验仅 90 真实，E3/E4 加 1k 注入消融 | 2026-05-06 |
| 实验路数 | 4 路（E1-E4） | 2026-05-06 |
| 输出 schema | JSON 风格（detection + 3 个 bin） | 2026-05-06 |
| Prompt | 固定 system prompt + 仅图像 user message | 2026-05-06 |
| 参数 bin 粒度 | chirp 7 / distance 6 / chi_eff 5 | 2026-05-06 |
| 注入规模 | ~1k | 2026-05-06 |
| 模型阶梯 | 8B (5090) → 27B (Spark) → MoE 可选 | 2026-05-06 |
| LoRA 配置 | bf16 LoRA, rank=8, alpha=16, dropout=0.1 | 2026-05-06 |
| Sky position 输出 | 不加 RA/Dec 字段；多探测器时延信息在我们输入中已消除 | 2026-05-07 |
| 窗口长度 | 8s → 4s；BBH chirp 占图比例足够，与 GWOSC quickview 对齐 | 2026-05-07 |
| Q-transform 处理 pipeline | 对齐 GWOSC 官方：去除显式 bandpass + 手动 whiten；统一用 q_transform 默认参数 | 2026-05-07 |
| 训练集负样本组成 | 三类：OFF-source + GravitySpy glitch + 正样本 | 2026-05-07 |
| 渲染色阶固定 | matplotlib imshow 加 vmin=0 / vmax=25.5 防 per-image normalization 视觉欺骗（对齐 LIGO colorbar）| 2026-05-19 |
| 训练/验证/测试切分 | 60/20/20 → **80/10/10**；扩大训练集占比 | 2026-05-19 |
| 探测器选择 | 训练仅用 H1+L1；V1 排除（灵敏度 2.5-3x 低 + chirp 视觉不可见 + MLGWSC-1 等标准） | 2026-05-20 |
| 多探测器样本组合 | 方案 A 独立样本（每 IFO 一图一样本）；不做多通道堆叠 / 多图输入 | 2026-05-20 |
| Jitter 扩展 | POS 5→9（±0.5s→±1s）/ NEG 5→9（±10s→±20s）；样本量约 ×1.8 | 2026-05-20 |
| **Li 2026 弃用** | **全面移除**：经审查发现其过采样复制 + 切分泄露导致 97.4% 不可信；不引用/不作 baseline/不作设计依据 | 2026-06-02 |
| MLGWSC-1 适配 | DS4（real O3a 噪声）作额外测试集；2048→4096Hz 升采样 + 滑窗 + 自跑 q_transform；评估用其 evaluate.py 出 sensitive-distance vs FAR | 2026-06-02 |
| 探测器处理认知 | 确认 LIGO 工程 + 主流 spectrogram-ML 均单探测器独立处理；G2Net 三通道堆叠仅 Kaggle 便利设计，非社区标准；我们方案 A 与 LIGO 论文图一致 | 2026-06-02 |
| 数据集口径修正 | 排除 V1 重生：train/val/test = 2884/366/330 → **2394/306/270**（共 2970，正/负各 1485）| 2026-06-15 |
| **纳入 Gemma 4 对比族** | Google 开源原生多模态；主对比 31B Dense（对位 Qwen3.6-27B）+ 调试 E4B；框架 Unsloth；详见 docs/03 | 2026-06-15 |
| 训练机 | 训练统一在 **DGX Spark**（`spark-91b6.local`，128GB）；不再依赖 RTX 5090 | 2026-06-15 |
| **评估口径** | 主指标用 **ROC-AUC + PR-AUC + FAR(FPR≤5%)工作点**；贪心 0.5-阈值 accuracy 会严重低估 recall，仅作参考 | 2026-06-17 |
| **主模型 LoRA 配方** | 实测保守设置(r8/lr5e-5/batch16)欠拟合；改 **r32/α32/lr2e-4/dropout0/有效batch8** 后 31B ROC-AUC 0.858→0.922 | 2026-06-17 |
| **输入分辨率** | Unsloth 默认 512px 偏低；改 Gemma4 token 预算 **560(≈1135px)** 吃满 1024 源（`image_tokens` 配置）| 2026-06-17 |
| 检测瓶颈认定 | 漏检集中在**低 SNR 弱信号**(efficiency vs SNR：<8 56% → 12-20 88%)；调色/位深无效，提升靠注入(E3/E4)或接受 SNR 下限 | 2026-06-17 |

> 📋 E1 完整训练/评估/诊断记录 → [`04_e1_experiments_and_findings.md`](04_e1_experiments_and_findings.md)。
