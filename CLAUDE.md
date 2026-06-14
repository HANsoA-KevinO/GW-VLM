# GW-VLM — 项目上下文（迁移交接文档）

> 本文件供新机器上的 Claude / 协作者快速接管项目。读完即可理解项目目标、当前进度、关键决策与下一步。
> 最后更新：2026-06（数据 pipeline 完成，即将进入训练）。

---

## 1. 项目是什么

用**预训练视觉语言模型（VLM，Qwen3-VL / Qwen3.6）微调**做**引力波（GW）检测 + 粗粒度物理参数估计**。输入是 LIGO strain 数据生成的 **Q-transform 时频图（spectrogram PNG）**，输出是结构化 JSON（检测 YES/NO + chirp_mass/distance/chi_eff 的离散 bin）。

**这是对前代项目 GW-TF 的完全重写**。GW-TF 用 LLaMA-3-8B + K-Means 把 spectrogram 量化成 256 token，丢失 99.8% 信息，且存在严重数据泄露（正样本复制 37 份后才随机切分 train/test）。GW-VLM 直接让 VLM 吃完整时频图，不做量化。

**目标产出**：一篇方法论扎实的论文（对标 MLGWSC-1 等社区标准），核心卖点是"VLM 直接处理时频图做 GW 检测 + 参数估计"这一空白。

---

## 2. 当前进度（截至迁移）

| 阶段 | 状态 |
|---|---|
| Stage 0 PoC（数据流概念验证） | ✅ 完成 |
| Stage 1 数据 pipeline（90 事件全量） | ✅ 完成 |
| **训练（首次 E1）** | ⏳ **下一步，尚未开始** |
| Stage 2 主实验（DGX Spark 27B） | 未开始 |
| 评估 / baseline / MLGWSC-1 对接 | 未开始 |

**数据集已就绪**：`output/dataset_{train,val,test}.jsonl` = train 2884 / val 366 / test 330，正负样本平衡，按事件 ID 切分（无泄露）。

---

## 3. 关键决策（必读，全部已与项目负责人对齐）

这些决策有充分论证，**不要随意推翻**。完整追溯见 `docs/02_research_design.md` §10。

| 决策 | 内容 | 理由 |
|---|---|---|
| **探测器** | 只用 **H1 + L1**，排除 V1 | V1 灵敏度低 2.5-3 倍，多数事件 SNR<5，chirp 视觉不可见；与 MLGWSC-1/Gabbard/Dingo 社区标准一致。V1 strain 文件保留磁盘备用 |
| **样本组合** | 方案 A：每 (事件, 探测器) 独立成样本，**不做多通道堆叠/多图输入** | 保持单探测器部署灵活性；与社区标准一致；数据量翻倍 |
| **窗口** | **4 秒**，中心对齐合并时刻 | BBH chirp 占图 5-50% 可见；对齐 GWOSC quickview |
| **渲染色阶** | 固定 **vmin=0, vmax=25.5** | 防 per-image normalization 视觉欺骗（否则强信号把背景压黑、模型学"亮度"捷径而非 chirp 形态）；对齐 LIGO colorbar |
| **处理流程** | 对齐 GWOSC 官方：直接 `q_transform()`，**不显式 bandpass/手动 whiten**（由 q_transform 内置） | 与审稿人/物理学家看到的 GWOSC 图一致 |
| **数据增广** | POS 9 jitter（±1s 间隔 0.25s）+ NEG 9 jitter（±20s 间隔 5s） | 让模型学 chirp 形态而非窗口位置；扩样本 ~1.8× |
| **切分** | **80/10/10**，按事件 ID（同事件所有样本绑同一集合） | 防泄露；小数据集最大化训练量 |
| **输出 schema** | JSON：`{detection, chirp_mass_bin, distance_bin, chi_eff_bin}` | 见 §4 |
| **bin 粒度** | chirp 7 / distance 6 / chi_eff 5 档 | 平衡信息量与样本均衡（见 config.py）|
| **不输出 RA/Dec** | sky position 不做 | 单探测器对齐窗口后时延信息已消除，无法重建 |
| **训练数据** | 主实验仅 90 真实事件；E3/E4 才加 ~1k 注入做消融 | — |
| **Li 2026 弃用** | 不引用/不作 baseline/不作设计依据 | 其过采样复制+切分泄露导致 97.4% accuracy 不可信 |
| **MLGWSC-1** | 用 DS4（real O3a 噪声）作额外测试集 + T2 CNN baseline | 需自行 `generate_data.py -d 4`（94GB 噪声底）|

---

## 4. 实验设计（4 路消融 E1–E4）

| 实验 | 训练数据 | 输出 | 目的 |
|---|---|---|---|
| **E1（主）** | 90 真实 + OFF-source | 纯检测 `{"detection":"YES/NO"}` | 主路径，先跑通 |
| E2 | 同上 | 检测 + 3 参数 bin | 多任务能力 |
| E3 | 真实 + ~1k 注入 | 纯检测 | 注入对检测的增益 |
| E4 | 真实 + ~1k 注入 | 检测 + bin | 注入对参数估计的增益 |

**输出 JSON schema**（E2/E4）：
```json
{"detection":"YES","chirp_mass_bin":"25-40","distance_bin":"400-800","chi_eff_bin":"-0.2-0.0"}
```
NO 时 bin 填 `"N/A"`。E1/E3 简化为 `{"detection":"YES"}`。

**固定 system prompt + 仅图像 user message**（具体见 `docs/02_research_design.md` §4.1）。

---

## 5. 硬件 / 训练阶梯

| Stage | 硬件 | 模型 | 精度 | 用途 |
|---|---|---|---|---|
| Stage 1 | （本机 SSH 直连）**DGX Spark** 或 RTX 5090 | Qwen3-VL-8B | bf16 LoRA（8B 装得下 32GB）| 4 路快速筛选 |
| Stage 2 | **DGX Spark**（128GB 统一内存）| Qwen3.6-27B Dense | bf16 LoRA | 主实验，论文数据 |
| Stage 3（可选）| DGX Spark | Qwen3.6-35B-A3B MoE | bf16 LoRA | Dense vs MoE 对照 |

- LoRA 配置：rank=8, alpha=16, dropout=0.1, AdamW
- **DGX Spark 用 bf16 LoRA 而非 QLoRA**（Qwen 官方推荐，保参数学习精度）
- 框架：Unsloth 或 LLaMA-Factory（**待定，见 §7 决策 1**）

---

## 6. 目录结构与数据再生

```
GW-VLM/
├── CLAUDE.md                    ← 本文件
├── README.md                    ← 简版项目说明
├── .gitignore
├── data_pipeline/
│   ├── config.py                ← 所有参数 + 事件加载 + bin 定义（自包含）
│   ├── events.csv               ← 90 GW 事件元数据（已随项目，不再依赖 GW-TF）
│   ├── requirements.txt         ← 冻结版本（Python 3.12）
│   └── scripts/
│       ├── 01_download_strain.py        # 下载 strain（--full=90事件 / --poc=5事件，8 路并行）
│       ├── 02_generate_spectrograms.py  # strain→Q-transform→1024×1024 PNG（vmax=25.5, 9 jitter）
│       ├── 06_build_dataset.py          # 扫 PNG → dataset.jsonl（中间格式）
│       ├── 07_split_by_event.py         # 按事件 ID 切 80/10/10
│       ├── audit_all_events.py          # 数据完整性审计（缺失/NaN/窗口）
│       └── check_*.py                   # 可视化核查（montage / 分布 / 三联图）
├── docs/
│   ├── 01_model_and_toolchain_investigation.md  # 模型选型调研
│   └── 02_research_design.md            ← 完整研究方案（权威，含 §10 决策追溯）
└── output/
    ├── raw_strain/      # 2.8GB strain 缓存（.gitignore，可再生）
    ├── spectrograms/    # 640MB PNG（.gitignore，可再生）
    └── dataset_*.jsonl  # 训练数据集（中间格式）
```

### 数据再生（新机器推荐方式 — 完全确定性）
```bash
cd GW-VLM
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r data_pipeline/requirements.txt
python data_pipeline/scripts/01_download_strain.py --full   # ~30-60min（下载 90 事件 H1/L1）
python data_pipeline/scripts/02_generate_spectrograms.py --full  # ~30-45min（生成 ~1925 PNG）
python data_pipeline/scripts/06_build_dataset.py            # 秒级
python data_pipeline/scripts/07_split_by_event.py           # 秒级（seed=42 固定）
```
**再生是确定性的**（seed=42 + events.csv 固定），新机器跑出的 dataset 与原机一致，且 `image_path` 自动写成新机器的绝对路径——**这就避免了路径迁移问题**。

### 中间格式 dataset.jsonl 单条
```json
{"image_path":"<绝对路径>.png",
 "label":{"detection":"NO","chirp_mass_bin":"N/A","distance_bin":"N/A","chi_eff_bin":"N/A"},
 "source_type":"real_neg_off","split_key":"GW150914","event_name":"GW150914",
 "ifo":"H1","jitter_idx":0,
 "metadata":{"kind":"BBH","chirp_mass":28.6,"luminosity_distance":440.0,"chi_eff":-0.01,"snr":26.0}}
```
⚠️ `image_path` 是**生成时的绝对路径**。新机器**重新生成**即自动正确；若直接拷 jsonl，需重映射路径。

---

## 7. 离训练还差什么（下一步工作清单）

### 轨 A：纯代码（任何机器可做）
1. **数据格式转换器**（`scripts/08_export_training_format.py`，**未写**）：
   - 把中间 jsonl → 训练器格式（system prompt + `<image>` + assistant JSON 目标）
   - `--schema detection_only|multitask` 切换 E1/E2 目标
   - 输出 `training_data/{e1,e2}/{train,val,test}.jsonl`
2. **训练脚本 + E1 配置**（`training/`，**未建目录**）
3. **评估脚本**（`evaluation/`，**未建**）：accuracy/F1/ROC-AUC/PR-AUC/confusion matrix + calibration

### 轨 B：需要 DGX Spark
4. 装框架（Unsloth / LLaMA-Factory）+ 下载 Qwen3-VL-8B
5. 跑 E1 训练 → 验证整条链路 work → 再上 E2-E4 → Stage 2 升 27B

### 训练前待敲定的 3 个决策
1. **框架**：Unsloth vs LLaMA-Factory（后者对 Qwen-VL 支持更成熟、配置更简单；前者省显存/快）
2. **Qwen3-VL-8B 确切 HF model ID** + 在 Spark 上确认可下载
3. **先跑 E1（纯检测）单独验证链路**，再扩 E2-E4

---

## 8. 关键坑 / 历史教训（避免重蹈）

- **vmax 必须固定**：早期没固定 vmin/vmax，matplotlib per-image 自动曝光让含强信号的 pos 图变纯黑、neg 图却"杂乱明亮"，模型会学"暗=有信号"的假捷径。已固定 vmax=25.5。
- **jitter 数组顺序**：扩 jitter 时新偏移要**追加在数组末尾**（j5-j8），不能插在前面，否则与已生成的 j0-j4 文件名错位。当前 `JITTER_OFFSETS`/`NEG_TIME_OFFSETS` 已是"旧值在前、新值在后"的兼容顺序。
- **数据缺失是物理事实**：90 事件中部分探测器无数据（V1 早期未启用、探测器 lockloss→NaN strain、单探测器事件如 GW190425 仅 L1）。`audit_all_events.py` 已统计。不是 bug。
- **GWOSC 下载偶发失败**：网络瞬态导致部分事件首次下载失败，**重跑 01 可救回**（已验证：60 个 MISSING → 重试后剩 20 个真缺失）。
- **方法族编码有 32% "Unclear"**：`docs/02_research_design.md` 相关；这是 scoping review 论文（GW-TF/SUBMIT）的问题，与 GW-VLM 训练无关，勿混淆。

---

## 9. 相关项目（勿混淆）

- **GW-TF**（`/Users/hansoakevino/code/GW-TF`，原 Mac）：前代项目，已弃用其方法。events.csv 来源（现已拷入本项目）。
- **SUBMIT**（`/Users/hansoakevino/code/SUBMIT`，原 Mac）：**另一篇独立的 scoping review 论文**（Universe MDPI 投稿），与 GW-VLM 训练无关，不要迁移混入。
- 本项目 = **GW-VLM**，只做 VLM 微调训练。

---

## 10. 迁移传输清单

**必传**（小）：
- 全部代码（`data_pipeline/`, `docs/`, `CLAUDE.md`, `README.md`, `.gitignore`）
- `data_pipeline/events.csv`（项目已自包含）

**不必传**（大且可再生，新机器重跑 01-07 即可）：
- `output/raw_strain/`（2.8GB）
- `output/spectrograms/`（640MB）
- `.venv/`（新机器重建）

**可选传**（省再生时间，但需重映射 image_path）：
- `output/dataset_*.jsonl` + `output/spectrograms/`

**推荐**：只传代码 + events.csv，到新机器（或直接在 DGX Spark）`pip install` 后重跑 01-07 再生数据——最干净，无路径问题。
