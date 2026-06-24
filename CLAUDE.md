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
| 训练基建（08 导出 + training/ + evaluation/ + 环境踩坑） | ✅ 完成 |
| **E1（纯检测）训练 + 评估** | ✅ **多模型跑完**：E4B / 31B 基线 / 31B v2 / 31B viridis 3ep / **Qwen3.6-27B viridis（最佳 ROC-AUC 0.940）** |
| 评估方法论（ROC-AUC / PR-AUC / FAR 工作点） | ✅ 确立（贪心 accuracy 会误导，改用 ROC+FAR） |
| 瓶颈诊断 | ✅ 漏检集中在**低 SNR 弱信号**（efficiency vs SNR） |
| 弱信号 recall 提升（注入 E3/E4 / viridis 彩色 / 转 E2） | ⏳ 探索中（viridis A/B 进行中） |
| Qwen3.6-27B 主实验 / MLGWSC-1 baseline / 鲁棒性场景 | 未开始 |

> **这两天的完整实验记录、结果表、关键发现、环境踩坑 → 见 [`docs/04_e1_experiments_and_findings.md`](docs/04_e1_experiments_and_findings.md)（必读）。**

**数据集已就绪**：`output/dataset_{train,val,test}.jsonl` = **train 2394 / val 306 / test 270**（共 2970，正/负各 1485），正负样本平衡，按事件 ID 切分（无泄露）。

> ⚠️ 数字更新（2026-06-15）：早期文档写的 2884/366/330 是**排除 V1 之前**的旧口径；`06_build_dataset` 现尊重 `config.DETECTORS`（H1+L1，排除 V1），重生后为 2394/306/270。

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
| 调试 | **DGX Spark**（SSH `spark-91b6.local`）| Qwen3-VL-8B / **Gemma 4 E4B** | bf16/QLoRA | 跑通链路、快速筛选 |
| Stage 2 主 | **DGX Spark**（128GB 统一内存）| Qwen3.6-27B Dense / **Gemma 4 31B Dense** | bf16 LoRA | 主实验，论文数据 |
| Stage 3（可选）| DGX Spark | Qwen3.6-35B-A3B MoE | bf16 LoRA | Dense vs MoE 对照 |

> **模型族二：Gemma 4（2026-06-15 纳入）**——Google 开源原生多模态。主对比 **Gemma 4 31B Dense**（`google/gemma-4-31B-it`，对位 Qwen3.6-27B），调试 **Gemma 4 E4B**（`google/gemma-4-E4B-it`）。Gemma 4 12B 是"编码器无关"统一多模态（原始 patch 直投 LLM），呼应本项目原生多模态主线，可作后续候选。详见 [`docs/03_gemma4_investigation.md`](docs/03_gemma4_investigation.md)。训练机现为 **DGX Spark**（非 5090）。

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

### 轨 A：纯代码（任何机器可做）✅ 已完成
1. ✅ **数据格式转换器** `scripts/08_export_training_format.py`：中间 jsonl → 统一 messages 格式（system prompt + image + assistant JSON）；`--schema detection_only|multitask` 切 E1/E2；`--image-path relative` 出相对路径便于跨机器；产物 `output/training_data/{e1,e2}/{train,val,test}.jsonl`。
2. ✅ **训练脚本 + E1 配置** `training/`：`train_vlm.py`（Unsloth `FastVisionModel` LoRA）+ `configs/e1_*.yaml`（gemma4_e4b / gemma4_31b / qwen3vl_8b / qwen36_27b）+ `requirements_train.txt`。
3. ✅ **评估脚本** `evaluation/`：`metrics.py`（accuracy/F1/ROC-AUC/PR-AUC/混淆矩阵）+ `evaluate.py`。鲁棒性场景（MLGWSC/Glitch/OOD）留后续。

### 轨 B：DGX Spark ✅ E1 已完成
4. ✅ 主机 venv 环境(torch cu130/GB10、torchvision、Unsloth)+ rsync 数据;所有踩坑见 [`docs/04`](docs/04_e1_experiments_and_findings.md) §5 / `training/spark_env.sh`。
5. ✅ E1 跑完 3 个模型(E4B / 31B 基线 / 31B 调参版),评估 + ROC-AUC + 损失曲线 + 混淆矩阵齐全。

### E1 关键结论(详见 docs/04)
- **最佳模型 = 🥇 Qwen3.6-27B(原生多模态,viridis,2ep)**:**ROC-AUC 0.940**,默认 0.5 阈值 recall 0.837(校准好,不用调阈值)。次优 Gemma4 31B v2(0.922)。
- **评估口径**:用 **ROC-AUC + FAR 工作点**,别用贪心 accuracy(0.5 阈值对 Gemma 会严重低估 recall)。
- **瓶颈 = 低 SNR 弱信号**;**调色(viridis≈灰度)/ 分辨率(>560)/ 堆 epoch(3ep≤2ep)都不能突破 ~0.92–0.94 天花板**(全已实验证伪),要靠**注入(E3/E4)**或接受 SNR 下限。
- **获胜配方**(主模型都该用):`bf16 LoRA, r=32, α=32, lr=2e-4, dropout=0, 有效batch=8`;Gemma 配 `image_tokens=560`,Qwen 用原生分辨率。
- **Qwen 评估须 `--no-unsloth`**(原生 transformers+PEFT 绕开 Qwen3.6 视觉推理 rope bug);跨机同步代码要**单文件 rsync + grep 验证落地**(目录 rsync 在内网抖动下会静默漏传)。

### 下一步候选
1. **viridis 彩色 A/B**(进行中):看彩色能否借预训练编码器提 AUC。
2. **注入管线 E3/E4**:提弱信号 recall(需写 `04_inject_signals`/`05`,装 lalsuite/pycbc)。
3. **补 31B 3ep + Qwen3.6-27B**:拿主表最终数;或 **转 E2**(检测+参数,`08 --schema multitask`)。

### 已敲定
- **框架**:**Unsloth**(回退 HF TRL / LLaMA-Factory)。
- **训练机**:DGX Spark(`super-cortant` = `kevin@spark-91b6.local`,经 NVIDIA Sync 别名);**主机 venv**,非 Docker。
- **HF model IDs**(用 unsloth 镜像免门):`unsloth/gemma-4-31B-it`、`unsloth/gemma-4-E4B-it`、`unsloth/Qwen3.6-27B`、`unsloth/Qwen3-VL-8B-Instruct`。

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
