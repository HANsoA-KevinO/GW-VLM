# 多模态物理融合(方法2):图 ⊕ 原始应变

> 状态:**两轮均为负结果(应变在检测上无增益,图是主力)**。第 1 轮(CNN+3B,§7.3):应变拖累。第 2 轮(**patch+自注意力编码器 + Qwen3.6-27B**,§7.4–7.6):用最强模型+更好编码器复判,**结论没翻盘**——B≈C(融合中性)、仅应变 0.52–0.56(比 CNN 的 0.43 进步但仍近随机)。根因是数据墙(90 事件),非模型/编码器;应变的价值在参数估计,前置=注入(E3/E4)。
> 范围:检测(E1),小模型原型先行,带"还要不要图片"的消融。日期:2026-06。

---

## 0. 一句话

给 VLM 在"看频谱图"之外再开一只"听原始应变"的耳朵——一个 1D 编码器直接吃**白化后的原始应变(含相位)**,把输出当作额外 token 和图像 token 一起喂进 LLM。**图负责检测的整体形态,应变负责图丢掉的精度/相位**,一个模型里融合。本轮先回答一个明确问题:**加了应变之后,还需不需要图片?**

---

## 1. 动机:为什么要融合,为什么图不够

E1/E2 的结论:
- 纯图**检测**很强(Qwen3.6-27B ROC-AUC 0.94–0.96);
- 纯图**参数估计基本失败**(distance 略高于随机,chirp mass / chi_eff ≈ 随机,且模型塌缩到"安全众数 bin")。

根因不是 bin 粒度、也不只是事件少,而是更深的一点:

> **Q-transform 图是精确数值数据(应变时序;振幅/频率/相位演化)的一张有损、降采样、上色的渲染。物理参数(尤其相位敏感量)由那套精确数值结构决定。让 VLM 从有损图片里"读回"这些精细量,是用错了输入。**

GW 社区做检测/参数靠的是**原始应变上的数值方法**(匹配滤波、1D 网络),而非图片;文献(arXiv:2312.04855)显示 **1D+2D 融合全面胜过任一单路**。所以正解是:**要精度就把原始应变直接作为第二路输入喂进去**,而不是把它扔了做成图让模型猜回来。

**本轮先做检测**:检测任务最清楚、不依赖额外数据,能直接验证融合机制是否 work,并回答"还要不要图"。参数估计真正发力还需要注入数据(见 §8),另立计划。

---

## 2. 为什么不是 thinking / 不加推理(定基调)

E2 诊断:开 thinking 自由生成时,模型把几乎所有正样本判 NO(生成式 recall≈0),而 teacher-forced(不思考)ROC-AUC 0.96。原因:思考套用"真信号=明亮 chirp 脊"的强信号先验,而我们的真信号大多不是亮线 → 把不思考时能用的**亚感知特征"推理掉了"**。

结论(本方案全程遵循):
- **训练/推理都关闭 thinking、保持一致**(`enable_thinking=False`)。这是还原训练分布,不是设限。
- "微调模型用不了推理"是误解:要推理须在训练数据里带推理轨迹(蒸馏/RL);但**检测是感知任务,推理帮不上、甚至有害**,故不开。

---

## 3. 架构(方法2:数值编码器 → token)

类比:VLM 现在用**视觉编码器**把一张图压成 ~1024 个"视觉 token",LLM 像读字一样读它们。方法2 就是给应变也配一个编码器,产出"应变 token",拼进同一序列。

```
原始应变(4s)──白化/降采样──► StrainEncoder1D ──► N 个"应变 token"(投影到 LLM hidden)
频谱图 ─────────────────────► 视觉编码器 ───────► ~1024 个"图像 token"
                              拼接 inputs_embeds:
        [ 图像token ‖ 应变token ‖ 文本(system+指令)token ]
                              │
                              ▼  Qwen-VL 语言模型(原生 transformers + LoRA)
                          生成 {"detection":"YES/NO"}(无 thinking)
```

**StrainEncoder1D**(新模块,从零训练、非 LoRA):
- 输入:白化应变,4s。下采样到 2048Hz → 8192 点(GW ML 常规,降长省显存)。
- 结构:几层 1D-Conv(stride 降采)+ 池化 → 输出 **N 个 token(8–16)**,各投影到 LLM hidden_dim。参考 GW 1D 检测网(Gabbard / AResGW)的卷积栈。

**融合 wrapper**(新模块,持有 PEFT-VLM + StrainEncoder),forward:
1. 用 VLM 自身路径把 input_ids + pixel_values 变成 `inputs_embeds`(图像占位符已替换为图 embeds);
2. StrainEncoder(strain) → strain_embeds;
3. 把 strain_embeds **插入序列固定位置**(图 token 之后、文本之前),同步扩展 `attention_mask`、`labels`(应变 token 标 -100)及 Qwen 的 position/rope;
4. 调 LLM forward(`inputs_embeds=...`)→ loss / logits。

**可训练参数** = VLM 的 LoRA(获胜配方 r32/α32/lr2e-4/dropout0)+ **整个 StrainEncoder(全量)**。保存/加载**分别存** LoRA adapter 与 `strain_encoder.pt`。

**为什么弃用 Unsloth**:`FastVisionModel` 全局 patch + `UnslothVisionDataCollator` 黑箱,塞不进自定义融合 token。改用原生 `AutoModelForImageTextToText` + PEFT `LoraConfig` + 自定义 collator(评估里 `--no-unsloth` 路径已是这套,有基础)。

---

## 4. 消融设计(回答"还要不要图片")

同一套代码、同一份数据,三个配置比 ROC-AUC(+ PR-AUC、FPR≤5% recall):

| 配置 | 输入 | 目的 |
|---|---|---|
| **(C) 仅图** | 图 + 文本 | baseline(= 现状,但在小模型 + 原生路径上重训,保证公平) |
| **(B) 图 + 应变** | 图 + 应变 + 文本 | 完整融合 |
| **(A) 仅应变** | 应变 + 文本(无图) | 看应变单独多强 |

- **A vs C**:应变 vs 图,谁强;
- **B vs max(A,C)**:融合有没有 1+1>1。

---

## 5. 数据:样本 → 白化应变(无歧义可复现)

- 原始应变:`output/raw_strain/{event}_{ifo}.hdf5`,4096Hz、**未白化**,每段含 256s 白化基底。
- 映射:`event_name + ifo + jitter_idx →(events.csv 的 GPS + config 的 jitter/neg_offset)→ 中心 GPS → crop 4s 窗口`。jitter 表在 `02_generate_spectrograms.py`,分类/偏移在 `data_pipeline/config.py`。
- 样本是**单探测器**,故应变也是该探测器的那 4s,与图天然对齐(多探测器相干为后续)。
- 预处理脚本 `data_pipeline/scripts/09_extract_strain.py`:加载 → crop 4s → `.whiten()` → bandpass(20–512Hz)+ 下采样 2048Hz → 存 `output/strain_arrays/<图同名>.npy`(float32, 8192)。

---

## 6. 涉及文件

**新建**:
- `data_pipeline/scripts/09_extract_strain.py`(应变预处理)
- `training/models/strain_encoder.py`(StrainEncoder1D)
- `training/fusion_model.py`(融合 wrapper)
- `training/collators/fusion_collator.py`(构建 inputs_embeds + mask + labels)
- `training/train_fusion.py`(原生 transformers + PEFT 训练,含 `use_image`/`use_strain` 消融开关)
- `training/configs/fusion_qwen2.5vl_3b.yaml`(原型)、`fusion_qwen36_27b.yaml`(正式)
- `evaluation/evaluate_fusion.py`(基于 evaluate_prob.py 的 `--no-unsloth` 路径 + 应变)

**改**:`CLAUDE.md`(新增原生 transformers 融合加载路线)。

---

## 7. 实测结果

### 7.1 数据预处理 ✅
- `output/strain_arrays/` 共 **2970** 个 .npy(精确匹配数据集),0 失败。
- 白化抽查:GW150914 H1 正样本(中心=合并时刻)RMS 峰值落在窗口 51%(=合并)、达中位 **1.81×**;噪声窗仅 1.20× —— **chirp 被白化保住、肉眼可辨**。

### 7.2 冒烟测试(小模型)✅
- 架构端到端跑通(图像散射 + 应变 token 注入 + 自算 M-RoPE + 反向 + 保存),loss 有限并下降(3.21→0.07)。
- **关键 bug(已修)**:`get_image_features` 返回 `BaseModelOutputWithPooling`,合并后的 LLM 维图像 embed 在 **`.pooler_output`**(非 `.last_hidden_state`,后者是视觉骨干 1280 维)。初版误用导致一直在用垃圾图像特征训练;限制图像分辨率后评估报 shape 错才暴露,修正后训练/评估两端通过。

### 7.3 消融结果(Qwen2.5-VL-3B,1 epoch,viridis,max_pixels=262144)
| 配置 | 输入 | **ROC-AUC** | PR-AUC | R@0.5 | R@FPR≤5% |
|---|---|---|---|---|---|
| **C 仅图** | 图 | **0.878** | 0.892 | 0.704 | 0.615 |
| B 图+应变 | 图+应变 | 0.795 | 0.822 | 0.593 | 0.563 |
| A 仅应变 | 应变 | 0.433（<随机) | 0.475 | 0.0 | 0.0 |

**结论(回答"加了应变还要不要图"):**
- **图是干活的那个,必须要(C=0.88)。**
- **应变分支在本设定下没用、甚至有害**:仅应变 A=0.433 **低于随机**(从零训的 1D 编码器没学会检测);融合 B=0.795 **反低于纯图 C=0.878**(无用的应变 token 给序列添了噪声、1 epoch 没学会忽略)。
- **根因 = 数据太少**:GW 界训 1D 应变检测器用**数万条注入**;我们只有 ~72 个训练事件,从零学的 StrainEncoder 学不出来(A 失败完全在预期内)。这印证了方案 §8 的判断——**应变/融合的价值被"够不够数据"门控住,而数据这关绕不开注入(E3/E4)**。

### 7.4 第 2 轮(2026-06-27):重开 27B —— 换编码器 + 换模型
第 1 轮负结果的诊断是"**从零 CNN 在小数据上学不出**",**不是融合方法本身错**。故第 1 轮"不放大 27B"的结论作废,按用户意见换两处再判:
- **变更①**:应变编码器 CNN → **patch + 线性投影 + 自注意力**(更省参,靠预训练 LLM 学,而非从零卷积);
- **变更②**:换更强的模型排除"模型太弱"这一 confound,最终定 **Qwen3.6-27B**(E1 最佳)。

### 7.5 第 2 轮训练设计(正在跑)
**目标**:同一问题(加应变还要不要图),用新编码器 + 强模型再判;并看新编码器能否**单独**学出应变(对比 CNN 的 0.43)。

**数据**(同前):每样本 = viridis 频谱图 + 对应白化 4s 应变(2048Hz / 8192 点,单探测器);检测 YES/NO。

**应变编码器 `StrainPatchEncoder`**(核心改动,`training/models/strain_encoder.py`):
- 8192 点切 N 段(`patch_size` 256→32 token / 128→64 token);
- 每段 `Linear(patch_size→hidden)` + GELU + Linear(小 MLP)→ 1 token + 可学位置编码;
- 过 3 层 `nn.TransformerEncoderLayer`(自注意力,块间交互)→ N 个应变 token;LayerNorm。
- 比 CNN 新参数少、归纳偏置轻,把"看懂 chirp"交给预训练 LLM(LoRA)。

**融合机制 `FusionVLM`**(不变):文本 embed → 图像特征(`get_image_features().pooler_output`)散射到图 token 位 → 应变 token 注入序列占位位 → `inputs_embeds` 喂 LLM;Qwen 自算 M-RoPE `position_ids`(关键字传参,兼容 Qwen2.5-VL 与 Qwen3.5 两套 `get_rope_index` 签名)。

**模型 = Qwen3.6-27B(架构 qwen3_5,E1 最佳)**,定型经一番排障:
- **Gemma4(E4B/31B)= PLE 逐层嵌入架构,与 inputs_embeds 注入式融合不兼容**(只给 embeds 时它会反推 input_ids → 拿 26 万词表逐比对 → 955 GiB OOM)→ 弃;
- **Qwen3.6-27B 用混合线性注意力(gated delta)**,默认走纯 torch 慢回退(~20s/batch)→ 本地编译 **fla + causal_conv1d**(GB10 / Blackwell / CUDA13 上编译成功)激活快路径 → **~5-6s/batch**;
- **bf16 全精度 + 梯度检查点**(省激活,前向峰值 ~79G / 121G;bf16 裸跑要 117G 会 OOM)。

**训练配方**(对齐 E1 获胜配方):原生 transformers + PEFT(非 Unsloth);LoRA r32/α32/dropout0;**有效 batch 8**(batch 1 × grad_accum 8;GPU batch 1 已 96% 满载);lr 2e-4(StrainEncoder lr 1e-3);**thinking 关闭**(训练/推理一致);2 epoch;max_pixels 262144(5 配置内部一致)。`num_workers=0`(崩溃不留孤儿)。

**消融 5 配置**:
| 配置 | 输入 | 应变编码器 | 答什么 |
|---|---|---|---|
| C | 仅图 | — | 27B 纯图基线 |
| B-32 / B-64 | 图+应变 | patch+attn(32/64) | **B vs C:加应变有没有用** |
| A-32 / A-64 | 仅应变 | patch+attn(32/64) | 新编码器能否**单独**学出应变(对比 CNN 0.43) |

各 2 epoch。代码:`training/{fusion_model,fusion_collator,train_fusion}.py` + `models/strain_encoder.py` + `evaluation/evaluate_fusion.py` + `configs/fusion_qwen36_27b.yaml` + `run_q27_ablation.sh`。

### 7.6 第 2 轮结果(Qwen3.6-27B,bf16,2 epoch,viridis,max_pixels 262144)
| 配置 | 输入 | **ROC-AUC** | PR-AUC | R@0.5 | R@FPR≤5% |
|---|---|---|---|---|---|
| **C 仅图** | 图 | **0.9557** | 0.9573 | 0.830 | 0.822 |
| B-32 图+应变 | 图+应变(32) | **0.9584** | 0.9594 | 0.830 | 0.785 |
| B-64 图+应变 | 图+应变(64) | 0.9560 | 0.9570 | 0.822 | 0.785 |
| A-32 仅应变 | 应变(32) | 0.5184 | 0.5104 | 0.993 | 0.0 |
| A-64 仅应变 | 应变(64) | 0.5556 | 0.5295 | 0.941 | 0.0 |

**结论(用最强模型 + 更好编码器复判,与第 1 轮一致):**
- **图是主力且很强**:C=0.956,与 E1 同档 → 融合代码图像路径无误、基线扎实。
- **加应变在检测上无增益(融合中性)**:B-32 比 C 仅 +0.003、B-64 持平;n=270 时 AUC 标准误 ~0.018,**差异在噪声内,统计上与纯图无别**;严格工作点 R@FPR≤5% 上 B(0.785)还略低于 C(0.822)。
- **仅应变接近随机(0.52/0.56),但比第 1 轮 CNN(0.43,低于随机)进步**:patch+自注意力编码器**确实学出了东西**(0.43→0.52–0.56),只是仍太弱,撼不动图的 0.96(且 R@FPR5%=0、R@0.5≈1 = 基本"全判 YES"+微弱信号)。

**对"加应变还要不要图"的最终回答:要图;应变在检测上不帮忙。** 关键:这次**换了最强模型(27B)+ 更强编码器,结论没翻盘** → 第 1 轮负结果**不是模型太弱、也不是 CNN 不行,是数据墙(90 事件)**:① 27B 纯图检测已 0.96,留给应变的提升空间本就极小;② 应变的真正价值在**参数估计(带相位),不在检测**——而那需**注入数据(E3/E4)**才喂得动。编码器换对了方向(应变可学性 0.43→0.55),但数据这关绕不开。

---

## 8. 风险与边界

- **Qwen `inputs_embeds` + 手工拼 token 的 rope/mask 坑**:我们在评估已踩过 `get_rope_index` 错位;手动改序列长度更易触发,需单步调试 attention_mask/position_ids/image_grid_thw,小模型先跑通。
- **白化正确性**:窗口边缘、PSD 估计——抽查频谱可见性把关,必要时对齐 02 的 q_transform 内部白化口径。
- **数据稀缺仍在**:90 事件对从零学的 StrainEncoder 也少;检测可能够,但**别指望融合一举解决参数**——参数仍需注入(E3/E4)。
- **小模型→27B 非线性迁移**:小模型结论是方向性的,最终以 27B 为准。
- **不在本轮**:注入管线(E3/E4)、多探测器相干融合、参数任务融合版、目标函数改造(序数/回归)——视本轮结论再排。这条线的远景(致密 GR 流形 + 偏离/异常检测,驱动理论)见讨论记录。
