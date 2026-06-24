# 论文原文索引(76 篇)

> 配合 [docs/05_training_methods_survey.md](../05_training_methods_survey.md) 阅读。每篇:标题(arXiv 链接)→ 本地 PDF → 一句话"对我们的意义"。
> ⭐ = 建议优先读;⚠️ = 避坑类(说明某做法对我们有害/无用)。
> 下载脚本:`bash docs/papers/download.sh`(读 `manifest.tsv`)。

---

## 🔥 建议最先读的 10 篇(我们决策的主要依据)

1. ⭐⭐ [Training Strategies for Deep Learning GW Searches](https://arxiv.org/abs/2106.03741) — [PDF](GW/training-strategies-usr__2106.03741.pdf) — **弱信号训练能泛化到强、反之不行**;课程要"早暴露弱信号";USR 修正低 FAR 排序。本项目弱信号策略的基石。
2. ⭐⭐ [MLGWSC-1 机器学习引力波搜索挑战](https://arxiv.org/abs/2209.11146) — [PDF](GW/mlgwsc-1__2209.11146.pdf) — 社区标准基准与评价口径(灵敏距离 vs FAR);高斯噪声达匹配滤波 95%、真实噪声仅 70%。
3. ⭐⭐ [AResGW:深度残差网做 GW 检测](https://arxiv.org/abs/2211.01520) — [PDF](GW/aresgw__2211.01520.pdf) — 当前 SOTA,1D ResNet+DAIN+SNR课程,真实噪声超匹配滤波。注入+课程的范本。
4. ⭐ [Does Prompt Loss Matter for SFT?](https://arxiv.org/abs/2401.13586) — [PDF](LOSS/does-prompt-loss-matter__2401.13586.pdf) — 证实"答案短、提示长"时模板token的损失处理很要紧。对应我们 A1(决策token损失聚焦)。
5. ⭐ [Learning Rate Matters: Vanilla LoRA May Suffice](https://arxiv.org/abs/2602.04998) — [PDF](PEFT/lora-lr-matters__2602.04998.pdf) — LoRA 变体祛魅:公平调 LR 后 DoRA/PiSSA 等优势消失。别在变体上花时间。
6. ⭐ [VLMs Are Few-Shot Audio Spectrogram Classifiers](https://arxiv.org/abs/2411.12058) — [PDF](VLM/vlm-fewshot-audio-spectrogram__2411.12058.pdf) — 离我们最近的类比:VLM 读频谱图图像分类可行但有限;渲染方式(colormap/坐标轴)显著影响结果。
7. ⭐ [Probabilities of Chat LLMs Are Miscalibrated but Predict Correctness](https://arxiv.org/abs/2402.13213) — [PDF](LOSS/chat-llm-miscalibrated__2402.13213.pdf) — 支持"用 P(YES) 当分数排序合理,但要事后校准"。对应 A2。
8. ⭐ [Comparative study of 1D and 2D CNN for GW detection](https://arxiv.org/abs/2312.04855) — [PDF](GW/1d-vs-2d-cnn__2312.04855.pdf) — 1D 对 BBH 更敏感、2D 对 BNS 好、1D+2D 集成最佳。我们 BBH 频谱图路线的现实定位。
9. ⭐ [CNNs: a magic bullet for GW detection?](https://arxiv.org/abs/1904.08693) — [PDF](GW/magic-bullet__1904.08693.pdf) — 为何 AUC/accuracy 会误导,应报"固定低 FAR 处 recall + efficiency-vs-SNR"。
10. ⭐ [Model Soups](https://arxiv.org/abs/2203.05482) — [PDF](PEFT/model-soups__2203.05482.pdf) — 多模型权重平均免费提精度/鲁棒性。对应 A4。

---

## PEFT / 微调取舍(15)

- ⭐ [Learning Rate Matters: Vanilla LoRA May Suffice](https://arxiv.org/abs/2602.04998) — [PDF](PEFT/lora-lr-matters__2602.04998.pdf) — 变体祛魅(LR 决定一切)。
- [Which LoRA? 多语种实证](https://arxiv.org/abs/2606.10428) — [PDF](PEFT/which-lora__2606.10428.pdf) — 变体间无显著差异,花哨方法知识保持更差。
- ⭐ [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) — [PDF](PEFT/lora-learns-less-forgets-less__2405.09673.pdf) — 小数据 LoRA 更稳、更不易遗忘。
- [LoRA vs Full FT: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228) — [PDF](PEFT/lora-vs-full-illusion__2410.21228.pdf) — intruder dimensions;别把 rank 压太低。
- [DoRA](https://arxiv.org/abs/2402.09353) — [PDF](PEFT/dora__2402.09353.pdf) — 方向+大小分解;对 rank 鲁棒,但非大涨点。
- [PiSSA](https://arxiv.org/abs/2404.02948) — [PDF](PEFT/pissa__2404.02948.pdf) — SVD 主成分初始化;主要是收敛快。
- [LoRA+](https://arxiv.org/abs/2402.12354) — [PDF](PEFT/lora-plus__2402.12354.pdf) — B 矩阵更大 LR;本质是调 LR。
- [rsLoRA](https://arxiv.org/abs/2312.03732) — [PDF](PEFT/rslora__2312.03732.pdf) — α/√r 缩放;仅高 rank 有意义。
- [VeRA](https://arxiv.org/abs/2310.11454) — [PDF](PEFT/vera__2310.11454.pdf) — 极省参但掉点;我们不缺显存,不适用。
- [AdaLoRA](https://arxiv.org/abs/2303.10512) — [PDF](PEFT/adalora__2303.10512.pdf) — 自适应 rank 预算;训练更贵。
- [MoRA](https://arxiv.org/abs/2405.12130) — [PDF](PEFT/mora__2405.12130.pdf) — 高有效秩;偏"记新知识"任务。
- [OLoRA](https://arxiv.org/abs/2406.01775) — [PDF](PEFT/olora__2406.01775.pdf) — 正交初始化;证据弱。
- ⚠️ [NEFTune](https://arxiv.org/abs/2310.05914) — [PDF](PEFT/neftune__2310.05914.pdf) — 嵌入加噪;只帮开放式对话,对分类无用。
- ⭐ [Model Soups](https://arxiv.org/abs/2203.05482) — [PDF](PEFT/model-soups__2203.05482.pdf) — 权重平均(A4)。
- ⭐ [WiSE-FT(鲁棒微调)](https://arxiv.org/abs/2109.01903) — [PDF](PEFT/wise-ft__2109.01903.pdf) — 微调权重与原始权重插值,抗分布漂移。

## 损失 / 难样本 / 课程 / 校准(19)

- ⭐ [Long-tail via Logit Adjustment](https://arxiv.org/abs/2007.07314) — [PDF](LOSS/logit-adjustment__2007.07314.pdf) — 决策位 logit 偏移补漏检(A 档可叠加)。
- [PolyLoss](https://arxiv.org/abs/2204.12511) — [PDF](LOSS/polyloss__2204.12511.pdf) — CE 上加一项,比 focal 灵活。
- [Class-Balanced Loss](https://arxiv.org/abs/1901.05555) — [PDF](LOSS/class-balanced-loss__1901.05555.pdf) — 有效样本数加权(我们已均衡,参考)。
- [Focal Loss](https://arxiv.org/abs/1708.02002) — [PDF](LOSS/focal-loss__1708.02002.pdf) — 难样本加权;高 γ 会过拟合噪声标签。
- ⚠️ [Label Smoothing 损害选择性分类](https://arxiv.org/abs/2403.14715) — [PDF](LOSS/label-smoothing-selective__2403.14715.pdf) — 伤"按置信排序卡阈值";建议不用。
- ⭐ [Does Prompt Loss Matter](https://arxiv.org/abs/2401.13586) — [PDF](LOSS/does-prompt-loss-matter__2401.13586.pdf) — short-completion 下模板token损失要紧(A1)。
- ⭐ [SFTKey:强调关键答案 token](https://arxiv.org/abs/2512.21017) — [PDF](LOSS/sftkey__2512.21017.pdf) — 直接命名"loss dilution",只在答案段算梯度(A1)。
- ⭐ [Rho-1: Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965) — [PDF](LOSS/rho-1__2404.07965.pdf) — 把损失聚焦到关键 token 是大杠杆。
- [SFT Needs to Unlock Token Priority](https://arxiv.org/abs/2602.01227) — [PDF](LOSS/sft-token-priority__2602.01227.pdf) — 连续 token 加权(A1 的软版)。
- [OHEM](https://arxiv.org/abs/1604.03540) — [PDF](LOSS/ohem__1604.03540.pdf) — 在线难例挖掘(对噪声敏感)。
- [GHM(梯度协调)](https://arxiv.org/abs/1811.05181) — [PDF](LOSS/ghm__1811.05181.pdf) — 同时压易样本和离群点,比 focal 更适合低 SNR。
- [NV-Retriever:positive-aware 难负挖掘](https://arxiv.org/abs/2407.15831) — [PDF](LOSS/nv-retriever__2407.15831.pdf) — 滤掉"其实是正样本的假难负"。
- [Learning to Reweight Examples](https://arxiv.org/abs/1803.09050) — [PDF](LOSS/learning-to-reweight__1803.09050.pdf) — 用干净验证集学样本权重,抗噪。
- [DisCL(扩散课程)](https://arxiv.org/abs/2410.13674) — [PDF](LOSS/discl__2410.13674.pdf) — 生成难度梯度样本,规避错标。
- [Temperature Scaling(校准)](https://arxiv.org/abs/1706.04599) — [PDF](LOSS/temperature-scaling__1706.04599.pdf) — 单调校准,不改 AUC。
- ⭐ [What Matters for Calibrating VLMs](https://arxiv.org/abs/2402.07417) — [PDF](LOSS/calibrating-vlms__2402.07417.pdf) — TS 对 VLM 一致有效、跨漂移。
- ⭐ [Chat LLM 概率校准差但排序好](https://arxiv.org/abs/2402.13213) — [PDF](LOSS/chat-llm-miscalibrated__2402.13213.pdf) — 用 P(YES) 排序合理(A2)。
- [Focal/TS/Properness 关系](https://arxiv.org/abs/2408.11598) — [PDF](LOSS/focal-ts-properness__2408.11598.pdf) — focal 改善校准但 improper,仍需事后 TS。
- [Can LLMs Express Their Uncertainty?](https://arxiv.org/abs/2306.13063) — [PDF](LOSS/llm-express-uncertainty__2306.13063.pdf) — token 概率优于口头置信度。

## 数据中心 / 增广 / 自监督(12)

- [SpecMix](https://arxiv.org/abs/2108.03020) — [PDF](DATA/specmix__2108.03020.pdf) — 频谱图条带混合,优于 Mixup/CutMix。
- [SpecAugment++](https://arxiv.org/abs/2103.16858) — [PDF](DATA/specaugment-pp__2103.16858.pdf) — 时间/频率掩码(频率掩码慎用于弱信号)。
- [Sage:GW 检测的 ML 偏差与缓解](https://arxiv.org/abs/2501.13846) — [PDF](DATA/sage-gw-biases__2501.13846.pdf) — 物理域激进增广(PSD缩放/噪声爆发/时间扭曲)。
- ⭐ [Machine Learning in GW Astronomy(综述)](https://arxiv.org/abs/2401.07406) — [PDF](DATA/ml-gw-astronomy-review__2401.07406.pdf) — 记录 AResGW 的注入+课程配方,领域全景。
- [Aframe:实时 CBC 检测流水线](https://arxiv.org/abs/2403.18661) — [PDF](DATA/realtime-cbc__2403.18661.pdf) — SNR 课程 + 拒绝采样(SNR>4)。
- [Frozen DINOv2 做 GW glitch](https://arxiv.org/abs/2605.28572) — [PDF](DATA/dinov2-gw-glitch__2605.28572.pdf) — 现成视觉 SSL 特征对 GW 谱开箱可用(纯 CPU)。
- [MAE 自预训练用于医学影像](https://arxiv.org/abs/2203.05573) — [PDF](DATA/mae-medical__2203.05573.pdf) — 小数据 MAE 优于 ImageNet 迁移。
- [Masked Spectrogram Modeling(音频 MAE)](https://arxiv.org/abs/2204.12260) — [PDF](DATA/masked-spectrogram-modeling__2204.12260.pdf) — 掩码谱建模通用音频表征。
- [SSL 音频表征做数据高效 ASC](https://arxiv.org/abs/2408.14862) — [PDF](DATA/ssl-audio-asc__2408.14862.pdf) — 有限标注下 SSL 显著提升。
- [BEATs](https://arxiv.org/abs/2212.09058) — [PDF](DATA/beats__2212.09058.pdf) — 声学 tokenizer 自监督。
- [Quality vs Quantity for Small Models](https://arxiv.org/abs/2411.15821) — [PDF](DATA/quality-vs-quantity-small__2411.15821.pdf) — 小数据"质量>数量"。
- [Time-Series 增广综述](https://arxiv.org/abs/2310.10060) — [PDF](DATA/timeseries-aug-survey__2310.10060.pdf) — 时序增广方法系统比较。

## VLM 专属 / 推理 / RL / TTA(20)

- [Qwen2-VL](https://arxiv.org/abs/2409.12191) — [PDF](VLM/qwen2-vl__2409.12191.pdf) — 原生动态分辨率(NaViT)。
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) — [PDF](VLM/qwen2.5-vl__2502.13923.pdf) — window attention 高分辨率。
- [Qwen3-VL](https://arxiv.org/abs/2511.21631) — [PDF](VLM/qwen3-vl__2511.21631.pdf) — SigLIP-2 + DeepStack。
- ⭐ [Task-Aware Resolution Optimization](https://arxiv.org/abs/2510.09822) — [PDF](VLM/task-aware-resolution__2510.09822.pdf) — 分辨率/视觉token有任务相关拐点,该扫描找平台。
- ⭐ [解冻视觉塔提升 Chart QA(+16.9)](https://arxiv.org/abs/2407.20174) — [PDF](VLM/chartqa-unfreeze-vision__2407.20174.pdf) — 域差距大时解冻视觉塔有大收益(频谱图类比)。
- [MedM-VL:好的医学 LVLM](https://arxiv.org/abs/2504.04323) — [PDF](VLM/medm-vl__2504.04323.pdf) — 解冻收益与域差距成正比。
- ⭐ [VLMs 读音频频谱图分类](https://arxiv.org/abs/2411.12058) — [PDF](VLM/vlm-fewshot-audio-spectrogram__2411.12058.pdf) — 最直接类比;渲染方式很重要。
- [DeepSeekMath(GRPO)](https://arxiv.org/abs/2402.03300) — [PDF](VLM/deepseekmath-grpo__2402.03300.pdf) — GRPO 来源。
- ⭐ [RL 真能拓展推理吗?](https://arxiv.org/abs/2504.13837) — [PDF](VLM/rl-incentivize-reasoning__2504.13837.pdf) — RLVR 只锐化、不拓展能力。
- ⭐ [Spurious Rewards](https://arxiv.org/abs/2506.10947) — [PDF](VLM/spurious-rewards__2506.10947.pdf) — 随机/错误奖励也能"涨点",Qwen 特有假象。
- ⭐ [Visual-RFT](https://arxiv.org/abs/2503.01785) — [PDF](VLM/visual-rft__2503.01785.pdf) — 少样本视觉 RL 胜过 SFT(唯一强对题的 RL 正面)。
- [Beyond Binary Rewards(RLCR)](https://arxiv.org/abs/2507.16806) — [PDF](VLM/rlcr__2507.16806.pdf) — 加校准奖励避免 RL 过自信。
- ⭐ [To CoT or not to CoT?](https://arxiv.org/abs/2409.12183) — [PDF](VLM/to-cot-or-not__2409.12183.pdf) — CoT 只在数学/符号大涨,感知任务收益小。
- ⭐ [Compositional CoT(CCoT)](https://arxiv.org/abs/2311.17076) — [PDF](VLM/ccot__2311.17076.pdf) — 纯 prompt"先结构化描述再答"(可低成本试)。
- [Visual CoT Dataset](https://arxiv.org/abs/2403.16999) — [PDF](VLM/visual-cot-dataset__2403.16999.pdf) — 先定位关键区域再答提升细粒度 VQA。
- [LLaVA-CoT](https://arxiv.org/abs/2411.10440) — [PDF](VLM/llava-cot__2411.10440.pdf) — 分阶段视觉推理。
- [Self-Consistency](https://arxiv.org/abs/2203.11171) — [PDF](VLM/self-consistency__2203.11171.pdf) — 多数投票(主要利于推理任务)。
- ⭐ [LoRA-Ensemble](https://arxiv.org/abs/2405.14438) — [PDF](VLM/lora-ensemble__2405.14438.pdf) — 单主干+多 LoRA 集成,单机最优集成形态。
- [Deep Ensembles 真有必要吗](https://arxiv.org/abs/2202.06985) — [PDF](VLM/deep-ensembles-necessary__2202.06985.pdf) — 集成 OOD 收益由 ID 表现决定。
- ⚠️ [I Can't Believe TTA Is Not Better](https://arxiv.org/abs/2604.09697) — [PDF](VLM/tta-not-better__2604.09697.pdf) — 几何/掩码类 TTA 多数变差;只用物理合法视图。

## GW 检测 ML 领域(10)

- ⭐⭐ [MLGWSC-1 基准](https://arxiv.org/abs/2209.11146) — [PDF](GW/mlgwsc-1__2209.11146.pdf) — 社区标准 + 评价口径。
- ⭐ [Gabbard 2018:DL 匹配匹配滤波](https://arxiv.org/abs/1712.06041) — [PDF](GW/gabbard2018__1712.06041.pdf) — 1D CNN 复现匹配滤波 ROC。
- [George & Huerta:Deep Filtering](https://arxiv.org/abs/1711.07966) — [PDF](GW/george-huerta__1711.07966.pdf) — 首用真实数据,检测+参数估计。
- ⭐⭐ [AResGW](https://arxiv.org/abs/2211.01520) — [PDF](GW/aresgw__2211.01520.pdf) — 真实噪声超匹配滤波;DAIN+SNR课程。
- [AResGW 增强版(ML 驱动新发现)](https://arxiv.org/abs/2407.07820) — [PDF](GW/aresgw-enhanced__2407.07820.pdf) — 分层 trigger 压低 FAR。
- ⭐⭐ [Training Strategies for DL GW Searches](https://arxiv.org/abs/2106.03741) — [PDF](GW/training-strategies-usr__2106.03741.pdf) — 弱→强泛化;USR;课程。
- ⭐ [1D vs 2D CNN for GW](https://arxiv.org/abs/2312.04855) — [PDF](GW/1d-vs-2d-cnn__2312.04855.pdf) — BBH 用 1D 更好,集成最佳。
- ⭐ [CNNs: a magic bullet?(评价批判)](https://arxiv.org/abs/1904.08693) — [PDF](GW/magic-bullet__1904.08693.pdf) — 为何别只看 AUC。
- ⭐ [LLM 读 CQT token 检测 GW](https://arxiv.org/abs/2512.04031) — [PDF](GW/llm-cqt-gw__2512.04031.pdf) — 最近先例(97.4% 但无 SNR 分层,慎读结论)。
- [Gravity Spy(Q-scan + CNN glitch 分类)](https://arxiv.org/abs/1611.04596) — [PDF](GW/gravity-spy__1611.04596.pdf) — 频谱图图像范式之源。
