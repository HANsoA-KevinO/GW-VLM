# E1 训练实验记录与关键发现

**日期**：2026-06-15 ~ 06-19
**范围**：E1（纯检测）在 DGX Spark 上的训练基建落地、多模型训练与评估、评估方法论的纠正、瓶颈诊断、数据再处理方向、全量 viridis 双主模型对决。
**一句话结论**：E1 检测最佳 **ROC-AUC 0.940（Qwen3.6-27B，原生多模态）** > Gemma4 31B（0.92）;"低 recall"主要是**判决阈值假象**;真正的天花板是**低 SNR 弱信号的物理信噪比**,提升它要靠**注入(E3/E4)**或接受 SNR 下限,而非调色/位深/堆 epoch(均已实验证伪)。

---

## 1. 训练/评估基建(本阶段新建)

| 文件 | 作用 |
|---|---|
| `data_pipeline/scripts/08_export_training_format.py` | 中间 jsonl → 统一 messages 训练格式(Qwen/Gemma 通用),`--schema detection_only/multitask` 切 E1/E2,`--image-path relative` 便于跨机器 |
| `training/train_vlm.py` | Unsloth `FastVisionModel` LoRA 训练器,YAML+CLI 驱动 |
| `training/configs/e1_*.yaml` | 各模型 E1 配置 |
| `training/spark_env.sh` | Spark 运行环境(代理/HF/Unsloth/CPATH 等开关一键 source) |
| `training/run_spark.sh` | detached 运行包装(train/eval) |
| `evaluation/metrics.py` | accuracy/F1/混淆矩阵 + YES/NO 解析 |
| `evaluation/evaluate.py` | 贪心解码评估(取 argmax YES/NO) |
| `evaluation/evaluate_prob.py` | **概率版评估**:抽每样本 P(YES) → ROC-AUC/PR-AUC/阈值扫描/工作点 + 每样本分数导出 |
| `evaluation/plot_loss.py` | 从 trainer_state.json 画 train/eval 损失曲线 |
| `evaluation/plot_snr.py` | P(YES) vs SNR 诊断(detection efficiency vs SNR) |
| `data_pipeline/scripts/render_compare.py` / `render_zoom.py` | 渲染对比(窗口×色图)视觉实验 |
| `02_generate_spectrograms.py` | 新增 `--cmap`/`--outdir`(支持 viridis 等彩色,灰度默认不变) |

数据集口径(排除 V1 后):**train 2394 / val 306 / test 270**,正负各 1485,按事件 ID 切分,seed=42。

---

## 2. 三个 E1 模型的配置与结果

| 模型 | 精度 | LoRA r/α | lr | 有效batch | 分辨率 | epoch(步) | 时长 |
|---|---|---|---|---|---|---|---|
| Gemma4 **E4B** | 4bit(QLoRA) | 32/32 | 2e-4 | 8 | 512(Unsloth默认) | 3(900) | ~2.5h |
| Gemma4 **31B 基线** | bf16 | **8/16** | **5e-5** | **16** | 512 | 3(450) | ~7.5h |
| Gemma4 **31B v2** | bf16 | 32/32 | 2e-4 | 8 | **560(~1135px)** | 2(600) | ~10h |
| Qwen3.6-27B | bf16 | 32/32 | 2e-4 | 8 | 原生 | 2(600,被迫停) | ~37h(慢) |

**评估结果(270 test 样本):**

| 模型 | 贪心 Acc | 贪心 Recall | **ROC-AUC** | PR-AUC | recall@FPR≤5% |
|---|---|---|---|---|---|
| E4B | 81.5% | 69% | 0.904 | 0.892 | 65% |
| 31B 基线 | 66% | 34% | 0.858 | 0.872 | 59% |
| **31B v2** | 76% | 53% | **0.922** 🥇 | **0.927** | **75%**(P 94%, acc 85%) |

> 31B v2 在 max-F1 阈值(~0.15)下:acc 86% / recall 81% / precision 89%。

---

## 2.5 全量 viridis 双主模型对决（2026-06-18~19）

按"一步到位、用新彩色图(viridis)"跑完两个主模型的全量 E1,统一获胜配方(r32/α32/lr2e-4/dropout0/有效batch8):

| 模型 | 色彩/epoch | 分辨率 | **ROC-AUC** | PR-AUC | R@0.5 | R@FPR≤5% | 时长 |
|---|---|---|---|---|---|---|---|
| 🥇 **Qwen3.6-27B** | viridis / **2** | 原生(~1024 img tok) | **0.940** | **0.945** | **0.837** | 0.741(P94%) | ~37h(被迫停2ep) |
| Gemma4 31B v2 | 灰度 / 2 | 560 | 0.922 | 0.927 | 0.62 | 0.75(P94%) | ~10h |
| Gemma4 31B | viridis / **3** | 560 | 0.905 | 0.909 | 0.622 | 0.689 | ~15h |
| Gemma4 E4B | viridis / 3 | 512 | 0.904 | 0.901 | 0.763 | 0.667 | ~2.5h |

**结论:**
- **Qwen3.6-27B(原生多模态)是 E1 最佳模型**:ROC-AUC 0.940,且**默认 0.5 阈值下 recall 就有 0.837**(Gemma 在 0.5 仅 ~0.62、要调阈值才追上),校准明显更好。弱信号也更强(SNR<8 档 recall 0.89@0.5)。仅 2 epoch 即超过 Gemma 31B 的 3 epoch。
- **第 3 个 epoch 没用、甚至轻微有害**:Gemma 31B viridis **3ep(0.905)< 灰度 2ep(0.922)**;train_loss 更低(0.030)但 test AUC 没升 → 过拟合迹象(差异在统计噪声内,即"无增益")。**印证了"补 epoch 不会有太大变化"的判断**。
- **彩色(viridis)无显著增益**:E4B viridis 0.9046 ≈ 灰度 0.9042;31B viridis(0.905)未超灰度(0.922)。彩色不亏(用预训练 RGB 编码器),但不是 recall 救星。
- **三类旋钮(色彩 / 分辨率>560 / epoch>2)都不能突破 ~0.92–0.94 的天花板** → 瓶颈是数据(物理 SNR),要靠注入(E3/E4)。

> ⚠️ Qwen 评估的工程坑(已解决,值得记):Qwen3.6 在 transformers 5.12.0+Unsloth 的**视觉推理 forward 有 `get_rope_index` 错位 bug**(训练 teacher-forcing 不触发、推理才触发)。修法:评估改用**原生 transformers + PEFT 加载(`--no-unsloth`)** 绕开 Unsloth 全局 patch;输入用未被 patch 的 `AutoProcessor`(产出自洽)。另外排查被两件事拖累:① 目录 `rsync` 在内网抖动时**静默中断、代码没真正落地**(必须单文件同步+`grep` 验证);② `pkill` 没杀净致**旧进程并发写同一日志**造成假象(必须确认进程归零再起)。

---

## 3. 关键发现(方法论 + 结论)

**① 贪心 accuracy 会误导 → 改用 ROC-AUC + FAR 工作点**
贪心解码 = 阈值 0.5。模型在 0.5 处极保守(precision 95%+、recall 低)。但 ROC-AUC 显示判别力其实很强(0.92)。**贪心数说 E4B(81.5%)赢 v2(76%)是假象**——按 ROC-AUC 是 **v2(0.922) > E4B(0.904) > 基线(0.858)**。**以后评估口径:ROC-AUC + PR-AUC + FPR≤5%(≈FAR)工作点**,这也是 docs/02 §7 本就要求的。

**② 31B 基线差 = 欠拟合(配方太保守),非模型不行**
基线用了 docs/02 给大模型的保守设置(rank8/lr5e-5/dropout0.1/batch16),欠拟合。换成 **E4B 验证有效的配方**(r32/α32/lr2e-4/dropout0/有效batch8)后,**ROC-AUC 0.858 → 0.922**(真·模型变强,非仅调阈值)。
- 依据:Unsloth 超参指南(rank16-32、lr2e-4 默认、dropout0)+ Thinking Machines "LoRA Without Regret"(容量要够 / LoRA 需高 lr / 大 batch 伤 LoRA / LoRA 加全层含 MLP)。

**③ "低 recall"主要是阈值假象**
v2 默认 0.5 时 recall 57%;**把阈值降到 ~0.18,recall 升到 75%(FPR 仅 4.4%)、acc 85%**。因为漏检的正样本 P(YES) 多在 0.2-0.45(偏 NO 但不强烈),AUC 高(分得开)→ 降阈值能用很小的误报代价换大量 recall。GW 领域本就按 FAR 选工作点,不用裸 accuracy。

**④ 真正的天花板 = 低 SNR 弱信号(诊断 A)**
P(YES) vs SNR:recall 随 SNR 单调升 —— **<8: 56% / 8-12: 62% / 12-20: 88%**;corr(logSNR, P(YES))=0.67。漏检集中在弱信号,符合物理。这张 **detection efficiency vs SNR** 曲线本身是论文 §7.2 要的结果。

**⑤ 调色/位深救不了弱信号(数据再处理调研)**
- "更多色阶级数"(色图/768/16-bit)对模型**无用**:灰度 256 级的量化步长(0.1 能量)已比噪声(±1-3)细 10-30 倍,再细全淹在噪声里;瓶颈是物理 SNR,不是比特深度。
- 视觉实验证明:**漏检的 SNR9 事件在任何渲染(2s窗/降vmax/viridis)下都没有可辨 chirp**(其峰值能量仅 15.5),信号本就埋在噪声里。
- **但 viridis 让中强信号明显更跳** → 对**冻结预训练编码器**可能有用(它在自然彩色图上学的滤波器对颜色梯度更敏感)→ 值得做 A/B(进行中)。
- 固定映射必须保留(动态逐图归一化会破坏"同能量→同亮度"一致性、引入亮度捷径)。

**⑥ 输入分辨率之前被严重浪费**
Gemma4 按 token 预算决定有效分辨率,档位 **70/140/280/560/1120 → ~410/570/803/1135/1606px**。Unsloth 默认对 Gemma4 兜底 **512px(偏低)**。对我们 1024 源,**560 档(~1135px)是甜点**(吃满源、不浪费);1120 是把 1024 插值放大、无新信息。代码用 `image_tokens` 配置控制(`resize="max"` + 设 `max_soft_tokens`)。

---

## 4. 提升弱信号 recall 的候选方向

1. **注入(E3/E4)** ⭐:造大量"可控低 SNR 信号 + 真噪声"训练样本,让模型学"看不见但可学"的弱信号统计特征(模型已能检测部分 SNR9 事件,说明有可学特征)。需建波形生成+SNR缩放管线(`04_inject_signals`/`05`,要装 lalsuite/pycbc),中等工程,但本就是研究设计的 E3/E4 阶段。
2. **viridis 彩色**(A/B 进行中):借预训练编码器先验,可能小幅提 AUC(主要利好中强信号)。
3. **接受 SNR 物理下限**:把 efficiency-vs-SNR 当结果,检测部分认定"够好"(0.92),转 **E2(检测+参数)** 或鲁棒性测试推进项目。

---

## 5. DGX Spark 环境与踩坑(全部已解决,见 spark_env.sh)

**主机 venv 路线(非 Docker)**:torch 2.12.0+cu130(GB10=sm_121 必需)、torchvision(Gemma4 图像处理器依赖)、Unsloth 2026.6.7 / transformers 5.12 / trl 0.26.1 / bitsandbytes(aarch64 可用)。

| 坑 | 解决 |
|---|---|
| Xet 下载 0 字节卡死 | `HF_HUB_DISABLE_XET=1`(走经典 HTTPS,遵守代理) |
| Unsloth 遥测 120s 超时误报 HF down | `UNSLOTH_DISABLE_STATISTICS=1` |
| Gemma4 处理器 ModuleNotFoundError | 其实是缺 `torchvision`(image_processing_gemma4 依赖) |
| Triton 编译 gcc 失败 | 缺 `Python.h` → 免 sudo:`apt-get download python3.12-dev libpython3.12-dev` + `dpkg -x` + `CPATH` 指过去 |
| dynamo fullgraph 编译 gemma4 失败 | `UNSLOTH_COMPILE_DISABLE=1`(走 eager) |
| 大 bf16 device_map auto 在 UMA 上 offload→accelerate 拒训 | `ACCELERATE_BYPASS_DEVICE_MAP=true`(写进 train_vlm.py 确保进环境) |
| 代理 IP 是 clash(DHCP 会变) | 失效时改 spark_env.sh 顶部 IP;训练/评估不需代理(模型已缓存,可 `HF_HUB_OFFLINE=1`) |

**SSH 运维经验**:① ControlMaster 复用 socket 损坏会导致所有连接 255 → 删 socket 文件即恢复;② 带 `setsid/nohup &` 的启动命令常"无输出"但其实已执行 → **只读重连核对,别盲目重杀**(否则误杀自己启动的进程);③ 长任务用"本地循环 + 每轮短连接轮询"监视,单次连接短、断了不影响下一轮。

---

## 6. 当前状态 / 下一步

- ✅ E1 基建 + 多模型训练评估 + 评估方法论(ROC/FAR)确立 + 瓶颈诊断完成。
- ✅ viridis A/B(彩色≈灰度,无显著增益)+ 全量双主模型对决(§2.5)完成。
- ✅ **Qwen3.6-27B 评估打通**(原生 transformers+PEFT,`--no-unsloth`)。
- 待定:① 注入管线(E3/E4)提弱信号 recall;② 补跑 Qwen3.6-27B 完整 3ep(本次因 272s/步被迫停在 2ep,但 2ep 已最佳);③ 转 E2(检测+参数)。
- **当前最佳 E1 模型**:🥇 **Qwen3.6-27B viridis 2ep(ROC-AUC 0.940,默认0.5阈值 recall 0.837)**,adapter 在 Spark `output/runs/e1_qwen36_27b_viridis_3ep/checkpoint-600/`;次优 Gemma4 31B v2(0.922)。
