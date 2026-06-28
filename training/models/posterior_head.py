"""
GaussianPosteriorHead:从一个特征向量(VLM 末层 hidden,或经典应变编码器的池化特征)
解码出物理参数的**后验**(高斯:每参数 μ + logσ),用 NLL 训练。

设计要点(见 docs/07 §A):
- 参数在**标准化空间**里建高斯:chirp_mass/distance 取 log(几何分布),chi_eff 取 identity,
  再各自 z-score(均值/方差由 train 正样本统计,存 norm_stats.json)。
- 头在 **fp32** 运行(NLL 对精度敏感),logσ clamp 稳住数值。
- 评估时 invert 回物理量:mass/distance 用 log-normal 中位数 exp(μ_t);并给可信区间 / PIT。

PARAM_NAMES 是参数顺序的**唯一真源**;08_export 发的是"按名字的 dict",加载端按 PARAM_NAMES 排序,
所以顺序解耦、不会错位。
"""
import math

import numpy as np
import torch
import torch.nn as nn

# 参数顺序的唯一真源;PARAM_LOG 指明哪些取 log 变换
PARAM_NAMES = ["chirp_mass", "distance", "chi_eff"]
PARAM_LOG = [True, True, False]
N_PARAMS = len(PARAM_NAMES)


# --- 标准化 / 反演(numpy,数据侧与评估侧共用)-------------------------------
def _fwd_transform(phys: np.ndarray) -> np.ndarray:
    """物理量 → 变换空间(log for mass/distance)。phys: [..., N_PARAMS]"""
    t = np.asarray(phys, dtype=np.float64).copy()
    for i, islog in enumerate(PARAM_LOG):
        if islog:
            t[..., i] = np.log(np.clip(t[..., i], 1e-6, None))
    return t


def compute_norm_stats(phys_positive: np.ndarray) -> dict:
    """用 train 正样本的物理参数算标准化统计(变换空间的 mean/std)。"""
    t = _fwd_transform(np.asarray(phys_positive, dtype=np.float64))
    mean = np.nanmean(t, axis=0)
    std = np.nanstd(t, axis=0) + 1e-6
    return {"names": PARAM_NAMES, "log": PARAM_LOG,
            "mean": mean.tolist(), "std": std.tolist()}


def standardize(phys: np.ndarray, stats: dict) -> np.ndarray:
    """物理量 → 标准化 z(训练目标)。NaN(负样本无参数)原样传出。"""
    t = _fwd_transform(phys)
    return (t - np.asarray(stats["mean"])) / np.asarray(stats["std"])


def invert(z_mu: np.ndarray, z_logstd: np.ndarray, stats: dict):
    """标准化后验 → 物理量。返回 (median_phys, t_mu, t_sigma):
    - median_phys:物理量点估计(mass/distance 用 exp(μ_t) 的 log-normal 中位数)
    - t_mu, t_sigma:变换空间的均值/标准差(用于 PIT / 可信区间)
    """
    mean = np.asarray(stats["mean"]); std = np.asarray(stats["std"])
    t_mu = z_mu * std + mean
    t_sigma = np.exp(z_logstd) * std
    median = t_mu.copy()
    for i, islog in enumerate(PARAM_LOG):
        if islog:
            median[..., i] = np.exp(t_mu[..., i])
    return median, t_mu, t_sigma


# --- 头 ---------------------------------------------------------------------
class GaussianPosteriorHead(nn.Module):
    def __init__(self, hidden_dim: int, n_params: int = N_PARAMS, mlp_hidden: int = 256,
                 dropout: float = 0.1, logstd_min: float = -7.0, logstd_max: float = 5.0):
        super().__init__()
        self.n_params = n_params
        self.logstd_min, self.logstd_max = logstd_min, logstd_max
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(),
        )
        self.out = nn.Linear(mlp_hidden, 2 * n_params)
        # 零初始化输出层 → 初始 μ=0, logσ=0(σ=1)→ 初始 NLL≈0.5·z²,数值良好,
        # 避免随机初始化下 logσ 极端 → inv-var 爆炸 → nan 梯度污染共享权重。
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor):
        """h: [B, hidden_dim](任意 dtype)→ (mu, logstd):[B, n_params] fp32"""
        h = h.float()
        z = self.out(self.mlp(self.norm(h)))
        mu, raw_logstd = z.chunk(2, dim=-1)
        logstd = torch.clamp(raw_logstd, self.logstd_min, self.logstd_max)
        return mu, logstd


def gaussian_nll(mu: torch.Tensor, logstd: torch.Tensor,
                 target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """高斯 NLL(标准化空间),只在 valid(正样本)上平均。
    mu/logstd/target: [B, P];valid: [B] bool。返回标量。

    关键:负样本 target 是 NaN。必须在进入 (target-mu) 前就把 NaN 清零,否则
    反向链里 "0(被 mask 的梯度) × NaN(局部导数) = NaN" 会污染头的梯度 → 全 nan。
    清零后该样本数值有限,再用 valid mask 掉其贡献(值与梯度都为 0)。
    """
    target = torch.nan_to_num(target, nan=0.0)        # 先消毒,再算
    inv_var = torch.exp(-2.0 * logstd)
    nll = 0.5 * ((target - mu) ** 2) * inv_var + logstd + 0.5 * math.log(2 * math.pi)  # [B,P]
    nll = nll.sum(dim=-1)                              # [B]
    v = valid.to(nll.dtype)
    return (nll * v).sum() / v.sum().clamp(min=1.0)
