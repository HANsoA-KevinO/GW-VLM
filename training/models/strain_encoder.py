"""
StrainEncoder1D:把白化后的 1D 应变(4s @ 2048Hz = 8192 点)编码成 N 个 token embedding,
投影到 LLM 的 hidden_dim,供多模态融合(方法2)拼进序列。

结构参考 GW 1D 检测网(Gabbard 2018 / AResGW):Conv1d(stride 降采)+ BN + GELU 堆叠
→ AdaptiveAvgPool1d 到固定 token 数 → Linear 投影到 hidden_dim。从零训练(非 LoRA)。
"""
import torch
import torch.nn as nn


class StrainEncoder1D(nn.Module):
    def __init__(self, hidden_dim: int, n_tokens: int = 12, in_len: int = 8192,
                 channels=(32, 64, 128, 256)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_tokens = n_tokens
        self.in_len = in_len

        layers = []
        c_in = 1
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=16, stride=4, padding=6),
                nn.BatchNorm1d(c_out),
                nn.GELU(),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)          # → [B, C, n_tokens]
        self.proj = nn.Linear(channels[-1], hidden_dim)     # → hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        """strain: [B, in_len] → [B, n_tokens, hidden_dim]"""
        if strain.dim() == 2:
            strain = strain.unsqueeze(1)                    # [B, 1, in_len]
        x = self.conv(strain)                               # [B, C, L']
        x = self.pool(x)                                    # [B, C, n_tokens]
        x = x.transpose(1, 2)                               # [B, n_tokens, C]
        x = self.proj(x)                                    # [B, n_tokens, hidden]
        return self.norm(x)
