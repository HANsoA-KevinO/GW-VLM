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


class StrainPatchEncoder(nn.Module):
    """Patch + 线性投影 + 自注意力(用户拍板的第 2 轮编码器)。

    设计思路:不用从零的卷积"嚼碎"波形,而是像 ViT 切图块一样把 8192 点切成
    n_patches 段,每段经一个【薄翻译器】(Linear[+小MLP])投影成 1 个 token,加可学位置
    编码后过几层自注意力(块间交互),输出 n_patches 个 token。新参数远少于 CNN,
    把"看懂 chirp"的重活留给预训练 LLM(LoRA)。

    n_tokens = in_len // patch_size(自动推导;patch_size=256→32, 128→64)。
    """

    def __init__(self, hidden_dim: int, in_len: int = 8192, patch_size: int = 256,
                 n_attn_layers: int = 3, n_heads: int = 8, mlp_proj: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        assert in_len % patch_size == 0, f"in_len {in_len} 不能被 patch_size {patch_size} 整除"
        self.hidden_dim = hidden_dim
        self.in_len = in_len
        self.patch_size = patch_size
        self.n_tokens = in_len // patch_size

        # 每段输入归一化(白化应变各段幅度差异大,稳一下)
        self.patch_norm = nn.LayerNorm(patch_size)
        if mlp_proj:
            self.proj = nn.Sequential(
                nn.Linear(patch_size, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.proj = nn.Linear(patch_size, hidden_dim)
        # 可学位置编码(让自注意力/LLM 知道块的先后)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        if n_attn_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.attn = nn.TransformerEncoder(enc_layer, num_layers=n_attn_layers)
        else:
            self.attn = nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, strain: torch.Tensor) -> torch.Tensor:
        """strain: [B, in_len] → [B, n_tokens, hidden_dim]"""
        B = strain.shape[0]
        x = strain.reshape(B, self.n_tokens, self.patch_size)   # 切块 [B, n_tokens, patch_size]
        x = self.patch_norm(x)
        x = self.proj(x)                                        # [B, n_tokens, hidden]
        x = x + self.pos
        x = self.attn(x)                                        # 块间自注意力
        return self.norm(x)
