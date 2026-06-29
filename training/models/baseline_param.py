"""经典基线:纯 1D 应变 → 参数后验(无 VLM)。

作用 = 统一后验头的**对照标尺**:量"基础模型(27B 融合表示)到底有没有
追平/超过一个只吃应变的经典小网"。同样的标准化目标 + 同样的 GaussianPosteriorHead,
只是特征来自一个小应变编码器(均值池化),而非 27B 的 hidden。
"""
import torch.nn as nn

from models.strain_encoder import StrainPatchEncoder, StrainEncoder1D
from models.posterior_head import GaussianPosteriorHead, N_PARAMS


class StrainParamBaseline(nn.Module):
    def __init__(self, hidden: int = 256, enc_type: str = "patch_attn", in_len: int = 8192,
                 patch_size: int = 256, n_attn_layers: int = 3, n_heads: int = 8,
                 mlp_hidden: int = 256, head_dropout: float = 0.1):
        super().__init__()
        if enc_type == "patch_attn":
            self.enc = StrainPatchEncoder(hidden, in_len=in_len, patch_size=patch_size,
                                          n_attn_layers=n_attn_layers, n_heads=n_heads)
        else:
            self.enc = StrainEncoder1D(hidden, in_len=in_len)
        self.head = GaussianPosteriorHead(hidden, n_params=N_PARAMS, mlp_hidden=mlp_hidden,
                                          dropout=head_dropout)

    def forward(self, strain):
        """strain: [B, in_len] → (mu, logstd): [B, N_PARAMS]"""
        feat = self.enc(strain).mean(dim=1)   # [B,N,H] → [B,H](均值池化)
        return self.head(feat)
