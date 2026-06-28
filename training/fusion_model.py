"""
FusionVLM:把 1D 应变编码器的输出作为额外 token 融进 VLM(方法2)。支持两家:

- **qwen**(Qwen2.5-VL / Qwen3.6-27B,M-RoPE):图像特征在 get_image_features 的
  `.pooler_output`;须自算 position_ids(get_rope_index)再传入,绕开内部 rope 错位。
- **gemma**(Gemma4-E4B = gemma3n,标准 RoPE):get_image_features 直接返回张量;
  **不传 position_ids**(模型自己按 1D 标准 RoPE 算),图像合并同为 masked_scatter。

通用策略:collator 在序列里预置 N 个【文本占位 token】(strain_mask 标其位置),
forward 里把这些占位位置的 embedding 换成应变编码器输出。
"""
import torch
import torch.nn as nn


def repair_misquantized_linears(model):
    """Unsloth bnb-4bit 检查点里 vision/audio 塔本应 fp(在 llm_int8_skip_modules),
    但 transformers 在加载预量化权重时把它们误包成 Linear4bit(无 quant_state),
    forward 触发 bnb 断言崩。这里把"假 4bit"层(权重未真正打包成 [N,1])换回 nn.Linear。"""
    try:
        import bitsandbytes as bnb
    except Exception:
        return 0
    fixed = 0
    for parent in list(model.modules()):
        for cname, child in list(parent.named_children()):
            if isinstance(child, bnb.nn.Linear4bit):
                w = child.weight
                if not (w.dim() == 2 and w.shape[1] == 1):   # 真 4bit 是 [N,1] 打包;否则是误包的 fp
                    lin = nn.Linear(w.shape[1], w.shape[0], bias=child.bias is not None)
                    lin.weight = nn.Parameter(w.data.to(torch.bfloat16))
                    if child.bias is not None:
                        lin.bias = nn.Parameter(child.bias.data.to(torch.bfloat16))
                    setattr(parent, cname, lin.to(w.device))
                    fixed += 1
    return fixed


class FusionVLM(nn.Module):
    def __init__(self, vlm, strain_encoder, image_token_id: int, model_family: str = "qwen"):
        super().__init__()
        self.vlm = vlm                       # PEFT 包好的 VLM
        self.strain_encoder = strain_encoder  # None 表示不用应变(消融)
        self.image_token_id = image_token_id
        self.model_family = model_family      # "qwen" | "gemma"

    def _base(self):
        m = self.vlm
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def forward(self, input_ids, attention_mask, labels=None,
                pixel_values=None, image_grid_thw=None, image_position_ids=None,
                strain=None, strain_mask=None, **_):
        base = self._base()
        is_gemma = self.model_family == "gemma"
        emb = base.get_input_embeddings()(input_ids)            # [B,S,H]

        # 1) 散射图像特征到 image-token 位置(两家都用 .pooler_output 取合并后的 LLM 维 embed)
        if pixel_values is not None:
            if is_gemma:
                # Gemma4:视觉塔需要 image_position_ids(patch 2D 坐标);合并 embed 在 .pooler_output
                img = base.get_image_features(pixel_values=pixel_values,
                                              image_position_ids=image_position_ids)
            else:
                # Qwen:.pooler_output(每图一个 [m_i,H] 的 tuple),非 .last_hidden_state(视觉骨干 1280 维)
                img = base.get_image_features(pixel_values, image_grid_thw)
            if hasattr(img, "pooler_output"):
                img = img.pooler_output
            if isinstance(img, (list, tuple)):
                img = torch.cat([x for x in img], dim=0)
            img = img.to(emb.dtype).reshape(-1, emb.shape[-1])
            img_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(emb)
            emb = emb.masked_scatter(img_mask, img)

        # 2) 把占位位置换成应变编码输出
        if strain is not None and self.strain_encoder is not None and strain_mask is not None:
            se = self.strain_encoder(strain).to(emb.dtype)      # [B,N,H]
            emb = emb.clone()
            emb[strain_mask] = se.reshape(-1, se.shape[-1])

        # 3) position_ids:Qwen 须自算 M-RoPE;Gemma 标准 RoPE 不传(模型自算)
        if is_gemma:
            return self.vlm(inputs_embeds=emb, attention_mask=attention_mask,
                            labels=labels, use_cache=False)
        mm_type = (input_ids == self.image_token_id).to(torch.int32)
        # 关键字传参,兼容 Qwen2.5-VL(含 second_per_grid_ts)与 Qwen3.5(无该参)两套签名
        pos, _ = base.model.get_rope_index(
            input_ids=input_ids, mm_token_type_ids=mm_type,
            image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        return self.vlm(inputs_embeds=emb, attention_mask=attention_mask,
                        position_ids=pos, labels=labels, use_cache=False)
