"""
FusionVLM:把 1D 应变编码器的输出作为额外 token 融进 Qwen2.5-VL(方法2)。

策略(绕开 Qwen M-RoPE 坑):
- collator 在序列里预置 N 个【文本占位 token】(strain_mask 标其位置),图像 token 不动;
- forward 里:文本 embed → 散射图像特征 → 把占位位置的 embedding 换成应变编码输出;
- 自己用 base.model.get_rope_index 算好 position_ids 再传入(传了就不会触发内部 rope,
  且占位是文本类型 → 正常 1D 位置;图像 token 数与 grid 一致 → 不错位)。
"""
import torch
import torch.nn as nn


class FusionVLM(nn.Module):
    def __init__(self, vlm, strain_encoder, image_token_id: int):
        super().__init__()
        self.vlm = vlm                       # PEFT 包好的 Qwen2.5-VL
        self.strain_encoder = strain_encoder  # None 表示不用应变(消融)
        self.image_token_id = image_token_id

    def _base(self):
        m = self.vlm
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def forward(self, input_ids, attention_mask, labels=None,
                pixel_values=None, image_grid_thw=None,
                strain=None, strain_mask=None, **_):
        base = self._base()
        emb = base.get_input_embeddings()(input_ids)            # [B,S,H]

        # 1) 散射图像特征到 image-token 位置
        # 注意:get_image_features 返回 BaseModelOutputWithPooling,合并后(LLM维)的图像
        # embed 在 .pooler_output(每图一个 [m_i, hidden] 的 tuple),不是 .last_hidden_state(视觉骨干1280维)。
        if pixel_values is not None:
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

        # 3) 自算 M-RoPE position_ids(占位=文本类型;图像 token 与 grid 一致)
        mm_type = (input_ids == self.image_token_id).to(torch.int32)
        pos, _ = base.model.get_rope_index(
            input_ids, mm_type, image_grid_thw, None, None, attention_mask)

        # 4) 走 PEFT 前向(LoRA 生效);传了 position_ids 就不再触发内部 rope
        return self.vlm(inputs_embeds=emb, attention_mask=attention_mask,
                        position_ids=pos, labels=labels, use_cache=False)
