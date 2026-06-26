"""
FusionCollator:把 (messages + 白化应变) 批成 FusionVLM 的输入。
- 渲染 prompt / full 两次得到 labels(只在 assistant 答案上算损失);
- 在序列【最前】预置 N 个文本占位 token(占位=strain),前向时被换成应变编码输出;
- 右 padding;pixel_values / image_grid_thw 按样本拼接(Qwen 风格)。
"""
import numpy as np
import torch


def render(processor, messages, images, add_gen, enable_thinking):
    kw = dict(tokenize=False, add_generation_prompt=add_gen)
    try:
        text = processor.apply_chat_template(messages, enable_thinking=enable_thinking, **kw)
    except TypeError:
        text = processor.apply_chat_template(messages, **kw)  # 模板不支持 thinking 参数
    return processor(text=[text], images=images or None, return_tensors="pt")


class FusionCollator:
    def __init__(self, processor, n_strain_tokens, image_token_id,
                 pad_id, enable_thinking=False):
        self.p = processor
        self.n = n_strain_tokens
        self.img_tok = image_token_id
        self.pad = pad_id
        self.think = enable_thinking

    def one(self, ex):
        msgs = ex["messages"]            # 完整对话(含 assistant)
        prompt_msgs = msgs[:-1]          # 去掉 assistant
        imgs = ex.get("images")          # [PIL] 或 None
        full = render(self.p, msgs, imgs, add_gen=False, enable_thinking=self.think)
        prm = render(self.p, prompt_msgs, imgs, add_gen=True, enable_thinking=self.think)
        ids = full["input_ids"][0]
        plen = prm["input_ids"].shape[1]
        labels = ids.clone()
        labels[:plen] = -100             # 只在答案上算损失

        # 预置 N 个占位(strain)
        n = self.n if ex.get("use_strain") else 0
        if n:
            place = torch.full((n,), self.pad, dtype=ids.dtype)
            ids = torch.cat([place, ids])
            labels = torch.cat([torch.full((n,), -100, dtype=labels.dtype), labels])
        smask = torch.zeros(len(ids), dtype=torch.bool)
        smask[:n] = True
        return {
            "input_ids": ids, "labels": labels, "strain_mask": smask,
            "pixel_values": full.get("pixel_values"),
            "image_grid_thw": full.get("image_grid_thw"),
            "strain": ex.get("strain"),
        }

    def __call__(self, batch):
        rows = [self.one(ex) for ex in batch]
        L = max(len(r["input_ids"]) for r in rows)
        B = len(rows)
        input_ids = torch.full((B, L), self.pad, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        smask = torch.zeros((B, L), dtype=torch.bool)
        for i, r in enumerate(rows):
            n = len(r["input_ids"])
            input_ids[i, :n] = r["input_ids"]
            labels[i, :n] = r["labels"]
            attn[i, :n] = 1
            smask[i, :n] = r["strain_mask"]
        out = {"input_ids": input_ids, "attention_mask": attn,
               "labels": labels, "strain_mask": smask}
        pv = [r["pixel_values"] for r in rows if r["pixel_values"] is not None]
        if pv:
            out["pixel_values"] = torch.cat(pv, dim=0)
            out["image_grid_thw"] = torch.cat(
                [r["image_grid_thw"] for r in rows if r["image_grid_thw"] is not None], dim=0)
        st = [r["strain"] for r in rows if r["strain"] is not None]
        if st:
            out["strain"] = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in st])
        return out
