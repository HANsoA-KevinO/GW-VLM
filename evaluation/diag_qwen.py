"""诊断 Qwen3.6 处理器:看 input_ids / attention_mask / image_grid_thw 是否一致。"""
import os, json, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

adapter = "output/runs/e1_qwen36_27b_viridis_3ep/checkpoint-600"
proc = AutoProcessor.from_pretrained(adapter)

# 取一个测试样本
rec = None
with open("output/training_data/e1/test.jsonl") as f:
    rec = json.loads(f.readline())
img_rel = None
msgs = []
for m in rec["messages"]:
    if m["role"] == "assistant":
        continue
    c = m["content"]
    if isinstance(c, list):
        nc = []
        for part in c:
            if part.get("type") == "image":
                img_rel = part["image"]
                p = os.path.join("output/spectrograms_viridis", img_rel)
                nc.append({"type": "image", "image": Image.open(p).convert("RGB")})
            else:
                nc.append(part)
        msgs.append({"role": m["role"], "content": nc})
    else:
        msgs.append(m)

full = msgs + [{"role": "assistant", "content": '{"detection": "YES"}'}]
images = [part["image"] for mm in full if isinstance(mm["content"], list)
          for part in mm["content"] if isinstance(part, dict) and part.get("type") == "image"]

# 路径 A:两步法
text = proc.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
A = proc(text=[text], images=images or None, return_tensors="pt")
print("=== 两步法 processor(text, images) ===")
for k, v in A.items():
    print(f"  {k}: shape={tuple(v.shape)}")
ipad = getattr(proc, "image_token_id", None) or getattr(getattr(proc, "tokenizer", None), "image_token_id", None)
try:
    img_tok = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    n_pad = int((A["input_ids"][0] == img_tok).sum())
    print(f"  <|image_pad|> id={img_tok} 计数={n_pad}")
except Exception as e:
    print("  image_pad 计数失败:", e)
if "image_grid_thw" in A:
    g = A["image_grid_thw"][0].tolist()
    print(f"  image_grid_thw={g} prod={g[0]*g[1]*g[2]} /merge^2(4)={g[0]*g[1]*g[2]//4}")
print(f"  input_ids==attention_mask 长度? {A['input_ids'].shape[1]} vs {A['attention_mask'].shape[1]}")

# 路径 B:一步法 apply_chat_template(tokenize=True)
B = proc.apply_chat_template(full, tokenize=True, add_generation_prompt=False,
                             return_dict=True, return_tensors="pt")
print("=== 一步法 apply_chat_template(tokenize=True) ===")
for k, v in B.items():
    try:
        print(f"  {k}: shape={tuple(v.shape)}")
    except Exception:
        pass
