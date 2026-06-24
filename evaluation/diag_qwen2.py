"""诊断2:先 import unsloth(模拟 eval 环境),再看 AutoProcessor 产出是否一致。不加载大模型。"""
import os, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
from PIL import Image

# A:import unsloth 之前的干净处理器
from transformers import AutoProcessor
adapter = "output/runs/e1_qwen36_27b_viridis_3ep/checkpoint-600"
proc_clean = AutoProcessor.from_pretrained(adapter)

# 触发 unsloth 全局 patch(不加载模型)
import unsloth  # noqa
proc_after = AutoProcessor.from_pretrained(adapter)

# 取一个样本
with open("output/training_data/e1/test.jsonl") as f:
    rec = json.loads(f.readline())
msgs = []
for m in rec["messages"]:
    if m["role"] == "assistant":
        continue
    c = m["content"]
    if isinstance(c, list):
        nc = []
        for part in c:
            if part.get("type") == "image":
                nc.append({"type": "image", "image": Image.open(
                    os.path.join("output/spectrograms_viridis", part["image"])).convert("RGB")})
            else:
                nc.append(part)
        msgs.append({"role": m["role"], "content": nc})
    else:
        msgs.append(m)
full = msgs + [{"role": "assistant", "content": '{"detection": "YES"}'}]
images = [p["image"] for mm in full if isinstance(mm["content"], list)
          for p in mm["content"] if isinstance(p, dict) and p.get("type") == "image"]


def show(tag, proc):
    text = proc.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    out = proc(text=[text], images=images, return_tensors="pt")
    ii = out["input_ids"].shape[1]
    am = out["attention_mask"].shape[1]
    mm = out["mm_token_type_ids"].shape[1] if "mm_token_type_ids" in out else "—"
    g = out["image_grid_thw"][0].tolist() if "image_grid_thw" in out else "—"
    gimg = (g[0] * g[1] * g[2] // 4) if isinstance(g, list) else "—"
    print(f"{tag}: input_ids={ii} attn={am} mm_type={mm} grid={g} grid_img_tok={gimg}")


show("import unsloth 前", proc_clean)
show("import unsloth 后", proc_after)
