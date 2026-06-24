"""
思考诊断:对 E2 测试集样本带【思考】生成,完整保存 <think> 原文 + 最终判定,
用于分析"为什么开思考后模型把正样本推理成 NO"。

逐样本即时追加写出(可中途 tail 分析)。默认只跑真实正样本(失败发生处);
--include-neg 也跑若干负样本做对照。

用法(Spark):
  python evaluation/diag_e2_thinking.py --adapter output/runs/e2_qwen36_27b_viridis \
      --image-root output/spectrograms_viridis --out ~/think_diag.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_test(path, image_root, max_samples=None):
    from PIL import Image
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rec = json.loads(line)
            gold, prompt, img_rel = None, [], None
            for m in rec["messages"]:
                if m["role"] == "assistant":
                    try:
                        gold = json.loads(m["content"])
                    except Exception:
                        gold = {}
                    continue
                c = m["content"]
                if isinstance(c, list):
                    nc = []
                    for part in c:
                        if part.get("type") == "image":
                            img_rel = part["image"]
                            p = part["image"]
                            if not os.path.isabs(p):
                                p = os.path.join(image_root, p)
                            nc.append({"type": "image", "image": Image.open(p).convert("RGB")})
                        else:
                            nc.append(part)
                    prompt.append({"role": m["role"], "content": nc})
                else:
                    prompt.append(m)
            items.append((prompt, gold, img_rel))
    return items


def extract_images(msgs):
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image":
                    out.append(part["image"])
    return out


def split_think(raw):
    """返回 (thinking原文, 答案部分)。"""
    think, ans = "", raw
    if "<think>" in raw and "</think>" in raw:
        think = raw.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        ans = raw.rsplit("</think>", 1)[-1]
    for t in ("<think>", "</think>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        ans = ans.replace(t, "")
    return think.strip(), ans.strip()


def final_det(ans):
    m = re.search(r'"detection"\s*:\s*"([^"]*)"', ans)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", type=Path, default=Path("output/training_data/e2/test.jsonl"))
    ap.add_argument("--image-root", default="output/spectrograms_viridis")
    ap.add_argument("--out", default=os.path.expanduser("~/think_diag.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-pos", type=int, default=135, help="最多跑多少真实正样本")
    ap.add_argument("--include-neg", type=int, default=15, help="另跑多少负样本做对照")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    genproc = AutoProcessor.from_pretrained(args.adapter)
    cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
    base_id = cfg["base_model_name_or_path"]
    print(f"[diag] base={base_id} + PEFT", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(base_id, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    device = model.device

    items = load_test(args.test, args.image_root)
    pos = [it for it in items if it[1].get("detection") == "YES"][:args.max_pos]
    neg = [it for it in items if it[1].get("detection") == "NO"][:args.include_neg]
    todo = [("pos", it) for it in pos] + [("neg", it) for it in neg]
    print(f"[diag] 跑 正样本{len(pos)} + 负样本{len(neg)} = {len(todo)} 条(带思考)", flush=True)

    out_f = open(args.out, "w")
    n_pos_no = 0
    t_start = time.time()
    for k, (kind, (msgs, gold, img_rel)) in enumerate(todo):
        text = genproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # 默认开思考
        imgs = extract_images(msgs)
        pin = genproc(text=[text], images=imgs or None, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**pin, max_new_tokens=args.max_new_tokens, do_sample=False)
        new = gen[0, pin["input_ids"].shape[1]:]
        raw = genproc.tokenizer.decode(new, skip_special_tokens=False)
        think, ans = split_think(raw)
        det = final_det(ans)
        if kind == "pos" and det != "YES":
            n_pos_no += 1
        rec = {"kind": kind, "image": img_rel, "gold": gold, "final_detection": det,
               "n_think_tokens": int(new.shape[0]), "thinking": think, "answer": ans}
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        if (k + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"  {k + 1}/{len(todo)}  正样本误判NO={n_pos_no}  用时{el/60:.1f}min", flush=True)
    out_f.close()
    print(f"[diag] 完成。正样本中思考后判NO的: {n_pos_no}/{len(pos)}  → 写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
