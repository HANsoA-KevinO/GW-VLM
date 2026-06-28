"""
多模态融合(方法2)检测评估:teacher-forced 取每样本 P(YES) → ROC-AUC / PR-AUC / 工作点。
复用 training/ 的 FusionCollator + FusionVLM;按 adapter 目录的 fusion_meta.json 还原消融配置
(use_image/use_strain/strain 参数),加载 PEFT adapter + strain_encoder.pt。

用法(Spark):
  python evaluation/evaluate_fusion.py --adapter output/runs/fusion_B \
      --image-root output/spectrograms_viridis --strain-root output/strain_arrays
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))


def load_items(test_path, image_root, strain_root, use_image, use_strain, max_samples):
    from PIL import Image
    items = []
    for i, line in enumerate(open(test_path)):
        if max_samples is not None and i >= max_samples:
            break
        rec = json.loads(line)
        gold, prompt, img, img_rel = None, [], None, None
        for m in rec["messages"]:
            if m["role"] == "assistant":
                try:
                    gold = json.loads(m["content"]).get("detection")
                except Exception:
                    gold = None
                continue
            c = m["content"]
            if isinstance(c, list):
                if use_image:
                    nc = []
                    for part in c:
                        if part.get("type") == "image":
                            img_rel = part["image"]
                            p = part["image"] if os.path.isabs(part["image"]) else os.path.join(image_root, part["image"])
                            img = Image.open(p).convert("RGB")
                            nc.append({"type": "image", "image": img})
                        else:
                            nc.append(part)
                    prompt.append({"role": m["role"], "content": nc})
                else:
                    for part in c:
                        if part.get("type") == "image":
                            img_rel = part["image"]
                    prompt.append({"role": m["role"],
                                   "content": [{"type": "text", "text": "Analyze the gravitational-wave data."}]})
            else:
                prompt.append(m)
        strain = None
        if use_strain and img_rel:
            strain = np.load(os.path.join(strain_root, Path(img_rel).stem + ".npy"))
        items.append((prompt, img, strain, gold))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", type=Path, default=ROOT / "output/training_data/e1/test.jsonl")
    ap.add_argument("--image-root", default="output/spectrograms_viridis")
    ap.add_argument("--strain-root", default="output/strain_arrays")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    from models.strain_encoder import StrainEncoder1D, StrainPatchEncoder
    from fusion_model import FusionVLM, repair_misquantized_linears
    from fusion_collator import FusionCollator

    meta = json.loads((Path(args.adapter) / "fusion_meta.json").read_text())
    use_image, use_strain = meta["use_image"], meta["use_strain"]
    family = meta.get("model_family", "qwen")
    enc_type = meta.get("strain_encoder_type", "cnn")
    n_tokens = meta["strain_n_tokens"]
    base_id = json.loads((Path(args.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"[eval-fusion] family={family} enc={enc_type} use_image={use_image} use_strain={use_strain} base={base_id}", flush=True)

    processor = AutoProcessor.from_pretrained(args.adapter)
    ip = getattr(processor, "image_processor", None)
    if family == "qwen" and ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = int(meta.get("max_pixels", 262144))   # 与训练一致
    quant = None
    if meta.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"])
    base = AutoModelForImageTextToText.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=quant)
    if family == "gemma":
        repair_misquantized_linears(base)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    hidden = model.get_base_model().config.text_config.hidden_size
    img_tok = model.get_base_model().config.image_token_id

    strain_enc = None
    if use_strain:
        if enc_type == "patch_attn":
            strain_enc = StrainPatchEncoder(
                hidden, in_len=meta["strain_in_len"], patch_size=int(meta["strain_patch_size"]),
                n_attn_layers=int(meta.get("strain_attn_layers", 3)),
                n_heads=int(meta.get("strain_attn_heads", 8)),
                mlp_proj=bool(meta.get("strain_mlp_proj", True))).to("cuda", torch.float32)
        else:
            strain_enc = StrainEncoder1D(hidden, n_tokens=n_tokens, in_len=meta["strain_in_len"],
                                         channels=tuple(meta["strain_channels"])).to("cuda", torch.float32)
        strain_enc.load_state_dict(torch.load(Path(args.adapter) / "strain_encoder.pt"))
        strain_enc.eval()
    fusion = FusionVLM(model, strain_enc, img_tok, model_family=family)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    collator = FusionCollator(processor, n_tokens, img_tok, pad_id, enable_thinking=False, family=family)

    items = load_items(args.test, args.image_root, args.strain_root, use_image, use_strain, args.max_samples)

    def ex(prompt, img, strain, ans):
        return {"messages": prompt + [{"role": "assistant", "content": ans}],
                "images": [img] if (use_image and img is not None) else None,
                "use_strain": use_strain, "strain": strain}

    y_true, y_score = [], []
    for k, (prompt, img, strain, gold) in enumerate(items):
        by = collator([ex(prompt, img, strain, '{"detection": "YES"}')])
        bn = collator([ex(prompt, img, strain, '{"detection": "NO"}')])
        a = by["input_ids"][0].tolist()
        b = bn["input_ids"][0].tolist()
        dp = 0
        while dp < min(len(a), len(b)) and a[dp] == b[dp]:
            dp += 1
        yid, nid = a[dp], b[dp]
        batch = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in by.items()}
        with torch.no_grad():
            out = fusion(**batch)
        lg = out.logits[0, dp - 1].float()
        pair = torch.softmax(torch.stack([lg[yid], lg[nid]]), dim=0)
        y_true.append(1 if gold == "YES" else 0)
        y_score.append(float(pair[0]))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(items)} ...", flush=True)

    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
    yt, ys = np.array(y_true), np.array(y_score)
    roc_auc = float(roc_auc_score(yt, ys))
    pr_auc = float(average_precision_score(yt, ys))

    def at(thr):
        yp = (ys >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return dict(threshold=round(float(thr), 4), recall=round(tp / (tp + fn), 4) if (tp + fn) else 0,
                    precision=round(tp / (tp + fp), 4) if (tp + fp) else 0,
                    accuracy=round((tp + tn) / len(yt), 4), fpr=round(fp / (fp + tn), 4) if (fp + tn) else 0)
    fpr_c, tpr_c, thr_c = roc_curve(yt, ys)
    op = {"default_0.5": at(0.5)}
    for t in (0.05, 0.10):
        idx = np.where(fpr_c <= t)[0]
        op[f"fpr<={t}"] = at(thr_c[idx[-1]] if len(idx) else 1.0)
    report = {"use_image": use_image, "use_strain": use_strain,
              "model_family": family, "strain_encoder_type": enc_type, "strain_n_tokens": n_tokens,
              "n": len(yt),
              "roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4), "operating_points": op}
    (Path(args.adapter) / "fusion_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== 融合评估 ===")
    print(f"use_image={use_image} use_strain={use_strain}  ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}")
    for kk, v in op.items():
        print(f"  [{kk}] R={v['recall']} P={v['precision']} acc={v['accuracy']} FPR={v['fpr']}")
    print(f"写入 {args.adapter}/fusion_eval.json")


if __name__ == "__main__":
    main()
