"""
统一(E5)评估:检测(teacher-forced P(YES))+ 后验头参数。
参数指标(真实正样本):NLL(标准化空间)、MAE/中位分数误差(物理)、bin 等价精度(对照 E2)、
PIT/覆盖率(校准:50%/90% 可信区间命中率应≈0.5/0.9)。

用法:
  python evaluation/evaluate_unified.py --adapter output/runs/unified_q27 \
      --test output/training_data/e5/test.jsonl \
      --image-root output/spectrograms_viridis --strain-root output/strain_arrays
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "data_pipeline"))

SQRT2 = math.sqrt(2.0)


def load_items(test_path, image_root, strain_root, use_image, use_strain, param_names, max_samples):
    from PIL import Image
    items = []
    for i, line in enumerate(open(test_path)):
        if max_samples is not None and i >= max_samples:
            break
        rec = json.loads(line)
        gold, prompt, img = None, [], None
        for m in rec["messages"]:
            if m["role"] == "assistant":
                try:
                    gold = json.loads(m["content"]).get("detection")
                except Exception:
                    gold = None
                continue
            c = m["content"]
            if isinstance(c, list) and use_image:
                nc = []
                for part in c:
                    if part.get("type") == "image":
                        p = part["image"] if os.path.isabs(part["image"]) else os.path.join(image_root, part["image"])
                        img = Image.open(p).convert("RGB")
                        nc.append({"type": "image", "image": img})
                    else:
                        nc.append(part)
                prompt.append({"role": m["role"], "content": nc})
            elif isinstance(c, list):
                prompt.append({"role": m["role"],
                               "content": [{"type": "text", "text": "Analyze the gravitational-wave data."}]})
            else:
                prompt.append(m)
        base = rec.get("basename")
        strain = np.load(os.path.join(strain_root, base + ".npy")) if (use_strain and base) else None
        t = rec.get("targets") or {}
        is_pos = all(t.get(n) is not None for n in param_names)
        true_phys = [float(t[n]) for n in param_names] if is_pos else None
        items.append((prompt, img, strain, gold, true_phys))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", type=Path, default=ROOT / "output/training_data/e5/test.jsonl")
    ap.add_argument("--image-root", default="output/spectrograms_viridis")
    ap.add_argument("--strain-root", default="output/strain_arrays")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    from models.strain_encoder import StrainEncoder1D, StrainPatchEncoder
    from models.posterior_head import GaussianPosteriorHead, PARAM_NAMES, standardize, invert
    from fusion_model import FusionVLM, repair_misquantized_linears
    from fusion_collator import FusionCollator
    from config import CHIRP_MASS_BINS, DISTANCE_BINS, CHI_EFF_BINS, assign_bin

    BINS = {"chirp_mass": CHIRP_MASS_BINS, "distance": DISTANCE_BINS, "chi_eff": CHI_EFF_BINS}

    adp = Path(args.adapter)
    meta = json.loads((adp / "fusion_meta.json").read_text())
    stats = json.loads((adp / "norm_stats.json").read_text())
    use_image, use_strain = meta["use_image"], meta["use_strain"]
    family = meta.get("model_family", "qwen")
    enc_type = meta.get("strain_encoder_type", "patch_attn")
    n_tokens = meta["strain_n_tokens"]
    base_id = json.loads((adp / "adapter_config.json").read_text())["base_model_name_or_path"]
    print(f"[eval-unified] family={family} enc={enc_type} use_image={use_image} use_strain={use_strain} base={base_id}", flush=True)

    processor = AutoProcessor.from_pretrained(adp)
    ip = getattr(processor, "image_processor", None)
    if family == "qwen" and ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = int(meta.get("max_pixels", 262144))
    quant = None
    if meta.get("load_in_4bit"):
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   llm_int8_skip_modules=["visual", "lm_head"])
    base = AutoModelForImageTextToText.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=quant)
    if family == "gemma":
        repair_misquantized_linears(base)
    model = PeftModel.from_pretrained(base, adp)
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
        strain_enc.load_state_dict(torch.load(adp / "strain_encoder.pt"))
        strain_enc.eval()
    head = GaussianPosteriorHead(hidden, n_params=len(PARAM_NAMES),
                                 mlp_hidden=int(meta.get("head_mlp_hidden", 256))).to("cuda", torch.float32)
    head.load_state_dict(torch.load(adp / "posterior_head.pt"))
    head.eval()
    fusion = FusionVLM(model, strain_enc, img_tok, model_family=family, param_head=head)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    collator = FusionCollator(processor, n_tokens, img_tok, pad_id, enable_thinking=False, family=family)
    items = load_items(args.test, args.image_root, args.strain_root, use_image, use_strain, PARAM_NAMES, args.max_samples)

    def ex(prompt, img, strain, ans):
        return {"messages": prompt + [{"role": "assistant", "content": ans}],
                "images": [img] if (use_image and img is not None) else None,
                "use_strain": use_strain, "strain": strain,
                "param_target": np.zeros(len(PARAM_NAMES), dtype="float32"),  # 占位:让 collator 出 param_pos
                "param_valid": True}

    y_true, y_score = [], []
    z_mu_l, z_ls_l, true_phys_l = [], [], []
    for k, (prompt, img, strain, gold, true_phys) in enumerate(items):
        by = collator([ex(prompt, img, strain, '{"detection": "YES"}')])
        bn = collator([ex(prompt, img, strain, '{"detection": "NO"}')])
        a, bb = by["input_ids"][0].tolist(), bn["input_ids"][0].tolist()
        dp = 0
        while dp < min(len(a), len(bb)) and a[dp] == bb[dp]:
            dp += 1
        yid, nid = a[dp], bb[dp]
        batch = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in by.items()}
        with torch.no_grad():
            out = fusion(**batch)
        lg = out.logits[0, dp - 1].float()
        pair = torch.softmax(torch.stack([lg[yid], lg[nid]]), dim=0)
        y_true.append(1 if gold == "YES" else 0)
        y_score.append(float(pair[0]))
        if true_phys is not None:
            z_mu_l.append(out.param_mu[0].float().cpu().numpy())
            z_ls_l.append(out.param_logstd[0].float().cpu().numpy())
            true_phys_l.append(true_phys)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(items)} ...", flush=True)

    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
    yt, ys = np.array(y_true), np.array(y_score)
    roc_auc = float(roc_auc_score(yt, ys)); pr_auc = float(average_precision_score(yt, ys))

    def at(thr):
        yp = (ys >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return dict(threshold=round(float(thr), 4), recall=round(tp / (tp + fn), 4) if (tp + fn) else 0,
                    precision=round(tp / (tp + fp), 4) if (tp + fp) else 0,
                    accuracy=round((tp + tn) / len(yt), 4), fpr=round(fp / (fp + tn), 4) if (fp + tn) else 0)
    fpr_c, _, thr_c = roc_curve(yt, ys)
    op = {"default_0.5": at(0.5)}
    for t in (0.05, 0.10):
        idx = np.where(fpr_c <= t)[0]
        op[f"fpr<={t}"] = at(thr_c[idx[-1]] if len(idx) else 1.0)

    # --- 参数指标(真实正样本)---
    params_report = {}
    if true_phys_l:
        zmu = np.array(z_mu_l); zls = np.array(z_ls_l); tp = np.array(true_phys_l)  # [Np,3]
        sigma_z = np.exp(zls)
        true_z = standardize(tp, stats)
        nll = 0.5 * ((true_z - zmu) / sigma_z) ** 2 + np.log(sigma_z) + 0.5 * math.log(2 * math.pi)  # [Np,3]
        median_phys, _, _ = invert(zmu, zls, stats)
        pit = 0.5 * (1.0 + np.vectorize(math.erf)((true_z - zmu) / (sigma_z * SQRT2)))  # [Np,3]
        for j, name in enumerate(PARAM_NAMES):
            bins = BINS[name]
            idxmap = {lab: i for i, (_, _, lab) in enumerate(bins)}
            exact = adj = 0
            for p_pred, p_true in zip(median_phys[:, j], tp[:, j]):
                bp, bt = assign_bin(float(p_pred), bins), assign_bin(float(p_true), bins)
                exact += (bp == bt)
                if bp in idxmap and bt in idxmap and abs(idxmap[bp] - idxmap[bt]) <= 1:
                    adj += 1
            n = len(tp)
            pj = pit[:, j]
            params_report[name] = {
                "nll": round(float(nll[:, j].mean()), 4),
                "mae": round(float(np.mean(np.abs(median_phys[:, j] - tp[:, j]))), 4),
                "median_frac_err": round(float(np.median(np.abs(median_phys[:, j] - tp[:, j]) / np.abs(tp[:, j]))), 4),
                "exact_bin_acc": round(exact / n, 4),
                "adjacent_bin_acc": round(adj / n, 4),
                "coverage_50": round(float(np.mean((pj > 0.25) & (pj < 0.75))), 4),
                "coverage_90": round(float(np.mean((pj > 0.05) & (pj < 0.95))), 4),
                "n_bins": len(bins), "chance": round(1.0 / len(bins), 4),
            }

    report = {"use_image": use_image, "use_strain": use_strain, "model_family": family, "n": len(yt),
              "detection": {"roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4), "operating_points": op},
              "params_on_true_positives": {"n_pos": len(true_phys_l), **params_report}}
    (adp / "unified_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== 统一评估 ===")
    print(f"检测 ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}  R@0.5={op['default_0.5']['recall']} R@FPR5%={op['fpr<=0.05']['recall']}")
    print(f"参数(真实正样本 n={len(true_phys_l)};对照 E2 exact: distance .442 / chirp_mass .239 / chi_eff .265):")
    for name, m in params_report.items():
        print(f"  {name:11s} NLL={m['nll']:.3f} MAE={m['mae']:.3f} bin-exact={m['exact_bin_acc']:.3f}(随机{m['chance']}) "
              f"adj={m['adjacent_bin_acc']:.3f} cov50={m['coverage_50']:.2f} cov90={m['coverage_90']:.2f}")
    print(f"写入 {adp}/unified_eval.json")


if __name__ == "__main__":
    main()
