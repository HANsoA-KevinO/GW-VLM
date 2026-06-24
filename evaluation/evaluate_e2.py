"""
E2(多任务:检测 + 物理参数 bin)评估。

- 检测:teacher-forced 取 P(YES) → ROC-AUC / PR-AUC / 工作点(同 E1 口径)。
- 参数:贪心生成完整 JSON → 解析 chirp_mass/distance/chi_eff bin → 在**真实正样本**上算
  每字段 bin 准确率(精确 + 邻接±1档,因为 bin 是有序的)+ 三项全对的联合准确率。

均用原生 transformers + PEFT 加载(--no-unsloth 思路),绕开 Qwen3.6 在 Unsloth 下的 rope bug。

用法(Spark):
  python evaluation/evaluate_e2.py --adapter output/runs/e2_qwen36_27b_viridis \
      --image-root output/spectrograms_viridis
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

FIELDS = ["chirp_mass_bin", "distance_bin", "chi_eff_bin"]


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


def strip_thinking(txt):
    """排除 <think>...</think> 包裹的推理,只留答案部分;再清掉残留特殊 token。"""
    if "</think>" in txt:
        txt = txt.rsplit("</think>", 1)[-1]
    for t in ("<think>", "</think>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        txt = txt.replace(t, "")
    return txt.strip()


def parse_json(txt):
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    d = {}
    for key in ["detection"] + FIELDS:
        mm = re.search(r'"%s"\s*:\s*"([^"]*)"' % key, txt)
        if mm:
            d[key] = mm.group(1)
    return d


def binkey(s):
    m = re.match(r"\s*(-?\d+\.?\d*)", str(s))
    return float(m.group(1)) if m else 0.0


def main():
    ap = argparse.ArgumentParser(description="E2 多任务评估(检测 ROC-AUC + 参数 bin 准确率)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", type=Path, default=Path("output/training_data/e2/test.jsonl"))
    ap.add_argument("--image-root", default="output/spectrograms_viridis")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)  # 无思考,JSON ~60 token,128 足够
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    genproc = AutoProcessor.from_pretrained(args.adapter)
    cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
    base_id = cfg["base_model_name_or_path"]
    print(f"[e2] 原生加载 base={base_id} + PEFT adapter", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(base_id, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    device = model.device

    items = load_test(args.test, args.image_root, args.max_samples)

    def build_full(msgs, ans):
        full = msgs + [{"role": "assistant", "content": ans}]
        text = genproc.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        imgs = extract_images(full)
        return genproc(text=[text], images=imgs or None, return_tensors="pt").to(device)

    def build_prompt(msgs):
        # enable_thinking=False 还原训练格式(模型是用纯 JSON、无思考 SFT 的)。
        # 实测:开思考时模型把全部样本判成 NO(off-distribution),关掉才复现训练时的 YES+bin 行为。
        text = genproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        imgs = extract_images(msgs)
        return genproc(text=[text], images=imgs or None, return_tensors="pt").to(device)

    y_true, y_score, rows = [], [], []
    dbg = 0
    gen_tok, gen_t = 0, 0.0  # 统计生成吞吐 tok/s
    for k, (msgs, gold, img_rel) in enumerate(items):
        # 1) 检测 P(YES):teacher-forced 分叉
        iy = build_full(msgs, '{"detection": "YES"}')
        ino = build_full(msgs, '{"detection": "NO"}')
        a = iy["input_ids"][0].tolist()
        b = ino["input_ids"][0].tolist()
        dp = 0
        while dp < min(len(a), len(b)) and a[dp] == b[dp]:
            dp += 1
        yid, nid = a[dp], b[dp]
        with torch.no_grad():
            out = model(**iy)
        lg = out.logits[0, dp - 1].float()
        pair = torch.softmax(torch.stack([lg[yid], lg[nid]]), dim=0)
        p_yes = float(pair[0])

        # 2) 参数:贪心生成完整 JSON
        pred = {}
        gen_txt = ""
        try:
            pin = build_prompt(msgs)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**pin, max_new_tokens=args.max_new_tokens, do_sample=False)
            new = gen[0, pin["input_ids"].shape[1]:]
            gen_t += time.time() - t0
            gen_tok += int(new.shape[0])
            raw = genproc.tokenizer.decode(new, skip_special_tokens=False)  # 保留 <think> 标签以便切分
            gen_txt = strip_thinking(raw)  # 排除思考,只留答案
            pred = parse_json(gen_txt)
        except Exception as e:
            gen_txt = f"<gen_error: {e}>"

        gold_det = 1 if gold.get("detection") == "YES" else 0
        y_true.append(gold_det)
        y_score.append(p_yes)
        rows.append({"image": img_rel, "gold": gold, "pred": pred, "p_yes": p_yes, "gen": gen_txt})
        if gold_det == 1 and dbg < 3:
            dbg += 1
            print(f"[debug pos#{dbg}] gold={json.dumps(gold, ensure_ascii=False)}", flush=True)
            print(f"           pred={json.dumps(pred, ensure_ascii=False)}", flush=True)
            print(f"           gen={gen_txt[:140]!r}", flush=True)
        if (k + 1) % 50 == 0:
            tps = gen_tok / gen_t if gen_t else 0
            print(f"  {k + 1}/{len(items)} ... 生成吞吐 {tps:.1f} tok/s ({gen_tok}tok/{gen_t:.0f}s)", flush=True)

    # ---- 检测指标 ----
    import numpy as np
    from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                                 precision_recall_curve, confusion_matrix)
    yt = np.array(y_true)
    ys = np.array(y_score)
    roc_auc = float(roc_auc_score(yt, ys))
    pr_auc = float(average_precision_score(yt, ys))

    def metrics_at(thr):
        yp = (ys >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        acc = (tp + tn) / len(yt)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return dict(threshold=round(float(thr), 4), accuracy=round(acc, 4),
                    precision=round(prec, 4), recall=round(rec, 4), fpr=round(fpr, 4),
                    tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

    fpr_c, tpr_c, thr_c = roc_curve(yt, ys)
    op = {"default_0.5": metrics_at(0.5)}
    prec_c, rec_c, thr_pr = precision_recall_curve(yt, ys)
    f1s = [2 * p * r / (p + r) if (p + r) else 0 for p, r in zip(prec_c, rec_c)]
    best_i = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    op["max_f1"] = metrics_at(thr_pr[best_i] if best_i < len(thr_pr) else 0.5)
    for tgt in (0.05, 0.10):
        idx = np.where(fpr_c <= tgt)[0]
        op[f"fpr<={tgt}"] = metrics_at(thr_c[idx[-1]] if len(idx) else 1.0)

    # ---- 参数指标(在真实正样本上)----
    pos = [r for r in rows if r["gold"].get("detection") == "YES"]
    npos = len(pos)
    params = {}
    for f in FIELDS:
        order = sorted({r["gold"][f] for r in pos if r["gold"].get(f) not in (None, "N/A")}, key=binkey)
        idxmap = {b: i for i, b in enumerate(order)}
        exact = sum(1 for r in pos if r["pred"].get(f) == r["gold"].get(f))
        adj = 0
        for r in pos:
            g, p = r["gold"].get(f), r["pred"].get(f)
            if g in idxmap and p in idxmap and abs(idxmap[g] - idxmap[p]) <= 1:
                adj += 1
        params[f] = {"exact_acc": round(exact / npos, 4) if npos else 0.0,
                     "adjacent_pm1_acc": round(adj / npos, 4) if npos else 0.0,
                     "n_bins": len(order), "chance": round(1.0 / max(1, len(order)), 4)}
    joint = sum(1 for r in pos if all(r["pred"].get(f) == r["gold"].get(f) for f in FIELDS))
    # 负样本上模型是否正确输出 N/A(次要)
    neg = [r for r in rows if r["gold"].get("detection") == "NO"]
    neg_na_ok = sum(1 for r in neg if all(r["pred"].get(f) in ("N/A", None) for f in FIELDS))

    report = {
        "n": len(yt), "n_pos": npos,
        "detection": {"roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4), "operating_points": op},
        "params_on_true_positives": params,
        "params_joint_all3_exact_acc": round(joint / npos, 4) if npos else 0.0,
        "neg_correctly_NA_rate": round(neg_na_ok / len(neg), 4) if neg else None,
    }
    out_dir = Path(args.out_dir or args.adapter)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with open(out_dir / "e2_per_sample.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== E2 评估 ===")
    print(f"检测: ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}  (0.5: R={op['default_0.5']['recall']} P={op['default_0.5']['precision']})")
    print(f"参数(真实正样本 n={npos}):")
    for f in FIELDS:
        pm = params[f]
        print(f"  {f}: 精确={pm['exact_acc']} 邻接±1={pm['adjacent_pm1_acc']} (随机={pm['chance']}, {pm['n_bins']}档)")
    print(f"  三项全对联合精确={report['params_joint_all3_exact_acc']}")
    print(f"  负样本正确输出N/A率={report['neg_correctly_NA_rate']}")
    print(f"写入 {out_dir}/e2_report.json + e2_per_sample.jsonl")


if __name__ == "__main__":
    main()
