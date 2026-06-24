"""
概率版 E1 检测评估：抽取每个样本的 P(YES)，算 ROC-AUC / PR-AUC + 阈值扫描曲线。

与 evaluate.py(贪心解码取 YES/NO)不同：这里对每个样本做**一次前向**，把输入喂到
assistant 答案的 `{"detection": "` 处，看**下一个 token 是 YES 还是 NO 的相对概率** →
得到连续分数 P(YES)。有了分数就能算阈值无关的 ROC-AUC、PR-AUC，以及 recall/precision
随阈值变化的曲线、不同工作点(默认0.5 / 低FPR)的指标。

用法（Spark）：
  python evaluation/evaluate_prob.py --adapter output/runs/e1_gemma4_31b_v2 \
      --no-4bit --image-tokens 560
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_test(path: Path, image_root: str, max_samples=None) -> list:
    """读 messages jsonl → [(prompt_messages 含 PIL 图, gold 0/1)]。"""
    from PIL import Image
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            rec = json.loads(line)
            gold, prompt_msgs, img_rel = None, [], None
            for m in rec["messages"]:
                if m["role"] == "assistant":
                    try:
                        gold = 1 if json.loads(m["content"]).get("detection") == "YES" else 0
                    except Exception:
                        gold = None
                    continue
                content = m["content"]
                if isinstance(content, list):
                    nc = []
                    for part in content:
                        if part.get("type") == "image":
                            img_rel = part["image"]  # 记下相对图名，用于和 SNR 关联
                            p = img_rel
                            if not os.path.isabs(p):
                                p = os.path.join(image_root, p)
                            nc.append({"type": "image", "image": Image.open(p).convert("RGB")})
                        else:
                            nc.append(part)
                    prompt_msgs.append({"role": m["role"], "content": nc})
                else:
                    prompt_msgs.append(m)
            items.append((prompt_msgs, gold, img_rel))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="概率版 E1 评估(ROC/PR/阈值)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", type=Path, default=Path("output/training_data/e1/test.jsonl"))
    ap.add_argument("--image-root", default="output/spectrograms")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--image-tokens", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--no-unsloth", action="store_true",
                    help="用原生 transformers+PEFT 加载(绕开 Unsloth 对 Qwen3.6 推理 forward 的 rope bug)")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor
    genproc = AutoProcessor.from_pretrained(args.adapter)

    if args.no_unsloth:
        # 原生 transformers + PEFT:不导入 unsloth → 模型 forward 无全局 patch,
        # get_rope_index 收到与处理器一致的输入(Qwen3.6 推理走这条)。
        from transformers import AutoModelForImageTextToText
        from peft import PeftModel
        cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
        base_id = cfg["base_model_name_or_path"]
        print(f"[prob] 原生加载 base={base_id} + PEFT adapter", flush=True)
        base = AutoModelForImageTextToText.from_pretrained(
            base_id, dtype=torch.bfloat16, device_map={"": 0})
        model = PeftModel.from_pretrained(base, args.adapter)
        model.eval()
    else:
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(args.adapter, load_in_4bit=not args.no_4bit)
        FastVisionModel.for_inference(model)
        # image_tokens 是 Gemma 专属(token 预算→分辨率);仅当处理器为 Gemma 式(有 max_soft_tokens)才覆盖。
        if args.image_tokens is not None:
            gip = getattr(genproc, "image_processor", None)
            if gip is not None and hasattr(gip, "max_soft_tokens"):
                gip.max_soft_tokens = args.image_tokens
                gip.image_seq_length = args.image_tokens
                if hasattr(genproc, "image_seq_length"):
                    genproc.image_seq_length = args.image_tokens

    tok = genproc.tokenizer
    # 找 `{"detection": "YES"}` 与 `..."NO"}` 第一处分叉的 token → 即 YES/NO 决策位
    yes_full = tok.encode('{"detection": "YES"}', add_special_tokens=False)
    no_full = tok.encode('{"detection": "NO"}', add_special_tokens=False)
    d = 0
    while d < min(len(yes_full), len(no_full)) and yes_full[d] == no_full[d]:
        d += 1
    prefix_ids = yes_full[:d]
    yes_id, no_id = yes_full[d], no_full[d]
    print(f"[prob] 共享前缀解码={tok.decode(prefix_ids)!r}  yes_tok={tok.decode([yes_id])!r} no_tok={tok.decode([no_id])!r}")

    items = load_test(args.test, args.image_root, args.max_samples)
    device = model.device
    y_true, y_score, imgs = [], [], []

    def build(msgs, ans):
        """原始 AutoProcessor 两步法(渲文本→展开图像),产出自洽输入(image_pad 数 = grid)。"""
        full = msgs + [{"role": "assistant", "content": ans}]
        text = genproc.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        images = [part["image"] for m in full if isinstance(m.get("content"), list)
                  for part in m["content"] if isinstance(part, dict) and part.get("type") == "image"]
        return genproc(text=[text], images=images or None, return_tensors="pt").to(device)

    # teacher-forced 单次前向取 P(YES):对每个样本建 YES/NO 两条完整序列,找首处分叉位 dp
    # (=决策 token 位置,自动适配各模型 BPE 上下文),前向 YES 序列读 logits[dp-1] → 预测该位。
    for k, (msgs, gold, img_rel) in enumerate(items):
        imgs.append(img_rel)
        iy = build(msgs, '{"detection": "YES"}')
        ino = build(msgs, '{"detection": "NO"}')
        a = iy["input_ids"][0].tolist()
        b = ino["input_ids"][0].tolist()
        dp = 0
        while dp < min(len(a), len(b)) and a[dp] == b[dp]:
            dp += 1
        yid, nid = a[dp], b[dp]  # YES 分支 / NO 分支在分叉位的 token
        with torch.no_grad():
            out = model(**iy)  # 原始处理器输入自洽,mm_token_type_ids 与 input_ids 等长
        lg = out.logits[0, dp - 1].float()  # 预测分叉位的 logits(在共享前缀内,YES/NO 等价)
        pair = torch.softmax(torch.stack([lg[yid], lg[nid]]), dim=0)
        y_true.append(gold)
        y_score.append(float(pair[0]))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(items)} ...", flush=True)

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
                    precision=round(prec, 4), recall=round(rec, 4),
                    fpr=round(fpr, 4), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

    # 工作点：默认0.5；最大化F1；以及几个低FPR点(近似FAR)
    fpr_c, tpr_c, thr_c = roc_curve(yt, ys)
    op = {"default_0.5": metrics_at(0.5)}
    # 最大 F1 阈值
    prec_c, rec_c, thr_pr = precision_recall_curve(yt, ys)
    f1s = [2 * p * r / (p + r) if (p + r) else 0 for p, r in zip(prec_c, rec_c)]
    best_i = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    op["max_f1"] = metrics_at(thr_pr[best_i] if best_i < len(thr_pr) else 0.5)
    # 低 FPR 工作点(误报率 ≤ 5% / 10%)→ 近似 FAR
    for target_fpr in (0.05, 0.10):
        idx = np.where(fpr_c <= target_fpr)[0]
        thr = thr_c[idx[-1]] if len(idx) else 1.0
        op[f"fpr<={target_fpr}"] = metrics_at(thr)

    report = {"n": len(yt), "roc_auc": round(roc_auc, 4), "pr_auc": round(pr_auc, 4),
              "operating_points": op}
    out_dir = Path(args.out_dir or args.adapter)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prob_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    # 每样本分数(供 SNR 诊断关联)
    with open(out_dir / "per_sample.jsonl", "w") as f:
        for img, g, sc in zip(imgs, y_true, y_score):
            f.write(json.dumps({"image": img, "gold": int(g), "p_yes": float(sc)}) + "\n")

    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr_c, tpr_c, color="tab:blue", lw=1.8, label=f"ROC (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    axes[0].set_xlabel("FPR (false alarm rate)"); axes[0].set_ylabel("TPR (recall)")
    axes[0].set_title("ROC curve"); axes[0].legend(); axes[0].grid(alpha=0.3)
    ths = np.linspace(0, 1, 101)
    axes[1].plot(ths, [metrics_at(t)["recall"] for t in ths], color="tab:green", label="recall")
    axes[1].plot(ths, [metrics_at(t)["precision"] for t in ths], color="tab:red", label="precision")
    axes[1].axvline(0.5, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("threshold"); axes[1].set_ylabel("score")
    axes[1].set_title("recall / precision vs threshold"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "roc_threshold.png", dpi=120)

    print("\n=== 概率版指标 ===")
    print(f"  ROC-AUC = {roc_auc:.4f}   PR-AUC = {pr_auc:.4f}   n={len(yt)}")
    for name, m in op.items():
        print(f"  [{name}] thr={m['threshold']} acc={m['accuracy']} P={m['precision']} R={m['recall']} FPR={m['fpr']} (tp{m['tp']}/fp{m['fp']}/fn{m['fn']}/tn{m['tn']})")
    print(f"写入 {out_dir}/prob_report.json + roc_threshold.png")


if __name__ == "__main__":
    main()
