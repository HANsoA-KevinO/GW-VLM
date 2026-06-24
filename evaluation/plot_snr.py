"""
诊断 A：把每样本 P(YES) 和该事件的 SNR 关联，看"漏检是不是集中在低 SNR 弱信号"。
- 左图：P(YES) vs SNR 散点(仅正样本/真实事件)。若 P 随 SNR 升高 → 模型置信度跟着信号强度走 → 弱信号是瓶颈。
- 右图：按 SNR 分箱的 recall(在给定阈值下)。

用法：
  python evaluation/plot_snr.py --per-sample output/runs/e1_gemma4_31b_v2/per_sample.jsonl \
      --dataset output/dataset_test.jsonl --threshold 0.18
"""
import argparse
import json
from pathlib import Path


def basename(p: str) -> str:
    return p.rsplit("/", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample", required=True, type=Path)
    ap.add_argument("--dataset", type=Path, default=Path("output/dataset_test.jsonl"))
    ap.add_argument("--threshold", type=float, default=0.18)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    snr_by = {}
    for line in open(args.dataset):
        r = json.loads(line)
        snr_by[basename(r["image_path"])] = r.get("metadata", {}).get("snr")

    pos, neg = [], []
    for line in open(args.per_sample):
        r = json.loads(line)
        snr = snr_by.get(basename(r["image"]))
        if r["gold"] == 1 and snr is not None:
            pos.append((float(snr), r["p_yes"]))
        elif r["gold"] == 0 and snr is not None:
            neg.append((float(snr), r["p_yes"]))

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if pos:
        ps, pp = zip(*pos)
        axes[0].scatter(ps, pp, c="tab:red", s=20, alpha=0.6, label="positive (real GW)")
    axes[0].axhline(args.threshold, color="gray", ls="--", lw=1, label=f"threshold={args.threshold}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("event SNR (log)"); axes[0].set_ylabel("P(YES)")
    axes[0].set_title("P(YES) vs SNR  (positives)"); axes[0].legend(); axes[0].grid(alpha=0.3)

    bins = [0, 8, 12, 20, 1e9]
    labels = ["<8", "8-12", "12-20", "20+"]
    rec = []
    for i in range(len(bins) - 1):
        sub = [p for s, p in pos if bins[i] <= s < bins[i + 1]]
        r = (sum(1 for p in sub if p >= args.threshold) / len(sub)) if sub else float("nan")
        rec.append((r, len(sub)))
    axes[1].bar(labels, [r if r == r else 0 for r, _ in rec], color="tab:green")
    for i, (r, n) in enumerate(rec):
        axes[1].text(i, (r if r == r else 0) + 0.02, f"n={n}", ha="center")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("SNR bin"); axes[1].set_ylabel(f"recall @ thr={args.threshold}")
    axes[1].set_title("recall vs SNR bin  (positives)"); axes[1].grid(alpha=0.3, axis="y")

    out = Path(args.out) if args.out else args.per_sample.parent / "snr_diagnosis.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)

    print(f"positives={len(pos)}  (negatives 不计：其 SNR 是事件SNR、对噪声窗无意义)")
    print(f"recall by SNR bin (thr={args.threshold}):")
    for l, (r, n) in zip(labels, rec):
        print(f"  {l:6}: recall={r:.2f} (n={n})" if r == r else f"  {l:6}: n=0")
    if pos:
        ps, pp = zip(*pos)
        c = float(np.corrcoef(np.log(np.array(ps)), np.array(pp))[0, 1])
        print(f"corr(log SNR, P(YES)) = {c:.3f}  (越接近1=置信度越跟着信号强度走)")
    print("saved", out)


if __name__ == "__main__":
    main()
