"""训练经典基线(strain→参数后验,无 VLM)。只在正样本上训 NLL,早停 val NLL。
用 e5 数据(targets+basename),与统一头同口径(同 norm_stats 逻辑、同 head)。

用法:
  python training/train_baseline_param.py --data-dir output/training_data/e5 \
      --strain-root output/strain_arrays --output-dir output/runs/baseline_param --epochs 60
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_pos(jsonl, strain_root, param_names, stats=None):
    """读正样本:返回 [(strain[8192], z_target[3])]。stats=None 时只收物理值用于算 stats。"""
    from models.posterior_head import standardize
    X, Y, phys = [], [], []
    for line in open(jsonl):
        r = json.loads(line)
        t = r.get("targets") or {}
        if not all(t.get(n) is not None for n in param_names):
            continue
        base = r.get("basename")
        p = np.array([float(t[n]) for n in param_names], dtype=float)
        phys.append(p)
        if stats is not None:
            X.append(np.load(f"{strain_root}/{base}.npy").astype("float32"))
            Y.append(standardize(p[None, :], stats)[0].astype("float32"))
    return X, Y, np.array(phys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="output/training_data/e5")
    ap.add_argument("--strain-root", default="output/strain_arrays")
    ap.add_argument("--output-dir", default="output/runs/baseline_param")
    ap.add_argument("--enc-type", default="patch_attn")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.baseline_param import StrainParamBaseline
    from models.posterior_head import compute_norm_stats, gaussian_nll, PARAM_NAMES

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dd = Path(args.data_dir)
    # norm_stats(train 正样本)
    _, _, train_phys = load_pos(dd / "train.jsonl", args.strain_root, PARAM_NAMES, stats=None)
    stats = compute_norm_stats(train_phys)
    json.dump(stats, open(out / "norm_stats.json", "w"), indent=2)

    Xtr, Ytr, _ = load_pos(dd / "train.jsonl", args.strain_root, PARAM_NAMES, stats=stats)
    Xva, Yva, _ = load_pos(dd / "val.jsonl", args.strain_root, PARAM_NAMES, stats=stats)
    if args.max_samples:
        Xtr, Ytr = Xtr[:args.max_samples], Ytr[:args.max_samples]
    print(f"[baseline] train 正样本 {len(Xtr)}  val {len(Xva)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = DataLoader(TensorDataset(torch.tensor(np.array(Xtr)), torch.tensor(np.array(Ytr))),
                    batch_size=args.batch_size, shuffle=True)
    Xva_t = torch.tensor(np.array(Xva)).to(dev); Yva_t = torch.tensor(np.array(Yva)).to(dev)
    valid_va = torch.ones(len(Xva), dtype=torch.bool, device=dev)

    net = StrainParamBaseline(hidden=args.hidden, enc_type=args.enc_type,
                              patch_size=args.patch_size).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_ep = float("inf"), -1
    for ep in range(args.epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            mu, ls = net(xb)
            loss = gaussian_nll(mu, ls, yb, torch.ones(len(xb), dtype=torch.bool, device=dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            mv, lv = net(Xva_t)
            vnll = gaussian_nll(mv, lv, Yva_t, valid_va).item()
        if vnll < best:
            best, best_ep = vnll, ep
            torch.save(net.state_dict(), out / "baseline.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep} val_nll={vnll:.4f} (best {best:.4f}@{best_ep})", flush=True)

    json.dump({"enc_type": args.enc_type, "hidden": args.hidden, "patch_size": args.patch_size,
               "param_names": PARAM_NAMES, "best_val_nll": best, "best_epoch": best_ep},
              open(out / "baseline_meta.json", "w"), indent=2)
    print(f"[baseline] done → {out} (best val_nll {best:.4f}@{best_ep})", flush=True)


if __name__ == "__main__":
    main()
