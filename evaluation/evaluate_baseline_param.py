"""评估经典基线(strain→参数后验)。与 evaluate_unified 同口径的参数指标:
NLL / MAE / bin 等价精度(对照 E2)/ PIT 覆盖率。只在真实正样本上。

用法:
  python evaluation/evaluate_baseline_param.py --adapter output/runs/baseline_param \
      --test output/training_data/e5/test.jsonl --strain-root output/strain_arrays
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "data_pipeline"))
SQRT2 = math.sqrt(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="output/runs/baseline_param")
    ap.add_argument("--test", default="output/training_data/e5/test.jsonl")
    ap.add_argument("--strain-root", default="output/strain_arrays")
    args = ap.parse_args()

    import torch
    from models.baseline_param import StrainParamBaseline
    from models.posterior_head import PARAM_NAMES, standardize, invert
    from config import CHIRP_MASS_BINS, TOTAL_MASS_BINS, CHI_EFF_BINS, assign_bin
    BINS = {"chirp_mass": CHIRP_MASS_BINS, "total_mass": TOTAL_MASS_BINS, "chi_eff": CHI_EFF_BINS}

    adp = Path(args.adapter)
    meta = json.loads((adp / "baseline_meta.json").read_text())
    stats = json.loads((adp / "norm_stats.json").read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = StrainParamBaseline(hidden=meta["hidden"], enc_type=meta["enc_type"],
                              patch_size=meta["patch_size"]).to(dev)
    net.load_state_dict(torch.load(adp / "baseline.pt", map_location=dev)); net.eval()

    X, phys = [], []
    for line in open(args.test):
        r = json.loads(line); t = r.get("targets") or {}
        if not r.get("param_eval", True):   # 跳过不可靠标签事件(与统一评估同口径)
            continue
        if not all(t.get(n) is not None for n in PARAM_NAMES):
            continue
        X.append(np.load(f"{args.strain_root}/{r['basename']}.npy").astype("float32"))
        phys.append([float(t[n]) for n in PARAM_NAMES])
    X = torch.tensor(np.array(X)).to(dev); tp = np.array(phys)
    with torch.no_grad():
        mu, ls = net(X)
    zmu = mu.float().cpu().numpy(); zls = ls.float().cpu().numpy()
    sigma_z = np.exp(zls); true_z = standardize(tp, stats)
    nll = 0.5 * ((true_z - zmu) / sigma_z) ** 2 + np.log(sigma_z) + 0.5 * math.log(2 * math.pi)
    median_phys, _, _ = invert(zmu, zls, stats)
    pit = 0.5 * (1.0 + np.vectorize(math.erf)((true_z - zmu) / (sigma_z * SQRT2)))

    rep = {"n_pos": len(tp)}
    for j, name in enumerate(PARAM_NAMES):
        bins = BINS[name]; idxmap = {lab: i for i, (_, _, lab) in enumerate(bins)}
        exact = adj = 0
        for pp, pt in zip(median_phys[:, j], tp[:, j]):
            bp, bt = assign_bin(float(pp), bins), assign_bin(float(pt), bins)
            exact += (bp == bt)
            if bp in idxmap and bt in idxmap and abs(idxmap[bp] - idxmap[bt]) <= 1:
                adj += 1
        n = len(tp); pj = pit[:, j]
        rep[name] = {"nll": round(float(nll[:, j].mean()), 4),
                     "mae": round(float(np.mean(np.abs(median_phys[:, j] - tp[:, j]))), 4),
                     "exact_bin_acc": round(exact / n, 4), "adjacent_bin_acc": round(adj / n, 4),
                     "coverage_50": round(float(np.mean((pj > 0.25) & (pj < 0.75))), 4),
                     "coverage_90": round(float(np.mean((pj > 0.05) & (pj < 0.95))), 4),
                     "chance": round(1.0 / len(bins), 4)}
    (adp / "baseline_eval.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n=== 经典基线参数评估(可靠标签真实正样本 n={len(tp)};对照 E2-old: chirp .239/chi .265)===")
    for name in PARAM_NAMES:
        m = rep[name]
        print(f"  {name:11s} NLL={m['nll']:.3f} MAE={m['mae']:.3f} bin-exact={m['exact_bin_acc']:.3f}"
              f"(随机{m['chance']}) adj={m['adjacent_bin_acc']:.3f} cov50={m['coverage_50']:.2f} cov90={m['coverage_90']:.2f}")
    print(f"写入 {adp}/baseline_eval.json")


if __name__ == "__main__":
    main()
