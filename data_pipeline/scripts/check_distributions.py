"""
步骤 2: 训练数据集多维度分布统计。

读 dataset_{train,val,test}.jsonl，分析：
  A. 信号类型分布（BBH/BNS/NSBH）
  B. 参数 bin 分布（chirp_mass 7 / distance 6 / chi_eff 5）
  C. SNR 分布（4-8 / 8-12 / 12-20 / 20+）
  D. 探测器分布（H1/L1）
  E. train/val/test 三集合的 bin 覆盖度对比
  F. 物理参数散点图（chirp_mass vs distance, chirp_mass vs SNR）

输出：
  output/dist_report.txt        各维度文本表格
  output/dist_histograms.png    6 张直方图
  output/dist_scatter.png       2 张物理参数散点图
  output/dist_per_split.txt     三集合 bin 覆盖度对比
  output/dist_data.json         机器可读统计数据

判断标准：
  - 每个 bin 至少 ≥10 样本
  - test 的 bin ⊆ train 的 bin（防止预测未学过的标签）
  - 探测器 H1/L1 应大致平衡
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    CHIRP_MASS_BINS,
    CHI_EFF_BINS,
    DISTANCE_BINS,
    OUTPUT_DIR,
)


REPORT_PATH = OUTPUT_DIR / "dist_report.txt"
PER_SPLIT_PATH = OUTPUT_DIR / "dist_per_split.txt"
DATA_PATH = OUTPUT_DIR / "dist_data.json"
HISTOGRAMS_PATH = OUTPUT_DIR / "dist_histograms.png"
SCATTER_PATH = OUTPUT_DIR / "dist_scatter.png"


SNR_BINS = [
    (0.0, 8.0, "4-8 (low)"),
    (8.0, 12.0, "8-12 (mid)"),
    (12.0, 20.0, "12-20 (high)"),
    (20.0, float("inf"), "20+ (loud)"),
]


def load_split(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def snr_bin_label(snr: float | None) -> str:
    if snr is None:
        return "n/a"
    for lo, hi, label in SNR_BINS:
        if lo <= snr < hi:
            return label
    return SNR_BINS[-1][2]


def analyze_split(samples: list[dict]) -> dict:
    """对单个 split 提取统计。"""
    kinds = Counter()
    chirp_bins = Counter()
    distance_bins = Counter()
    chi_eff_bins = Counter()
    snr_bin_counts = Counter()
    ifos = Counter()
    source_types = Counter()
    detection_counts = Counter()

    scatter_data = {
        "chirp_mass": [],
        "luminosity_distance": [],
        "snr": [],
        "kind": [],
        "detection": [],
    }

    for s in samples:
        meta = s.get("metadata", {})
        label = s.get("label", {})
        source_types[s.get("source_type", "?")] += 1
        ifos[s.get("ifo", "?")] += 1
        detection_counts[label.get("detection", "?")] += 1

        # 正样本才统计参数 bin
        if label.get("detection") == "YES":
            chirp_bins[label.get("chirp_mass_bin", "N/A")] += 1
            distance_bins[label.get("distance_bin", "N/A")] += 1
            chi_eff_bins[label.get("chi_eff_bin", "N/A")] += 1
            kinds[meta.get("kind", "?")] += 1
            snr_bin_counts[snr_bin_label(meta.get("snr"))] += 1

            if meta.get("chirp_mass") is not None and meta.get("luminosity_distance") is not None:
                scatter_data["chirp_mass"].append(meta["chirp_mass"])
                scatter_data["luminosity_distance"].append(meta["luminosity_distance"])
                scatter_data["snr"].append(meta.get("snr", 0))
                scatter_data["kind"].append(meta.get("kind", "?"))
                scatter_data["detection"].append("YES")

    return {
        "n_samples": len(samples),
        "kinds": dict(kinds),
        "chirp_bins": dict(chirp_bins),
        "distance_bins": dict(distance_bins),
        "chi_eff_bins": dict(chi_eff_bins),
        "snr_bins": dict(snr_bin_counts),
        "ifos": dict(ifos),
        "source_types": dict(source_types),
        "detection": dict(detection_counts),
        "scatter": scatter_data,
    }


def format_count_table(title: str, splits: dict[str, dict], key: str, order: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    header = f"  {'bin':<25s} | {'Train':>8s} | {'Val':>6s} | {'Test':>6s} | {'Total':>7s}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    total_per_split = {name: 0 for name in splits}
    for bin_label in order:
        train_v = splits["Train"][key].get(bin_label, 0)
        val_v = splits["Val"][key].get(bin_label, 0)
        test_v = splits["Test"][key].get(bin_label, 0)
        total = train_v + val_v + test_v
        for name, v in [("Train", train_v), ("Val", val_v), ("Test", test_v)]:
            total_per_split[name] += v
        lines.append(f"  {bin_label:<25s} | {train_v:>8d} | {val_v:>6d} | {test_v:>6d} | {total:>7d}")
    # 行尾汇总
    tot_t = total_per_split["Train"]; tot_v = total_per_split["Val"]; tot_te = total_per_split["Test"]
    lines.append("  " + "-" * (len(header) - 2))
    lines.append(f"  {'TOTAL':<25s} | {tot_t:>8d} | {tot_v:>6d} | {tot_te:>6d} | {tot_t+tot_v+tot_te:>7d}")
    lines.append("")
    return lines


def check_warnings(splits: dict) -> list[str]:
    """检测潜在问题：空 bin、test 包含 train 没见过的 bin。"""
    warnings = []
    keys_to_check = [
        ("chirp_bins", [b[2] for b in CHIRP_MASS_BINS]),
        ("distance_bins", [b[2] for b in DISTANCE_BINS]),
        ("chi_eff_bins", [b[2] for b in CHI_EFF_BINS]),
        ("snr_bins", [b[2] for b in SNR_BINS]),
        ("kinds", ["BBH", "BNS", "NSBH"]),
    ]
    for key, all_bins in keys_to_check:
        train_bins = set(splits["Train"][key].keys())
        val_bins = set(splits["Val"][key].keys())
        test_bins = set(splits["Test"][key].keys())

        # test 包含 train 没见过的 bin
        leaked = test_bins - train_bins
        if leaked:
            warnings.append(f"⚠️  {key}: Test 含 Train 没见过的 bin: {leaked}")

        # val 类似
        leaked_v = val_bins - train_bins
        if leaked_v:
            warnings.append(f"⚠️  {key}: Val 含 Train 没见过的 bin: {leaked_v}")

        # 完全空 bin
        all_bins_set = set(all_bins)
        empty = all_bins_set - (train_bins | val_bins | test_bins)
        if empty:
            warnings.append(f"⚠️  {key}: 完全无样本的 bin: {empty}")

        # Train 中样本数 < 10 的 bin
        sparse_train = [b for b in train_bins if splits["Train"][key].get(b, 0) < 10]
        if sparse_train:
            warnings.append(f"⚠️  {key}: Train 中样本 < 10 的 bin: {sparse_train}")

    return warnings


def render_histograms(splits: dict) -> None:
    """6 子图：kind, chirp, distance, chi_eff, snr, ifo——按 train/val/test 堆叠"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_specs = [
        ("kinds", ["BBH", "BNS", "NSBH"], "Signal Type (POS only)"),
        ("chirp_bins", [b[2] for b in CHIRP_MASS_BINS], "Chirp Mass Bin (POS only)"),
        ("distance_bins", [b[2] for b in DISTANCE_BINS], "Distance Bin (POS only)"),
        ("chi_eff_bins", [b[2] for b in CHI_EFF_BINS], "Chi_eff Bin (POS only)"),
        ("snr_bins", [b[2] for b in SNR_BINS], "SNR Bin (POS only)"),
        ("ifos", ["H1", "L1"], "Detector (all samples)"),
    ]

    colors = {"Train": "#1f77b4", "Val": "#ff7f0e", "Test": "#2ca02c"}

    for ax, (key, all_bins, title) in zip(axes, plot_specs):
        x = np.arange(len(all_bins))
        width = 0.27
        for i, name in enumerate(["Train", "Val", "Test"]):
            heights = [splits[name][key].get(b, 0) for b in all_bins]
            ax.bar(x + (i - 1) * width, heights, width, label=name, color=colors[name])
        ax.set_xticks(x)
        ax.set_xticklabels(all_bins, rotation=30, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Sample count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(HISTOGRAMS_PATH, dpi=100)
    plt.close(fig)
    print(f"Histograms → {HISTOGRAMS_PATH}")


def render_scatter(splits: dict) -> None:
    """2 子图：chirp_mass vs distance, chirp_mass vs SNR"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    kind_colors = {"BBH": "#1f77b4", "BNS": "#d62728", "NSBH": "#2ca02c"}
    split_markers = {"Train": "o", "Val": "s", "Test": "^"}

    for split_name, marker in split_markers.items():
        sc = splits[split_name]["scatter"]
        for kind in set(sc.get("kind", [])):
            mask = [k == kind for k in sc["kind"]]
            cm = [v for v, m in zip(sc["chirp_mass"], mask) if m]
            dist = [v for v, m in zip(sc["luminosity_distance"], mask) if m]
            snr = [v for v, m in zip(sc["snr"], mask) if m]
            axes[0].scatter(cm, dist, c=kind_colors.get(kind, "gray"), marker=marker,
                            label=f"{split_name}-{kind}", alpha=0.5, s=20)
            axes[1].scatter(cm, snr, c=kind_colors.get(kind, "gray"), marker=marker,
                            label=f"{split_name}-{kind}", alpha=0.5, s=20)

    axes[0].set_xlabel("Chirp mass (M_sun)")
    axes[0].set_ylabel("Luminosity distance (Mpc)")
    axes[0].set_title("Chirp mass × Distance (POS samples)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)

    axes[1].set_xlabel("Chirp mass (M_sun)")
    axes[1].set_ylabel("SNR")
    axes[1].set_title("Chirp mass × SNR (POS samples)")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig(SCATTER_PATH, dpi=100)
    plt.close(fig)
    print(f"Scatter → {SCATTER_PATH}")


def main() -> None:
    splits = {
        "Train": analyze_split(load_split(OUTPUT_DIR / "dataset_train.jsonl")),
        "Val": analyze_split(load_split(OUTPUT_DIR / "dataset_val.jsonl")),
        "Test": analyze_split(load_split(OUTPUT_DIR / "dataset_test.jsonl")),
    }

    lines = [
        "# GW-VLM 训练数据集多维度分布统计",
        "",
        "## 总体样本数",
    ]
    for name in ["Train", "Val", "Test"]:
        s = splits[name]
        lines.append(f"  {name}: {s['n_samples']} 样本"
                     f"  (YES: {s['detection'].get('YES', 0)}, NO: {s['detection'].get('NO', 0)})")
    lines.append("")

    # 各维度表格
    lines.extend(format_count_table("A. 信号类型分布（正样本）", splits, "kinds", ["BBH", "BNS", "NSBH"]))
    lines.extend(format_count_table("B1. chirp_mass_bin 分布（正样本）", splits, "chirp_bins", [b[2] for b in CHIRP_MASS_BINS]))
    lines.extend(format_count_table("B2. distance_bin 分布（正样本）", splits, "distance_bins", [b[2] for b in DISTANCE_BINS]))
    lines.extend(format_count_table("B3. chi_eff_bin 分布（正样本）", splits, "chi_eff_bins", [b[2] for b in CHI_EFF_BINS]))
    lines.extend(format_count_table("C. SNR 分布（正样本）", splits, "snr_bins", [b[2] for b in SNR_BINS]))
    lines.extend(format_count_table("D. 探测器分布（所有样本）", splits, "ifos", ["H1", "L1"]))

    # 警告
    warnings = check_warnings(splits)
    lines.append("## 检测到的问题")
    if not warnings:
        lines.append("  ✅ 无问题")
    else:
        for w in warnings:
            lines.append(f"  {w}")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"详细报告 → {REPORT_PATH}")

    # 简短的 per-split 比较
    per_split_lines = [
        "# Train / Val / Test 各维度覆盖度对比",
        "",
        "目标：Test 和 Val 的 bin 集合应 ⊆ Train 的 bin 集合（防泄露），且各 bin 至少 1 个样本。",
        "",
    ]
    for key, name, all_bins in [
        ("chirp_bins", "chirp_mass_bin", [b[2] for b in CHIRP_MASS_BINS]),
        ("distance_bins", "distance_bin", [b[2] for b in DISTANCE_BINS]),
        ("chi_eff_bins", "chi_eff_bin", [b[2] for b in CHI_EFF_BINS]),
        ("snr_bins", "snr_bin", [b[2] for b in SNR_BINS]),
    ]:
        train_set = set(splits["Train"][key].keys())
        val_set = set(splits["Val"][key].keys())
        test_set = set(splits["Test"][key].keys())
        per_split_lines.append(f"## {name}")
        per_split_lines.append(f"  Train 含: {sorted(train_set)}")
        per_split_lines.append(f"  Val   含: {sorted(val_set)}")
        per_split_lines.append(f"  Test  含: {sorted(test_set)}")
        miss_val = train_set - val_set
        miss_test = train_set - test_set
        leak_val = val_set - train_set
        leak_test = test_set - train_set
        if miss_val:
            per_split_lines.append(f"  ⚠️  Val 缺少（Train 有但 Val 无）: {sorted(miss_val)}")
        if miss_test:
            per_split_lines.append(f"  ⚠️  Test 缺少（Train 有但 Test 无）: {sorted(miss_test)}")
        if leak_val or leak_test:
            per_split_lines.append(f"  🛑 泄露（Val/Test 含 Train 没见过的 bin）: val={leak_val}, test={leak_test}")
        per_split_lines.append("")

    PER_SPLIT_PATH.write_text("\n".join(per_split_lines) + "\n")
    print(f"切分覆盖度 → {PER_SPLIT_PATH}")

    # 渲染图
    render_histograms(splits)
    render_scatter(splits)

    # 机器可读
    DATA_PATH.write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "scatter"} for k, v in splits.items()}, indent=2))
    print(f"机器可读 → {DATA_PATH}")
    print()
    print("=== 主要发现 ===")
    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  ✅ 所有维度通过基本检查（无空 bin / 无切分泄露 / 无样本过少）")


if __name__ == "__main__":
    main()
