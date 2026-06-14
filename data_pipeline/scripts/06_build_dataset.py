"""
扫描 spectrograms/ 目录，按文件名规则将 PNG 归类并生成 JSONL 数据集。

文件名规则（与 02/04/05 脚本约定一致）：
  {event_name}_{ifo}_pos*.png       → 正样本（真实事件 ON-source）
  {event_name}_{ifo}_neg*.png       → 负样本（OFF-source 纯噪声）
  inject_{id}_{ifo}.png             → 正样本（注入信号，metadata 在伴随 .json 中）
  glitch_{glitch_id}_{ifo}.png      → 负样本（GravitySpy hard negative）

输出：
  output/dataset.jsonl   每行 1 个样本，含 image_path + label + split_key
  output/dataset_stats.txt   类别分布统计

JSONL schema（中间格式，训练时 transform 成具体框架格式）：
  {
    "image_path": "/absolute/path.png",
    "label": {"detection": "YES"|"NO", "chirp_mass_bin": ..., "distance_bin": ..., "chi_eff_bin": ...},
    "source_type": "real_pos" | "real_neg_off" | "inject_pos" | "glitch_neg",
    "split_key": "GW150914" | "inject_xxx" | "glitch_xxx",  # 按此 key 切 60/20/20，防泄露
    "event_name": "GW150914",       # 真实事件时填，注入/glitch 为 null
    "ifo": "H1",
    "jitter_idx": 0,                  # jitter 索引（real_pos/real_neg_off）
    "metadata": {...}                  # 注入时填源参数；glitch 时填类别
  }
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CHIRP_MASS_BINS,
    CHI_EFF_BINS,
    DETECTORS,
    DISTANCE_BINS,
    OUTPUT_DIR,
    SPECTROGRAMS_DIR,
    assign_bin,
    load_events_from_csv,
)


DATASET_PATH = OUTPUT_DIR / "dataset.jsonl"
STATS_PATH = OUTPUT_DIR / "dataset_stats.txt"


REAL_PAT = re.compile(r"^(?P<event>GW\d+(?:_\d+)?)_(?P<ifo>H1|L1|V1)_(?P<kind>pos|neg)(?:_j(?P<j>\d+))?\.png$")
INJECT_PAT = re.compile(r"^inject_(?P<id>[A-Za-z0-9_-]+)_(?P<ifo>H1|L1|V1)\.png$")
GLITCH_PAT = re.compile(r"^glitch_(?P<id>[A-Za-z0-9_-]+)_(?P<ifo>H1|L1|V1)\.png$")


def build_real_label(event: dict, kind: str) -> dict:
    """对真实事件 PNG 生成 label。"""
    if kind == "pos":
        return {
            "detection": "YES",
            "chirp_mass_bin": assign_bin(event.get("chirp_mass"), CHIRP_MASS_BINS),
            "distance_bin": assign_bin(event.get("luminosity_distance"), DISTANCE_BINS),
            "chi_eff_bin": assign_bin(event.get("chi_eff"), CHI_EFF_BINS),
        }
    return {
        "detection": "NO",
        "chirp_mass_bin": "N/A",
        "distance_bin": "N/A",
        "chi_eff_bin": "N/A",
    }


def build_inject_label(meta: dict) -> dict:
    return {
        "detection": "YES",
        "chirp_mass_bin": assign_bin(meta.get("chirp_mass"), CHIRP_MASS_BINS),
        "distance_bin": assign_bin(meta.get("luminosity_distance"), DISTANCE_BINS),
        "chi_eff_bin": assign_bin(meta.get("chi_eff"), CHI_EFF_BINS),
    }


def build_glitch_label() -> dict:
    return {
        "detection": "NO",
        "chirp_mass_bin": "N/A",
        "distance_bin": "N/A",
        "chi_eff_bin": "N/A",
    }


def build_dataset() -> list[dict]:
    events = {e["name"]: e for e in load_events_from_csv()}
    pngs = sorted(SPECTROGRAMS_DIR.glob("*.png"))

    samples = []
    unmatched = []
    for png in pngs:
        m = REAL_PAT.match(png.name)
        if m:
            event_name = m.group("event")
            ifo = m.group("ifo")
            if ifo not in DETECTORS:  # 尊重 config.DETECTORS（如 V1 已排除则磁盘有 V1 PNG 也不入库）
                continue
            kind = m.group("kind")
            j_idx = int(m.group("j")) if m.group("j") else 0
            event = events.get(event_name)
            if event is None:
                unmatched.append(f"{png.name}: no event metadata")
                continue
            samples.append({
                "image_path": str(png.resolve()),
                "label": build_real_label(event, kind),
                "source_type": "real_pos" if kind == "pos" else "real_neg_off",
                "split_key": event_name,
                "event_name": event_name,
                "ifo": ifo,
                "jitter_idx": j_idx,
                "metadata": {
                    "kind": event["kind"],
                    "chirp_mass": event["chirp_mass"],
                    "luminosity_distance": event["luminosity_distance"],
                    "chi_eff": event["chi_eff"],
                    "snr": event["snr"],
                },
            })
            continue

        m = INJECT_PAT.match(png.name)
        if m:
            inject_id = m.group("id")
            ifo = m.group("ifo")
            if ifo not in DETECTORS:
                continue
            meta_path = png.with_suffix(".json")
            if not meta_path.exists():
                unmatched.append(f"{png.name}: missing inject metadata json")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            samples.append({
                "image_path": str(png.resolve()),
                "label": build_inject_label(meta),
                "source_type": "inject_pos",
                "split_key": f"inject_{inject_id}",
                "event_name": None,
                "ifo": ifo,
                "jitter_idx": 0,
                "metadata": meta,
            })
            continue

        m = GLITCH_PAT.match(png.name)
        if m:
            glitch_id = m.group("id")
            ifo = m.group("ifo")
            if ifo not in DETECTORS:
                continue
            meta_path = png.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            samples.append({
                "image_path": str(png.resolve()),
                "label": build_glitch_label(),
                "source_type": "glitch_neg",
                "split_key": f"glitch_{glitch_id}",
                "event_name": None,
                "ifo": ifo,
                "jitter_idx": 0,
                "metadata": meta,
            })
            continue

        unmatched.append(png.name)

    if unmatched:
        print(f"[warn] {len(unmatched)} 个文件未匹配规则（前 5 个）: {unmatched[:5]}")

    return samples


def write_outputs(samples: list[dict]) -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    src_counts = Counter(s["source_type"] for s in samples)
    det_counts = Counter(s["label"]["detection"] for s in samples)
    ifo_counts = Counter(s["ifo"] for s in samples)

    lines = [
        f"Total samples: {len(samples)}",
        "",
        "By source_type:",
        *[f"  {k}: {v}" for k, v in src_counts.most_common()],
        "",
        "By detection label:",
        *[f"  {k}: {v}" for k, v in det_counts.most_common()],
        "",
        "By ifo:",
        *[f"  {k}: {v}" for k, v in ifo_counts.most_common()],
    ]
    STATS_PATH.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWritten: {DATASET_PATH}")
    print(f"Stats:   {STATS_PATH}")


def main() -> None:
    samples = build_dataset()
    write_outputs(samples)


if __name__ == "__main__":
    main()
