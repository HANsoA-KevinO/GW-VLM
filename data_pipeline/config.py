"""
GW-VLM 数据 pipeline config。
- POC_EVENTS：5 个代表性事件，Stage 0 PoC 用
- load_events_from_csv()：从 GW-TF/events.csv 读 93 全量事件，Stage 1+ 用
- 信号处理参数对所有阶段统一
"""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_STRAIN_DIR = OUTPUT_DIR / "raw_strain"
SPECTROGRAMS_DIR = OUTPUT_DIR / "spectrograms"
MONTAGE_PATH = OUTPUT_DIR / "montage.png"

# 自包含：events.csv 随项目分发（data_pipeline/events.csv）。
# 可用环境变量 GWVLM_EVENTS_CSV 覆盖。
import os
EVENTS_CSV_PATH = Path(
    os.environ.get("GWVLM_EVENTS_CSV", Path(__file__).resolve().parent / "events.csv")
)


POC_EVENTS = [
    {
        "name": "GW150914",
        "version": "GW150914-v3",
        "gps": 1126259462.4,
        "kind": "BBH",
        "snr": 24.0,
        "chirp_mass": 28.6,
        "negative_offset": 100.0,
    },
    {
        "name": "GW190521",
        "version": "GW190521-v3",
        "gps": 1242442967.4,
        "kind": "BBH",
        "snr": 14.4,
        "chirp_mass": 64.0,
        "negative_offset": 100.0,
    },
    {
        "name": "GW170817",
        "version": "GW170817-v3",
        "gps": 1187008882.4,
        "kind": "BNS",
        "snr": 32.4,
        "chirp_mass": 1.186,
        "negative_offset": 600.0,
    },
    {
        "name": "GW200115_042309",
        "version": "GW200115_042309-v2",
        "gps": 1263097407.7,
        "kind": "NSBH",
        "snr": 11.0,
        "chirp_mass": 2.42,
        "negative_offset": 100.0,
    },
    {
        "name": "GW200322_091133",
        "version": "GW200322_091133-v1",
        "gps": 1268903511.3,
        "kind": "marginal",
        "snr": 4.5,
        "chirp_mass": 15.0,
        "negative_offset": 100.0,
    },
]

EVENTS = POC_EVENTS  # 向后兼容别名


def _classify_kind(chirp_mass: float) -> str:
    """按 chirp_mass 推算源类型。"""
    if chirp_mass < 2.5:
        return "BNS"
    if chirp_mass < 5.0:
        return "NSBH"
    return "BBH"


def _negative_offset_for(chirp_mass: float) -> float:
    """BNS 长 inspiral 需更早负样本时刻；其他 100s 即够。"""
    return 600.0 if chirp_mass < 5.0 else 100.0


def load_events_from_csv(csv_path: Path = EVENTS_CSV_PATH) -> list[dict]:
    """从 events.csv 读所有事件，自动推算 kind 与 negative_offset。

    必需字段：name、shortName、gps、chirp_mass_source（缺失时跳过）。
    可选字段：luminosity_distance、chi_eff（缺失时为 None，影响 bin 标注）。
    """
    events = []
    skipped = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            short = row.get("shortName", "").strip()
            gps_str = row.get("gps", "").strip()
            cm_str = row.get("chirp_mass_source", "").strip()
            snr_str = row.get("network_matched_filter_snr", "").strip()
            dist_str = row.get("luminosity_distance", "").strip()
            chi_str = row.get("chi_eff", "").strip()

            if not (name and short and gps_str and cm_str):
                skipped.append(name or "<unknown>")
                continue

            try:
                gps = float(gps_str)
                cm = float(cm_str)
                snr = float(snr_str) if snr_str else None
                dist = float(dist_str) if dist_str else None
                chi = float(chi_str) if chi_str else None
            except ValueError:
                skipped.append(name)
                continue

            events.append({
                "name": name,
                "version": short,
                "gps": gps,
                "kind": _classify_kind(cm),
                "snr": snr,
                "chirp_mass": cm,
                "luminosity_distance": dist,
                "chi_eff": chi,
                "negative_offset": _negative_offset_for(cm),
            })

    if skipped:
        print(f"[load_events_from_csv] 跳过 {len(skipped)} 个缺字段事件: {skipped}")
    return events


# ---------------------------------------------------------------------------
# 参数 bin 定义（按 02_research_design.md §4.2）
# ---------------------------------------------------------------------------

# (lower, upper, label) — upper 不含；最后一档 upper 用 inf 表示开放区间
CHIRP_MASS_BINS = [
    (0.0, 2.5, "1-2.5"),
    (2.5, 5.0, "2.5-5"),
    (5.0, 15.0, "5-15"),
    (15.0, 25.0, "15-25"),
    (25.0, 40.0, "25-40"),
    (40.0, 60.0, "40-60"),
    (60.0, float("inf"), "60+"),
]

DISTANCE_BINS = [
    (0.0, 200.0, "<200"),
    (200.0, 400.0, "200-400"),
    (400.0, 800.0, "400-800"),
    (800.0, 1600.0, "800-1600"),
    (1600.0, 3200.0, "1600-3200"),
    (3200.0, float("inf"), "3200+"),
]

CHI_EFF_BINS = [
    (-float("inf"), -0.2, "<-0.2"),
    (-0.2, 0.0, "-0.2-0.0"),
    (0.0, 0.2, "0.0-0.2"),
    (0.2, 0.4, "0.2-0.4"),
    (0.4, float("inf"), "0.4+"),
]


def assign_bin(value: float | None, bins: list[tuple]) -> str:
    """将连续值分入对应 bin；None 返回 'N/A'。"""
    if value is None:
        return "N/A"
    for lo, hi, label in bins:
        if lo <= value < hi:
            return label
    return bins[-1][2]  # 兜底返回最后一档


# 连续参数(unified schema 用):参数名 → 中间 dataset metadata 里的 key。
# 顺序的唯一真源在 training/models/posterior_head.py 的 PARAM_NAMES;这里只管"从哪取值"。
PARAM_METADATA_KEYS = {
    "chirp_mass": "chirp_mass",
    "distance": "luminosity_distance",
    "chi_eff": "chi_eff",
}

DETECTORS = ["H1", "L1"]  # V1 排除（O3 BNS 距离仅 45-51 Mpc，多数事件 SNR<5，与 MLGWSC-1 / Gabbard 2018 等社区标准对齐）


SAMPLE_RATE = 4096
WINDOW_SECONDS = 4.0
BANDPASS_LOW = 20.0
BANDPASS_HIGH = 512.0
Q_RANGE = (4.0, 64.0)
FREQ_RANGE = (20.0, 512.0)
PSD_DURATION = 256.0


IMAGE_SIZE = 1024


MONTAGE_CELL_SIZE = 256


# ---------------------------------------------------------------------------
# 噪声池扩充(03b):从 GWOSC 拉干净 O3 噪声 → 负样本池 + 注入背景池(GPS 冻结、互斥)
# ---------------------------------------------------------------------------
RAW_NOISE_DIR = OUTPUT_DIR / "raw_noise"
NOISE_POOL_MANIFEST = OUTPUT_DIR / "noise_pool_manifest.jsonl"
# O3a / O3b GPS 区间(GWOSC 公开数据)
O3A_RANGE = (1238166018, 1253977218)
O3B_RANGE = (1256655618, 1269363618)
# 数据质量旗标:CBC CAT2(同事件用的搜索质量)+ 无硬件注入。取 H1∩L1 同时段。
NOISE_DQ_FLAGS = ["H1_CBC_CAT2", "L1_CBC_CAT2", "H1_NO_CBC_HW_INJ", "L1_NO_CBC_HW_INJ"]
NOISE_EVENT_GUARD = 180.0        # 事件 GPS ±此秒数 veto(含 GWTC 全部 + 我们 events.csv)
NOISE_SEG_PAD = 8.0              # 段两端留白(白化边缘腐蚀 + 窗口半长)
NOISE_SUBSEG_HALF = 64.0        # 注入子段半长(与 04 一致)
NOISE_NEG_MIN_DUR = 2 * WINDOW_SECONDS + 2 * NOISE_SEG_PAD          # neg 段最短(只需 4s 窗)= 24s
NOISE_INJBG_MIN_DUR = 2 * (NOISE_SUBSEG_HALF + NOISE_SEG_PAD) + 40  # injbg 段最短(容多个注入中心)= 184s
NEG_FRACTION = 0.6              # 干净段中 ~60% 当负样本,~40% 当注入背景
NOISE_NEG_WINDOWS_PER_SEG = 12  # 每个 neg 段每探测器切多少独立窗口(实际数量按配平回调)
NOISE_GLITCH_MAX = 8.0          # 幅度筛:白化窗口 |max| 超此值视为 glitch,丢弃(挡未编目 glitch)
