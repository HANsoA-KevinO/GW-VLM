"""
为多模态融合(方法2)提取【白化后的原始应变】数组,与每张频谱图一一对齐。

对 dataset.jsonl 里的每个样本,切出和其频谱图同一段 4s 窗口的应变,
经 白化 → 带通(20-512Hz)→ 降采样(2048Hz)→ 存成 output/strain_arrays/<图同名>.npy
(float32, 8192 点)。窗口中心、jitter、负样本偏移与 02_generate_spectrograms.py 完全一致。

效率:每个 (event, ifo) 只加载+白化整段一次,再裁出该探测器的全部 jitter 窗口。
白化用整段做 PSD 基底(对齐 q_transform 用整段的口径);窗口都安全地落在白化边缘腐蚀区之内。

用法(本地,需 gwpy → 用 .venv-render):
  .venv-render/bin/python data_pipeline/scripts/09_extract_strain.py
  .venv-render/bin/python data_pipeline/scripts/09_extract_strain.py --limit 3   # 只跑前3事件,自检
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from gwpy.timeseries import TimeSeries

from config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    DETECTORS,
    OUTPUT_DIR,
    RAW_STRAIN_DIR,
    WINDOW_SECONDS,
    load_events_from_csv,
)

# 必须与 02_generate_spectrograms.py 保持一致(顺序敏感:j0-j4 旧,j5-j8 新增)
JITTER_OFFSETS = [-0.5, -0.25, 0.0, 0.25, 0.5, -1.0, -0.75, 0.75, 1.0]
NEG_TIME_OFFSETS = [-10.0, -5.0, 0.0, 5.0, 10.0, -20.0, -15.0, 15.0, 20.0]

STRAIN_ARRAYS_DIR = OUTPUT_DIR / "strain_arrays"
TARGET_SR = 2048                         # 降采样目标(Nyquist 1024 > 512 带通上限,安全)
N_SAMPLES = int(WINDOW_SECONDS * TARGET_SR)  # 4s * 2048 = 8192
WHITEN_FFTLEN = 4.0                       # 白化 PSD 的 FFT 长度(秒)
WHITEN_OVERLAP = 2.0


def valid_basenames(dataset_path: Path) -> set:
    """从 dataset.jsonl 取所有样本的 basename(无扩展名),只为这些样本产出 npy。"""
    names = set()
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            names.add(Path(rec["image_path"]).stem)  # e.g. GW150914_H1_pos_j0
    return names


def fix_length(arr: np.ndarray, n: int) -> np.ndarray:
    """裁/补到精确 n 点(crop 端点可能多/少 1 点)。"""
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)))


def targets_for(event: dict):
    """返回 [(label, center_gps), ...],与 02 的命名/中心一致。"""
    gps = event["gps"]
    neg_offset = event["negative_offset"]
    pos = [(f"pos_j{i}", gps + d) for i, d in enumerate(JITTER_OFFSETS)]
    neg = [(f"neg_j{i}", gps - neg_offset + d) for i, d in enumerate(NEG_TIME_OFFSETS)]
    return pos + neg


def process_event_detector(event: dict, ifo: str, want: set, out_dir: Path,
                           overwrite: bool) -> tuple:
    name = event["name"]
    strain_path = RAW_STRAIN_DIR / f"{name}_{ifo}.hdf5"
    if not strain_path.exists():
        return 0, 0  # 该探测器无数据(正常,非 bug)

    # 该 (event,ifo) 需要产出的目标(只保留在 dataset 里的)
    tgts = [(lab, c) for lab, c in targets_for(event)
            if f"{name}_{ifo}_{lab}" in want]
    if not tgts:
        return 0, 0
    if not overwrite:
        tgts = [(lab, c) for lab, c in tgts
                if not (out_dir / f"{name}_{ifo}_{lab}.npy").exists()]
    if not tgts:
        return 0, 0

    # 整段加载 + 白化 + 带通 + 降采样(一次)
    strain = TimeSeries.read(strain_path, format="hdf5")
    white = strain.whiten(WHITEN_FFTLEN, WHITEN_OVERLAP)
    white = white.bandpass(BANDPASS_LOW, BANDPASS_HIGH)
    white = white.resample(TARGET_SR)
    t0 = white.t0.value
    t_end = t0 + white.duration.value

    half = WINDOW_SECONDS / 2.0
    ok = fail = 0
    for lab, center in tgts:
        ws, we = center - half, center + half
        try:
            if ws < t0 or we > t_end:
                raise ValueError(f"window {ws:.1f}-{we:.1f} 超出 {t0:.1f}-{t_end:.1f}")
            seg = white.crop(ws, we)
            arr = fix_length(np.asarray(seg.value, dtype=np.float32), N_SAMPLES)
            np.save(out_dir / f"{name}_{ifo}_{lab}.npy", arr)
            ok += 1
        except Exception as exc:
            print(f"  [fail] {name}_{ifo}_{lab}: {exc.__class__.__name__}: {exc}")
            fail += 1
    return ok, fail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=OUTPUT_DIR / "dataset.jsonl")
    ap.add_argument("--outdir", type=Path, default=STRAIN_ARRAYS_DIR)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个事件(自检)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    want = valid_basenames(args.dataset)
    events = load_events_from_csv()
    if args.limit:
        events = events[:args.limit]
    print(f"目标样本(dataset): {len(want)}  | 事件: {len(events)}  | 输出: {args.outdir}")
    print(f"每条: 白化→带通{BANDPASS_LOW:.0f}-{BANDPASS_HIGH:.0f}Hz→{TARGET_SR}Hz→{N_SAMPLES}点 float32\n")

    total_ok = total_fail = 0
    for ev in events:
        eok = efail = 0
        for ifo in DETECTORS:
            ok, fail = process_event_detector(ev, ifo, want, args.outdir, args.overwrite)
            eok += ok
            efail += fail
        total_ok += eok
        total_fail += efail
        if eok or efail:
            print(f"  {ev['name']:24s} ok={eok} fail={efail}")

    print(f"\n完成: 成功 {total_ok}  失败 {total_fail}  | 输出目录 {args.outdir}")
    n_npy = len(list(args.outdir.glob('*.npy')))
    print(f"目录内 .npy 总数: {n_npy} (目标 {len(want)})")


if __name__ == "__main__":
    main()
