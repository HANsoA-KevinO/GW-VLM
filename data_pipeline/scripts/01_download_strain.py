"""
下载 GW 事件的 H1/L1/V1 strain。

模式：
- --poc: 仅 5 个代表性事件（Stage 0 PoC，已完成）
- --full（默认）: events.csv 全量 ~93 事件（Stage 1+）

用 gwpy.TimeSeries.fetch_open_data 拉一个足够长的段：
[gps - max_negative_offset - PSD_DURATION - pad, gps + pad]，
确保下游脚本能取出 PSD 估计基底 + ON/OFF 两个窗口。
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gwpy.timeseries import TimeSeries

from config import (
    DETECTORS,
    POC_EVENTS,
    PSD_DURATION,
    RAW_STRAIN_DIR,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    load_events_from_csv,
)

DEFAULT_WORKERS = 8


def download_segment(event: dict, ifo: str) -> str:
    """下载一个事件 + 探测器的 strain 段，保存为 HDF5。返回单行状态字符串。"""
    name = event["name"]
    gps = event["gps"]
    neg_offset = event["negative_offset"]

    half_window = WINDOW_SECONDS / 2.0
    pad = 8.0
    start = gps - neg_offset - half_window - PSD_DURATION - pad
    end = gps + half_window + pad

    out_path = RAW_STRAIN_DIR / f"{name}_{ifo}.hdf5"
    if out_path.exists():
        return f"[skip] {out_path.name} 已存在"

    try:
        ts = TimeSeries.fetch_open_data(ifo, start, end, sample_rate=SAMPLE_RATE, cache=True)
        ts.write(out_path, format="hdf5", overwrite=True)
        return f"[ok]   {out_path.name}  ({len(ts)} samples, {end - start:.1f}s)"
    except Exception as exc:
        return f"[fail] {name} {ifo}: {exc.__class__.__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--poc", action="store_true", help="只下载 5 PoC 事件")
    group.add_argument("--full", action="store_true", help="下载 events.csv 全量（默认）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数 (默认 {DEFAULT_WORKERS})")
    args = parser.parse_args()

    if args.poc:
        events = POC_EVENTS
        mode = "POC (5 events)"
    else:
        events = load_events_from_csv()
        mode = f"FULL ({len(events)} events)"

    RAW_STRAIN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Mode: {mode}, workers={args.workers}", flush=True)
    print(f"Output: {RAW_STRAIN_DIR}\n", flush=True)

    tasks = [(event, ifo) for event in events for ifo in DETECTORS]
    total = len(tasks)
    print(f"Submitting {total} download tasks ({len(events)} events × {len(DETECTORS)} detectors)\n", flush=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_segment, ev, ifo): (ev["name"], ifo) for ev, ifo in tasks}
        for fut in as_completed(futures):
            completed += 1
            name, ifo = futures[fut]
            status = fut.result()
            print(f"[{completed:3d}/{total}] {status}", flush=True)


if __name__ == "__main__":
    main()
