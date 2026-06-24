"""
对每条 strain 生成 ON-source 和 OFF-source 时频图。

模式：
- --poc: 5 PoC 事件，每事件每探测器 2 张图（pos/neg）—— Stage 0 PoC
- --full（默认）: 90 全量事件，每事件每探测器 5 jitter × 2 = 10 张图 —— Stage 1+
  正样本通过 ±0.5s jitter 让信号在窗口里位置变化
  负样本通过 5 个不同 OFF-source 时刻（GPS - neg_offset ± 偏移）实现噪声多样性

处理流程对齐 GWOSC quickview：直接 q_transform，由内置 whitening 处理。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from PIL import Image

from config import (
    DETECTORS,
    POC_EVENTS,
    FREQ_RANGE,
    IMAGE_SIZE,
    Q_RANGE,
    RAW_STRAIN_DIR,
    SPECTROGRAMS_DIR,
    WINDOW_SECONDS,
    load_events_from_csv,
)


# 顺序兼容旧文件名：j0-j4 与早期 PoC 版本一致，j5-j8 是 Stage 1 v0.3 新增。
# 整体覆盖范围 ±1.0s (POS) / ±20s (NEG)，9 个 jitter。
JITTER_OFFSETS = [-0.5, -0.25, 0.0, 0.25, 0.5, -1.0, -0.75, 0.75, 1.0]
NEG_TIME_OFFSETS = [-10.0, -5.0, 0.0, 5.0, 10.0, -20.0, -15.0, 15.0, 20.0]


def render_qtransform(strain: TimeSeries, center_gps: float, out_path: Path,
                      cmap: str = "gray") -> None:
    """从一段 strain 中以 center_gps 为中心切显示窗口，对齐 GWOSC quickview 流程：
    直接调 q_transform，由其内置 whitening 处理；输入整段 strain 提供 PSD 估计基底。
    cmap：'gray'（默认，灰度单通道）或 'viridis' 等（彩色三通道）。固定 vmin0/vmax25.5 不变。
    """
    half_window = WINDOW_SECONDS / 2.0
    win_start = center_gps - half_window
    win_end = center_gps + half_window

    if win_start < strain.t0.value or win_end > (strain.t0.value + strain.duration.value):
        raise ValueError(
            f"strain 不覆盖所需窗口 "
            f"(strain {strain.t0.value:.1f}-{strain.t0.value + strain.duration.value:.1f}, "
            f"need {win_start:.1f}-{win_end:.1f})"
        )

    qt = strain.q_transform(
        qrange=Q_RANGE,
        frange=FREQ_RANGE,
        outseg=(win_start, win_end),
    )

    fig, ax = plt.subplots(figsize=(8, 8), dpi=IMAGE_SIZE // 8)
    ax.imshow(
        np.asarray(qt.value).T,
        aspect="auto",
        origin="lower",
        extent=[float(qt.xindex[0].value), float(qt.xindex[-1].value), FREQ_RANGE[0], FREQ_RANGE[1]],
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=25.5,
    )
    ax.set_yscale("log")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    fig.savefig(out_path, format="png", dpi=IMAGE_SIZE // 8, pad_inches=0)
    plt.close(fig)

    mode = "L" if cmap == "gray" else "RGB"   # 彩色保存为 RGB 三通道
    img = Image.open(out_path).convert(mode)
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img.save(out_path, format="PNG")


def process_event_detector(event: dict, ifo: str, jitter: bool,
                           out_dir: Path = SPECTROGRAMS_DIR, cmap: str = "gray") -> None:
    name = event["name"]
    gps = event["gps"]
    neg_offset = event["negative_offset"]

    strain_path = RAW_STRAIN_DIR / f"{name}_{ifo}.hdf5"
    if not strain_path.exists():
        print(f"  [skip] {name} {ifo}: strain 文件不存在")
        return

    strain = TimeSeries.read(strain_path, format="hdf5")

    if jitter:
        pos_centers = [(f"pos_j{i}", gps + d) for i, d in enumerate(JITTER_OFFSETS)]
        neg_centers = [(f"neg_j{i}", gps - neg_offset + d) for i, d in enumerate(NEG_TIME_OFFSETS)]
        targets = pos_centers + neg_centers
    else:
        targets = [("pos", gps), ("neg", gps - neg_offset)]

    for label, center in targets:
        out_path = out_dir / f"{name}_{ifo}_{label}.png"
        if out_path.exists():
            continue
        try:
            render_qtransform(strain, center, out_path, cmap=cmap)
            print(f"  [ok]   {out_path.name}")
        except Exception as exc:
            print(f"  [fail] {out_path.name}: {exc.__class__.__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--poc", action="store_true", help="5 PoC 事件，每事件每探测器 2 图")
    group.add_argument("--full", action="store_true", help="全量事件 + 5 jitter（默认）")
    parser.add_argument("--cmap", default="gray", help="色图：gray(默认) / viridis / ...")
    parser.add_argument("--outdir", default=None, help="输出目录(默认 output/spectrograms)")
    args = parser.parse_args()
    out_dir = Path(args.outdir) if args.outdir else SPECTROGRAMS_DIR

    if args.poc:
        events = POC_EVENTS
        jitter = False
        mode = "POC (5 events, 2 imgs/event/ifo)"
    else:
        events = load_events_from_csv()
        jitter = True
        mode = f"FULL ({len(events)} events, 10 imgs/event/ifo)"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Mode: {mode}  |  cmap={args.cmap}")
    print(f"Output: {out_dir}\n")

    for event in events:
        snr_str = f"SNR ~{event['snr']}" if event.get("snr") is not None else "SNR n/a"
        print(f"=== {event['name']} ({event['kind']}, {snr_str}) ===")
        for ifo in DETECTORS:
            process_event_detector(event, ifo, jitter=jitter, out_dir=out_dir, cmap=args.cmap)
        print()


if __name__ == "__main__":
    main()
