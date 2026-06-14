"""
对 8 个代表性事件，绘制 GWOSC quickview 风格的"三联图"：
  列 1：Time-series（ON-source 4s 窗口的原始 strain 时域）
  列 2：ASD（整段 strain 376s 的 amplitude spectral density）
  列 3：Q-transform spectrogram（复用已生成的 pos_j0.png）

行 = (event, ifo) 组合，H1 在上 L1 在下，事件按 chirp_mass 排序。

目的：让用户对原始数据从三个角度同时核对——时域、频域、时频。
对齐 GWOSC quickview 工具（gwosc-tutorial/quickview）的 4 fig 流程。

输出: output/data_triplet_montage.png
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from PIL import Image, ImageDraw, ImageFont

from config import (
    OUTPUT_DIR,
    RAW_STRAIN_DIR,
    SPECTROGRAMS_DIR,
    load_events_from_csv,
)


OUT_PATH = OUTPUT_DIR / "data_triplet_montage.png"


SELECTED = [
    "GW170817",
    "GW190425",
    "GW200115_042309",
    "GW190814",
    "GW190412",
    "GW150914",
    "GW190521",
    "GW200322_091133",
    "GW190413_052954",  # 含 glitch 的事件
]

DETECTORS_TO_PLOT = ["H1", "L1"]
CELL_W = 380
CELL_H = 280
ROW_HEADER_W = 200
COL_HEADER_H = 36
GAP = 2


def render_timeseries(strain: TimeSeries, gps: float) -> Image.Image:
    """ON-source 4s strain 时域图。"""
    half = 2.0
    seg = strain.crop(gps - half, gps + half)
    t = np.arange(len(seg)) / seg.sample_rate.value - half  # 相对时间
    fig, ax = plt.subplots(figsize=(CELL_W / 100, CELL_H / 100), dpi=100)
    ax.plot(t, seg.value, linewidth=0.4, color="#1f4e79")
    ax.set_xlim(-half, half)
    ax.set_xlabel("Time relative to merger [s]", fontsize=8)
    ax.set_ylabel("Strain", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((CELL_W, CELL_H), Image.LANCZOS)


def render_asd(strain: TimeSeries) -> Image.Image:
    """整段 strain ASD 图。"""
    asd = strain.asd(fftlength=4)
    fig, ax = plt.subplots(figsize=(CELL_W / 100, CELL_H / 100), dpi=100)
    ax.loglog(asd.frequencies.value, asd.value, linewidth=0.6, color="#2d6e2d")
    ax.set_xlim(10, 2000)
    ax.set_ylim(1e-24, 1e-19)
    ax.set_xlabel("Frequency [Hz]", fontsize=8)
    ax.set_ylabel("ASD [strain/√Hz]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(which="both", alpha=0.3, linewidth=0.4)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((CELL_W, CELL_H), Image.LANCZOS)


def load_qtransform_png(event_name: str, ifo: str) -> Image.Image:
    """复用 02 生成的 pos_j0 PNG。"""
    src = SPECTROGRAMS_DIR / f"{event_name}_{ifo}_pos_j0.png"
    cell = Image.new("RGB", (CELL_W, CELL_H), color=(60, 60, 60))
    if src.exists():
        img = Image.open(src).convert("RGB").resize((CELL_W, CELL_H), Image.LANCZOS)
        cell.paste(img, (0, 0))
    else:
        draw = ImageDraw.Draw(cell)
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        text = "N/A (no PNG)"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((CELL_W - tw) // 2, (CELL_H - th) // 2), text, fill=(200, 200, 200), font=font)
    return cell


def render_row(event: dict, ifo: str) -> tuple:
    """生成一个 (event, ifo) 行的 3 个图 + 状态。"""
    strain_path = RAW_STRAIN_DIR / f"{event['name']}_{ifo}.hdf5"
    if not strain_path.exists():
        return None, None, None, "MISSING"
    try:
        strain = TimeSeries.read(strain_path, format="hdf5")
        # NaN 检查
        if np.isnan(strain.value).sum() / len(strain.value) > 0.05:
            return None, None, None, "NAN"

        ts_img = render_timeseries(strain, event["gps"])
        asd_img = render_asd(strain)
        qt_img = load_qtransform_png(event["name"], ifo)
        return ts_img, asd_img, qt_img, "OK"
    except Exception as exc:
        return None, None, None, f"ERR: {exc.__class__.__name__}"


def main() -> None:
    events_dict = {e["name"]: e for e in load_events_from_csv()}
    rows = []
    for name in SELECTED:
        ev = events_dict.get(name)
        if ev is None:
            print(f"[skip] {name} not in events.csv")
            continue
        for ifo in DETECTORS_TO_PLOT:
            rows.append((ev, ifo))

    n_rows = len(rows)
    n_cols = 3  # time-series, ASD, Q-transform

    width = ROW_HEADER_W + n_cols * CELL_W + (n_cols + 1) * GAP
    height = COL_HEADER_H + n_rows * CELL_H + (n_rows + 1) * GAP

    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_hdr = ImageFont.truetype("Arial.ttf", 14)
        font_row = ImageFont.truetype("Arial.ttf", 12)
    except OSError:
        font_hdr = ImageFont.load_default()
        font_row = ImageFont.load_default()

    col_labels = ["Time-series (ON-source 4s)", "ASD (376s segment)", "Q-transform (pos_j0)"]
    for j, label in enumerate(col_labels):
        x = ROW_HEADER_W + j * CELL_W + (j + 1) * GAP + CELL_W // 2
        bbox = draw.textbbox((0, 0), label, font=font_hdr)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 10), label, fill=(0, 0, 0), font=font_hdr)

    for i, (ev, ifo) in enumerate(rows):
        y_top = COL_HEADER_H + i * CELL_H + (i + 1) * GAP
        snr = ev.get("snr")
        snr_str = f"SNR={snr:.1f}" if snr is not None else "SNR=n/a"
        header_lines = [
            ev["name"],
            f"{ifo}  {ev['kind']}",
            f"M={ev['chirp_mass']:.2f}",
            f"{snr_str}",
        ]
        draw.multiline_text((6, y_top + 8), "\n".join(header_lines), fill=(0, 0, 0), font=font_row, spacing=3)

        ts_img, asd_img, qt_img, status = render_row(ev, ifo)
        for j, img in enumerate([ts_img, asd_img, qt_img]):
            x_left = ROW_HEADER_W + j * CELL_W + (j + 1) * GAP
            if img is None:
                cell = Image.new("RGB", (CELL_W, CELL_H), color=(220, 220, 220))
                d2 = ImageDraw.Draw(cell)
                try:
                    f2 = ImageFont.truetype("Arial.ttf", 16)
                except OSError:
                    f2 = ImageFont.load_default()
                d2.text((CELL_W // 2 - 30, CELL_H // 2 - 8), status, fill=(150, 0, 0), font=f2)
                canvas.paste(cell, (x_left, y_top))
            else:
                canvas.paste(img, (x_left, y_top))

        print(f"  [{i+1:2d}/{n_rows}] {ev['name']} {ifo}: {status}")

    canvas.save(OUT_PATH, format="PNG")
    print()
    print(f"Saved: {OUT_PATH}  ({canvas.size[0]}x{canvas.size[1]})")
    print(f"Open with: open {OUT_PATH}")


if __name__ == "__main__":
    main()
