"""
步骤 1: 肉眼图像质量核实。

选 8 个代表性事件（覆盖 BBH 高/中/低 SNR + BNS + NSBH + marginal），
每事件拼 H1/L1/V1 × pos/neg = 6 张图（jitter_idx=0）成大 montage。

判断标准：
- BBH pos 列：能看到"向上弯亮线"在中央
- pos vs neg 对比：pos 有 chirp，neg 均匀散斑
- 同事件 H1 vs L1：chirp 形态应一致

输出: output/visual_check_montage.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from config import (
    DETECTORS,
    SPECTROGRAMS_DIR,
    OUTPUT_DIR,
    load_events_from_csv,
)


OUT_PATH = OUTPUT_DIR / "visual_check_montage.png"


SELECTED_EVENT_NAMES = [
    "GW150914",            # BBH 经典强信号 M~28, SNR~24
    "GW190521",            # 大 BBH M~64, SNR~14
    "GW170817",            # BNS 唯一确认 M~1.19, SNR~32
    "GW190425",            # BNS 候选 M~1.44
    "GW200115_042309",     # NSBH M~2.42, SNR~11
    "GW190814",            # 极端质量比 M~6
    "GW190412",            # 不对称 BBH M~13
    "GW200322_091133",     # marginal SNR~4.5
]

KINDS = ["pos", "neg"]
JITTER_IDX = 0  # 每事件取 jitter=0 那张
CELL_SIZE = 256
ROW_HEADER_W = 280
COL_HEADER_H = 36
GAP = 2


def cell_for(event_name: str, ifo: str, kind: str) -> Image.Image:
    """加载 PNG 并缩放成 cell；缺失则灰底 + N/A 字样。"""
    src = SPECTROGRAMS_DIR / f"{event_name}_{ifo}_{kind}_j{JITTER_IDX}.png"
    cell = Image.new("L", (CELL_SIZE, CELL_SIZE), color=60)
    if src.exists():
        img = Image.open(src).convert("L").resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
        cell.paste(img, (0, 0))
        return cell

    draw = ImageDraw.Draw(cell)
    try:
        font = ImageFont.truetype("Arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    text = "N/A"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((CELL_SIZE - tw) // 2, (CELL_SIZE - th) // 2), text, fill=200, font=font)
    return cell


def main() -> None:
    events_by_name = {e["name"]: e for e in load_events_from_csv()}

    rows: list[dict] = []
    missing = []
    for name in SELECTED_EVENT_NAMES:
        if name in events_by_name:
            rows.append(events_by_name[name])
        else:
            missing.append(name)
    if missing:
        print(f"[warn] events.csv 找不到: {missing}")

    n_rows = len(rows)
    n_cols = len(DETECTORS) * len(KINDS)

    width = ROW_HEADER_W + n_cols * CELL_SIZE + (n_cols + 1) * GAP
    height = COL_HEADER_H + n_rows * CELL_SIZE + (n_rows + 1) * GAP
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)

    try:
        font_hdr = ImageFont.truetype("Arial.ttf", 14)
        font_row = ImageFont.truetype("Arial.ttf", 13)
    except OSError:
        font_hdr = ImageFont.load_default()
        font_row = ImageFont.load_default()

    col_labels = []
    for ifo in DETECTORS:
        for kind in KINDS:
            col_labels.append(f"{ifo}-{kind}")
    for j, label in enumerate(col_labels):
        x = ROW_HEADER_W + j * CELL_SIZE + (j + 1) * GAP + CELL_SIZE // 2
        bbox = draw.textbbox((0, 0), label, font=font_hdr)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 10), label, fill=0, font=font_hdr)

    for i, event in enumerate(rows):
        y_top = COL_HEADER_H + i * CELL_SIZE + (i + 1) * GAP
        snr = event.get("snr")
        snr_str = f"SNR={snr:.1f}" if snr is not None else "SNR=n/a"
        header_lines = [
            event["name"],
            f"{event['kind']}  {snr_str}",
            f"M_chirp={event['chirp_mass']:.2f}",
        ]
        if event.get("luminosity_distance") is not None:
            header_lines.append(f"D={event['luminosity_distance']:.0f} Mpc")
        header = "\n".join(header_lines)
        draw.multiline_text((10, y_top + 18), header, fill=0, font=font_row, spacing=4)

        for j, ifo in enumerate(DETECTORS):
            for k, kind in enumerate(KINDS):
                col = j * len(KINDS) + k
                x_left = ROW_HEADER_W + col * CELL_SIZE + (col + 1) * GAP
                cell = cell_for(event["name"], ifo, kind)
                canvas.paste(cell, (x_left, y_top))

    canvas.save(OUT_PATH, format="PNG")
    print(f"Saved: {OUT_PATH}  ({canvas.size[0]}x{canvas.size[1]})")
    print()
    print("=== Selected events ===")
    for e in rows:
        snr = e.get("snr")
        snr_str = f"{snr:.1f}" if snr is not None else "n/a"
        dist = e.get("luminosity_distance")
        dist_str = f"{dist:.0f}Mpc" if dist is not None else "n/a"
        print(f"  {e['name']:22s} {e['kind']:6s} M={e['chirp_mass']:6.2f}  SNR={snr_str:>5s}  D={dist_str}")
    print()
    print("Open with: open", OUT_PATH)


if __name__ == "__main__":
    main()
