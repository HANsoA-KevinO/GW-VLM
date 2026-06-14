"""
Stage 0 Step 4: 把 24-30 张 PNG 拼成一张 montage。

行 = 事件（5 行）
列 = [H1-pos, H1-neg, L1-pos, L1-neg, V1-pos, V1-neg]（6 列）
缺失的 cell 留白并标 "N/A"。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from config import (
    DETECTORS,
    EVENTS,
    MONTAGE_CELL_SIZE,
    MONTAGE_PATH,
    SPECTROGRAMS_DIR,
)


KINDS = ["pos", "neg"]
LABEL_HEIGHT = 28
ROW_HEADER_WIDTH = 180


def cell_for(event_name: str, ifo: str, kind: str) -> Image.Image:
    """加载一个 PNG 并缩到 cell 大小；缺失则返回带 N/A 字样的灰底。"""
    src = SPECTROGRAMS_DIR / f"{event_name}_{ifo}_{kind}.png"
    cell = Image.new("L", (MONTAGE_CELL_SIZE, MONTAGE_CELL_SIZE), color=80)
    if src.exists():
        img = Image.open(src).convert("L").resize((MONTAGE_CELL_SIZE, MONTAGE_CELL_SIZE), Image.LANCZOS)
        cell.paste(img, (0, 0))
    else:
        draw = ImageDraw.Draw(cell)
        text = "N/A"
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((MONTAGE_CELL_SIZE - tw) // 2, (MONTAGE_CELL_SIZE - th) // 2), text, fill=200, font=font)
    return cell


def make_montage() -> None:
    n_rows = len(EVENTS)
    n_cols = len(DETECTORS) * len(KINDS)

    width = ROW_HEADER_WIDTH + n_cols * MONTAGE_CELL_SIZE
    height = LABEL_HEIGHT + n_rows * MONTAGE_CELL_SIZE

    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    col_labels = []
    for ifo in DETECTORS:
        for kind in KINDS:
            col_labels.append(f"{ifo}-{kind}")

    for j, label in enumerate(col_labels):
        x = ROW_HEADER_WIDTH + j * MONTAGE_CELL_SIZE + MONTAGE_CELL_SIZE // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 4), label, fill=0, font=font)

    for i, event in enumerate(EVENTS):
        y_top = LABEL_HEIGHT + i * MONTAGE_CELL_SIZE
        header = f"{event['name']}\n{event['kind']}  SNR~{event['snr']}"
        draw.multiline_text((6, y_top + 8), header, fill=0, font=font, spacing=4)
        for j, ifo in enumerate(DETECTORS):
            for k, kind in enumerate(KINDS):
                col = j * len(KINDS) + k
                x_left = ROW_HEADER_WIDTH + col * MONTAGE_CELL_SIZE
                cell = cell_for(event["name"], ifo, kind)
                canvas.paste(cell, (x_left, y_top))

    canvas.save(MONTAGE_PATH, format="PNG")
    print(f"Saved montage → {MONTAGE_PATH}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    make_montage()
