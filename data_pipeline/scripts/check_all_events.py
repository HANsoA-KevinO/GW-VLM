"""
把所有 90 事件 × 6 列（H1/L1/V1 × pos/neg）拼成大 montage。
每事件取 jitter_idx=0 的图作为代表。
按 chirp_mass 升序排列（BNS → BBH 大质量 → marginal）。

按 PAGE_SIZE 事件一页分多个 PNG 输出，避免单 PNG 过大。

输出: output/all_events_page{N}.png
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


PAGE_SIZE = 30  # 每页最多 30 事件
CELL_SIZE = 240
ROW_HEADER_W = 260
COL_HEADER_H = 32
GAP = 2
JITTER_IDX = 0

KINDS = ["pos", "neg"]


def cell_for(event_name: str, ifo: str, kind: str) -> Image.Image:
    src = SPECTROGRAMS_DIR / f"{event_name}_{ifo}_{kind}_j{JITTER_IDX}.png"
    cell = Image.new("L", (CELL_SIZE, CELL_SIZE), color=60)
    if src.exists():
        img = Image.open(src).convert("L").resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
        cell.paste(img, (0, 0))
        return cell

    draw = ImageDraw.Draw(cell)
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    text = "N/A"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((CELL_SIZE - tw) // 2, (CELL_SIZE - th) // 2), text, fill=200, font=font)
    return cell


def render_page(events_in_page: list[dict], page_num: int, total_pages: int) -> Path:
    n_rows = len(events_in_page)
    n_cols = len(DETECTORS) * len(KINDS)

    width = ROW_HEADER_W + n_cols * CELL_SIZE + (n_cols + 1) * GAP
    height = COL_HEADER_H + n_rows * CELL_SIZE + (n_rows + 1) * GAP

    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)

    try:
        font_hdr = ImageFont.truetype("Arial.ttf", 14)
        font_row = ImageFont.truetype("Arial.ttf", 12)
        font_page = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font_hdr = ImageFont.load_default()
        font_row = ImageFont.load_default()
        font_page = ImageFont.load_default()

    col_labels = []
    for ifo in DETECTORS:
        for kind in KINDS:
            col_labels.append(f"{ifo}-{kind}")
    for j, label in enumerate(col_labels):
        x = ROW_HEADER_W + j * CELL_SIZE + (j + 1) * GAP + CELL_SIZE // 2
        bbox = draw.textbbox((0, 0), label, font=font_hdr)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 8), label, fill=0, font=font_hdr)

    draw.text((10, 8), f"Page {page_num}/{total_pages}", fill=0, font=font_page)

    for i, event in enumerate(events_in_page):
        y_top = COL_HEADER_H + i * CELL_SIZE + (i + 1) * GAP
        snr = event.get("snr")
        snr_str = f"SNR={snr:.1f}" if snr is not None else "SNR=n/a"
        dist = event.get("luminosity_distance")
        dist_str = f"D={dist:.0f}Mpc" if dist is not None else "D=n/a"
        header_lines = [
            event["name"],
            f"{event['kind']}",
            f"M={event['chirp_mass']:.2f}",
            f"{snr_str}",
            f"{dist_str}",
        ]
        header = "\n".join(header_lines)
        draw.multiline_text((8, y_top + 12), header, fill=0, font=font_row, spacing=2)

        for j, ifo in enumerate(DETECTORS):
            for k, kind in enumerate(KINDS):
                col = j * len(KINDS) + k
                x_left = ROW_HEADER_W + col * CELL_SIZE + (col + 1) * GAP
                cell = cell_for(event["name"], ifo, kind)
                canvas.paste(cell, (x_left, y_top))

    out = OUTPUT_DIR / f"all_events_page{page_num}.png"
    canvas.save(out, format="PNG")
    return out


def main() -> None:
    events = load_events_from_csv()
    # 按 (kind 顺序, chirp_mass) 排序，让 BNS / NSBH / BBH 分组聚集
    kind_order = {"BNS": 0, "NSBH": 1, "BBH": 2}
    events.sort(key=lambda e: (kind_order.get(e["kind"], 9), e["chirp_mass"]))

    total = len(events)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"Total events: {total}, page size: {PAGE_SIZE}, total pages: {total_pages}")

    for page_num in range(1, total_pages + 1):
        start = (page_num - 1) * PAGE_SIZE
        events_in_page = events[start:start + PAGE_SIZE]
        out = render_page(events_in_page, page_num, total_pages)
        sample_first = events_in_page[0]["name"]
        sample_last = events_in_page[-1]["name"]
        print(f"  Page {page_num}: {len(events_in_page)} events ({sample_first} ~ {sample_last}) → {out}  ({out.stat().st_size // 1024} KB)")

    print()
    print("Open all pages:")
    for n in range(1, total_pages + 1):
        print(f"  open {OUTPUT_DIR / f'all_events_page{n}.png'}")


if __name__ == "__main__":
    main()
