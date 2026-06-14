"""
对所有 90 事件 × 3 探测器逐一审计：
  1. raw_strain HDF5 是否存在
  2. strain 内容是否 NaN（NaN 比例 > 5% 视为坏）
  3. strain 时长是否覆盖 ON-source（GPS ± window/2）和 OFF-source（GPS - neg_offset ± window/2）窗口
  4. spectrograms PNG 是否齐全（每事件每探测器期望 5 pos + 5 neg = 10 张）

输出：
  output/audit_report.txt     人类可读详细报告
  output/audit_summary.txt    按问题类型分类汇总
  output/audit_data.json      机器可读，后续修复脚本输入

每条记录的 status 之一：
  OK             ：strain 完整 + 10 张图齐全
  MISSING_STRAIN ：raw_strain 文件不存在（物理或下载失败）
  NAN_STRAIN     ：strain 文件存在但全 NaN（>5% NaN）
  WINDOW_SHORT   ：strain 时长不够覆盖 neg 窗口（脚本配置 bug）
  PARTIAL_IMG    ：strain OK 但 PNG 数量 < 期望（部分 jitter 失败）
  NO_IMG         ：strain OK 但完全没生成 PNG
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from gwpy.timeseries import TimeSeries

from config import (
    DETECTORS,
    OUTPUT_DIR,
    PSD_DURATION,
    RAW_STRAIN_DIR,
    SPECTROGRAMS_DIR,
    WINDOW_SECONDS,
    load_events_from_csv,
)


REPORT_PATH = OUTPUT_DIR / "audit_report.txt"
SUMMARY_PATH = OUTPUT_DIR / "audit_summary.txt"
DATA_PATH = OUTPUT_DIR / "audit_data.json"

EXPECTED_JITTER_COUNT = 5
NAN_THRESHOLD = 0.05  # NaN 比例 > 5% 视为坏数据


def inspect_strain(path: Path, gps: float, neg_offset: float) -> dict:
    """返回 strain 文件诊断信息。"""
    info: dict = {"path_exists": path.exists()}
    if not info["path_exists"]:
        return info

    try:
        ts = TimeSeries.read(path, format="hdf5")
    except Exception as exc:
        info["read_error"] = f"{exc.__class__.__name__}: {exc}"
        return info

    data = ts.value
    n_total = len(data)
    n_nan = int(np.isnan(data).sum())
    nan_ratio = n_nan / n_total if n_total else 1.0
    info.update({
        "duration": float(ts.duration.value),
        "samples": n_total,
        "nan_count": n_nan,
        "nan_ratio": nan_ratio,
        "is_nan_bad": nan_ratio > NAN_THRESHOLD,
        "t0": float(ts.t0.value),
        "t_end": float(ts.t0.value + ts.duration.value),
    })

    half = WINDOW_SECONDS / 2.0
    pos_start = gps - half
    pos_end = gps + half
    neg_start = gps - neg_offset - half
    neg_end = gps - neg_offset + half

    info["covers_pos"] = info["t0"] <= pos_start and pos_end <= info["t_end"]
    info["covers_neg"] = info["t0"] <= neg_start and neg_end <= info["t_end"]
    info["needed_neg_window"] = (neg_start, neg_end)
    return info


def count_pngs(event_name: str, ifo: str) -> dict:
    """统计该 event/ifo 的 PNG 数量。"""
    pos = [SPECTROGRAMS_DIR / f"{event_name}_{ifo}_pos_j{i}.png" for i in range(EXPECTED_JITTER_COUNT)]
    neg = [SPECTROGRAMS_DIR / f"{event_name}_{ifo}_neg_j{i}.png" for i in range(EXPECTED_JITTER_COUNT)]
    return {
        "pos_count": sum(1 for p in pos if p.exists()),
        "neg_count": sum(1 for p in neg if p.exists()),
        "pos_expected": EXPECTED_JITTER_COUNT,
        "neg_expected": EXPECTED_JITTER_COUNT,
    }


def classify(strain_info: dict, png_info: dict) -> str:
    """根据 strain 与 PNG 状态分类。"""
    if not strain_info.get("path_exists"):
        return "MISSING_STRAIN"
    if "read_error" in strain_info:
        return "MISSING_STRAIN"
    if strain_info.get("is_nan_bad"):
        return "NAN_STRAIN"

    pos_ok = png_info["pos_count"] == png_info["pos_expected"]
    neg_ok = png_info["neg_count"] == png_info["neg_expected"]

    if pos_ok and neg_ok:
        return "OK"
    if png_info["pos_count"] == 0 and png_info["neg_count"] == 0:
        return "NO_IMG"
    if not strain_info.get("covers_neg") and pos_ok and png_info["neg_count"] == 0:
        return "WINDOW_SHORT"
    return "PARTIAL_IMG"


def audit() -> tuple[list, dict]:
    events = load_events_from_csv()
    records = []
    by_event: dict[str, dict] = {}

    for ev in events:
        event_name = ev["name"]
        gps = ev["gps"]
        neg_offset = ev["negative_offset"]
        chirp = ev["chirp_mass"]
        kind = ev["kind"]

        event_record = {
            "event_name": event_name,
            "kind": kind,
            "chirp_mass": chirp,
            "negative_offset": neg_offset,
            "gps": gps,
            "detectors": {},
            "summary": {},
        }

        for ifo in DETECTORS:
            strain_info = inspect_strain(
                RAW_STRAIN_DIR / f"{event_name}_{ifo}.hdf5",
                gps, neg_offset,
            )
            png_info = count_pngs(event_name, ifo)
            status = classify(strain_info, png_info)

            event_record["detectors"][ifo] = {
                "status": status,
                "strain": strain_info,
                "pngs": png_info,
            }
            records.append({
                "event_name": event_name,
                "kind": kind,
                "chirp_mass": chirp,
                "ifo": ifo,
                "status": status,
                "pos_count": png_info["pos_count"],
                "neg_count": png_info["neg_count"],
                "nan_ratio": strain_info.get("nan_ratio"),
                "covers_neg": strain_info.get("covers_neg"),
            })

        statuses = [event_record["detectors"][i]["status"] for i in DETECTORS]
        event_record["summary"]["all_ok"] = all(s == "OK" for s in statuses)
        event_record["summary"]["any_ok"] = any(s == "OK" for s in statuses)
        event_record["summary"]["status_counts"] = dict(Counter(statuses))
        by_event[event_name] = event_record

    return records, by_event


def write_report(records: list, by_event: dict) -> None:
    lines = [
        "# GW-VLM 全量事件数据审计报告",
        f"# 事件总数: {len(by_event)}",
        f"# 探测器组合总数: {len(records)}",
        "",
        "## 按事件详细状态（按 chirp_mass 分组）",
        "",
    ]

    sorted_events = sorted(by_event.values(), key=lambda e: (e["kind"], e["chirp_mass"]))

    for ev in sorted_events:
        snr_status_str = " | ".join(
            f"{i}: {ev['detectors'][i]['status']}" for i in DETECTORS
        )
        lines.append(f"{ev['event_name']:24s}  {ev['kind']:6s}  M={ev['chirp_mass']:6.2f}  neg_off={ev['negative_offset']:5.0f}s  | {snr_status_str}")

        for ifo in DETECTORS:
            d = ev["detectors"][ifo]
            if d["status"] == "OK":
                continue
            details = []
            s = d["strain"]
            if not s.get("path_exists"):
                details.append("strain 文件不存在")
            elif "read_error" in s:
                details.append(f"strain 读失败: {s['read_error']}")
            elif s.get("is_nan_bad"):
                details.append(f"NaN 比例 {s['nan_ratio']*100:.1f}%")
            elif not s.get("covers_neg"):
                t0, te = s.get("t0"), s.get("t_end")
                need = s.get("needed_neg_window")
                details.append(
                    f"strain 覆盖 {t0:.0f}-{te:.0f}（{(te or 0)-(t0 or 0):.0f}s），neg 窗口需要 {need[0]:.0f}-{need[1]:.0f}"
                )
            p = d["pngs"]
            details.append(f"pos {p['pos_count']}/{p['pos_expected']}, neg {p['neg_count']}/{p['neg_expected']}")
            lines.append(f"    └ {ifo}: {d['status']}  | {' | '.join(details)}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"详细报告 → {REPORT_PATH}")


def write_summary(records: list, by_event: dict) -> None:
    status_counter = Counter(r["status"] for r in records)
    kind_x_status = defaultdict(lambda: Counter())
    ifo_x_status = defaultdict(lambda: Counter())
    for r in records:
        kind_x_status[r["kind"]][r["status"]] += 1
        ifo_x_status[r["ifo"]][r["status"]] += 1

    fully_ok_events = sum(1 for e in by_event.values() if e["summary"]["all_ok"])
    partly_ok_events = sum(1 for e in by_event.values() if e["summary"]["any_ok"] and not e["summary"]["all_ok"])
    fully_broken_events = sum(1 for e in by_event.values() if not e["summary"]["any_ok"])

    fixable = [
        (r["event_name"], r["ifo"], r["status"], r["nan_ratio"])
        for r in records if r["status"] == "WINDOW_SHORT"
    ]
    nan_cases = [
        (r["event_name"], r["ifo"], f"{r['nan_ratio']*100:.0f}%")
        for r in records if r["status"] == "NAN_STRAIN"
    ]
    missing_cases = [
        (r["event_name"], r["ifo"])
        for r in records if r["status"] == "MISSING_STRAIN"
    ]
    partial_cases = [
        (r["event_name"], r["ifo"], f"{r['pos_count']}+{r['neg_count']}/10")
        for r in records if r["status"] == "PARTIAL_IMG"
    ]

    lines = [
        "# GW-VLM 数据审计汇总",
        "",
        "## 总体状态",
        f"  事件总数: {len(by_event)}",
        f"  探测器组合总数 (event × ifo): {len(records)}",
        f"  完全 OK 的事件（3 探测器都有完整 10 图）: {fully_ok_events}",
        f"  部分 OK 的事件（至少 1 探测器 OK）: {partly_ok_events}",
        f"  完全失败的事件（无任何探测器 OK）: {fully_broken_events}",
        "",
        "## 按 status 分类（探测器组合数）",
        *[f"  {k}: {v}" for k, v in status_counter.most_common()],
        "",
        "## 按 kind × status 分类",
        f"  {'kind':6s}  " + "  ".join(f"{s:>14s}" for s in status_counter.keys()),
    ]
    for kind in sorted(kind_x_status.keys()):
        row = "  ".join(f"{kind_x_status[kind].get(s, 0):>14d}" for s in status_counter.keys())
        lines.append(f"  {kind:6s}  {row}")

    lines += [
        "",
        "## 按 ifo × status 分类",
        f"  {'ifo':4s}  " + "  ".join(f"{s:>14s}" for s in status_counter.keys()),
    ]
    for ifo in DETECTORS:
        row = "  ".join(f"{ifo_x_status[ifo].get(s, 0):>14d}" for s in status_counter.keys())
        lines.append(f"  {ifo:4s}  {row}")

    lines += [
        "",
        f"## 可修复的 WINDOW_SHORT 案例（{len(fixable)} 条；删 strain 重下即可）",
    ]
    for ev, ifo, st, _ in fixable:
        lines.append(f"  - {ev:24s} {ifo}")

    lines += [
        "",
        f"## NAN_STRAIN 案例（{len(nan_cases)} 条；物理事实，无法修复）",
    ]
    for ev, ifo, ratio in nan_cases:
        lines.append(f"  - {ev:24s} {ifo}  (NaN {ratio})")

    lines += [
        "",
        f"## MISSING_STRAIN 案例（{len(missing_cases)} 条；GWOSC 无数据/下载失败）",
    ]
    for ev, ifo in missing_cases:
        lines.append(f"  - {ev:24s} {ifo}")

    lines += [
        "",
        f"## PARTIAL_IMG 案例（{len(partial_cases)} 条；strain OK 但部分 jitter 失败）",
    ]
    for ev, ifo, info in partial_cases:
        lines.append(f"  - {ev:24s} {ifo}  ({info})")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"汇总报告 → {SUMMARY_PATH}")


def write_data(records: list, by_event: dict) -> None:
    DATA_PATH.write_text(json.dumps({"records": records, "by_event": by_event}, indent=2, default=str))
    print(f"机器可读 → {DATA_PATH}")


def main() -> None:
    print("Auditing all events × detectors...")
    records, by_event = audit()
    write_report(records, by_event)
    write_summary(records, by_event)
    write_data(records, by_event)
    print()
    print("=== Quick summary ===")
    status_counter = Counter(r["status"] for r in records)
    for k, v in status_counter.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
