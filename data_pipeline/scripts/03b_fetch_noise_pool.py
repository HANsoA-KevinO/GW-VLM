"""注入管线 第0步:从 GWOSC 拉【干净的真实 O3 噪声池】,GPS 冻结分成两个互斥子池:
  role=neg   → 用来造更多检测负样本(每段切多个 4s 窗)
  role=injbg → 用来当注入背景(每段注入信号)
两池按 GPS 不重叠 → 一段噪声绝不会既当负样本又当注入背景(用户的硬约束)。

干净度:DQ 旗标(CBC_CAT2 + 无硬件注入,H1∩L1 同时段)→ 减去事件 veto(GWTC 全部事件
+ 我们 events.csv 的 90 个,各 ±180s)→ 留干净段。合成数据全进 train(07),val/test 仍全真实。

依赖 gwpy+gwosc → .venv-render:
  .venv-render/bin/python data_pipeline/scripts/03b_fetch_noise_pool.py --n-injbg 85 --n-neg 125
自检:  --n-injbg 1 --n-neg 1
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.utils import iers
iers.conf.auto_download = False
from gwpy.segments import DataQualityFlag, Segment, SegmentList
from gwpy.timeseries import TimeSeries
from gwosc import datasets

from config import (RAW_NOISE_DIR, NOISE_POOL_MANIFEST, O3A_RANGE, O3B_RANGE, NOISE_DQ_FLAGS,
                    NOISE_EVENT_GUARD, NOISE_NEG_MIN_DUR, NOISE_INJBG_MIN_DUR, SAMPLE_RATE,
                    DETECTORS, load_events_from_csv)

HOST = "https://gwosc.org"
DAY = 86400


def day_clean(t, d1, veto):
    """一天内的干净 H1∩L1 同时段(DQ ∩),减去事件 veto。失败返回空。"""
    try:
        acts = [DataQualityFlag.fetch_open_data(f, t, d1, host=HOST).active for f in NOISE_DQ_FLAGS]
        inter = acts[0]
        for a in acts[1:]:
            inter = inter & a
        return (inter.coalesce() - veto).coalesce()
    except Exception as e:
        print(f"  [warn] {int(t)}-{int(d1)} DQ 失败: {e.__class__.__name__}", flush=True)
        return SegmentList()


def scan_candidates(veto, need, rng):
    """增量逐天扫 O3a→O3b,收够 need 个候选干净段(≥NEG_MIN)就停。"""
    cand = []
    for nm, (lo, hi) in [("O3a", O3A_RANGE), ("O3b", O3B_RANGE)]:
        t = lo
        while t < hi:
            d1 = min(t + DAY, hi)
            for s in day_clean(t, d1, veto):
                if float(s[1] - s[0]) >= NOISE_NEG_MIN_DUR:
                    cand.append(s)
            t = d1
            if len(cand) >= need:
                print(f"  扫到 {nm} {int(t)}:候选 {len(cand)} 段(够了)", flush=True)
                return cand
        print(f"  {nm} 扫完:候选累计 {len(cand)}", flush=True)
    return cand


def event_veto():
    """GWTC 全部事件 + 我们 events.csv 的 GPS,各 ±GUARD,合并成 veto SegmentList。"""
    gps = []
    try:
        for name in datasets.find_datasets(type="event"):
            try:
                gps.append(float(datasets.event_gps(name)))
            except Exception:
                pass
    except Exception as e:
        print(f"  [warn] GWTC 事件列表拉取失败,仅用 events.csv: {e}", flush=True)
    for ev in load_events_from_csv():
        gps.append(float(ev["gps"]))
    veto = SegmentList([Segment(t - NOISE_EVENT_GUARD, t + NOISE_EVENT_GUARD) for t in gps]).coalesce()
    print(f"  事件 veto: {len(gps)} 个 GPS → {len(veto)} 段", flush=True)
    return veto


def fetch_seg(seg, ifo):
    """拉一段某探测器 strain(4096Hz),NaN 检查。失败/含NaN 返回 None。"""
    s0, s1 = float(seg[0]), float(seg[1])
    try:
        ts = TimeSeries.fetch_open_data(ifo, s0, s1, sample_rate=SAMPLE_RATE, cache=True, host=HOST)
        if not np.isfinite(ts.value).all():
            return None
        return ts
    except Exception as e:
        print(f"  [warn] fetch {ifo} {int(s0)} 失败: {e.__class__.__name__}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-injbg", type=int, default=85, help="注入背景段数(每段≥184s)")
    ap.add_argument("--n-neg", type=int, default=125, help="负样本段数(每段≥24s)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    RAW_NOISE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    veto = event_veto()
    need = (args.n_injbg + args.n_neg) * 2   # 2x 余量(部分太短/拉取失败)
    print(f"[03b] 增量扫干净段(目标候选 {need})...", flush=True)
    segs = scan_candidates(veto, need, rng)
    print(f"[03b] 候选干净段 {len(segs)} 段(≥{NOISE_NEG_MIN_DUR:.0f}s)", flush=True)
    rng.shuffle(segs)
    injbg_pool = [s for s in segs if float(s[1] - s[0]) >= NOISE_INJBG_MIN_DUR]
    chosen = {}   # segid -> (seg, role)
    # 先选 injbg(从够长的里取前 n-injbg)
    for s in injbg_pool[:args.n_injbg]:
        segid = f"{int(s[0])}_{int(s[1]-s[0])}"
        chosen[segid] = (s, "injbg")
    # 再选 neg(从剩下的里取,任何 ≥24s 的)
    used = set(chosen)
    n_neg = 0
    for s in segs:
        segid = f"{int(s[0])}_{int(s[1]-s[0])}"
        if segid in used:
            continue
        chosen[segid] = (s, "neg"); used.add(segid); n_neg += 1
        if n_neg >= args.n_neg:
            break
    n_injbg = sum(1 for _, r in chosen.values() if r == "injbg")
    print(f"[03b] 选定 injbg {n_injbg} 段 + neg {n_neg} 段;开始拉 strain...", flush=True)

    # 断言:两池 GPS 互斥(段本就 coalesce 不重叠 → 必然;再保险查一遍)
    negsl = SegmentList([s for s, r in chosen.values() if r == "neg"]).coalesce()
    injsl = SegmentList([s for s, r in chosen.values() if r == "injbg"]).coalesce()
    assert len(negsl & injsl) == 0, "neg/injbg GPS 重叠!"

    mf = open(NOISE_POOL_MANIFEST, "w"); ok = 0
    items = [(segid, seg, role, ifo) for segid, (seg, role) in chosen.items() for ifo in DETECTORS]
    saved = {}
    def work(item):
        segid, seg, role, ifo = item
        out = RAW_NOISE_DIR / f"noise_{segid}_{ifo}.hdf5"
        if out.exists():
            return (segid, role, True)
        ts = fetch_seg(seg, ifo)
        if ts is None:
            return (segid, role, False)
        ts.write(out, format="hdf5", overwrite=True)
        return (segid, role, True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for segid, role, good_ in ex.map(work, items):
            saved.setdefault(segid, {"role": role, "ifos": []})
            if good_:
                saved[segid]["ifos"].append(True)
    # 只保留 H1+L1 都成功的段
    for segid, (seg, role) in chosen.items():
        info = saved.get(segid, {"ifos": []})
        if sum(info["ifos"]) == len(DETECTORS):
            mf.write(json.dumps({"segid": segid, "gps_start": float(seg[0]), "gps_end": float(seg[1]),
                                 "dur": float(seg[1]-seg[0]), "role": role,
                                 "ifos": DETECTORS}, ensure_ascii=False) + "\n")
            ok += 1
    mf.close()
    print(f"[03b] 完成:H1+L1 齐全的段 {ok} → {RAW_NOISE_DIR} ; manifest {NOISE_POOL_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
