"""注入管线:从噪声池的 role==neg 段渲染【检测负样本】(图 + 应变),manifest 驱动。

为什么单独写(不复用 09):09 用 dataset.jsonl 过滤输出,放 06 前会出 0 个 npy(对抗审查 bug#2)。
这里直接从 noise_pool_manifest 驱动,同一循环里出 PNG + npy(窗口中心共用,天然对齐)。
glitch 屏蔽:白化窗口 |max| 超阈值视为 glitch 丢弃(挡未编目 glitch,审查 bug#5)。

命名:noise_{segid}_{ifo}_neg_w{ii}.png / .npy(06 用 NOISE_PAT 归为 source_type=noise_neg)。
依赖 gwpy+matplotlib+pillow → .venv-render:
  .venv-render/bin/python data_pipeline/scripts/05b_render_noise_negatives.py --target 3000
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.utils import iers
iers.conf.auto_download = False
from gwpy.timeseries import TimeSeries

from config import (BANDPASS_LOW, BANDPASS_HIGH, WINDOW_SECONDS, OUTPUT_DIR, RAW_NOISE_DIR,
                    NOISE_POOL_MANIFEST, SPECTROGRAMS_DIR, DETECTORS, NOISE_SEG_PAD, NOISE_GLITCH_MAX)

_spec = importlib.util.spec_from_file_location(
    "gen02", Path(__file__).resolve().parent / "02_generate_spectrograms.py")
gen02 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gen02)
render_qtransform = gen02.render_qtransform

SPECTRO_VIRIDIS = OUTPUT_DIR / "spectrograms_viridis"
STRAIN_ARRAYS = OUTPUT_DIR / "strain_arrays"
TARGET_SR, N_SAMPLES = 2048, int(WINDOW_SECONDS * 2048)
WHITEN_FFTLEN, WHITEN_OVERLAP = 4.0, 2.0


def fix_length(a, n):
    return a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))


def centers_for(segid, ifo, t0, tend, n):
    """seed=(segid,ifo) 的 n 个窗口中心,严格落在白化腐蚀区内部。"""
    rng = np.random.default_rng(abs(hash((segid, ifo))) % (2**32))
    half = WINDOW_SECONDS / 2
    lo, hi = t0 + NOISE_SEG_PAD + half, tend - NOISE_SEG_PAD - half
    if hi <= lo:
        return []
    return [float(rng.uniform(lo, hi)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000, help="目标负样本总数(配平注入数)")
    args = ap.parse_args()
    SPECTRO_VIRIDIS.mkdir(parents=True, exist_ok=True); STRAIN_ARRAYS.mkdir(parents=True, exist_ok=True)

    neg_segs = [r for r in (json.loads(l) for l in open(NOISE_POOL_MANIFEST)) if r["role"] == "neg"]
    if not neg_segs:
        print("[render-neg] manifest 无 role==neg 段"); return
    per = max(1, args.target // (len(neg_segs) * len(DETECTORS)) + 1)  # 每段每探测器窗口数
    print(f"[render-neg] neg 段 {len(neg_segs)} | 每段每探测器 {per} 窗 | 目标 ~{args.target}", flush=True)

    half = WINDOW_SECONDS / 2
    ok = glitch = fail = 0
    for r in neg_segs:
        segid = r["segid"]
        for ifo in DETECTORS:
            hdf5 = RAW_NOISE_DIR / f"noise_{segid}_{ifo}.hdf5"
            if not hdf5.exists():
                continue
            try:
                raw = TimeSeries.read(hdf5, format="hdf5")
                white = raw.whiten(WHITEN_FFTLEN, WHITEN_OVERLAP).bandpass(BANDPASS_LOW, BANDPASS_HIGH).resample(TARGET_SR)
                t0, tend = raw.t0.value, raw.t0.value + raw.duration.value
                for ii, c in enumerate(centers_for(segid, ifo, t0, tend, per)):
                    if ok >= args.target:
                        break
                    arr = fix_length(np.asarray(white.crop(c - half, c + half).value, dtype=np.float32), N_SAMPLES)
                    if float(np.max(np.abs(arr))) > NOISE_GLITCH_MAX:   # glitch 屏蔽
                        glitch += 1; continue
                    stem = f"noise_{segid}_{ifo}_neg_w{ii:02d}"
                    render_qtransform(raw, c, SPECTRO_VIRIDIS / f"{stem}.png", cmap="viridis")
                    np.save(STRAIN_ARRAYS / f"{stem}.npy", arr)
                    ok += 1
            except Exception as e:
                print(f"  [fail] noise_{segid}_{ifo}: {e.__class__.__name__}: {e}"); fail += 1
        if ok >= args.target:
            break
        if ok % 200 < per:
            print(f"  进度 {ok}/{args.target}", flush=True)
    print(f"[render-neg] 完成 {ok} 负样本(丢 glitch {glitch},失败 {fail})→ {SPECTRO_VIRIDIS}", flush=True)


if __name__ == "__main__":
    main()
