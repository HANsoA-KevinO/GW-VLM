"""注入管线 第2步:对每个注入子段,渲染 viridis 频谱图 + 提白化应变 + 写 sidecar。

- 图:复用 02 的 render_qtransform(完全相同的 q_transform/vmin/vmax/尺寸/色图)→ 无"合成 vs 真实"捷径。
- 应变:复用 09 的 白化→带通→降采样→裁 4s→8192 链路。
- sidecar(供 06 的 build_inject_label + 08 unified targets):chirp_mass / luminosity_distance / chi_eff / snr。

输出:
  output/spectrograms_viridis/inject_{id}_{ifo}.png   (+ 同名 .json sidecar)
  output/strain_arrays/inject_{id}_{ifo}.npy

依赖 gwpy+matplotlib+pillow → .venv-inject:
  .venv-inject/bin/python data_pipeline/scripts/05_render_injections.py
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from gwpy.timeseries import TimeSeries

from config import (BANDPASS_LOW, BANDPASS_HIGH, WINDOW_SECONDS, OUTPUT_DIR)

# 复用 02 的 render_qtransform(文件名以数字开头,用 importlib 按路径加载)
_spec = importlib.util.spec_from_file_location(
    "gen02", Path(__file__).resolve().parent / "02_generate_spectrograms.py")
gen02 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gen02)
render_qtransform = gen02.render_qtransform

RAW_INJ_DIR = OUTPUT_DIR / "raw_strain_inject"
MANIFEST = OUTPUT_DIR / "injections_manifest.jsonl"
SPECTRO_VIRIDIS = OUTPUT_DIR / "spectrograms_viridis"
STRAIN_ARRAYS = OUTPUT_DIR / "strain_arrays"
TARGET_SR, N_SAMPLES = 2048, int(WINDOW_SECONDS * 2048)
WHITEN_FFTLEN, WHITEN_OVERLAP = 4.0, 2.0


def fix_length(arr, n):
    return arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)))


def main():
    SPECTRO_VIRIDIS.mkdir(parents=True, exist_ok=True)
    STRAIN_ARRAYS.mkdir(parents=True, exist_ok=True)
    man = {r["id"]: r for r in (json.loads(l) for l in open(MANIFEST))}
    print(f"[render-inj] manifest {len(man)} 注入 | 子段目录 {RAW_INJ_DIR}", flush=True)

    ok = fail = 0
    for hdf5 in sorted(RAW_INJ_DIR.glob("inject_*.hdf5")):
        stem = hdf5.stem                       # inject_{id}_{ifo}
        iid, ifo = stem.split("_")[1], stem.split("_")[2]
        rec = man.get(iid)
        if rec is None:
            continue
        center = rec["inj_center"]
        try:
            ts = TimeSeries.read(hdf5, format="hdf5")
            # 图(viridis,与真实同管线)
            render_qtransform(ts, center, SPECTRO_VIRIDIS / f"{stem}.png", cmap="viridis")
            # 应变(与 09 同链路)
            w = ts.whiten(WHITEN_FFTLEN, WHITEN_OVERLAP).bandpass(BANDPASS_LOW, BANDPASS_HIGH).resample(TARGET_SR)
            seg = w.crop(center - WINDOW_SECONDS / 2, center + WINDOW_SECONDS / 2)
            np.save(STRAIN_ARRAYS / f"{stem}.npy",
                    fix_length(np.asarray(seg.value, dtype=np.float32), N_SAMPLES))
            # sidecar(06 读)。注入无红移→采样质量即 detector-frame;直接当 _det 目标。
            # total_mass 旧 manifest 可能没有 → 用 mass1+mass2 补(向后兼容)。
            rec_tm = rec.get("total_mass", rec["mass1"] + rec["mass2"])
            json.dump({"chirp_mass_det": rec["chirp_mass"], "total_mass_det": rec_tm,
                       "chi_eff": rec["chi_eff"], "snr": rec["snr"],
                       "luminosity_distance": rec["distance"], "mass1": rec["mass1"],
                       "mass2": rec["mass2"], "approx": "IMRPhenomXPHM"},
                      open(SPECTRO_VIRIDIS / f"{stem}.json", "w"))
            ok += 1
            if ok % 100 == 0:
                print(f"  [{ok}] {stem}", flush=True)
        except Exception as e:
            print(f"  [fail] {stem}: {e.__class__.__name__}: {e}"); fail += 1
    print(f"[render-inj] 完成 {ok} 失败 {fail} → 图 {SPECTRO_VIRIDIS} / 应变 {STRAIN_ARRAYS}", flush=True)


if __name__ == "__main__":
    main()
