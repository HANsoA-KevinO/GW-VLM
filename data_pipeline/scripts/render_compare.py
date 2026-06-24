"""视觉对比实验:对几个事件(强参照 + 漏检弱信号),并排渲染不同 {窗口}×{色图}。
固定 vmin0/vmax25.5(一致),只变窗口长度和 colormap,看哪种能让弱 chirp 显出来。
用法: .venv-render/bin/python data_pipeline/scripts/render_compare.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

from config import load_events_from_csv, RAW_STRAIN_DIR, Q_RANGE, FREQ_RANGE

events = {e["name"]: e for e in load_events_from_csv()}

# (事件, 探测器, 说明)
picks = [
    ("GW150914", "H1", "SNR24 强(参照)"),
    ("GW190413_052954", "H1", "SNR9 弱(漏检 P=0.06)"),
    ("GW190719_215514", "H1", "SNR7.9 弱(漏检 P=0.20)"),
]
# (列名, 窗口秒, colormap)
variants = [
    ("4s gray (current)", 4.0, "gray"),
    ("2s gray", 2.0, "gray"),
    ("4s viridis", 4.0, "viridis"),
    ("2s viridis", 2.0, "viridis"),
]

fig, axes = plt.subplots(len(picks), len(variants),
                         figsize=(3.4 * len(variants), 3.4 * len(picks)))
for i, (name, ifo, desc) in enumerate(picks):
    ev = events.get(name)
    gps = ev["gps"]
    strain = TimeSeries.read(str(RAW_STRAIN_DIR / f"{name}_{ifo}.hdf5"), format="hdf5")
    for j, (vname, window, cmap) in enumerate(variants):
        ax = axes[i][j]
        try:
            qt = strain.q_transform(qrange=Q_RANGE, frange=FREQ_RANGE,
                                    outseg=(gps - window / 2, gps + window / 2))
            ax.imshow(np.asarray(qt.value).T, aspect="auto", origin="lower",
                      extent=[float(qt.xindex[0].value), float(qt.xindex[-1].value),
                              FREQ_RANGE[0], FREQ_RANGE[1]],
                      cmap=cmap, interpolation="nearest", vmin=0, vmax=25.5)
            ax.set_yscale("log")
        except Exception as e:
            ax.text(0.5, 0.5, f"{type(e).__name__}\n{str(e)[:40]}", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(vname, fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{name}_{ifo}\n{desc}", fontsize=8)

fig.suptitle("窗口 × 色图 对比(固定 vmax=25.5)", fontsize=13)
fig.tight_layout()
out = Path("output/render_compare.png")
fig.savefig(out, dpi=110)
print(f"saved {out}")
