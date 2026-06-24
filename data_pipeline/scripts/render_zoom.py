"""把一个漏检弱事件放大单看,推几个更激进的变体,判断 chirp 到底能不能被渲染出来。"""
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
NAME, IFO = "GW190413_052954", "H1"   # SNR9, 漏检 P=0.06
ev = events[NAME]; gps = ev["gps"]
strain = TimeSeries.read(str(RAW_STRAIN_DIR / f"{NAME}_{IFO}.hdf5"), format="hdf5")

# (说明, 窗口, colormap, vmax, qrange)
variants = [
    ("4s gray vmax25.5 (current)", 4.0, "gray", 25.5, Q_RANGE),
    ("2s gray vmax25.5", 2.0, "gray", 25.5, Q_RANGE),
    ("1s gray vmax25.5", 1.0, "gray", 25.5, Q_RANGE),
    ("2s gray vmax10", 2.0, "gray", 10.0, Q_RANGE),
    ("2s viridis vmax25.5", 2.0, "viridis", 25.5, Q_RANGE),
    ("2s viridis vmax10", 2.0, "viridis", 10.0, Q_RANGE),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for ax, (desc, window, cmap, vmax, qr) in zip(axes, variants):
    try:
        qt = strain.q_transform(qrange=qr, frange=FREQ_RANGE,
                                outseg=(gps - window / 2, gps + window / 2))
        ax.imshow(np.asarray(qt.value).T, aspect="auto", origin="lower",
                  extent=[float(qt.xindex[0].value), float(qt.xindex[-1].value),
                          FREQ_RANGE[0], FREQ_RANGE[1]],
                  cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_yscale("log")
    except Exception as e:
        ax.text(0.5, 0.5, str(e)[:50], ha="center")
    ax.set_title(desc, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{NAME}_{IFO}  SNR9  (model missed, P=0.06)", fontsize=14)
fig.tight_layout()
fig.savefig("output/render_zoom.png", dpi=100)
print("saved output/render_zoom.png  qpeak=%.1f" % float(np.asarray(strain.q_transform(qrange=Q_RANGE, frange=FREQ_RANGE, outseg=(gps-1, gps+1)).value).max()))
