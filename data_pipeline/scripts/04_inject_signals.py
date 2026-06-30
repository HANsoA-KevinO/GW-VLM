"""注入管线 第1步:生成 GR 模板波形(IMRPhenom),按目标 SNR 注入【真实 O3 噪声】。

思路(GW-ML 标准配方,MLGWSC/AResGW):
- 采源参数(分量质量→chirp_mass+q;对齐自旋→chi_eff;各向同性 incl/sky/pol),覆盖各 bin。
- get_td_waveform 生成 hp/hc,按 H1/L1 天线响应+时延投影(一条注入 → H1+L1 两样本,物理一致)。
- host 噪声 = 复用 output/raw_strain/{train事件}_{ifo}.hdf5 的干净段(真 O3,只用 train 事件防泄漏)。
- 目标 SNR:估 host PSD → pycbc.filter.sigma 算最优 SNR → 抽目标网络 SNR(低端过采样)→ 缩放命中;
  标签 distance = 缩放后真实距离。
- 存:注入后的子段(±SUBSEG_HALF 秒,4096Hz,hdf5,格式同 raw_strain)+ injections_manifest.jsonl。

依赖 pycbc+lalsuite → .venv-inject:
  .venv-inject/bin/python data_pipeline/scripts/04_inject_signals.py --n 1200
自检:  --n 4 --test   (固定高 SNR,验证可见+回收)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.utils import iers
iers.conf.auto_download = False          # 用 astropy 自带 IERS-A,禁联网下载(否则 Detector 调用会卡死在 SSL)
from gwpy.timeseries import TimeSeries

from config import RAW_STRAIN_DIR, OUTPUT_DIR, DETECTORS, load_events_from_csv

SAMPLE_RATE = 4096
F_LOWER = 20.0
SUBSEG_HALF = 64.0
APPROX = "IMRPhenomXPHM"
RAW_INJ_DIR = OUTPUT_DIR / "raw_strain_inject"
MANIFEST = OUTPUT_DIR / "injections_manifest.jsonl"


def sample_params(rng, test=False):
    if test:
        m1, m2, chi = 35.0, 30.0, 0.0          # 自检:清晰响 BBH
    else:
        # 均匀采【要预测的目标量】→ 拉平训练分布,逼模型读波形形态而非猜众数(回应
        # new-E2 暴露的众数预测失败)。chirp 覆盖真实 detector-frame 范围(真实上到 131)。
        # 拒采保物理:总质量/主星质量限在真实 detector 上限附近,剔掉高chirp+极端低q的非物理角。
        m1, m2 = 30.0, 25.0
        for _ in range(40):
            chirp = rng.uniform(3.0, 135.0)
            q = rng.uniform(0.15, 1.0)          # m2/m1,均匀
            eta = q / (1.0 + q) ** 2            # 对称质量比
            mtot = chirp / eta ** 0.6           # detector-frame 总质量
            c1, c2 = mtot / (1.0 + q), mtot * q / (1.0 + q)
            if mtot <= 400.0 and c1 <= 200.0:
                m1, m2 = c1, c2; break
        chi = rng.uniform(-0.6, 0.6)            # chi_eff 直接均匀(覆盖真实[-0.29,0.68])
    s1z = s2z = float(chi)                       # 对齐自旋=chi_eff → chi_eff 精确均匀
    return dict(mass1=float(m1), mass2=float(m2), spin1z=float(s1z), spin2z=float(s2z),
                chi_eff=float((m1 * s1z + m2 * s2z) / (m1 + m2)),
                chirp_mass=float((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2),
                total_mass=float(m1 + m2), q=float(m2 / m1),
                inclination=float(np.arccos(rng.uniform(-1, 1))),
                coa_phase=float(rng.uniform(0, 2 * np.pi)),
                polarization=float(rng.uniform(0, 2 * np.pi)),
                ra=float(rng.uniform(0, 2 * np.pi)),
                dec=float(np.arcsin(rng.uniform(-1, 1))), stream="flat")


def sample_target_snr(rng):
    return float(4.0 + 16.0 * rng.beta(1.5, 3.0))   # [4,20],低端偏多


def train_hosts():
    train_events = set()
    for line in open(OUTPUT_DIR / "dataset_train.jsonl"):
        r = json.loads(line)
        if str(r.get("source_type", "")).startswith("real"):
            train_events.add(r["event_name"])
    evmap = {e["name"]: e for e in load_events_from_csv()}
    return [evmap[n] for n in sorted(train_events)
            if n in evmap and all((RAW_STRAIN_DIR / f"{n}_{ifo}.hdf5").exists() for ifo in DETECTORS)]


def injbg_hosts():
    """噪声池 role==injbg 段当注入背景 host(伪 host dict)。"""
    from config import NOISE_POOL_MANIFEST, RAW_NOISE_DIR
    hosts = []
    if not NOISE_POOL_MANIFEST.exists():
        return hosts
    for line in open(NOISE_POOL_MANIFEST):
        r = json.loads(line)
        if r["role"] != "injbg":
            continue
        if all((RAW_NOISE_DIR / f"noise_{r['segid']}_{ifo}.hdf5").exists() for ifo in DETECTORS):
            hosts.append({"name": f"noise_{r['segid']}", "is_noise": True, "segid": r["segid"],
                          "gps": r["gps_end"] + 1e6, "negative_offset": 0.0})  # sentinel:窗口按全段算
    return hosts


def host_strain_path(host, ifo):
    """按 host 类型返回 strain 文件路径(噪声 host 在 raw_noise/,真实事件在 raw_strain/)。"""
    from config import RAW_NOISE_DIR
    if host.get("is_noise"):
        return RAW_NOISE_DIR / f"noise_{host['segid']}_{ifo}.hdf5"
    return RAW_STRAIN_DIR / f"{host['name']}_{ifo}.hdf5"


_MSUN_S = 4.925491e-6   # G·Msun/c³ [秒]


def f_lower_for(mc_solar, dur_s=14.0):
    """选 f_lower 使牛顿啁啾时长≈dur_s(避免轻质量 BNS 从 20Hz 生成超长波形拖死)。
    重 BBH → <20Hz → 取 20;轻 BNS → 抬高 f_lower 把时长压到 ~14s(merger 居中,可见部分够)。"""
    mc = mc_solar * _MSUN_S
    pif = (5.0 / (256.0 * dur_s) * mc ** (-5.0 / 3.0)) ** (3.0 / 8.0)
    return max(F_LOWER, float(pif / np.pi))


def project(p, det, t_gps):
    """投影到探测器的波形(pycbc TS,merger 在自身 epoch 的 t=0;distance=1000Mpc 占位)。"""
    from pycbc.waveform import get_td_waveform
    from pycbc.detector import Detector
    flo = f_lower_for(p["chirp_mass"])
    hp, hc = get_td_waveform(approximant=APPROX, mass1=p["mass1"], mass2=p["mass2"],
                             spin1z=p["spin1z"], spin2z=p["spin2z"],
                             inclination=p["inclination"], coa_phase=p["coa_phase"],
                             distance=1000.0, delta_t=1.0 / SAMPLE_RATE, f_lower=flo)
    fp, fc = Detector(det).antenna_pattern(p["ra"], p["dec"], p["polarization"], t_gps)
    return (fp * hp + fc * hc), flo


def add_waveform(noise_np, noise_t0, dt, h_pycbc, t_merger):
    """把 h(merger 对齐到 t_merger)按时间索引加进 noise_np(就地返回新数组)。"""
    hs = float(h_pycbc.start_time) + t_merger          # h 平移:merger(t=0)→ t_merger
    hv = np.asarray(h_pycbc.numpy(), dtype=np.float64)
    i0 = int(round((hs - noise_t0) / dt))
    a = noise_np.copy()
    lo = max(0, i0); hi = min(len(a), i0 + len(hv))
    if hi > lo:
        a[lo:hi] += hv[(lo - i0):(lo - i0) + (hi - lo)]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    from pycbc.filter import sigma
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    from pycbc.detector import Detector

    RAW_INJ_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    hosts = train_hosts() + injbg_hosts()   # 真实事件 off-source + 噪声池 injbg 段
    if not hosts:
        raise RuntimeError("无可用 host(train 事件 raw_strain 或 噪声池 injbg)")
    n_noise = sum(1 for h in hosts if h.get("is_noise"))
    print(f"[inject] host {len(hosts)}(其中噪声池 {n_noise}) | 目标 {args.n} 注入 × {len(DETECTORS)} | approx={APPROX}", flush=True)

    mf = open(MANIFEST, "w"); ok = 0
    skip = {"window": 0, "snr": 0, "distance": 0, "error": 0}
    host_cache = {}   # 缓存 host 整段读取(同 host 复用,省重复 IO)
    for i in range(args.n):
        p = sample_params(rng, test=args.test)
        host = hosts[rng.integers(len(hosts))]
        gps, neg = host["gps"], host["negative_offset"]
        try:
            if (host["name"], "H1") not in host_cache:
                host_cache[(host["name"], "H1")] = TimeSeries.read(host_strain_path(host, "H1"), format="hdf5")
            h1 = host_cache[(host["name"], "H1")]
        except Exception:
            skip["error"] += 1; continue
        t0, tend = h1.t0.value, h1.t0.value + h1.duration.value
        if host.get("is_noise"):   # 噪声 host:整段(已 event-veto、远离任何事件),用全段窗口
            lo, hi = t0 + SUBSEG_HALF + 8, tend - SUBSEG_HALF - 8
        else:
            lo = t0 + SUBSEG_HALF + 8; hi = min(gps - neg - 30, tend - SUBSEG_HALF - 8)
        if hi <= lo:
            skip["window"] += 1; continue
        inj_center = float(rng.uniform(lo, hi))

        # 每探测器:子段 + PSD + 投影波形 + 最优 SNR
        per = {}; net_sq = 0.0
        try:
            for ifo in DETECTORS:
                key = (host["name"], ifo)
                if key not in host_cache:
                    host_cache[key] = TimeSeries.read(host_strain_path(host, ifo), format="hdf5")
                sub = host_cache[key].crop(inj_center - SUBSEG_HALF, inj_center + SUBSEG_HALF).to_pycbc()
                psd = inverse_spectrum_truncation(
                    interpolate(sub.psd(4), sub.delta_f),
                    int(4 * sub.sample_rate), low_frequency_cutoff=F_LOWER)
                h, _ = project(p, ifo, inj_center)
                hh = h.copy(); hh.resize(len(sub))
                opt = float(sigma(hh, psd=psd, low_frequency_cutoff=F_LOWER))
                per[ifo] = (sub, h, opt); net_sq += opt ** 2
        except Exception as e:
            skip["error"] += 1; continue
        net_opt = net_sq ** 0.5
        if not np.isfinite(net_opt) or net_opt < 1e-3:
            skip["snr"] += 1; continue
        target = 30.0 if args.test else sample_target_snr(rng)
        scale = target / net_opt
        p["distance"] = float(1000.0 / scale); p["snr"] = float(target)
        # 物理距离守卫:低质量源要达目标 SNR 需极近 → 距离不真实就跳过(避免垃圾标签)
        if not (5.0 <= p["distance"] <= 12000.0):
            skip["distance"] += 1; continue

        iid = f"{i:05d}"
        for ifo, (sub, h, opt) in per.items():
            dt_geo = Detector(ifo).time_delay_from_earth_center(p["ra"], p["dec"], inj_center)
            noise_np = np.asarray(sub.numpy(), dtype=np.float64)
            inj_np = add_waveform(noise_np, float(sub.start_time), float(sub.delta_t),
                                  h * scale, inj_center + dt_geo)
            g = TimeSeries(inj_np, t0=float(sub.start_time), sample_rate=SAMPLE_RATE,
                           name=f"inject_{iid}_{ifo}")
            g.write(RAW_INJ_DIR / f"inject_{iid}_{ifo}.hdf5", format="hdf5", overwrite=True)
        mf.write(json.dumps({"id": iid, "host": host["name"], "inj_center": inj_center,
                             "net_opt_snr_1000Mpc": net_opt, **p}, ensure_ascii=False) + "\n")
        mf.flush(); ok += 1
        if ok % 50 == 0 or args.test:
            print(f"  [{ok}] id{iid} {host['name']} mc={p['chirp_mass']:.1f} d={p['distance']:.0f} "
                  f"chi={p['chi_eff']:.2f} snr={p['snr']:.1f}", flush=True)
    mf.close()
    print(f"[inject] 完成 {ok} → {RAW_INJ_DIR} ; manifest {MANIFEST} | 跳过 {skip}", flush=True)


if __name__ == "__main__":
    main()
