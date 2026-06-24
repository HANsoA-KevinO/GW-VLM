"""
生成 E1 结果看板（单文件自包含 HTML）。
所有图表由**实际结果数据**在浏览器里用原生 SVG 渲染（非嵌入 PNG 截图）：
  · 训练/验证损失曲线  ← trainer_state.json
  · ROC 曲线叠加 + AUC 对比  ← per_sample.jsonl + prob_report.json
  · 判决阈值扫描（recall/precision vs 阈值）  ← per_sample.jsonl
  · 混淆矩阵  ← prob_report.json operating_points
  · 检测效率 vs SNR（分档 recall）  ← per_sample.jsonl + dataset_test.jsonl

用法：python3 reports/build_dashboard.py   → reports/e1_dashboard.html
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "output" / "runs"
OUT = Path(__file__).resolve().parent / "e1_dashboard.html"

MODELS = [
    {"key": "qwen", "name": "Qwen3.6-27B", "family": "Qwen · 原生多模态",
     "tag": "viridis · 2 epoch · 原生分辨率", "rank": 1, "best": True, "color": "#21918c",
     "dir": RUNS / "e1_qwen36_27b_viridis_3ep" / "checkpoint-600",
     "note": "E1 最佳。默认 0.5 阈值 recall 就 0.837（无需调阈值），校准最好；弱信号也更强（SNR&lt;8 档 recall 0.89）。仅 2 epoch 即超过 Gemma 31B 的 3 epoch。"},
    {"key": "g31v2", "name": "Gemma4 31B v2", "family": "Gemma 4 · 原生多模态",
     "tag": "灰度 · 2 epoch · 560 tok", "rank": 2, "best": False, "color": "#3b528b",
     "dir": RUNS / "e1_gemma4_31b_v2",
     "note": "获胜配方提出者（r32/α32/lr2e-4/dropout0/有效batch8/560）。0.5 阈值偏保守（recall 0.62）需调阈值；FPR≤5% 工作点 recall 75% / precision 94%。"},
    {"key": "g31vir", "name": "Gemma4 31B", "family": "Gemma 4 · 原生多模态",
     "tag": "viridis · 3 epoch · 560 tok", "rank": 3, "best": False, "color": "#5ec962",
     "dir": RUNS / "e1_gemma4_31b_viridis_3ep",
     "note": "全量 viridis + 第 3 个 epoch。结果 0.905 ≤ 灰度 2ep 的 0.922（差异在统计噪声内）；train_loss 更低但 test 没升 = 过拟合迹象。证明补 epoch + 彩色均无增益。"},
    {"key": "e4bvir", "name": "Gemma4 E4B", "family": "Gemma 4 · 边缘版（调试）",
     "tag": "viridis · 3 epoch · 512 tok", "rank": 4, "best": False, "color": "#7e4ea3",
     "dir": RUNS / "e1_gemma4_e4b_viridis",
     "note": "最小调试模型（~17GB）。viridis 0.9046 ≈ 灰度 0.9042 —— 彩色 A/B 对照证明色彩无显著增益。"},
]

OP_LABELS = {"default_0.5": "默认 (阈值 0.5)", "max_f1": "最大 F1",
             "fpr<=0.05": "FPR ≤ 5% (≈FAR)", "fpr<=0.1": "FPR ≤ 10%"}


# ---------- 数据读取与计算（纯 Python） ----------
def load_per_sample(d):
    p = d / "per_sample.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def roc_points(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda x: (-x[0]))
    P = sum(labels) or 1
    N = (len(labels) - sum(labels)) or 1
    tp = fp = 0
    pts = [[0.0, 0.0]]
    for s, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        pts.append([round(fp / N, 4), round(tp / P, 4)])
    return pts


def sweep(scores, labels, n=51):
    P = sum(labels) or 1
    out = []
    for i in range(n):
        t = i / (n - 1)
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and y == 0)
        rec = tp / P
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        out.append([round(t, 3), round(rec, 4), round(prec, 4)])
    return out


def load_loss(d):
    p = d / "trainer_state.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text()).get("log_history", [])
    tr = [[x["step"], round(x["loss"], 4)] for x in h if "loss" in x]
    ev = [[x["step"], round(x["eval_loss"], 4)] for x in h if "eval_loss" in x]
    return {"train": tr, "eval": ev}


# SNR 来源：dataset_test.jsonl，按图片 basename 关联
def load_snr_map():
    p = ROOT / "output" / "dataset_test.jsonl"
    m = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        base = Path(r["image_path"]).name
        snr = (r.get("metadata") or {}).get("snr")
        det = (r.get("label") or {}).get("detection")
        m[base] = {"snr": snr, "pos": det == "YES"}
    return m


SNR_BINS = [(0, 8, "<8"), (8, 12, "8–12"), (12, 20, "12–20"), (20, 1e9, "≥20")]


def snr_recall(per, snr_map, thr=0.5):
    counts = {b[2]: [0, 0] for b in SNR_BINS}  # [hit, total]
    for row in per:
        base = Path(row["image"]).name
        info = snr_map.get(base)
        if not info or not info["pos"] or info["snr"] is None:
            continue
        s = info["snr"]
        for lo, hi, lab in SNR_BINS:
            if lo <= s < hi:
                counts[lab][1] += 1
                if row["p_yes"] >= thr:
                    counts[lab][0] += 1
                break
    bins = [b[2] for b in SNR_BINS if counts[b[2]][1] > 0]  # 丢掉空档(如测试集无 SNR≥20)
    recall = [round(counts[b][0] / counts[b][1], 3) for b in bins]
    ns = [counts[b][1] for b in bins]
    return {"bins": bins, "recall": recall, "n": ns, "thr": thr}


snr_map = load_snr_map()
data = []
for m in MODELS:
    rep = json.loads((m["dir"] / "prob_report.json").read_text())
    per = load_per_sample(m["dir"])
    scores = [r["p_yes"] for r in per]
    labels = [int(r["gold"]) for r in per]
    data.append({
        "key": m["key"], "name": m["name"], "family": m["family"], "tag": m["tag"],
        "rank": m["rank"], "best": m["best"], "color": m["color"], "note": m["note"],
        "auc": rep["roc_auc"], "pr_auc": rep["pr_auc"], "n": rep["n"],
        "ops": rep["operating_points"],
        "loss": load_loss(m["dir"]),
        "roc": roc_points(scores, labels),
        "sweep": sweep(scores, labels),
        "snr": snr_recall(per, snr_map, 0.5),
    })

PAYLOAD = json.dumps(data, ensure_ascii=False)

# ---------- 汇总表 ----------
def pct(x):
    return f"{x*100:.1f}%" if x is not None else "—"


rows = ""
for d in data:
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(d["rank"], "")
    op05 = d["ops"]["default_0.5"]
    far = d["ops"].get("fpr<=0.05", {})
    cls = ' class="best-row"' if d["best"] else ""
    rows += f"""<tr{cls}>
      <td class="rank">{medal} {d['rank']}</td>
      <td><b>{d['name']}</b><div class="sub">{d['family']}</div></td>
      <td class="mono">{d['tag']}</td>
      <td class="num big" style="color:{d['color']}">{d['auc']:.3f}</td>
      <td class="num">{d['pr_auc']:.3f}</td>
      <td class="num">{pct(op05['recall'])}</td>
      <td class="num">{pct(far.get('recall'))} <span class="sub">/ P {pct(far.get('precision'))}</span></td>
    </tr>"""

findings = [
    ("🏆 原生多模态 Qwen3.6 胜出", "ROC-AUC <b>0.940</b> &gt; Gemma4 31B（0.922）。默认 0.5 阈值 recall 就 0.837（Gemma 仅 ~0.62），校准更好，仅 2 epoch。"),
    ("📉 第 3 个 epoch 没用甚至有害", "Gemma4 31B viridis 3ep（0.905）≤ 灰度 2ep（0.922）；train_loss 更低但 test 没升 = 过拟合。"),
    ("🎨 彩色（viridis）无显著增益", "E4B viridis 0.9046 ≈ 灰度 0.9042。彩色不亏（用预训练 RGB 编码器），但不是 recall 救星。"),
    ("🔬 天花板是物理 SNR", "色彩 / 分辨率(&gt;560) / epoch(&gt;2) 都突破不了 ~0.92–0.94。瓶颈是低 SNR 弱信号，要靠注入(E3/E4)。"),
    ("📏 评估须用 ROC-AUC + FAR", "贪心 accuracy（0.5 阈值）会严重低估 recall，是阈值假象。应以 ROC-AUC + 低 FPR 工作点（≈FAR）为准。"),
    ("🛠 工程坑已解决", "Qwen3.6 视觉推理在 transformers 5.12+Unsloth 有 rope bug → 评估改用原生 transformers+PEFT。跨机同步须单文件 rsync + 验证落地。"),
]
find_html = "".join(f'<div class="finding"><h4>{t}</h4><p>{d}</p></div>' for t, d in findings)

cards = ""
for d in data:
    badge = '<span class="badge-best">最佳</span>' if d["best"] else ""
    op_rows = ""
    for k, lab in OP_LABELS.items():
        op = d["ops"].get(k)
        if not op:
            continue
        op_rows += f"""<tr><td>{lab}</td><td class="mono">{op['threshold']}</td>
          <td>{pct(op['accuracy'])}</td><td>{pct(op['precision'])}</td>
          <td><b>{pct(op['recall'])}</b></td><td>{pct(op['fpr'])}</td></tr>"""
    cards += f"""<section class="card" id="card-{d['key']}">
      <div class="card-head"><h3>{d['name']} {badge}</h3><div class="tag mono">{d['tag']}</div></div>
      <div class="chips">
        <div class="chip"><span>ROC-AUC</span><b style="color:{d['color']}">{d['auc']:.3f}</b></div>
        <div class="chip"><span>PR-AUC</span><b>{d['pr_auc']:.3f}</b></div>
        <div class="chip"><span>测试样本</span><b>{d['n']}</b></div>
      </div>
      <p class="note">{d['note']}</p>
      <table class="op"><thead><tr><th>工作点</th><th>阈值</th><th>Acc</th><th>Precision</th><th>Recall</th><th>FPR</th></tr></thead>
        <tbody>{op_rows}</tbody></table>
      <div class="card-charts">
        <div class="mini"><div class="mini-t">训练 / 验证损失</div><div class="svg-host" data-chart="loss" data-key="{d['key']}"></div></div>
        <div class="mini"><div class="mini-t">混淆矩阵 (阈值 0.5)</div><div class="svg-host" data-chart="cm" data-key="{d['key']}"></div></div>
        <div class="mini"><div class="mini-t">Recall / Precision vs 阈值</div><div class="svg-host" data-chart="sweep" data-key="{d['key']}"></div></div>
      </div>
    </section>"""

dataset_chips = """
  <div class="chip"><span>训练</span><b>2394</b></div><div class="chip"><span>验证</span><b>306</b></div>
  <div class="chip"><span>测试</span><b>270</b></div><div class="chip"><span>正/负</span><b>1485 / 1485</b></div>
  <div class="chip"><span>探测器</span><b>H1 + L1</b></div><div class="chip"><span>窗口</span><b>4 s @ 合并</b></div>
"""

HTML = f"""<!DOCTYPE html><html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GW-VLM · E1 检测实验结果</title>
<style>
  :root {{ --purple:#440154; --indigo:#3b528b; --teal:#21918c; --green:#5ec962; --yellow:#fde725;
    --ink:#1a1a2e; --muted:#6b7280; --line:#e7e7ee; --bg:#f6f6fb; --card:#fff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; color:var(--ink); background:var(--bg); line-height:1.6; }}
  .mono {{ font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace; font-size:.85em; }}
  header.hero {{ background:linear-gradient(120deg,var(--purple),var(--indigo) 45%,var(--teal) 80%,var(--green)); color:#fff; padding:54px 24px 46px; text-align:center; }}
  header.hero h1 {{ margin:0 0 8px; font-size:2rem; letter-spacing:.5px; }}
  header.hero p {{ margin:4px 0; opacity:.92; }}
  .hero .pill {{ display:inline-block; margin-top:14px; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35); padding:6px 16px; border-radius:999px; }}
  main {{ max-width:1080px; margin:0 auto; padding:0 20px 64px; }}
  .section-title {{ font-size:1.3rem; margin:42px 0 16px; padding-left:12px; border-left:5px solid var(--teal); }}
  .summary {{ background:var(--card); border-radius:16px; overflow:hidden; box-shadow:0 6px 24px rgba(20,20,50,.07); margin-top:-28px; position:relative; }}
  table.rank {{ width:100%; border-collapse:collapse; }}
  table.rank th, table.rank td {{ padding:13px 14px; text-align:left; border-bottom:1px solid var(--line); }}
  table.rank thead th {{ background:#fafafe; font-size:.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:.4px; }}
  table.rank td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  table.rank td.big {{ font-size:1.15rem; font-weight:700; }}
  table.rank td.rank {{ white-space:nowrap; font-weight:600; }}
  .sub {{ color:var(--muted); font-size:.8rem; }}
  tr.best-row {{ background:linear-gradient(90deg,rgba(94,201,98,.13),rgba(253,231,37,.10)); }}
  .findings {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }}
  .finding {{ background:var(--card); border-radius:14px; padding:16px 18px; box-shadow:0 3px 12px rgba(20,20,50,.05); border-top:3px solid var(--teal); }}
  .finding h4 {{ margin:0 0 6px; font-size:1.02rem; }}
  .finding p {{ margin:0; color:#374151; font-size:.93rem; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:10px; margin:6px 0 14px; }}
  .chip {{ background:#f1f5f9; border-radius:10px; padding:8px 14px; display:flex; flex-direction:column; }}
  .chip span {{ color:var(--muted); font-size:.74rem; }}
  .chip b {{ font-size:1.05rem; }}
  .dataset {{ display:flex; flex-wrap:wrap; gap:10px; background:var(--card); border-radius:14px; padding:16px 18px; box-shadow:0 3px 12px rgba(20,20,50,.05); }}
  .panel {{ background:var(--card); border-radius:16px; padding:20px; box-shadow:0 4px 18px rgba(20,20,50,.06); margin:16px 0; }}
  .panel h3 {{ margin:0 0 4px; font-size:1.05rem; }}
  .panel .cap {{ color:var(--muted); font-size:.85rem; margin-bottom:8px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:16px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:.85rem; margin-top:6px; }}
  .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
  .legend i {{ width:14px; height:4px; border-radius:2px; display:inline-block; }}
  .card {{ background:var(--card); border-radius:16px; padding:22px 22px 26px; margin:18px 0; box-shadow:0 4px 18px rgba(20,20,50,.06); }}
  .card-head {{ display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:8px; border-bottom:1px solid var(--line); padding-bottom:12px; }}
  .card-head h3 {{ margin:0; font-size:1.35rem; }}
  .card-head .tag {{ color:var(--muted); }}
  .badge-best {{ background:linear-gradient(90deg,var(--teal),var(--green)); color:#fff; font-size:.72rem; padding:3px 10px; border-radius:999px; margin-left:8px; vertical-align:middle; }}
  .note {{ color:#374151; font-size:.95rem; }}
  table.op {{ width:100%; border-collapse:collapse; margin:8px 0 6px; font-size:.88rem; }}
  table.op th, table.op td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; }}
  table.op th:first-child, table.op td:first-child {{ text-align:left; }}
  table.op thead th {{ color:var(--muted); font-weight:600; font-size:.78rem; }}
  .card-charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; margin-top:14px; }}
  .mini {{ background:#fbfbfe; border:1px solid var(--line); border-radius:12px; padding:10px 8px 6px; }}
  .mini-t {{ font-size:.82rem; color:var(--muted); text-align:center; margin-bottom:2px; }}
  .svg-host svg {{ width:100%; height:auto; display:block; }}
  text {{ font-family:-apple-system,"PingFang SC",sans-serif; }}
  footer {{ text-align:center; color:var(--muted); font-size:.85rem; padding:30px 20px; }}
  .tip {{ position:fixed; pointer-events:none; background:rgba(26,26,46,.94); color:#fff; padding:5px 9px; border-radius:6px; font-size:.78rem; opacity:0; transition:opacity .1s; z-index:50; white-space:nowrap; }}
</style></head>
<body>
<header class="hero">
  <h1>GW-VLM · E1 检测实验结果</h1>
  <p>视觉语言模型直接读 Q-transform 时频图做引力波检测</p>
  <p style="font-size:.92rem;opacity:.85">Qwen3.6 / Gemma 4 · LoRA 微调 · DGX Spark · 270 测试样本</p>
  <div class="pill">最佳 ROC-AUC&nbsp;<b>0.940</b>&nbsp;· Qwen3.6-27B（原生多模态）</div>
</header>
<main>
  <div class="summary"><table class="rank">
    <thead><tr><th>排名</th><th>模型</th><th>配置</th><th>ROC-AUC</th><th>PR-AUC</th><th>Recall@0.5</th><th>Recall@FPR≤5%</th></tr></thead>
    <tbody>{rows}</tbody></table></div>

  <h2 class="section-title">总体对比</h2>
  <div class="grid2">
    <div class="panel"><h3>ROC-AUC 对比</h3><div class="cap">阈值无关的判别力（越接近 1 越强）</div><div class="svg-host" id="aucBar"></div></div>
    <div class="panel"><h3>ROC 曲线叠加</h3><div class="cap">TPR(recall) vs FPR(误报率)；左上角越靠越好</div><div class="svg-host" id="rocAll"></div><div class="legend" id="rocLeg"></div></div>
  </div>
  <div class="panel"><h3>检测效率 vs SNR（分档 recall @ 阈值 0.5）</h3>
    <div class="cap">漏检集中在低信噪比弱信号——这是物理瓶颈，不是模型/调色问题。各档样本数标在柱顶。</div>
    <div class="svg-host" id="snrBar"></div><div class="legend" id="snrLeg"></div></div>

  <h2 class="section-title">关键发现</h2>
  <div class="findings">{find_html}</div>

  <h2 class="section-title">数据集</h2><div class="dataset">{dataset_chips}</div>

  <h2 class="section-title">各模型详情 & 图表</h2>
  {cards}

  <footer>GW-VLM E1（纯检测）· 全部图表由实际结果数据原生渲染（无 PNG 截图）<br>
    评估口径：每样本 P(YES) → ROC-AUC / PR-AUC / 阈值扫描 / 低 FPR 工作点（≈FAR）。</footer>
</main>
<div class="tip" id="tip"></div>
<script>
const DATA = {PAYLOAD};
const byKey = Object.fromEntries(DATA.map(d=>[d.key,d]));
const tip = document.getElementById('tip');
const NS='http://www.w3.org/2000/svg';
function el(t,a){{const e=document.createElementNS(NS,t);for(const k in (a||{{}}))e.setAttribute(k,a[k]);return e;}}
function showTip(ev,html){{tip.innerHTML=html;tip.style.opacity=1;tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';}}
function hideTip(){{tip.style.opacity=0;}}

// 通用坐标轴框架
function frame(W,H,m){{const svg=el('svg',{{viewBox:`0 0 ${{W}} ${{H}}`}});const g=el('g');svg.appendChild(g);
  return {{svg,g,ix:x=>m.l+(x-m.x0)/(m.x1-m.x0)*(W-m.l-m.r),iy:y=>H-m.b-(y-m.y0)/(m.y1-m.y0)*(H-m.t-m.b),W,H,m}};}}
function axes(F,opts){{const {{svg,m,W,H}}=F;
  svg.appendChild(el('line',{{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#cbd5e1'}}));
  svg.appendChild(el('line',{{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#cbd5e1'}}));
  (opts.yticks||[]).forEach(v=>{{const y=F.iy(v);
    svg.appendChild(el('line',{{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:'#eef0f5'}}));
    const t=el('text',{{x:m.l-6,y:y+3,'text-anchor':'end','font-size':10,fill:'#9ca3af'}});t.textContent=opts.yfmt?opts.yfmt(v):v;svg.appendChild(t);}});
  if(opts.xlabel){{const t=el('text',{{x:(m.l+W-m.r)/2,y:H-4,'text-anchor':'middle','font-size':10,fill:'#6b7280'}});t.textContent=opts.xlabel;svg.appendChild(t);}}
  if(opts.ylabel){{const t=el('text',{{x:12,y:(m.t+H-m.b)/2,'text-anchor':'middle','font-size':10,fill:'#6b7280',transform:`rotate(-90 12 ${{(m.t+H-m.b)/2}})`}});t.textContent=opts.ylabel;svg.appendChild(t);}}
}}
function polyline(F,pts,color,w){{const d=pts.map((p,i)=>(i?'L':'M')+F.ix(p[0]).toFixed(1)+' '+F.iy(p[1]).toFixed(1)).join(' ');
  F.svg.appendChild(el('path',{{d,fill:'none',stroke:color,'stroke-width':w||2,'stroke-linejoin':'round'}}));}}

// 1) AUC 横向条形
(function(){{const host=document.getElementById('aucBar');const W=480,bh=34,H=DATA.length*bh+30;
  const F=frame(W,H,{{l:120,r:50,t:8,b:22,x0:0.8,x1:0.96,y0:0,y1:1}});
  F.svg.appendChild(el('line',{{x1:F.ix(0.8),y1:8,x2:F.ix(0.8),y2:H-22,stroke:'#e7e7ee'}}));
  [0.85,0.90,0.95].forEach(v=>{{const x=F.ix(v);F.svg.appendChild(el('line',{{x1:x,y1:8,x2:x,y2:H-22,stroke:'#eef0f5'}}));
    const t=el('text',{{x,y:H-8,'text-anchor':'middle','font-size':10,fill:'#9ca3af'}});t.textContent=v;F.svg.appendChild(t);}});
  DATA.forEach((d,i)=>{{const y=14+i*bh;const x0=F.ix(0.8),x1=F.ix(d.auc);
    const r=el('rect',{{x:x0,y,width:Math.max(1,x1-x0),height:bh-14,rx:4,fill:d.color,opacity:d.best?1:.82}});F.svg.appendChild(r);
    const nm=el('text',{{x:114,y:y+(bh-14)/2+4,'text-anchor':'end','font-size':11,fill:'#374151'}});nm.textContent=d.name;F.svg.appendChild(nm);
    const vl=el('text',{{x:x1+5,y:y+(bh-14)/2+4,'font-size':11,'font-weight':700,fill:d.color}});vl.textContent=d.auc.toFixed(3);F.svg.appendChild(vl);}});
  host.appendChild(F.svg);}})();

// 2) ROC 叠加
(function(){{const host=document.getElementById('rocAll');const W=480,H=380;
  const F=frame(W,H,{{l:42,r:14,t:12,b:34,x0:0,x1:1,y0:0,y1:1}});
  axes(F,{{yticks:[0,.25,.5,.75,1],yfmt:v=>v,xlabel:'FPR（误报率）',ylabel:'TPR（recall）'}});
  F.svg.appendChild(el('line',{{x1:F.ix(0),y1:F.iy(0),x2:F.ix(1),y2:F.iy(1),stroke:'#d1d5db','stroke-dasharray':'4 4'}}));
  DATA.forEach(d=>polyline(F,d.roc,d.color,d.best?2.6:1.8));
  host.appendChild(F.svg);
  document.getElementById('rocLeg').innerHTML=DATA.map(d=>`<span><i style="background:${{d.color}}"></i>${{d.name}} (${{d.auc.toFixed(3)}})</span>`).join('');
}})();

// 3) SNR 分档 grouped bar
(function(){{const host=document.getElementById('snrBar');const bins=DATA[0].snr.bins;
  const W=760,H=300,m={{l:42,r:14,t:14,b:40}};const F=frame(W,H,{{l:m.l,r:m.r,t:m.t,b:m.b,x0:0,x1:1,y0:0,y1:1}});
  axes(F,{{yticks:[0,.25,.5,.75,1],yfmt:v=>(v*100)+'%',ylabel:'recall'}});
  const gw=(W-m.l-m.r)/bins.length, bw=gw/(DATA.length+1);
  bins.forEach((b,bi)=>{{const gx=m.l+bi*gw;
    const lt=el('text',{{x:gx+gw/2,y:H-22,'text-anchor':'middle','font-size':11,fill:'#374151'}});lt.textContent='SNR '+b;F.svg.appendChild(lt);
    DATA.forEach((d,di)=>{{const r=d.snr.recall[bi];if(r==null)return;const x=gx+bw*0.5+di*bw;const y=F.iy(r);
      const rect=el('rect',{{x,y,width:bw*0.86,height:F.iy(0)-y,rx:2,fill:d.color,opacity:d.best?1:.8}});
      rect.addEventListener('mousemove',e=>showTip(e,`${{d.name}}<br>SNR ${{b}}: recall ${{(r*100).toFixed(0)}}% (n=${{d.snr.n[bi]}})`));
      rect.addEventListener('mouseleave',hideTip);F.svg.appendChild(rect);}});
    const nt=el('text',{{x:gx+gw/2,y:m.t+8,'text-anchor':'middle','font-size':9,fill:'#9ca3af'}});nt.textContent='n='+DATA[0].snr.n[bi];F.svg.appendChild(nt);
  }});
  host.appendChild(F.svg);
  document.getElementById('snrLeg').innerHTML=DATA.map(d=>`<span><i style="background:${{d.color}}"></i>${{d.name}}</span>`).join('');
}})();

// 4) 每模型：损失 / 混淆 / 阈值扫描
// 损失用【对数纵轴】:线性轴会把塌陷后的细节(0.03 vs 0.005、过拟合)全压到 0 附近看不见。
function drawLoss(host,d){{const L=d.loss;if(!L){{host.innerHTML='<p class="sub">无损失数据</p>';return;}}
  const W=320,H=200;const steps=L.train.map(p=>p[0]).concat(L.eval.map(p=>p[0]));
  const vals=L.train.map(p=>p[1]).concat(L.eval.map(p=>p[1])).filter(v=>v>0);
  const lo=Math.min(...vals),hi=Math.max(...vals);
  const y0=Math.log10(lo)-0.12,y1=Math.log10(hi)+0.05;
  const F=frame(W,H,{{l:46,r:10,t:10,b:30,x0:0,x1:Math.max(...steps),y0,y1}});
  const ticks=[];for(let e=Math.floor(y1);e>=Math.ceil(y0);e--){{ticks.push(e);}}
  axes(F,{{yticks:ticks,yfmt:lv=>{{const r=Math.pow(10,lv);return r>=1?r.toFixed(0):(r>=0.01?r.toFixed(2):r.toExponential(0));}},xlabel:'step（对数纵轴）'}});
  polyline(F,L.train.map(p=>[p[0],Math.log10(p[1])]),d.color,1.6);
  L.eval.forEach(p=>{{const c=el('circle',{{cx:F.ix(p[0]),cy:F.iy(Math.log10(p[1])),r:3.5,fill:'#fff',stroke:d.color,'stroke-width':2}});
    c.addEventListener('mousemove',e=>showTip(e,`eval loss ${{p[1]}} @step ${{p[0]}}`));c.addEventListener('mouseleave',hideTip);F.svg.appendChild(c);}});
  host.appendChild(F.svg);
  host.insertAdjacentHTML('beforeend',`<div class="legend" style="justify-content:center"><span><i style="background:${{d.color}}"></i>train</span><span><i style="background:#fff;border:2px solid ${{d.color}};width:10px;height:10px;border-radius:50%"></i>eval(每epoch)</span></div>`);
}}
function drawCM(host,d){{const op=d.ops['default_0.5'];const W=240,H=200;const svg=el('svg',{{viewBox:`0 0 ${{W}} ${{H}}`}});
  const cells=[['TP',op.tp,'#127a4e'],['FN',op.fn,'#b45309'],['FP',op.fp,'#b45309'],['TN',op.tn,'#127a4e']];
  const mx=Math.max(op.tp,op.fn,op.fp,op.tn),cw=80,ch=64,ox=70,oy=24;
  ['预测 NO','预测 YES'].forEach((t,i)=>{{const e=el('text',{{x:ox+cw/2+i*cw,y:oy-6,'text-anchor':'middle','font-size':10,fill:'#6b7280'}});e.textContent=t;svg.appendChild(e);}});
  ['真 YES','真 NO'].forEach((t,i)=>{{const e=el('text',{{x:ox-6,y:oy+ch/2+i*ch+4,'text-anchor':'end','font-size':10,fill:'#6b7280'}});e.textContent=t;svg.appendChild(e);}});
  // 列:0=预测NO 1=预测YES;行:0=真YES 1=真NO  →  TP(真YES,预YES)=右上 等
  const order=[[op.tp,'TP',1,0,true],[op.fn,'FN',0,0,false],[op.fp,'FP',1,1,false],[op.tn,'TN',0,1,true]];
  order.forEach(([v,lab,cx,cy,good])=>{{const x=ox+cx*cw,y=oy+cy*ch;const al=0.18+0.72*(v/mx);
    svg.appendChild(el('rect',{{x,y,width:cw-4,height:ch-4,rx:6,fill:good?'#21918c':'#e8924f','fill-opacity':al}}));
    const n=el('text',{{x:x+cw/2-2,y:y+ch/2,'text-anchor':'middle','font-size':18,'font-weight':700,fill:'#1a1a2e'}});n.textContent=v;svg.appendChild(n);
    const l=el('text',{{x:x+cw/2-2,y:y+ch/2+15,'text-anchor':'middle','font-size':9,fill:'#6b7280'}});l.textContent=lab;svg.appendChild(l);}});
  host.appendChild(svg);}}
function drawSweep(host,d){{const W=320,H=200;const F=frame(W,H,{{l:40,r:10,t:10,b:28,x0:0,x1:1,y0:0,y1:1}});
  axes(F,{{yticks:[0,.5,1],yfmt:v=>(v*100)+'%',xlabel:'阈值'}});
  F.svg.appendChild(el('line',{{x1:F.ix(0.5),y1:F.m.t,x2:F.ix(0.5),y2:H-F.m.b,stroke:'#cbd5e1','stroke-dasharray':'3 3'}}));
  polyline(F,d.sweep.map(p=>[p[0],p[1]]),'#21918c',2);
  polyline(F,d.sweep.map(p=>[p[0],p[2]]),'#e8924f',2);
  host.appendChild(F.svg);
  host.insertAdjacentHTML('beforeend','<div class="legend" style="justify-content:center"><span><i style="background:#21918c"></i>recall</span><span><i style="background:#e8924f"></i>precision</span></div>');
}}
document.querySelectorAll('.svg-host[data-chart]').forEach(h=>{{const d=byKey[h.dataset.key],c=h.dataset.chart;
  if(c==='loss')drawLoss(h,d);else if(c==='cm')drawCM(h,d);else if(c==='sweep')drawSweep(h,d);}});
</script>
</body></html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"已生成 {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
print(f"模型 {len(data)} 个；图表全部由数据原生渲染（损失/ROC/阈值扫描/混淆/SNR/AUC）")
