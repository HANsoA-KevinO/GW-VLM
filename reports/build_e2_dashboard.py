"""
E2(多任务:检测+参数)结果分析看板。单文件自包含 HTML,所有图由数据原生 SVG 渲染。
重点:参数混淆矩阵(看"输出安全众数 bin"的塌缩模式)、gold vs pred 分布、检测曲线、思考诊断 reasoning。

用法:python3 reports/build_e2_dashboard.py  → reports/e2_dashboard.html
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "output" / "runs" / "e2_qwen36_27b_viridis"
OUT = Path(__file__).resolve().parent / "e2_dashboard.html"

FIELDS = [("chirp_mass_bin", "Chirp Mass (质量)"), ("distance_bin", "Distance (距离)"),
          ("chi_eff_bin", "χ_eff (有效自旋)")]


def binkey(s):
    m = re.match(r"\s*(-?\d+\.?\d*)", str(s))
    return float(m.group(1)) if m else -1e9


# ---- 读数据 ----
rows = [json.loads(l) for l in (RUN / "e2_per_sample.jsonl").read_text().splitlines() if l.strip()]
report = json.loads((RUN / "e2_report.json").read_text())
pos = [r for r in rows if r["gold"].get("detection") == "YES"]

# SNR 映射(用于检测分档)
snr_map = {}
for line in (ROOT / "output" / "dataset_test.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    snr_map[Path(r["image_path"]).name] = (r.get("metadata") or {}).get("snr")

# 检测 ROC / sweep
det_scores = [r["p_yes"] for r in rows]
det_labels = [1 if r["gold"].get("detection") == "YES" else 0 for r in rows]


def roc_points(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
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
        out.append([round(t, 3), round(tp / P, 4), round(tp / (tp + fp), 4) if (tp + fp) else 1.0])
    return out


# 参数混淆矩阵 + 分布
def param_analysis(field):
    golds = sorted({r["gold"][field] for r in pos if r["gold"].get(field) not in (None, "N/A")}, key=binkey)
    preds_extra = sorted({r["pred"].get(field) for r in pos
                          if r["pred"].get(field) not in (None, "N/A") and r["pred"].get(field) not in golds}, key=binkey)
    col_bins = golds + preds_extra + ["N/A"]
    # 行=gold, 列=pred
    mat = [[0] * len(col_bins) for _ in golds]
    gi = {b: i for i, b in enumerate(golds)}
    ci = {b: i for i, b in enumerate(col_bins)}
    for r in pos:
        g = r["gold"].get(field)
        p = r["pred"].get(field) or "N/A"
        if g in gi:
            mat[gi[g]][ci.get(p, ci["N/A"])] += 1
    gold_dist = {b: sum(1 for r in pos if r["gold"].get(field) == b) for b in golds}
    pred_dist = {b: sum(1 for r in pos if (r["pred"].get(field) or "N/A") == b) for b in col_bins}
    rep = report["params_on_true_positives"][field]
    # 条件准确率(仅检出YES)
    dety = [r for r in pos if r["pred"].get("detection") == "YES"]
    cond = sum(1 for r in dety if r["pred"].get(field) == r["gold"].get(field)) / len(dety) if dety else 0
    return {"field": field, "gold_bins": golds, "col_bins": col_bins, "matrix": mat,
            "gold_dist": [gold_dist[b] for b in golds], "pred_dist": [pred_dist[b] for b in col_bins],
            "exact": rep["exact_acc"], "adjacent": rep["adjacent_pm1_acc"],
            "chance": rep["chance"], "cond": round(cond, 4), "n_bins": rep["n_bins"]}


# 损失
def load_loss():
    p = RUN / "checkpoint-900" / "trainer_state.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text()).get("log_history", [])
    return {"train": [[x["step"], round(x["loss"], 4)] for x in h if "loss" in x],
            "eval": [[x["step"], round(x["eval_loss"], 4)] for x in h if "eval_loss" in x]}


# 思考诊断:取代表性 reasoning(2条弱信号判NO + 2条判YES)
def thinking_samples():
    dg = [json.loads(l) for l in (RUN / "think_diag.jsonl").read_text().splitlines() if l.strip()]
    no = [d for d in dg if d.get("kind") == "pos" and d.get("final_detection") != "YES"][:3]
    yes = [d for d in dg if d.get("kind") == "pos" and d.get("final_detection") == "YES"][:2]
    out = []
    for d in yes + no:
        out.append({"gold": "%s / %s" % (d["gold"].get("chirp_mass_bin"), d["gold"].get("distance_bin")),
                    "det": d.get("final_detection"), "tok": d.get("n_think_tokens"),
                    "text": (d.get("answer") or d.get("thinking") or "")[:1400]})
    n_no = sum(1 for d in dg if d.get("kind") == "pos" and d.get("final_detection") != "YES")
    n_pos = sum(1 for d in dg if d.get("kind") == "pos")
    return out, n_no, n_pos


params = [param_analysis(f) for f, _ in FIELDS]
loss = load_loss()
think, think_no, think_n = thinking_samples()
det = report["detection"]
PAYLOAD = json.dumps({
    "roc": roc_points(det_scores, det_labels), "sweep": sweep(det_scores, det_labels),
    "ops": det["operating_points"], "loss": loss, "params": params,
    "field_labels": dict(FIELDS),
}, ensure_ascii=False)

# 生成式 recall
dety = sum(1 for r in pos if r["pred"].get("detection") == "YES")


def pct(x):
    return f"{x*100:.1f}%" if x is not None else "—"


# 参数卡 HTML
pcards = ""
for f, label in FIELDS:
    pa = next(p for p in params if p["field"] == f)
    pcards += f"""<div class="pcard">
      <h4>{label}</h4>
      <div class="pmetrics">
        <span class="m"><b style="color:{'#127a4e' if pa['exact']>pa['chance'] else '#b91c1c'}">{pct(pa['exact'])}</b><i>精确</i></span>
        <span class="m"><b>{pct(pa['cond'])}</b><i>条件精确</i></span>
        <span class="m"><b>{pct(pa['adjacent'])}</b><i>邻接±1</i></span>
        <span class="m"><b class="sub">{pct(pa['chance'])}</b><i>随机基线</i></span>
      </div>
      <div class="svg-host" data-cm="{f}"></div>
      <div class="svg-host" data-dist="{f}"></div>
    </div>"""

# 思考 reasoning 卡
tcards = ""
for t in think:
    color = "#127a4e" if t["det"] == "YES" else "#b91c1c"
    tcards += f"""<div class="tcard">
      <div class="thead">GOLD(质量/距离): <b>{t['gold']}</b> · 思考后判定 <b style="color:{color}">{t['det']}</b> · {t['tok']}tok</div>
      <pre class="treason">{t['text'].replace('<','&lt;').replace('>','&gt;')}</pre>
    </div>"""

op05 = det["operating_points"]["default_0.5"]

HTML = f"""<!DOCTYPE html><html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GW-VLM · E2 多任务结果分析</title>
<style>
  :root {{ --purple:#440154; --indigo:#3b528b; --teal:#21918c; --green:#5ec962; --yellow:#fde725;
    --ink:#1a1a2e; --muted:#6b7280; --line:#e7e7ee; --bg:#f6f6fb; --card:#fff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif; color:var(--ink); background:var(--bg); line-height:1.6; }}
  .mono {{ font-family:"SF Mono",ui-monospace,Menlo,monospace; font-size:.85em; }}
  header.hero {{ background:linear-gradient(120deg,var(--purple),var(--indigo) 50%,var(--teal) 85%,var(--green)); color:#fff; padding:48px 24px 40px; text-align:center; }}
  header.hero h1 {{ margin:0 0 6px; font-size:1.9rem; }}
  header.hero p {{ margin:3px 0; opacity:.92; }}
  .hero .pills {{ margin-top:14px; display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }}
  .hero .pill {{ background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.35); padding:6px 16px; border-radius:999px; font-size:.9rem; }}
  main {{ max-width:1120px; margin:0 auto; padding:0 20px 64px; }}
  .section-title {{ font-size:1.3rem; margin:40px 0 14px; padding-left:12px; border-left:5px solid var(--teal); }}
  .panel {{ background:var(--card); border-radius:16px; padding:20px; box-shadow:0 4px 18px rgba(20,20,50,.06); margin:14px 0; }}
  .panel h3 {{ margin:0 0 4px; font-size:1.05rem; }} .panel .cap {{ color:var(--muted); font-size:.85rem; margin-bottom:8px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:16px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .chip {{ background:#f1f5f9; border-radius:10px; padding:8px 14px; display:flex; flex-direction:column; }}
  .chip span {{ color:var(--muted); font-size:.74rem; }} .chip b {{ font-size:1.05rem; }}
  .insight {{ background:linear-gradient(90deg,rgba(94,201,98,.12),rgba(253,231,37,.08)); border-left:4px solid var(--teal); border-radius:10px; padding:12px 16px; margin:12px 0; font-size:.95rem; }}
  table.op {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  table.op th,table.op td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; }}
  table.op th:first-child,table.op td:first-child {{ text-align:left; }} table.op thead th {{ color:var(--muted); font-size:.78rem; }}
  .pgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:16px; }}
  .pcard {{ background:var(--card); border-radius:14px; padding:16px; box-shadow:0 3px 12px rgba(20,20,50,.05); }}
  .pcard h4 {{ margin:0 0 8px; }} .pmetrics {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:8px; }}
  .pmetrics .m {{ display:flex; flex-direction:column; }} .pmetrics b {{ font-size:1.1rem; }} .pmetrics i {{ font-style:normal; color:var(--muted); font-size:.72rem; }}
  .sub {{ color:var(--muted); }}
  .svg-host svg {{ width:100%; height:auto; display:block; }}
  .tcard {{ background:var(--card); border-radius:12px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 8px rgba(20,20,50,.05); }}
  .thead {{ font-size:.85rem; color:#374151; margin-bottom:6px; }}
  .treason {{ white-space:pre-wrap; font-size:.82rem; color:#374151; background:#fbfbfe; border:1px solid var(--line); border-radius:8px; padding:10px; max-height:220px; overflow:auto; margin:0; font-family:-apple-system,"PingFang SC",sans-serif; }}
  footer {{ text-align:center; color:var(--muted); font-size:.85rem; padding:28px 20px; }}
  .tip {{ position:fixed; pointer-events:none; background:rgba(26,26,46,.94); color:#fff; padding:5px 9px; border-radius:6px; font-size:.78rem; opacity:0; transition:opacity .1s; z-index:50; }}
</style></head>
<body>
<header class="hero">
  <h1>GW-VLM · E2 多任务结果分析</h1>
  <p>Qwen3.6-27B · 检测 + 物理参数(chirp mass / distance / χ_eff)· viridis · 270 测试样本</p>
  <div class="pills">
    <div class="pill">检测 ROC-AUC <b>{det['roc_auc']:.3f}</b></div>
    <div class="pill">distance 精确 <b>{pct(params[1]['exact'])}</b>(随机{pct(params[1]['chance'])})</div>
    <div class="pill">生成式 recall <b>{dety}/{len(pos)}</b></div>
  </div>
</header>
<main>
  <h2 class="section-title">① 检测</h2>
  <div class="panel"><div class="chips">
    <div class="chip"><span>ROC-AUC</span><b style="color:var(--teal)">{det['roc_auc']:.4f}</b></div>
    <div class="chip"><span>PR-AUC</span><b>{det['pr_auc']:.4f}</b></div>
    <div class="chip"><span>Recall@0.5</span><b>{pct(op05['recall'])}</b></div>
    <div class="chip"><span>Precision@0.5</span><b>{pct(op05['precision'])}</b></div>
    <div class="chip"><span>vs E1 单检测</span><b>0.940 → {det['roc_auc']:.3f}</b></div>
  </div>
  <div class="insight">多任务(加参数预测)<b>没有拖累检测、反而略升</b>(0.940→{det['roc_auc']:.3f})——辅助任务起了正则作用。</div>
  </div>
  <div class="grid2">
    <div class="panel"><h3>ROC 曲线</h3><div class="cap">TPR vs FPR,左上角越靠越好</div><div class="svg-host" id="roc"></div></div>
    <div class="panel"><h3>Recall / Precision vs 阈值</h3><div class="cap">绿=recall 橙=precision,灰虚线=0.5</div><div class="svg-host" id="sweep"></div></div>
  </div>
  <div class="panel"><h3>工作点</h3><table class="op"><thead><tr><th>工作点</th><th>阈值</th><th>Acc</th><th>Precision</th><th>Recall</th><th>FPR</th></tr></thead><tbody id="ops"></tbody></table></div>

  <h2 class="section-title">② 物理参数(核心:看"输出安全众数 bin"的塌缩)</h2>
  <div class="insight">
    <b>关键发现:</b>模型倾向输出每个参数的"安全众数 bin",而非真正按图预测——
    <b>chirp mass 几乎只输出 25-40 一档(众数塌缩,≈随机)</b>;
    <b>distance 用了远端两档、偏向"远"但有真实区分(2.2×随机)</b>;
    <b>χ_eff 塌向 0.0-0.2(近零自旋先验)</b>。混淆矩阵按行(gold)归一化着色,看每个真实 bin 的预测都流向了哪。
  </div>
  <div class="pgrid">{pcards}</div>

  <h2 class="section-title">③ 损失曲线(E2 训练,对数纵轴)</h2>
  <div class="panel"><div class="svg-host" id="loss"></div></div>

  <h2 class="section-title">④ 思考诊断:为什么开 thinking 会把检测打崩</h2>
  <div class="insight">
    带【思考】自由生成时,正样本 <b>{think_no}/{think_n}</b> 被判 NO(生成式 recall≈0);而 teacher-forced(不思考)ROC-AUC {det['roc_auc']:.3f}。
    原因:思考套用"真信号=明亮 chirp 脊"的强信号先验,而我们的真信号大多不是亮线 → 把不思考时能用的亚感知证据"推理掉了"。下面是模型推理原文样本(绿=判YES、红=判NO):
  </div>
  {tcards}

  <footer>GW-VLM E2 多任务 · 全部图表由实际数据原生渲染 · 数据:output/runs/e2_qwen36_27b_viridis/</footer>
</main>
<div class="tip" id="tip"></div>
<script>
const D={PAYLOAD};
const tip=document.getElementById('tip'),NS='http://www.w3.org/2000/svg';
function el(t,a){{const e=document.createElementNS(NS,t);for(const k in(a||{{}}))e.setAttribute(k,a[k]);return e;}}
function showTip(ev,h){{tip.innerHTML=h;tip.style.opacity=1;tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';}}
function hideTip(){{tip.style.opacity=0;}}
function frame(W,H,m){{const svg=el('svg',{{viewBox:`0 0 ${{W}} ${{H}}`}});return {{svg,ix:x=>m.l+(x-m.x0)/(m.x1-m.x0)*(W-m.l-m.r),iy:y=>H-m.b-(y-m.y0)/(m.y1-m.y0)*(H-m.t-m.b),W,H,m}};}}
function axes(F,o){{const {{svg,m,W,H}}=F;svg.appendChild(el('line',{{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#cbd5e1'}}));svg.appendChild(el('line',{{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#cbd5e1'}}));
  (o.yticks||[]).forEach(v=>{{const y=F.iy(v);svg.appendChild(el('line',{{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:'#eef0f5'}}));const t=el('text',{{x:m.l-6,y:y+3,'text-anchor':'end','font-size':10,fill:'#9ca3af'}});t.textContent=o.yfmt?o.yfmt(v):v;svg.appendChild(t);}});
  if(o.xlabel){{const t=el('text',{{x:(m.l+W-m.r)/2,y:H-3,'text-anchor':'middle','font-size':10,fill:'#6b7280'}});t.textContent=o.xlabel;svg.appendChild(t);}}}}
function poly(F,pts,c,w){{F.svg.appendChild(el('path',{{d:pts.map((p,i)=>(i?'L':'M')+F.ix(p[0]).toFixed(1)+' '+F.iy(p[1]).toFixed(1)).join(' '),fill:'none',stroke:c,'stroke-width':w||2}}));}}

// ROC
(function(){{const F=frame(480,360,{{l:42,r:14,t:12,b:30,x0:0,x1:1,y0:0,y1:1}});axes(F,{{yticks:[0,.5,1],xlabel:'FPR'}});
  F.svg.appendChild(el('line',{{x1:F.ix(0),y1:F.iy(0),x2:F.ix(1),y2:F.iy(1),stroke:'#d1d5db','stroke-dasharray':'4 4'}}));
  poly(F,D.roc,'#21918c',2.4);document.getElementById('roc').appendChild(F.svg);}})();
// sweep
(function(){{const F=frame(480,360,{{l:42,r:14,t:12,b:30,x0:0,x1:1,y0:0,y1:1}});axes(F,{{yticks:[0,.5,1],yfmt:v=>(v*100)+'%',xlabel:'阈值'}});
  F.svg.appendChild(el('line',{{x1:F.ix(.5),y1:F.m.t,x2:F.ix(.5),y2:360-F.m.b,stroke:'#cbd5e1','stroke-dasharray':'3 3'}}));
  poly(F,D.sweep.map(p=>[p[0],p[1]]),'#21918c',2);poly(F,D.sweep.map(p=>[p[0],p[2]]),'#e8924f',2);document.getElementById('sweep').appendChild(F.svg);}})();
// ops table
{{const lab={{'default_0.5':'默认 0.5','max_f1':'最大F1','fpr<=0.05':'FPR≤5%','fpr<=0.1':'FPR≤10%'}};let h='';
  for(const k in D.ops){{const o=D.ops[k];h+=`<tr><td>${{lab[k]||k}}</td><td class="mono">${{o.threshold}}</td><td>${{(o.accuracy*100).toFixed(1)}}%</td><td>${{(o.precision*100).toFixed(1)}}%</td><td><b>${{(o.recall*100).toFixed(1)}}%</b></td><td>${{(o.fpr*100).toFixed(1)}}%</td></tr>`;}}
  document.getElementById('ops').innerHTML=h;}}

// 混淆矩阵(行=gold,按行归一化着色)+ 分布
D.params.forEach(pa=>{{
  const host=document.querySelector(`[data-cm="${{pa.field}}"]`);if(!host)return;
  const rows=pa.gold_bins,cols=pa.col_bins,cw=Math.max(34,Math.min(54,260/cols.length)),ch=30,ox=88,oy=46;
  const W=ox+cols.length*cw+10,H=oy+rows.length*ch+14;const svg=el('svg',{{viewBox:`0 0 ${{W}} ${{H}}`}});
  cols.forEach((c,j)=>{{const t=el('text',{{x:ox+j*cw+cw/2,y:oy-6,'text-anchor':'end','font-size':8.5,fill:'#6b7280',transform:`rotate(-40 ${{ox+j*cw+cw/2}} ${{oy-6}})`}});t.textContent=c;svg.appendChild(t);}});
  const tt=el('text',{{x:ox+cols.length*cw/2,y:12,'text-anchor':'middle','font-size':10,fill:'#374151'}});tt.textContent='预测 →';svg.appendChild(tt);
  rows.forEach((r,i)=>{{const rs=pa.matrix[i].reduce((a,b)=>a+b,0)||1;
    const lt=el('text',{{x:ox-6,y:oy+i*ch+ch/2+3,'text-anchor':'end','font-size':9,fill:'#374151'}});lt.textContent=r;svg.appendChild(lt);
    cols.forEach((c,j)=>{{const v=pa.matrix[i][j],frac=v/rs;const isNA=c==='N/A';
      const al=0.08+0.85*frac;const fill=isNA?`rgba(180,90,40,${{al}})`:(r===c?`rgba(18,122,78,${{al}})`:`rgba(59,82,139,${{al}})`);
      const rect=el('rect',{{x:ox+j*cw,y:oy+i*ch,width:cw-2,height:ch-2,rx:3,fill:fill}});
      rect.addEventListener('mousemove',e=>showTip(e,`真 ${{r}} → 预测 ${{c}}<br>${{v}} 个 (${{(frac*100).toFixed(0)}}%)`));rect.addEventListener('mouseleave',hideTip);svg.appendChild(rect);
      if(v){{const n=el('text',{{x:ox+j*cw+cw/2-1,y:oy+i*ch+ch/2+3,'text-anchor':'middle','font-size':9,'font-weight':v===Math.max(...pa.matrix[i])?700:400,fill:frac>0.5?'#fff':'#1a1a2e'}});n.textContent=v;svg.appendChild(n);}}
    }});}});
  const yl=el('text',{{x:10,y:oy+rows.length*ch/2,'text-anchor':'middle','font-size':9,fill:'#6b7280',transform:`rotate(-90 10 ${{oy+rows.length*ch/2}})`}});yl.textContent='真实 (gold)';svg.appendChild(yl);
  host.appendChild(svg);
  // gold vs pred 分布条
  const host2=document.querySelector(`[data-dist="${{pa.field}}"]`);
  const allb=pa.col_bins,gd=pa.gold_bins,W2=W,bw=Math.max(28,(W2-50)/allb.length),H2=120;
  const mx=Math.max(...pa.gold_dist,...pa.pred_dist)||1;const s2=el('svg',{{viewBox:`0 0 ${{W2}} ${{H2}}`}});
  const cap=el('text',{{x:8,y:12,'font-size':9.5,fill:'#6b7280'}});cap.textContent='分布: ▣gold ▣pred (该参数被预测成哪些档)';s2.appendChild(cap);
  allb.forEach((b,j)=>{{const x=40+j*bw;const gv=gd.includes(b)?pa.gold_dist[gd.indexOf(b)]:0,pv=pa.pred_dist[j];
    s2.appendChild(el('rect',{{x:x,y:H2-18-(gv/mx*78),width:bw*0.4,height:gv/mx*78,fill:'#94a3b8'}}));
    s2.appendChild(el('rect',{{x:x+bw*0.45,y:H2-18-(pv/mx*78),width:bw*0.4,height:pv/mx*78,fill:b==='N/A'?'#e8924f':'#21918c'}}));
    const t=el('text',{{x:x+bw*0.4,y:H2-6,'text-anchor':'middle','font-size':8,fill:'#6b7280'}});t.textContent=b;s2.appendChild(t);}});
  host2.appendChild(s2);
}});

// 损失(对数)
(function(){{const L=D.loss;if(!L)return;const W=900,H=240,host=document.getElementById('loss');
  const vals=L.train.map(p=>p[1]).concat(L.eval.map(p=>p[1])).filter(v=>v>0);const lo=Math.min(...vals),hi=Math.max(...vals);
  const F=frame(W,H,{{l:50,r:12,t:12,b:28,x0:0,x1:Math.max(...L.train.map(p=>p[0])),y0:Math.log10(lo)-0.12,y1:Math.log10(hi)+0.05}});
  const ticks=[];for(let e=Math.floor(F.m.y1);e>=Math.ceil(F.m.y0);e--)ticks.push(e);
  axes(F,{{yticks:ticks,yfmt:lv=>{{const r=Math.pow(10,lv);return r>=1?r.toFixed(0):(r>=0.01?r.toFixed(2):r.toExponential(0));}},xlabel:'step(对数纵轴)'}});
  poly(F,L.train.map(p=>[p[0],Math.log10(p[1])]),'#3b528b',1.6);
  L.eval.forEach(p=>{{const c=el('circle',{{cx:F.ix(p[0]),cy:F.iy(Math.log10(p[1])),r:4,fill:'#fff',stroke:'#21918c','stroke-width':2}});c.addEventListener('mousemove',e=>showTip(e,`eval loss ${{p[1]}} @${{p[0]}}`));c.addEventListener('mouseleave',hideTip);F.svg.appendChild(c);}});
  host.appendChild(F.svg);host.insertAdjacentHTML('beforeend','<div style="text-align:center;font-size:.8rem;color:#6b7280">蓝线=train,空心圈=每epoch eval</div>');
}})();
</script></body></html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"已生成 {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
print(f"正样本{len(pos)} 生成式检出{dety} · 思考诊断 {think_no}/{think_n} 判NO · 参数混淆矩阵 {len(params)}个")
