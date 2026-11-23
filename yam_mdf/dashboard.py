"""Live local dashboard for MDF-lite training runs (Weights & Biases style), standard library only.

  source env.sh && PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_mdf.dashboard [--port 8790] [--root outputs/mdf]
  (or plain: python3 -m yam_mdf.dashboard) then open http://127.0.0.1:8790

Reads outputs/mdf/<run>/{config.json, log.jsonl, ckpt_best.pt, ckpt_last.pt} (written by yam_mdf.train) and never
writes anything. GET /api/runs?since={"<run>": [gen, n], ...} returns, per run, its config, status and the log records
after the first n complete lines (gen changes when a log is truncated or replaced: the client then gets everything).
"""
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
LIVE_SEC = 60  # a log modified this recently counts as live


def _clean(v):  # NaN / inf are not valid JSON for the browser
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def _mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return None


class LogCache:
    """log.jsonl files parsed incrementally: only bytes appended since the last read are parsed. One entry per complete
    line (None for a line that does not parse); a partially written last line waits for its newline."""

    def __init__(self):
        self.lock, self.files, self.gens = threading.Lock(), {}, {}

    def read(self, path):
        with self.lock:
            try:
                st = path.stat()
            except OSError:
                return [], 0
            e = self.files.get(path)
            if e is None or st.st_ino != e["ino"] or st.st_size < e["off"]:
                self.gens[path] = gen = self.gens.get(path, -1) + 1
                e = self.files[path] = dict(ino=st.st_ino, off=0, recs=[], gen=gen)
            if st.st_size > e["off"]:
                with open(path, "rb") as f:
                    f.seek(e["off"])
                    buf = f.read(st.st_size - e["off"])
                end = buf.rfind(b"\n") + 1
                for ln in buf[:end].split(b"\n")[:-1]:
                    try:
                        r = json.loads(ln)
                        r = _clean(r) if isinstance(r, dict) else None
                    except ValueError:
                        r = None
                    e["recs"].append(r)
                e["off"] += end
            return e["recs"], e["gen"]


def run_info(d, since, cache, now):
    cfgp, logp = d / "config.json", d / "log.jsonl"
    try:
        cfg = _clean(json.loads(cfgp.read_text()))
        cfg = cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        cfg = {}
    cfg.pop("meta", None)  # large and not shown
    recs, gen = cache.read(logp)
    s = since.get(d.name)
    sg, sn = s if isinstance(s, list) and len(s) == 2 else (None, 0)
    reset = sg != gen or not isinstance(sn, int) or sn > len(recs)
    start = 0 if reset else sn
    last = next((r for r in reversed(recs) if r and isinstance(r.get("step"), (int, float))), None)
    step, total = (last["step"] if last else 0), cfg.get("steps")
    mt = max([t for t in (_mtime(cfgp), _mtime(logp)) if t] or [0])
    status = ("finished" if total and step >= total else "live" if now - mt < LIVE_SEC else "stale")
    return dict(name=d.name, config=cfg, gen=gen, n=len(recs), reset=reset, status=status, mtime=mt, step=step,
                records=[r for r in recs[start:] if r], ckpt={k: _mtime(d / f"ckpt_{k}.pt") for k in ("best", "last")})


def make_handler(root, cache):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path != "/api/runs":
                return self.send(404, b"not found", "text/plain")
            try:
                since = json.loads(parse_qs(u.query).get("since", ["{}"])[0])
                since = since if isinstance(since, dict) else {}
            except ValueError:
                since = {}
            now = time.time()
            dirs = sorted(d for d in root.iterdir() if d.is_dir() and ((d / "config.json").exists() or (d / "log.jsonl").exists())) if root.is_dir() else []
            body = json.dumps(dict(now=now, root=str(root), runs=[run_info(d, since, cache, now) for d in dirs]))
            self.send(200, body.encode(), "application/json")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--root", default=str(ROOT / "outputs" / "mdf"), help="directory holding one sub-directory per run")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(root, LogCache()))
    srv.daemon_threads = True
    print(f"MDF dashboard: http://{a.host}:{a.port}  (runs in {root})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MDF-lite runs</title><style>
:root{--bg:#111318;--panel:#191c23;--panel2:#20242d;--border:#2b303b;--text:#e3e6ec;--muted:#8b93a3;--grid:#262a33;--accent:#5b9bd5;
--live:#4cc27a;--fin:#5b9bd5;--stale:#8b93a3;color-scheme:dark}
:root[data-theme=light]{--bg:#f4f5f7;--panel:#fff;--panel2:#f0f2f5;--border:#dde1e7;--text:#1d2230;--muted:#667085;--grid:#eceef2;
--accent:#2f6fb3;--live:#1f9d55;--fin:#2f6fb3;--stale:#8a93a3;color-scheme:light}
@media (prefers-color-scheme:light){:root:not([data-theme=dark]){--bg:#f4f5f7;--panel:#fff;--panel2:#f0f2f5;--border:#dde1e7;
--text:#1d2230;--muted:#667085;--grid:#eceef2;--accent:#2f6fb3;--live:#1f9d55;--fin:#2f6fb3;--stale:#8a93a3;color-scheme:light}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:8px 16px;background:var(--panel);border-bottom:1px solid var(--border)}
header h1{font-size:15px;margin:0 8px 0 0;font-weight:600}header .grow{flex:1}header label{display:flex;gap:6px;align-items:center;color:var(--muted)}
button{background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:3px 9px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}input[type=range]{width:120px;accent-color:var(--accent)}input[type=checkbox]{accent-color:var(--accent)}
.wrap{display:flex;align-items:flex-start}aside{width:250px;flex:none;position:sticky;top:45px;max-height:calc(100vh - 45px);overflow:auto;
padding:12px;border-right:1px solid var(--border)}aside .hd{display:flex;gap:6px;align-items:center;margin-bottom:8px;color:var(--muted)}
aside .hd span{flex:1;font-weight:600;color:var(--text)}.run{display:flex;gap:8px;align-items:flex-start;padding:6px;border-radius:6px;cursor:pointer}
.run:hover{background:var(--panel2)}.run.off{opacity:.5}.sw{width:11px;height:11px;border-radius:3px;flex:none;margin-top:3px}
.nm{font-weight:600;word-break:break-all}.meta{color:var(--muted);font-size:12px}.st{font-size:10px;text-transform:uppercase;letter-spacing:.04em;
padding:0 5px;border-radius:8px;border:1px solid currentColor;margin-right:4px}.st.live{color:var(--live)}.st.finished{color:var(--fin)}.st.stale{color:var(--stale)}
main{flex:1;min-width:0;padding:12px 16px 40px}h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:14px 0 8px;font-weight:600}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(380px,1fr))}.grid.small{grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px 10px 6px;min-width:0}
.card .ct{display:flex;gap:8px;align-items:baseline}.card h3{margin:0;font-size:13px;font-weight:600;flex:1}.card .note{color:var(--muted);font-size:11px}
.card .ct label{color:var(--muted);font-size:11px;display:flex;gap:3px;align-items:center}.card canvas{display:block;width:100%;height:230px}
.grid.small canvas{height:170px}#tip{position:fixed;z-index:9;pointer-events:none;display:none;background:var(--panel);border:1px solid var(--border);
border-radius:6px;padding:6px 8px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.3);max-width:360px}#tip b{display:block;margin-bottom:3px}
#tip .r{display:flex;gap:6px;align-items:center;white-space:nowrap}#tip .r i{width:9px;height:9px;border-radius:2px;flex:none}#tip .r span{flex:1}
#tip .r em{font-style:normal;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
.tw{overflow:auto;background:var(--panel);border:1px solid var(--border);border-radius:8px}th,td{padding:5px 8px;border-bottom:1px solid var(--border);
text-align:right;white-space:nowrap}th{cursor:pointer;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel)}
th:hover{color:var(--text)}td:first-child,th:first-child{text-align:left}tr.off td{opacity:.45}.empty{color:var(--muted);padding:30px;text-align:center}
@media (max-width:900px){.wrap{flex-direction:column;align-items:stretch}aside{position:static;width:100%;max-height:none;border-right:0;border-bottom:1px solid var(--border)}
.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>MDF-lite runs</h1><label title="EMA smoothing of training curves">smoothing <input id=sm type=range min=0 max=0.99 step=0.01>
<span id=smv></span></label><label><input id=auto type=checkbox checked> auto refresh 5 s</label><button id=now>refresh</button>
<span class=grow></span><span id=upd class=meta></span><button id=theme title="theme">theme: auto</button></header>
<div class=wrap><aside><div class=hd><span>Runs</span><button id=all>all</button><button id=none>none</button></div><div id=runs></div>
<div id=rootp class=meta style="margin-top:10px;word-break:break-all"></div></aside>
<main><div id=empty class=empty style="display:none">No runs found yet.</div>
<h2>Training loss</h2><div class=grid id=g_train></div><h2>Per-modality training loss</h2><div class="grid small" id=g_mod></div>
<h2>Held-out evaluation</h2><div class=grid id=g_eval></div><h2>Optimizer</h2><div class=grid id=g_opt></div>
<div id=othersec style="display:none"><h2>Other metrics</h2><div class=grid id=g_other></div></div>
<h2>Summary</h2><div class=tw><table id=tbl></table></div></main></div><div id=tip></div>
<script>
const $=s=>document.querySelector(s);
const PAL=['#5b9bd5','#f28e2b','#e15759','#4fc1b9','#6abf4b','#e8c22d','#b07aa1','#ff8fa3','#a1785c','#8f8fd8','#3fb0e0','#d4a5ff'];
const LS={get(k,d){try{const v=localStorage.getItem('mdfdash.'+k);return v==null?d:JSON.parse(v)}catch(e){return d}},
  set(k,v){try{localStorage.setItem('mdfdash.'+k,JSON.stringify(v))}catch(e){}}};
const KNOWN=new Set(['step','lr','sec','loss','action','proprio','force','vision','act_rmse','hold_rmse','grip_mae','act_rmse_full','act_rmse_pyramid','best']);
let R={},order=[],colors=LS.get('colors',{}),sel=LS.get('sel',{}),smooth=LS.get('smooth',0.6),hover=null,timer=null,sortK='name',sortD=1,
  noisyPick='latest',T={},rootPath='',serverNow=0;
const num=v=>typeof v==='number'&&isFinite(v);
const fmt=v=>v==null||!isFinite(v)?'–':v===0?'0':Math.abs(v)<1e-3||Math.abs(v)>=1e5?v.toExponential(2):+v.toPrecision(4)+'';
const fstep=v=>v>=1000?(+(v/1000).toPrecision(3))+'k':''+Math.round(v);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function colorOf(n){if(!(n in colors)){let h=0;for(const c of n)h=(h*31+c.charCodeAt(0))>>>0;const used=new Set(Object.values(colors));
  let i=h%PAL.length;for(let k=0;k<PAL.length&&used.has(i);k++)i=(i+1)%PAL.length;colors[n]=i;LS.set('colors',colors)}return PAL[colors[n]%PAL.length]}
const selRuns=()=>order.filter(n=>sel[n]!==false).map(n=>R[n]);
function ema(p,a){if(!a)return p;let s=0,w=0;return p.map(([x,y])=>{s=a*s+(1-a)*y;w=a*w+(1-a);return[x,s/w]})}
const pts=(recs,k)=>recs.filter(x=>num(x[k])&&num(x.step)).map(x=>[x.step,x[k]]);
function trainSeries(k){return selRuns().map(r=>{const raw=pts(r.recs,k);return{label:r.name,color:r.color,pts:ema(raw,smooth),raw:smooth>0?raw:null}})}
function evalSeries(k){return selRuns().map(r=>({label:r.name,color:r.color,pts:pts(r.evals,k),marks:k==='act_rmse'?r.evals.filter(x=>x.best).map(x=>x.step):null}))}
// ---- charts
const charts=[];
function card(gid,id,title,o={}){const el=document.createElement('div');el.className='card';
  el.innerHTML=`<div class=ct><h3>${esc(title)}</h3>${o.logToggle?`<label><input type=checkbox ${o.logy?'checked':''}> log y</label>`:''}${o.toggle?`<button>${o.toggle}</button>`:''}</div><canvas></canvas><div class=note>${o.note||''}</div>`;
  $('#'+gid).appendChild(el);const ch=Object.assign({id,el,cv:el.querySelector('canvas'),kind:'line'},o);ch.logy=LS.get('logy.'+id,!!o.logy);
  const cb=el.querySelector('input');if(cb){cb.checked=ch.logy;cb.onchange=()=>{ch.logy=cb.checked;LS.set('logy.'+id,ch.logy);draw(ch)}}
  const bt=el.querySelector('button');if(bt)bt.onclick=()=>{o.onToggle(bt);draw(ch)};
  ch.cv.addEventListener('mousemove',e=>onMove(ch,e));ch.cv.addEventListener('mouseleave',()=>{hover=null;$('#tip').style.display='none';drawAll()});
  charts.push(ch);return ch}
card('g_train','loss','train loss (total)',{logToggle:1,logy:1,series:()=>trainSeries('loss'),note:'EMA-smoothed; raw shown faintly'});
for(const m of['action','proprio','force','vision'])card('g_mod',m,m,{logToggle:1,logy:1,series:()=>trainSeries(m)});
card('g_eval','act','held-out act_rmse (rad)',{dots:1,series:()=>{const S=[];for(const r of selRuns()){
  S.push({label:r.name,color:r.color,pts:pts(r.evals,'act_rmse'),marks:r.evals.filter(x=>x.best).map(x=>x.step)});
  S.push({label:r.name+' hold',color:r.color,pts:pts(r.evals,'hold_rmse'),dash:[5,4],alpha:.7,width:1.2})}return S},
  note:'solid: model (diamond = new best checkpoint) · dashed: repeat-last-action baseline (hold_rmse) · lower is better'});
card('g_eval','sched','act_rmse: full vs pyramid sampling',{dots:1,series:()=>{const S=[];for(const r of selRuns()){
  const s=r.config.sample_schedule||'full',o=s==='full'?'pyramid':'full',d={full:[],pyramid:[6,3]};
  S.push({label:r.name+' '+s,color:r.color,pts:pts(r.evals,'act_rmse'),dash:d[s]});
  const k='act_rmse_'+o;if(r.evals.some(x=>num(x[k])))S.push({label:r.name+' '+o,color:r.color,pts:pts(r.evals,k),dash:d[o],alpha:.8})}return S},
  note:'solid: full (all future tokens together) · dashed: pyramid (near future first); act_rmse is scored with the run\'s sample_schedule'});
const noisy=card('g_eval','noisy','noisy observations: aware vs unaware',{kind:'bars',toggle:'latest eval',onToggle:b=>{noisyPick=noisyPick==='latest'?'best':'latest';b.textContent=noisyPick+' eval'},
  note:'act_rmse with noise σ on the history · solid: model told the noise level (aware) · faded: not told (unaware) · line: clean act_rmse'});
card('g_eval','grip','grip_mae (gripper, 0..1)',{dots:1,series:()=>evalSeries('grip_mae')});
card('g_opt','lr','learning rate',{zero:1,series:()=>selRuns().map(r=>({label:r.name,color:r.color,pts:pts(r.recs,'lr')}))});
function ensureOther(){const ks=new Set();for(const r of selRuns())for(const x of r.recs)for(const k in x)if(!KNOWN.has(k)&&!k.startsWith('noisy_')&&num(x[k]))ks.add(k);
  for(const k of ks)if(!charts.some(c=>c.id==='x.'+k)){card('g_other','x.'+k,k,{logToggle:1,dots:1,series:()=>selRuns().map(r=>{
    const all=r.recs.filter(x=>num(x.step)),raw=pts(all,k),dense=raw.length>all.length/2;
    return{label:r.name,color:r.color,pts:dense?ema(raw,smooth):raw,raw:dense&&smooth>0?raw:null}})})}
  $('#othersec').style.display=charts.some(c=>c.id.startsWith('x.'))?'':'none'}
function niceTicks(a,b,n){if(!(b>a))return[a];let st=Math.pow(10,Math.floor(Math.log10((b-a)/n)));const e=(b-a)/n/st;
  st*=e>=7.5?10:e>=3.5?5:e>=1.5?2:1;const t=[];for(let v=Math.ceil(a/st)*st;v<=b+st*1e-9;v+=st)t.push(Math.abs(v)<st*1e-9?0:v);return t}
function logTicks(l0,l1){const t=[],d=[];for(let k=Math.floor(l0);k<=Math.ceil(l1);k++)for(const m of[1,2,5]){const v=m*Math.pow(10,k),l=Math.log10(v);
  if(l>=l0-1e-9&&l<=l1+1e-9){t.push(v);if(m===1)d.push(v)}}if(t.length<=8)return t;const s=Math.ceil(d.length/6);return d.length>=2?d.filter((_,i)=>i%s===0):t.filter(v=>/^[15]/.test(v.toExponential()))}
function setup(ch){const c=ch.cv,W=c.clientWidth,H=c.clientHeight,d=window.devicePixelRatio||1;
  if(c.width!==Math.round(W*d)||c.height!==Math.round(H*d)){c.width=Math.round(W*d);c.height=Math.round(H*d)}
  const g=c.getContext('2d');g.setTransform(d,0,0,d,0,0);g.clearRect(0,0,W,H);g.font='11px system-ui,sans-serif';return[g,W,H]}
function xMax(){let m=0;for(const r of selRuns())m=Math.max(m,r.step||0);return m||1}
function noData(g,W,H,ch){g.fillStyle=T.muted;g.textAlign='center';g.fillText('no data yet',W/2,H/2);ch.lay=null}
function axes(g,ch,L,Tp,pw,ph,yt,sy,xt,sx){g.strokeStyle=T.grid;g.fillStyle=T.muted;g.lineWidth=1;g.textAlign='right';
  for(const v of yt){const y=Math.round(sy(v))+.5;g.beginPath();g.moveTo(L,y);g.lineTo(L+pw,y);g.stroke();g.fillText(fmt(v),L-5,y+3.5)}
  g.textAlign='center';for(const[v,lab]of xt){const x=Math.round(sx(v))+.5;g.beginPath();g.moveTo(x,Tp);g.lineTo(x,Tp+ph);g.stroke();g.fillText(lab,x,Tp+ph+14)}}
function drawLine(ch){const[g,W,H]=setup(ch),S=ch.S=ch.series(),L=50,Tp=8,pw=W-L-10,ph=H-Tp-22,ok=v=>num(v)&&(!ch.logy||v>0);
  let y0=Infinity,y1=-Infinity;for(const s of S)for(const a of[s.pts,s.raw||[]])for(const p of a)if(ok(p[1])){y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1])}
  if(!(y1>=y0))return noData(g,W,H,ch);
  if(ch.logy){y0=Math.log10(y0);y1=Math.log10(y1)}if(ch.zero&&!ch.logy)y0=Math.min(0,y0);
  if(y1-y0<1e-12){const m=ch.logy?.5:(Math.abs(y1)||1)*.1;y0-=m;y1+=m}const pad=(y1-y0)*.05;y1+=pad;if(!(ch.zero&&y0===0))y0-=pad;
  const X1=xMax(),tf=v=>ch.logy?Math.log10(v):v,sx=x=>L+x/X1*pw,sy=v=>Tp+(1-(tf(v)-y0)/(y1-y0))*ph;
  axes(g,ch,L,Tp,pw,ph,ch.logy?logTicks(y0,y1):niceTicks(y0,y1,5),sy,niceTicks(0,X1,6).map(v=>[v,fstep(v)]),sx);
  g.save();g.beginPath();g.rect(L,Tp,pw,ph);g.clip();g.lineJoin='round';
  const path=a=>{g.beginPath();let f=1;for(const[x,y]of a){if(!ok(y)){f=1;continue}f?g.moveTo(sx(x),sy(y)):g.lineTo(sx(x),sy(y));f=0}g.stroke()};
  for(const s of S){g.strokeStyle=g.fillStyle=s.color;g.lineWidth=1;if(s.raw){g.globalAlpha=.2;g.setLineDash([]);path(s.raw)}
    g.globalAlpha=s.alpha||1;g.setLineDash(s.dash||[]);g.lineWidth=s.width||1.7;path(s.pts);g.setLineDash([]);
    if(ch.dots&&s.pts.length<200)for(const[x,y]of s.pts)if(ok(y)){g.beginPath();g.arc(sx(x),sy(y),2.2,0,7);g.fill()}
    if(s.marks)for(const[x,y]of s.pts)if(ok(y)&&s.marks.includes(x)){const X=sx(x),Y=sy(y);g.beginPath();g.moveTo(X,Y-6);g.lineTo(X+5,Y);g.lineTo(X,Y+6);g.lineTo(X-5,Y);
      g.closePath();g.fill();g.strokeStyle=T.bg;g.lineWidth=1;g.stroke();g.strokeStyle=s.color}}
  g.globalAlpha=1;g.restore();
  if(hover!=null){g.strokeStyle=T.muted;g.setLineDash([3,3]);g.beginPath();g.moveTo(Math.round(sx(hover))+.5,Tp);g.lineTo(Math.round(sx(hover))+.5,Tp+ph);g.stroke();g.setLineDash([])}
  ch.lay={L,pw,X1}}
function noisyRows(){return selRuns().map(r=>{const rec=noisyPick==='best'?r.best:r.evals[r.evals.length-1];return rec?{r,rec}:null}).filter(Boolean)}
function drawBars(ch){const[g,W,H]=setup(ch),rows=noisyRows(),sg=new Set();
  for(const{rec}of rows)for(const k in rec){const m=k.match(/^noisy_s(.+)_(aware|unaware)$/);if(m&&num(rec[k]))sg.add(m[1])}
  const sig=[...sg].sort((a,b)=>a-b),L=50,Tp=8,pw=W-L-10,ph=H-Tp-22;if(!sig.length)return noData(g,W,H,ch);
  let y1=0;for(const{rec}of rows)for(const k in rec)if((k.startsWith('noisy_s')||k==='act_rmse')&&num(rec[k]))y1=Math.max(y1,rec[k]);y1*=1.08;
  const sy=v=>Tp+(1-v/y1)*ph,gw=pw/sig.length;axes(g,ch,L,Tp,pw,ph,niceTicks(0,y1,5),sy,sig.map((s,i)=>[i+.5,'σ = '+s]),x=>L+x*gw);
  const n=rows.length,bw=Math.max(2,Math.min(22,gw*.8/(2*n)));
  sig.forEach((s,i)=>{const x0=L+i*gw+(gw-bw*2*n)/2;rows.forEach(({r,rec},j)=>{const x=x0+j*2*bw;g.fillStyle=r.color;
    for(const[q,a]of[['aware',1],['unaware',.4]]){const v=rec[`noisy_s${s}_${q}`];if(!num(v))continue;g.globalAlpha=a;
      g.fillRect(x+(q==='aware'?0:bw)+.5,sy(v),bw-1,sy(0)-sy(v))}g.globalAlpha=1;
    if(num(rec.act_rmse)){g.strokeStyle=T.text;g.lineWidth=1.5;g.beginPath();g.moveTo(x,sy(rec.act_rmse));g.lineTo(x+2*bw,sy(rec.act_rmse));g.stroke()}})});
  ch.lay={L,pw,gw,sig,rows}}
const draw=ch=>ch.kind==='bars'?drawBars(ch):drawLine(ch);
function readTheme(){const cs=getComputedStyle(document.documentElement);for(const k of['grid','muted','text','bg'])T[k]=cs.getPropertyValue('--'+k).trim()}
let raf=0;function drawAll(){if(raf)return;raf=requestAnimationFrame(()=>{raf=0;readTheme();charts.forEach(draw)})}
function near(a,x){let lo=0,hi=a.length-1;if(hi<0)return null;while(hi-lo>1){const m=(lo+hi)>>1;a[m][0]<x?lo=m:hi=m}return Math.abs(a[lo][0]-x)<=Math.abs(a[hi][0]-x)?a[lo]:a[hi]}
function onMove(ch,e){if(!ch.lay)return;const rc=ch.cv.getBoundingClientRect(),x=e.clientX-rc.left,tip=$('#tip');let h='';
  if(ch.kind==='bars'){const i=Math.floor((x-ch.lay.L)/ch.lay.gw),s=ch.lay.sig[i];if(s==null){tip.style.display='none';return}
    h=`<b>σ = ${s} (${noisyPick} eval)</b>`+ch.lay.rows.map(({r,rec})=>{const a=rec[`noisy_s${s}_aware`],u=rec[`noisy_s${s}_unaware`];
      return`<div class=r><i style="background:${r.color}"></i><span>${esc(r.name)} <em>@${fstep(rec.step)}</em></span>aware ${fmt(a)} · unaware ${fmt(u)} · clean ${fmt(rec.act_rmse)}${num(a)&&num(u)?` <em>Δ ${fmt(u-a)}</em>`:''}</div>`}).join('')}
  else{const st=Math.max(0,Math.min(1,(x-ch.lay.L)/ch.lay.pw))*ch.lay.X1;let best=null;
    for(const s of ch.S){const p=near(s.pts,st);if(p&&(best==null||Math.abs(p[0]-st)<Math.abs(best-st)))best=p[0]}if(best==null)return;hover=best;drawAll();
    h=`<b>step ${best}</b>`+ch.S.map(s=>{const p=near(s.pts,best);if(!p)return'';const rw=s.raw&&near(s.raw,best);
      return`<div class=r><i style="background:${s.color};opacity:${s.alpha||1}"></i><span>${esc(s.label)}${p[0]!==best?` <em>@${fstep(p[0])}</em>`:''}</span>${fmt(p[1])}${rw&&rw[0]===p[0]&&smooth>0?` <em>raw ${fmt(rw[1])}</em>`:''}</div>`}).join('')}
  tip.innerHTML=h;tip.style.display='block';const tw=tip.offsetWidth,th=tip.offsetHeight;
  tip.style.left=Math.min(e.clientX+14,innerWidth-tw-8)+'px';tip.style.top=(e.clientY+14+th>innerHeight?e.clientY-th-10:e.clientY+14)+'px'}
// ---- sidebar + table
function sidebar(){$('#runs').innerHTML=order.map(n=>{const r=R[n],tot=r.config.steps,last=r.recs[r.recs.length-1];
  const eta=r.status==='live'&&last&&num(last.sec)&&last.step&&tot>last.step?' · ETA '+Math.max(1,Math.round(last.sec/last.step*(tot-last.step)/60))+' min':'';
  return`<label class="run ${sel[n]===false?'off':''}"><input type=checkbox data-n="${esc(n)}" ${sel[n]!==false?'checked':''}><span class=sw style="background:${r.color}"></span>
  <div><div class=nm>${esc(n)}</div><div class=meta><span class="st ${r.status}">${r.status}</span>${fstep(r.step||0)} / ${tot?fstep(tot):'?'}${eta}</div>
  <div class=meta>best act_rmse ${r.best?fmt(r.best.act_rmse)+' @'+fstep(r.best.step):'–'}</div></div></label>`}).join('');
  $('#runs').querySelectorAll('input').forEach(c=>c.onchange=()=>{sel[c.dataset.n]=c.checked;LS.set('sel',sel);render()})}
const COLS=[['name','run'],['status','status'],['noise','noise_mode'],['sched','schedule'],['step','step'],['total','steps'],['batch','batch'],['lr','lr'],
  ['tr','train eps'],['va','val eps'],['best','best act_rmse'],['bstep','@step'],['last','last act_rmse'],['hold','hold_rmse'],['bgrip','grip_mae@best'],
  ['other','other sched@best'],['time','elapsed'],['ck','ckpt best / last']];
const ago=t=>t?(s=>s<90?Math.round(s)+'s':s<5400?Math.round(s/60)+'m':s<172800?Math.round(s/3600)+'h':Math.round(s/86400)+'d')(serverNow-t)+' ago':'–';
function rowOf(r){const c=r.config,b=r.best||{},l=r.evals[r.evals.length-1]||{},last=r.recs[r.recs.length-1]||{},s=c.sample_schedule||'full',o=s==='full'?'pyramid':'full';
  const eps=v=>Array.isArray(v)?v.length:v;
  return{name:r.name,status:r.status,noise:c.noise_mode,sched:(c.sample_schedule||'full*')+(s==='pyramid'&&c.uncertainty!=null?' u='+c.uncertainty:''),step:r.step,total:c.steps,batch:c.batch,lr:c.lr,
    tr:eps(c.train_episodes),va:eps(c.val_episodes),best:b.act_rmse,bstep:b.step,last:l.act_rmse,hold:b.hold_rmse,bgrip:b.grip_mae,other:b['act_rmse_'+o],time:last.sec,
    ck:r.ckpt}}
function table(){const rows=order.map(n=>[R[n],rowOf(R[n])]);
  rows.sort((A,B)=>{const a=A[1][sortK],b=B[1][sortK];if(a==null)return 1;if(b==null)return-1;return(typeof a==='number'?a-b:String(a).localeCompare(String(b)))*sortD});
  const cell=(k,v)=>k==='name'?`<span class=sw style="display:inline-block;margin-right:6px;background:${R[v].color}"></span>${esc(v)}`:k==='status'?`<span class="st ${v}">${v}</span>`:
    k==='ck'?ago(v.best)+' / '+ago(v.last):k==='time'?(num(v)?(v/60).toFixed(1)+' min':'–'):typeof v==='number'?(['step','total','bstep','batch','tr','va'].includes(k)?String(v):fmt(v)):esc(v??'–');
  $('#tbl').innerHTML='<thead><tr>'+COLS.map(([k,t])=>`<th data-k="${k}">${t}${sortK===k?(sortD>0?' ▲':' ▼'):''}</th>`).join('')+'</tr></thead><tbody>'+
    rows.map(([r,o])=>`<tr class="${sel[r.name]===false?'off':''}">`+COLS.map(([k])=>`<td>${cell(k,o[k])}</td>`).join('')+'</tr>').join('')+'</tbody>';
  $('#tbl').querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortD=sortK===k?-sortD:1;sortK=k;table()})}
function render(){$('#empty').style.display=order.length?'none':'';$('#rootp').textContent=rootPath;sidebar();ensureOther();table();drawAll()}
// ---- data
async function refresh(){const since={};for(const n in R)since[n]=[R[n].gen,R[n].n];
  try{const res=await fetch('api/runs?since='+encodeURIComponent(JSON.stringify(since)),{cache:'no-store'}),d=await res.json();serverNow=d.now;rootPath=d.root;
    const seen=new Set();for(const x of d.runs){seen.add(x.name);let r=R[x.name];if(!r||x.reset)r=R[x.name]={name:x.name,recs:[],color:colorOf(x.name)};
      r.recs.push(...x.records);Object.assign(r,{gen:x.gen,n:x.n,config:x.config||{},status:x.status,step:x.step,ckpt:x.ckpt});
      r.evals=r.recs.filter(z=>num(z.act_rmse)&&num(z.step));r.best=r.evals.reduce((a,z)=>!a||z.act_rmse<a.act_rmse?z:a,null)}
    for(const n in R)if(!seen.has(n))delete R[n];order=Object.keys(R).sort();
    $('#upd').textContent='updated '+new Date().toLocaleTimeString();render()}catch(e){$('#upd').textContent='refresh failed: '+e.message}}
function schedule(){clearInterval(timer);timer=$('#auto').checked?setInterval(refresh,5000):null}
$('#sm').value=smooth;$('#smv').textContent=(+smooth).toFixed(2);
$('#sm').oninput=e=>{smooth=+e.target.value;$('#smv').textContent=smooth.toFixed(2);LS.set('smooth',smooth);drawAll()};
$('#auto').onchange=schedule;$('#now').onclick=refresh;
$('#all').onclick=()=>{order.forEach(n=>sel[n]=true);LS.set('sel',sel);render()};$('#none').onclick=()=>{order.forEach(n=>sel[n]=false);LS.set('sel',sel);render()};
let theme=LS.get('theme','auto');const applyTheme=()=>{theme==='auto'?document.documentElement.removeAttribute('data-theme'):document.documentElement.dataset.theme=theme;
  $('#theme').textContent='theme: '+theme;drawAll()};
$('#theme').onclick=()=>{theme={auto:'dark',dark:'light',light:'auto'}[theme];LS.set('theme',theme);applyTheme()};
matchMedia('(prefers-color-scheme: light)').addEventListener('change',drawAll);new ResizeObserver(drawAll).observe($('main'));
applyTheme();refresh();schedule();
</script></body></html>"""

if __name__ == "__main__":
    main()
