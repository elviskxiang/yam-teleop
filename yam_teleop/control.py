"""Always-on control page for recording with lerobot-record (same layout as the old teleop page).

  source env.sh && PYTHONPATH=$LR_PYTHONPATH $LR_PY -m yam_teleop.control      ->  http://127.0.0.1:8780

Idle: live camera previews (opened here) and a Start button. Start launches ./record.sh <dataset> as a child process
(lerobot-record, LeRobot's standard recorder; the cameras are handed over to it first) and the first episode begins.
During the session the page relays the session page that the yam_follower plugin serves on :8781 (cameras, arm state,
episode state, gripper), and its buttons press the keys lerobot-record listens for:
  recording -> Stop & save (right arrow) / Discard (left arrow)
  waiting   -> Start next episode (right arrow) / End session (Esc: both arms park, torque off)
After each episode the session WAITS (reset period of an hour) until you press Start next episode, like the old page.
When the session ends, the cameras come back to the idle preview.
"""
import argparse
import http.client
import json
import os
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2

from yam_teleop.cam_wall import STREAMS_JS, V4L2_CID_EXPOSURE_AUTO_PRIORITY, set_v4l2_ctrl

ROOT = Path(__file__).resolve().parent.parent
SESSION_PORT = 8781
CAMS = None


def rig_cams():
    from lerobot_robot_yam.config_yam_follower import rig_cameras
    return {k: str(v.index_or_path) for k, v in rig_cameras().items()}


class Preview:
    """Idle camera preview: open every camera, keep the latest JPEG; close() hands them to the session."""

    def __init__(self, cams):
        self.cams, self.jpg, self.stop, self.threads = cams, {n: (None, 0) for n in cams}, threading.Event(), []

    def open(self):
        self.stop.clear(); self.threads = []
        for n, p in self.cams.items():
            t = threading.Thread(target=self._run, args=(n, p), daemon=True); t.start(); self.threads.append(t)

    def _run(self, n, p):
        if n == "c922":
            try:
                set_v4l2_ctrl(p, V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)
            except OSError:
                pass
        def opener():
            c = cv2.VideoCapture(p, cv2.CAP_V4L2)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); c.set(cv2.CAP_PROP_FRAME_WIDTH, 640); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return c
        cap = opener()
        try:
            while not self.stop.is_set():
                ok, f = cap.read() if cap.isOpened() else (False, None)
                if not ok:                                  # busy (session still releasing it) or replugged: reopen
                    cap.release(); time.sleep(0.5)
                    if not self.stop.is_set():
                        cap = opener()
                    continue
                ok, b = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.jpg[n] = (b.tobytes(), self.jpg[n][1] + 1)
        finally:
            cap.release()

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=3)
        self.jpg = {n: (None, 0) for n in self.cams}


class Control:
    def __init__(self, mock):
        self.mock = mock
        self.proc, self.log, self.started, self.dataset = None, deque(maxlen=400), None, None
        self.preview = Preview(rig_cams()); self.preview.open()
        self.lock = threading.Lock()

    def session_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, dataset, task):
        with self.lock:
            if self.session_alive():
                return "a session is already running"
            dataset = "".join(c for c in dataset if c.isalnum() or c in "_-") or "screw_from_rest"
            self.preview.close()
            env = dict(os.environ, RESET_S="3600", WEB_PORT=str(SESSION_PORT), DISPLAY_DATA="false", MOCK="true" if self.mock else "false",
                       TASK=task.strip() or dataset)
            self.log.clear(); self.dataset, self.started = dataset, time.time()
            self.proc = subprocess.Popen([str(ROOT / "record.sh"), dataset, "200"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
            threading.Thread(target=self._pump, daemon=True).start()
            return f"starting a session on {dataset}: connecting the arms (about 10 s), then episode recording begins"

    def _pump(self):
        p = self.proc
        for line in p.stdout:
            self.log.append(line.rstrip())
        p.wait()
        self.log.append(f"--- session ended (exit code {p.returncode}) ---")
        time.sleep(1.0)
        self.preview.open()                                 # cameras back to the idle preview

    def dataset_episodes(self, name):
        try:
            return json.loads((ROOT / "data" / "lerobot" / name / "meta" / "info.json").read_text())["total_episodes"]
        except Exception:
            return 0

    def session(self, method, path, body=None, timeout=2):
        c = http.client.HTTPConnection("127.0.0.1", SESSION_PORT, timeout=timeout)
        try:
            c.request(method, path, body=json.dumps(body) if body is not None else None)
            r = c.getresponse()
            return json.loads(r.read())
        finally:
            c.close()

    def status(self, dataset):
        s = dict(session=self.session_alive(), dataset=self.dataset or dataset, log=list(self.log)[-12:])
        s["episodes_in_dataset"] = self.dataset_episodes(s["dataset"])
        if s["session"]:
            try:
                s["live"] = self.session("GET", "/status")
            except OSError:
                s["live"] = None                            # still connecting
        return s


PAGE = r"""<!doctype html><meta charset=utf-8><title>YAM record</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}body{margin:0;background:#101214;color:#dde;font:13px/1.35 system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:10px;padding:10px;align-items:start}
.cams{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:6px;position:sticky;top:10px;height:calc(100vh - 20px);z-index:1;background:#101214}
@media(max-width:900px){.wrap{grid-template-columns:1fr}.cams{top:0;height:46vh;padding-bottom:6px}}
.cams figure{margin:0;position:relative;min-height:0;background:#000;border-radius:4px}.cams img{width:100%;height:100%;object-fit:contain;display:block}
.cams figcaption{position:absolute;left:6px;top:4px;font-size:11px;background:#0009;padding:1px 5px;border-radius:3px}
.panel{display:flex;flex-direction:column;gap:10px}.box{background:#181b1f;border:1px solid #2a2f36;border-radius:6px;padding:8px}
h2{font-size:12px;margin:0 0 6px;color:#9ab;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
button{background:#262b33;color:#dde;border:1px solid #3a414c;border-radius:5px;padding:9px 4px;font:inherit;cursor:pointer;min-width:0}
button:hover{background:#313846}button:disabled{opacity:.35;cursor:default}button.go{background:#1f5a34;border-color:#3c9a5c}
button.rec{background:#8a1f1f;border-color:#d04848}button.warn{background:#6b2a2a;border-color:#a44}
.row{display:flex;gap:5px;flex-wrap:wrap;align-items:center;margin-bottom:5px}.row>button{flex:1}
input[type=text]{width:100%;box-sizing:border-box;background:#0e1013;color:#dde;border:1px solid #3a414c;border-radius:5px;padding:7px;font:inherit;margin-bottom:5px}
pre{margin:0;font:11.5px/1.4 ui-monospace,monospace;white-space:pre-wrap}#msg{font-weight:600}.bad{color:#f88}.good{color:#8d8}
.big{font-size:15px;font-weight:700}.recording{color:#ff6b6b}.following{color:#8d8}.held,.engaging,.saving,.discarding{color:#fb4}.fault{color:#f66}
#log{max-height:160px;overflow:auto;color:#89a;font-size:10.5px}kbd{background:#2a2f36;border-radius:3px;padding:0 4px;font:11px ui-monospace,monospace}
</style>
<div class=wrap><div class=cams id=cams></div>
<div class=panel>
 <div class=box><h2>Recording</h2><div id=ep class=big>…</div><pre id=epd></pre>
  <div id=idle><input type=text id=ds value="screw_from_rest" placeholder="dataset (data/lerobot/&lt;name&gt;)"><input type=text id=task value="screw_from_rest" placeholder="task label">
   <div class=row><button class=rec onclick="post('/start',{dataset:ds.value,task:task.value})">● Start recording</button></div></div>
  <div id=rec style=display:none><div class=row><button id=b1 class=go onclick="post('/key',{k:'right'})">■ Stop episode</button><button id=b2 onclick="post('/key',{k:'left'})">✕ Stop &amp; discard</button></div>
   <div class=row><button class=warn onclick="if(confirm('End the session? Both arms park at rest, then torque is cut.'))post('/key',{k:'esc'})">End session (park arms)</button></div></div></div>
 <div class=box><h2>Follower (right arm)</h2><div id=mode class=big>-</div><div id=blocked class=bad></div></div>
 <div class=box><h2>Gripper</h2><div class=row><button class=g onclick="post('/grip',{g:1})">Open <kbd>V</kbd></button><button class=g onclick="post('/grip',{g:0.62})">Cube width <kbd>B</kbd></button><button class=g onclick="post('/grip',{g:0.45})">Squeeze <kbd>C</kbd></button><button class=g onclick="post('/grip',{g:0})">Close <kbd>Z</kbd></button></div></div>
 <div class=box><h2>State</h2><div id=msg></div><pre id=st></pre></div>
 <div class=box><h2>Session log</h2><pre id=log></pre></div>
</div></div>
<script src="/streams.js"></script><script>
let camsKey='';
function setCams(n){const k=n.join('|');if(k===camsKey)return;camsKey=k;cams.innerHTML=n.map((c,i)=>`<figure><img data-cam=${i}><figcaption>${c}</figcaption></figure>`).join('');camStreams('',cams.querySelectorAll('img[data-cam]'))}
fetch('/cams').then(r=>r.json()).then(setCams);
async function post(p,b){try{const r=await (await fetch(p,{method:'POST',body:JSON.stringify(b||{})})).json();msg.textContent=r.msg;msg.className=/^(a session|not|no session|request)/.test(r.msg)?'bad':'good'}catch(e){msg.textContent='request failed: '+e;msg.className='bad'}}
const f=v=>v==null?'-':v.map(x=>x.toFixed(2)).join(' ');
setInterval(async()=>{try{const s=await (await fetch('/status?ds='+encodeURIComponent(ds.value))).json();
 idle.style.display=s.session?'none':'';rec.style.display=s.session?'':'none';
 document.querySelectorAll('button.g').forEach(b=>b.disabled=!(s.session&&s.live));
 log.textContent=s.log.join('\n');log.scrollTop=log.scrollHeight;
 const L=s.live;
 if(!s.session){ep.textContent='IDLE: arms off, cameras previewing';ep.className='big';epd.textContent=`dataset ${s.dataset}: ${s.episodes_in_dataset} episodes`;mode.textContent='-';blocked.textContent='';st.textContent='';return}
 if(!L){ep.textContent='STARTING: connecting arms and cameras…';ep.className='big engaging';epd.textContent=`dataset ${s.dataset}: ${s.episodes_in_dataset} episodes`;return}
 const e=L.episode,rec_=e.state==='recording',wait=e.state==='waiting';
 ep.textContent=rec_?'● RECORDING episode '+e.episode+'  '+e.elapsed+' s'
  :wait?(e.discard?'WAITING: episode '+e.episode+' will be DISCARDED, then re-recorded':'WAITING: episode '+e.episode+' will be saved')
  :({stopping:'STOPPING the episode…',saving:'SAVING the episode (a few seconds)…',discarding:'DISCARDING, then re-recording…',
     'stopping session':'ENDING SESSION: arms parking…',stopped:'SESSION ENDED',starting:'STARTING…'}[e.state]||e.state.toUpperCase());
 ep.className='big '+(rec_?'recording':e.state.split(' ')[0]);
 epd.textContent=`dataset ${s.dataset}: ${e.in_dataset??s.episodes_in_dataset} episodes saved   (${e.saved??0} this session)`+(wait?'\nleave the arms where you like: they keep following until the next episode starts':'');
 b1.textContent=wait?'▶ Start next episode':'■ Stop episode';b1.className=wait?'rec':'go';b1.disabled=!(rec_||wait);b2.disabled=!rec_;
 mode.textContent=L.fault?'FAULT: '+L.fault:L.mode.toUpperCase();mode.className='big '+(L.fault?'fault':L.mode);blocked.textContent=L.blocked?'blocked: '+L.blocked:'';
 st.textContent=`gripper target ${L.grip.toFixed(2)}\nleader q    ${f(L.leader_q)}\nfollower q  ${f(L.pos)}\ncommand     ${f(L.cmd)}\neffort      ${f(L.effort)}`}catch(e){}},300);
</script>"""


def main():
    ap = argparse.ArgumentParser(description="always-on recording control page")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--mock", action="store_true", help="sessions use simulated arms (cameras are real)")
    a = ap.parse_args()
    ctl = Control(a.mock)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, body, ct="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            try:
                if self.path.startswith("/status"):
                    ds = unquote(self.path.split("ds=")[1]) if "ds=" in self.path else "screw_from_rest"
                    self._send(ctl.status(ds))
                elif self.path == "/cams":
                    self._send(list(ctl.preview.cams))
                elif self.path == "/streams.js":
                    self._send(STREAMS_JS.encode(), "text/javascript")
                elif self.path == "/streams":
                    self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                    self.send_header("Cache-Control", "no-store"); self.end_headers()
                    names, seen = list(ctl.preview.cams), {}
                    while True:
                        if ctl.session_alive():             # relay the session's stream (same camera order)
                            try:
                                c = http.client.HTTPConnection("127.0.0.1", SESSION_PORT, timeout=5); c.request("GET", "/streams")
                                r = c.getresponse()
                                while ctl.session_alive():
                                    chunk = r.read1(65536)
                                    if not chunk:
                                        break
                                    self.wfile.write(chunk)
                                c.close()
                            except OSError:
                                time.sleep(0.5)
                            continue
                        sent = False
                        for i, n in enumerate(names):
                            jpg, seq = ctl.preview.jpg.get(n, (None, 0))
                            if jpg is None or seen.get(n) == seq:
                                continue
                            seen[n] = seq; sent = True
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                                             + b"\r\nX-Cam: " + str(i).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        if not sent:
                            time.sleep(0.01)
                else:
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/start":
                msg = ctl.start(b.get("dataset", "screw_from_rest"), b.get("task", ""))
            elif self.path in ("/key", "/grip"):
                if not ctl.session_alive():
                    msg = "no session running"
                else:
                    try:
                        msg = ctl.session("POST", self.path, b)["msg"]
                    except OSError:
                        msg = "session still connecting"
            else:
                msg = "unknown"
            try:
                self._send(dict(msg=msg))
            except (BrokenPipeError, ConnectionResetError):
                pass

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); srv.daemon_threads = True
    print(f"YAM record control page: http://127.0.0.1:{a.port}" + ("  (MOCK arms)" if a.mock else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    if ctl.session_alive():
        print("a recording session is still running; it keeps going (end it with Esc)", flush=True)
    os._exit(0)                                         # camera threads blocked in cv2 would abort a normal exit


if __name__ == "__main__":
    main()
