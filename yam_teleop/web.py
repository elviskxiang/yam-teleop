"""Web page for a lerobot-record session, served from inside the yam_follower plugin (same layout as the old teleop page).

  http://127.0.0.1:8780   cameras (2x2, pinned), follower state, episode state, episode and gripper buttons

The page cannot reach into lerobot-record directly, so:
  * episode state: read from lerobot-record's own log lines ("Recording episode N", "Reset the environment", ...);
  * episode buttons: press the keys lerobot-record listens for (right arrow = save and next, left = re-record,
    Esc = stop), through pynput, exactly as if typed;
  * gripper buttons: set the shared gripper target the leader plugin sends (yam_teleop.arms).
Cameras: the robot's own LeRobot cameras (read_latest), re-encoded to JPEG, all four on one connection (/streams).
"""
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from yam_teleop import arms as A
from yam_teleop.cam_wall import STREAMS_JS

log = logging.getLogger(__name__)


class EpisodeLog(logging.Handler):
    """Tracks lerobot-record's episode state from its log messages. LeRobot 0.5's order per episode:
         "Recording episode N" -> record (until time-out or right/left arrow)
         -> "Reset the environment" -> wait (until reset_time_s or right arrow)
         -> left arrow was pressed during the episode? "Re-record episode", nothing saved, same N again
            otherwise save_episode() (encodes the videos: a few seconds) -> "Recording episode N+1"
       Esc ends the loop; an episode in progress or waiting is still saved (unless it was discarded).
    N is the dataset's episode count, so episodes in the dataset = N while recording episode N.
    States: starting, recording, stopping, waiting, saving, discarding, stopped."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.state, self.episode, self.t0 = "starting", None, None
        self.discard = False            # the current episode will be thrown away (left arrow)
        self.first = None               # dataset episode count when the session started
        self.after_save = None          # dataset episode count after the last save (yam_teleop.record_session)
        self.final = None               # dataset episode count after the session stopped

    @property
    def in_dataset(self):
        if self.final is not None:
            return self.final
        if self.after_save is not None:
            return self.after_save
        return self.episode if self.episode is not None else self.first

    def emit(self, record):
        try:
            m = record.getMessage()
        except Exception:
            return
        if (g := re.match(r"Recording episode (\d+)", m)):
            n = int(g.group(1))
            if self.first is None:
                self.first = n
            self.state, self.episode, self.t0, self.discard, self.after_save = "recording", n, time.time(), False, None
        elif m.startswith("Reset the environment"):
            self.state, self.t0 = "waiting", time.time()
        elif m.startswith("Saving episode"):
            self.state = "saving"
        elif (g := re.match(r"Episode (\d+) saved", m)):
            self.after_save, self.state, self.t0 = int(g.group(1)) + 1, "waiting", time.time()
        elif m.startswith("Re-record episode"):
            self.state = "discarding"
        elif m.startswith("Stop recording"):
            if self.after_save is not None:
                self.final = self.after_save
            elif self.episode is not None:
                self.final = self.episode + (0 if self.discard else 1)
            self.state, self.t0 = "stopped", time.time()

    def key(self, k):
        """A key was pressed from the page: show the transition at once (LeRobot logs it only later)."""
        if self.state == "recording" and k in ("right", "left"):
            self.state, self.t0 = "stopping", time.time()
            self.discard = k == "left"
        elif self.state == "waiting" and k == "right":
            self.state = "discarding" if self.discard else "starting next"
        elif k == "esc":
            self.state = "stopping session"


PAGE = r"""<!doctype html><meta charset=utf-8><title>YAM record</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}body{margin:0;background:#101214;color:#dde;font:13px/1.35 system-ui,sans-serif}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:10px;padding:10px;align-items:start}
.cams{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:6px;position:sticky;top:10px;height:calc(100vh - 20px);z-index:1;background:#101214}
@media(max-width:900px){.wrap{grid-template-columns:1fr}.cams{top:0;height:46vh;padding-bottom:6px}}
.cams figure{margin:0;position:relative;min-height:0;background:#000;border-radius:4px}.cams img{width:100%;height:100%;object-fit:contain;display:block}
.cams figcaption{position:absolute;left:6px;top:4px;font-size:11px;background:#0009;padding:1px 5px;border-radius:3px}
.panel{display:flex;flex-direction:column;gap:10px}.box{background:#181b1f;border:1px solid #2a2f36;border-radius:6px;padding:8px}
h2{font-size:12px;margin:0 0 6px;color:#9ab;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
button{background:#262b33;color:#dde;border:1px solid #3a414c;border-radius:5px;padding:8px 4px;font:inherit;cursor:pointer;min-width:0}
button:hover{background:#313846}button.go{background:#1f5a34;border-color:#3c9a5c}button.rec{background:#8a1f1f;border-color:#d04848}
button.warn{background:#6b2a2a;border-color:#a44}.row{display:flex;gap:5px;flex-wrap:wrap;align-items:center;margin-bottom:5px}.row>button{flex:1}
pre{margin:0;font:11.5px/1.4 ui-monospace,monospace;white-space:pre-wrap}#msg{font-weight:600}.bad{color:#f88}.good{color:#8d8}
.big{font-size:15px;font-weight:700}.recording{color:#ff6b6b}.following{color:#8d8}.held,.engaging{color:#fb4}.fault{color:#f66}
kbd{background:#2a2f36;border-radius:3px;padding:0 4px;font:11px ui-monospace,monospace}
</style>
<div class=wrap><div class=cams id=cams></div>
<div class=panel>
 <div class=box><h2>Follower (right arm)</h2><div id=mode class=big>…</div><div id=blocked class=bad></div></div>
 <div class=box><h2>Episode (lerobot-record)</h2><div id=ep class=big>…</div><pre id=epd></pre>
  <div class=row style="margin-top:6px"><button class=go onclick="post('/key',{k:'right'})">Save &amp; next <kbd>→</kbd></button><button onclick="post('/key',{k:'left'})">Re-record <kbd>←</kbd></button></div>
  <div class=row><button class=warn onclick="if(confirm('Stop the session? Both arms park at rest, then torque is cut.'))post('/key',{k:'esc'})">Stop session <kbd>Esc</kbd></button></div></div>
 <div class=box><h2>Gripper</h2><div class=row><button onclick="post('/grip',{g:1})">Open <kbd>V</kbd></button><button onclick="post('/grip',{g:0.62})">Cube width <kbd>B</kbd></button><button onclick="post('/grip',{g:0.45})">Squeeze <kbd>C</kbd></button><button onclick="post('/grip',{g:0})">Close <kbd>Z</kbd></button></div></div>
 <div class=box><h2>State</h2><div id=msg></div><pre id=st></pre></div>
 <div class=box style="color:#89a">Keys work from any window (LeRobot listens globally): <kbd>→</kbd> save &amp; next, <kbd>←</kbd> re-record, <kbd>Esc</kbd> stop, <kbd>V</kbd><kbd>B</kbd><kbd>C</kbd><kbd>Z</kbd> gripper.</div>
</div></div>
<script src="/streams.js"></script><script>
fetch('/cams').then(r=>r.json()).then(n=>{cams.innerHTML=n.map((c,i)=>`<figure><img data-cam=${i}><figcaption>${c}</figcaption></figure>`).join('');camStreams('',cams.querySelectorAll('img[data-cam]'))});
async function post(p,b){try{const r=await (await fetch(p,{method:'POST',body:JSON.stringify(b||{})})).json();msg.textContent=r.msg;msg.className='good'}catch(e){msg.textContent='request failed: '+e;msg.className='bad'}}
const f=v=>v==null?'-':v.map(x=>x.toFixed(2)).join(' ');
setInterval(async()=>{try{const s=await (await fetch('/status')).json();
 mode.textContent=s.fault?'FAULT: '+s.fault:s.mode.toUpperCase();mode.className='big '+(s.fault?'fault':s.mode);blocked.textContent=s.blocked?'blocked: '+s.blocked:'';
 const e=s.episode;ep.textContent=e.state==='recording'?'● RECORDING episode '+e.episode+'  '+e.elapsed+' s':e.state==='waiting'?(e.discard?'WAITING: episode '+e.episode+' discarded':'WAITING: episode '+e.episode+' saved')+', → starts the next':e.state.toUpperCase();
 ep.className='big '+(e.state==='recording'?'recording':'');epd.textContent=`in dataset ${e.in_dataset??'-'}   saved this session ${e.saved??0}   gripper target ${s.grip.toFixed(2)}`;
 st.textContent=`leader q    ${f(s.leader_q)}\nfollower q  ${f(s.pos)}\ncommand     ${f(s.cmd)}\neffort      ${f(s.effort)}`}catch(e){mode.textContent='page server not reachable (session ended?)';mode.className='big fault'}},250);
</script>"""


class WebPage:
    def __init__(self, robot, port):
        self.robot, self.port = robot, port
        self.eplog = EpisodeLog()
        logging.getLogger().addHandler(self.eplog)
        self.names = list(robot.cameras)
        self.jpg = {n: (None, 0) for n in self.names}
        self._stop = threading.Event()
        self.kb = None
        try:
            from pynput.keyboard import Controller, Key
            self.kb, self.keys = Controller(), {"right": Key.right, "left": Key.left, "esc": Key.esc}
        except Exception as e:
            log.warning(f"page episode buttons unavailable (no keyboard injection): {e}")

    def _encoder(self):
        """~20 fps per camera, JPEG once per frame for all page viewers."""
        while not self._stop.is_set():
            for n in self.names:
                try:
                    f = self.robot.cameras[n].read_latest()
                    ok, b = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        self.jpg[n] = (b.tobytes(), self.jpg[n][1] + 1)
                except Exception:
                    pass
            time.sleep(0.05)

    def status(self):
        r, e = self.robot, self.eplog
        s = r._state or {}
        lst = lambda v: None if v is None else [float(x) for x in v]
        return dict(mode=r.mode, blocked=r.blocked, fault=r.fault, grip=A.grip_target(), pos=lst(s.get("pos")),
                    effort=lst(s.get("effort")), cmd=lst(r.cmd), leader_q=lst(A.leader_q()),
                    episode=dict(state=e.state, episode=e.episode, discard=e.discard, in_dataset=e.in_dataset,
                                 saved=None if e.first is None or e.in_dataset is None else e.in_dataset - e.first,
                                 elapsed=None if e.t0 is None else round(time.time() - e.t0, 1)))

    def start(self):
        page = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ct="application/json"):
                b = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b)))
                self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                try:
                    if self.path == "/status":
                        self._send(page.status())
                    elif self.path == "/cams":
                        self._send(page.names)
                    elif self.path == "/streams.js":
                        self._send(STREAMS_JS.encode(), "text/javascript")
                    elif self.path == "/streams":
                        self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                        self.send_header("Cache-Control", "no-store"); self.end_headers()
                        seen = {n: -1 for n in page.names}
                        while not page._stop.is_set():
                            sent = False
                            for i, n in enumerate(page.names):
                                jpg, seq = page.jpg[n]
                                if jpg is None or seq == seen[n]:
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
                if self.path == "/grip":
                    A.set_grip_target(float(b.get("g", 1.0))); msg = f"gripper target {A.grip_target():.2f}"
                elif self.path == "/key" and page.kb is not None and b.get("k") in page.keys:
                    e = page.eplog
                    allowed = {"recording": ("right", "left", "esc"), "waiting": ("right", "esc")}.get(e.state, ("esc",))
                    if b["k"] not in allowed:
                        msg = f"not now: the session is {e.state}"
                        try:
                            self._send(dict(msg=msg))
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        return
                    k = page.keys[b["k"]]; page.kb.press(k); page.kb.release(k)
                    e.key(b["k"])
                    msg = {"stopping": "episode stopped: saving it now (a few seconds)",
                           "starting next": "starting the next episode",
                           "discarding": "recording the discarded episode again",
                           "stopping session": "ending the session: both arms park, then torque is cut"}.get(e.state, e.state)
                    if e.state == "stopping" and e.discard:
                        msg = "episode stopped and DISCARDED (not saved)"
                else:
                    msg = "not available"
                try:
                    self._send(dict(msg=msg))
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True, name="yam_web").start()
        threading.Thread(target=self._encoder, daemon=True, name="yam_web_jpeg").start()
        log.info(f"YAM record page on http://127.0.0.1:{self.port}")

    def stop(self):
        self._stop.set()
        logging.getLogger().removeHandler(self.eplog)
        try:
            self.srv.shutdown(); self.srv.server_close()
        except Exception:
            pass
