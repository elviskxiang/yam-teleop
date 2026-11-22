"""Live 2x2 wall of the four rig cameras in the browser.

    source env.sh && $CAM_PY -m yam_teleop.cam_wall [--port 8090]

Devices are addressed by USB port path (the two Microdias share serial SN0001,
and /dev/videoN numbers shift on replug). Each camera has a list of candidate
nodes; the first one that delivers frames wins (the D415 exposes several
nodes and only some carry colour).
"""
import argparse
import glob
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

try:  # a busy camera makes OpenCV warn on every retry; we report it in the page instead
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def holders(dev):
    """Names of other processes that have this video node open."""
    try:
        target = os.path.realpath(dev)
    except OSError:
        return []
    names = set()
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == target and int(fd.split("/")[2]) != os.getpid():
                names.add(open(f"/proc/{fd.split('/')[2]}/comm").read().strip())
        except OSError:
            pass
    return sorted(names)

BY_PATH = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:"
CAMS = [
    ("D415 (tripod, top)", sorted(glob.glob(BY_PATH + "12:1.3-video-index0"))
     + ["/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_415_Intel_R__RealSense_TM__Depth_Camera_415-video-index0"]),
    ("C922", ["/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_A579FDAF-video-index0"]),
    ("Microdia left (side)", [BY_PATH + "2.1:1.0-video-index0"]),
    ("Microdia right (base plate)", [BY_PATH + "10:1.0-video-index0"]),
]

# V4L2 controls applied every time a camera opens (they reset on replug). The C922's auto-exposure defaults to
# exposure_auto_priority = 1, which lets it halve its frame rate to 15 fps in dim light (a whole demo episode was
# recorded at 15 fps that way); 0 keeps auto-exposure but holds 30 fps, making up with gain.
V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903
CONTROLS = {"C922": {V4L2_CID_EXPOSURE_AUTO_PRIORITY: 0}}


def set_v4l2_ctrl(dev, cid, value):
    import fcntl, struct
    VIDIOC_S_CTRL, VIDIOC_G_CTRL = 0xC008561C, 0xC008561B
    fd = os.open(dev, os.O_RDWR)
    try:
        fcntl.ioctl(fd, VIDIOC_S_CTRL, bytearray(struct.pack("Ii", cid, value)))
        b = bytearray(struct.pack("Ii", cid, 0)); fcntl.ioctl(fd, VIDIOC_G_CTRL, b)
        return struct.unpack("Ii", b)[1]
    finally:
        os.close(fd)


class Cam(threading.Thread):
    def __init__(self, name, candidates):
        super().__init__(daemon=True)
        self.name_, self.candidates = name, candidates
        self.jpg, self.dev, self.fps, self.seq, self.why, self.t_ns = None, None, 0.0, 0, "", 0
        self.cond = threading.Condition()

    def open(self):
        for dev in self.candidates:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for cid, value in CONTROLS.get(self.name_, {}).items():
                try:
                    got = set_v4l2_ctrl(dev, cid, value)
                    if got != value:
                        print(f"{self.name_}: control {cid:#x} is {got}, wanted {value}", flush=True)
                except OSError as e:
                    print(f"{self.name_}: could not set control {cid:#x}: {e}", flush=True)
            ok, frame = cap.read()
            if ok and frame is not None and frame.ndim == 3:
                self.dev = dev
                return cap
            cap.release()
        return None

    def run(self):
        while True:
            cap = self.open()
            if cap is None:
                self.dev, self.jpg = None, None
                busy = sorted({p for d in self.candidates for p in holders(d)})
                why = f"in use by {', '.join(busy)}" if busy else "not found"
                if why != self.why:
                    print(f"{self.name_}: {why}", flush=True)
                self.why = why
                time.sleep(2)
                continue
            if self.why:
                print(f"{self.name_}: streaming", flush=True)
            self.why = ""
            t0, n = time.time(), 0
            while True:
                ok, frame = cap.read()
                t_ns = time.time_ns()          # capture time, sent with each frame (X-Timestamp-Ns) for the logger
                if not ok:
                    cap.release()
                    break
                n += 1
                if time.time() - t0 > 1:
                    self.fps, t0, n = n / (time.time() - t0), time.time(), 0
                jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                with self.cond:
                    self.jpg, self.seq, self.t_ns = jpg, self.seq + 1, t_ns
                    self.cond.notify_all()


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Rig cameras</title>
<style>
body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:6px;padding:6px}
figure{margin:0;position:relative;background:#000;aspect-ratio:4/3}
img{width:100%;height:100%;object-fit:contain;display:block}
figcaption{position:absolute;left:0;top:0;padding:3px 8px;background:#000a;border-bottom-right-radius:4px}
</style></head><body><div class=grid>
%s
</div><script src="/streams.js"></script><script>
camStreams('', document.querySelectorAll('img[data-cam]'));
setInterval(async()=>{const s=await (await fetch('/status')).json();
s.forEach((c,i)=>document.getElementById('c'+i).textContent=
 c.name+'  ·  '+(c.dev?c.dev.split('/').pop()+'  ·  '+c.fps.toFixed(0)+' fps':'NO SIGNAL — '+c.why))},1000);
</script></body></html>"""

# All cameras over ONE connection (/streams: multipart parts tagged X-Cam). A browser allows only 6 connections per host, so
# one <img src=/stream/i> per camera starves the 5th+ stream as soon as two camera pages are open (wall + teleop page).
STREAMS_JS = r"""
function camStreams(base, imgs){
  const urls=[];
  async function run(){
    try{
      const r=await fetch(base+'/streams',{cache:'no-store'}); const rd=r.body.getReader();
      let buf=new Uint8Array(0); const dec=new TextDecoder();
      const cat=(a,b)=>{const c=new Uint8Array(a.length+b.length);c.set(a);c.set(b,a.length);return c};
      const find=(b,from)=>{for(let i=from;i<b.length-3;i++)if(b[i]===13&&b[i+1]===10&&b[i+2]===13&&b[i+3]===10)return i;return -1};
      for(;;){
        const {value,done}=await rd.read(); if(done)break; buf=cat(buf,value);
        for(;;){
          const s=dec.decode(buf.subarray(0,Math.min(buf.length,64))).indexOf('--f'); if(s<0)break;
          const e=find(buf,s); if(e<0)break;
          const h=dec.decode(buf.subarray(s,e)); const n=+(/Content-Length: *(\d+)/i.exec(h)||[])[1]; const cam=+(/X-Cam: *(\d+)/i.exec(h)||[])[1];
          if(buf.length<e+4+n)break;
          const jpg=buf.slice(e+4,e+4+n); buf=buf.slice(e+4+n);
          const img=imgs[cam]; if(!img)continue;
          const u=URL.createObjectURL(new Blob([jpg],{type:'image/jpeg'})); img.src=u; if(urls[cam])URL.revokeObjectURL(urls[cam]); urls[cam]=u;
        }
      }
    }catch(e){}
    setTimeout(run,1000);
  }
  run();
}
"""


def make_handler(cams):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                figs = "\n".join(
                    f'<figure><img data-cam={i}><figcaption id=c{i}>{c.name_}</figcaption></figure>'
                    for i, c in enumerate(cams))
                body = PAGE.replace("%s", figs).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/status":
                import json
                body = json.dumps([{"name": c.name_, "dev": c.dev, "fps": c.fps, "why": c.why} for c in cams]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/streams.js":
                body = STREAMS_JS.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/streams":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.send_header("Access-Control-Allow-Origin", "*")   # the teleop page (port 8770) reads it with fetch()
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seqs = [-1] * len(cams)
                try:
                    while True:
                        sent = False
                        for i, cam in enumerate(cams):
                            with cam.cond:
                                jpg, seq, t_ns = cam.jpg, cam.seq, cam.t_ns
                            if jpg is None or seq == seqs[i]:
                                continue
                            seqs[i] = seq; sent = True
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                                             + b"\r\nX-Cam: " + str(i).encode() + b"\r\nX-Timestamp-Ns: " + str(t_ns).encode()
                                             + b"\r\n\r\n" + jpg + b"\r\n")
                        if not sent:
                            time.sleep(0.003)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path.startswith("/stream/"):
                cam = cams[int(self.path.rsplit("/", 1)[1])]
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        with cam.cond:
                            cam.cond.wait_for(lambda: cam.seq != seq, timeout=5)
                            jpg, seq, t_ns = cam.jpg, cam.seq, cam.t_ns
                        if jpg is None:
                            continue
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\nX-Timestamp-Ns: " + str(t_ns).encode()
                                         + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    cams = [Cam(n, c) for n, c in CAMS]
    try:  # bind first so a second copy fails before touching the cameras
        srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(cams))
    except OSError as e:
        raise SystemExit(f"port {args.port} busy ({e.strerror}) - is cam_wall already running? "
                         f"try http://localhost:{args.port} or pass --port")
    for c in cams:
        c.start()
    print(f"http://localhost:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    # hard exit: daemon threads blocked inside cv2 read() abort the interpreter on normal shutdown
    os._exit(0)
