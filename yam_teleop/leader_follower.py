"""Right-arm teleop: the LEFT YAM is the leader, the RIGHT YAM follows it. Browser page for engaging and recording episodes.

  source env.sh && $HW_PY -m yam_teleop.leader_follower [--mock] [--port 8780]

The leader floats under gravity compensation (URDF model, all six joints incl. the wrist) and is never commanded while you
work: you move it by hand. The arms are mounted identically 0.61 m apart, so the follower copies the leader's joint angles
and its gripper makes the same motion in the world, shifted 0.61 m toward -y.
The follower's gripper is driven from the page / keys (the leader's lead-screw gripper cannot be pushed by hand); its force is
capped by the rig's LeashGripper (13 N), which the stock servo is not (it burst the cube twice).

Follower safety, every control tick (100 Hz):
  * starts FLOATING; 'Engage' ramps it at 0.35 rad/s from where it is to the leader's pose, then it follows;
  * the target is clipped to the joint limits and slewed at most --max-vel rad/s per joint, and sent with its velocity as a
    feed-forward (--vff) so the motors' damping does not make the follower trail the leader;
  * a target that the rig model (robotics-rubiks-cube poc/generated/scene.xml) says touches the table, the floor or the leader,
    or puts the grasp point below the table top, is not sent: the follower holds and the page says why;
  * if the follower falls > 0.3 rad behind its command (it hit something) it holds where it is and stops following; an
    episode being recorded is discarded. Engage again to continue.
'Park and release' ramps both arms to rest, then cuts torque. SIGINT / SIGTERM park and release too. Never kill -9 this
process while an arm is raised.

Episodes: Start / Stop (space) / Discard (X) on the page. Only the state changes here; yam_teleop.record writes the files.
"""
import argparse
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np

from yam_teleop.gravity import GRAV_FACTOR, UrdfGravity

LEADER, FOLLOWER = "left", "right"
DT = 0.01
# the gains that make the real arm track like the rig's sim (robotics-rubiks-cube scripts/hw/yam_io.py: SIM_KP/KD x 'match')
KP = np.array([80.0, 80.0, 80.0, 30.0, 15.0, 15.0])
KD = np.array([11.0, 11.0, 11.0, 1.5, 2.25, 2.25])
J_LO = np.array([-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944 + np.radians(5)])
J_HI = np.array([3.05433, 3.66519, 3.66519, 1.5708, 1.5708, 2.0944 - np.radians(5)])
ENGAGE_VEL, TRACK_STOP, ENGAGED_TOL = 0.35, 0.30, 0.02
FLOOR_Z = 0.735                       # grasp point may not go below the table top (m, world)
GRIP_VEL = 1.5                        # normalized gripper units per second
VFF_ALPHA = 0.25                      # low-pass on the velocity feed-forward (~40 ms at 100 Hz)
VFF_GAIN = 1.0                        # fraction of the command velocity fed forward (0 = position-only, the old behaviour)
GRIP_PRESETS = {"open": 1.0, "cube": 0.62, "squeeze": 0.45, "close": 0.0}   # close: fully shut, still force-capped (13 N)


class Collision:
    """Rig model: does the follower at q touch the table / floor / the leader at ql?"""

    def __init__(self, scene_xml):
        self.m = mujoco.MjModel.from_xml_path(scene_xml)
        self.d = mujoco.MjData(self.m)
        self.qadr = {s: np.array([self.m.joint(f"{s}_joint{i}").qposadr[0] for i in range(1, 7)]) for s in (LEADER, FOLLOWER)}
        fq = int(self.m.joint("cube_free").qposadr[0])
        self.d.qpos[fq:fq + 3] = (3.0, 3.0, 1.0)            # the real cube is not modelled: keep the sim cube out of it
        self.base = self.d.qpos.copy()
        self.side = [self.m.body(self.m.geom_bodyid[g]).name.split("_")[0] for g in range(self.m.ngeom)]
        self.env = {self.m.geom("table").id, self.m.geom("floor").id}
        self.pinch = self.m.site(f"{FOLLOWER}_pinch").id

    def check(self, qf, ql):
        d = self.d
        d.qpos[:] = self.base; d.qpos[self.qadr[FOLLOWER]] = qf; d.qpos[self.qadr[LEADER]] = ql
        mujoco.mj_forward(self.m, d)
        if d.site_xpos[self.pinch][2] < FLOOR_Z:
            return f"grasp point would go below the table top ({d.site_xpos[self.pinch][2] * 1000:.0f} mm)"
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > -0.001:
                continue
            a, b = self.side[c.geom[0]], self.side[c.geom[1]]
            if {a, b} == {LEADER, FOLLOWER}:
                return "follower would touch the leader arm"
            if FOLLOWER in (a, b) and (c.geom[0] in self.env or c.geom[1] in self.env):
                return "follower would touch the table"
        return None

    def ee(self):
        return [round(float(v), 4) for v in self.d.site_xpos[self.pinch]]


class MockLeader:
    """--mock: a leader that moves by itself (slow sine on every joint) so the follower and the recorder have motion."""

    def __init__(self):
        self.t0 = time.time()

    def get_observations(self):
        t = time.time() - self.t0
        q = np.array([0.4, 0.9, 0.9, -0.3, 0.3, 0.5]) * np.sin(0.4 * t) + np.array([0.0, 0.9, 0.9, 0.0, 0.0, 0.0])
        return dict(joint_pos=q, joint_vel=np.zeros(6), joint_eff=np.zeros(6), gripper_pos=np.ones(1),
                    gripper_vel=np.zeros(1), gripper_eff=np.zeros(1))

    def enter_gravity_comp_idle(self):
        pass

    def command_joint_pos(self, q):
        pass


class Teleop:
    def __init__(self, mock, scene_xml, grav_factor=GRAV_FACTOR, max_vel=1.5, grip_N=13.0, task="", friction=False):
        self.mock, self.max_vel, self.grav_factor = mock, float(max_vel), np.asarray(grav_factor, float)
        self.col = Collision(scene_xml)
        self.gmodel = UrdfGravity()                        # telemetry (control-loop thread only)
        self.lock = threading.Lock()
        self.mode, self.msg, self.blocked = "idle", "follower floating: press Engage", None
        self.cmd = None                                     # follower command (7): q6 + gripper 0..1; None = floating
        self.grip_target = 1.0
        self.vff = np.zeros(6)                              # follower joint velocity feed-forward (rad/s)
        self.ep = dict(id=1, state="idle", task=task, t_start=None, saved=0, discarded=0, last=None)
        self.snap = None
        if mock:
            import yam_io as Y
            self.arm = {LEADER: MockLeader(), FOLLOWER: Y.MockArm(np.r_[0.0, 0.9, 0.9, 0, 0, 0, 1.0])}
        else:
            import yam_io as Y
            from i2rt.robots.get_robot import get_yam_robot
            from i2rt.robots.utils import ArmType, GripperType
            self.arm = {}
            for s in (LEADER, FOLLOWER):
                r = get_yam_robot(channel=f"can_{s}", arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, zero_gravity_mode=True,
                                  gravity_comp_factor=self.grav_factor, enable_auto_recovery=True, gripper_kp=None, gripper_kd=0.1)
                r.kdl = UrdfGravity()                       # gravity from the URDF instead of the SDK's MJCF
                r.temp_record_flag = True
                r.use_coulomb_friction = bool(friction) and s == LEADER
                if s == FOLLOWER:
                    r._gripper_force_limiter = Y.LeashGripper(r, grip_N, Y.GripperMap(), 0.031); r._limit_gripper_force = grip_N
                    r.update_kp_kd(np.r_[KP, r._kp[6]], np.r_[KD, r._kd[6]])   # stored; applied with the first command
                self.arm[s] = r
        self.released = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ---- control loop
    def _obs(self, s):
        o = self.arm[s].get_observations()
        return o, np.asarray(o["joint_pos"], float), float(np.ravel(o["gripper_pos"])[0])

    def _loop(self):
        nxt = time.time()
        while not self._stop.is_set():
            try:
                with self.lock:
                    if not self.released:
                        self._tick()
            except Exception as e:                          # never let the loop die silently with the follower engaged
                with self.lock:
                    why = f"error in control loop: {type(e).__name__}: {e}"
                    if self.mode in ("engaging", "following"):
                        try:
                            self._hold(why)
                        except Exception:
                            self.mode, self.msg = "held", why
                    else:
                        self.msg = why
            nxt += DT
            time.sleep(max(0.0, nxt - time.time()))
            if time.time() - nxt > 0.1:
                nxt = time.time()

    def _tick(self):
        lo, lq, lg = self._obs(LEADER)
        fo, fq, fg = self._obs(FOLLOWER)
        self.blocked = None
        if self.mode in ("engaging", "following"):
            target = np.clip(lq, J_LO, J_HI)
            vel = ENGAGE_VEL if self.mode == "engaging" else self.max_vel
            step = np.clip(target - self.cmd[:6], -vel * DT, vel * DT)
            q_new = self.cmd[:6] + step
            g_new = self.cmd[6] + np.clip(self.grip_target - self.cmd[6], -GRIP_VEL * DT, GRIP_VEL * DT)
            hit = self.col.check(q_new, lq)
            if hit:
                self.blocked = hit                          # hold the joints this tick; the gripper may still move
                q_new = self.cmd[:6]
            # velocity feed-forward: the motors run MIT mode, tau = kp (q* - q) + kd (v* - v). With v* = 0 the damping
            # brakes every motion and the arm trails a moving target by kd v / kp (wrist roll: 0.26 rad at 1.5 rad/s,
            # just under the 0.3 rad stop). v* = the command's own velocity, low-passed against leader-encoder jitter.
            v_raw = (q_new - self.cmd[:6]) / DT
            self.vff = (1 - VFF_ALPHA) * self.vff + VFF_ALPHA * v_raw if not hit else np.zeros(6)
            self.cmd = np.r_[q_new, g_new]
            self._send(self.cmd, self.vff * VFF_GAIN)
            if self.mode == "engaging" and np.abs(target - q_new).max() < ENGAGED_TOL:
                self.mode, self.msg = "following", "following the leader"
            lag = float(np.abs(self.cmd[:6] - fq).max())
            if lag > TRACK_STOP:
                self._hold(f"STOPPED: follower joint {int(np.abs(self.cmd[:6] - fq).argmax()) + 1} fell {lag:.2f} rad behind "
                           f"(touching something?). Holding; press Engage to continue.", fq=fq, fg=fg)
        self.snap = self._snapshot(lo, lq, fo, fq)

    def _send(self, cmd, vff):
        """Follower command: joint positions (+ gripper) with a joint velocity feed-forward (gripper: 0, the force cap
        relies on its damping)."""
        a = self.arm[FOLLOWER]
        if not hasattr(a, "_command_lock"):                 # mock arm
            a.command_joint_pos(cmd)
            return
        # Same as the SDK's command_joint_state, but the command is built first and swapped in under the lock: the SDK
        # resets it to all zeros (kp = kd = 0) outside the lock, so its control thread can catch a zero-stiffness tick.
        from i2rt.robots.motor_chain_robot import JointCommands
        c = JointCommands.init_all_zero(len(a.motor_chain))
        c.pos = a._joint_to_motor_pos(a.remapper.to_robot_joint_pos_space(a._clip_robot_joint_pos_command(cmd)))
        c.vel = a._joint_to_motor_vel(a.remapper.to_robot_joint_vel_space(np.r_[vff, 0.0]))
        c.kp, c.kd = a._kp, a._kd
        with a._command_lock:
            a._commands = c

    def _hold(self, why, fq=None, fg=None):
        if fq is None:
            _, fq, fg = self._obs(FOLLOWER)
        if self.mode in ("engaging", "following") or self.cmd is not None:
            self.cmd = np.r_[fq, self.cmd[6] if self.cmd is not None else fg]
            self.vff = np.zeros(6)
            self.arm[FOLLOWER].command_joint_pos(self.cmd)
        self.mode, self.msg = "held", why
        if self.ep["state"] == "recording":
            self._end_episode("discarded")
            self.msg += " The episode being recorded was discarded."

    def _snapshot(self, lo, lq, fo, fq):
        arr = lambda o, k: [float(v) for v in np.ravel(o[k])] if k in o else None
        a = self.arm[FOLLOWER]
        mt = a.get_motor_torques() if hasattr(a, "get_motor_torques") else None
        follower = {k: arr(fo, k) for k in ("joint_pos", "joint_vel", "joint_eff", "gripper_pos", "gripper_vel", "gripper_eff",
                                            "temp_mos", "temp_rotor")}
        follower.update(grav_torque=[float(v) for v in self.gmodel.gravity(fq) * self.grav_factor],
                        motor_torque_cmd=None if mt is None else [float(v) for v in np.ravel(mt)])
        leader = {k: arr(lo, k) for k in ("joint_pos", "joint_vel", "joint_eff", "gripper_pos", "temp_mos", "temp_rotor")}
        ep = dict(self.ep)
        ep["elapsed_s"] = None if ep["t_start"] is None else round(time.time() - ep["t_start"], 1)
        return dict(t_ns=time.time_ns(), mode=self.mode, msg=self.msg, blocked=self.blocked, episode=ep,
                    action=None if self.cmd is None else [float(v) for v in self.cmd],
                    action_vel=None if self.cmd is None else [float(v) for v in self.vff * VFF_GAIN],
                    follower=follower, leader=leader)

    # ---- commands from the page (each takes the lock)
    def engage(self):
        if self.mode in ("engaging", "following"):
            return "already engaged"
        _, fq, fg = self._obs(FOLLOWER)
        self.cmd = np.r_[fq, fg]                            # start exactly where it is: no jump
        self.grip_target = fg
        self.vff = np.zeros(6)
        self.arm[FOLLOWER].command_joint_pos(self.cmd)
        self.mode, self.msg = "engaging", "engaging: follower moving slowly to the leader's pose"
        return self.msg

    def disengage(self):
        if self.ep["state"] == "recording":
            self._end_episode("saved")
        self.arm[FOLLOWER].enter_gravity_comp_idle()
        self.cmd, self.mode, self.msg = None, "idle", "follower floating: press Engage"
        return self.msg

    def grip(self, g):
        self.grip_target = float(np.clip(g, 0.0, 1.0))
        return f"gripper target {self.grip_target:.2f}"

    def start_episode(self, task):
        if self.mode != "following":
            return "engage first: episodes are recorded only while the follower follows"
        if self.ep["state"] == "recording":
            return "already recording"
        if task is not None:
            self.ep["task"] = task.strip()
        if not self.ep["task"]:
            return "type a task description first (it is the language label of the episode)"
        self.ep.update(state="recording", t_start=time.time())
        self.msg = f"RECORDING episode {self.ep['id']}"
        return self.msg

    def _end_episode(self, how):
        self.ep["saved" if how == "saved" else "discarded"] += 1
        done = self.ep["id"]
        self.ep.update(state=how, t_start=None, last=dict(id=done, how=how))   # the recorder closes / deletes episode `done`
        self.ep["id"] += 1
        return f"episode {done} {how}"

    def stop_episode(self, how):
        if self.ep["state"] != "recording":
            return "not recording"
        self.msg = self._end_episode(how)
        return self.msg

    def park(self, release):
        if self.ep["state"] == "recording":
            self._end_episode("saved")
        q0 = {s: np.r_[self._obs(s)[1], self._obs(s)[2]] for s in (LEADER, FOLLOWER)}
        if self.mode in ("engaging", "following", "held") and self.cmd is not None:
            q0[FOLLOWER] = self.cmd.copy()
        seconds = max(5.0, max(float(np.abs(q0[s][:6]).max()) for s in q0) / 0.3)
        n = int(seconds * 100)
        for i in range(1, n + 1):
            u = i / n; u = u * u * (3 - 2 * u)
            for s in q0:
                c = q0[s].copy(); c[:6] = (1 - u) * q0[s][:6]
                self.arm[s].command_joint_pos(c)
            time.sleep(seconds / n)
        time.sleep(0.5)
        for s in q0:
            self.arm[s].enter_gravity_comp_idle()
        self.cmd, self.mode = None, "idle"
        if release:
            for s in q0:
                if hasattr(self.arm[s], "close"):
                    self.arm[s].close()
            self.released, self.mode = True, "released"
            self.msg = "parked and released: torque is off. Restart the program to continue."
        else:
            self.msg = "parked at rest; both arms floating"
        return self.msg

    def do(self, fn, *a):
        with self.lock:
            if self.released:
                return "arms are released; restart the program"
            try:
                return fn(*a)
            except Exception as e:
                self.msg = f"error: {type(e).__name__}: {e}"
                return self.msg


PAGE = r"""<!doctype html><meta charset=utf-8><title>YAM teleop</title><meta name=viewport content="width=device-width,initial-scale=1">
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
input[type=text]{width:100%;box-sizing:border-box;background:#0e1013;color:#dde;border:1px solid #3a414c;border-radius:5px;padding:7px;font:inherit}
pre{margin:0;font:11.5px/1.4 ui-monospace,monospace;white-space:pre-wrap}#msg{font-weight:600}.bad{color:#f88}.good{color:#8d8}
#mode{font-size:15px;font-weight:700}#mode.recording{color:#ff6b6b}#mode.following{color:#8d8}#mode.held{color:#fb4}
kbd{background:#2a2f36;border-radius:3px;padding:0 4px;font:11px ui-monospace,monospace}
</style>
<div class=wrap><div class=cams id=cams></div>
<div class=panel>
 <div class=box><h2>Follower (right arm)</h2><div id=mode>…</div>
  <div class=row style="margin-top:6px"><button class=go onclick="post('/engage')">Engage <kbd>E</kbd></button><button onclick="post('/disengage')">Float <kbd>Q</kbd></button></div>
  <div id=blocked class=bad></div></div>
 <div class=box><h2>Episode</h2>
  <input type=text id=task placeholder="task, e.g. pick up the cube and place it on the tape">
  <div class=row style="margin-top:6px"><button id=recbtn class=rec onclick="toggleRec()">Start <kbd>space</kbd></button><button onclick="post('/episode/discard')">Discard <kbd>X</kbd></button></div>
  <pre id=ep></pre></div>
 <div class=box><h2>Gripper</h2><div class=row><button onclick="post('/grip',{g:1})">Open <kbd>V</kbd></button><button onclick="post('/grip',{g:0.62})">Cube width <kbd>B</kbd></button><button onclick="post('/grip',{g:0.45})">Squeeze <kbd>C</kbd></button><button onclick="post('/grip',{g:0})">Close <kbd>Z</kbd></button></div></div>
 <div class=box><h2>State</h2><div id=msg>…</div><pre id=st></pre></div>
 <div class=box><div class=row><button onclick="post('/park')">Park at rest</button><button class=warn onclick="if(confirm('Park both arms and cut torque?'))post('/release')">Park and release</button></div></div>
</div></div>
<script>
const CAM='http://'+(location.hostname==='localhost'?'127.0.0.1':'localhost')+':8090';
fetch('/cams').then(r=>r.json()).then(n=>{cams.innerHTML=n.map((c,i)=>`<figure><img data-cam=${i}><figcaption>${c}</figcaption></figure>`).join('');
 const sc=document.createElement('script');sc.src=CAM+'/streams.js';sc.onload=()=>camStreams(CAM,cams.querySelectorAll('img[data-cam]'));document.head.appendChild(sc)});
let S=null;
async function post(p,b){try{const r=await (await fetch(p,{method:'POST',body:JSON.stringify(b||{})})).json();msg.textContent=r.msg;msg.className=/^(STOPPED|error|engage first|type|not )/.test(r.msg)?'bad':'good'}catch(e){msg.textContent='request failed: '+e}}
function toggleRec(){if(S&&S.episode.state==='recording')post('/episode/stop');else post('/episode/start',{task:task.value})}
const f=v=>v==null?'-':v.map(x=>x.toFixed(2)).join(' ');
function show(s){S=s;const rec=s.episode.state==='recording';mode.textContent=rec?'● RECORDING episode '+s.episode.id+'  '+s.episode.elapsed_s+' s':s.mode.toUpperCase();
 mode.className=rec?'recording':s.mode;recbtn.textContent=rec?'Stop and save (space)':'Start (space)';blocked.textContent=s.blocked?'blocked: '+s.blocked:'';
 if(document.activeElement!==task&&!task.value&&s.episode.task)task.value=s.episode.task;
 ep.textContent=`next id ${s.episode.id}   saved ${s.episode.saved}   discarded ${s.episode.discarded}\ntask: ${s.episode.task||'-'}`;
 const F=s.follower,L=s.leader;st.textContent=`${s.msg}\n\nleader q   ${f(L.joint_pos)}\nfollower q ${f(F.joint_pos)}  grip ${f(F.gripper_pos)}\naction     ${f(s.action)}\nfollower eff  ${f(F.joint_eff)}\nfollower grav ${f(F.grav_torque)}\ntemps (rotor) ${F.temp_rotor?F.temp_rotor.join(' '):'-'}`}
setInterval(async()=>{try{show(await (await fetch('/telemetry')).json())}catch(e){}},200);
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'){if(e.key==='Enter')e.target.blur();return}const k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();toggleRec()}else if(k==='x')post('/episode/discard');else if(k==='e')post('/engage');else if(k==='q')post('/disengage');
 else if(k==='v')post('/grip',{g:1});else if(k==='b')post('/grip',{g:0.62});else if(k==='c')post('/grip',{g:0.45});else if(k==='z')post('/grip',{g:0})});
</script>"""
CAM_NAMES = ["D415 (tripod, top)", "C922", "Microdia left", "Microdia right"]


def main():
    global VFF_GAIN
    ap = argparse.ArgumentParser(description="right-arm teleop with the left arm as leader")
    ap.add_argument("--mock", action="store_true", help="no hardware: a self-moving leader and a simulated follower")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--max-vel", type=float, default=1.5, help="follower joint speed limit while following (rad/s)")
    ap.add_argument("--grav-factor", type=float, nargs=6, default=list(GRAV_FACTOR))
    ap.add_argument("--friction", action="store_true", help="Coulomb friction feed-forward on the leader (lighter by hand)")
    ap.add_argument("--vff", type=float, default=VFF_GAIN,
                    help="velocity feed-forward gain 0..1 (1 = the follower keeps up with fast wrist turns; 0 = position only)")
    ap.add_argument("--task", default="", help="default task description for episodes")
    ap.add_argument("--scene", default=os.path.join(os.environ.get("RUBIKS", ""), "poc", "generated", "scene.xml"))
    a = ap.parse_args()
    VFF_GAIN = float(np.clip(a.vff, 0.0, 1.0))
    if not os.path.isfile(a.scene):
        raise SystemExit(f"rig scene not found: {a.scene} (source env.sh, or pass --scene); it is needed for collision checks")
    tele = Teleop(a.mock, a.scene, a.grav_factor, a.max_vel, task=a.task, friction=a.friction)

    def bail(*_):
        print("signal: parking both arms before exit ...", flush=True)
        if not tele.released:
            tele.do(tele.park, True)
        raise SystemExit(0)
    signal.signal(signal.SIGINT, bail); signal.signal(signal.SIGTERM, bail)

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ct="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            try:
                if self.path == "/telemetry":
                    self._send(tele.snap or {})
                elif self.path == "/cams":
                    self._send(CAM_NAMES)
                elif self.path == "/config":
                    self._send(dict(leader=LEADER, follower=FOLLOWER, kp=KP.tolist(), kd=KD.tolist(), max_vel=tele.max_vel, vff_gain=VFF_GAIN,
                                    grav_factor=tele.grav_factor.tolist(), gravity_model=tele.gmodel.path,
                                    link_specs=tele.gmodel.specs(), j_lo=J_LO.tolist(), j_hi=J_HI.tolist(), mock=tele.mock))
                elif self.path.startswith("/telemetry/stream"):
                    hz = 100.0
                    if "hz=" in self.path:
                        hz = min(100.0, max(1.0, float(self.path.split("hz=")[1].split("&")[0])))
                    self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
                    last, period, nxt = None, 1.0 / hz, time.time()
                    while True:
                        s = tele.snap
                        if s is not None and s is not last:
                            self.wfile.write((json.dumps(s) + "\n").encode()); self.wfile.flush(); last = s
                        nxt += period
                        time.sleep(max(0.0, nxt - time.time()))
                        if time.time() - nxt > 0.5:
                            nxt = time.time()
                else:
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            routes = {"/engage": (tele.engage,), "/disengage": (tele.disengage,), "/grip": (tele.grip, b.get("g", 1.0)),
                      "/episode/start": (tele.start_episode, b.get("task")), "/episode/stop": (tele.stop_episode, "saved"),
                      "/episode/discard": (tele.stop_episode, "discarded"), "/park": (tele.park, False), "/release": (tele.park, True)}
            r = routes.get(self.path)
            msg = tele.do(*r) if r else "unknown command"
            try:
                self._send(dict(msg=msg))
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    print(f"YAM teleop on http://127.0.0.1:{a.port}  ({'MOCK' if a.mock else 'REAL arms'}; leader={LEADER}, follower={FOLLOWER}; "
          f"URDF gravity x {tele.grav_factor.tolist()}; velocity feed-forward x {VFF_GAIN})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
