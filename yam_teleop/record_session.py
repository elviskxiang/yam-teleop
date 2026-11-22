"""lerobot-record, with each episode saved as soon as it stops (instead of after the reset / wait period).

  python -m yam_teleop.record_session <the same arguments as lerobot-record>      (record.sh runs this)

LeRobot 0.5's record() does: record episode -> reset period -> save_episode(). With a long reset period used as "wait
until the operator starts the next episode", an episode would sit unsaved until then. This runs LeRobot's own record()
unchanged, with two small hooks:
  * when the reset period starts (record_loop called without a dataset), the episode just recorded is saved first,
    unless it was discarded (left arrow) or the session is stopping (then record() saves it itself);
  * record()'s own save_episode() after the reset is skipped once if that episode was already saved.
Logs "Saving episode N" / "Episode N saved" for the control page.
"""
import logging

import lerobot.scripts.lerobot_record as R

log = logging.getLogger("yam_teleop.record_session")
_state = {"dataset": None, "skip_next_save": False}
_orig_record_loop = R.record_loop


def _record_loop(*args, **kw):
    ds = kw.get("dataset")
    events = kw.get("events", args[1] if len(args) > 1 else None)
    if ds is not None:                                   # an episode is about to be recorded
        if _state["dataset"] is not ds:
            _state["dataset"] = ds
            orig_save = ds.save_episode

            def save_episode(*a, **k):
                if _state["skip_next_save"]:
                    _state["skip_next_save"] = False
                    return None
                return orig_save(*a, **k)
            ds.save_episode = save_episode
        _state["skip_next_save"] = False
        return _orig_record_loop(*args, **kw)
    d = _state["dataset"]                                # reset / wait period
    if d is not None and events is not None and not events.get("rerecord_episode") and not events.get("stop_recording") \
            and not _state["skip_next_save"]:
        n = d.num_episodes
        log.info(f"Saving episode {n}")
        d.save_episode()                                 # still the original (skip flag is False)
        _state["skip_next_save"] = True
        log.info(f"Episode {n} saved")
    return _orig_record_loop(*args, **kw)


R.record_loop = _record_loop

if __name__ == "__main__":
    R.main()
