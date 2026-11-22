"""Change the task label of recorded episodes (the 'episode' metadata the converter reads), without touching the data.

  source env.sh && $REC_PY -m yam_teleop.relabel "screw_from_rest" data/episodes/<name>/episode_00001.mcap [...]

Each file is copied record by record into a new file with the new task, checked (same message count per topic, same
attachments), and only then swapped in for the original. The per-sample /telemetry JSON keeps the text that was typed at
recording time (a raw log); the dataset label comes from the metadata.
"""
import json
import os
import sys

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer


def counts(path):
    with open(path, "rb") as f:
        s = make_reader(f).get_summary()
        return {c.topic: s.statistics.channel_message_counts.get(i, 0) for i, c in s.channels.items()}, s.statistics.attachment_count


def relabel(path, task):
    tmp = path + ".relabel.part"
    with open(path, "rb") as f, open(tmp, "wb") as out:
        r = make_reader(f)
        meta = {m.name: dict(m.metadata) for m in r.iter_metadata()}
        old = meta.get("episode", {}).get("task")
        meta.setdefault("episode", {})["task"] = task
        if old is not None and old != task:
            meta["episode"]["task_as_recorded"] = old
        w = Writer(out, compression=CompressionType.NONE, chunk_size=4 * 1024 * 1024)
        w.start(profile="", library="yam_teleop.relabel")
        sid, cid = {0: 0}, {}
        for sch, ch, msg in r.iter_messages(log_time_order=False):
            if sch is not None and sch.id not in sid:
                sid[sch.id] = w.register_schema(sch.name, sch.encoding, sch.data)
            if ch.id not in cid:
                cid[ch.id] = w.register_channel(ch.topic, ch.message_encoding, sid[sch.id if sch else 0], ch.metadata)
            w.add_message(cid[ch.id], msg.log_time, msg.data, msg.publish_time, msg.sequence)
        for a in r.iter_attachments():
            w.add_attachment(a.create_time, a.log_time, a.name, a.media_type, a.data)
        for name, m in meta.items():
            w.add_metadata(name, m)
        w.finish()
    a, b = counts(path), counts(tmp)
    if a != b:
        os.remove(tmp)
        raise SystemExit(f"{path}: copy differs ({a} vs {b}); original left untouched")
    os.replace(tmp, path)
    return old


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    task = sys.argv[1]
    for p in sys.argv[2:]:
        old = relabel(p, task)
        print(json.dumps(dict(file=p, task=task, was=old)))
