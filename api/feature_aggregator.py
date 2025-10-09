# api/feature_aggregator.py
import time, json, os, threading
from collections import deque, defaultdict
from statistics import mean, pstdev

# In-memory store: session_id -> deque of events (keystroke/mouse)
# This is a demo feature-store. In prod use Redis/ClickHouse/feature-store.
_MAX_EVENTS = 2000
_sessions = defaultdict(lambda: deque(maxlen=_MAX_EVENTS))
_lock = threading.Lock()

def push_event(session_id: str, event: dict):
    """Push a raw event (keystroke/mouse) into the session buffer."""
    ts = event.get("timestamp", int(time.time()*1000))
    with _lock:
        _sessions[session_id].append(dict(event, _received_ts=ts))

def clear_session(session_id: str):
    with _lock:
        _sessions.pop(session_id, None)

def _compute_from_events(events):
    """Compute simple aggregated features from raw events list."""
    keystrokes = [e for e in events if e.get("event_type") == "keystroke"]
    mouse = [e for e in events if e.get("event_type") in ("mouse","scroll")]

    # Keystroke timing features (payload must have downDuration / time)
    hold_times = []
    inter_key = []
    backspace_count = 0
    last_time = None
    for e in keystrokes:
        p = e.get("payload", {})
        if "downDuration" in p:
            try:
                hold_times.append(float(p.get("downDuration", 0)))
            except Exception:
                pass
        t = p.get("time")
        if t is not None:
            try:
                t = float(t)
                if last_time is not None:
                    inter_key.append(abs(t - last_time))
                last_time = t
            except Exception:
                pass
        if p.get("key","").lower() in ("backspace", "\b"):
            backspace_count += 1

    avg_hold = float(mean(hold_times)) if hold_times else 0.0
    std_hold = float(pstdev(hold_times)) if len(hold_times) > 1 else 0.0
    inter_mean = float(mean(inter_key)) if inter_key else 0.0
    backspace_freq = (backspace_count / len(keystrokes)) if keystrokes else 0.0

    # Mouse features (very simple)
    jitter = 0.0
    if mouse:
        # use payload "dx"/"dy" if present; fallback to jitter random
        diffs = []
        last_pos = None
        for e in mouse:
            p = e.get("payload",{})
            x = p.get("x"); y = p.get("y")
            if x is not None and y is not None:
                if last_pos is not None:
                    dx = float(x) - float(last_pos[0])
                    dy = float(y) - float(last_pos[1])
                    diffs.append((dx*dx + dy*dy)**0.5)
                last_pos = (x,y)
        jitter = float(mean(diffs)) if diffs else 0.0

    return {
        "event_count": len(events),
        "avg_hold": round(avg_hold, 3),
        "std_hold": round(std_hold, 3),
        "interkey_mean": round(inter_mean, 3),
        "backspace_freq": round(backspace_freq, 4),
        "mouse_jitter": round(jitter, 3),
        "last_ts": events[-1].get("_received_ts") if events else None
    }

def get_latest(session_id: str):
    """Return latest aggregates for a session (in-memory)."""
    with _lock:
        evs = list(_sessions.get(session_id, []))
    return _compute_from_events(evs)

# Demo persistence: append aggregate to data/aggregates.jsonl
DATA_AGG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "aggregates.jsonl"))
os.makedirs(os.path.dirname(DATA_AGG_FILE), exist_ok=True)

def flush_all():
    """Persist all current aggregates to disk (append-only) and clear memory."""
    to_write = []
    with _lock:
        for sid, deq in list(_sessions.items()):
            agg = _compute_from_events(list(deq))
            record = {"session_id": sid, "ts": int(time.time()), "agg": agg}
            to_write.append(record)
        _sessions.clear()

    if to_write:
        with open(DATA_AGG_FILE, "a", encoding="utf-8") as f:
            for r in to_write:
                f.write(json.dumps(r) + "\n")
    return len(to_write)
