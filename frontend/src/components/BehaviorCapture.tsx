// BehaviorCapture.tsx
import React, { useEffect, useRef, useState } from "react";

/**
 * BehaviorCapture component - collects keystrokes/mouse events,
 * streams each event to the backend websocket, and exposes session id/fingerprint.
 *
 * Props:
 *  - sessionId: string | null        (optional, display)
 *  - onSessionChange: (newId: string) => void   (called when server gives a new session id)
 *
 * Use:
 *   <BehaviorCapture sessionId={sessionId} onSessionChange={setSessionId} />
 *
 * This component will:
 *  - call POST /behavior/start to create a session (if sessionId not provided)
 *  - open ws://localhost:8000/behavior_ws
 *  - on events, send JSON { session_id, event, _client_ts } over the ws
 *  - keep an internal list of captured events, accessible via `.getEvents()` if needed
 */

type BehaviorCaptureProps = {
  sessionId?: string | null;
  onSessionChange?: (id: string) => void;
};

type EventRecord = {
  type: string;
  ts: number;
  x?: number;
  y?: number;
  key?: string;
  hold?: number;
  button?: number;
  extra?: Record<string, any>;
};

const API_BASE = (window as any).__API_BASE__ || "http://localhost:8000";

export default function BehaviorCapture({ sessionId: propSessionId, onSessionChange }: BehaviorCaptureProps) {
  const [sessionId, setSessionId] = useState<string | null>(propSessionId ?? null);
  const [fingerprint, setFingerprint] = useState<string | null>(null);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
  const wsRef = useRef<WebSocket | null>(null);

  // captured events stored locally; kept for submission
  const eventsRef = useRef<EventRecord[]>([]);
  // track key hold times
  const keyDownTimestamps = useRef<Record<string, number>>({});
  // reconnection backoff
  const reconnectAttempts = useRef<number>(0);
  const shouldReconnect = useRef<boolean>(true);

  // If parent gave a sessionId prop, keep in sync
  useEffect(() => {
    if (propSessionId) {
      setSessionId(propSessionId);
    }
  }, [propSessionId]);

  // Expose a getter on window for debugging if you want (optional)
  useEffect(() => {
    (window as any).__behavior_capture_get_events__ = () => eventsRef.current.slice();
    (window as any).__behavior_capture_session__ = () => ({ sessionId, fingerprint });
  }, [sessionId, fingerprint]);

  // create or refresh session by calling /behavior/start
  async function startSession() {
    try {
      const res = await fetch(`${API_BASE}/behavior/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ /* optionally send metadata */ }),
      });
      const j = await res.json();
      if (j && j.session_id) {
        setSessionId(j.session_id);
        setFingerprint(j.device_fingerprint || null);
        if (onSessionChange) onSessionChange(j.session_id);
        console.info("Behavior session started", j);
      } else {
        console.warn("behavior/start did not return session id", j);
      }
    } catch (e) {
      console.error("Failed to start behavior session", e);
    }
  }

  // Build websocket url
  function wsUrl() {
    // If backend served over https/wss adapt accordingly; for dev we use ws.
    const base = (API_BASE || "http://localhost:8000").replace(/^http/, "ws");
    return `${base.replace(/\/+$/, "")}/behavior_ws`;
  }

  // connect WS and set handlers
  function connectWs() {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;
    setWsStatus("connecting");
    shouldReconnect.current = true;
    const url = wsUrl();
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectAttempts.current = 0;
        setWsStatus("connected");
        console.info("Behavior WS connected");
      };

      ws.onmessage = (ev) => {
        // messages broadcast from server are JSON
        try {
          const data = JSON.parse(ev.data);
          // simple debug log for now
          // if devtools opened: you can inspect WS frames in Network
          // Also we could add a small UI list if needed
          // console.debug("WS recv:", data);
          // no other action required
        } catch (err) {
          // not JSON - ignore
        }
      };

      ws.onclose = () => {
        setWsStatus("disconnected");
        console.info("Behavior WS closed");
        wsRef.current = null;
        if (shouldReconnect.current) scheduleReconnect();
      };

      ws.onerror = (err) => {
        console.warn("Behavior WS error", err);
        try {
          ws.close();
        } catch (e) {}
      };
    } catch (e) {
      console.error("Failed to open WS", e);
      scheduleReconnect();
    }
  }

  function scheduleReconnect() {
    reconnectAttempts.current = Math.min(reconnectAttempts.current + 1, 10);
    const delay = Math.min(30000, 1000 * Math.pow(1.5, reconnectAttempts.current));
    setTimeout(() => {
      if (shouldReconnect.current) connectWs();
    }, delay);
  }

  // gracefully stop ws reconnect attempts / close socket
  function stopWs() {
    shouldReconnect.current = false;
    try {
      if (wsRef.current) {
        wsRef.current.close();
      }
    } catch (e) {}
    wsRef.current = null;
    setWsStatus("disconnected");
  }

  // send an event object over ws (augmented with session id + timestamp)
  function sendEventOverWs(evt: EventRecord | Record<string, any>) {
    const payload = {
      session_id: sessionId,
      device_fingerprint: fingerprint,
      event: evt,
      _client_ts: Date.now(),
    };
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(payload));
      }
    } catch (e) {
      // ignore send errors; WS will reconnect
    }
  }

  // helper to push into local event list and send via ws
  function pushEvent(evt: EventRecord) {
    eventsRef.current.push(evt);
    sendEventOverWs(evt);
  }

  // Attach DOM listeners once we have a session
  useEffect(() => {
    // start session if missing
    if (!sessionId) {
      startSession();
      // session will be set when /behavior/start returns
    } else {
      // If we have a sessionId, ensure websocket is connected
      connectWs();
    }

    // event handlers
    function onKeyDown(e: KeyboardEvent) {
      const ts = Date.now();
      const key = e.key;
      // record keydown timestamp
      keyDownTimestamps.current[key] = ts;
      const ev: EventRecord = { type: "keydown", ts, key, extra: { code: e.code, repeat: e.repeat } };
      pushEvent(ev);
    }

    function onKeyUp(e: KeyboardEvent) {
      const ts = Date.now();
      const key = e.key;
      let hold: number | undefined = undefined;
      if (keyDownTimestamps.current[key]) {
        hold = ts - keyDownTimestamps.current[key];
        delete keyDownTimestamps.current[key];
      }
      const ev: EventRecord = { type: "keyup", ts, key, hold };
      pushEvent(ev);
    }

    let lastMouseEmit = 0;
    function onMouseMove(e: MouseEvent) {
      const now = Date.now();
      // throttle mousemove emission a bit to avoid flooding (e.g. 30ms)
      if (now - lastMouseEmit < 30) return;
      lastMouseEmit = now;
      const ev: EventRecord = { type: "mousemove", ts: now, x: e.clientX, y: e.clientY };
      pushEvent(ev);
    }

    function onClick(e: MouseEvent) {
      const now = Date.now();
      const ev: EventRecord = { type: "click", ts: now, x: e.clientX, y: e.clientY, button: e.button };
      pushEvent(ev);
    }

    function onMouseDown(e: MouseEvent) {
      const now = Date.now();
      const ev: EventRecord = { type: "mousedown", ts: now, x: e.clientX, y: e.clientY, button: e.button };
      pushEvent(ev);
    }

    function onFocus(e: FocusEvent) {
      const now = Date.now();
      const ev: EventRecord = { type: "focus", ts: now, extra: { target: (e.target as HTMLElement)?.tagName } };
      pushEvent(ev);
    }

    function onBlur(e: FocusEvent) {
      const now = Date.now();
      const ev: EventRecord = { type: "blur", ts: now, extra: { target: (e.target as HTMLElement)?.tagName } };
      pushEvent(ev);
    }

    // attach listeners to window/document
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("click", onClick);
    window.addEventListener("mousedown", onMouseDown);
    window.addEventListener("focus", onFocus, true);
    window.addEventListener("blur", onBlur, true);

    // cleanup on unmount
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("click", onClick);
      window.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("focus", onFocus, true);
      window.removeEventListener("blur", onBlur, true);
      stopWs();
    };
    // We intentionally run this effect once; sessionId changes will call connectWs separately.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // If sessionId becomes available later (after /behavior/start), open WS
  useEffect(() => {
    if (sessionId) {
      connectWs();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // expose helper to parent or other code
  (window as any).__get_behavior_events__ = () => eventsRef.current.slice();

  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ padding: 18, borderRadius: 10, background: "rgba(255,255,255,0.02)", boxShadow: "0 6px 24px rgba(0,0,0,0.5)" }}>
        <h3 style={{ margin: 0 }}>Behavior capture is active</h3>
        <p style={{ marginTop: 8, color: "#bfc8d5" }}>
          This demo sends anonymized behavior to the backend. Session:{" "}
          <code style={{ background: "#111", padding: "2px 6px", borderRadius: 6 }}>{sessionId ?? "beh_missing"}</code>
        </p>
        <p style={{ marginTop: 8, color: "#9aa4b3" }}>
          Fingerprint: <code style={{ background: "#111", padding: "2px 6px", borderRadius: 6 }}>{fingerprint ?? "fp_missing"}</code>
        </p>
        <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <button
              onClick={() => {
                startSession();
                connectWs();
              }}
              style={{ padding: "8px 12px", borderRadius: 8, background: "#2b6cff", color: "white", border: "none", cursor: "pointer" }}
            >
              Start / Refresh session
            </button>
            <button
              onClick={() => {
                // clear local captured events
                eventsRef.current = [];
                keyDownTimestamps.current = {};
                alert("Cleared captured events (local buffer).");
              }}
              style={{ marginLeft: 10, padding: "8px 12px", borderRadius: 8, background: "#3a3a3a", color: "#fff", border: "none", cursor: "pointer" }}
            >
              Clear events
            </button>
          </div>

          <div style={{ textAlign: "right" }}>
            <span style={{ color: wsStatus === "connected" ? "#66f" : "#ff9f43", fontWeight: 600 }}>
              WS {wsStatus === "connected" ? "connected" : wsStatus === "connecting" ? "connecting" : "disconnected"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}