// frontend/src/components/BehaviorCapture.tsx
import React, { useEffect, useRef } from "react";

export default function BehaviorCapture({ sessionId }: { sessionId: string }) {
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Use 127.0.0.1 to avoid hostname issues
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/behavior");
    wsRef.current = ws;

    ws.onopen = () => console.log("WS open");
    ws.onmessage = (e) => console.log("WS msg:", e.data);
    ws.onclose = () => console.log("WS closed");
    ws.onerror = (ev) => console.warn("WS error", ev);

    return () => {
      try {
        ws.close();
      } catch {}
    };
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const ev = {
        event_type: "keystroke",
        user_id: sessionId,
        timestamp: Date.now(),
        payload: { key: e.key, event: e.type, time: Date.now(), downDuration: 0 },
        client_meta: {
          fingerprint_id: "fp_demo",
          ua: navigator.userAgent,
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        },
      };
      try {
        wsRef.current?.send(JSON.stringify(ev));
      } catch (err) {
        // ignore send errors (disconnected)
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sessionId]);

  return <div style={{ margin: "12px 0" }}>Behavior capture active — type in the form to stream events.</div>;
}
