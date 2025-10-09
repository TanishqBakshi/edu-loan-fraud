import React, { useEffect, useMemo, useState } from "react";

/**
 * BehaviorCapture component
 * Props:
 *  - sessionId?: string  (optional - if not provided we generate one)
 *
 * This component is a *lightweight demo-only* behavior capture stub:
 * it does not send real keystroke streams to a backend here — it generates
 * a demo session id and shows UI. The loan form will use this id when it
 * sends a payload to backend (if needed).
 */

type Props = {
  sessionId?: string;
};

function makeId() {
  // short stable-like id using timestamp + random
  return `sess_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export default function BehaviorCapture(props: Props) {
  // use passed id or generate one once
  const session = useMemo(() => props.sessionId || makeId(), [props.sessionId]);

  const [status] = useState<string>("active"); // simple placeholder state
  const [consented, setConsented] = useState<boolean>(true);

  useEffect(() => {
    // If you want to expose the sessionId globally (for debug)
    // window.__behavior_session = session; // avoid polluting in prod
    return () => {
      // cleanup if you later add event listeners
    };
  }, [session]);

  return (
    <div style={{
      background: "var(--panel)",
      padding: 16,
      borderRadius: 12,
      marginBottom: 16,
      boxShadow: "var(--card-shadow)"
    }}>
      <div style={{display: "flex", justifyContent: "space-between", alignItems: "center"}}>
        <div>
          <strong style={{fontSize: 15}}>Behavior capture {status === "active" ? "active" : "inactive"}</strong>
          <div style={{opacity: 0.7, marginTop:6, fontSize:13}}>This demo only sends anonymized behavior summary to the local backend. Accept to continue.</div>
        </div>

        <div style={{textAlign: "right"}}>
          <div style={{fontSize:12, color: "var(--muted)"}}>Session</div>
          <div style={{fontWeight:700}}>{session}</div>
        </div>
      </div>

      <div style={{marginTop:12, display:"flex", gap:8, alignItems:"center"}}>
        <label style={{display:"flex", alignItems:"center", gap:8, cursor:"pointer"}}>
          <input type="checkbox" checked={consented} onChange={() => setConsented(s => !s)} />
          <span style={{fontSize:13}}>I consent to demo behavior capture</span>
        </label>
      </div>
    </div>
  );
}
