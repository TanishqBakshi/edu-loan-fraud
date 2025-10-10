// LoanForm.tsx
import React, { useEffect, useState } from "react";
import BehaviorCapture from "./BehaviorCapture";

const API_BASE = (window as any).__API_BASE__ || "http://localhost:8000";

type LoanFormProps = {
  sessionId?: string | null;
  onResult?: (result: any) => void;
};

export default function LoanForm({ sessionId: initialSessionId, onResult }: LoanFormProps) {
  const [sessionId, setSessionId] = useState<string | null>(initialSessionId ?? null);
  const [fingerprint, setFingerprint] = useState<string | null>(null);

  const [name, setName] = useState("Tanishq Bakshi");
  const [email, setEmail] = useState("tanishqbakshi112@gmail.com");
  const [dob, setDob] = useState("2006-01-02");
  const [phone, setPhone] = useState("+917015259188");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<any>(null);

  // keep a small polling to fetch session/fingerprint if BehaviorCapture set them on window
  useEffect(() => {
    // poll once to pick up session created by BehaviorCapture if available
    const maybe = (window as any).__behavior_capture_session__;
    if (maybe) {
      try {
        const s = maybe();
        if (s && s.sessionId) {
          setSessionId(s.sessionId);
          setFingerprint(s.fingerprint || null);
        }
      } catch (e) {}
    }
    // also fetch again after a short delay to allow BehaviorCapture to start
    const t = setTimeout(() => {
      try {
        const s = (window as any).__behavior_capture_session__?.();
        if (s && s.sessionId) {
          setSessionId(s.sessionId);
          setFingerprint(s.fingerprint || null);
        }
      } catch (e) {}
    }, 1200);
    return () => clearTimeout(t);
  }, []);

  // gather local captured events from window helper created by BehaviorCapture
  function getCapturedEvents(): any[] {
    try {
      const v = (window as any).__get_behavior_events__?.();
      if (Array.isArray(v)) return v;
    } catch (e) {}
    return [];
  }

  async function handleSubmit(e?: React.FormEvent) {
    if (e) e.preventDefault();
    setSubmitting(true);
    setError(null);

    const behavior_events = getCapturedEvents();

    const payload = {
      applicant_id: `app_${Math.random().toString(36).slice(2, 8)}`,
      name,
      dob,
      email,
      phone,
      documents: [], // demo
      behavior_events,
      device_fingerprint: fingerprint,
      session_id: sessionId,
    };

    try {
      const res = await fetch(`${API_BASE}/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        // try to parse HTML or text
        const txt = await res.text();
        setError(txt);
        setSubmitting(false);
        return;
      }

      const j = await res.json();
      setLastResponse(j);
      if (onResult) onResult(j);
    } catch (err: any) {
      setError(String(err?.message ?? err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      {/* Behavior capture UI (inline so user sees session/fingerprint controls) */}
      <BehaviorCapture sessionId={sessionId} onSessionChange={(id) => setSessionId(id)} />

      <form onSubmit={handleSubmit} style={{ padding: 18, borderRadius: 8, background: "rgba(0,0,0,0.35)" }}>
        <div style={{ marginBottom: 12 }}>
          <label style={{ display: "block", fontWeight: 600 }}>Full name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} style={{ width: "100%", padding: 10, borderRadius: 6 }} />
        </div>
        <div style={{ marginBottom: 12 }}>
          <label style={{ display: "block", fontWeight: 600 }}>Email</label>
          <input value={email} onChange={(e) => setEmail(e.target.value)} style={{ width: "100%", padding: 10, borderRadius: 6 }} />
        </div>
        <div style={{ marginBottom: 12 }}>
          <label style={{ display: "block", fontWeight: 600 }}>DOB</label>
          <input value={dob} onChange={(e) => setDob(e.target.value)} style={{ width: "100%", padding: 10, borderRadius: 6 }} />
        </div>
        <div style={{ marginBottom: 12 }}>
          <label style={{ display: "block", fontWeight: 600 }}>Phone</label>
          <input value={phone} onChange={(e) => setPhone(e.target.value)} style={{ width: "100%", padding: 10, borderRadius: 6 }} />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button type="submit" disabled={submitting} style={{ padding: "10px 16px", borderRadius: 8, background: "#5b4bff", color: "white", border: "none" }}>
            {submitting ? "Submitting..." : "Submit Application"}
          </button>
          <div>
            <span style={{ color: "#9aa4b3" }}>Fingerprint:</span>{" "}
            <code style={{ background: "#0f1720", padding: "4px 8px", borderRadius: 6 }}>{fingerprint ?? "fp_missing"}</code>
          </div>
        </div>

        <div style={{ marginTop: 18, color: "#9aa4b3" }}>
          Ready to submit — backend at <code>http://localhost:8000</code>
        </div>

        <div style={{ marginTop: 12 }}>
          {error && (
            <div style={{ color: "tomato", whiteSpace: "pre-wrap" }}>
              Error: {String(error)}
            </div>
          )}

          {lastResponse && (
            <div style={{ marginTop: 12, color: "#cbd5e1" }}>
              <strong>Last response:</strong>
              <pre style={{ background: "#0b0b0b", padding: 12, borderRadius: 8, overflowX: "auto" }}>{JSON.stringify(lastResponse, null, 2)}</pre>
            </div>
          )}
        </div>

        <div style={{ marginTop: 8, color: "#9aa4b3" }}>
          <small>Session id: <code style={{ background: "#0f1720", padding: "4px 8px", borderRadius: 6 }}>{sessionId ?? "beh_missing"}</code></small>
        </div>
      </form>
    </div>
  );
}