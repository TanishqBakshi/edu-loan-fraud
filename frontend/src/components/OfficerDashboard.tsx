// OfficerDashboard.tsx
import React, { useEffect, useState } from "react";

const API_BASE = (window as any).__API_BASE__ || "http://localhost:8000";

type ScoredRecord = {
  id: number;
  applicant_id: string;
  session_id?: string;
  fp?: string;
  ip?: string;
  created_at?: number;
  score?: number;
  risk_label?: string;
  model_used?: string;
  reasons?: any[];
  raw_input?: any;
};

export default function OfficerDashboard() {
  const [items, setItems] = useState<ScoredRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/officer/list`);
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `status ${res.status}`);
      }
      const j = await res.json();
      setItems((j.results || []) as ScoredRecord[]);
    } catch (err: any) {
      setError(String(err?.message || err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // simple polling to keep realtime-ish
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{ padding: 18 }}>
      <h1>Loan Officer Dashboard</h1>
      <div style={{ marginBottom: 12 }}>
        <button onClick={load} style={{ padding: "8px 12px", borderRadius: 8, background: "#0b63ff", color: "#fff", border: "none" }}>
          Refresh
        </button>
      </div>

      {loading && <div>Loading...</div>}
      {error && <div style={{ color: "tomato" }}>Error: {error}</div>}

      {!loading && items.length === 0 && <div style={{ color: "#9aa4b3" }}>No records yet.</div>}

      <div style={{ display: "grid", gap: 12, marginTop: 12 }}>
        {items.map((it) => (
          <div key={it.id} style={{ padding: 12, borderRadius: 8, background: "rgba(255,255,255,0.02)" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <div><strong>{it.applicant_id}</strong> — {new Date((it.created_at || Date.now()) * 1000).toLocaleString()}</div>
              <div>{it.score !== undefined ? `${Number(it.score).toFixed(3)}` : "—"}</div>
            </div>
            <div style={{ marginTop: 8 }}>
              <strong>Risk:</strong> {it.risk_label ?? "—"}
            </div>
            <div style={{ marginTop: 8 }}>
              <strong>Model:</strong> {it.model_used ?? "—"}
            </div>
            <div style={{ marginTop: 8 }}>
              <details>
                <summary>Reasons & raw</summary>
                <pre style={{ background: "#080808", padding: 8, borderRadius: 6, overflowX: "auto" }}>{JSON.stringify({ reasons: it.reasons, raw_input: it.raw_input }, null, 2)}</pre>
              </details>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}