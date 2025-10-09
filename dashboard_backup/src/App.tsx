// dashboard/src/App.tsx
import React, { useEffect, useState } from "react";
import "./App.css";

type DecisionRow = {
  ts: number;
  applicant_id: string;
  score: number;
  risk_label: string;
  reasons: Array<{ code: string; explanation: string }>;
  model_version?: string;
};

function timeFmt(ts?: number) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleString();
}

function ShapBar({ shap }: { shap: Record<string, number> }) {
  const entries = Object.entries(shap || {});
  if (!entries.length) return <div className="no-shap">No SHAP</div>;
  const max = Math.max(...entries.map(([, v]) => Math.abs(v))) || 1;
  return (
    <div className="shap-bars">
      {entries.map(([k, v]) => (
        <div key={k} className="shap-row">
          <div className="shap-name">{k}</div>
          <div className="shap-bar-wrap">
            <div
              className="shap-bar"
              style={{
                width: `${(Math.abs(v) / max) * 100}%`,
                background: v >= 0 ? "#ef4444" : "#10b981",
                opacity: 0.9,
              }}
            />
            <div className="shap-value">{v.toFixed(3)}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [rows, setRows] = useState<DecisionRow[]>([]);
  const [selected, setSelected] = useState<DecisionRow | null>(null);
  const [shap, setShap] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);

  async function loadDecisions() {
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/decisions");
      const data = await res.json();
      setRows(data.rows || []);
    } catch (e) {
      console.error("Failed to load decisions", e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDecisions();
    const id = setInterval(loadDecisions, 5000);
    return () => clearInterval(id);
  }, []);

  async function openDetail(row: DecisionRow) {
    setSelected(row);
    try {
      const vec = [120, 40, 0.02, 150, 0.5, row.score || 50, 0.8];
      const r = await fetch("http://127.0.0.1:8000/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: vec }),
      });
      const dj = await r.json();
      setShap(dj.shap_values || {});
    } catch (e) {
      console.warn("Explain failed:", e);
      setShap({});
    }
  }

  return (
    <div style={{ padding: 20, fontFamily: "Inter, Arial, sans-serif" }}>
      <h1>Loan Fraud — Officer Dashboard (Demo)</h1>
      <div style={{ display: "flex", gap: 16 }}>
        <div style={{ width: 520 }}>
          <div style={{ marginBottom: 8 }}>
            <button onClick={loadDecisions}>Refresh</button>
            <span style={{ marginLeft: 12 }}>{loading ? "Loading..." : `${rows.length} rows`}</span>
          </div>
          <div className="list">
            {rows.map((r) => (
              <div key={`${r.applicant_id}-${r.ts}`} className="row">
                <div style={{ flex: 1 }}>
                  <div className="id">{r.applicant_id}</div>
                  <div className="meta">{timeFmt(r.ts)} • {r.model_version || "v?"}</div>
                </div>
                <div style={{ width: 140, textAlign: "right" }}>
                  <div className={`badge ${r.risk_label}`}>{r.risk_label}</div>
                  <div className="score">{r.score}</div>
                  <div><button onClick={() => openDetail(r)}>View</button></div>
                </div>
              </div>
            ))}
            {rows.length === 0 && <div>No decisions yet</div>}
          </div>
        </div>

        <div style={{ flex: 1 }}>
          <h3>Detail</h3>
          {selected ? (
            <div>
              <div><strong>Applicant</strong>: {selected.applicant_id}</div>
              <div><strong>Score</strong>: {selected.score} — <em>{selected.risk_label}</em></div>
              <div style={{ marginTop: 8 }}>
                <strong>Reasons</strong>
                <ul>
                  {selected.reasons.map((rs, i) => <li key={i}><b>{rs.code}</b>: {rs.explanation}</li>)}
                </ul>
              </div>
              <div style={{ marginTop: 8 }}>
                <strong>Explainability (SHAP)</strong>
                <ShapBar shap={shap} />
              </div>
            </div>
          ) : (
            <div>Select an application to see details.</div>
          )}
        </div>
      </div>
    </div>
  );
}
