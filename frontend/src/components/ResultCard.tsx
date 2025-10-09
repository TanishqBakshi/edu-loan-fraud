import React from "react";

export default function ResultCard({ result }: { result: any | null }) {
  if (!result) {
    return (
      <div className="card">
        <h3>Application result</h3>
        <div style={{ color: "#9aa" }}>No result yet — submit the form to see scoring & reasons.</div>
      </div>
    );
  }

  if (result.error) {
    return (
      <div className="card">
        <h3>Application result</h3>
        <div className="error">{String(result.error)}</div>
      </div>
    );
  }

  return (
    <div className="card result">
      <h3>Fraud Score</h3>
      <div className="big-score">{result.score ?? "—"}</div>
      <div className="risk">Risk label: <strong>{result.risk_label ?? "—"}</strong></div>

      <div style={{ marginTop: 12 }}>
        <strong>Reasons</strong>
        <ul>
          {(result.reasons ?? []).map((r: any, i: number) => <li key={i}><b>{r.code}</b>: {r.explanation}</li>)}
        </ul>
      </div>

      <div style={{ marginTop: 12 }}>
        <strong>Model</strong>
        <div style={{ color: "#9aa" }}>{result.model_version ?? "v?"}</div>
      </div>
    </div>
  );
}
