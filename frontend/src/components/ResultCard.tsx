// ResultCard.tsx
import React from "react";

type ResultCardProps = {
  result: any | null;
};

export default function ResultCard({ result }: ResultCardProps) {
  if (!result) {
    return (
      <div style={{ padding: 18, borderRadius: 8, background: "rgba(255,255,255,0.02)" }}>
        <h3 style={{ marginTop: 0 }}>Fraud Score</h3>
        <div style={{ color: "#9aa4b3" }}>No result yet — submit the form to see scoring & reasons.</div>
      </div>
    );
  }

  const { score, risk_label, model_used, reasons, scores_detail } = result;

  return (
    <div style={{ padding: 18, borderRadius: 8, background: "rgba(255,255,255,0.02)" }}>
      <h3 style={{ marginTop: 0 }}>Fraud Score</h3>
      <div style={{ fontSize: 36, fontWeight: 700 }}>{typeof score === "number" ? score : "--"}</div>
      <div style={{ marginTop: 8 }}>
        <strong>Risk label:</strong> {risk_label ?? "—"}
      </div>
      <div style={{ marginTop: 12 }}>
        <strong>Reasons</strong>
        <ul>
          {(reasons || []).map((r: any, i: number) => (
            <li key={i}><strong>{r.type}</strong>: {r.msg || JSON.stringify(r)}</li>
          ))}
        </ul>
      </div>
      <div style={{ marginTop: 8 }}>
        <strong>Model</strong>
        <div>{model_used ?? "unknown"}</div>
      </div>

      <div style={{ marginTop: 12 }}>
        <strong>raw:</strong>
        <pre style={{ background: "#0b0b0b", padding: 12, borderRadius: 8, overflow: "auto" }}>{JSON.stringify(result, null, 2)}</pre>
      </div>
    </div>
  );
}