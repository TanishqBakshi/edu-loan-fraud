import React from "react";

interface ResultCardProps {
  result: any;
}

const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  if (!result) return null;

  return (
    <div
      style={{
        background: "#111",
        color: "#fff",
        padding: "20px",
        borderRadius: "12px",
        marginTop: "16px",
      }}
    >
      <h3>Fraud Score</h3>
      <p>
        Score: <strong>{result.score.toFixed(3)}</strong>
      </p>
      <p>
        Risk label: <strong>{result.risk_label}</strong>
      </p>
      <p>Model used: {result.model_used}</p>
      <h4>Reasons:</h4>
      <ul>
        {result.reasons &&
          result.reasons.map((r: any, idx: number) => (
            <li key={idx}>
              [{r.type}] {r.msg}
            </li>
          ))}
      </ul>
    </div>
  );
};

export default ResultCard;