import React, { useEffect, useState } from "react";

const backendBase = "http://localhost:8000";

const OfficerDashboard: React.FC = () => {
  const [records, setRecords] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchRecords = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${backendBase}/officer/list`);
      const data = await res.json();
      setRecords(data.results || []);
    } catch (e) {
      console.error("Error fetching officer data:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
    const interval = setInterval(fetchRecords, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ padding: "24px", color: "#fff" }}>
      <h2>Loan Officer Dashboard</h2>
      <button onClick={fetchRecords} disabled={loading}>
        Refresh
      </button>
      {loading && <p>Loading...</p>}
      {records.length === 0 ? (
        <p>No applications yet.</p>
      ) : (
        <div>
          {records.map((rec: any) => (
            <div
              key={rec.id}
              style={{
                marginTop: "16px",
                background: "#111",
                padding: "16px",
                borderRadius: "8px",
              }}
            >
              <h4>{rec.raw_input?.name}</h4>
              <p>Email: {rec.raw_input?.email}</p>
              <p>Score: {rec.score?.toFixed(3)}</p>
              <p>Risk: {rec.risk_label}</p>
              <p>Model: {rec.model_used}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default OfficerDashboard;