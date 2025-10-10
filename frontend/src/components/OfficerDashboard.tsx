import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

/**
 * OfficerDashboard.tsx
 *
 * Fetches recent scored applications from the backend and shows:
 *  - Line chart for score over time
 *  - Score histogram (bar)
 *  - Risk-label distribution (pie)
 *  - Recent table
 *
 * This is purposely verbose and defensive about types so it fits into your existing repo.
 */

/* ---------------------------
   Config
   --------------------------- */
const BACKEND_BASE = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const POLL_INTERVAL_MS = 5000;

/* ---------------------------
   Types
   --------------------------- */
type RawScoredRow = {
  id?: number;
  applicant_id?: string | null;
  session_id?: string | null;
  fp?: string | null;
  ip?: string | null;
  created_at?: number | string | null;
  score?: number | null;
  risk_label?: string | null;
  model_used?: string | null;
  reasons?: any; // backend sends array; keep flexible
  raw_input?: any;
  raw_features?: any;
};

type NormalizedRow = {
  id: number | string;
  applicant_id: string;
  session_id: string | null;
  fp: string | null;
  ip: string | null;
  created_at: number;
  score: number | null;
  risk_label: string | null;
  model_used: string | null;
  reasons: any;
  raw_input: any;
  raw_features: any;
  status?: string;
};

/* ---------------------------
   Helper utilities
   --------------------------- */

function normalizeRow(d: RawScoredRow): NormalizedRow {
  const id = d.id ?? d.applicant_id ?? "unknown";
  // created_at may be timestamp (float) or string; coerce to number
  let created = Date.now();
  if (typeof d.created_at === "number") created = d.created_at;
  if (typeof d.created_at === "string") {
    const v = Number(d.created_at);
    if (!Number.isNaN(v)) created = v;
  }
  const scoreVal = typeof d.score === "number" ? d.score : null;
  return {
    id,
    applicant_id: (d.applicant_id as string) ?? (d.raw_input?.applicant_id as string) ?? "anon",
    session_id: (d.session_id as string) ?? null,
    fp: (d.fp as string) ?? null,
    ip: (d.ip as string) ?? null,
    created_at: created,
    score: scoreVal,
    risk_label: d.risk_label ?? (d.raw_input?.eval?.risk_label ?? null) ?? null,
    model_used: d.model_used ?? (d.raw_input?.eval?.model ?? null) ?? null,
    reasons: d.reasons ?? d.raw_input?.eval?.reasons ?? d.raw_input?.reasons ?? [],
    raw_input: d.raw_input ?? {},
    raw_features: d.raw_features ?? {},
    status: "scored",
  };
}

/* build timeseries array: {timeLabel, score} */
function buildTimeSeries(data: NormalizedRow[]) {
  // sort ascending by created_at
  const sorted = data.slice().sort((a, b) => a.created_at - b.created_at);
  return sorted.map((r) => ({
    timeLabel: new Date(r.created_at * 1000).toLocaleTimeString(), // created_at stored in seconds in backend
    ts: r.created_at,
    score: r.score ?? 0,
    id: r.id,
  }));
}

/* build histogram buckets for scores 0.0-1.0 with step */
function buildScoreBuckets(data: NormalizedRow[], step = 0.1) {
  const buckets: { name: string; count: number; rangeMin: number; rangeMax: number }[] = [];
  for (let start = 0; start < 1 + 1e-9; start += step) {
    const min = parseFloat(start.toFixed(2));
    const max = parseFloat(Math.min(1.0, start + step).toFixed(2));
    buckets.push({
      name: `${min.toFixed(2)}-${max.toFixed(2)}`,
      count: 0,
      rangeMin: min,
      rangeMax: max,
    });
  }
  data.forEach((d) => {
    const s = typeof d.score === "number" ? d.score : null;
    if (s === null) return;
    // find first bucket where rangeMin <= s < rangeMax (last bucket includes 1.0)
    for (let i = 0; i < buckets.length; i++) {
      const b = buckets[i];
      const isLast = i === buckets.length - 1;
      if ((s >= b.rangeMin && s < b.rangeMax) || (isLast && s <= b.rangeMax)) {
        b.count += 1;
        break;
      }
    }
  });
  return buckets;
}

function buildRiskDistribution(data: NormalizedRow[]) {
  const map: Record<string, number> = {};
  data.forEach((d) => {
    const label = (d.risk_label ?? "unknown").toString();
    map[label] = (map[label] || 0) + 1;
  });
  return Object.entries(map).map(([name, value]) => ({ name, value }));
}

/* colors for pie chart */
const RISK_COLORS: Record<string, string> = {
  HIGH: "#ff8a65",
  MEDIUM: "#ffcc00",
  LOW: "#4fc3f7",
  unknown: "#9e9e9e",
};

/* ---------------------------
   Component
   --------------------------- */

export default function OfficerDashboard(): JSX.Element {
  const [items, setItems] = useState<NormalizedRow[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchList = async (limit = 200) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_BASE}/officer/list?limit=${limit}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Backend request failed: ${res.status} ${res.statusText} — ${text}`);
      }
      const payload = await res.json();
      const rows = (payload.results ?? []) as RawScoredRow[];
      const normalized = rows.map(normalizeRow);
      setItems(normalized);
    } catch (e: any) {
      console.error("Failed to fetch officer list:", e);
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  };

  // initial + poll
  useEffect(() => {
    fetchList();
    const id = setInterval(() => {
      fetchList();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  // memo charts data
  const timeseries = useMemo(() => buildTimeSeries(items), [items]);
  const buckets = useMemo(() => buildScoreBuckets(items, 0.1), [items]);
  const riskDist = useMemo(() => buildRiskDistribution(items), [items]);

  return (
    <div style={{ padding: 28 }}>
      <h1 style={{ marginBottom: 12, fontSize: 28 }}>Loan Officer Dashboard</h1>

      <div style={{ display: "flex", gap: 18, alignItems: "flex-start", marginBottom: 18 }}>
        <div style={{ flex: 1 }}>
          <button
            onClick={() => fetchList()}
            style={{
              background: "#1976d2",
              color: "#fff",
              border: "none",
              padding: "10px 14px",
              borderRadius: 8,
              cursor: "pointer",
              marginBottom: 12,
            }}
          >
            Refresh
          </button>

          {loading ? <div style={{ color: "#888" }}>Loading...</div> : null}
          {error ? (
            <div style={{ color: "tomato", marginTop: 8 }}>
              Error fetching list: {error}
            </div>
          ) : null}

          {items.length === 0 && !loading ? (
            <div style={{ color: "#999", marginTop: 18 }}>No records yet.</div>
          ) : null}
        </div>

        <div style={{ width: 420 }}>
          <div style={{ background: "#0f1620", padding: 12, borderRadius: 12 }}>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8, color: "#f5f7fa" }}>
              Fraud Score (recent)
            </div>
            <div style={{ height: 180 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeseries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timeLabel" minTickGap={10} />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="score" stroke="#8884d8" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 420px", gap: 18 }}>
        {/* Left: Histogram + Recent table */}
        <div>
          <div style={{ background: "#0f1620", padding: 12, borderRadius: 12, marginBottom: 18 }}>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>Score distribution</div>
            <div style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={buckets}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" interval={0} angle={-30} textAnchor="end" height={60} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ background: "#0f1620", padding: 12, borderRadius: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 6 }}>Recent scored applications</div>
              <div style={{ color: "#9aa4b2", fontSize: 12 }}>{items.length} records</div>
            </div>

            <div style={{ marginTop: 8 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", color: "#e6eef6" }}>
                <thead style={{ textAlign: "left", fontSize: 13, color: "#9aa4b2" }}>
                  <tr>
                    <th style={{ padding: "6px 8px" }}>ID</th>
                    <th style={{ padding: "6px 8px" }}>Applicant</th>
                    <th style={{ padding: "6px 8px" }}>Score</th>
                    <th style={{ padding: "6px 8px" }}>Risk</th>
                    <th style={{ padding: "6px 8px" }}>Model</th>
                  </tr>
                </thead>
                <tbody>
                  {items.slice(0, 10).map((r) => (
                    <tr key={String(r.id)} style={{ borderTop: "1px solid rgba(255,255,255,0.03)" }}>
                      <td style={{ padding: "8px" }}>{String(r.id)}</td>
                      <td style={{ padding: "8px" }}>{r.applicant_id}</td>
                      <td style={{ padding: "8px" }}>{r.score !== null ? r.score.toFixed(3) : "—"}</td>
                      <td style={{ padding: "8px" }}>{r.risk_label ?? "—"}</td>
                      <td style={{ padding: "8px" }}>{r.model_used ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right: Pie + Raw JSON view */}
        <div>
          <div style={{ background: "#0f1620", padding: 12, borderRadius: 12, marginBottom: 18 }}>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>Risk label distribution</div>
            <div style={{ width: "100%", height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskDist}
                    dataKey="value"
                    nameKey="name"
                    outerRadius={80}
                    label
                  >
                    {riskDist.map((entry, idx) => {
                      const color = RISK_COLORS[entry.name as string] ?? "#8884d8";
                      return <Cell key={`cell-${idx}`} fill={color} />;
                    })}
                  </Pie>
                  <Legend verticalAlign="bottom" />
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ background: "#0f1620", padding: 12, borderRadius: 12 }}>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>Last raw item</div>
            {items.length === 0 ? (
              <div style={{ color: "#9aa4b2" }}>No data</div>
            ) : (
              <pre style={{ whiteSpace: "pre-wrap", color: "#c8d7e6", fontSize: 12, maxHeight: 280, overflow: "auto" }}>
                {JSON.stringify(items[0], null, 2)}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}