import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const BACKEND_BASE = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

/**
 * OfficerLogin.tsx
 * Simple username/password login UI that calls /auth/token and stores token in localStorage.
 * Demo-only: plaintext password.
 */

export default function OfficerLogin(): JSX.Element {
  const [username, setUsername] = useState<string>("officer");
  const [password, setPassword] = useState<string>("demo1234");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // backend expects application/x-www-form-urlencoded (OAuth2 form)
      const body = new URLSearchParams();
      body.append("username", username);
      body.append("password", password);

      const r = await fetch(`${BACKEND_BASE}/auth/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
      });

      if (!r.ok) {
        const text = await r.text();
        throw new Error(`Login failed: ${r.status} ${r.statusText} — ${text}`);
      }
      const data = await r.json();
      const token = data.access_token;
      if (!token) throw new Error("No token returned");
      // store token demo-style
      localStorage.setItem("officer_token", token);
      // navigate to officer dashboard
      navigate("/officer");
    } catch (e: any) {
      console.error("Login error", e);
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 30, maxWidth: 540 }}>
      <h2>Officer login</h2>
      <p style={{ color: "#b6c2cc" }}>Please sign in to access the officer dashboard.</p>
      <form onSubmit={handleSubmit} style={{ display: "grid", gap: 12 }}>
        <label style={{ fontSize: 13 }}>Username</label>
        <input value={username} onChange={(e) => setUsername(e.target.value)} style={{ padding: 8, borderRadius: 6 }} />
        <label style={{ fontSize: 13 }}>Password</label>
        <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" style={{ padding: 8, borderRadius: 6 }} />
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button disabled={loading} style={{ padding: "8px 12px", background: "#5b6cff", color: "white", border: "none", borderRadius: 6 }}>
            {loading ? "Signing in…" : "Sign in"}
          </button>
          <button
            type="button"
            onClick={() => {
              setUsername("officer");
              setPassword("demo1234");
            }}
            style={{ background: "transparent", border: "1px solid #333", color: "#ddd", padding: "8px 12px", borderRadius: 6 }}
          >
            Fill demo creds
          </button>
        </div>

        {error ? <div style={{ color: "tomato" }}>{error}</div> : null}
        <div style={{ fontSize: 12, color: "#9aa" }}>Demo account: <b>officer / demo1234</b></div>
      </form>
    </div>
  );
}