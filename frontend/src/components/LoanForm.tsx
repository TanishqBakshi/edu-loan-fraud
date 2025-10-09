// frontend/src/components/LoanForm.tsx
import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import FingerprintJS from "@fingerprintjs/fingerprintjs";

type Props = {
  sessionId?: string;                       // from BehaviorCapture
  onResult: (res: any) => void;            // App will receive the scoring result
  backendUrl?: string;                     // optional override (default http://localhost:8000)
};

function makeApplicantId() {
  return `app_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
}

export default function LoanForm({ sessionId, onResult, backendUrl = "http://localhost:8000" }: Props) {
  // form fields
  const [name, setName] = useState<string>("Tanishq Bakshi");
  const [email, setEmail] = useState<string>("tanishqbakshi112@gmail.com");
  const [dob, setDob] = useState<string>("2006-01-02");
  const [phone, setPhone] = useState<string>("+917015259188");

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [fingerprint, setFingerprint] = useState<string | null>(null);
  const [ipAddress, setIpAddress] = useState<string | null>(null);
  const [country, setCountry] = useState<string>("Unknown");
  const behaviorSummaryId = sessionId || `beh_${Date.now().toString(36).slice(-6)}`;

  // generate fingerprint on mount
  useEffect(() => {
    let canceled = false;
    (async () => {
      try {
        // init agent
        const fp = await FingerprintJS.load();
        const result = await fp.get();
        if (!canceled) setFingerprint(String(result.visitorId));
      } catch (err) {
        // fallback short hash if fingerprint library fails
        const fallback = `fp_${Math.random().toString(36).slice(2, 12)}`;
        if (!canceled) setFingerprint(fallback);
      }
    })();
    return () => { canceled = true; };
  }, []);

  // try to detect public ip (best-effort). fallback to localhost
  useEffect(() => {
    let canceled = false;
    (async () => {
      try {
        const resp = await fetch("https://api.ipify.org?format=json");
        if (!resp.ok) throw new Error("ipify failed");
        const j = await resp.json();
        if (!canceled) setIpAddress(j.ip || "127.0.0.1");
      } catch (e) {
        if (!canceled) setIpAddress("127.0.0.1");
      }
    })();
    return () => { canceled = true; };
  }, []);

  // derive country from browser locale if possible
  useEffect(() => {
    try {
      const loc = Intl.DateTimeFormat().resolvedOptions().locale || navigator.language || "";
      // locale often like "en-US" -> extract US
      const parts = loc.split("-");
      if (parts.length > 1) setCountry(parts[1]);
      else setCountry("Unknown");
    } catch {
      setCountry("Unknown");
    }
  }, []);

  const submit = async (ev?: React.FormEvent) => {
    if (ev) ev.preventDefault();
    setError(null);
    setLoading(true);

    // basic validation
    if (!name || !email) {
      setError("Name and email are required.");
      setLoading(false);
      return;
    }

    const applicant_id = makeApplicantId();

    // build payload the backend expects (fields present in many demo examples)
    const payload: any = {
      applicant_id,
      name,
      dob,
      email,
      phone,
      ip_address: ipAddress || "127.0.0.1",
      location_country: country || "Unknown",
      documents: [
        // keep this minimal — backend sample used ocr results arrays — we provide empty placeholders
        { type: "id_card", ocr_text: "", ocr_confidence: 0 },
      ],
      behavior_summary_id: behaviorSummaryId,
      device_fingerprint: fingerprint || "unknown_fingerprint",
    };

    try {
      const resp = await axios.post(`${backendUrl.replace(/\/$/, "")}/score`, payload, {
        headers: { "Content-Type": "application/json" },
        timeout: 20000,
      });

      // success -> send to parent (App) to display
      if (resp && resp.data) {
        onResult(resp.data);
      } else {
        setError("No response data from server.");
      }
    } catch (err: any) {
      // helpful error parsing: if server returns 422/400 show details from body
      const serverMsg = err?.response?.data ?? null;
      if (err?.response?.status === 422 && serverMsg) {
        // if the backend returns validation details (422), show short message
        setError(`Validation error: ${JSON.stringify(serverMsg)}`);
      } else if (err?.message) {
        setError(`Network error or no response from server.`);
      } else {
        setError("Unknown error during submit.");
      }
      // still bubble up null result
      onResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ background: "var(--panel)", padding: 24, borderRadius: 12, boxShadow: "var(--card-shadow)" }}>
      <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <label style={{ fontSize: 14, opacity: 0.9 }}>Full name</label>
        <input value={name} onChange={(e) => setName(e.target.value)} style={{ padding: 12, borderRadius: 8 }} />

        <label style={{ fontSize: 14, opacity: 0.9 }}>Email</label>
        <input value={email} onChange={(e) => setEmail(e.target.value)} style={{ padding: 12, borderRadius: 8 }} />

        <label style={{ fontSize: 14, opacity: 0.9 }}>DOB</label>
        <input type="date" value={dob} onChange={(e) => setDob(e.target.value)} style={{ padding: 12, borderRadius: 8 }} />

        <label style={{ fontSize: 14, opacity: 0.9 }}>Phone</label>
        <input value={phone} onChange={(e) => setPhone(e.target.value)} style={{ padding: 12, borderRadius: 8 }} />

        <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 6 }}>
          <button disabled={loading} type="submit" style={{
            background: "linear-gradient(90deg,#3b82f6,#8b5cf6)",
            color: "white",
            border: "none",
            padding: "10px 18px",
            borderRadius: 8,
            cursor: "pointer",
            fontWeight: 700
          }}>
            {loading ? "Submitting…" : "Submit Application"}
          </button>

          <div style={{ fontSize: 13, color: "var(--muted)" }}>
            Fingerprint:&nbsp;
            <code style={{ background: "rgba(255,255,255,0.04)", padding: "3px 6px", borderRadius: 4 }}>
              {fingerprint ?? "loading..."}
            </code>
          </div>
        </div>

        <div style={{ marginTop: 8 }}>
          {error ? (
            <div style={{ color: "#ff7676", fontWeight: 600 }}>Error: {error}</div>
          ) : (
            <div style={{ color: "var(--muted)" }}>
              {loading ? "Submitting…" : "Ready to submit — backend at "}<code style={{ marginLeft: 6 }}>{backendUrl}</code>
            </div>
          )}
        </div>

        <div style={{ fontSize: 12, marginTop: 8, color: "var(--muted)" }}>
          <div>Session id: <code style={{ background: "rgba(255,255,255,0.02)", padding: "2px 6px", borderRadius: 4 }}>{behaviorSummaryId}</code></div>
          <div style={{ marginTop: 6 }}>IP: <code style={{ background: "rgba(255,255,255,0.02)", padding: "2px 6px", borderRadius: 4 }}>{ipAddress || "..."}</code> • Country: <strong>{country}</strong></div>
        </div>
      </form>
    </div>
  );
}
