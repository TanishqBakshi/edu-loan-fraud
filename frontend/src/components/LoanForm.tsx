import React, { useState } from "react";

interface LoanFormProps {
  behaviorData: any[];
  onResult: (result: any) => void;
}

const LoanForm: React.FC<LoanFormProps> = ({ behaviorData, onResult }) => {
  const [form, setForm] = useState({
    name: "",
    email: "",
    dob: "",
    phone: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const backendBase = "http://localhost:8000";

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${backendBase}/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          applicant_id: "app_" + Math.random().toString(36).slice(2, 8),
          name: form.name,
          email: form.email,
          dob: form.dob,
          phone: form.phone,
          device_fingerprint: "fp_demo_" + Math.random().toString(36).slice(2, 6),
          behavior_events: behaviorData,
        }),
      });

      const data = await res.json();
      onResult(data);
    } catch (err: any) {
      console.error("LoanForm error:", err);
      setError(err.message || "Failed to submit");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        background: "#111",
        color: "#fff",
        padding: "20px",
        borderRadius: "12px",
      }}
    >
      <div>
        <label>Full name</label>
        <input
          name="name"
          value={form.name}
          onChange={handleChange}
          required
          style={{ width: "100%", marginBottom: "10px" }}
        />
      </div>
      <div>
        <label>Email</label>
        <input
          name="email"
          value={form.email}
          onChange={handleChange}
          required
          style={{ width: "100%", marginBottom: "10px" }}
        />
      </div>
      <div>
        <label>DOB</label>
        <input
          name="dob"
          value={form.dob}
          onChange={handleChange}
          required
          style={{ width: "100%", marginBottom: "10px" }}
        />
      </div>
      <div>
        <label>Phone</label>
        <input
          name="phone"
          value={form.phone}
          onChange={handleChange}
          required
          style={{ width: "100%", marginBottom: "10px" }}
        />
      </div>

      <button type="submit" disabled={loading}>
        {loading ? "Submitting..." : "Submit Application"}
      </button>

      {error && <div style={{ color: "red", marginTop: "10px" }}>{error}</div>}
    </form>
  );
};

export default LoanForm;