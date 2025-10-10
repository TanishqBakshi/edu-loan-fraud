// src/App.tsx
import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import LoanForm from "./components/LoanForm";
import ResultCard from "./components/ResultCard";
import OfficerDashboard from "./components/OfficerDashboard";

export default function App(): JSX.Element {
  const [result, setResult] = useState<any | null>(null);

  return (
    <Router>
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 18,
          }}
        >
          <div style={{ fontSize: 18, fontWeight: 700 }}>
            Edu Loan — Application
          </div>
          <nav>
            <Link
              to="/"
              style={{
                color: "#bfc8d5",
                textDecoration: "none",
                marginRight: 16,
              }}
            >
              Student
            </Link>
            <Link
              to="/officer"
              style={{ color: "#bfc8d5", textDecoration: "none" }}
            >
              Officer
            </Link>
          </nav>
        </header>

        <main style={{ display: "flex", gap: 18 }}>
          <div style={{ flex: 1 }}>
            <Routes>
              {/* Student page */}
              <Route
                path="/"
                element={<LoanForm onResult={(r) => setResult(r)} />}
              />
              {/* Officer dashboard */}
              <Route path="/officer" element={<OfficerDashboard />} />
            </Routes>
          </div>

          {/* Result card sidebar */}
          <aside style={{ width: 420 }}>
            <ResultCard result={result} />
          </aside>
        </main>

        <footer
          style={{
            marginTop: 36,
            color: "#9aa4b3",
            textAlign: "center",
          }}
        >
          Local demo — backend at{" "}
          <code>http://localhost:8000</code>
        </footer>
      </div>
    </Router>
  );
}