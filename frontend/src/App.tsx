import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import BehaviorCapture from "./components/BehaviorCapture";
import LoanForm from "./components/LoanForm";
import ResultCard from "./components/ResultCard";
import OfficerDashboard from "./components/OfficerDashboard";

export default function App() {
  const [behaviorData, setBehaviorData] = useState<any[]>([]);
  const [result, setResult] = useState<any>(null);

  return (
    <Router>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "12px 24px",
          background: "#000",
          color: "#fff",
        }}
      >
        <h2>Edu Loan — Application</h2>
        <nav>
          <Link to="/" style={{ color: "#bfc", marginRight: "16px" }}>
            Student
          </Link>
          <Link to="/officer" style={{ color: "#bfc" }}>
            Officer
          </Link>
        </nav>
      </header>

      <main style={{ padding: "24px" }}>
        <Routes>
          <Route
            path="/"
            element={
              <div style={{ display: "flex", gap: "24px" }}>
                <div style={{ flex: 1 }}>
                  <BehaviorCapture onBehaviorData={setBehaviorData} />
                  <LoanForm behaviorData={behaviorData} onResult={setResult} />
                </div>
                <div style={{ width: "420px" }}>
                  <ResultCard result={result} />
                </div>
              </div>
            }
          />
          <Route path="/officer" element={<OfficerDashboard />} />
        </Routes>
      </main>
    </Router>
  );
}