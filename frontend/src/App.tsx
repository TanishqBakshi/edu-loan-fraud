import React, { useState, useEffect } from "react";
import BehaviorCapture from "./components/BehaviorCapture";
import LoanForm from "./components/LoanForm";
import ResultCard from "./components/ResultCard";
import "./App.css";

export default function App() {
  const [result, setResult] = useState<any | null>(null);

  useEffect(() => {
    // Listen for result updates from LoanForm
    function handleScoreEvent(e: any) {
      setResult(e.detail);
    }

    window.addEventListener("scoreResult", handleScoreEvent);

    // Load last score if available
    const last = localStorage.getItem("last_score");
    if (last) {
      try {
        setResult(JSON.parse(last));
      } catch {}
    }

    return () => {
      window.removeEventListener("scoreResult", handleScoreEvent);
    };
  }, []);

  return (
    <div className="App dark">
      <header>
        <h1>Student Loan Application — Demo (Dark)</h1>
      </header>

      <main style={{ display: "flex", justifyContent: "center", padding: "40px" }}>
        <div style={{ flex: 1, maxWidth: 600 }}>
          <BehaviorCapture />
          <LoanForm onResult={setResult} />
        </div>

        <aside style={{ width: 420, marginLeft: 24 }}>
          <ResultCard result={result} />
        </aside>
      </main>

      <footer style={{ textAlign: "center", padding: "16px", opacity: 0.6 }}>
        Local demo — connect backend at{" "}
        <code>http://localhost:8000</code>
      </footer>
    </div>
  );
}
