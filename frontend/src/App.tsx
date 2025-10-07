// frontend/src/App.tsx
import React, { useState } from "react";
import BehaviorCapture from "./components/BehaviorCapture";
import "./App.css";

function App() {
  const [consent, setConsent] = useState(false);
  const sessionId = "temp_session_" + Math.random().toString(36).slice(2, 9);

  return (
    <div style={{ padding: 24, fontFamily: "Inter, Arial, sans-serif" }}>
      <h1>Student Loan Application — Demo</h1>

      {!consent ? (
        <div>
          <p>
            To speed up verification we capture typing & mouse behavior. This is a demo — accept to continue.
          </p>
          <button onClick={() => setConsent(true)} style={{ padding: "8px 14px", borderRadius: 6 }}>
            I consent
          </button>
        </div>
      ) : (
        <>
          <BehaviorCapture sessionId={sessionId} />
          <form
            onSubmit={(e) => {
              e.preventDefault();
              alert("Application submitted (demo). Use curl to POST to /score for scoring.");
            }}
            style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 12, maxWidth: 420 }}
          >
            <input name="name" placeholder="Full name" required />
            <input name="email" placeholder="Email" type="email" required />
            <button type="submit" style={{ padding: "8px 12px", borderRadius: 6 }}>
              Submit Application
            </button>
          </form>
        </>
      )}
    </div>
  );
}

export default App;
