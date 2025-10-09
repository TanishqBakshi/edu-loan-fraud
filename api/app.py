import os
import json
import time
import io
import joblib
import numpy as np
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from feature_aggregator import push_event, get_latest, flush_all

# =====================================================
#                 Edu Loan Fraud API
# =====================================================

app = FastAPI(title="Edu Loan Fraud API")
MODEL_VERSION = "v0.1.0"

# Allow CORS for frontend/dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Pydantic Models ====================

class BehaviorEvent(BaseModel):
    event_type: str
    user_id: str
    timestamp: int
    payload: dict
    client_meta: dict

class ApplicationSubmission(BaseModel):
    applicant_id: str
    name: str
    dob: str
    email: str
    phone: str
    documents: List[dict]
    behavior_summary_id: str | None = None
    device_fingerprint: str | None = None

# ================= ML Model Load ======================

ML_MODEL = None
try:
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "models", "rf.joblib"))
    if os.path.exists(model_path):
        ML_MODEL = joblib.load(model_path)
        print(f"✅ Loaded ML model from {model_path}")
    else:
        print("⚠️ No model found at", model_path)
except Exception as e:
    print("⚠️ Failed to load model:", e)
    ML_MODEL = None

# ================= Behavior Event WS ==================

@app.websocket("/behavior")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            await push_event(data)
    except WebSocketDisconnect:
        await flush_all()
    except Exception as e:
        print("WebSocket error:", e)
        await flush_all()

# =====================================================
#                   Scoring Endpoint
# =====================================================

@app.post("/score")
async def score_application(app_data: ApplicationSubmission):
    score = 0
    reasons = []

    # Example rule-based scores (will be replaced later with model)
    for doc in app_data.documents:
        if doc.get("ocr_confidence", 100) < 70:
            score += 10
            reasons.append({"code": "LOW_OCR", "explanation": f"OCR confidence {doc.get('ocr_confidence')}%"})

    if app_data.device_fingerprint and "susp" in app_data.device_fingerprint:
        score += 25
        reasons.append({"code": "DEVICE_MISMATCH", "explanation": "Fingerprint suspicious"})

    score = min(score, 100)
    label = "HIGH" if score >= 60 else ("MEDIUM" if score >= 30 else "LOW")

    response = {
        "applicant_id": app_data.applicant_id,
        "score": round(score, 2),
        "risk_label": label,
        "reasons": reasons,
        "model_version": MODEL_VERSION,
        "explainability": {"shap_values": {}}
    }

    # Append decision log safely
    import traceback
    try:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        os.makedirs(data_dir, exist_ok=True)
        log_path = os.path.join(data_dir, "decision_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": int(time.time()), "decision": response}) + "\n")
    except Exception as e:
        print("⚠️ Failed to write decision log:", e)
        traceback.print_exc()

    return JSONResponse(response)

# =====================================================
#         Explainability + Decision Endpoints
# =====================================================

FEATURE_NAMES = [
    "avg_hold_time", "std_hold", "backspace_freq",
    "interkey_mean", "mouse_jitter", "ocr_confidence", "device_fp_sim"
]

def compute_shap_for_vector(model, X):
    """
    Compute SHAP values for a single vector X (shape [1, n]).
    """
    try:
        import shap
    except Exception as e:
        print("⚠️ SHAP import failed:", e)
        return {}

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[-1]
        arr = shap_vals[0].tolist()
        return {fname: float(v) for fname, v in zip(FEATURE_NAMES, arr)}
    except Exception as e:
        print("⚠️ SHAP computation failed:", e)
        return {}

class ExplainRequest(BaseModel):
    features: List[float]

@app.post("/explain")
async def explain(req: ExplainRequest):
    if ML_MODEL is None:
        return {"shap_values": {}, "model_version": MODEL_VERSION, "note": "no_model_loaded"}

    if len(req.features) != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail=f"expected {len(FEATURE_NAMES)} features: {FEATURE_NAMES}")

    try:
        shap_map = compute_shap_for_vector(ML_MODEL, [req.features])
        return {"shap_values": shap_map, "model_version": MODEL_VERSION}
    except Exception as e:
        print("Explain error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ========== Decision Log APIs ==========

def read_decision_log(limit=200):
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "decision_log.jsonl"))
    if not os.path.exists(data_path):
        return []
    out = []
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except:
                        continue
    except Exception as e:
        print("Error reading log:", e)
        return []
    return out[-limit:]

@app.get("/decisions")
async def get_decisions(limit: int = 100):
    rows = read_decision_log(limit)
    simplified = []
    for r in reversed(rows):
        dec = r.get("decision", {})
        simplified.append({
            "ts": r.get("ts"),
            "applicant_id": dec.get("applicant_id"),
            "score": dec.get("score"),
            "risk_label": dec.get("risk_label"),
            "reasons": dec.get("reasons", []),
            "model_version": dec.get("model_version")
        })
    return {"count": len(simplified), "rows": simplified}

@app.get("/decision/{applicant_id}")
async def get_decision(applicant_id: str):
    rows = read_decision_log(limit=1000)
    for r in reversed(rows):
        dec = r.get("decision", {})
        if dec.get("applicant_id") == applicant_id:
            return {"found": True, "ts": r.get("ts"), "decision": dec}
    return {"found": False}

# =====================================================
#                Root Health Endpoint
# =====================================================

@app.get("/")
async def root():
    return {"message": "Edu Loan Fraud API is running", "model_loaded": ML_MODEL is not None}
