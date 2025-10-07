# api/app.py
import time
import json
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(title="Edu Loan Fraud API")

MODEL_VERSION = "v0.0.1"

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
    documents: list
    behavior_summary_id: str = None
    device_fingerprint: str = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Education Loan Fraud Detection API"}

# WebSocket endpoint (demo)
@app.websocket("/ws/behavior")
async def ws_behavior(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            # For demo: print a trimmed version to container logs
            print("RECV_BEHAVIOR:", data[:300])
            # In real system, push to Redis or aggregator
            await ws.send_text(json.dumps({"status": "received"}))
    except WebSocketDisconnect:
        print("WebSocket client disconnected")

# Simple scoring endpoint (demo logic)
@app.post("/score")
async def score_application(app_data: ApplicationSubmission):
    # Demo heuristics -> returns score & reasons
    score = 5.0
    reasons = []

    if app_data.device_fingerprint and str(app_data.device_fingerprint).endswith("susp"):
        score += 80
        reasons.append({"code": "DEVICE_MISMATCH", "explanation": "Fingerprint suspicious"})

    for doc in app_data.documents:
        if doc.get("ocr_confidence", 100) < 70:
            score += 10
            reasons.append({"code": "LOW_OCR", "explanation": f"OCR confidence {doc.get('ocr_confidence')}%"})

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

    # safe append to local repo data/decision_log.jsonl and log errors
    import os, traceback

    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(DATA_DIR, exist_ok=True)
    LOG_PATH = os.path.join(DATA_DIR, "decision_log.jsonl")

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": int(time.time()), "decision": response}) + "\n")
    except Exception as e:
        print("ERROR writing decision log:", e)
        traceback.print_exc()

    return JSONResponse(response)
