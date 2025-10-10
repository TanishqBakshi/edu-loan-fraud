# api/app.py
import os
import json
import time
import sqlite3
import traceback
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# -------------------
# Config
# -------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "ml", "models")
DB_DIR = os.path.join(ROOT_DIR, "data")
DB_PATH = os.path.join(DB_DIR, "scored_apps.db")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

os.makedirs(DB_DIR, exist_ok=True)

# -------------------
# App setup
# -------------------
app = FastAPI(title="Edu Loan Fraud - API (with persistence)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, narrow this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# DB helpers
# -------------------
def init_db():
    """Create sqlite DB and scored_applications table if missing."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scored_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id TEXT,
            session_id TEXT,
            fp TEXT,
            ip TEXT,
            created_at REAL,
            score REAL,
            risk_label TEXT,
            model_used TEXT,
            reasons_json TEXT,
            raw_input_json TEXT,
            raw_features_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def save_scored_application(
    applicant_id: str,
    session_id: Optional[str],
    fp: Optional[str],
    ip: Optional[str],
    score: float,
    risk_label: str,
    model_used: str,
    reasons: List[Dict[str, Any]],
    raw_input: Dict[str, Any],
    raw_features: Dict[str, Any],
):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO scored_applications
        (applicant_id, session_id, fp, ip, created_at, score, risk_label, model_used, reasons_json, raw_input_json, raw_features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            applicant_id,
            session_id,
            fp,
            ip,
            time.time(),
            float(score),
            risk_label,
            model_used,
            json.dumps(reasons),
            json.dumps(raw_input),
            json.dumps(raw_features),
        ),
    )
    conn.commit()
    conn.close()

def get_recent_scored(limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, applicant_id, session_id, fp, ip, created_at, score, risk_label, model_used, reasons_json, raw_input_json, raw_features_json
        FROM scored_applications
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append(
            {
                "id": r[0],
                "applicant_id": r[1],
                "session_id": r[2],
                "fp": r[3],
                "ip": r[4],
                "created_at": r[5],
                "score": r[6],
                "risk_label": r[7],
                "model_used": r[8],
                "reasons": json.loads(r[9]) if r[9] else None,
                "raw_input": json.loads(r[10]) if r[10] else None,
                "raw_features": json.loads(r[11]) if r[11] else None,
            }
        )
    return results

# -------------------
# Models & metadata loading
# -------------------
models = {}
metadata = {}

def load_models_and_metadata():
    global models, metadata
    models = {}
    metadata = {}
    # load metadata if exists
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    # load standard models
    for name in ["rf", "lr", "isoforest", "autoencoder"]:
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        if os.path.exists(path):
            try:
                obj = joblib.load(path)
                models[name] = obj
                print(f"Loaded {name} model from {path}")
            except Exception as e:
                print(f"Failed to load {name} from {path}: {e}")
        else:
            # missing — that's fine; we fallback at scoring
            pass

# -------------------
# Small feature extraction helper
# -------------------
def simple_extract_features_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a minimal set of features derived from behavior events.
    This is intentionally simple — in your production pipeline replace with the
    same feature extraction used in training and listed in metadata['feature_columns'].
    """
    features = {
        "events_count": 0,
        "keydown_count": 0,
        "keyup_count": 0,
        "mouse_move": 0,
        "clicks": 0,
        "backspace_count": 0,
        "avg_hold_time": None,
        "avg_interkey": None,
    }
    hold_times = []
    inter_times = []
    last_ts = None
    for ev in events or []:
        features["events_count"] += 1
        t = ev.get("type", "")
        ts = ev.get("ts", None)
        if t == "keydown":
            features["keydown_count"] += 1
            key = ev.get("key", "")
            if key in ("Backspace", "Delete"):
                features["backspace_count"] += 1
            if last_ts is not None and ts is not None:
                inter_times.append(max(0, ts - last_ts))
        elif t == "keyup":
            features["keyup_count"] += 1
        elif t == "mousemove":
            features["mouse_move"] += 1
        elif t in ("click", "mousedown"):
            features["clicks"] += 1
        if "hold" in ev:
            hold_times.append(ev["hold"])
        if ts is not None:
            last_ts = ts
    features["avg_hold_time"] = (sum(hold_times) / len(hold_times)) if hold_times else 0.0
    features["avg_interkey"] = (sum(inter_times) / len(inter_times)) if inter_times else 0.0
    return features

# -------------------
# Utilities: attempt to build an ordered ndarray for model predict
# -------------------
def build_feature_vector_from_input(raw_input: Dict[str, Any], extracted_feats: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Use metadata['feature_columns'] if present; otherwise create a small vector from `extracted_feats`.
    Returns (vector_or_None, features_dict_used)
    """
    # If metadata lists feature_columns, try to assemble in that order
    feat_dict = {}
    try:
        if metadata and "feature_columns" in metadata:
            cols = metadata.get("feature_columns", [])
            for c in cols:
                # main sources: extracted behavior features OR top-level fields in input
                if c in extracted_feats:
                    feat_dict[c] = extracted_feats[c]
                elif c in raw_input:
                    # numeric cast where possible
                    try:
                        feat_dict[c] = float(raw_input[c])
                    except Exception:
                        feat_dict[c] = 0.0
                else:
                    feat_dict[c] = 0.0
            arr = np.array([feat_dict[c] for c in cols], dtype=float).reshape(1, -1)
            return arr, feat_dict
    except Exception as e:
        print("Failed to build vector from metadata feature_columns:", e)
    # Fallback: use a small hand-crafted ordering
    fallback_order = [
        "events_count", "keydown_count", "keyup_count",
        "clicks", "mouse_move", "backspace_count",
        "avg_hold_time", "avg_interkey"
    ]
    for c in fallback_order:
        feat_dict[c] = float(extracted_feats.get(c, 0.0) or 0.0)
    arr = np.array([feat_dict[c] for c in fallback_order], dtype=float).reshape(1, -1)
    return arr, feat_dict

# -------------------
# Risk label mapping helper
# -------------------
def score_to_label(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"

# -------------------
# Startup: init DB & load models
# -------------------
init_db()
load_models_and_metadata()

# -------------------
# Pydantic model for request
# -------------------
class ScoreRequest(BaseModel):
    applicant_id: Optional[str]
    name: Optional[str]
    dob: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    documents: Optional[List[Dict[str, Any]]] = []
    behavior_events: Optional[List[Dict[str, Any]]] = []
    behavior_summary_id: Optional[str] = None
    device_fingerprint: Optional[str] = None
    session_id: Optional[str] = None

# -------------------
# API endpoints
# -------------------

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys()), "metadata": bool(metadata)}

# -------------------
# Minimal behavior capture endpoints (start + websocket)
# -------------------

# in-memory store for captured event lists per session id
# Note: for the demo we use memory; for production persist to DB or buffer/store
behavior_store: Dict[str, List[Dict[str, Any]]] = {}

@app.post("/behavior/start")
async def behavior_start(request: Request):
    """
    Start a behavior capture session.
    Frontend sends a POST to this to acquire a session_id to open websocket.
    Returns: { session_id: "...", ws_url: "ws://..." }.
    """
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    # generate a session id
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    # init store
    behavior_store[session_id] = []
    # build ws url (frontend uses host same as API)
    # if front-end is served from other host, adjust logic there
    ws_scheme = "ws"
    host = request.client.host if request.client else "127.0.0.1"
    # prefer returning host:port of backend from request url base
    # but provide simple default
    # get base host:port from request.url
    base = str(request.url).split(request.url.path)[0]
    ws_url = base.replace("http", "ws") + "/behavior_ws?session_id=" + session_id
    return {"session_id": session_id, "ws_url": ws_url}


@app.websocket("/behavior_ws")
async def behavior_ws(websocket: WebSocket):
    """
    WebSocket endpoint to receive behaviour events JSON as they stream from the frontend.
    Frontend should connect to: ws://<api-host>:<port>/behavior_ws?session_id=<id>
    Messages must be JSON objects representing events; we append them to behavior_store[session_id].
    """
    # fastapi WebSocket accepts connections but we will check session_id param
    await websocket.accept()
    try:
        params = websocket.query_params
        session_id = params.get("session_id")
        if not session_id:
            # require session id
            await websocket.send_json({"error": "missing session_id query param"})
            await websocket.close(code=1008)
            return

        # if not present, initialize as well
        if session_id not in behavior_store:
            behavior_store[session_id] = []

        # Accept and receive messages loop
        while True:
            data = await websocket.receive_text()
            # try parse as JSON
            try:
                evt = json.loads(data)
            except Exception:
                # ignore malformed messages but send a warning
                try:
                    await websocket.send_json({"warning": "malformed json"})
                except Exception:
                    pass
                continue

            # store event
            behavior_store[session_id].append(evt)

            # optional: echo ack (keeps client aware)
            try:
                await websocket.send_json({"ack": True, "received": len(behavior_store[session_id])})
            except Exception:
                pass

    except WebSocketDisconnect:
        # client disconnected
        return
    except Exception as e:
        try:
            await websocket.close()
        except Exception:
            pass
        print("behavior_ws error:", e)
        return
    
@app.post("/score")
async def score_endpoint(req: Request, payload: ScoreRequest):
    """
    Score an application and persist the scored result to the sqlite DB.
    Returns JSON with score, risk_label, reasons, model info.
    """
    try:
        body_client = await req.json()
    except Exception:
        body_client = payload.dict()

    # extract client ip
    client_host = None
    try:
        client_host = req.client.host
    except Exception:
        client_host = "unknown"

    # Basic inputs
    applicant_id = payload.applicant_id or (body_client.get("applicant_id") if isinstance(body_client, dict) else None) or "anon"
    session_id = payload.session_id or body_client.get("session_id") if isinstance(body_client, dict) else None
    fp = payload.device_fingerprint or body_client.get("device_fingerprint") if isinstance(body_client, dict) else None
    events = payload.behavior_events or body_client.get("behavior_events", [])
    documents = payload.documents or body_client.get("documents", [])

    # 1) feature extraction
    extracted = simple_extract_features_from_events(events)

    # 2) attempt to build feature vector that matches training
    vector, used_features = build_feature_vector_from_input(body_client if isinstance(body_client, dict) else payload.dict(), extracted)

    # 3) scoring - try to use RF if present, else LR, else anomaly
    scores = {}
    reasons = []

    # device fingerprint basic check (demo-level)
    # If metadata contains a demo fingerprint pattern we could check; otherwise produce neutral reason
    if fp:
        reasons.append({"type": "DEVICE_FP", "msg": "Fingerprint received", "fp": fp})
    else:
        reasons.append({"type": "DEVICE_FP", "msg": "No fingerprint provided"})

    # behavior capture summary
    reasons.append({"type": "BEHAVIOR", "msg": f"events_captured:{extracted.get('events_count',0)}"})

    model_used = "none"
    final_score = 0.0

    # Helper to safe predict_proba
    def safe_predict_proba(m, vec):
        try:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(vec)
                # if binary classification, take positive class prob
                if p.shape[1] >= 2:
                    return float(p[0, 1])
                else:
                    return float(p[0, 0])
            else:
                # fallback to decision_function or predict
                if hasattr(m, "decision_function"):
                    df = m.decision_function(vec)
                    return float((df - df.min()) / (df.max() - df.min() + 1e-9)) if np.ndim(df)>0 else float(df)
                elif hasattr(m, "predict"):
                    pr = m.predict(vec)
                    return float(pr[0])
        except Exception as e:
            print("safe_predict_proba error:", e)
        return None

    # Try random forest
    if "rf" in models and vector is not None:
        try:
            rf = models["rf"]
            p_rf = safe_predict_proba(rf, vector)
            if p_rf is not None:
                scores["rf"] = p_rf
                model_used = "rf"
                reasons.append({"type":"MODEL","model":"rf","msg":"RandomForest produced a probability"})
        except Exception as e:
            print("RF scoring failed:", e)

    # Try logistic regression if rf not present
    if "lr" in models and ("rf" not in scores or scores.get("rf") is None):
        try:
            lr = models["lr"]
            p_lr = safe_predict_proba(lr, vector)
            if p_lr is not None:
                scores["lr"] = p_lr
                model_used = "lr"
                reasons.append({"type":"MODEL","model":"lr","msg":"Logistic Regression produced a probability"})
        except Exception as e:
            print("LR scoring failed:", e)

    # Try anomaly detectors for anomaly signal (higher -> more anomalous)
    # isoforest: predict_proba not always present; use score_samples (negative -> outlier)
    if "isoforest" in models:
        try:
            iso = models["isoforest"]
            if hasattr(iso, "score_samples"):
                iso_score = float(iso.score_samples(vector)[0]) if vector is not None else None
                # map iso_score to [0,1] anomaly severity (simple)
                iso_severity = None
                if iso_score is not None:
                    # iso_score is higher for inliers; convert to anomaly probability
                    iso_severity = float(1.0 - (1.0 / (1.0 + abs(iso_score))))
                    scores["isoforest"] = iso_severity
                    reasons.append({"type":"ANOMALY","model":"isoforest","score":iso_severity,"msg":"IsolationForest produced anomaly score"})
        except Exception as e:
            print("Iso scoring failed:", e)

    if "autoencoder" in models:
        try:
            ae = models["autoencoder"]
            if hasattr(ae, "predict"):
                # If autoencoder implemented as regressor - compute reconstruction error heuristic
                rec = ae.predict(vector) if vector is not None else None
                if rec is not None:
                    mse = float(np.mean((vector - rec) ** 2))
                    # scale to 0-1 roughly
                    ae_sev = min(1.0, mse / (mse + 1.0))
                    scores["autoencoder"] = ae_sev
                    reasons.append({"type":"ANOMALY","model":"autoencoder","score":ae_sev,"msg":"Autoencoder reconstruction error"})
        except Exception as e:
            print("Autoencoder scoring failed:", e)

    # If at least one discriminative score exists, build ensemble (weighted average)
    discriminative_keys = [k for k in ("rf", "lr") if k in scores]
    if discriminative_keys:
        vals = [scores[k] for k in discriminative_keys]
        final_score = float(sum(vals) / len(vals))
        model_used = discriminative_keys[0]  # primary model for display
    else:
        # fallback: use anomaly severity if present
        if "isoforest" in scores:
            final_score = float(scores["isoforest"])
            model_used = "isoforest"
        elif "autoencoder" in scores:
            final_score = float(scores["autoencoder"])
            model_used = "autoencoder"
        else:
            # last resort heuristic: simple function of events_count and backspace frequency
            ev = extracted.get("events_count", 0)
            bs = extracted.get("backspace_count", 0)
            heuristic = 1.0 - min(1.0, ev / 1000.0)  # fewer events -> higher suspicion
            heuristic += min(1.0, bs / (ev + 1.0)) * 0.2
            final_score = max(0.0, min(1.0, heuristic / 1.2))
            model_used = "heuristic"
            reasons.append({"type":"MODEL","model":"heuristic","msg":"Fallback heuristic used"})

    # Clip and label mapping
    final_score = float(max(0.0, min(1.0, final_score)))
    label = score_to_label(final_score)

    # Prepare response
    resp = {
        "score": final_score,
        "risk_label": label,
        "model_used": model_used,
        "scores_detail": scores,
        "reasons": reasons,
        "session_id": session_id,
    }

    # Persist the scored application
    try:
        save_scored_application(
            applicant_id=applicant_id,
            session_id=session_id,
            fp=fp,
            ip=client_host,
            score=final_score,
            risk_label=label,
            model_used=model_used,
            reasons=reasons,
            raw_input=body_client if isinstance(body_client, dict) else payload.dict(),
            raw_features=used_features or extracted,
        )
    except Exception as e:
        print("Failed to save scored application:", e, traceback.format_exc())

    return resp

@app.get("/officer/list")
def officer_list(limit: int = 50):
    """
    Return recent scored applications for a simple officer dashboard.
    """
    try:
        rows = get_recent_scored(limit)
        return {"count": len(rows), "results": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------
# Reload models endpoint (manual trigger)
# -------------------
@app.post("/models/reload")
def reload_models():
    try:
        load_models_and_metadata()
        return {"status": "ok", "models": list(models.keys()), "metadata_present": bool(metadata)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------
# Run with: uvicorn api.app:app --reload
# -------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)