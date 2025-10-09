# app.py - full file (copy & replace your existing api/app.py)
import os
import json
import time
from typing import List, Optional, Dict, Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------
# Models / file locations
# -----------------------
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml", "models"))
MODEL_FILES = {
    "rf": "rf.joblib",
    "lr": "lr.joblib",
    "autoencoder": "autoencoder.joblib",
    "isoforest": "isoforest.joblib",
}
LOADED_MODELS: Dict[str, Any] = {}

# -----------------------
# FastAPI app + CORS
# -----------------------
app = FastAPI(title="Edu Loan Fraud API", description="Scoring + explainability endpoints for EDU loan demo")

# CORS configuration (inserted here intentionally)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # add production origins here if needed, e.g. "https://dashboard.example.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # or ["*"] during development (not recommended for prod)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------
# Pydantic request/response models
# -----------------------
class BehaviorSummary(BaseModel):
    id: str = Field(..., alias="behavior_summary_id")
    # We accept raw summary metadata here; frontend sends captured metrics
    avg_hold_time: Optional[float] = None
    std_hold: Optional[float] = None
    backspace_freq: Optional[float] = None
    interkey_mean: Optional[float] = None
    mouse_jitter: Optional[float] = None
    ocr_confidence: Optional[float] = None
    device_fp_sim: Optional[float] = None


class DocumentItem(BaseModel):
    type: str
    ocr_text: Optional[str]
    ocr_confidence: Optional[int]


class ApplicationSubmission(BaseModel):
    applicant_id: str
    name: str
    dob: str
    email: str
    phone: str
    documents: List[DocumentItem] = []
    behavior_summary_id: Optional[str] = None
    device_fingerprint: Optional[str] = None
    # allow extra fields for future expansion
    class Config:
        extra = "allow"


class ScoreResult(BaseModel):
    model_scores: Dict[str, float]
    final_score: float
    reasons: List[Dict[str, Any]] = []
    model_versions: Dict[str, str] = {}


# -----------------------
# Utilities - load models
# -----------------------
def load_models():
    global LOADED_MODELS
    LOADED_MODELS = {}
    # try to load each expected model file if present
    for key, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.isfile(path):
            try:
                obj = joblib.load(path)
                LOADED_MODELS[key] = obj
                print(f"Loaded {key} model from {path}")
            except Exception as e:
                LOADED_MODELS[key] = None
                print(f"Failed to load {key} from {path}: {repr(e)}")
        else:
            LOADED_MODELS[key] = None
            print(f"Model file not found for {key}: expected {path}")


# run at import
load_models()


# -----------------------
# Health / root endpoint
# -----------------------
@app.get("/")
def root():
    loaded = {k: (v is not None) for k, v in LOADED_MODELS.items()}
    return {"message": "Edu Loan Fraud API is running", "model_loaded": loaded}


# -----------------------
# small helper: basic input validation for required fields
# -----------------------
def validate_submission(payload: dict):
    required = ["applicant_id", "name", "dob", "email", "phone"]
    missing = [r for r in required if r not in payload or payload[r] in (None, "")]
    if missing:
        raise HTTPException(status_code=422, detail=[{"type": "missing", "loc": ["body"] , "msg": "Field required", "input": payload, "missing": missing}])

    # ensure device fingerprint present (frontend will include fp)
    # Note: per your request you want a "basic fingerprint check" only; we accept whatever is sent
    if "device_fingerprint" not in payload or not payload["device_fingerprint"]:
        # we do not block submission for no fingerprint, but we'll note it
        pass


# -----------------------
# Scoring helpers - simplistic
# -----------------------
def make_feature_vector(sub: ApplicationSubmission) -> Dict[str, float]:
    # Create a simple vector from available behavior features / docs
    # This is intentionally shallow — your ML pipeline normally expects exact feature ordering and normalization.
    # Here we create a best-effort dict that your saved scikit pipelines should be able to accept as a single-row DF.
    vec = {
        "avg_hold_time": None,
        "std_hold": None,
        "backspace_freq": None,
        "interkey_mean": None,
        "mouse_jitter": None,
        "ocr_confidence": None,
        "device_fp_sim": None,
    }
    # If the frontend included a behavior_summary object in the payload, use it (some clients POST it inline)
    if hasattr(sub, "__dict__") and sub.behavior_summary_id:
        # real features should be assembled from the aggregated event store.
        # For demo purposes we leave them None or 0 so pipelines won't crash.
        pass
    # fill defaults (0 instead of None to avoid model predict errors)
    for k in list(vec.keys()):
        if vec[k] is None:
            vec[k] = 0.0
    return vec


def safe_predict_proba(model, X_row):
    """
    If model has predict_proba, return probability for positive class.
    If model only has decision_function or predict, try to convert to 0-1.
    If unavailable, raise ValueError.
    """
    try:
        if hasattr(model, "predict_proba"):
            # sklearn classifiers return [[p0, p1]]
            probs = model.predict_proba([X_row])
            # if two columns, return second col; else try last column
            if probs.shape[1] >= 2:
                return float(probs[0, 1])
            else:
                return float(probs[0, -1])
        elif hasattr(model, "predict"):
            pred = model.predict([X_row])
            return float(pred[0])
        elif hasattr(model, "decision_function"):
            df = model.decision_function([X_row])
            # map decision function to a 0-1 by logistic transform for display
            import math
            s = 1.0 / (1.0 + math.exp(-float(df[0])))
            return s
        else:
            raise ValueError("Model has no prediction method")
    except Exception as e:
        raise


# -----------------------
# POST /score endpoint
# -----------------------
@app.post("/score")
async def score_application(submission: ApplicationSubmission, request: Request):
    # Basic validation
    try:
        validate_submission(submission.dict(by_alias=True))
    except HTTPException as he:
        # re-raise so FastAPI returns structured 422
        raise he

    # Build feature vector for the models (the real pipeline uses a DataFrame with exact columns)
    features = make_feature_vector(submission)

    # For demonstration we will attempt to pass a single-row dict to models directly.
    # Your actual trained pipelines may expect a DataFrame; adjust accordingly if necessary.
    X_row = list(features.values())

    model_scores = {}
    reasons = []

    # 1) Logistic Regression
    lr = LOADED_MODELS.get("lr")
    if lr:
        try:
            p = safe_predict_proba(lr, X_row)
            model_scores["lr"] = p
            # add simple reason extraction: coef weights (best-effort)
            try:
                if hasattr(lr, "coef_"):
                    # grab largest absolute coefficient feature as reason (demo)
                    import numpy as np
                    coefs = np.array(lr.coef_).ravel()
                    idx = int(np.argmax(np.abs(coefs)))
                    feat_name = list(features.keys())[idx] if idx < len(features) else "feature"
                    reasons.append({"source": "lr", "code": "LR_TOP_FEATURE", "explanation": f"Top LR feature: {feat_name}"})
            except Exception:
                pass
        except Exception as e:
            reasons.append({"source": "lr", "code": "LR_ERROR", "explanation": f"LR predict failed: {repr(e)}"})
    else:
        reasons.append({"source": "lr", "code": "LR_MISSING", "explanation": "Logistic Regression model not loaded"})

    # 2) Random Forest
    rf = LOADED_MODELS.get("rf")
    if rf:
        try:
            p = safe_predict_proba(rf, X_row)
            model_scores["rf"] = p
            # simple feature importance reason
            try:
                if hasattr(rf, "feature_importances_"):
                    import numpy as np
                    fi = np.array(rf.feature_importances_)
                    idx = int(np.argmax(fi))
                    feat_name = list(features.keys())[idx] if idx < len(features) else "feature"
                    reasons.append({"source": "rf", "code": "RF_TOP_FEATURE", "explanation": f"Top RF feature: {feat_name}"})
            except Exception:
                pass
        except Exception as e:
            reasons.append({"source": "rf", "code": "RF_ERROR", "explanation": f"RF predict failed: {repr(e)}"})
    else:
        reasons.append({"source": "rf", "code": "RF_MISSING", "explanation": "Random Forest model not loaded"})

    # 3) Anomaly detectors (IsoForest + Autoencoder)
    # IsoForest - typically lower means more anomalous; here we invert and scale to 0-1
    isof = LOADED_MODELS.get("isoforest")
    if isof:
        try:
            if hasattr(isof, "decision_function"):
                score = float(isof.decision_function([X_row])[0])
                # normalize loosely to 0-1 via logistic-ish
                import math
                iso_prob = 1.0 / (1.0 + math.exp(-score))
                model_scores["isoforest"] = iso_prob
                reasons.append({"source": "isoforest", "code": "ANOMALY", "explanation": "Anomaly detector produced a score"})
            else:
                model_scores["isoforest"] = 0.0
        except Exception as e:
            reasons.append({"source": "isoforest", "code": "ISO_ERROR", "explanation": f"IsoForest failed: {repr(e)}"})
    else:
        reasons.append({"source": "isoforest", "code": "ISO_MISSING", "explanation": "IsoForest model not loaded"})

    auto = LOADED_MODELS.get("autoencoder")
    if auto:
        try:
            # if the autoencoder returns reconstruction error, we can map to anomaly probability
            if hasattr(auto, "predict"):
                rec = auto.predict([X_row])
                # simple mapping -> anomaly prob (demo only)
                import numpy as np, math
                # compute L2 recon error
                err = float(np.linalg.norm(np.array(X_row) - np.array(rec).ravel()))
                a_prob = 1.0 / (1.0 + math.exp(-err))
                model_scores["autoencoder"] = a_prob
                reasons.append({"source": "autoencoder", "code": "RECON_ERR", "explanation": "Autoencoder reconstruction anomaly score used"})
            else:
                model_scores["autoencoder"] = 0.0
        except Exception as e:
            reasons.append({"source": "autoencoder", "code": "AUTO_ERROR", "explanation": f"Autoencoder failed: {repr(e)}"})
    else:
        reasons.append({"source": "autoencoder", "code": "AUTO_MISSING", "explanation": "Autoencoder model not loaded"})

    # Compose final score (weighted average of available model scores)
    available = [v for v in model_scores.values() if v is not None]
    if available:
        final_score = sum(available)/len(available)
    else:
        final_score = 0.0
        reasons.append({"source": "system", "code": "NO_MODELS", "explanation": "No scoring models available"})

    # Additional logic: basic device fingerprint check (basic fingerprint history check disabled per request)
    # We only mark an alert if device_fingerprint is missing OR matches known suspicious token (demo)
    if not submission.device_fingerprint:
        reasons.append({"source": "device_fp", "code": "NO_FP", "explanation": "No device fingerprint provided"})
    else:
        # demo rule: if fingerprint equals string "fp_demo" it's low confidence (this is just demo logic)
        if submission.device_fingerprint == "fp_demo":
            reasons.append({"source": "device_fp", "code": "FP_PLACEHOLDER", "explanation": "Placeholder fingerprint (demo) used"})

    # time-based suspicious login example - front-end should send metadata; this is a placeholder example
    # (You mentioned time detection across countries; that requires storing last login timestamps per account and comparing).
    # Here we just add a placeholder explanation entry.
    # reasons.append({"source":"temporal","code":"TIME_CHECK","explanation":"Temporal login check not implemented on server side (requires history store)"})

    # Build response
    resp = ScoreResult(
        model_scores=model_scores,
        final_score=final_score,
        reasons=reasons,
        model_versions={k: ("loaded" if LOADED_MODELS.get(k) else "missing") for k in MODEL_FILES.keys()}
    )

    return JSONResponse(status_code=200, content=resp.dict())

# -----------------------
# Enable reloading of models on demand (optional admin endpoint)
# -----------------------
@app.post("/admin/reload-models")
def admin_reload_models():
    load_models()
    return {"reloaded": {k: (LOADED_MODELS[k] is not None) for k in LOADED_MODELS.keys()}}


# -----------------------
# On startup, log the model status
# -----------------------
@app.on_event("startup")
def startup_event():
    # If ml/models directory doesn't exist, print helpful message
    if not os.path.isdir(MODEL_DIR):
        print(f"Warning: model directory {MODEL_DIR} does not exist. Create it and put model joblib files there.")
    print("Application startup complete. Model status:")
    for k, v in LOADED_MODELS.items():
        print(f"  {k}: {'loaded' if v is not None else 'missing / failed'}")
