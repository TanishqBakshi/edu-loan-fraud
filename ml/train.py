#!/usr/bin/env python3
"""
ml/train.py

Train baseline and advanced fraud-detection models on a labeled CSV.

Produces:
  - ml/models/rf.joblib
  - ml/models/lr.joblib
  - ml/models/autoencoder.joblib
  - ml/models/isoforest.joblib
  - ml/models/metadata.json
  - ml/models/eval_report.json

Expected CSV format (columns):
  - applicant_id (optional)
  - avg_hold_time, std_hold, backspace_freq, interkey_mean, mouse_jitter,
    ocr_confidence, device_fp_sim
  - label (0/1)  -- 1 means fraudulent
"""

import os
import argparse
import json
import time
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import joblib

# -----------------------
# Simple wrappers
# -----------------------
class AutoencoderWrapper:
    """
    Wrap an MLPRegressor trained as an autoencoder (X -> X).
    Provides:
      - reconstruction_error(X) -> array of errors
      - decision_function(X) -> higher => less anomalous (to mimic sklearn API, we invert errors)
    """
    def __init__(self, model, scaler_mean=0.0, scaler_std=1.0):
        self.model = model
        self.mean = scaler_mean
        self.std = scaler_std

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        # model.predict returns reconstructed X
        X_recon = self.model.predict(X)
        err = np.mean((X - X_recon) ** 2, axis=1)
        return err

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # return values similar to sklearn.decision_function: larger => less anomalous
        err = self.reconstruction_error(X)
        # invert and normalize to have higher = less anomalous
        return -err

class IsolationForestWrapper:
    """
    Simple wrapper to keep consistent API; we save the fitted IsolationForest instance.
    """
    def __init__(self, model):
        self.model = model

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # sklearn IsolationForest: larger scores => less anomalous
        return self.model.decision_function(X)

# -----------------------
# Utilities
# -----------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity: drop rows without label
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column with 0/1 values.")
    # Drop rows with NaN in required features
    required = ["avg_hold_time","std_hold","backspace_freq","interkey_mean","mouse_jitter","ocr_confidence","device_fp_sim"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df = df.dropna(subset=required + ["label"])
    return df

def build_baseline_lr() -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])
    return pipe

def build_baseline_rf() -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    return pipe

def build_autoencoder(hidden_layer_sizes=(16,8,16), random_state=42) -> MLPRegressor:
    # MLPRegressor as autoencoder: fit X->X
    ae = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation="relu",
                      solver="adam", max_iter=300, random_state=random_state)
    return ae

def save_joblib(obj, path: str):
    joblib.dump(obj, path)
    print(f"Saved artifact: {path}")

# -----------------------
# Train + Evaluate
# -----------------------
def train_and_evaluate(data_csv: str, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(data_csv)
    feature_cols = ["avg_hold_time","std_hold","backspace_freq","interkey_mean","mouse_jitter","ocr_confidence","device_fp_sim"]
    X = df[feature_cols].values.astype(float)
    y = df["label"].astype(int).values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {"timestamp": int(time.time()), "models": {}}

    # --- Logistic Regression ---
    print("Training Logistic Regression...")
    lr = build_baseline_lr()
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:,1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    pr_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
    print(f"LR AUC={auc_lr:.4f} precision={pr_lr:.3f} recall={rec_lr:.3f} f1={f1_lr:.3f}")
    save_joblib(lr, os.path.join(out_dir, "lr.joblib"))
    results["models"]["lr"] = {"auc": auc_lr, "precision": pr_lr, "recall": rec_lr, "f1": f1_lr}

    # --- Random Forest ---
    print("Training Random Forest...")
    rf = build_baseline_rf()
    rf.fit(X_train, y_train)
    if hasattr(rf, "predict_proba"):
        y_prob_rf = rf.predict_proba(X_test)[:,1]
    else:
        y_prob_rf = rf.predict(X_test)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    pr_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
    print(f"RF AUC={auc_rf:.4f} precision={pr_rf:.3f} recall={rec_rf:.3f} f1={f1_rf:.3f}")
    save_joblib(rf, os.path.join(out_dir, "rf.joblib"))
    results["models"]["rf"] = {"auc": auc_rf, "precision": pr_rf, "recall": rec_rf, "f1": f1_rf}

    # --- Isolation Forest (anomaly detector) ---
    print("Training IsolationForest (anomaly detector)...")
    # We fit IsolationForest on "normal" class (label==0) if enough samples exist.
    try:
        normal_mask = (y_train == 0)
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        iso.fit(X_train[normal_mask])
        # decision_function: higher => less anomalous
        iso_scores = iso.decision_function(X_test)
        # For metric compute AUC treating lower decision_function -> anomaly -> label 1
        iso_auc = roc_auc_score(y_test, -iso_scores)
        print(f"IsoForest AUC (anomaly)={iso_auc:.4f}")
        save_joblib(IsolationForestWrapper(iso), os.path.join(out_dir, "isoforest.joblib"))
        results["models"]["isoforest"] = {"auc": iso_auc}
    except Exception as e:
        print("IsolationForest training failed:", e)

    # --- Autoencoder (MLPRegressor) as deep anomaly detector ---
    print("Training Autoencoder (MLPRegressor) for reconstruction anomaly detection...")
    try:
        # Train autoencoder on normal class only (label==0)
        normal_mask = (y_train == 0)
        if normal_mask.sum() < 10:
            print("Not enough normal samples for AE training; using full train set.")
            ae_train_X = X_train
        else:
            ae_train_X = X_train[normal_mask]

        # scale features for AE
        scaler = StandardScaler()
        ae_train_scaled = scaler.fit_transform(ae_train_X)
        n_features = ae_train_scaled.shape[1]
        # small bottleneck
        hidden = (max(8, n_features*2), max(4, n_features//1), max(8, n_features*2))
        # create MLPRegressor (autoencoder)
        ae = build_autoencoder(hidden_layer_sizes=(16,8,16))
        ae.fit(ae_train_scaled, ae_train_scaled)  # X -> X
        # compute reconstruction error on test set
        X_test_scaled = scaler.transform(X_test)
        recon = ae.predict(X_test_scaled)
        rec_err = np.mean((X_test_scaled - recon)**2, axis=1)
        # higher error -> more anomalous -> treat as positive
        ae_auc = roc_auc_score(y_test, rec_err)
        print(f"Autoencoder AUC (reconstruction)={ae_auc:.4f}")
        # wrap and save
        ae_wrapper = AutoencoderWrapper(ae, scaler_mean=float(scaler.mean_.mean()), scaler_std=float(scaler.scale_.mean()))
        # Save scaler inside wrapper? We only need the MLP; but save wrapper for API usage
        save_joblib(ae_wrapper, os.path.join(out_dir, "autoencoder.joblib"))
        results["models"]["autoencoder"] = {"auc": ae_auc}
    except Exception as e:
        print("Autoencoder training failed:", e)

    # --- Eval report + metadata ---
    eval_report = {
        "timestamp": int(time.time()),
        "metrics": results["models"]
    }
    with open(os.path.join(out_dir, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
    metadata = {
        "created_at": int(time.time()),
        "model_versions": {k: f"{int(time.time())}" for k in ["lr","rf","autoencoder","isoforest"]},
        "feature_columns": feature_cols
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Training complete. Artifacts saved to", out_dir)
    return {"eval": eval_report, "metadata": metadata}

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--data", required=True, help="path to labeled CSV (sample_labeled.csv)")
    parser.add_argument("--out", required=True, help="output folder for models (ml/models)")
    args = parser.parse_args()
    out = args.out
    res = train_and_evaluate(args.data, out)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
