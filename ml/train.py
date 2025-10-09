# ml/train.py
import argparse
import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

def build_rf():
    return Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])

def build_lr():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV input path")
    parser.add_argument("--out", required=True, help="Directory to save models")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    # expected columns: applicant_id + feature cols + label
    if "label" not in df.columns:
        raise SystemExit("CSV missing 'label' column")

    X = df.drop(columns=["label","applicant_id"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "rf": build_rf(),
        "lr": build_lr()
    }

    report = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
        auc = roc_auc_score(y_test, prob) if prob is not None else None
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        acc = accuracy_score(y_test, pred)
        fname = os.path.join(args.out, f"{name}.joblib")
        joblib.dump(model, fname)
        report[name] = {"auc": auc, "precision": float(prec), "recall": float(rec), "f1": float(f1), "acc": float(acc), "path": fname}

    # Save metadata
    meta = {"models": report, "version": "v0.1.0"}
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete. Artifacts saved to", args.out)
    print(json.dumps(report, indent=2))
