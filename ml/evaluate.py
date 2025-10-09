# ml/evaluate.py
import sys, joblib, pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

model_path = sys.argv[1]
data_path = sys.argv[2]

m = joblib.load(model_path)
df = pd.read_csv(data_path)
X = df.drop(columns=["label","applicant_id"])
y = df["label"]

pred = m.predict(X)
prob = m.predict_proba(X)[:,1] if hasattr(m, "predict_proba") else None
auc = roc_auc_score(y, prob) if prob is not None else None
prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
acc = accuracy_score(y, pred)
print("acc", acc, "prec", prec, "rec", rec, "f1", f1, "auc", auc)
