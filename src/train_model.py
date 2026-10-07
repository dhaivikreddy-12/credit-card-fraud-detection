"""Fraud detection pipeline on the real credit card transaction dataset."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.load_data import load

TARGET = "Class"

df = load()
print(f"Loaded {len(df)} transactions")
print(f"Fraud rate: {df[TARGET].mean():.3%}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# On this dataset PCA components are already on a comparable scale,
# but a scaler is cheap insurance when new columns are added.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_res, y_res = smote.fit_resample(X_train_s, y_train)
print(f"After SMOTE: normal={int((y_res==0).sum())}, fraud={int((y_res==1).sum())}")

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2),
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
}

best = None
for name, model in models.items():
    model.fit(X_res, y_res)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    ap = average_precision_score(y_test, y_prob)
    print(f"\n--- {name} ---")
    print(f"Average precision: {ap:.3f}")
    prec, rec, thresh = precision_recall_curve(y_test, y_prob)
    f1 = 2 * (prec * rec) / np.maximum(prec + rec, 1e-9)
    k = int(np.argmax(f1[:-1]))
    print(f"Best F1 {f1[k]:.3f} at threshold {thresh[k]:.3f} (precision {prec[k]:.3f}, recall {rec[k]:.3f})")
    if best is None or ap > best[0]:
        best = (ap, name, y_prob, prec, rec, thresh, f1, k)

ap, best_name, y_prob, prec, rec, thresh, f1, k = best
print(f"\nBest model: {best_name} (average precision {ap:.3f})")

y_pred = (y_prob >= thresh[k]).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix at best threshold:")
print(cm)
tp, fn = cm[1, 1], cm[1, 0]
print(f"Fraud caught: {tp}/{tp+fn} ({tp/(tp+fn):.1%} recall)")

plt.plot(rec, prec, color="#c44e52")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve ({best_name})")
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=120)
print("Saved PR curve to pr_curve.png")
