"""Fraud detection pipeline with SMOTE and threshold tuning."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/transactions.csv")
print(f"Fraud rate: {df['is_fraud'].mean():.3%}")

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_r, y_train_r = smote.fit_resample(X_train_s, y_train)
print(f"After SMOTE: normal={sum(y_train_r==0)}, fraud={sum(y_train_r==1)}")

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_r, y_train_r)

y_prob = model.predict_proba(X_test_s)[:, 1]

prec, rec, thresh = precision_recall_curve(y_test, y_prob)

f1 = 2 * (prec * rec) / np.maximum(prec + rec, 1e-9)
best = np.argmax(f1[:-1])
print(f"\nBest threshold: {thresh[best]:.3f}")
print(f"Best F1       : {f1[best]:.3f}")
print(f"Precision     : {prec[best]:.3f}")
print(f"Recall        : {rec[best]:.3f}")

y_pred = (y_prob >= thresh[best]).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)
print(f"Fraud caught : {cm[1,1]} / {cm[1,1]+cm[1,0]} ({rec[best]:.1%})")

import matplotlib.pyplot as plt
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Fraud Detection)")
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=120)
print("\nSaved PR curve to pr_curve.png")
