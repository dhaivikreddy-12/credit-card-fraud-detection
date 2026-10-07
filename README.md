# 💳 Credit Card Fraud Detection

> *"Out of 10,000 transactions, maybe 5 are fraud. Find them without crying wolf on the other 9,995."*

This is the problem that taught me how **class imbalance** really works. In fraud detection, catching fraud matters far more than overall accuracy — a model that's 99.9% "accurate" but misses all fraud is useless. This project digs into precision, recall, and the F1/F2 trade-off properly.

## What this project does

- Loads a realistic transaction dataset with a severe imbalance (~0.5% fraud).
- Handles imbalance with **SMOTE** (synthetic minority oversampling).
- Trains **Random Forest** and compares it with a baseline.
- Evaluates with precision/recall, F1, and the confusion matrix.
- Shows how changing the decision threshold shifts the precision/recall balance.

## The dataset

Synthetic but realistic (`data/transactions.csv`), ~20,000 transactions:

| Feature                | Description                          |
|------------------------|--------------------------------------|
| `amount`               | Transaction amount ($)               |
| `hour`                 | Hour of day (0–23)                   |
| `merchant_risk`        | Merchant risk score (0–1)            |
| `distance_km`          | Distance from home (km)              |
| `foreign_transaction`  | 1 if international, else 0           |
| `is_fraud`             | Target (1 = fraud)                   |

## How to run it

```bash
pip install -r requirements.txt

# Generate data + full pipeline + evaluation
python fraud.py

# Visualise the imbalance
python explore.py
```

## What I learned

- Why accuracy is the *wrong* metric here — and precision/recall are right.
- What SMOTE does under the hood and when it helps (and when it doesn't).
- How to tune the decision threshold instead of just taking 0.5.
- That real-world data is almost never clean or balanced.

## Results

With SMOTE and threshold tuning, the model catches **~85–90% of fraud** while keeping false alarms low. The key insight: remember it's a trade-off, and you pick the point that matches the business cost.

---

*Built with Python, pandas, scikit-learn, imbalanced-learn, matplotlib. Made for learning, by a student, for students.*
