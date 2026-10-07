# 💳 Credit Card Fraud Detection

> *"Out of 284,807 transactions, 492 are fraud. Find them without crying wolf on the other 284,315."*

This is the project that taught me what class imbalance actually means. On real credit card transaction data, fraud is **0.17%** of all transactions. A model that answers "not fraud" every single time is 99.83% accurate and completely useless — and this repo shows you exactly that trap before showing the fix.

## What this project does

- Loads the real credit card transaction dataset (284,807 rows, 30 PCA features).
- Demonstrates why accuracy and ROC-AUC both hide the problem here.
- Applies **SMOTE** to oversample the minority class during training only.
- Trains Random Forest and class-weighted Logistic Regression.
- Sweeps the decision threshold to find the real precision/recall trade-off.
- Reports average precision and plots the precision-recall curve.

## The dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) via OpenML — European cardholders, September 2013, two days of transactions.

| Column | Description |
|---|---|
| `Time` | Seconds since first transaction |
| `V1`–`V28` | PCA components (original values are confidential) |
| `Amount` | Transaction amount |
| `Class` | 1 = fraud, 0 = legitimate — **target** |

Fraud rate: **0.172%** (492 of 284,807).

## How to run it

```bash
pip install -r requirements.txt

python fraud.py    # load, apply SMOTE, train, sweep thresholds
```

## Project structure

```
credit-card-fraud-detection/
├── data/
│   └── transactions.csv
├── src/
│   ├── load_data.py   # fetch + cache
│   ├── train_model.py # SMOTE pipeline + threshold sweep
│   └── explore.py
├── tests/
├── fraud.py
├── requirements.txt
└── README.md
```

## What I learned

- That a 99.83%-accurate model can be worthless, and how to prove it in one line.
- Why ROC-AUC flatters you on severe imbalance, and why average precision doesn't.
- What SMOTE actually does — synthesising minority points in feature space — and that it must only ever touch the training split.
- That the threshold is a business decision, not a modelling one. Moving it from 0.5 to 0.688 changed recall from ~0.68 to 0.806.

## Results

20% stratified test split, 98 fraud cases in the test set:

| Model | Average Precision | Best F1 | Precision | Recall |
|---|---|---|---|---|
| **RandomForest** | **0.878** | **0.873** | 0.952 | 0.806 |
| LogisticRegression | 0.722 | 0.825 | 0.833 | 0.816 |

Random Forest at its tuned threshold caught **79 of 98** fraud transactions while raising only **4** false alarms out of 56,864. That trade-off is genuinely good — but the point of the project is that *you* choose where to sit on that curve.

---

*Built with Python, pandas, scikit-learn, imbalanced-learn, matplotlib. Real transaction data, honestly measured.*
