"""Load the real Credit Card Fraud dataset (284,807 transactions).

Source: OpenML 'creditcard' (UCI-derived, Kaggle Credit Card Fraud Detection).
Class imbalance is severe: about 0.172% of transactions are fraud.
"""
import os
import pandas as pd

CSV_PATH = "data/transactions.csv"


def load():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(name="creditcard", version=1, as_frame=True)
    df = bunch.frame.copy()
    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} transactions -> {CSV_PATH}")
    print(f"Fraud rate: {df['Class'].mean():.3%}")
