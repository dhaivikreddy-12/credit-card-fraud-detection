"""Generate synthetic transaction data with heavy class imbalance."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(99)

n = 20000
fraud_rate = 0.005

n_fraud = int(n * fraud_rate)
n_normal = n - n_fraud

amount = np.concatenate([
    np.clip(rng.lognormal(3.8, 0.7, n_normal), 1, 500),
    np.clip(rng.lognormal(4.6, 1.1, n_fraud), 1, 2000),
])
hour = rng.integers(0, 24, n)
merchant_risk = np.concatenate([
    rng.beta(2, 8, n_normal),
    rng.beta(8, 2, n_fraud),
])
distance = np.concatenate([
    rng.exponential(15, n_normal),
    rng.exponential(120, n_fraud),
])
foreign = np.concatenate([
    rng.binomial(1, 0.12, n_normal),
    rng.binomial(1, 0.7, n_fraud),
])
is_fraud = np.array([0]*n_normal + [1]*n_fraud, dtype=int)

df = pd.DataFrame({
    "amount": amount.round(2),
    "hour": hour,
    "merchant_risk": merchant_risk.round(4),
    "distance_km": distance.round(2),
    "foreign_transaction": foreign,
    "is_fraud": is_fraud,
})
df = df.sample(frac=1, random_state=99).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions.csv", index=False)
print(f"Generated {len(df)} transactions, fraud rate {df['is_fraud'].mean():.3%}")
