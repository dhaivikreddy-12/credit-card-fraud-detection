"""Explore the fraud dataset imbalance."""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/transactions.csv")
os.makedirs("plots", exist_ok=True)

sns.countplot(data=df, x="is_fraud")
plt.title("Fraud vs Normal Transactions")
plt.tight_layout()
plt.savefig("plots/imbalance.png", dpi=120)
plt.close()

sns.boxplot(data=df, x="is_fraud", y="amount")
plt.title("Transaction Amount by Class")
plt.tight_layout()
plt.savefig("plots/amount.png", dpi=120)
plt.close()

print("Saved plots to plots/")
