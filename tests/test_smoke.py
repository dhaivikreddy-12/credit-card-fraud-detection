"""Behavioural tests for the credit card fraud detector.

The cached transactions.csv is ~142 MB, so every data test uses a bounded
nrows read. load() is never called (it would read the whole file) and
train_model.py is never imported (it trains at import time).
"""
import ast
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "transactions.csv"
NROWS = 200_000
TARGET = "Class"


def _make_imbalanced(n=200, n_pos=20, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    return X, y


def test_readme_and_license_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()


def test_load_data_exposes_csv_path_under_data():
    """Import the module only (no load() call) and inspect CSV_PATH."""
    spec = importlib.util.spec_from_file_location(
        "fraud_load_data", ROOT / "src" / "load_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    csv_path = pathlib.Path(module.CSV_PATH)
    assert csv_path.parent == pathlib.Path("data")
    assert csv_path.name == "transactions.csv"
    assert callable(module.load)


def test_bounded_read_has_class_column_and_severe_imbalance():
    if not CSV.is_file():
        pytest.skip("cached data/transactions.csv is not present")
    df = pd.read_csv(CSV, nrows=NROWS)
    assert TARGET in df.columns
    assert len(df) == NROWS
    fraud_rate = float(df[TARGET].mean())
    assert 0.0 <= fraud_rate < 0.01, f"expected severe imbalance, got {fraud_rate:.4%}"


def test_train_model_declares_class_target():
    tree = ast.parse((ROOT / "src" / "train_model.py").read_text(encoding="utf-8"))
    targets = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TARGET" for t in node.targets)
    ]
    assert TARGET in targets


def test_smote_upsamples_minority_to_requested_ratio():
    imblearn = pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE

    X, y = _make_imbalanced()
    assert (y == 1).sum() == 20

    X_res, y_res = SMOTE(random_state=42, sampling_strategy=0.3).fit_resample(X, y)
    n_majority = int((y_res == 0).sum())
    n_minority = int((y_res == 1).sum())
    assert len(y_res) == len(y) + n_minority - 20
    assert n_minority > 20, "SMOTE should synthesise new minority samples"
    assert n_minority == pytest.approx(0.3 * n_majority, rel=0.05)
    assert X_res.shape[1] == X.shape[1]


def test_forest_trains_on_smote_output_and_pr_curve_computes():
    imblearn = pytest.importorskip("imblearn")
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_curve

    X, y = _make_imbalanced()
    X_res, y_res = SMOTE(random_state=42, sampling_strategy=0.5).fit_resample(X, y)

    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X_res, y_res)
    preds = model.predict(X_res)
    assert preds.shape[0] == X_res.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})

    y_prob = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    assert len(precision) == len(recall) == len(thresholds) + 1
    assert np.all((precision >= 0) & (precision <= 1))
    assert np.all((recall >= 0) & (recall <= 1))