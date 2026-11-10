"""Leakage-aware PaySim feature schema shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "step", "amount", "log_amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "sender_balance_change",
    "receiver_balance_change", "sender_balance_inconsistent",
    "receiver_balance_inconsistent", "amount_to_sender_balance",
    "amount_to_receiver_balance",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
IDENTIFIER_COLUMNS = {"nameOrig", "nameDest"}
LEAKAGE_COLUMNS = {"isFraud", "isFlaggedFraud"}
REQUIRED_INPUTS = {
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
}


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build only transaction-time features; labels and account IDs are excluded."""
    missing = sorted(REQUIRED_INPUTS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing transaction fields: {', '.join(missing)}")
    result = pd.DataFrame(index=frame.index)
    result["type"] = frame["type"].fillna("UNKNOWN").astype(str)
    for field in ("step", "amount", "oldbalanceOrg", "newbalanceOrig",
                  "oldbalanceDest", "newbalanceDest"):
        result[field] = pd.to_numeric(frame[field], errors="coerce")
        result[field] = result[field].replace([np.inf, -np.inf], np.nan).clip(lower=0)
    result["log_amount"] = np.log1p(result["amount"])
    result["sender_balance_change"] = result["oldbalanceOrg"] - result["newbalanceOrig"]
    result["receiver_balance_change"] = result["newbalanceDest"] - result["oldbalanceDest"]
    sender_delta = result["amount"] - result["sender_balance_change"]
    receiver_delta = result["amount"] - result["receiver_balance_change"]
    result["sender_balance_inconsistent"] = (sender_delta.abs() > 0.01).astype(float)
    result["receiver_balance_inconsistent"] = (receiver_delta.abs() > 0.01).astype(float)
    result["amount_to_sender_balance"] = result["amount"] / result["oldbalanceOrg"].replace(0, np.nan)
    result["amount_to_receiver_balance"] = result["amount"] / result["oldbalanceDest"].replace(0, np.nan)
    return result[FEATURES]


def build_preprocessor():
    """Return a train-fitted imputation/encoding/scaling pipeline component."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")
