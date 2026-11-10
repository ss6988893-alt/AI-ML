import numpy as np
import pandas as pd
import pytest

from sentinel_ai.features.engineering import FEATURES, engineer_features


def sample():
    return pd.DataFrame({"step": [1], "type": ["TRANSFER"], "amount": [100.0],
        "oldbalanceOrg": [100.0], "newbalanceOrig": [0.0], "oldbalanceDest": [0.0],
        "newbalanceDest": [100.0], "isFraud": [1], "isFlaggedFraud": [0],
        "nameOrig": ["private-a"], "nameDest": ["private-b"]})


def test_features_share_schema_and_exclude_leakage_and_identifiers():
    result = engineer_features(sample())
    assert list(result.columns) == FEATURES
    assert not {"isFraud", "isFlaggedFraud", "nameOrig", "nameDest"} & set(result.columns)
    assert result.iloc[0]["sender_balance_change"] == 100
    assert result.iloc[0]["receiver_balance_change"] == 100


def test_zero_denominators_become_missing_not_infinite():
    result = engineer_features(sample())
    assert np.isnan(result.iloc[0]["amount_to_receiver_balance"])
    assert np.isfinite(result.select_dtypes("number").dropna(axis=1).to_numpy()).all()


def test_missing_required_input_fails_clearly():
    with pytest.raises(ValueError, match="oldbalanceDest"):
        engineer_features(sample().drop(columns="oldbalanceDest"))


def test_invalid_numeric_values_are_coerced_for_training_imputation():
    frame = sample()
    frame["amount"] = frame["amount"].astype(object)
    frame.loc[0, "amount"] = "bad"
    result = engineer_features(frame)
    assert np.isnan(result.iloc[0]["amount"])
