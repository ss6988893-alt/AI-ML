import pandas as pd
import pytest

from sentinel_ai.data.validation import (
    EXPECTED_COLUMNS,
    DataValidationError,
    require_valid,
    validate_frame,
)


@pytest.fixture
def paysim_frame():
    return pd.DataFrame({
        "step": [1, 1, 2], "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [10.0, 20.0, 30.0], "nameOrig": ["a", "b", "c"],
        "oldbalanceOrg": [20.0, 30.0, 40.0], "newbalanceOrig": [10.0, 10.0, 10.0],
        "nameDest": ["x", "y", "z"], "oldbalanceDest": [0.0, 0.0, 0.0],
        "newbalanceDest": [10.0, 20.0, 30.0], "isFraud": [0, 1, 0],
        "isFlaggedFraud": [0, 0, 0],
    })


def test_valid_schema_and_fraud_summary(paysim_frame):
    report = validate_frame(paysim_frame)
    assert report.valid
    assert report.rows == 3
    assert report.fraud_count == 1
    assert report.fraud_prevalence == pytest.approx(1 / 3)
    assert report.missing_columns == []
    require_valid(report)


def test_missing_required_column_is_reported(paysim_frame):
    report = validate_frame(paysim_frame.drop(columns="step"))
    assert "step" in report.missing_columns
    with pytest.raises(DataValidationError, match="step"):
        require_valid(report)


def test_non_numeric_values_and_invalid_target_are_reported(paysim_frame):
    paysim_frame["amount"] = paysim_frame["amount"].astype(object)
    paysim_frame.loc[0, "amount"] = "unknown"
    paysim_frame.loc[1, "isFraud"] = 2
    report = validate_frame(paysim_frame)
    assert report.invalid_numeric_counts["amount"] == 1
    assert report.invalid_target_count == 1
    with pytest.raises(DataValidationError):
        require_valid(report)


def test_nulls_duplicates_negatives_and_excluded_flag_are_visible(paysim_frame):
    frame = pd.concat([paysim_frame, paysim_frame.iloc[[0]]], ignore_index=True)
    frame.loc[2, "newbalanceOrig"] = None
    frame.loc[1, "amount"] = -1
    report = validate_frame(frame)
    assert report.null_counts["newbalanceOrig"] == 1
    assert report.duplicate_rows == 1
    assert report.negative_value_counts["amount"] == 1
    assert not report.valid
    with pytest.raises(DataValidationError, match="negative"):
        require_valid(report)
    # Rule flag is reported for inspection, but later modeling explicitly excludes it.
    assert report.flagged_fraud_count == 0
    assert len(EXPECTED_COLUMNS) == 11
