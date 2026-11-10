"""PaySim schema validation and data-quality reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)
NUMERIC_COLUMNS = (
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "isFraud", "isFlaggedFraud",
)
NON_NEGATIVE_COLUMNS = (
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest",
)


class DataValidationError(ValueError):
    """Raised when a dataset cannot safely be interpreted as PaySim data."""


@dataclass(frozen=True)
class DataQualityReport:
    path: str
    rows: int
    columns: int
    missing_columns: list[str]
    extra_columns: list[str]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    duplicate_rows: int
    fraud_count: int | None
    fraud_prevalence: float | None
    invalid_numeric_counts: dict[str, int]
    negative_value_counts: dict[str, int]
    invalid_target_count: int
    flagged_fraud_count: int | None
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_frame(frame: pd.DataFrame, path: str = "<memory>") -> DataQualityReport:
    """Inspect a full in-memory frame and fail only for schema/target violations."""
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    extra = [str(column) for column in frame.columns if column not in EXPECTED_COLUMNS]
    numeric_invalid: dict[str, int] = {}
    negative: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        conversion_failures = values.isna() & frame[column].notna()
        non_finite = values.notna() & ~np.isfinite(values)
        numeric_invalid[column] = int((conversion_failures | non_finite).sum())
        if column in NON_NEGATIVE_COLUMNS:
            negative[column] = int((values < 0).sum())

    invalid_target = 0
    fraud_count: int | None = None
    prevalence: float | None = None
    flagged_count: int | None = None
    if "isFraud" in frame:
        target = pd.to_numeric(frame["isFraud"], errors="coerce")
        invalid_target = int((~target.isin([0, 1])).sum())
        fraud_count = int((target == 1).sum())
        prevalence = float(fraud_count / len(frame)) if len(frame) else 0.0
    if "isFlaggedFraud" in frame:
        flagged = pd.to_numeric(frame["isFlaggedFraud"], errors="coerce")
        flagged_count = int((flagged == 1).sum())

    valid = (
        not missing
        and invalid_target == 0
        and all(count == 0 for count in numeric_invalid.values())
        and all(count == 0 for count in negative.values())
    )
    return DataQualityReport(
        path=path,
        rows=int(len(frame)),
        columns=int(len(frame.columns)),
        missing_columns=missing,
        extra_columns=extra,
        dtypes={str(k): str(v) for k, v in frame.dtypes.items()},
        null_counts={str(k): int(v) for k, v in frame.isna().sum().items()},
        duplicate_rows=int(frame.duplicated().sum()),
        fraud_count=fraud_count,
        fraud_prevalence=prevalence,
        invalid_numeric_counts=numeric_invalid,
        negative_value_counts=negative,
        invalid_target_count=invalid_target,
        flagged_fraud_count=flagged_count,
        valid=valid,
    )


def validate_csv(path: str | Path, chunksize: int = 250_000) -> DataQualityReport:
    """Validate a CSV in chunks, limiting memory for the full 6M-row source.

    Duplicate counts are exact within each chunk; duplicates spanning chunk
    boundaries are not counted. Use a full dataset fingerprint/dedup pass when
    exact global duplicate detection is required.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}. See docs/data-acquisition.md.")
    rows = 0
    nulls: dict[str, int] = {}
    bad_numeric: dict[str, int] = {}
    negatives: dict[str, int] = {}
    duplicates = frauds = flagged = invalid_target = 0
    dtypes: dict[str, str] = {}
    missing: list[str] | None = None
    extra: list[str] = []
    for chunk in pd.read_csv(source, chunksize=chunksize):
        report = validate_frame(chunk, str(source))
        if missing is None:
            missing = report.missing_columns
            extra = report.extra_columns
        rows += report.rows
        duplicates += report.duplicate_rows
        invalid_target += report.invalid_target_count
        frauds += report.fraud_count or 0
        flagged += report.flagged_fraud_count or 0
        dtypes = report.dtypes
        for col, count in report.null_counts.items():
            nulls[col] = nulls.get(col, 0) + count
        for col, count in report.invalid_numeric_counts.items():
            bad_numeric[col] = bad_numeric.get(col, 0) + count
        for col, count in report.negative_value_counts.items():
            negatives[col] = negatives.get(col, 0) + count
    missing = missing or list(EXPECTED_COLUMNS)
    valid = (
        not missing
        and invalid_target == 0
        and all(v == 0 for v in bad_numeric.values())
        and all(v == 0 for v in negatives.values())
    )
    return DataQualityReport(
        path=str(source), rows=rows, columns=len(dtypes), missing_columns=missing,
        extra_columns=extra, dtypes=dtypes, null_counts=nulls, duplicate_rows=duplicates,
        fraud_count=frauds, fraud_prevalence=(frauds / rows if rows else 0.0),
        invalid_numeric_counts=bad_numeric, negative_value_counts=negatives,
        invalid_target_count=invalid_target, flagged_fraud_count=flagged, valid=valid,
    )


def require_valid(report: DataQualityReport) -> None:
    """Raise actionable errors when required schema or values are invalid."""
    issues = []
    if report.missing_columns:
        issues.append(f"missing required columns: {', '.join(report.missing_columns)}")
    if report.invalid_target_count:
        issues.append(f"{report.invalid_target_count} invalid isFraud values (expected 0/1)")
    bad = {k: v for k, v in report.invalid_numeric_counts.items() if v}
    if bad:
        issues.append(f"non-numeric values: {bad}")
    bad_negative = {k: v for k, v in report.negative_value_counts.items() if v}
    if bad_negative:
        issues.append(f"negative values in non-negative fields: {bad_negative}")
    if issues:
        raise DataValidationError("Dataset validation failed: " + "; ".join(issues))
