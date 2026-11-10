# PaySim data acquisition and Week 1 validation

## Download

1. Create/sign in to a Kaggle account and accept the dataset's terms on the [PaySim dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1).
2. Download the archive from Kaggle's dataset page (or use Kaggle CLI after configuring your own credentials).
3. Extract `PS_20174392719_1491204439457_log.csv` into `data/raw/`.
4. Keep the raw file private and local. `data/raw/**` is excluded from Git; never commit Kaggle credentials or raw transaction data.

No credentials or download automation are required by this project. The dataset is not assumed to exist locally.

## Validation and EDA

Run from the repository root:

```powershell
python -m sentinel_ai.data.explore --input data/raw/PS_20174392719_1491204439457_log.csv
```

The validator streams CSV chunks (default 250,000 rows) and reports row/column counts, expected/missing/extra columns, dtypes, null counts, duplicate rows, numeric conversion failures, non-finite values, negative numeric values, target validity, fraud count/prevalence, and flagged-rule count. It exits with an exception for a missing file; the report's `valid` flag and `require_valid` helper reject schema, invalid numeric, non-binary target, and negative non-negative-field violations. Nulls and duplicates are reported for review rather than silently repaired. Negative balance/amount counts need investigation.

Duplicate counts are exact within each chunk; a row duplicate that crosses chunk boundaries may not be counted. Global exact duplicate detection on millions of records needs a separate external sort/hash pass and is not claimed by this first milestone.

The exploratory summary defaults to a fixed-seed 5% sample from each chunk. `--sample-fraction` must be in `(0, 1]`; the sampled summary is clearly labeled in JSON. The full quality scan runs regardless of the EDA sample rate. Sampled type counts and numeric descriptions are not full-dataset aggregates. To analyze every row, set fraction to 1, ensuring sufficient memory for the EDA summary frame.

To create Parquet part files after validation, run `python -m sentinel_ai.data.prepare --output-dir data/processed/paysim_parquet`. Conversion streams one CSV chunk at a time and refuses to reuse an existing output directory. Processed data is Git-ignored.

Outputs: `reports/data-quality/data_quality.json` and `exploratory_summary.json`. Reports are generated artifacts and do not contain account IDs in aggregate output.

## Schema policy

Expected columns are the eleven PaySim columns listed in `src/sentinel_ai/data/validation.py`. `isFraud` must be binary. `isFlaggedFraud` is reported only for data inspection and is excluded from future model features to avoid rule leakage. Account IDs are not model features and must not appear in dashboard, logs, explanations, or sample API data. No preprocessing or imputation is performed by the validator.
