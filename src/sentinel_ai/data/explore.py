"""Command line data validation and exploratory summary for PaySim CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sentinel_ai.data.validation import validate_csv, validate_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--output-dir", default="reports/data-quality")
    parser.add_argument("--sample-fraction", type=float, default=0.05,
                        help="Optional reproducible row sample fraction (0, 1].")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser


def run(input_path: Path, output_dir: Path, fraction: float, seed: int, chunksize: int) -> None:
    if not 0 < fraction <= 1:
        raise ValueError("--sample-fraction must be in (0, 1].")
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = validate_csv(input_path, chunksize)
    # Keep the EDA working set bounded to at most one sampled fraction of each chunk.
    samples = []
    offset = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        if fraction == 1:
            samples.append(chunk)
        else:
            samples.append(chunk.sample(frac=fraction, random_state=seed + offset))
        offset += 1
    frame = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    quality_path = output_dir / "data_quality.json"
    quality_path.write_text(json.dumps(quality.to_dict(), indent=2), encoding="utf-8")

    numeric = frame.select_dtypes(include="number")
    summary = {
        "source": str(input_path),
        "sampled": fraction < 1,
        "sample_fraction": fraction,
        "seed": seed,
        "rows_analyzed": int(len(frame)),
        "type_counts": frame["type"].value_counts(dropna=False).to_dict()
        if "type" in frame else {},
        "numeric_describe": numeric.describe().replace({float("inf"): None}).to_dict(),
        "fraud_rate_by_type": frame.groupby("type", dropna=False)["isFraud"].mean().to_dict()
        if {"type", "isFraud"}.issubset(frame.columns) else {},
    }
    (output_dir / "exploratory_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"Quality report: {quality_path}")
    print(f"EDA summary: {output_dir / 'exploratory_summary.json'}")
    print(f"Rows analyzed: {len(frame):,}; fraud prevalence: {quality.fraud_prevalence}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(Path(args.input), Path(args.output_dir), args.sample_fraction, args.seed, args.chunksize)
