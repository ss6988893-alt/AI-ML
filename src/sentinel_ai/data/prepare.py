"""Validate PaySim and convert its CSV into bounded Parquet part files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sentinel_ai.data.validation import require_valid, validate_csv


def prepare(input_path: Path, output_dir: Path, chunksize: int = 250_000) -> int:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; choose a new path")
    report = validate_csv(input_path, chunksize)
    require_valid(report)
    output_dir.mkdir(parents=True)
    count = 0
    for index, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
        target = output_dir / f"part-{index:05d}.parquet"
        chunk.to_parquet(target, index=False)
        count += len(chunk)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--output-dir", default="data/processed/paysim_parquet")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    rows = prepare(Path(args.input), Path(args.output_dir), args.chunksize)
    print(f"Wrote {rows:,} rows as Parquet parts to {args.output_dir}")


if __name__ == "__main__":
    main()
