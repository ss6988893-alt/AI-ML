"""Load saved model bundles; API startup never trains a model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib


def load_model(path: str | Path):
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"No trained model artifact at {artifact}")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError("Model artifact has an unsupported format")
    return bundle


def load_metadata(path: str | Path) -> dict:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))
