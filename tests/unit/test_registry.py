import joblib
import pytest

from sentinel_ai.models.registry import load_model


def test_missing_model_artifact_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained model"):
        load_model(tmp_path / "missing.joblib")


def test_invalid_bundle_is_rejected(tmp_path):
    path = tmp_path / "invalid.joblib"
    joblib.dump({"prediction": 0.9}, path)
    with pytest.raises(ValueError, match="unsupported format"):
        load_model(path)
