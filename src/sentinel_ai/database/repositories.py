"""Small persistence operations, kept separate from API route handlers."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sentinel_ai.database.models import ModelVersion, Prediction


def register_model(session: Session, metadata: dict, artifact_uri: str) -> None:
    version = metadata.get("model_version", "unknown")
    existing = session.scalar(select(ModelVersion).where(ModelVersion.version == version))
    if existing:
        return
    try:
        with session.begin_nested():
            session.add(ModelVersion(
                version=version,
                training_dataset_fingerprint=metadata.get("dataset_fingerprint_sha256", "unknown"),
                model_artifact_uri=artifact_uri,
                metrics_json={"validation": metadata.get("validation_metrics", {}),
                              "test": metadata.get("test_metrics", {})},
                configuration_json={"family": metadata.get("model_family"),
                                    "threshold": metadata.get("selected_threshold")},
                status="active",
            ))
            session.flush()
    except IntegrityError:
        # Another worker may register the same immutable version concurrently.
        existing = session.scalar(select(ModelVersion).where(ModelVersion.version == version))
        if existing is None:
            raise


def save_prediction(session: Session, **fields) -> Prediction:
    prediction = Prediction(**fields)
    session.add(prediction)
    session.commit()
    session.refresh(prediction)
    return prediction


def list_predictions(session: Session, start: datetime | None = None,
                     end: datetime | None = None, model_version: str | None = None,
                     risk_label: str | None = None, limit: int = 100) -> list[Prediction]:
    query = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
    if start:
        query = query.where(Prediction.created_at >= start)
    if end:
        query = query.where(Prediction.created_at <= end)
    if model_version:
        query = query.where(Prediction.model_version == model_version)
    if risk_label:
        query = query.where(Prediction.risk_label == risk_label)
    return list(session.scalars(query))
