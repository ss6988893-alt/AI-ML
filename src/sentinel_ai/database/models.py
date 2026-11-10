"""SQLAlchemy persistence models; migrations define the production schema."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, JSON, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


def utc_now():
    return datetime.now(timezone.utc)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    training_dataset_fingerprint: Mapped[str] = mapped_column(String(128), default="unknown")
    model_artifact_uri: Mapped[str] = mapped_column(Text, default="")
    metrics_json: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    configuration_json: Mapped[dict] = mapped_column(JSON_TYPE, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="active")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.version"), index=True)
    fraud_score: Mapped[float] = mapped_column(Float)
    risk_label: Mapped[str] = mapped_column(String(20), index=True)
    decision_threshold: Mapped[float] = mapped_column(Float)
    explanation_json: Mapped[dict | None] = mapped_column(JSON_TYPE, nullable=True)
    safe_feature_summary: Mapped[dict | None] = mapped_column(JSON_TYPE, nullable=True)


class PredictionFeedback(Base):
    __tablename__ = "prediction_feedback"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id: Mapped[str] = mapped_column(ForeignKey("predictions.id", ondelete="CASCADE"), index=True)
    reviewed_label: Mapped[str] = mapped_column(String(30))
    review_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)


Index("ix_predictions_created_model", Prediction.created_at, Prediction.model_version)
