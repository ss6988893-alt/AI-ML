"""PostgreSQL integration tests run when TEST_DATABASE_URL is configured."""
import os
import pytest
import uuid
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from sentinel_ai.database.models import ModelVersion, Prediction
from sentinel_ai.database.repositories import list_predictions, save_prediction

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")
def test_postgres_connection_and_jsonb_support():
    engine = create_engine(os.environ["TEST_DATABASE_URL"], pool_pre_ping=True)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT 1")).scalar_one() == 1
        tables = connection.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))
        assert {row[0] for row in tables} >= {"predictions", "model_versions", "prediction_feedback"}
    version = "integration-" + str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    try:
        with Session(engine) as session:
            session.add(ModelVersion(version=version, training_dataset_fingerprint="fixture",
                                     model_artifact_uri="test-only", metrics_json={"partition": "test"},
                                     configuration_json={"threshold": 0.5}, status="test"))
            session.commit()
            row = save_prediction(session, request_id=request_id, model_version=version,
                fraud_score=0.2, risk_label="low", decision_threshold=0.5,
                safe_feature_summary={"type": "TRANSFER", "amount": 12.0})
            retrieved = list_predictions(session, limit=10)
            assert any(item.id == row.id for item in retrieved)
            assert row.safe_feature_summary["amount"] == 12.0
    finally:
        with Session(engine) as session:
            prediction = session.scalar(select(Prediction).where(Prediction.request_id == request_id))
            if prediction:
                session.delete(prediction)
            model = session.scalar(select(ModelVersion).where(ModelVersion.version == version))
            if model:
                session.delete(model)
            session.commit()
    engine.dispose()
