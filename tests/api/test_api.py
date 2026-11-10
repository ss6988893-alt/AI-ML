import joblib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sentinel_ai.api.main import create_app
from sentinel_ai.config import Settings
from sentinel_ai.database.models import Base


class TestPipeline:
    """Deterministic test double; never used for product predictions or metrics."""
    def predict_proba(self, features):
        import numpy as np
        score = (features["amount"].fillna(0).to_numpy() > 500).astype(float) * 0.7 + 0.1
        return np.column_stack([1 - score, score])


@pytest.fixture
def client(tmp_path):
    db_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(db_engine)
    Session = sessionmaker(bind=db_engine, expire_on_commit=False)

    def session_override():
        with Session() as session:
            yield session

    model_path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": TestPipeline(), "metadata": {
        "model_version": "test-model", "model_family": "test-double",
        "selected_threshold": 0.5, "training_timestamp": "test-only",
    }}, model_path)
    settings = Settings(database_url="sqlite://", model_path=str(model_path), max_batch_size=2,
                        risk_medium_threshold=0.25, risk_high_threshold=0.7)
    app = create_app(settings, session_override)
    app.state.test_session_override = session_override
    with TestClient(app) as test_client:
        yield test_client
    db_engine.dispose()


def transaction(amount=100):
    return {"step": 1, "type": "TRANSFER", "amount": amount, "oldbalanceOrg": 1000,
            "newbalanceOrig": 1000-amount, "oldbalanceDest": 0, "newbalanceDest": amount}


def test_health_readiness_and_model_info(client):
    assert client.get("/health").json()["model_loaded"]
    assert client.get("/ready").json()["model_loaded"]
    assert client.get("/model/info").json()["model_version"] == "test-model"


def test_prediction_persists_record_without_identifiers(client):
    response = client.post("/predict", json={"transaction": transaction()})
    assert response.status_code == 200
    result = response.json()
    assert result["fraud_probability"] == pytest.approx(0.1)
    assert result["risk_category"] == "low"
    rows = client.get("/predictions").json()
    assert len(rows) == 1 and "nameOrig" not in rows[0]
    overview = client.get("/overview").json()
    assert overview["total_predictions"] == 1


def test_invalid_transaction_and_extra_identifier_rejected(client):
    bad = transaction(-1)
    assert client.post("/predict", json={"transaction": bad}).status_code == 422
    missing = transaction(); missing.pop("type")
    assert client.post("/predict", json={"transaction": missing}).status_code == 422
    with_id = transaction(); with_id["nameOrig"] = "should-not-be-accepted"
    response = client.post("/predict", json={"transaction": with_id})
    assert response.status_code == 422
    assert "should-not-be-accepted" not in response.text


def test_batch_predictions_and_bounds(client):
    response = client.post("/predict/batch", json={"transactions": [transaction(), transaction(900)]})
    assert response.status_code == 200 and len(response.json()) == 2
    too_many = client.post("/predict/batch", json={"transactions": [transaction()] * 3})
    assert too_many.status_code == 413


def test_high_score_maps_to_high_risk(client):
    response = client.post("/predict", json={"transaction": transaction(900)})
    assert response.status_code == 200
    assert response.json()["fraud_probability"] == pytest.approx(0.8)
    assert response.json()["risk_category"] == "high"


def test_metrics_endpoint(client):
    client.get("/health")
    metrics = client.get("/metrics").json()
    assert metrics["request_count"] >= 1


def test_database_failure_returns_safe_error(client):
    from sqlalchemy.exc import OperationalError

    class BrokenSession:
        def scalar(self, *args, **kwargs):
            raise OperationalError("query", {}, RuntimeError("secret host name"))
        def rollback(self):
            pass
    def broken_dependency():
        yield BrokenSession()
    client.app.dependency_overrides[client.app.state.test_session_override] = broken_dependency
    response = client.post("/predict", json={"transaction": transaction()})
    assert response.status_code == 503
    assert "secret host name" not in response.text


def test_missing_model_returns_unavailable_response(client, tmp_path):
    from sentinel_ai.config import Settings
    from fastapi.testclient import TestClient
    app = create_app(Settings(database_url="sqlite://", model_path=str(tmp_path / "not-found.joblib")),
                     client.app.state.test_session_override)
    with TestClient(app) as missing_client:
        response = missing_client.post("/predict", json={"transaction": transaction()})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "model_unavailable"
