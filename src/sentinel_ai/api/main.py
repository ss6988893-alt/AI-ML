"""FastAPI prediction service. The API loads a saved model and never trains."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from sentinel_ai.config import Settings, get_settings
from sentinel_ai.database import repositories
from sentinel_ai.database.session import get_session, make_engine
from sentinel_ai.features.engineering import engineer_features
from sentinel_ai.models.registry import load_model
from sentinel_ai.monitoring.metrics import RequestMetrics
from sentinel_ai.monitoring.metrics import population_stability_index
from sentinel_ai.api.schemas import (BatchPredictionRequest, PredictionRecord,
                                     PredictionRequest, PredictionResponse, Transaction)


def create_app(settings: Settings | None = None, database_dependency=get_session) -> FastAPI:
    config = settings or get_settings()
    metrics = RequestMetrics()
    health_engine = make_engine(config.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model_bundle = load_model(config.model_path)
            app.state.model_error = None
        except Exception:
            app.state.model_bundle = None
            app.state.model_error = "Model artifact is unavailable or invalid"
        try:
            yield
        finally:
            health_engine.dispose()

    app = FastAPI(title="Sentinel AI", version="0.1.0",
                  description="Educational PaySim fraud-risk scoring API.", lifespan=lifespan)
    app.state.model_bundle = None
    app.state.model_error = "Model artifact has not been loaded"

    @app.exception_handler(RequestValidationError)
    async def sanitized_validation_error(request: Request, exc: RequestValidationError):
        # Pydantic errors can contain submitted values; omit input/context to avoid echoing IDs.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(error.get("loc", ())), "msg": error.get("msg", "Invalid input"),
             "type": error.get("type", "value_error")}
            for error in exc.errors()
        ]})

    @app.middleware("http")
    async def collect_metrics(request: Request, call_next):
        started = time.perf_counter()
        is_error = False
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            is_error = response.status_code >= 400
            return response
        except Exception:
            is_error = True
            raise
        finally:
            metrics.record((time.perf_counter() - started) * 1000, is_error, status_code == 422)

    def get_bundle(request: Request):
        if request.app.state.model_bundle is None:
            raise HTTPException(status_code=503, detail={"code": "model_unavailable",
                                                        "message": "No valid trained model is loaded"})
        return request.app.state.model_bundle

    def category(score: float) -> str:
        if score >= config.risk_high_threshold:
            return "high"
        if score >= config.risk_medium_threshold:
            return "medium"
        return "low"

    def explain(bundle: dict, feature_frame: pd.DataFrame) -> dict:
        try:
            import shap
            pipeline = bundle["pipeline"]
            estimator = pipeline[-1]
            transformed = pipeline[:-1].transform(feature_frame)
            feature_names = list(pipeline[:-1].get_feature_names_out())
            if hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
                values = explainer.shap_values(transformed)
                base = explainer.expected_value
            elif hasattr(estimator, "coef_"):
                explainer = shap.LinearExplainer(estimator, transformed)
                values = explainer.shap_values(transformed)
                base = explainer.expected_value
            else:
                return {"available": False, "reason": "This model family has no supported SHAP explainer"}
            if isinstance(values, list):
                values = values[-1]
            if getattr(values, "ndim", 0) == 3:
                values = values[:, :, -1]
            row_values = values[0]
            contributions = sorted(
                [{"feature": name.split("__", 1)[-1], "contribution": float(value)}
                 for name, value in zip(feature_names, row_values)],
                key=lambda item: abs(item["contribution"]), reverse=True,
            )[:8]
            return {"available": True, "base_value": float(base[-1] if hasattr(base, "__len__") else base),
                    "top_features": contributions}
        except Exception:
            return {"available": False, "reason": "Explanation could not be generated for this prediction"}

    def predict_one(transaction: Transaction, include_explanation: bool, bundle: dict,
                    session: Session) -> PredictionResponse:
        raw = transaction.model_dump()
        features = engineer_features(pd.DataFrame([raw]))
        pipeline = bundle["pipeline"]
        score = float(pipeline.predict_proba(features)[0, 1])
        metadata = bundle["metadata"]
        threshold = float(metadata.get("selected_threshold", 0.5))
        risk = category(score)
        explanation = explain(bundle, features) if include_explanation else None
        request_id, prediction_id = str(uuid.uuid4()), str(uuid.uuid4())
        try:
            repositories.register_model(session, metadata, config.model_path)
            repositories.save_prediction(
                session, id=prediction_id, request_id=request_id,
                model_version=metadata.get("model_version", "unknown"),
                fraud_score=score, risk_label=risk, decision_threshold=threshold,
                explanation_json=explanation,
                safe_feature_summary={"type": raw["type"], "amount": raw["amount"], "step": raw["step"]},
            )
        except SQLAlchemyError as exc:
            session.rollback()
            raise HTTPException(status_code=503, detail={"code": "persistence_unavailable",
                                                        "message": "Prediction storage is unavailable"}) from exc
        metrics.record_prediction(score, risk)
        return PredictionResponse(
            request_id=request_id, prediction_id=prediction_id, fraud_probability=score,
            risk_category=risk, decision_threshold=threshold,
            model_version=metadata.get("model_version", "unknown"), explanation=explanation,
        )

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_loaded": request.app.state.model_bundle is not None}

    @app.get("/ready")
    def ready(request: Request):
        database_ok = False
        try:
            with health_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            database_ok = True
        except SQLAlchemyError:
            pass
        model_ok = request.app.state.model_bundle is not None
        return {"ready": model_ok and database_ok, "model_loaded": model_ok,
                "database_reachable": database_ok}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request,
                bundle=Depends(get_bundle), session: Session = Depends(database_dependency)):
        return predict_one(body.transaction, body.include_explanation, bundle, session)

    @app.post("/predict/batch", response_model=list[PredictionResponse])
    def predict_batch(body: BatchPredictionRequest, request: Request,
                      bundle=Depends(get_bundle), session: Session = Depends(database_dependency)):
        if len(body.transactions) > config.max_batch_size:
            raise HTTPException(status_code=413, detail={"code": "batch_too_large",
                "message": f"Batch limit is {config.max_batch_size} transactions"})
        return [predict_one(item, body.include_explanation, bundle, session)
                for item in body.transactions]

    @app.get("/metrics")
    def get_metrics():
        return metrics.snapshot()

    @app.get("/model/info")
    def model_info(request: Request):
        bundle = request.app.state.model_bundle
        if bundle is None:
            return {"loaded": False, "metadata": None}
        metadata = bundle["metadata"]
        return {"loaded": True, "model_version": metadata.get("model_version"),
                "model_family": metadata.get("model_family"),
                "training_timestamp": metadata.get("training_timestamp"),
                "evaluation": {"validation": metadata.get("validation_metrics"),
                               "test": metadata.get("test_metrics"),
                               "candidate_validation": metadata.get("candidate_validation_metrics"),
                               "precision_recall_curves": metadata.get("precision_recall_curves"),
                               "global_feature_importance": metadata.get("global_feature_importance"),
                               "candidate_configurations": metadata.get("candidate_configurations")},
                "threshold": metadata.get("selected_threshold"),
                "split": metadata.get("split"),
                "threshold_precision_constraint_met": metadata.get("threshold_precision_constraint_met")}

    @app.get("/monitoring/drift")
    def drift(limit: int = Query(default=1000, ge=1, le=5000),
              session: Session = Depends(database_dependency), bundle=Depends(get_bundle)):
        from sentinel_ai.database.models import Prediction
        recent = list(session.scalars(select(Prediction).order_by(
            Prediction.created_at.desc()).limit(limit)))
        metadata = bundle["metadata"]
        reference = metadata.get("reference_samples", {})
        output = {}
        for feature in ("amount", "step"):
            values = [row.safe_feature_summary.get(feature) for row in recent
                      if row.safe_feature_summary and row.safe_feature_summary.get(feature) is not None]
            psi = population_stability_index(reference.get(feature, []), values)
            output[feature] = {"psi": psi, "status": (
                "insufficient_data" if psi is None else "warning" if psi >= 0.2 else "no_warning"
            ), "reference_count": len(reference.get(feature, [])), "recent_count": len(values)}
        return {"window_predictions": len(recent), "label_metrics_available": False,
                "note": "Unlabeled prediction drift is not a measure of fraud accuracy.",
                "features": output}

    @app.get("/monitoring/drift/simulation")
    def drift_simulation(bundle=Depends(get_bundle)):
        reference = bundle["metadata"].get("reference_samples", {})
        output = {}
        for feature in ("amount", "step"):
            values = np.asarray(reference.get(feature, []), dtype=float)
            if not len(values):
                psi, count = None, 0
            else:
                simulated = values * 1.5 + max(float(np.mean(values)) * 0.5, 1.0)
                psi = population_stability_index(values, simulated)
                count = len(simulated)
            output[feature] = {"psi": psi, "status": "simulation",
                               "reference_count": len(values), "simulated_count": count}
        return {"simulation_only": True, "persisted": False, "features": output,
                "note": "Synthetic transformed values demonstrate PSI behavior; these are not observed transactions or accuracy results."}

    @app.get("/overview")
    def overview(session: Session = Depends(database_dependency)):
        from sentinel_ai.database.models import Prediction
        total = session.scalar(select(func.count(Prediction.id))) or 0
        grouped = session.execute(select(Prediction.risk_label, func.count(Prediction.id))
                                  .group_by(Prediction.risk_label)).all()
        categories = {name: count for name, count in grouped}
        high = categories.get("high", 0)
        recent = repositories.list_predictions(session, limit=500)
        return {"total_predictions": total, "high_risk_predictions": high,
                "high_risk_proportion": high / total if total else None,
                "risk_category_counts": categories,
                "recent_activity": [{"created_at": row.created_at.isoformat(),
                    "risk_category": row.risk_label, "fraud_probability": row.fraud_score,
                    "model_version": row.model_version} for row in reversed(recent)]}

    @app.get("/predictions", response_model=list[PredictionRecord])
    def prediction_history(start: datetime | None = None, end: datetime | None = None,
                           model_version: str | None = None,
                           risk_category: str | None = Query(default=None, pattern="^(low|medium|high)$"),
                           limit: int = Query(default=100, ge=1, le=500),
                           session: Session = Depends(database_dependency)):
        rows = repositories.list_predictions(session, start, end, model_version, risk_category, limit)
        return [PredictionRecord(request_id=row.request_id, prediction_id=row.id,
                                 created_at=row.created_at, model_version=row.model_version,
                                 fraud_probability=row.fraud_score, risk_category=row.risk_label,
                                 decision_threshold=row.decision_threshold) for row in rows]

    return app


app = create_app()
