"""Validated, identifier-free request and response contracts."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(ge=0)
    type: str = Field(min_length=1, max_length=32)
    amount: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceOrg: float = Field(ge=0, allow_inf_nan=False)
    newbalanceOrig: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceDest: float = Field(ge=0, allow_inf_nan=False)
    newbalanceDest: float = Field(ge=0, allow_inf_nan=False)


class PredictionRequest(BaseModel):
    transaction: Transaction
    include_explanation: bool = False


class BatchPredictionRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)
    include_explanation: bool = False


class PredictionResponse(BaseModel):
    request_id: str
    prediction_id: str
    fraud_probability: float
    risk_category: str
    decision_threshold: float
    model_version: str
    explanation: dict | None = None
    explanation_note: str = "Feature contributions support investigation; they do not prove fraud or establish causality."


class PredictionRecord(BaseModel):
    request_id: str
    prediction_id: str
    created_at: datetime
    model_version: str
    fraud_probability: float
    risk_category: str
    decision_threshold: float
