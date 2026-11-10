"""Environment-backed application settings."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    data_path: str = "data/raw/PS_20174392719_1491204439457_log.csv"
    sample_fraction: float = Field(default=0.1, gt=0, le=1)
    random_seed: int = 42
    min_precision: float = Field(default=0.5, ge=0, le=1)
    database_url: str = "sqlite:///./sentinel.db"
    model_path: str = "artifacts/model.joblib"
    model_metadata_path: str = "artifacts/model_metadata.json"
    max_batch_size: int = Field(default=100, ge=1, le=1000)
    risk_medium_threshold: float = Field(default=0.25, ge=0, le=1)
    risk_high_threshold: float = Field(default=0.70, ge=0, le=1)
    mlflow_tracking_uri: str = "file:./mlruns"


@lru_cache
def get_settings() -> Settings:
    return Settings()
