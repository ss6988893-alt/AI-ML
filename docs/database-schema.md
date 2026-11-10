# Database schema

PostgreSQL is the primary application store, managed with SQLAlchemy and Alembic. Revision `0001_initial` creates the tables. SQLite can be used in isolated tests.

```mermaid
erDiagram
  MODEL_VERSIONS ||--o{ PREDICTIONS : scores
  PREDICTIONS ||--o{ PREDICTION_FEEDBACK : reviewed_as
  MODEL_VERSIONS {
    uuid id PK
    string version UK
    timestamp created_at
    string training_dataset_fingerprint
    string model_artifact_uri
    json metrics_json
    json configuration_json
    string status
  }
  PREDICTIONS {
    uuid id PK
    uuid request_id UK
    timestamp created_at
    string model_version FK
    float fraud_score
    string risk_label
    float decision_threshold
    json explanation_json
    json safe_feature_summary
  }
  PREDICTION_FEEDBACK {
    uuid id PK
    uuid prediction_id FK
    string reviewed_label
    timestamp review_timestamp
    string reviewer_note
  }
```

Raw training transactions will remain in CSV/Parquet files, not be inserted into PostgreSQL. Feedback is a reviewed label, never an automatic retraining signal.
