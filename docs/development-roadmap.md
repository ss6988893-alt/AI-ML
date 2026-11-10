# Eight-week development roadmap

The implementation code for these milestones is present. Environment and data verification remains open as recorded in `implementation-status.md`.

1. **Dataset and EDA:** acquisition instructions, schema validation, data-quality report, sampling, and exploratory analysis. Code is implemented; real data run awaits dataset.
2. **Features and leakage:** shared training/inference transforms; chronological split; feature tests.
3. **Baselines:** dummy and logistic regression; partition-specific metrics.
4. **Candidate/evaluation:** XGBoost; validation-only threshold selection; one held-out test evaluation.
5. **Explainability/tracking:** SHAP with bounded samples; MLflow; versioned artifacts and metadata.
6. **API/database:** FastAPI; PostgreSQL models/repository; Alembic; persistence and API tests.
7. **Dashboard/containers:** Streamlit API client and Docker Compose stack.
8. **Monitoring/CI/polish:** drift and latency reporting; GitHub Actions; verified docs and genuine screenshots/results. Verification is still outstanding.
