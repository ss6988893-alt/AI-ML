# Architecture (target design)

The codebase implements the flow shown below. Running the full flow requires the PaySim source file, installed dependencies, and a reachable PostgreSQL/MLflow setup.

```mermaid
flowchart LR
  Kaggle[PaySim CSV] --> Validate[Chunked validation and EDA]
  Validate --> Train[Chronological training and evaluation]
  Train --> Registry[Versioned model artifact]
  Train --> MLflow[MLflow tracking]
  Dashboard[Streamlit dashboard] --> API[FastAPI]
  API --> Predict[Shared preprocessing and inference]
  Registry --> Predict
  Predict --> DB[(PostgreSQL predictions and model versions)]
  DB --> Dashboard
  API --> Monitor[Latency, errors, score and feature drift]
```

## Principal prediction workflow

```mermaid
sequenceDiagram
  participant U as Dashboard/client
  participant A as FastAPI
  participant F as Feature pipeline/model
  participant D as PostgreSQL
  U->>A: Validated transaction
  A->>F: Transform with saved preprocessing and score
  F-->>A: Probability, threshold and version
  A->>D: Persist prediction record
  A-->>U: Safe prediction response and request ID
```

Model artifacts are produced by the offline training command and loaded by the API; the API does not train on startup. Prediction scores, operational thresholds, risk categories, and reviewed labels are distinct concepts. This educational system does not connect to payment rails and does not claim calibrated probabilities.

## Prediction timing

The input contract includes PaySim's before/after balance fields. This implementation scores a transaction record **after the transaction's balance updates are available**, for retrospective triage and analysis. It is not a pre-authorization model: `newbalanceOrig` and `newbalanceDest` would be unavailable before the ledger update. A true pre-authorization model would need to remove those fields and re-engineer/retrain using only pre-transaction data.
