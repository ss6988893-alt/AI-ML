# API examples

FastAPI is served at `http://localhost:8000`; interactive docs are at `/docs`. The request below is illustrative only, not a measured or guaranteed outcome.

```json
{
  "transaction": {
    "step": 12,
    "type": "TRANSFER",
    "amount": 125.0,
    "oldbalanceOrg": 500.0,
    "newbalanceOrig": 375.0,
    "oldbalanceDest": 10.0,
    "newbalanceDest": 135.0
  },
  "include_explanation": false
}
```

With an actually loaded model, the response includes a score, risk category, applied threshold, model version, and request/prediction IDs. Without a saved trained model the API returns HTTP 503; it never generates placeholder scores. Batch requests use `POST /predict/batch` and are bounded by `MAX_BATCH_SIZE`. Only transaction-time fields are accepted; account IDs and target labels are rejected as extra fields.

Other endpoints are `GET /health`, `/ready`, `/metrics`, `/model/info`, `/predictions`, `/overview`, and `/monitoring/drift`. `GET /predictions` accepts optional ISO date range, model version, risk category, and a maximum 500-row limit.

The transaction schema includes old and new balances. The expected scoring point is after the transaction is recorded and those balance updates are known. This API does not represent a pre-authorization decision service.
