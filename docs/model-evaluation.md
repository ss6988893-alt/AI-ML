# Model evaluation

No model metrics are published until a reproducible run uses the real dataset. The training command writes measured results to `reports/model/evaluation.json`.

The split is chronological by PaySim `step`: earlier steps for training, the next interval for validation and threshold selection, and the latest interval held out for a single final test evaluation. Boundaries and data fingerprint are recorded. No test-set metric informs model selection.

The report includes precision, recall, F1, average precision/PR-AUC, ROC-AUC, confusion matrix, FP/FN counts, false positives per 1,000, precision-recall curve, batch average inference time, and sampled single-record p50/p95 inference latency, with the partition named for every result. Accuracy is not the primary selection metric. Threshold selection uses validation data and the configured precision constraint; if unsupported, F1 is the fallback and constraint failure is recorded. Stated recall/precision/latency goals are targets only. Calibration is not assumed.

The dummy baseline, unweighted Logistic Regression, class-weighted Logistic Regression, and class-weighted XGBoost are compared on validation. No resampling is applied. Test data is only scored once after selection and thresholding. Fixture behavior is not model evaluation.
