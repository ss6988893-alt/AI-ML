"""Reproducibly train and compare PaySim baselines and XGBoost candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sentinel_ai.features.engineering import NUMERIC_FEATURES, build_preprocessor, engineer_features
from sentinel_ai.models.evaluation import curve_points, evaluate, select_threshold
from sentinel_ai.data.validation import require_valid, validate_csv
from sentinel_ai.config import get_settings


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path, fraction: float, seed: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}. Follow docs/data-acquisition.md")
    chunks = []
    for index, chunk in enumerate(pd.read_csv(path, chunksize=250_000)):
        chunks.append(chunk if fraction == 1 else chunk.sample(frac=fraction, random_state=seed + index))
    frame = pd.concat(chunks, ignore_index=True)
    if not {"step", "isFraud"}.issubset(frame.columns):
        raise ValueError("Training data must include step and isFraud")
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["isFraud"] = pd.to_numeric(frame["isFraud"], errors="coerce")
    selected_rows = len(frame)
    frame = frame.dropna(subset=["step", "isFraud"])
    frame = frame[frame["isFraud"].isin([0, 1])].sort_values("step").reset_index(drop=True)
    frame.attrs["dropped_invalid_step_or_target_rows"] = selected_rows - len(frame)
    if frame["isFraud"].nunique() < 2:
        raise ValueError("Training sample must contain both target classes")
    return frame


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    steps = np.sort(frame["step"].unique())
    if len(steps) < 3:
        raise ValueError("At least three distinct transaction steps are required")
    train_end = steps[max(0, int(len(steps) * 0.70) - 1)]
    validation_end = steps[max(1, int(len(steps) * 0.85) - 1)]
    train = frame[frame.step <= train_end]
    validation = frame[(frame.step > train_end) & (frame.step <= validation_end)]
    test = frame[frame.step > validation_end]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition")
    boundaries = {
        "train_step_max": float(train_end),
        "validation_step_min": float(validation.step.min()),
        "validation_step_max": float(validation_end),
        "test_step_min": float(test.step.min()),
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    return train, validation, test, boundaries


def make_preprocessor():
    return build_preprocessor()


def build_candidates(y_train):
    from xgboost import XGBClassifier

    candidates = {
        "dummy_prior": make_pipeline(make_preprocessor(), DummyClassifier(strategy="prior")),
        "logistic_default": make_pipeline(make_preprocessor(), LogisticRegression(max_iter=1000, random_state=42)),
        "logistic_balanced": make_pipeline(make_preprocessor(), LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42)),
    }
    positives = max(int((y_train == 1).sum()), 1)
    ratio = max(float((y_train == 0).sum()) / positives, 1.0)
    candidates["xgboost_weighted"] = make_pipeline(make_preprocessor(), XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=ratio, eval_metric="logloss",
        random_state=42, n_jobs=4,
    ))
    return candidates


def probabilities(pipeline, features):
    return pipeline.predict_proba(features)[:, 1]


def global_shap_importance(pipeline, training_features: pd.DataFrame,
                           seed: int, sample_size: int = 300) -> list[dict]:
    """Bounded global SHAP summary. Empty means this model/explainer was unsupported."""
    try:
        import shap
        sample = training_features.sample(n=min(sample_size, len(training_features)),
                                          random_state=seed)
        preprocessor, estimator = pipeline[:-1], pipeline[-1]
        transformed = preprocessor.transform(sample)
        names = list(preprocessor.get_feature_names_out())
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(transformed)
        elif hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, transformed)
            values = explainer.shap_values(transformed)
        else:
            return []
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        importance = np.mean(np.abs(values), axis=0)
        order = np.argsort(importance)[::-1][:30]
        return [{"feature": names[i].split("__", 1)[-1], "importance": float(importance[i])}
                for i in order]
    except Exception:
        return []


def run(args) -> dict:
    source = Path(args.input)
    quality_report = validate_csv(source)
    require_valid(quality_report)
    dataset_fingerprint = fingerprint(source)
    frame = load_data(source, args.sample_fraction, args.seed)
    train, validation, test, split = chronological_split(frame)
    for partition_name, partition in (("train", train), ("validation", validation), ("test", test)):
        if partition["isFraud"].nunique() < 2:
            raise ValueError(
                f"The {partition_name} partition lacks one target class; increase the sample fraction or inspect the chronological boundaries."
            )
    X_train = engineer_features(train)
    X_validation = engineer_features(validation)
    X_test = engineer_features(test)
    y_train, y_validation, y_test = train.isFraud.astype(int), validation.isFraud.astype(int), test.isFraud.astype(int)
    output = Path(args.output_dir)
    artifact_path = Path(args.model_path)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    tracking = None
    if not args.disable_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("sentinel-ai-paysim")
            tracking = mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow is required; use --disable-mlflow only for an explicit local run") from exc
    results = {}
    pipelines = {}
    candidates = build_candidates(y_train)
    for name, pipeline in candidates.items():
        if tracking:
            run_context = tracking.start_run(run_name=name)
        else:
            from contextlib import nullcontext
            run_context = nullcontext()
        with run_context:
            pipeline.fit(X_train, y_train)
            inference_started = time.perf_counter()
            score = probabilities(pipeline, X_validation)
            inference_elapsed = time.perf_counter() - inference_started
            metrics = evaluate(y_validation, score, 0.5)
            metrics["inference_latency_ms_per_record"] = float(inference_elapsed * 1000 / max(len(y_validation), 1))
            metrics["partition"] = "validation"
            results[name] = metrics
            pipelines[name] = pipeline
            if tracking:
                run_params = {"model_family": name, "seed": args.seed,
                              "sample_fraction": args.sample_fraction,
                              "split_method": "chronological step",
                              "feature_config": "paysim-v1",
                              "feature_columns": ",".join(["type", *NUMERIC_FEATURES]),
                              "dataset_fingerprint": dataset_fingerprint,
                              "train_rows": len(train), "validation_rows": len(validation),
                              "test_rows": len(test)}
                run_params.update({f"estimator_{key}": str(value)[:400]
                                   for key, value in pipeline[-1].get_params().items()
                                   if isinstance(value, (str, int, float, bool, type(None)))})
                tracking.log_params(run_params)
                tracking.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    selected = max(results, key=lambda name: (results[name]["average_precision"], results[name]["recall"]))
    selected_pipeline = pipelines[selected]
    validation_scores = probabilities(selected_pipeline, X_validation)
    threshold = select_threshold(y_validation, validation_scores, args.min_precision)
    validation_metrics = evaluate(y_validation, validation_scores, threshold)
    validation_metrics["partition"] = "validation"
    test_started = time.perf_counter()
    test_scores = probabilities(selected_pipeline, X_test)
    latency_ms = (time.perf_counter() - test_started) * 1000 / max(len(y_test), 1)
    for _ in range(5):
        selected_pipeline.predict_proba(X_test.iloc[:1])
    single_latencies = []
    for row_index in range(min(len(X_test), 100)):
        started = time.perf_counter()
        selected_pipeline.predict_proba(X_test.iloc[row_index:row_index + 1])
        single_latencies.append((time.perf_counter() - started) * 1000)
    test_metrics = evaluate(y_test, test_scores, threshold)
    test_metrics["partition"] = "test"
    test_metrics["inference_latency_ms_per_record"] = latency_ms
    test_metrics["single_record_latency_ms_p50"] = float(np.percentile(single_latencies, 50))
    test_metrics["single_record_latency_ms_p95"] = float(np.percentile(single_latencies, 95))
    timestamp = datetime.now(timezone.utc).isoformat()
    model_version = "sentinel-" + timestamp.replace(":", "").replace("-", "")[:15]
    versioned_path = artifact_path.parent / f"{model_version}.joblib"
    metadata = {
        "model_version": model_version,
        "model_family": selected, "training_timestamp": timestamp,
        "benchmark_environment": {"python": sys.version.split()[0],
                                  "platform": platform.platform(),
                                  "processor": platform.processor() or "not reported by OS",
                                  "logical_cpu_count": os.cpu_count()},
        "dataset_fingerprint_sha256": dataset_fingerprint, "dataset_path_label": source.name,
        "sample_fraction": args.sample_fraction, "random_seed": args.seed,
        "dropped_rows_missing_step_or_target": frame.attrs.get("dropped_invalid_step_or_target_rows", 0),
        "split_method": "chronological by step", "split": split,
        "selected_threshold": threshold, "minimum_precision_constraint": args.min_precision,
        "threshold_precision_constraint_met": validation_metrics["precision"] >= args.min_precision,
        "candidate_configurations": {name: {key: str(value) for key, value in pipeline[-1].get_params().items()
                                              if isinstance(value, (str, int, float, bool, type(None)))}
                                     for name, pipeline in candidates.items()},
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "candidate_validation_metrics": results, "features": ["type", *NUMERIC_FEATURES],
        "global_feature_importance": global_shap_importance(selected_pipeline, X_train, args.seed),
        "global_feature_importance_method": "mean_absolute_shap_value",
        "target": "isFraud", "excluded_inputs": ["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"],
        "reference_samples": {
            "amount": train["amount"].sample(n=min(len(train), 5000), random_state=args.seed).tolist(),
            "step": train["step"].sample(n=min(len(train), 5000), random_state=args.seed).tolist(),
        },
        "precision_recall_curves": {
            "validation": curve_points(y_validation, validation_scores),
            "test": curve_points(y_test, test_scores),
        },
    }
    import joblib
    bundle = {"pipeline": selected_pipeline, "metadata": metadata}
    joblib.dump(bundle, versioned_path)
    shutil.copyfile(versioned_path, artifact_path)
    (output / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (Path(args.metadata_path)).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "precision_recall_curves.json").write_text(
        json.dumps(metadata["precision_recall_curves"]), encoding="utf-8")
    if tracking:
        with tracking.start_run(run_name="selected-final"):
            tracking.log_params({"selected_model": selected, "threshold": threshold,
                                 "model_version": model_version, "training_timestamp": timestamp,
                                 "dataset_fingerprint": dataset_fingerprint,
                                 "split_configuration": "chronological by step",
                                 "feature_config": "paysim-v1",
                                 "feature_columns": ",".join(["type", *NUMERIC_FEATURES])})
            tracking.log_metrics({f"test_{k}": v for k, v in test_metrics.items()
                                  if isinstance(v, (int, float))})
            tracking.log_artifact(str(versioned_path))
    print(json.dumps({"selected": selected, "threshold": threshold,
                      "validation": validation_metrics, "test": test_metrics}, indent=2))
    return metadata


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=settings.data_path)
    parser.add_argument("--sample-fraction", type=float, default=settings.sample_fraction)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--min-precision", type=float, default=settings.min_precision)
    parser.add_argument("--output-dir", default="reports/model")
    parser.add_argument("--model-path", default=settings.model_path)
    parser.add_argument("--metadata-path", default=settings.model_metadata_path)
    parser.add_argument("--mlflow-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--disable-mlflow", action="store_true",
                        help="Explicit development-only bypass for MLflow tracking")
    args = parser.parse_args()
    if not 0 < args.sample_fraction <= 1 or not 0 <= args.min_precision <= 1:
        parser.error("sample fraction must be (0,1], precision must be [0,1]")
    run(args)


if __name__ == "__main__":
    main()
