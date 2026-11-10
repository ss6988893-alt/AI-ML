"""Partition-labeled model performance and candidate comparison."""
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from api_client import get

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
st.title("Model performance")
try:
    info = get("/model/info")
except Exception as exc:
    st.error("Model metadata is unavailable from the API."); st.caption(str(exc)); st.stop()
if not info.get("loaded"):
    st.info("No trained model metadata is available. Run the documented training command first."); st.stop()
st.caption(f"Selected: {info.get('model_family')} · Version {info.get('model_version')} · Trained {info.get('training_timestamp')}")
split = info.get("split") or {}
if split:
    st.caption(f"Chronological split by step: train through {split.get('train_step_max')}; "
               f"validation {split.get('validation_step_min')}–{split.get('validation_step_max')}; "
               f"test from {split.get('test_step_min')}. Rows: {split.get('rows')}")
for partition in ("validation", "test"):
    report = info.get("evaluation", {}).get(partition)
    if not report:
        continue
    st.subheader(f"Selected model — {partition.title()} partition")
    cols = st.columns(5)
    for col, key, label in zip(cols, ("precision", "recall", "f1", "average_precision", "roc_auc"), ("Precision", "Recall", "F1", "PR-AUC", "ROC-AUC")):
        value = report.get(key)
        col.metric(label, f"{value:.3f}" if value is not None else "—")
    cm = report["confusion_matrix"]
    st.caption(f"Threshold {report.get('threshold'):.4f}; FP {cm['fp']}; FN {cm['fn']}; false positives per 1,000: {report['false_positives_per_1000']:.2f}")
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = ff.create_annotated_heatmap(matrix, x=["Predicted negative", "Predicted positive"], y=["Actual negative", "Actual positive"], colorscale="Blues", showscale=True)
    st.plotly_chart(fig, use_container_width=True)
candidates = info.get("evaluation", {}).get("candidate_validation", {})
if candidates:
    st.subheader("Candidate comparison — validation partition")
    table = pd.DataFrame(candidates).T.reset_index(names="model")
    fields = [c for c in ["model", "precision", "recall", "f1", "average_precision", "roc_auc"] if c in table]
    st.dataframe(table[fields], use_container_width=True, hide_index=True)
    if "average_precision" in table:
        st.plotly_chart(px.bar(table, x="model", y="average_precision", title="Average precision by validation candidate"), use_container_width=True)
importance = info.get("evaluation", {}).get("global_feature_importance", [])
if importance:
    st.subheader("Global feature importance")
    st.plotly_chart(px.bar(pd.DataFrame(importance).sort_values("importance"), x="importance", y="feature", orientation="h", title="Model feature importance"), use_container_width=True)
curves = info.get("evaluation", {}).get("precision_recall_curves", {})
if curves:
    st.subheader("Precision–recall curve")
    partitions = [k for k in ("validation", "test") if k in curves]
    chosen = st.radio("Partition", partitions, horizontal=True)
    curve = curves[chosen]
    st.plotly_chart(px.line(x=curve["recall"], y=curve["precision"], labels={"x": "Recall", "y": "Precision"}, title=f"Precision–recall ({chosen})"), use_container_width=True)
st.info("Validation data selected the model and threshold. Test metrics are held-out final evaluation; probability calibration is not implied.")
