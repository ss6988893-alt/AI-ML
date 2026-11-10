"""Operational latency, request errors, and unlabeled feature drift."""
import pandas as pd
import plotly.express as px
import streamlit as st
from api_client import get

st.set_page_config(page_title="Monitoring", page_icon="📡", layout="wide")
st.title("Monitoring")
window = st.slider("Recent prediction window for drift", 100, 5000, 1000, 100)
try:
    metrics, drift, model = get("/metrics"), get("/monitoring/drift", {"limit": window}), get("/model/info")
except Exception as exc:
    st.error("Monitoring data is unavailable. Check the API and database."); st.caption(str(exc)); st.stop()
a, b, c, d = st.columns(4)
a.metric("API requests", metrics["request_count"])
b.metric("Errors", metrics["error_count"])
c.metric("Latency p95", f"{metrics['latency_ms_p95']:.1f} ms" if metrics["latency_ms_p95"] is not None else "—")
d.metric("Loaded model", model.get("model_version") or "None")
st.subheader("Feature drift (PSI)")
st.caption(f"Validation errors: {metrics['validation_error_count']} · fraud-score histogram bins: {metrics['fraud_score_histogram']}")
st.caption(f"Reference window is sampled training data; recent window has {drift['window_predictions']} stored predictions.")
rows = [{"feature": name, **item} for name, item in drift["features"].items()]
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    values = [row for row in rows if row["psi"] is not None]
    if values:
        st.plotly_chart(px.bar(pd.DataFrame(values), x="feature", y="psi", color="status", title="Population Stability Index"), use_container_width=True)
st.info(drift["note"])
st.caption("PSI is a warning signal for distribution change, not a measure of fraud accuracy. It does not trigger automatic retraining.")
if st.button("Run drift simulation"):
    try:
        simulated = get("/monitoring/drift/simulation")
        st.warning(simulated["note"])
        st.dataframe(pd.DataFrame([{"feature": key, **value} for key, value in simulated["features"].items()]), hide_index=True)
    except Exception as exc:
        st.error("Simulation is unavailable because no reference model is loaded.")
        st.caption(str(exc))
