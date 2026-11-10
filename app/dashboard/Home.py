"""Sentinel AI portfolio dashboard overview."""
import streamlit as st
import pandas as pd
import plotly.express as px
from api_client import get

st.set_page_config(page_title="Sentinel AI | Risk Intelligence", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.stApp {background: #0b1220; color: #e7edf7}
[data-testid="stMetric"] {background:#131e31;padding:16px;border:1px solid #24344d;border-radius:12px}
section[data-testid="stSidebar"] {background:#101a2b}
</style>
""", unsafe_allow_html=True)
st.title("🛡️ Sentinel AI")
st.caption("Synthetic transaction risk intelligence · Educational demonstration")
try:
    metrics, model, overview = get("/metrics"), get("/model/info"), get("/overview")
except Exception as exc:
    st.error("The API is unavailable. Start the API and database, then refresh this page.")
    st.caption(str(exc)); st.stop()
if not model.get("loaded"):
    st.info("No trained model is loaded yet. Train a model on PaySim, then restart the API.")
records = pd.DataFrame(overview["recent_activity"])
total = overview["total_predictions"]
cats = overview["risk_category_counts"]
high = overview["high_risk_predictions"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions processed", f"{total:,}")
c2.metric("High risk predictions", f"{high:,}")
c3.metric("High risk share", f"{(high / total * 100):.1f}%" if total else "—")
c4.metric("API error rate", f"{metrics['error_rate']:.1%}" if metrics.get("error_rate") is not None else "—")
st.caption("These are application prediction counts, not confirmed fraud cases.")
left, right = st.columns(2)
with left:
    st.subheader("Risk category distribution")
    if total:
        chart = pd.DataFrame({"Risk category": list(cats), "Predictions": list(cats.values())})
        st.plotly_chart(px.pie(chart, names="Risk category", values="Predictions", hole=0.55), use_container_width=True)
    else:
        st.info("No predictions have been processed yet.")
with right:
    st.subheader("Recent prediction activity")
    if not records.empty:
        records["created_at"] = pd.to_datetime(records["created_at"])
        trend = records.set_index("created_at").resample("h").size().rename("Predictions").reset_index()
        st.plotly_chart(px.line(trend, x="created_at", y="Predictions"), use_container_width=True)
    else:
        st.info("Prediction history is empty.")
st.subheader("Recent activity")
if records.empty:
    st.info("No stored predictions yet.")
else:
    st.dataframe(records[["created_at", "model_version", "fraud_probability", "risk_category"]].head(10), use_container_width=True, hide_index=True)
