"""Stored prediction history filters without account identifiers."""
import pandas as pd
import streamlit as st
from datetime import datetime, time, timedelta
from api_client import get

st.set_page_config(page_title="Prediction History", page_icon="🧾", layout="wide")
st.title("Prediction history")
with st.form("filters"):
    a, b, c, d = st.columns(4)
    risk = a.selectbox("Risk category", ["All", "low", "medium", "high"])
    version = b.text_input("Model version (optional)")
    limit = c.slider("Maximum rows", 10, 500, 100, 10)
    today = datetime.now().date()
    dates = d.date_input("Date range", value=(today - timedelta(days=30), today), max_value=today)
    submit = st.form_submit_button("Apply filters")
if submit or "history_loaded" not in st.session_state:
    params = {"limit": limit}
    if risk != "All": params["risk_category"] = risk
    if version: params["model_version"] = version
    if len(dates) == 2:
        params["start"] = datetime.combine(dates[0], time.min).isoformat()
        params["end"] = datetime.combine(dates[1], time.max).isoformat()
    try:
        st.session_state.history = get("/predictions", params)
        st.session_state.history_loaded = True
    except Exception as exc:
        st.error("Prediction history could not be loaded from the API."); st.caption(str(exc))
rows = st.session_state.get("history", [])
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No prediction records match these filters.")
