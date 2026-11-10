"""Manual and CSV transaction scoring through FastAPI."""
import pandas as pd
import streamlit as st
from api_client import post

st.set_page_config(page_title="Transaction Analysis", page_icon="🔎", layout="wide")
st.title("Transaction analysis")
st.caption("Predictions are risk indicators for review; they do not confirm fraud.")
with st.form("transaction"):
    c1, c2 = st.columns(2)
    with c1:
        step = st.number_input("Step", min_value=0, value=1)
        tx_type = st.selectbox("Transaction type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        amount = st.number_input("Amount", min_value=0.0, value=100.0)
        old_org = st.number_input("Sender balance before", min_value=0.0, value=1000.0)
    with c2:
        new_org = st.number_input("Sender balance after", min_value=0.0, value=900.0)
        old_dest = st.number_input("Receiver balance before", min_value=0.0, value=0.0)
        new_dest = st.number_input("Receiver balance after", min_value=0.0, value=100.0)
        explain = st.checkbox("Request an explanation (when supported)")
    submitted = st.form_submit_button("Score transaction", type="primary")
if submitted:
    payload = {"transaction": {"step": int(step), "type": tx_type, "amount": amount,
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest}, "include_explanation": explain}
    try:
        result = post("/predict", payload)
        a, b, c = st.columns(3)
        a.metric("Fraud risk score", f"{result['fraud_probability']:.1%}")
        b.metric("Risk category", result["risk_category"].title())
        c.metric("Decision threshold", f"{result['decision_threshold']:.1%}")
        st.caption(f"Model {result['model_version']} · Request {result['request_id']}")
        if result.get("explanation"):
            if result["explanation"].get("available"):
                st.write("Most influential features for this prediction")
                st.dataframe(pd.DataFrame(result["explanation"]["top_features"]), hide_index=True)
            else:
                st.info(result["explanation"].get("reason", "Explanation unavailable"))
        st.caption(result["explanation_note"])
    except Exception as exc:
        st.error("Prediction could not be completed. Check that the API, model, and database are ready.")
        st.caption(str(exc))
st.divider()
st.subheader("Batch analysis")
file = st.file_uploader("Upload a CSV with the seven transaction fields", type=["csv"])
required = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
if file and st.button("Score uploaded transactions"):
    try:
        frame = pd.read_csv(file)
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            st.error(f"CSV is missing required columns: {', '.join(missing)}")
        else:
            transactions = frame[required].where(pd.notna(frame[required]), None).to_dict("records")
            output = []
            for start in range(0, len(transactions), 100):
                output.extend(post("/predict/batch", {"transactions": transactions[start:start+100]}))
            st.success(f"Scored {len(output):,} transactions")
            st.dataframe(pd.DataFrame([{"risk_score": r["fraud_probability"], "risk_category": r["risk_category"], "model_version": r["model_version"]} for r in output]), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error("CSV validation or batch prediction failed.")
        st.caption(str(exc))
