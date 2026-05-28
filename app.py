# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Dropout Risk Predictor", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>🎓 Student Dropout Risk Predictor</h1>
<p style='text-align: center;'>Advanced Analytics Dashboard</p>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

# Risk mapping
def map_risk(prob):
    if prob < 0.33:
        return "Low Risk"
    elif prob < 0.66:
        return "Medium Risk"
    else:
        return "High Risk"

# Color mapping
def color_risk(val):
    if val == "High Risk":
        return 'background-color: #ff4d4d; color: white'
    elif val == "Medium Risk":
        return 'background-color: #ffd633'
    elif val == "Low Risk":
        return 'background-color: #66cc66'
    return ''

# Sidebar
st.sidebar.header("🔍 Filters")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    input_df = df.copy()

    for col in input_df.columns:
        if col in encoders:
            le = encoders[col]
            input_df[col] = input_df[col].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    input_df = input_df[columns]

    probs = model.predict_proba(input_df)[:, 1]
    risks = [map_risk(p) for p in probs]

    df["Dropout_Probability"] = probs.round(2)
    df["Predicted_Risk"] = risks

    # =============================
    # Filters
    # =============================
    risk_filter = st.sidebar.multiselect(
        "Filter by Risk",
        options=df["Predicted_Risk"].unique(),
        default=df["Predicted_Risk"].unique()
    )

    search = st.sidebar.text_input("Search by Student ID")

    filtered_df = df[df["Predicted_Risk"].isin(risk_filter)]

    if search:
        filtered_df = filtered_df[filtered_df.astype(str).apply(lambda row: row.str.contains(search).any(), axis=1)]

    # =============================
    # Metrics
    # =============================
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total", len(filtered_df))
    col2.metric("High Risk", sum(filtered_df["Predicted_Risk"] == "High Risk"))
    col3.metric("Low Risk", sum(filtered_df["Predicted_Risk"] == "Low Risk"))

    # =============================
    # Styled Table
    # =============================
    st.markdown("### 🎯 Prediction Results")
    st.dataframe(filtered_df.style.applymap(color_risk, subset=["Predicted_Risk"]), use_container_width=True)

    # =============================
    # Charts
    # =============================
    st.markdown("### 📈 Risk Distribution")
    st.bar_chart(filtered_df["Predicted_Risk"].value_counts())

    st.markdown("### 🥧 Risk Pie Chart")
    st.plotly_chart({
        "data": [{
            "labels": filtered_df["Predicted_Risk"].value_counts().index.tolist(),
            "values": filtered_df["Predicted_Risk"].value_counts().values.tolist(),
            "type": "pie"
        }]
    })

    # =============================
    # Download
    # =============================
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data", csv, "filtered_predictions.csv")

else:
    st.info("Upload a CSV file to begin")

st.markdown("---")
st.markdown("<p style='text-align: center;'>🚀 Advanced Dashboard Ready</p>", unsafe_allow_html=True)
