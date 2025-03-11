import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import time
import pandas as pd
import os

# Load Model & Scaler
model_path = os.path.join(os.getcwd(), "model", "health_risk_model.pkl")
scaler_path = os.path.join(os.getcwd(), "model", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Function to Calculate BMI
def calculate_bmi(weight, height, weight_unit, height_unit):
    if weight_unit == "g":
        weight /= 1000  # Convert grams to kg
    if height_unit == "cm":
        height /= 100  # Convert cm to meters
    bmi = weight / (height ** 2)
    return round(bmi, 2)

# Streamlit UI Configuration
st.set_page_config(page_title="Health Risk Prediction", layout="centered")
st.title("🩺 AI Health Risk Prediction")
st.write("Enter individual details OR upload a file for bulk health risk assessment.")

# **🔹 Individual Input Form**
st.header("📌 Predict for a Single Individual")

with st.form("health_form"):
    age = st.number_input("🔢 Age", min_value=1, max_value=120, value=30)
    weight_unit = st.radio("⚖️ Weight Input Type", ["kg", "g"])
    weight = st.number_input("⚖️ Enter your Weight", min_value=1.0, max_value=500.0, value=70.0, step=0.1)
    height_unit = st.radio("📏 Height Input Type", ["m", "cm"])
    height = st.number_input("📏 Enter your Height", min_value=30.0, max_value=250.0, value=170.0, step=0.1)

    if st.form_submit_button("📊 Calculate BMI"):
        st.session_state.bmi = calculate_bmi(weight, height, weight_unit, height_unit)
        st.success(f"📊 Your BMI is: {st.session_state.bmi}")

# **🔹 Prediction Section**
if "bmi" in st.session_state:
    st.write(f"📊 Your BMI: **{st.session_state.bmi}**")
    steps_per_day = st.slider("🚶 Steps per Day", min_value=0, max_value=30000, value=5000, step=500)
    sleep_hours = st.slider("💤 Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    air_quality = st.number_input("🌍 Air Quality Index (AQI)", min_value=0, max_value=500, value=100)
    family_history = st.radio("🏥 Family History of Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("🔮 Predict Risk"):
        with st.spinner("Analyzing health data... ⏳"):
            time.sleep(2)
        
        # Ensure feature names match the trained model
        features = pd.DataFrame([[age, st.session_state.bmi, steps_per_day, sleep_hours, air_quality, family_history]], 
                                columns=["age", "bmi", "steps_per_day", "sleep_hours", "air_quality", "family_history"])
        features_scaled = scaler.transform(features)

        # Predict Risk
        risk = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        risk_level = "🔴 High" if risk == 1 else "🟢 Low"

        with st.expander("📊 Prediction Results", expanded=True):
            st.markdown(f"### **Risk Level:** {risk_level}")
            st.markdown(f"**Confidence:** {probability * 100:.2f}%")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Risk Confidence (%)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "red" if risk == 1 else "green"}}
            ))
            st.plotly_chart(fig)

        # Health Tips Based on Risk Factors
        risk_reasons = []
        if st.session_state.bmi < 18.5 or st.session_state.bmi > 25:
            risk_reasons.append("Unhealthy BMI (Underweight/Overweight).")
        if sleep_hours < 6:
            risk_reasons.append("Inadequate Sleep (Less than 6 hours).")
        if steps_per_day < 5000:
            risk_reasons.append("Low Physical Activity.")
        if air_quality > 150:
            risk_reasons.append("Poor Air Quality.")
        if family_history == 1:
            risk_reasons.append("Family History of Diseases.")

        if risk == 1:
            st.warning("⚠️ Your risk is high! Follow a healthy lifestyle.")
            if risk_reasons:
                st.error("**Possible reasons for high risk:**\n- " + "\n- ".join(risk_reasons))
        else:
            st.balloons()
            st.success("🎉 Your risk is low! Keep up the good habits.")  

# **🔹 Bulk Prediction from CSV File**
# **🔹 Option 2: Bulk Prediction from a CSV File**
# **🔹 Bulk Prediction from CSV File**
st.header("📌 Bulk Prediction from File")
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename columns to match expected format
    rename_dict = {
        "age": "age",
        "weight": "weight",
        "weight_unit": "weight_unit",
        "height": "height",
        "height_unit": "height_unit",
        "steps": "steps_per_day",
        "sleep": "sleep_hours",
        "aqi": "air_quality",
        "family_history": "family_history"
    }
    df.rename(columns=rename_dict, inplace=True)

    # Check for missing columns
    expected_columns = {"age", "weight", "weight_unit", "height", "height_unit", "steps_per_day", "sleep_hours", "air_quality", "family_history"}
    missing_columns = expected_columns - set(df.columns)

    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
    else:
        st.success("✅ File uploaded successfully! Processing data...")

        # Calculate BMI using appropriate units
        df["bmi"] = df.apply(lambda row: calculate_bmi(row["weight"], row["height"], row["weight_unit"], row["height_unit"]), axis=1)

        # Select required features
        df_selected = df[["age", "bmi", "steps_per_day", "sleep_hours", "air_quality", "family_history"]]

        # Scale Data
        df_scaled = scaler.transform(df_selected)

        # Predict Risks
        df["risk"] = model.predict(df_scaled)
        df["risk_confidence"] = model.predict_proba(df_scaled)[:, 1] * 100
        df["risk_level"] = df["risk"].apply(lambda x: "🔴 High Risk" if x == 1 else "🟢 Low Risk")

        # Display Results
        st.write("### 📊 Prediction Results:")
        st.dataframe(df[["age", "bmi", "steps_per_day", "sleep_hours", "air_quality", "family_history", "risk_level", "risk_confidence"]])

        # Downloadable File
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "health_risk_predictions.csv", "text/csv")

        # **🔹 Data Visualization: Risk vs Non-Risk**
        st.subheader("📊 Risk Distribution Visualization")
        risk_counts = df["risk_level"].value_counts()

        # **Bar Chart - Risk Distribution**
        bar_fig = go.Figure(
            data=[go.Bar(x=risk_counts.index, y=risk_counts.values, marker=dict(color=["green", "red"]))],
            layout=go.Layout(title="Risk vs Non-Risk Count", xaxis=dict(title="Risk Level"), yaxis=dict(title="Number of Individuals"))
        )
        st.plotly_chart(bar_fig)

        # **Pie Chart - Risk Proportion**
        pie_fig = go.Figure(
            data=[go.Pie(labels=risk_counts.index, values=risk_counts.values, hole=0.4)],
            layout=go.Layout(title="Proportion of Risk vs Non-Risk")
        )
        st.plotly_chart(pie_fig)

st.write("🚀 **Developed with AI for smarter health risk predictions!**")

