import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import time
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
import os
model_path = os.path.join(os.getcwd(), "model", "health_risk_model.pkl")
model = joblib.load(model_path)

scaler = joblib.load("model/scaler.pkl")

# Function to calculate BMI
def calculate_bmi(weight, height, weight_unit, height_unit):
    if weight_unit == "g":
        weight = weight / 1000  
    if height_unit == "cm":
        height = height / 100  
    bmi = weight / (height ** 2)
    return round(bmi, 2)

# Streamlit UI
st.set_page_config(page_title="Health Risk Prediction", layout="centered")
st.title("🩺 AI Health Risk Prediction")
st.write("Enter your details to get a personalized risk assessment and health tips!")

# User input form
with st.form("health_form"):
    age = st.number_input("🔢 Age", min_value=1, max_value=120, value=30)
    weight_unit = st.radio("⚖️ Weight Input Type", ["kg", "g"])
    weight = st.number_input("⚖️ Enter your Weight", min_value=1.0, max_value=500.0, value=70.0, step=0.1)
    height_unit = st.radio("📏 Height Input Type", ["m", "cm"])
    height = st.number_input("📏 Enter your Height", min_value=30.0, max_value=250.0, value=170.0, step=0.1)

    if st.form_submit_button("📊 Calculate BMI"):
        st.session_state.age = age  # Store age
        st.session_state.bmi = calculate_bmi(weight, height, weight_unit, height_unit)
        st.success(f"📊 Your BMI is: {st.session_state.bmi}")

if "bmi" in st.session_state:
    st.write(f"📊 Your BMI: **{st.session_state.bmi}**")
    steps_per_day = st.slider("🛋 Steps per Day", min_value=0, max_value=30000, value=5000, step=500)
    sleep_hours = st.slider("🎤 Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    air_quality = st.number_input("🌍 Air Quality Index (AQI)", min_value=0, max_value=500, value=100)
    family_history = st.radio("🏥 Family History of Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    if st.button("🔮 Predict Risk"):
        with st.spinner("Analyzing your health data... ⏳"):
            time.sleep(2)
        
        features = np.array([[st.session_state.age, st.session_state.bmi, steps_per_day, sleep_hours, air_quality, family_history]])
        features_scaled = scaler.transform(features)
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

        risk_reasons = []
        if st.session_state.bmi < 18.5 or st.session_state.bmi > 25:
            risk_reasons.append("Unhealthy BMI (Underweight/Overweight).")
        if sleep_hours < 6:
            risk_reasons.append("Inadequate Sleep (Less than 6 hours).")
        if steps_per_day < 5000:
            risk_reasons.append("Low Physical Activity (Less than 5000 steps).")
        if air_quality > 150:
            risk_reasons.append("Poor Air Quality (AQI above 150).")
        if family_history == 1:
            risk_reasons.append("Family History of Diseases.")

        if risk == 1:
            st.warning("⚠️ Your risk is high! Follow a healthy lifestyle.")
            if risk_reasons:
                st.error("**Possible reasons for high risk:**\n- " + "\n- ".join(risk_reasons))
        else:
            st.balloons()
            st.success("🎉 Your risk is low! Keep up the good habits.")  

    ### 📈 Health Risk Forecasting ###
    if "age" in st.session_state:
        st.subheader("📈 Health Risk Forecasting Over Time")

        # Predict risk for future ages
        years = [st.session_state.age, st.session_state.age + 5, st.session_state.age + 10, st.session_state.age + 20]
        risk_forecast = []

        for future_age in years:
            future_features = np.array([[future_age, st.session_state.bmi, steps_per_day, sleep_hours, air_quality, family_history]])
            future_scaled = scaler.transform(future_features)
            future_risk = model.predict_proba(future_scaled)[0][1] * 100
            risk_forecast.append(future_risk)

        # Convert to DataFrame for visualization
        df_forecast = pd.DataFrame({"Age": years, "Risk (%)": risk_forecast})

        # Plot risk trend
        fig, ax = plt.subplots()
        ax.plot(df_forecast["Age"], df_forecast["Risk (%)"], marker="o", linestyle="-", color="red")
        ax.set_xlabel("Age")
        ax.set_ylabel("Predicted Health Risk (%)")
        ax.set_title("Health Risk Prediction Over the Years")
        ax.grid()
        st.pyplot(fig)

        # Tips for staying healthy
        st.subheader("📝 Health Tips for the Future")
        tips = []

        if st.session_state.bmi < 18.5 or st.session_state.bmi > 25:
            tips.append("Maintain a balanced diet to keep a healthy BMI.")
        if sleep_hours < 6:
            tips.append("Ensure at least 7-8 hours of sleep daily.")
        if steps_per_day < 5000:
            tips.append("Increase daily physical activity to at least 10,000 steps.")
        if air_quality > 150:
            tips.append("Use an air purifier or avoid highly polluted areas.")
        if family_history == 1:
            tips.append("Get regular health check-ups to monitor risk factors.")

        if tips:
            st.info("💡 **Recommended Health Tips:**\n- " + "\n- ".join(tips))
        else:
            st.success("🎉 You're on the right track! Keep following healthy habits.")

