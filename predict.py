import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("model/health_risk_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Sample input data
data = {
    "age": 30,
    "bmi": 25,
    "steps_per_day": 5000,
    "sleep_hours": 7,
    "air_quality": 2,
    "family_history": 1
}

# Convert input data to a NumPy array
features = np.array([[ 
    data["age"], data["bmi"], data["steps_per_day"], 
    data["sleep_hours"], data["air_quality"], data["family_history"] 
]])

# Scale features
features_scaled = scaler.transform(features)

# Make prediction
risk = model.predict(features_scaled)[0]
probability = model.predict_proba(features_scaled)[0][1]

# Display result
print(f"Risk Level: {'High' if risk == 1 else 'Low'}")
print(f"Confidence: {probability * 100:.2f}%")
