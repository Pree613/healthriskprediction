import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create a dataset (Replace with real data)
data = {
    "age": np.random.randint(20, 80, 1000),
    "bmi": np.random.uniform(18, 35, 1000),
    "steps_per_day": np.random.randint(1000, 15000, 1000),
    "sleep_hours": np.random.uniform(4, 10, 1000),
    "air_quality": np.random.randint(0, 500, 1000),
    "family_history": np.random.randint(0, 2, 1000),  # 1: Yes, 0: No
    "disease_risk": np.random.randint(0, 2, 1000)  # Target (1: High Risk, 0: Low Risk)
}

df = pd.DataFrame(data)

# Split data
X = df.drop(columns=["disease_risk"])
y = df["disease_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Create a directory to save the model
os.makedirs("model", exist_ok=True)

# Save model & scaler
joblib.dump(model, "model/health_risk_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model and scaler saved successfully in 'model/' directory.")
