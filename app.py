from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/health_risk_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Homepage route
@app.route("/", methods=["GET"])
def home():
    return "Health Risk Prediction API is running! Use /predict to get predictions."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([[
            data["age"], data["bmi"], data["steps_per_day"], 
            data["sleep_hours"], data["air_quality"], data["family_history"]
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        risk = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return jsonify({
            "risk_level": "High" if risk == 1 else "Low",
            "confidence": f"{probability * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
