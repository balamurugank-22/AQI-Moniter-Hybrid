import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify, render_template
import pandas as pd
from predict_advanced_realtime import AdvHybridAQIPredictor
from interactive_predict import classify_aqi

app = Flask(__name__)

# Initialize model globally so it only loads once
print("Initializing the Hybrid AQI Model...")
try:
    predictor = AdvHybridAQIPredictor()
    print("Advanced Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None

def build_prediction_sequence(current_values, model_features, look_back):
    repeated_rows = [current_values.copy() for _ in range(look_back)]
    return pd.DataFrame(repeated_rows, columns=model_features)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded properly on the server."}), 500

    data = request.json
    if not data:
        return jsonify({"error": "No input data provided."}), 400

    model_features = predictor.features
    model_values = {}
    
    for feature in model_features:
        # If the advanced model expects a 'City_' feature, fallback to the local feature if not provided
        base_feature = feature.replace("City_", "")
        val = data.get(feature, data.get(base_feature, 0.0))
        try:
            model_values[feature] = float(val)
        except ValueError:
            model_values[feature] = 0.0

    try:
        prediction_sequence = build_prediction_sequence(
            model_values,
            model_features,
            predictor.look_back,
        )
        
        predicted_aqi = predictor.predict_realtime(prediction_sequence)
        category = classify_aqi(predicted_aqi)
        
        return jsonify({
            "aqi": float(predicted_aqi),
            "category": category,
            "inputs_used": model_values
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n====================================")
    print(" Starting AQI Monitor Web App")
    print("====================================")
    print("Go to http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000, use_reloader=False)
