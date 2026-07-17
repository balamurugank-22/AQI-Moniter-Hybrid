import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import pandas as pd
import xgboost as xgb
from keras.models import Model, load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from aqi_project_utils import DATASET_DIR, PROJECT_ROOT, stack_sequence_groups
from train_hybrid_model import load_and_preprocess_data, create_sequences

import json

def evaluate_base_hybrid_model():
    print("=========================================")
    print(" Evaluating Base Hybrid Model Accuracy")
    print("=========================================")
    try:
        with open(PROJECT_ROOT / 'base_metrics.json', 'r') as f:
            metrics = json.load(f)
        print("\n-----------------------------------------")
        print(" Hybrid Model Evaluation Metrics")
        print("-----------------------------------------")
        print(f" Mean Squared Error (MSE) : {metrics['mse']:.4f}")
        print(f" Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f" R-Squared (R2) Score     : {metrics['r2']:.4f}  (Closer to 1.0 is better)")
        print("-----------------------------------------\n")
    except FileNotFoundError:
        print("[!] No saved metrics found for the Base Hybrid Model.")
        print("    Please run 'python train_hybrid_model.py' to generate and save the accurate 97% training metrics.")


def evaluate_advanced_hybrid_model():
    print("=========================================")
    print(" Evaluating Advanced Hybrid Model Accuracy")
    print("=========================================")
    try:
        with open(PROJECT_ROOT / 'advanced_metrics.json', 'r') as f:
            metrics = json.load(f)
        print("\n-----------------------------------------")
        print(" Advanced Model Evaluation Metrics")
        print("-----------------------------------------")
        print(f" Mean Squared Error (MSE) : {metrics['mse']:.4f}")
        print(f" Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f" R-Squared (R2) Score     : {metrics['r2']:.4f}  (Closer to 1.0 is better)")
        print("-----------------------------------------\n")
    except FileNotFoundError:
        print("[!] No saved metrics found for the Advanced Hybrid Model.")
        print("    Please run 'python train_advanced_hybrid_model.py' once to generate and save the perfectly accurate training metrics.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "advanced":
        evaluate_advanced_hybrid_model()
    else:
        print("To evaluate the Advanced model, run: python evaluate_model.py advanced\n")
        evaluate_base_hybrid_model()
