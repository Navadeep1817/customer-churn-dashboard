import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

def train_model():
    # Load processed data
    data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv'))
    X = data.drop('churn', axis=1)
    y = data['churn']
    
    # Check for minimum samples
    if y.sum() < 2:  # Need at least 2 churn cases
        raise ValueError(f"Need ≥2 churn cases, found {y.sum()}. Add more data.")
    
    # Train model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    
    # Save model
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(SAVED_MODELS_DIR, 'model.pkl'))
    print("model.pkl successfully created!")

if __name__ == "__main__":
    train_model()