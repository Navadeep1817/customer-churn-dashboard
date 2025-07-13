import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib
from pathlib import Path

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def run_pipeline():
    # Use real data or adjusted sample data
    data = pd.DataFrame({
        'tenure': [12, 24, 36, 6, 18, 48],
        'MonthlyCharges': [29.85, 56.95, 89.10, 45.0, 75.0, 65.0],
        'TotalCharges': [358.2, 1366.8, 3207.6, 270.0, 1350.0, 3120.0],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Partner': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 
                    'Month-to-month', 'One year', 'Two year'],
        'Churn': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes']  # 3 "Yes" cases
    })

    # Define features
    numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical = ['gender', 'Partner', 'Contract']

    # Preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical),
        ('cat', categorical_pipeline, categorical)
    ])

    # Process data
    X = data.drop('Churn', axis=1)
    y = data['Churn'].map({'Yes': 1, 'No': 0})
    X_processed = preprocessor.fit_transform(X)

    # Save artifacts
    joblib.dump(preprocessor, os.path.join(SAVED_MODELS_DIR, 'preprocessor.pkl'))
    processed_data = pd.DataFrame(X_processed)
    processed_data['churn'] = y.values
    processed_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv'), index=False)
    print("Data pipeline executed successfully!")

if __name__ == "__main__":
    run_pipeline()