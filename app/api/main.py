from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Optional
import numpy as np

app = FastAPI(title="Customer Churn Prediction API",
              description="API for predicting customer churn probability",
              version="1.0.0")

# Load model artifacts
try:
    model = joblib.load('models/saved_models/model.pkl')
    preprocessor = joblib.load('models/saved_models/preprocessor.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

class CustomerData(BaseModel):
    tenure: int = Field(..., gt=0, example=12, 
                       description="Number of months the customer has been with the company")
    MonthlyCharges: float = Field(..., gt=0, example=29.85,
                                description="Monthly charges for the customer")
    TotalCharges: float = Field(..., gt=0, example=358.2,
                              description="Total charges accumulated by the customer")
    gender: str = Field(..., example="Male",
                       description="Gender of the customer", 
                       pattern="^(Male|Female|Other)$")
    Partner: str = Field(..., example="Yes",
                        description="Whether the customer has a partner",
                        pattern="^(Yes|No)$")
    Contract: str = Field(..., example="Month-to-month",
                         description="Contract type",
                         pattern="^(Month-to-month|One year|Two year)$")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 29.85,
                "TotalCharges": 358.2,
                "gender": "Male",
                "Partner": "Yes",
                "Contract": "Month-to-month"
            }
        }

@app.post("/predict",
         summary="Predict churn probability",
         response_description="Dictionary containing churn probability and risk level")
async def predict(data: CustomerData):
    """
    Predicts the probability of customer churn based on input features.
    
    - **tenure**: Months with company (must be positive)
    - **MonthlyCharges**: Current monthly charges (must be positive)
    - **TotalCharges**: Total charges to date (must be positive)
    - **gender**: Male/Female/Other
    - **Partner**: Yes/No
    - **Contract**: Month-to-month/One year/Two year
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Transform features
        processed = preprocessor.transform(input_df)
        
        # Predict
        proba = model.predict_proba(processed)[0, 1]
        
        return {
            "churn_probability": float(proba),
            "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low",
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Prediction failed",
                "message": str(e),
                "success": False
            }
        )

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessor_loaded": True
    }