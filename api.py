from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict
import os

# Load the trained SVM model
try:
    model = joblib.load("outputs/trained_svm_model.pkl")
    model_type = "SVM"
    print("SVM model loaded successfully")
except FileNotFoundError:
    print("Warning: SVM model not found. Please train the model first using model_training_svm.py")
    model = None
    model_type = None

app = FastAPI(
    title="Breast Cancer Diagnosis API",
    description="API for predicting breast cancer diagnosis using machine learning",
    version="1.0.0"
)

# Feature names (same as in your training)
REQUIRED_FEATURES = [
    'concave points_mean', 'area_worst', 'fractal_dimension_worst', 
    'smoothness_worst', 'symmetry_worst', 'fractal_dimension_mean',
    'smoothness_mean', 'compactness_se', 'fractal_dimension_se', 
    'concave points_se', 'symmetry_se', 'perimeter_se', 'concavity_se',
    'symmetry_mean', 'smoothness_se', 'texture_se', 'texture_mean'
]

class PatientData(BaseModel):
    """Input data model for prediction"""
    concave_points_mean: float
    area_worst: float
    fractal_dimension_worst: float
    smoothness_worst: float
    symmetry_worst: float
    fractal_dimension_mean: float
    smoothness_mean: float
    compactness_se: float
    fractal_dimension_se: float
    concave_points_se: float
    symmetry_se: float
    perimeter_se: float
    concavity_se: float
    symmetry_mean: float
    smoothness_se: float
    texture_se: float
    texture_mean: float
    
    @validator('*', pre=True)
    def validate_numeric(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("All features must be numeric")
        if v < 0:
            raise ValueError("Feature values must be non-negative")
        return float(v)

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str  # "Benign" or "Malignant"
    probability: float
    confidence: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Breast Cancer Diagnosis API",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_type if model_type else "None",
        "sklearn_class": type(model).__name__ if model else "None",
        "n_features": len(REQUIRED_FEATURES),
        "features": REQUIRED_FEATURES,
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """Make prediction for a single patient"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert input to DataFrame with correct column names
        input_dict = {
            'concave points_mean': patient_data.concave_points_mean,
            'area_worst': patient_data.area_worst,
            'fractal_dimension_worst': patient_data.fractal_dimension_worst,
            'smoothness_worst': patient_data.smoothness_worst,
            'symmetry_worst': patient_data.symmetry_worst,
            'fractal_dimension_mean': patient_data.fractal_dimension_mean,
            'smoothness_mean': patient_data.smoothness_mean,
            'compactness_se': patient_data.compactness_se,
            'fractal_dimension_se': patient_data.fractal_dimension_se,
            'concave points_se': patient_data.concave_points_se,
            'symmetry_se': patient_data.symmetry_se,
            'perimeter_se': patient_data.perimeter_se,
            'concavity_se': patient_data.concavity_se,
            'symmetry_mean': patient_data.symmetry_mean,
            'smoothness_se': patient_data.smoothness_se,
            'texture_se': patient_data.texture_se,
            'texture_mean': patient_data.texture_mean
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of malignant
        
        # SVM doesn't have feature importances - return empty dict
        feature_importance = {}
        
        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            prediction="Malignant" if prediction == 1 else "Benign",
            probability=round(probability, 4),
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(patients: List[PatientData]):
    """Make predictions for multiple patients"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large. Maximum 100 patients.")
    
    try:
        results = []
        for i, patient in enumerate(patients):
            # Convert to prediction format
            prediction_result = await predict(patient)
            results.append({
                "patient_id": i + 1,
                **prediction_result.dict()
            })
        
        return {"predictions": results, "total_patients": len(patients)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/features")
async def get_required_features():
    """Get list of required features for prediction"""
    return {
        "required_features": REQUIRED_FEATURES,
        "total_features": len(REQUIRED_FEATURES),
        "description": "All features must be provided as numeric values"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)