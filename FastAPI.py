from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI()

# Input schema using Pydantic
class Applicant(BaseModel):
    experience_years: int
    technical_score: int

# Load trained model and scaler
model = joblib.load(r"C:/TurkcellGYK_YZ/Machine Learning/Supervised Learning/5. SVM/SVM_Project/model.joblib")
scaler = joblib.load(r"C:/TurkcellGYK_YZ/Machine Learning/Supervised Learning/5. SVM/SVM_Project/scaler.joblib")

@app.get("/")
def root():
   return {
       "title": "🤖 AI Hiring Prediction System",
       "description": "SVM-based candidate evaluation API",
       "version": "1.0",
       "endpoints": {
           "POST /predict": "Make hiring prediction",
           "GET /test": "Run test predictions",
           "GET /docs": "API documentation"
       },
       "usage": {
           "experience_years": "0-10 years",
           "technical_score": "0-100 points"
       },
       "example": {
           "experience_years": 5,
           "technical_score": 75
       }
   }


@app.post("/predict")
def make_prediction(applicant: Applicant):
    
    # Input verisini ölçeklendir
    input_scaled = scaler.transform([[applicant.experience_years, applicant.technical_score]])
    
    # Tahmin yap
    prediction = model.predict(input_scaled)
    
    # modelinizde 0=alınmadı, 1=alındı
    result = "❌ Not Hired" if prediction[0] == 0 else "✅ Hired"
    
    return {
        "prediction": result,
        "prediction_value": int(prediction[0]),
        "experience": applicant.experience_years,
        "technical_score": applicant.technical_score
    }

@app.get("/test")
def test_predictions():
    test_cases = [
        {"experience": 1, "technical_score": 50},  # Az tecrübe, düşük skor
        {"experience": 5, "technical_score": 85},  # İyi tecrübe, yüksek skor
        {"experience": 0, "technical_score": 30},  # Hiç tecrübe, çok düşük skor
        {"experience": 8, "technical_score": 95}   # Çok tecrübe, çok yüksek skor
    ]
    
    results = []
    for case in test_cases:
        input_scaled = scaler.transform([[case["experience"], case["technical_score"]]])
        prediction = model.predict(input_scaled)
        result = "❌ Not Hired" if prediction[0] == 0 else "✅ Hired"
        
        results.append({
            "input": case,
            "prediction": result,
            "prediction_value": int(prediction[0])
        })
    
    return {"test_results": results}