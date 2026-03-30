"""
FastAPI server for heart disease prediction model.
Deployment-ready inference service with async support and monitoring.
"""
import time
import json
from typing import List
from pydantic import BaseModel
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Initialize model
model = xgb.XGBClassifier()
model.load_model("models/xgb_seed42_d4.json")

app = FastAPI(
    title="Heart Disease Classifier API",
    description="XGBoost-based binary classifier for heart disease prediction",
    version="1.0.0"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    """13 cardiac features for prediction"""
    age: float
    sex: float              # 1=M, 0=F
    cp: float              # chest pain type (0-3)
    trestbps: float        # resting BP (mmHg)
    chol: float            # serum cholesterol (mg/dl)
    fbs: float             # fasting blood sugar > 120 (0/1)
    restecg: float         # resting ECG (0-2)
    thalach: float         # max heart rate achieved
    exang: float           # exercise induced angina (0/1)
    oldpeak: float         # ST depression
    slope: float           # ST slope (0-2)
    ca: float              # major vessels (0-4)
    thal: float            # thalassemia (0-3)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    latency_ms: float
    confidence: str

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    return {"status": "healthy", "model": "xgb_seed42_d4", "version": "1.0"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict heart disease risk.
    
    Returns:
        - prediction: 0 (no disease) or 1 (disease present)
        - probability: confidence [0-1]
        - latency_ms: inference time in milliseconds
        - confidence: "High" (>0.8), "Medium" (0.5-0.8), "Low" (<0.5)
    """
    start = time.time()
    
    try:
        # Extract features in correct order
        features = np.array([[
            request.age, request.sex, request.cp, request.trestbps,
            request.chol, request.fbs, request.restecg, request.thalach,
            request.exang, request.oldpeak, request.slope, request.ca, request.thal
        ]])
        
        # Predict
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0, 1]
        
        # Confidence leveling
        if proba > 0.8:
            confidence = "High"
        elif proba > 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        latency = (time.time() - start) * 1000  # ms
        
        return PredictionResponse(
            prediction=int(pred),
            probability=round(float(proba), 4),
            latency_ms=round(latency, 2),
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction for high-throughput scenarios"""
    start = time.time()
    results = []
    
    for req in requests:
        r = await predict(req)
        results.append(r.dict())
    
    latency = (time.time() - start) * 1000
    return {
        "count": len(results),
        "results": results,
        "total_latency_ms": round(latency, 2),
        "avg_latency_ms": round(latency / len(results), 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
