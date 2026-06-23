from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CustomerData(BaseModel):
    gender: str
    age: float
    partner: str
    dependents: str
    tenure: float
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract_type: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float
    avg_monthly_usage_gb: float
    support_tickets: float
    satisfaction_score: float


class BatchCustomerData(BaseModel):
    customers: List[CustomerData]


class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_prediction: int
    churn_probability: float
    risk_level: str
    risk_factors: Dict[str, float]


def create_app(
    model_path: Path = Path("models/saved/best_model.pkl"),
    preprocessor_path: Path = Path("models/saved/preprocessor.pkl"),
) -> FastAPI:
    from src.models.predict import Predictor

    app = FastAPI(
        title="Telco Churn Prediction API",
        description="ML-powered customer churn prediction service",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    predictor = Predictor(model_path, preprocessor_path)

    @app.get("/")
    def root():
        return {
            "service": "Telco Churn Prediction API",
            "version": "1.0.0",
            "status": "healthy",
            "endpoints": {
                "predict": "/predict - Single customer prediction",
                "predict_batch": "/predict/batch - Batch prediction",
                "health": "/health - Health check",
            },
        }

    @app.get("/health")
    def health():
        return {"status": "healthy", "model_loaded": True}

    @app.post("/predict", response_model=PredictionResponse)
    def predict_single(customer: CustomerData):
        try:
            df = pd.DataFrame([customer.dict()])
            result = predictor.predict_with_details(df)
            risk_factors = predictor.get_risk_factors(df)

            return PredictionResponse(
                customer_id=None,
                churn_prediction=int(result["prediction"].iloc[0]),
                churn_probability=float(result["churn_probability"].iloc[0]),
                risk_level=str(result["risk_level"].iloc[0]),
                risk_factors=risk_factors["risk_factors"],
            )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch")
    def predict_batch(data: BatchCustomerData):
        try:
            df = pd.DataFrame([c.dict() for c in data.customers])
            results = predictor.predict_with_details(df)

            return {
                "predictions": results["prediction"].tolist(),
                "probabilities": results["churn_probability"].tolist(),
                "risk_levels": results["risk_level"].tolist(),
                "summary": {
                    "total": len(results),
                    "predicted_churn": int(results["prediction"].sum()),
                    "churn_rate": float(results["prediction"].mean()),
                },
            }
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
