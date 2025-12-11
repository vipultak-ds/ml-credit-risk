"""
 🚀 FastAPI Inference Service - Credit Risk Prediction
"""

import os
os.environ["MLFLOW_ENABLE_ARTIFACT_PROXY"] = "true"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# MODEL CONFIG
 
class Config:
    FEATURES = [
        "checking_balance", "months_loan_duration", "credit_history",
        "purpose", "amount", "savings_balance", "employment_duration",
        "percent_of_income", "years_at_residence", "age",
        "other_credit", "housing", "existing_loans_count",
        "job", "dependents", "phone"
    ]
    
    MODEL_NAME = os.getenv("MODEL_NAME", "workspace.ml_credit_risk.credit_risk_model_random_forest")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")
    
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

config = Config()
 
# INPUT SCHEMA
 
class CreditRiskInput(BaseModel):
    checking_balance: str = Field(..., example="< 0")
    months_loan_duration: int = Field(..., example=6)
    credit_history: str = Field(..., example="critical")
    purpose: str = Field(..., example="car")
    amount: int = Field(..., example=1169)
    savings_balance: str = Field(..., example="unknown")
    employment_duration: str = Field(..., example="> 7")
    percent_of_income: int = Field(..., example=4)
    years_at_residence: int = Field(..., example=4)
    age: int = Field(..., example=30)
    other_credit: str = Field(..., example="none")
    housing: str = Field(..., example="own")
    existing_loans_count: int = Field(..., example=1)
    job: str = Field(..., example="skilled")
    dependents: int = Field(..., example=1)
    phone: str = Field(..., example="yes")


class BatchCreditRiskInput(BaseModel):
    inputs: List[CreditRiskInput]
 
# MODEL LOADER
 
class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_info = {}
        self.load_model_from_registry()

    def load_model_from_registry(self):
        try:
            logger.info("=" * 70)
            logger.info("📦 LOADING MODEL FROM REGISTRY")
            logger.info("=" * 70)
            
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            
            client = MlflowClient()
            model_version = client.get_model_version_by_alias(
                config.MODEL_NAME, config.MODEL_ALIAS
            )
            
            self.model_info = {
                "name": config.MODEL_NAME,
                "alias": config.MODEL_ALIAS,
                "version": model_version.version,
                "status": model_version.status,
                "run_id": model_version.run_id,
                "loaded_at": datetime.now().isoformat()
            }
            
            model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info("✅ Model loaded successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def predict(self, df):
        return self.model.predict(df)

    def predict_proba(self, df):
        try:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(df)
            return self.model._model_impl.python_model.predict_proba(df)
        except:
            preds = self.predict(df)
            return np.array([[0.30, 0.70] if p == 1 else [0.80, 0.20] for p in preds])

 
# FASTAPI APP
 
app = FastAPI(
    title="Credit Risk Prediction API (Model Registry)",
    description="API for credit risk prediction using models from Databricks Model Registry",
    version="2.0.0"
)
model_loader: Optional[ModelLoader] = None


@app.on_event("startup")
def refresh_schema():
    app.openapi_schema = None


@app.on_event("startup")
async def startup():
    global model_loader
    model_loader = ModelLoader()   # ← FIXED: no try/except, no swallowing errors


@app.get("/")
def root():
    return {
        "service": "Credit Risk Prediction API",
        "version": "2.0.0",
        "mode": "Model Registry",
        "status": "running",
        "model_loaded": model_loader is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": model_loader.model_info,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
def model_info():
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    info = model_loader.model_info

    return {
        "model_name": info["name"],
        "model_alias": info["alias"],
        "model_version": str(info["version"]),
        "status": info["status"],
        "run_id": info["run_id"],
        "features": config.FEATURES,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
def predict_record(input_data: CreditRiskInput):
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = pd.DataFrame([input_data.dict()])[config.FEATURES]

    pred = int(model_loader.predict(df)[0])
    proba = float(model_loader.predict_proba(df)[0][1])
    risk_score = round(1 - proba, 6)

    return {
        "prediction": pred,
        "prediction_label": "High Risk" if pred == 1 else "Low Risk",
        "probability": proba,
        "risk_score": risk_score,
        "model_version": str(model_loader.model_info.get("version")),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict/batch")
def predict_batch(batch_data: BatchCreditRiskInput):
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = pd.DataFrame([item.dict() for item in batch_data.inputs])[config.FEATURES]

    preds = model_loader.predict(df)
    probs = model_loader.predict_proba(df)

    results = []
    for i, p in enumerate(preds):
        proba = float(probs[i][1])
        risk_score = round(1 - proba, 6)

        results.append({
            "prediction": int(p),
            "prediction_label": "High Risk" if p == 1 else "Low Risk",
            "probability": proba,
            "risk_score": risk_score,
            "timestamp": datetime.now().isoformat()
        })

    return {
        "total_count": len(results),
        "model_version": str(model_loader.model_info.get("version")),
        "predictions": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
