"""
🚀 FastAPI Inference Service - Credit Risk Prediction
Serves the Production model from MLflow Model Registry
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION (UPDATED)
# ========================================

class Config:
    """Application configuration"""
    
    USE_SECRET = os.getenv("USE_DATABRICKS_SECRETS", "false").lower() == "true"
    SECRET_SCOPE = os.getenv("SECRET_SCOPE")
    SECRET_HOST_KEY = os.getenv("SECRET_HOST_KEY")
    SECRET_TOKEN_KEY = os.getenv("SECRET_TOKEN_KEY")

    if USE_SECRET:
        try:
            from databricks.sdk import WorkspaceClient
            ws = WorkspaceClient()

            DATABRICKS_HOST = ws.secrets.get(secret_scope=SECRET_SCOPE, key=SECRET_HOST_KEY)
            DATABRICKS_TOKEN = ws.secrets.get(secret_scope=SECRET_SCOPE, key=SECRET_TOKEN_KEY)

            print("🔐 Using Databricks Secret Manager credentials")
        except Exception as e:
            print(f"❌ Failed to load secrets: {e}")
            raise
    else:
        print("⚠ Using .env credentials (not from Secrets Manager)")
        DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
        DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

    MODEL_NAME = os.getenv("MODEL_NAME", "workspace.ml_credit_risk.credit_risk_model_random_forest")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")

    FEATURES = [
        "checking_balance", "months_loan_duration", "credit_history",
        "purpose", "amount", "savings_balance", "employment_duration",
        "percent_of_income", "years_at_residence", "age",
        "other_credit", "housing", "existing_loans_count",
        "job", "dependents", "phone"
    ]


config = Config()

# ========================================
# PYDANTIC MODELS (API CONTRACTS)
# ========================================

class CreditRiskInput(BaseModel):
    checking_balance: str = Field(..., example="< 0 DM")
    months_loan_duration: int = Field(..., ge=1, le=72, example=6)
    credit_history: str = Field(..., example="critical")
    purpose: str = Field(..., example="car (new)")
    amount: int = Field(..., ge=250, le=20000, example=1169)
    savings_balance: str = Field(..., example="unknown")
    employment_duration: str = Field(..., example="> 7 yrs")
    percent_of_income: int = Field(..., ge=1, le=4, example=4)
    years_at_residence: int = Field(..., ge=1, le=4, example=4)
    age: int = Field(..., ge=18, le=100, example=67)
    other_credit: str = Field(..., example="none")
    housing: str = Field(..., example="own")
    existing_loans_count: int = Field(..., ge=1, le=4, example=2)
    job: str = Field(..., example="skilled employee")
    dependents: int = Field(..., ge=1, le=2, example=1)
    phone: str = Field(..., example="yes")


class BatchCreditRiskInput(BaseModel):
    inputs: List[CreditRiskInput]


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    risk_score: float
    timestamp: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    timestamp: str


# ========================================
# MODEL LOADER
# ========================================

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_uri = None
        self._initialize_mlflow()
        self._load_model()
    
    def _initialize_mlflow(self):
        try:
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            logger.info("✅ MLflow initialized successfully")
        except Exception as e:
            logger.error(f"❌ MLflow initialization failed: {e}")
            raise
    
    def _load_model(self):
        try:
            self.model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
            logger.info(f"📦 Loading model: {self.model_uri}")
            self.model = mlflow.pyfunc.load_model(self.model_uri)

            from mlflow.tracking import MlflowClient
            client = MlflowClient()

            mv = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
            self.model_version = mv.version
            
            logger.info(f"✅ Model loaded → Version: {self.model_version}")

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise


# ========================================
# FASTAPI APPLICATION (UNCHANGED)
# ========================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="MLOps inference service for credit risk classification",
    version="1.0.0",
    docs_url="/docs"
)

try:
    model_loader = ModelLoader()
except Exception:
    model_loader = None


@app.get("/")
async def root():
    return {"message": "API running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    ok = model_loader is not None
    return HealthResponse(
        status="healthy" if ok else "unhealthy",
        model_loaded=ok,
        model_name=config.MODEL_NAME,
        model_version=str(model_loader.model_version) if ok else "N/A",
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info")
async def model_info():
    return {
        "model_name": config.MODEL_NAME,
        "model_version": str(model_loader.model_version),
        "features": config.FEATURES,
        "endpoint": "/predict"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CreditRiskInput):
    df = pd.DataFrame([input_data.dict()])[config.FEATURES]
    pred = int(model_loader.predict(df)[0])
    prob = model_loader.predict_proba(df)[0]
    label = "High Risk" if pred == 1 else "Low Risk"

    return PredictionResponse(
        prediction=pred,
        prediction_label=label,
        probability=float(prob[pred]),
        risk_score=float(prob[1]),
        timestamp=datetime.now().isoformat(),
        model_version=str(model_loader.model_version)
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(batch_data: BatchCreditRiskInput):
    df = pd.DataFrame([item.dict() for item in batch_data.inputs])[config.FEATURES]
    
    preds = model_loader.predict(df)
    probs = model_loader.predict_proba(df)
    
    results = [
        PredictionResponse(
            prediction=int(pred),
            prediction_label="High Risk" if pred == 1 else "Low Risk",
            probability=float(probs[i][pred]),
            risk_score=float(probs[i][1]),
            timestamp=datetime.now().isoformat(),
            model_version=str(model_loader.model_version)
        )
        for i, pred in enumerate(preds)
    ]
    
    return BatchPredictionResponse(
        predictions=results,
        total_count=len(results),
        timestamp=datetime.now().isoformat()
    )


# ========================================
# RUN SERVER (UNCHANGED)
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
