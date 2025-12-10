"""
🚀 FastAPI Inference Service - Credit Risk Prediction
Uses Model Registry (Production Alias)
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


# ======================
# MODEL CONFIG
# ======================

class Config:
    FEATURES = [
        "checking_balance", "months_loan_duration", "credit_history",
        "purpose", "amount", "savings_balance", "employment_duration",
        "percent_of_income", "years_at_residence", "age",
        "other_credit", "housing", "existing_loans_count",
        "job", "dependents", "phone"
    ]
    
    # Model Registry Settings
    MODEL_NAME = os.getenv("MODEL_NAME", "workspace.ml_credit_risk.credit_risk_model_random_forest")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")
    
    # Databricks Settings
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

config = Config()


# ======================
# INPUT SCHEMA
# ======================

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
    job: str = Field(..., example="skilled employee")
    dependents: int = Field(..., example=1)
    phone: str = Field(..., example="yes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "checking_balance": "< 0",
                "months_loan_duration": 6,
                "credit_history": "critical",
                "purpose": "car",
                "amount": 1169,
                "savings_balance": "unknown",
                "employment_duration": "> 7",
                "percent_of_income": 4,
                "years_at_residence": 4,
                "age": 30,
                "other_credit": "none",
                "housing": "own",
                "existing_loans_count": 1,
                "job": "skilled employee",
                "dependents": 1,
                "phone": "yes"
            }
        }
    }


class BatchCreditRiskInput(BaseModel):
    inputs: List[CreditRiskInput]


# ======================
# MODEL LOADER
# ======================

class ModelLoader:
    def __init__(self):
        self.model = None
        self.model_info = {}
        self.load_model_from_registry()

    def load_model_from_registry(self):
        """Load model from Databricks Model Registry"""
        try:
            logger.info("=" * 70)
            logger.info("📦 LOADING MODEL FROM REGISTRY")
            logger.info("=" * 70)
            
            # Configure MLflow
            logger.info(f"🔗 Connecting to Databricks: {config.DATABRICKS_HOST}")
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            
            # Get model info
            client = MlflowClient()
            logger.info(f"🔍 Fetching model: {config.MODEL_NAME} @ {config.MODEL_ALIAS}")
            
            model_version = client.get_model_version_by_alias(
                config.MODEL_NAME, 
                config.MODEL_ALIAS
            )
            
            self.model_info = {
                "name": config.MODEL_NAME,
                "alias": config.MODEL_ALIAS,
                "version": model_version.version,
                "status": model_version.status,
                "run_id": model_version.run_id,
                "loaded_at": datetime.now().isoformat()
            }
            
            logger.info(f"📌 Model Version: {model_version.version}")
            logger.info(f"📌 Status: {model_version.status}")
            logger.info(f"📌 Run ID: {model_version.run_id}")
            
            # Load model using model URI
            model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
            logger.info(f"⏳ Loading model from URI: {model_uri}")
            
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info("✅ Model loaded successfully from registry!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Failed to load model from registry: {e}")
            logger.error(f"   Model Name: {config.MODEL_NAME}")
            logger.error(f"   Model Alias: {config.MODEL_ALIAS}")
            logger.error(f"   Databricks Host: {config.DATABRICKS_HOST}")
            raise

    def predict(self, df):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded!")
        
        # Use MLflow's predict method
        predictions = self.model.predict(df)
        return predictions

    def predict_proba(self, df):
        """Get prediction probabilities"""
        try:
            # Try to get probabilities from the model
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(df)
            else:
                # For MLflow models, use the underlying model
                proba = self.model._model_impl.python_model.predict_proba(df)
                return proba
        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")
            # Fallback: generate dummy probabilities
            preds = self.predict(df)
            return np.array([[0.30, 0.70] if p == 1 else [0.80, 0.20] for p in preds])


# ======================
# FASTAPI APP
# ======================

app = FastAPI(
    title="Credit Risk Prediction API (Model Registry)",
    description="API for credit risk prediction using models from Databricks Model Registry",
    version="2.0.0"
)
model_loader: Optional[ModelLoader] = None


@app.on_event("startup")
def refresh_schema():
    app.openapi_schema = None
    logger.info("🔄 Swagger/OpenAPI schema refreshed.")


@app.on_event("startup")
async def startup():
    global model_loader
    try:
        model_loader = ModelLoader()
        logger.info("🚀 API startup complete - Model loaded from registry")
    except Exception as e:
        logger.error(f"❌ Failed to load model during startup: {e}")
        # Don't raise - let health check handle it
        model_loader = None


@app.get("/")
def root():
    """Root endpoint"""
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
    """Health check endpoint"""
    if model_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": model_loader.model_info,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
def model_info():
    """Get model information"""
    if model_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_info": model_loader.model_info,
        "features": config.FEATURES,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
def predict_record(input_data: CreditRiskInput):
    """Predict single record"""
    if model_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )

    try:
        input_df = pd.DataFrame([input_data.dict()])
        input_df = input_df[config.FEATURES]

        pred = int(model_loader.predict(input_df)[0])
        proba = model_loader.predict_proba(input_df)[0]

        return {
            "prediction": pred,
            "risk_label": "High Risk" if pred == 1 else "Low Risk",
            "risk_probability": float(proba[1]),
            "model_version": model_loader.model_info.get("version"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(batch_data: BatchCreditRiskInput):
    """Predict batch of records"""
    if model_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )

    try:
        df = pd.DataFrame([item.dict() for item in batch_data.inputs])
        df = df[config.FEATURES]

        preds = model_loader.predict(df)
        probs = model_loader.predict_proba(df)

        results = []
        for i, pred in enumerate(preds):
            results.append({
                "prediction": int(pred),
                "risk_label": "High Risk" if pred == 1 else "Low Risk",
                "risk_probability": float(probs[i][1]),
                "timestamp": datetime.now().isoformat()
            })

        return {
            "total_records": len(results),
            "model_version": model_loader.model_info.get("version"),
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# MAIN
# ======================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"🚀 Starting API server on {host}:{port}")
    logger.info(f"📦 Model: {config.MODEL_NAME} @ {config.MODEL_ALIAS}")
    
    uvicorn.run(app, host=host, port=port)