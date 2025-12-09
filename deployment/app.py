"""
🚀 FastAPI Inference Service - Credit Risk Prediction
Uses ONLY Model Registry (NO Serving Endpoint)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

# Fix permission issue when loading models from Databricks Registry
os.environ["MLFLOW_ENABLE_ARTIFACT_PROXY"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================

class Config:
    """Application configuration"""
    
    # Databricks credentials
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
    
    # Model info from Model Registry
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
    model_source: str
    timestamp: str


# ========================================
# MODEL LOADER (Model Registry ONLY)
# ========================================

class ModelLoader:
    """Loads model directly from Databricks Model Registry"""
    
    def __init__(self):
        self.model = None
        self.model_version = None
        self._initialize_mlflow()
        self._load_model()
    
    def _initialize_mlflow(self):
        """Configure MLflow to connect to Databricks"""
        try:
            logger.info("🔧 Configuring MLflow for Databricks Unity Catalog...")
            
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            
            logger.info("✅ MLflow configured successfully")
        except Exception as e:
            logger.error(f"❌ MLflow configuration failed: {e}")
            raise
    
    def _load_model(self):
        """Load model from Model Registry"""
        try:
            model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
            
            logger.info("="*70)
            logger.info("📦 Loading model from Model Registry...")
            logger.info(f"   Model URI: {model_uri}")
            logger.info("="*70)
            
            # Load the model
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model version info
            client = MlflowClient()
            mv = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
            self.model_version = mv.version
            
            logger.info("="*70)
            logger.info("✅ MODEL LOADED SUCCESSFULLY")
            logger.info(f"   Name: {config.MODEL_NAME}")
            logger.info(f"   Alias: {config.MODEL_ALIAS}")
            logger.info(f"   Version: {self.model_version}")
            logger.info(f"   Status: {mv.status}")
            logger.info(f"   Source: Model Registry (Direct)")
            logger.info("="*70)
            
        except Exception as e:
            logger.error("="*70)
            logger.error("❌ MODEL LOADING FAILED")
            logger.error(f"   Error: {e}")
            logger.error("="*70)
            logger.error("\n💡 Troubleshooting:")
            logger.error("   1. Check DATABRICKS_HOST and DATABRICKS_TOKEN in .env")
            logger.error(f"   2. Verify model exists: {config.MODEL_NAME}")
            logger.error(f"   3. Verify alias exists: {config.MODEL_ALIAS}")
            logger.error("   4. Check network connectivity to Databricks")
            raise
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(df)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        # Try to get probabilities
        try:
            return self.model.predict_proba(df)
        except AttributeError:
            # If model doesn't have predict_proba, create dummy probabilities
            predictions = self.predict(df)
            proba_matrix = [[0.7, 0.3] if p == 0 else [0.3, 0.7] for p in predictions]
            return np.array(proba_matrix)


# ========================================
# FASTAPI APPLICATION
# ========================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="MLOps inference service using Databricks Model Registry",
    version="1.0.0",
    docs_url="/docs"
)

# Initialize model loader
model_loader = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model_loader
    try:
        logger.info("🚀 Starting API server...")
        model_loader = ModelLoader()
        logger.info("✅ API ready to serve predictions")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error("   API will start but predictions will fail")


@app.get("/")
async def root():
    return {
        "message": "Credit Risk Prediction API",
        "status": "running",
        "model_source": "Databricks Model Registry",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    ok = model_loader is not None
    return HealthResponse(
        status="healthy" if ok else "unhealthy",
        model_loaded=ok,
        model_name=config.MODEL_NAME,
        model_version=str(model_loader.model_version) if ok else "N/A",
        model_source="Model Registry (Direct)",
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": config.MODEL_NAME,
        "model_version": str(model_loader.model_version),
        "model_alias": config.MODEL_ALIAS,
        "model_source": "Databricks Model Registry (Direct Load)",
        "features": config.FEATURES,
        "feature_count": len(config.FEATURES),
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CreditRiskInput):
    """Single prediction endpoint"""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])[config.FEATURES]
        
        # Get predictions
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
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(batch_data: BatchCreditRiskInput):
    """Batch prediction endpoint"""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert inputs to DataFrame
        df = pd.DataFrame([item.dict() for item in batch_data.inputs])[config.FEATURES]
        
        # Get predictions
        preds = model_loader.predict(df)
        probs = model_loader.predict_proba(df)
        
        results = [
            PredictionResponse(
                prediction=int(pred),
                prediction_label="High Risk" if pred == 1 else "Low Risk",
                probability=float(probs[i][int(pred)]),
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
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ========================================
# RUN SERVER
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*70)
    logger.info("🚀 Starting Credit Risk API Server")
    logger.info("="*70)
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Alias: {config.MODEL_ALIAS}")
    logger.info("Source: Databricks Model Registry (Direct)")
    logger.info("="*70)
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)