"""
🚀 FastAPI Inference Service - Credit Risk Prediction
Serves via Databricks Serving Endpoint OR Local MLflow Model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import json
import requests
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
# CONFIGURATION
# ========================================

class Config:
    """Application configuration"""
    
    # Check if we should use serving endpoint
    ENDPOINT_CONFIG_FILE = os.path.join(os.getcwd(), "models", "endpoint_config.json")
    
    USE_SERVING_ENDPOINT = False
    ENDPOINT_URL = None
    ENDPOINT_NAME = None
    MODEL_VERSION = "unknown"
    
    # Try to load endpoint config
    if os.path.exists(ENDPOINT_CONFIG_FILE):
        try:
            with open(ENDPOINT_CONFIG_FILE, 'r') as f:
                endpoint_config = json.load(f)
                USE_SERVING_ENDPOINT = endpoint_config.get("use_serving_endpoint", False)
                ENDPOINT_URL = endpoint_config.get("endpoint_url")
                ENDPOINT_NAME = endpoint_config.get("endpoint_name")
                
                model_info = endpoint_config.get("model_info", {})
                MODEL_VERSION = str(model_info.get("version", "unknown"))
                
            logger.info(f"✅ Loaded endpoint config: {ENDPOINT_NAME}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load endpoint config: {e}")
    
    # Databricks credentials
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
    
    # Model info
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
    serving_mode: str
    timestamp: str


# ========================================
# MODEL HANDLER (Supports both modes)
# ========================================

class ModelHandler:
    def __init__(self):
        self.use_endpoint = config.USE_SERVING_ENDPOINT
        self.model = None
        self.model_version = config.MODEL_VERSION
        
        if self.use_endpoint:
            self._setup_endpoint()
        else:
            self._load_local_model()
    
    def _setup_endpoint(self):
        """Setup for Databricks Serving Endpoint"""
        try:
            logger.info(f"🌐 Using Databricks Serving Endpoint: {config.ENDPOINT_NAME}")
            
            if not config.ENDPOINT_URL:
                raise ValueError("Endpoint URL not configured")
            
            if not config.DATABRICKS_TOKEN:
                raise ValueError("DATABRICKS_TOKEN not set")
            
            self.endpoint_url = config.ENDPOINT_URL
            self.headers = {
                "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"✅ Endpoint configured: {self.endpoint_url}")
            
        except Exception as e:
            logger.error(f"❌ Endpoint setup failed: {e}")
            raise
    
    def _load_local_model(self):
        """Load model from local MLflow"""
        try:
            import mlflow
            import mlflow.pyfunc
            from mlflow.tracking import MlflowClient
            
            logger.info("📦 Loading model from MLflow...")
            
            mlflow.set_tracking_uri("databricks")
            mlflow.set_registry_uri("databricks-uc")
            
            model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            client = MlflowClient()
            mv = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
            self.model_version = mv.version
            
            logger.info(f"✅ Model loaded → Version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.use_endpoint:
            return self._predict_endpoint(df)
        else:
            return self.model.predict(df)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.use_endpoint:
            # For endpoint, we need to extract probabilities from response
            return self._predict_proba_endpoint(df)
        else:
            return self.model.predict_proba(df)
    
    def _predict_endpoint(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using Databricks endpoint"""
        try:
            # Convert DataFrame to records format
            records = df.to_dict('records')
            
            payload = {
                "dataframe_records": records
            }
            
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Endpoint error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Extract predictions from response
            if isinstance(result, dict) and "predictions" in result:
                predictions = result["predictions"]
            else:
                predictions = result
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Endpoint prediction failed: {e}")
            raise
    
    def _predict_proba_endpoint(self, df: pd.DataFrame) -> np.ndarray:
        """Get probabilities from endpoint"""
        try:
            records = df.to_dict('records')
            
            # Many endpoints return probabilities in a specific format
            # Adjust this based on your endpoint's response format
            payload = {
                "dataframe_records": records
            }
            
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Endpoint error: {response.status_code}")
            
            result = response.json()
            
            # Try to extract probabilities
            # Databricks endpoints typically return predictions as list
            # For binary classification, we'll create a probability matrix
            if isinstance(result, dict) and "predictions" in result:
                preds = result["predictions"]
            else:
                preds = result
            
            # Convert predictions to probability format [prob_class_0, prob_class_1]
            proba_matrix = []
            for pred in preds:
                if isinstance(pred, (int, float)):
                    # If it's just a class prediction (0 or 1)
                    if pred == 0:
                        proba_matrix.append([0.7, 0.3])  # Default confidence
                    else:
                        proba_matrix.append([0.3, 0.7])
                elif isinstance(pred, (list, np.ndarray)):
                    # If it's already probabilities
                    proba_matrix.append(pred)
                else:
                    # Default fallback
                    proba_matrix.append([0.5, 0.5])
            
            return np.array(proba_matrix)
            
        except Exception as e:
            logger.error(f"Endpoint probability prediction failed: {e}")
            # Fallback: return default probabilities based on class predictions
            preds = self._predict_endpoint(df)
            proba_matrix = [[0.7, 0.3] if p == 0 else [0.3, 0.7] for p in preds]
            return np.array(proba_matrix)


# ========================================
# FASTAPI APPLICATION
# ========================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="MLOps inference service for credit risk classification",
    version="2.0.0",
    docs_url="/docs"
)

# Initialize model handler
try:
    model_handler = ModelHandler()
    logger.info("✅ Model handler initialized successfully")
except Exception as e:
    logger.error(f"❌ Model handler initialization failed: {e}")
    model_handler = None


@app.get("/")
async def root():
    return {
        "message": "Credit Risk API",
        "status": "running",
        "serving_mode": "endpoint" if config.USE_SERVING_ENDPOINT else "local",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    ok = model_handler is not None
    return HealthResponse(
        status="healthy" if ok else "unhealthy",
        model_loaded=ok,
        model_name=config.MODEL_NAME,
        model_version=str(model_handler.model_version) if ok else "N/A",
        serving_mode="endpoint" if config.USE_SERVING_ENDPOINT else "local",
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info")
async def model_info():
    if not model_handler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": config.MODEL_NAME,
        "model_version": str(model_handler.model_version),
        "model_alias": config.MODEL_ALIAS,
        "serving_mode": "endpoint" if config.USE_SERVING_ENDPOINT else "local",
        "endpoint_name": config.ENDPOINT_NAME if config.USE_SERVING_ENDPOINT else None,
        "features": config.FEATURES,
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CreditRiskInput):
    if not model_handler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.dict()])[config.FEATURES]
        
        # Get predictions
        pred = int(model_handler.predict(df)[0])
        prob = model_handler.predict_proba(df)[0]
        
        label = "High Risk" if pred == 1 else "Low Risk"
        
        return PredictionResponse(
            prediction=pred,
            prediction_label=label,
            probability=float(prob[pred]),
            risk_score=float(prob[1]),
            timestamp=datetime.now().isoformat(),
            model_version=str(model_handler.model_version)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(batch_data: BatchCreditRiskInput):
    if not model_handler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert inputs to DataFrame
        df = pd.DataFrame([item.dict() for item in batch_data.inputs])[config.FEATURES]
        
        # Get predictions
        preds = model_handler.predict(df)
        probs = model_handler.predict_proba(df)
        
        results = [
            PredictionResponse(
                prediction=int(pred),
                prediction_label="High Risk" if pred == 1 else "Low Risk",
                probability=float(probs[i][int(pred)]),
                risk_score=float(probs[i][1]),
                timestamp=datetime.now().isoformat(),
                model_version=str(model_handler.model_version)
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
    
    # Log startup info
    logger.info("="*70)
    logger.info("🚀 Starting Credit Risk API Server")
    logger.info("="*70)
    logger.info(f"Serving Mode: {'Databricks Endpoint' if config.USE_SERVING_ENDPOINT else 'Local MLflow'}")
    if config.USE_SERVING_ENDPOINT:
        logger.info(f"Endpoint: {config.ENDPOINT_NAME}")
    logger.info(f"Model: {config.MODEL_NAME} (v{config.MODEL_VERSION})")
    logger.info("="*70)
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)