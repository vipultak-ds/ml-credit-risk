"""
Test script to verify Model Registry loading works with Artifact Proxy
Run this locally to test before deploying
"""

import os
import sys

# ⚠️ CRITICAL: Set BEFORE any imports
os.environ["MLFLOW_ENABLE_ARTIFACT_PROXY"] = "true"

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Set additional env vars
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST", "")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN", "")

print("="*70)
print("🧪 TESTING MODEL REGISTRY WITH ARTIFACT PROXY")
print("="*70)

# Verify environment
print("\n1️⃣ Environment Check:")
print(f"   MLFLOW_ENABLE_ARTIFACT_PROXY = {os.environ.get('MLFLOW_ENABLE_ARTIFACT_PROXY')}")
print(f"   DATABRICKS_HOST = {os.environ.get('DATABRICKS_HOST', 'NOT SET')[:30]}...")
print(f"   DATABRICKS_TOKEN = {'SET' if os.environ.get('DATABRICKS_TOKEN') else 'NOT SET'}")

if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
    print("\n❌ Missing credentials in .env file!")
    sys.exit(1)

# Now import MLflow
print("\n2️⃣ Importing MLflow...")
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Configure MLflow
print("\n3️⃣ Configuring MLflow...")
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
print("   ✅ MLflow configured")

# Model details
MODEL_NAME = "workspace.ml_credit_risk.credit_risk_model_random_forest"
MODEL_ALIAS = "Production"

# Test 1: Check if model exists
print(f"\n4️⃣ Checking if model exists: {MODEL_NAME}")
try:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    print(f"   ✅ Model found!")
    print(f"      Version: {mv.version}")
    print(f"      Status: {mv.status}")
    print(f"      Run ID: {mv.run_id}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 2: Try to load model
print(f"\n5️⃣ Loading model from registry...")
print("   ⏳ This may take 30-90 seconds on first load...")
print("   (Downloading artifacts via Databricks proxy)")

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("\n   ✅ MODEL LOADED SUCCESSFULLY!")
    print("   Artifact proxy is working correctly")
    
except Exception as e:
    print(f"\n   ❌ MODEL LOADING FAILED!")
    print(f"   Error: {e}")
    
    error_str = str(e)
    
    if "AccessDenied" in error_str or "s3:" in error_str:
        print("\n" + "="*70)
        print("🔍 DIAGNOSIS: S3 Access Denied")
        print("="*70)
        print("\nThe artifact proxy is NOT working in your Databricks workspace.")
        print("\n📋 This means:")
        print("   ❌ Direct Model Registry loading won't work")
        print("   ❌ Community Edition API restrictions apply")
        print("\n🔧 YOUR OPTIONS:")
        print("\n   Option 1: Use Serving Endpoint (RECOMMENDED)")
        print("   - Your previous setup that was working")
        print("   - Use app_serving.py instead of app.py")
        print("   - No model download needed")
        print("\n   Option 2: Download model manually")
        print("   - Run Databricks notebook to export model")
        print("   - Use app_local.py with local model files")
        print("\n   Option 3: Upgrade Databricks")
        print("   - Standard/Premium tier has full API access")
        print("   - Direct Model Registry loading will work")
        print("="*70)
    else:
        print(f"\n💡 Other error: {error_str}")
    
    sys.exit(1)

# Test 3: Make a prediction
print("\n6️⃣ Testing prediction...")

import pandas as pd

test_data = pd.DataFrame([{
    "checking_balance": "< 0 DM",
    "months_loan_duration": 6,
    "credit_history": "critical",
    "purpose": "car (new)",
    "amount": 1169,
    "savings_balance": "unknown",
    "employment_duration": "> 7 yrs",
    "percent_of_income": 4,
    "years_at_residence": 4,
    "age": 67,
    "other_credit": "none",
    "housing": "own",
    "existing_loans_count": 2,
    "job": "skilled employee",
    "dependents": 1,
    "phone": "yes"
}])

try:
    prediction = model.predict(test_data)
    print(f"   ✅ Prediction: {prediction[0]}")
    print(f"   Risk: {'High' if prediction[0] == 1 else 'Low'}")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nModel Registry with Artifact Proxy is working correctly.")
print("You can now deploy using the Model Registry approach.")
print("\n📋 Next steps:")
print("   1. Update deployment/app.py with the fixed version")
print("   2. Run: python deployment/app.py")
print("   3. Or deploy via GitHub Actions workflow")
print("="*70)