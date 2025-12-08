"""
📦 Pull Production Model (Simplified + Hardcoded Version)
"""

import os
import sys
import json
from datetime import datetime
import requests
import mlflow
from mlflow.tracking import MlflowClient

print("=" * 60)
print("📦 PRODUCTION MODEL SETUP")
print("=" * 60)


# ---------------------- FIXED CONFIG ----------------------
MODEL_NAME = "workspace.ml_credit_risk.credit_risk_model_random_forest"   # <---- HARD CODED
MODEL_ALIAS = "Production"  # alias from registry

# Serving Endpoint (Hardcoded)
SERVING_ENDPOINT_NAME = "credit-risk-model-random_forest-prod"    # <---- HARD CODED

# Required ENV for authentication
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

LOCAL_MODEL_DIR = "models"
ENDPOINT_CONFIG = f"{LOCAL_MODEL_DIR}/endpoint_config.json"
METADATA_FILE = f"{LOCAL_MODEL_DIR}/model_metadata.json"


# ---------------------- VALIDATIONS ----------------------
def validate_env():
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        print("❌ ERROR: Missing Databricks credentials.")
        print("➡️ Set DATABRICKS_HOST and DATABRICKS_TOKEN in environment variables.")
        sys.exit(1)

    print(f"✅ Running against Databricks → {DATABRICKS_HOST}")


# ---------------------- MLflow ----------------------
def connect_mlflow():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return MlflowClient()


def fetch_model_metadata(client):
    print("\n🔍 Fetching Production model version...")

    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(mv.run_id)

    metadata = {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": mv.version,
        "run_id": mv.run_id,
        "status": mv.status,
        "metrics": dict(run.data.metrics),
        "pulled_at": datetime.now().isoformat()
    }

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Saved metadata → {METADATA_FILE}")
    return metadata


# ---------------------- VALIDATE SERVING ENDPOINT ----------------------
def test_serving_endpoint():
    print(f"\n🔍 Validating serving endpoint: {SERVING_ENDPOINT_NAME}")

    url = f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"

    payload = {
        "dataframe_records": [{
            "checking_balance": "< 0 DM",
            "months_loan_duration": 6,
            "credit_history": "critical",
            "purpose": "radio/tv",
            "amount": 1169,
            "savings_balance": "unknown",
            "employment_duration": "< 1 year",
            "percent_of_income": 4,
            "years_at_residence": 4,
            "age": 67,
            "other_credit": "none",
            "housing": "own",
            "existing_loans_count": 2,
            "job": "skilled",
            "dependents": 1,
            "phone": "yes"
        }]
    }

    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print("🎯 Serving endpoint working → Prediction OK")
        print("Response:", response.json())
        return True

    print(f"❌ Serving endpoint failed ({response.status_code}) → {response.text}")
    return False


# ---------------------- SAVE CONFIG ----------------------
def save_endpoint_config(metadata):
    config = {
        "use_serving_endpoint": True,
        "endpoint_name": SERVING_ENDPOINT_NAME,
        "endpoint_url": f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
        "model": metadata
    }

    with open(ENDPOINT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)

    print(f"💾 Saved endpoint config → {ENDPOINT_CONFIG}")


# ---------------------- MAIN ----------------------
def main():
    validate_env()

    client = connect_mlflow()
    metadata = fetch_model_metadata(client)

    if not test_serving_endpoint():
        print("\n❌ Deployment Required: Serving endpoint not live.")
        return False

    save_endpoint_config(metadata)

    print("\n🚀 Successfully configured production model via serving endpoint.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
