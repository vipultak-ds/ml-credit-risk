"""
📦 Pull Production Model (Hardcoded Endpoint Version)
"""

import os
import sys
import json
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import requests

# Load .env file
load_dotenv()

print("=" * 70)
print("📦 PRODUCTION MODEL SETUP (Hardcoded Endpoint)")
print("=" * 70)


# ---------------------- FIXED CONFIG ---------------------- #

MODEL_NAME = "workspace.ml_credit_risk.credit_risk_model_random_forest"
MODEL_ALIAS = "Production"

# 🔥 HARD-CODED ENDPOINT
SERVING_ENDPOINT_NAME = "credit-risk-model-random_forest-prod"

# Auth
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

# Output paths
LOCAL_MODEL_DIR = "models"
ENDPOINT_CONFIG_FILE = os.path.join(LOCAL_MODEL_DIR, "endpoint_config.json")
METADATA_FILE = os.path.join(LOCAL_MODEL_DIR, "model_metadata.json")


# ---------------------- FUNCTIONS ---------------------- #

def validate_credentials():
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        print("❌ Missing DATABRICKS_HOST or DATABRICKS_TOKEN")
        sys.exit(1)
    print(f"✅ Connected to Databricks → {DATABRICKS_HOST}")


def connect_mlflow():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return MlflowClient()


def fetch_model_metadata(client):
    print("\n🔍 Fetching Production model metadata...")

    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(mv.run_id)

    metadata = {
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": mv.version,
        "run_id": mv.run_id,
        "status": mv.status,
        "metrics": dict(run.data.metrics),
        "timestamp": datetime.now().isoformat()
    }

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Model metadata saved → {METADATA_FILE}")
    return metadata


def check_endpoint():
    print(f"\n🔍 Checking serving endpoint: {SERVING_ENDPOINT_NAME}")

    url = f"{DATABRICKS_HOST}/api/2.0/serving-endpoints/{SERVING_ENDPOINT_NAME}"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("✅ Serving endpoint exists")
        return True

    print(f"❌ Endpoint missing or inaccessible → {response.status_code}")
    return False


def test_prediction():
    print("\n🧪 Testing inference...")

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
        print("🎯 Prediction Successful →", response.json())
        return True

    print(f"❌ Prediction Failed → {response.status_code} | {response.text}")
    return False


def save_endpoint_config(metadata):
    data = {
        "use_serving_endpoint": True,
        "endpoint_name": SERVING_ENDPOINT_NAME,
        "endpoint_url": f"{DATABRICKS_HOST}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
        "model_info": metadata,
        "saved_at": datetime.now().isoformat()
    }

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    with open(ENDPOINT_CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"💾 Endpoint config saved → {ENDPOINT_CONFIG_FILE}")


# ---------------------- MAIN EXECUTION ---------------------- #

def main():
    validate_credentials()

    client = connect_mlflow()
    metadata = fetch_model_metadata(client)

    if not check_endpoint():
        print("\n❌ No valid endpoint available. Deploy first.")
        return False

    if not test_prediction():
        print("\n❌ Endpoint exists but inference failed.")
        return False

    save_endpoint_config(metadata)

    print("\n🚀 Model successfully configured using serving endpoint.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
