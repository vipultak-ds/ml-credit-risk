"""
📦 Pull Production Model from MLflow Registry OR Use Serving Endpoint (Auto-Resolved Endpoint Name)
"""

import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

print("=" * 70)
print("📦 PRODUCTION MODEL SETUP")
print("=" * 70)


# ---------------------- CONFIGURATION ----------------------
class Config:

    USE_DATABRICKS_SECRETS = os.getenv("USE_DATABRICKS_SECRETS", "false").lower() == "true"
    SECRET_SCOPE = os.getenv("SECRET_SCOPE", None)
    SECRET_HOST_KEY = os.getenv("SECRET_HOST_KEY", None)
    SECRET_TOKEN_KEY = os.getenv("SECRET_TOKEN_KEY", None)

    if USE_DATABRICKS_SECRETS and "dbutils" in globals():
        try:
            DATABRICKS_HOST = dbutils.secrets.get(SECRET_SCOPE, SECRET_HOST_KEY)
            DATABRICKS_TOKEN = dbutils.secrets.get(SECRET_SCOPE, SECRET_TOKEN_KEY)
            print("🔐 Databricks secrets loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Databricks secrets: {e}")
            sys.exit(1)
    else:
        DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
        DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

    MODEL_NAME = os.getenv("MODEL_NAME", "workspace.ml_credit_risk.credit_risk_model_random_forest")
    MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")

    # If ENV not set → auto generate from MODEL_NAME
    SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", None)

    LOCAL_MODEL_DIR = "models"
    ENDPOINT_CONFIG_FILE = os.path.join(LOCAL_MODEL_DIR, "endpoint_config.json")
    METADATA_FILE = os.path.join(LOCAL_MODEL_DIR, "model_metadata.json")


config = Config()


# ---------------------- HELPERS ----------------------
def auto_generate_endpoint_name(model_name: str):
    """Automatically create endpoint name based on registered model name."""
    base = model_name.split(".")[-1]  # credit_risk_model_random_forest
    # Convert underscores → hyphen except before model type
    return f"{base.replace('_model', '-model')}-prod"


def validate_credentials():
    if not config.DATABRICKS_HOST or not config.DATABRICKS_TOKEN:
        print("❌ ERROR: Missing Databricks credentials.")
        sys.exit(1)
    print("✅ Credentials validated")


def initialize_mlflow():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    print(f"\n🔧 Connected to MLflow at → {config.DATABRICKS_HOST[:30]}***")
    return MlflowClient()


def get_model_info(client):
    print("\n🔍 Fetching model info...")
    mv = client.get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
    run = client.get_run(mv.run_id)

    model_info = {
        "model_name": config.MODEL_NAME,
        "alias": config.MODEL_ALIAS,
        "version": mv.version,
        "run_id": mv.run_id,
        "status": mv.status,
        "metrics": dict(run.data.metrics),
        "timestamp": datetime.now().isoformat()
    }

    print(f"📌 Production Version: v{mv.version}")
    return model_info


def detect_endpoint_name():
    """Determine endpoint name correctly."""
    if config.SERVING_ENDPOINT_NAME:
        print(f"⚙️ Using endpoint from ENV → {config.SERVING_ENDPOINT_NAME}")
        return config.SERVING_ENDPOINT_NAME
    
    generated = auto_generate_endpoint_name(config.MODEL_NAME)
    print(f"✨ Auto-generated endpoint name → {generated}")
    return generated


def check_endpoint(name):
    print(f"\n🔍 Checking endpoint: {name}")
    url = f"{config.DATABRICKS_HOST}/api/2.0/serving-endpoints/{name}"
    r = requests.get(url, headers={"Authorization": f"Bearer {config.DATABRICKS_TOKEN}"})

    if r.status_code == 200:
        print("✅ Endpoint exists and responding.")
        return True, r.json()
    
    print(f"❌ Endpoint not found ({r.status_code})")
    return False, None


def test_inference(name):
    print("\n🧪 Testing prediction...")

    url = f"{config.DATABRICKS_HOST}/serving-endpoints/{name}/invocations"

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
        "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    res = requests.post(url, json=payload, headers=headers)

    if res.status_code == 200:
        print("🎯 Prediction OK →", res.json())
        return True

    print("❌ Prediction failed →", res.text)
    return False


def save_config(model_info, endpoint_name):
    os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)

    data = {
        "use_serving_endpoint": True,
        "endpoint_name": endpoint_name,
        "endpoint_url": f"{config.DATABRICKS_HOST}/serving-endpoints/{endpoint_name}/invocations",
        "model_info": model_info,
        "saved_at": datetime.now().isoformat()
    }

    with open(config.ENDPOINT_CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"💾 Saved config → {config.ENDPOINT_CONFIG_FILE}")


# ---------------------- MAIN LOGIC ----------------------
def main():
    validate_credentials()
    client = initialize_mlflow()
    model_info = get_model_info(client)

    endpoint_name = detect_endpoint_name()

    exists, ep_info = check_endpoint(endpoint_name)

    if exists and test_inference(endpoint_name):
        save_config(model_info, endpoint_name)
        print("\n🎉 MODEL READY WITH SERVING ENDPOINT 🚀")
        return True

    print("\n❌ No working endpoint found. Deployment required.")
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
