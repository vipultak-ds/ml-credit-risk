"""
📦 Pull Production Model from MLflow Registry OR Use Serving Endpoint (Config-Driven Endpoint Name)
"""

import os
import sys
import json
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import requests

# Optional: for reading pipeline_config.yml
from pathlib import Path
import yaml

# Load environment variables
load_dotenv()

print("=" * 70)
print("📦 PRODUCTION MODEL SETUP")
print("=" * 70)


# ---------------------- CONFIGURATION ----------------------
class Config:
    def __init__(self):
        # Secrets / Auth
        self.USE_DATABRICKS_SECRETS = os.getenv("USE_DATABRICKS_SECRETS", "false").lower() == "true"
        self.SECRET_SCOPE = os.getenv("SECRET_SCOPE", None)
        self.SECRET_HOST_KEY = os.getenv("SECRET_HOST_KEY", None)
        self.SECRET_TOKEN_KEY = os.getenv("SECRET_TOKEN_KEY", None)

        # Databricks Host / Token
        if self.USE_DATABRICKS_SECRETS and "dbutils" in globals():
            try:
                self.DATABRICKS_HOST = dbutils.secrets.get(self.SECRET_SCOPE, self.SECRET_HOST_KEY)
                self.DATABRICKS_TOKEN = dbutils.secrets.get(self.SECRET_SCOPE, self.SECRET_TOKEN_KEY)
                print("🔐 Databricks secrets loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Databricks secrets: {e}")
                sys.exit(1)
        else:
            self.DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
            self.DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

        # Model registry name + alias
        self.MODEL_NAME = os.getenv(
            "MODEL_NAME",
            "workspace.ml_credit_risk.credit_risk_model_random_forest"
        )
        self.MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")

        # Explicit endpoint name (if provided)
        self.SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", None)

        # Local files
        self.LOCAL_MODEL_DIR = "models"
        self.ENDPOINT_CONFIG_FILE = os.path.join(self.LOCAL_MODEL_DIR, "endpoint_config.json")
        self.METADATA_FILE = os.path.join(self.LOCAL_MODEL_DIR, "model_metadata.json")

        # Pipeline config path (same pattern as serving script, but repo-relative)
        self.PIPELINE_CONFIG_PATHS = [
            Path("dev_env/pipeline_config.yml"),
            Path("pipeline_config.yml"),
        ]


config = Config()


# ---------------------- HELPERS ----------------------
def validate_credentials():
    if not config.DATABRICKS_HOST or not config.DATABRICKS_TOKEN:
        print("❌ ERROR: Missing Databricks credentials (DATABRICKS_HOST / DATABRICKS_TOKEN).")
        sys.exit(1)
    print("✅ Credentials validated")


def initialize_mlflow():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    print(f"\n🔧 Connected to MLflow at → {config.DATABRICKS_HOST[:40]}***")
    return MlflowClient()


def get_model_info(client: MlflowClient):
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
        "timestamp": datetime.now().isoformat(),
    }

    print(f"📌 Production Version: v{mv.version}")
    print("📊 Metrics:")
    for k, v in model_info["metrics"].items():
        print(f"   - {k}: {v}")

    # Save metadata file (optional but handy)
    os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)
    with open(config.METADATA_FILE, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"💾 Saved model metadata → {config.METADATA_FILE}")

    return model_info


def auto_generate_endpoint_name_from_model_name(model_name: str) -> str:
    """
    Fallback heuristic if neither ENV nor pipeline_config.yml available.
    Example:
      workspace.ml_credit_risk.credit_risk_model_random_forest
      -> credit-risk-model-random_forest-prod
    (close to how your serving script builds it)
    """
    base = model_name.split(".")[-1]  # credit_risk_model_random_forest

    # Split into tokens
    parts = base.split("_")  # ["credit", "risk", "model", "random", "forest"]

    # Make "credit-risk-model" from first 3 tokens
    if len(parts) >= 3:
        base_name = "-".join(parts[0:3])  # credit-risk-model
        model_type = "_".join(parts[3:]) if len(parts) > 3 else "model"
    else:
        # fallback very basic
        base_name = base.replace("_", "-")
        model_type = "model"

    endpoint_name = f"{base_name}-{model_type}-prod"
    return endpoint_name


def load_endpoint_name_from_pipeline_config() -> str | None:
    """
    Try to read dev_env/pipeline_config.yml same as serving script.
    Uses:
      endpoint_name_format: "{base_name}-{model_type}-prod"
      base_name: "credit_risk_model"
      model_type: "random_forest"
    So endpoint_name = "credit-risk-model-random_forest-prod"
    """
    for path in config.PIPELINE_CONFIG_PATHS:
        if path.exists():
            try:
                print(f"\n📋 Loading endpoint name from pipeline config: {path}")
                with path.open("r") as f:
                    pipeline_cfg = yaml.safe_load(f)

                model_type = pipeline_cfg["model"]["type"]
                base_name = pipeline_cfg["model"]["base_name"]
                endpoint_format = pipeline_cfg["serving"]["endpoint_name_format"]

                endpoint_name = endpoint_format.format(
                    base_name=base_name.replace("_", "-"),
                    model_type=model_type,
                )
                print(f"✅ Endpoint derived from pipeline_config.yml → {endpoint_name}")
                return endpoint_name
            except Exception as e:
                print(f"⚠️ Failed to derive endpoint from {path}: {e}")
                return None

    print("ℹ️ pipeline_config.yml not found in expected locations, skipping config-based endpoint discovery.")
    return None


def detect_endpoint_name() -> str:
    """
    Resolution order:
    1. SERVING_ENDPOINT_NAME env (highest priority)
    2. pipeline_config.yml (same logic as serving script)
    3. auto_generate_endpoint_name_from_model_name (fallback)
    """
    # 1️⃣ Explicit ENV override
    if config.SERVING_ENDPOINT_NAME:
        print(f"\n⚙️ Using endpoint from ENV → {config.SERVING_ENDPOINT_NAME}")
        return config.SERVING_ENDPOINT_NAME

    # 2️⃣ Config-driven (recommended – keeps serving & pull scripts in sync)
    cfg_name = load_endpoint_name_from_pipeline_config()
    if cfg_name:
        return cfg_name

    # 3️⃣ Fallback heuristic
    generated = auto_generate_endpoint_name_from_model_name(config.MODEL_NAME)
    print(f"\n✨ Auto-generated endpoint name (fallback) → {generated}")
    return generated


def check_endpoint(name: str):
    print(f"\n🔍 Checking endpoint: {name}")
    url = f"{config.DATABRICKS_HOST}/api/2.0/serving-endpoints/{name}"
    headers = {"Authorization": f"Bearer {config.DATABRICKS_TOKEN}"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        print(f"❌ Error calling serving endpoint API: {e}")
        return False, None

    if r.status_code == 200:
        print("✅ Endpoint exists and responding.")
        return True, r.json()

    print(f"❌ Endpoint not found (status: {r.status_code}) → {r.text}")
    return False, None


def test_inference(name: str) -> bool:
    print("\n🧪 Testing prediction on serving endpoint...")

    url = f"{config.DATABRICKS_HOST}/serving-endpoints/{name}/invocations"

    payload = {
        "dataframe_records": [
            {
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
                "phone": "yes",
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=20)
    except Exception as e:
        print(f"❌ Error calling inference endpoint: {e}")
        return False

    if res.status_code == 200:
        print("🎯 Prediction OK →", res.json())
        return True

    print(f"❌ Prediction failed (status: {res.status_code}) → {res.text}")
    return False


def save_config(model_info, endpoint_name: str):
    os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)

    data = {
        "use_serving_endpoint": True,
        "endpoint_name": endpoint_name,
        "endpoint_url": f"{config.DATABRICKS_HOST}/serving-endpoints/{endpoint_name}/invocations",
        "model_info": model_info,
        "saved_at": datetime.now().isoformat(),
    }

    with open(config.ENDPOINT_CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"💾 Saved endpoint config → {config.ENDPOINT_CONFIG_FILE}")
    print(f"   → {data['endpoint_url']}")


# ---------------------- MAIN LOGIC ----------------------
def main() -> bool:
    validate_credentials()
    client = initialize_mlflow()
    model_info = get_model_info(client)

    endpoint_name = detect_endpoint_name()

    exists, _ = check_endpoint(endpoint_name)

    if exists and test_inference(endpoint_name):
        save_config(model_info, endpoint_name)
        print("\n🎉 MODEL READY WITH SERVING ENDPOINT 🚀")
        return True

    print("\n❌ No working endpoint found or inference failed. Deployment required before pull script.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
