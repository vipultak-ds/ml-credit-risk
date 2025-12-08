"""
📦 Pull Production Model from MLflow Registry
"""

import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

print("="*70)
print("📦 PULLING PRODUCTION MODEL FROM MLFLOW REGISTRY")
print("="*70)

# CONFIGURATION

class Config:

    # Read base ENV variables
    USE_DATABRICKS_SECRETS = os.getenv("USE_DATABRICKS_SECRETS", "false").lower() == "true"
    SECRET_SCOPE = os.getenv("SECRET_SCOPE", None)
    SECRET_HOST_KEY = os.getenv("SECRET_HOST_KEY", None)
    SECRET_TOKEN_KEY = os.getenv("SECRET_TOKEN_KEY", None)

    # If using Databricks Secrets (only inside Databricks runtime)
    if USE_DATABRICKS_SECRETS and "dbutils" in globals():
        try:
            DATABRICKS_HOST = dbutils.secrets.get(SECRET_SCOPE, SECRET_HOST_KEY)
            DATABRICKS_TOKEN = dbutils.secrets.get(SECRET_SCOPE, SECRET_TOKEN_KEY)
            print("🔐 Databricks secrets loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Databricks secrets: {e}")
            sys.exit(1)
    else:
        # Normal local .env mode
        DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
        DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

    MODEL_NAME = os.getenv(
        "MODEL_NAME",
        "workspace.ml_credit_risk.credit_risk_model_random_forest"
    )
    MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Production")

    # Local paths
    LOCAL_MODEL_DIR = "models"
    LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "production_model")
    METADATA_FILE = os.path.join(LOCAL_MODEL_DIR, "model_metadata.json")

config = Config()

# VALIDATION

def validate_credentials():
    """Validate Databricks credentials"""
    if not config.DATABRICKS_HOST:
        print("❌ ERROR: DATABRICKS_HOST missing")
        sys.exit(1)

    if not config.DATABRICKS_TOKEN:
        print("❌ ERROR: DATABRICKS_TOKEN missing")
        sys.exit(1)

    print("✅ Credentials validated")

# MLFLOW INITIALIZATION

def initialize_mlflow():
    """Initialize MLflow connection"""
    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")

        print(f"\n🔧 MLflow Configuration:")
        print(f"   Tracking URI: databricks")
        print(f"   Registry URI: databricks-uc")
        print(f"   Host: {config.DATABRICKS_HOST}")

        return MlflowClient()

    except Exception as e:
        print(f"❌ MLflow initialization failed: {e}")
        sys.exit(1)

# MODEL FETCHING

def get_production_model_info(client):
    """Get production model information"""
    try:
        print(f"\n🔍 Fetching model information...")
        print(f"   Model: {config.MODEL_NAME}")
        print(f"   Alias: @{config.MODEL_ALIAS}")

        model_version = client.get_model_version_by_alias(
            config.MODEL_NAME,
            config.MODEL_ALIAS
        )

        run = client.get_run(model_version.run_id)

        info = {
            "model_name": config.MODEL_NAME,
            "alias": config.MODEL_ALIAS,
            "version": int(model_version.version),
            "run_id": model_version.run_id,
            "status": model_version.status,
            "creation_timestamp": model_version.creation_timestamp,
            "current_stage": model_version.current_stage,
            "description": model_version.description,
            "tags": dict(model_version.tags) if model_version.tags else {},
            "metrics": dict(run.data.metrics) if run else {},
            "params": dict(run.data.params) if run else {},
            "pulled_at": datetime.now().isoformat()
        }

        print(f"\n📊 Model Information:")
        print(f"   Version: v{info['version']}")
        print(f"   Run ID: {info['run_id']}")
        print(f"   Status: {info['status']}")

        if info['metrics']:
            print(f"\n   📈 Metrics:")
            for metric, value in list(info['metrics'].items())[:5]:
                print(f"      {metric}: {value:.4f}")

        return info

    except Exception as e:
        print(f"❌ Failed to fetch model info: {e}")
        sys.exit(1)

# MODEL DOWNLOAD

def download_model(model_info):
    """Download model from MLflow Registry"""
    try:
        os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)

        if os.path.exists(config.LOCAL_MODEL_PATH):
            print(f"\n🗑️  Removing existing model...")
            shutil.rmtree(config.LOCAL_MODEL_PATH)

        model_uri = f"models:/{config.MODEL_NAME}@{config.MODEL_ALIAS}"

        print(f"\n📥 Downloading model...")
        print(f"   Source: {model_uri}")
        print(f"   Destination: {config.LOCAL_MODEL_PATH}")

        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=config.LOCAL_MODEL_PATH
        )

        print(f"✅ Model downloaded successfully")

        save_metadata(model_info)

        return True

    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

# METADATA MANAGEMENT

def save_metadata(model_info):
    try:
        with open(config.METADATA_FILE, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"✅ Metadata saved: {config.METADATA_FILE}")

    except Exception as e:
        print(f"⚠️  Failed to save metadata: {e}")

# VERIFICATION

def verify_download():
    try:
        print(f"\n🔍 Verifying download...")

        if not os.path.exists(config.LOCAL_MODEL_PATH):
            print(f"❌ Model directory not found: {config.LOCAL_MODEL_PATH}")
            return False

        model_files = os.listdir(config.LOCAL_MODEL_PATH)
        print(f"   Found {len(model_files)} artifacts")

        if os.path.exists(config.METADATA_FILE):
            with open(config.METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            print(f"   Model version: v{metadata['version']}")

        print(f"✅ Verification passed")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

# TEST MODEL LOADING

def test_model_load():
    try:
        print(f"\n🧪 Testing model load...")

        model = mlflow.pyfunc.load_model(config.LOCAL_MODEL_PATH)

        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")

        return True

    except Exception as e:
        print(f"⚠️  Model load test failed: {e}")
        return False

# MAIN EXECUTION

def main():
    validate_credentials()
    client = initialize_mlflow()
    model_info = get_production_model_info(client)
    success = download_model(model_info)

    if not success:
        print("\n❌ Model pull FAILED")
        sys.exit(1)

    verify_download()
    test_model_load()

    print("\n" + "="*70)
    print("✅ MODEL PULL COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n📦 Model Details:")
    print(f"   Name: {config.MODEL_NAME}")
    print(f"   Version: v{model_info['version']}")
    print(f"   Alias: @{config.MODEL_ALIAS}")
    print(f"   Location: {config.LOCAL_MODEL_PATH}")
    print(f"\n📋 Next Steps:")
    print(f"   1. Start API: python app.py")
    print(f"   2. Test API: curl http://localhost:8000/health")
    print(f"   3. View docs: http://localhost:8000/docs")
    print("="*70)

if __name__ == "__main__":
    main()
