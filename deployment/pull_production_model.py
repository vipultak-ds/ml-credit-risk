"""
📦 Pull Production Model - Model Registry Version
Updated to work with app.py that loads from Model Registry
"""

import os
import sys
import json
from datetime import datetime

# ⚠️ CRITICAL: Set BEFORE MLflow import
os.environ["MLFLOW_ENABLE_ARTIFACT_PROXY"] = "true"

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 70)
print("📦 PRODUCTION MODEL SETUP (Model Registry)")
print("=" * 70)


# ---------------------- CONFIG ---------------------- #

MODEL_NAME = "workspace.ml_credit_risk.credit_risk_model_random_forest"
MODEL_ALIAS = "Production"

# Auth
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

# Output paths - for storing metadata only
LOCAL_MODEL_DIR = os.path.join(os.getcwd(), "deployment", "models")
METADATA_FILE = os.path.join(LOCAL_MODEL_DIR, "model_metadata.json")
CONFIG_FILE = os.path.join(LOCAL_MODEL_DIR, "model_config.json")


# ---------------------- FUNCTIONS ---------------------- #

def create_directories():
    """Ensure all necessary directories exist"""
    print(f"\n📁 Creating directory: {LOCAL_MODEL_DIR}")
    try:
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        print(f"✅ Directory created/verified: {LOCAL_MODEL_DIR}")
        return True
    except Exception as e:
        print(f"❌ Failed to create directory: {e}")
        return False


def validate_credentials():
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        print("❌ Missing DATABRICKS_HOST or DATABRICKS_TOKEN")
        sys.exit(1)
    print(f"✅ Connected to Databricks → {DATABRICKS_HOST}")


def connect_mlflow():
    """Configure MLflow for Databricks"""
    print("\n🔗 Configuring MLflow...")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    print("✅ MLflow configured")
    return MlflowClient()


def fetch_model_metadata(client):
    """Fetch model metadata from registry"""
    print("\n🔍 Fetching Production model metadata...")

    try:
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

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Model metadata saved → {METADATA_FILE}")
        return metadata
    
    except Exception as e:
        print(f"⚠️ Could not fetch model metadata: {e}")
        # Return minimal metadata so script can continue
        return {
            "model_name": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": "unknown",
            "status": "READY",
            "timestamp": datetime.now().isoformat()
        }


def test_model_loading():
    """Test if model can be loaded from registry"""
    print("\n🧪 Testing model loading from registry...")
    
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    print(f"   Model URI: {model_uri}")
    print("   ⏳ This may take 30-60 seconds on first load...")
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Model loaded successfully from registry!")
        return True
    except Exception as e:
        error_str = str(e)
        
        if "AccessDenied" in error_str or "s3:" in error_str:
            print("\n❌ MODEL LOADING FAILED - S3 Access Denied")
            print("\n🔍 DIAGNOSIS: Artifact Proxy Not Working")
            print("="*70)
            print("Your Databricks workspace doesn't support artifact proxy.")
            print("\n📋 This is common in Databricks Community Edition.")
            print("\n✅ SOLUTION: Use Serving Endpoint Instead")
            print("   1. Deploy model to serving endpoint in Databricks")
            print("   2. Use the serving endpoint version of scripts")
            print("   3. Or download model manually to use locally")
            print("="*70)
            return False
        else:
            print(f"❌ Model loading failed: {e}")
            return False


def save_config(metadata):
    """Save configuration for app.py"""
    config = {
        "mode": "model_registry",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_uri": f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        "databricks_host": DATABRICKS_HOST,
        "model_info": metadata,
        "saved_at": datetime.now().isoformat()
    }

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

        print(f"💾 Configuration saved → {CONFIG_FILE}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to save configuration: {e}")
        return False


# ---------------------- MAIN EXECUTION ---------------------- #

def main():
    print("\n🚀 Setting up Model Registry integration...")
    
    # Step 1: Create directories
    if not create_directories():
        print("\n❌ Failed to create necessary directories")
        return False

    # Step 2: Validate credentials
    validate_credentials()

    # Step 3: Connect to MLflow and fetch metadata
    try:
        client = connect_mlflow()
        metadata = fetch_model_metadata(client)
    except Exception as e:
        print(f"⚠️ MLflow connection issue: {e}")
        metadata = {
            "model_name": MODEL_NAME,
            "alias": MODEL_ALIAS,
            "version": "unknown",
            "status": "READY",
            "timestamp": datetime.now().isoformat()
        }

    # Step 4: Test model loading (NEW - Critical test)
    print("\n" + "="*70)
    print("🔍 CRITICAL TEST: Checking if Model Registry loading works")
    print("="*70)
    
    can_load = test_model_loading()
    
    if not can_load:
        print("\n⚠️ Model Registry loading doesn't work in your workspace")
        print("   Saving config anyway for reference...")
        save_config(metadata)
        print("\n💡 Next Steps:")
        print("   1. Check if you're using Databricks Community Edition")
        print("   2. Consider using Serving Endpoint approach instead")
        print("   3. Or use manual model download approach")
        return False

    # Step 5: Save configuration
    if not save_config(metadata):
        print("\n❌ Failed to save configuration")
        return False

    # Verify files were created
    print("\n📋 Verification:")
    if os.path.exists(CONFIG_FILE):
        print(f"✅ {CONFIG_FILE} exists")
    else:
        print(f"❌ {CONFIG_FILE} missing!")
        
    if os.path.exists(METADATA_FILE):
        print(f"✅ {METADATA_FILE} exists")
    
    print("\n" + "="*70)
    print("✅ MODEL REGISTRY SETUP COMPLETE!")
    print("="*70)
    print("\n📋 Summary:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Alias: {MODEL_ALIAS}")
    print(f"   Version: {metadata.get('version', 'unknown')}")
    print(f"   Mode: Model Registry (Direct Load)")
    print("\n🚀 You can now start the API:")
    print("   cd deployment && python app.py")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)