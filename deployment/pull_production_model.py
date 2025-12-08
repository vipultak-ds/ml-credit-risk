"""
📦 Pull Production Model from MLflow Registry OR Use Serving Endpoint
"""

import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
import shutil
import tempfile
from pathlib import Path
import requests
import pickle
import yaml

# Load environment variables
load_dotenv()

print("="*70)
print("📦 PRODUCTION MODEL SETUP")
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
    
    # Serving endpoint configuration
    USE_SERVING_ENDPOINT = os.getenv("USE_SERVING_ENDPOINT", "true").lower() == "true"
    SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", "credit-risk-model-random-forest-prod")

    # Local paths
    LOCAL_MODEL_DIR = "models"
    LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "production_model")
    METADATA_FILE = os.path.join(LOCAL_MODEL_DIR, "model_metadata.json")
    ENDPOINT_CONFIG_FILE = os.path.join(LOCAL_MODEL_DIR, "endpoint_config.json")

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
        print(f"   Host: {config.DATABRICKS_HOST[:30]}***")

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

# SERVING ENDPOINT CHECK

def check_serving_endpoint():
    """Check if serving endpoint exists and is ready"""
    try:
        print(f"\n🔍 Checking serving endpoint...")
        print(f"   Endpoint: {config.SERVING_ENDPOINT_NAME}")
        
        headers = {
            "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        url = f"{config.DATABRICKS_HOST}/api/2.0/serving-endpoints/{config.SERVING_ENDPOINT_NAME}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            endpoint_info = response.json()
            state = endpoint_info.get("state", {})
            
            print(f"   Status: {state.get('ready', 'Unknown')}")
            print(f"   Config Update: {state.get('config_update', 'Unknown')}")
            
            is_ready = "READY" in str(state.get('ready', ''))
            
            if is_ready:
                print(f"✅ Serving endpoint is READY")
                return True, endpoint_info
            else:
                print(f"⚠️  Serving endpoint exists but not ready")
                return False, endpoint_info
        else:
            print(f"⚠️  Serving endpoint not found (Status: {response.status_code})")
            return False, None
            
    except Exception as e:
        print(f"⚠️  Failed to check serving endpoint: {e}")
        return False, None

# TEST SERVING ENDPOINT

def test_serving_endpoint(endpoint_info):
    """Test serving endpoint with sample data"""
    try:
        print(f"\n🧪 Testing serving endpoint...")

        headers = {
            "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }

        test_data = {
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
                    "phone": "yes"
                }
            ]
        }

        # FIXED URL
        url = f"{config.DATABRICKS_HOST}/api/2.0/serving-endpoints/{config.SERVING_ENDPOINT_NAME}/invocations"
        print(f"➡️ Calling endpoint: {url}")

        response = requests.post(url, headers=headers, json=test_data, timeout=30)

        if response.status_code == 200:
            predictions = response.json()
            print(f"✅ Endpoint test successful!")
            print(f"   Prediction: {predictions}")
            return True
        else:
            print(f"⚠️ Endpoint test failed ({response.status_code})")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        return False

# SAVE ENDPOINT CONFIGURATION

def save_endpoint_config(model_info, endpoint_info):
    """Save endpoint configuration for API to use"""
    try:
        os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)
        
        endpoint_config = {
            "use_serving_endpoint": True,
            "endpoint_name": config.SERVING_ENDPOINT_NAME,
            # 🔥 FIXED URL HERE
            "endpoint_url": f"{config.DATABRICKS_HOST}/api/2.0/serving-endpoints/{config.SERVING_ENDPOINT_NAME}/invocations",
            "model_info": model_info,
            "endpoint_state": endpoint_info.get("state") if endpoint_info else None,
            "config_version": endpoint_info.get("config", {}).get("config_version") if endpoint_info else None,
            "created_at": datetime.now().isoformat()
        }
        
        with open(config.ENDPOINT_CONFIG_FILE, 'w') as f:
            json.dump(endpoint_config, f, indent=2)
        
        print(f"\n✅ Endpoint configuration saved: {config.ENDPOINT_CONFIG_FILE}")
        
        # Also save metadata
        model_info['use_serving_endpoint'] = True
        model_info['serving_endpoint_name'] = config.SERVING_ENDPOINT_NAME
        
        with open(config.METADATA_FILE, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Metadata saved: {config.METADATA_FILE}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Failed to save endpoint config: {e}")
        return False


# DATABRICKS REST API METHOD (Fallback)

def download_via_rest_api(model_info):
    """Download model artifacts using Databricks REST API"""
    try:
        print(f"\n   🔄 Attempting to download model via REST API...")
        
        headers = {
            "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Get artifact list from run
        list_url = f"{config.DATABRICKS_HOST}/api/2.0/mlflow/artifacts/list"
        list_params = {
            "run_id": model_info['run_id'],
            "path": "model"
        }
        
        response = requests.get(list_url, headers=headers, params=list_params)
        response.raise_for_status()
        
        artifacts = response.json().get('files', [])
        print(f"   📁 Found {len(artifacts)} artifact files")
        
        if not artifacts:
            raise Exception("No artifacts found in model path")
        
        # Create local directory structure
        os.makedirs(config.LOCAL_MODEL_PATH, exist_ok=True)
        
        # Download each artifact
        download_count = 0
        for artifact in artifacts:
            artifact_path = artifact['path']
            is_dir = artifact.get('is_dir', False)
            
            if is_dir:
                # Create directory
                local_dir = os.path.join(config.LOCAL_MODEL_PATH, 
                                        artifact_path.replace('model/', ''))
                os.makedirs(local_dir, exist_ok=True)
            else:
                # Download file
                file_url = f"{config.DATABRICKS_HOST}/api/2.0/mlflow/artifacts/get"
                file_params = {
                    "run_id": model_info['run_id'],
                    "path": artifact_path
                }
                
                file_response = requests.get(file_url, headers=headers, params=file_params)
                file_response.raise_for_status()
                
                # Save file
                local_file = os.path.join(config.LOCAL_MODEL_PATH,
                                         artifact_path.replace('model/', ''))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                
                with open(local_file, 'wb') as f:
                    f.write(file_response.content)
                
                download_count += 1
        
        print(f"   ✅ Downloaded {download_count} files via REST API")
        return True
        
    except Exception as e:
        print(f"   ⚠️  REST API download failed: {str(e)}")
        return False

# MAIN SETUP LOGIC

def main():
    """Main setup logic - prefer serving endpoint over local model"""
    
    validate_credentials()
    client = initialize_mlflow()
    model_info = get_production_model_info(client)
    
    # Check if we should use serving endpoint
    if config.USE_SERVING_ENDPOINT:
        print(f"\n{'='*70}")
        print(f"🚀 USING DATABRICKS SERVING ENDPOINT (Recommended)")
        print(f"{'='*70}")
        
        endpoint_ready, endpoint_info = check_serving_endpoint()
        
        if endpoint_ready:
            # Test the endpoint
            test_success = test_serving_endpoint(endpoint_info)
            
            # Save endpoint configuration
            save_endpoint_config(model_info, endpoint_info)
            
            print(f"\n{'='*70}")
            print(f"✅ SERVING ENDPOINT SETUP COMPLETED")
            print(f"{'='*70}")
            print(f"\n📦 Configuration:")
            print(f"   Endpoint: {config.SERVING_ENDPOINT_NAME}")
            print(f"   Model: {config.MODEL_NAME}")
            print(f"   Version: v{model_info['version']}")
            print(f"   Status: {'READY ✅' if test_success else 'READY (Test Failed ⚠️)'}")
            print(f"\n📋 Next Steps:")
            print(f"   1. Start API: python app.py")
            print(f"   2. API will use serving endpoint automatically")
            print(f"   3. Test API: curl http://localhost:8000/health")
            print(f"   4. View docs: http://localhost:8000/docs")
            print(f"{'='*70}")
            
            return True
        else:
            print(f"\n⚠️  Serving endpoint not ready, falling back to local download...")
    
    # Fallback: Try to download model locally
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING MODEL LOCALLY (Fallback)")
    print(f"{'='*70}")
    
    os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)
    
    if os.path.exists(config.LOCAL_MODEL_PATH):
        print(f"\n🗑️  Removing existing model...")
        shutil.rmtree(config.LOCAL_MODEL_PATH)
    
    # Try REST API download
    if download_via_rest_api(model_info):
        print(f"\n✅ Model downloaded successfully via REST API")
        
        # Save metadata
        model_info['use_serving_endpoint'] = False
        with open(config.METADATA_FILE, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ LOCAL MODEL SETUP COMPLETED")
        print(f"{'='*70}")
        print(f"\n📦 Model Details:")
        print(f"   Name: {config.MODEL_NAME}")
        print(f"   Version: v{model_info['version']}")
        print(f"   Location: {config.LOCAL_MODEL_PATH}")
        print(f"\n📋 Next Steps:")
        print(f"   1. Start API: python app.py")
        print(f"   2. Test API: curl http://localhost:8000/health")
        print(f"   3. View docs: http://localhost:8000/docs")
        print(f"{'='*70}")
        
        return True
    
    # If everything fails
    print(f"\n❌ Both serving endpoint and local download failed")
    print(f"\n💡 Recommendations:")
    print(f"   1. Check if serving endpoint exists: {config.SERVING_ENDPOINT_NAME}")
    print(f"   2. Verify AWS S3 permissions for model artifacts")
    print(f"   3. Contact Databricks admin for access")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)