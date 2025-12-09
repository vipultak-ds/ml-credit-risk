#!/bin/bash

set -e

echo "======================================================================"
echo "🚀 LOCAL MLOPS DEPLOYMENT (Model Registry Mode)"
echo "======================================================================"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# -------------------------------------------------------
# STEP 1: Environment Check
# -------------------------------------------------------

echo -e "\n${BLUE}[STEP 1/5]${NC} Checking environment..."

if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found${NC}"
    echo "Creating .env from template..."

    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo -e "${YELLOW}⚠️  Please update .env with actual credentials${NC}"
        exit 1
    else
        echo -e "${RED}❌ .env.template not found${NC}"
        echo ""
        echo "Please create .env with:"
        echo "  DATABRICKS_HOST=https://your-workspace.cloud.databricks.com"
        echo "  DATABRICKS_TOKEN=dapi..."
        echo "  MODEL_NAME=workspace.ml_credit_risk.credit_risk_model_random_forest"
        echo "  MODEL_ALIAS=Production"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Environment file found${NC}"
source .env

# Validate credentials
if [ -z "$DATABRICKS_HOST" ] || [ -z "$DATABRICKS_TOKEN" ]; then
    echo -e "${RED}❌ Missing DATABRICKS_HOST or DATABRICKS_TOKEN in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Databricks credentials configured${NC}"
echo -e "${BLUE}ℹ️  Mode: Direct from Model Registry${NC}"

# Copy .env to deployment
if [ ! -f "deployment/.env" ]; then
    cp .env deployment/.env
    echo -e "${GREEN}✅ .env copied to deployment/${NC}"
fi

# -------------------------------------------------------
# STEP 2: Python Environment
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 2/5]${NC} Setting up Python environment..."

# Check if venv exists, if not ask to create
if [ ! -d "venv" ]; then
    echo "Virtual environment not found."
    read -p "Create virtual environment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 -m venv venv
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    else
        echo -e "${YELLOW}⚠️  Skipping virtual environment${NC}"
    fi
fi

if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠️  Using system Python${NC}"
fi

# -------------------------------------------------------
# STEP 3: Install Dependencies
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 3/5]${NC} Installing dependencies..."

pip install --upgrade pip --quiet

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✅ Dependencies installed from requirements.txt${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found, installing core packages...${NC}"
    pip install mlflow fastapi uvicorn pandas numpy python-dotenv --quiet
    echo -e "${GREEN}✅ Core dependencies installed${NC}"
fi

# -------------------------------------------------------
# STEP 4: Verify Model Access
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 4/5]${NC} Verifying Model Registry access..."

python3 << EOF
import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    
    # Configure MLflow
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    
    # Get model info
    model_name = os.getenv("MODEL_NAME", "workspace.ml_credit_risk.credit_risk_model_random_forest")
    model_alias = os.getenv("MODEL_ALIAS", "Production")
    
    print(f"   Checking model: {model_name} @ {model_alias}")
    
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, model_alias)
    
    print(f"   ✓ Model found: Version {mv.version}")
    print(f"   ✓ Status: {mv.status}")
    sys.exit(0)
    
except Exception as e:
    print(f"   ✗ Model verification failed: {e}")
    print("")
    print("   Troubleshooting:")
    print("   1. Check DATABRICKS_HOST and DATABRICKS_TOKEN")
    print("   2. Verify MODEL_NAME exists in Unity Catalog")
    print("   3. Verify MODEL_ALIAS is set on the model")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Model Registry access verified${NC}"
else
    echo -e "${RED}❌ Model verification failed${NC}"
    exit 1
fi

# -------------------------------------------------------
# STEP 5: Start API
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 5/5]${NC} Starting FastAPI..."

# Kill existing process if running
if [ -f "deployment/api.pid" ]; then
    OLD_PID=$(cat deployment/api.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Stopping existing API (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
    rm deployment/api.pid
fi

# Start API
cd deployment
rm -f api.log

nohup python app.py > api.log 2>&1 &
API_PID=$!
echo $API_PID > api.pid

echo -e "${GREEN}✅ API started (PID: $API_PID)${NC}"

# Wait for startup
echo -e "\n${BLUE}Waiting for API to start...${NC}"
echo "   (This may take 10-30 seconds while model loads)"

sleep 5

if ! ps -p $API_PID > /dev/null 2>&1; then
    echo -e "${RED}❌ API failed to start${NC}"
    echo ""
    echo "Error log:"
    tail -30 api.log
    exit 1
fi

# Health check
echo -e "\n${BLUE}Performing health check...${NC}"
for i in {1..15}; do
    HEALTH_CHECK=$(curl -s http://localhost:8000/health 2>/dev/null || echo "failed")
    
    if echo "$HEALTH_CHECK" | grep -q "healthy"; then
        echo -e "${GREEN}✅ API HEALTH OK${NC}"
        echo ""
        echo "$HEALTH_CHECK" | python -m json.tool 2>/dev/null
        break
    fi
    
    if [ $i -eq 15 ]; then
        echo -e "${RED}❌ HEALTH CHECK FAILED${NC}"
        echo ""
        echo "Last 30 lines of api.log:"
        tail -30 api.log
        exit 1
    fi
    
    echo "   Attempt $i/15..."
    sleep 2
done

echo ""
echo "======================================================================"
echo -e "${GREEN}🚀 LOCAL DEPLOYMENT SUCCESSFUL${NC}"
echo "======================================================================"
echo ""
echo "📋 Mode: Model Registry (Direct Load)"
echo ""
echo "🌐 Available URLs:"
echo "   Health:     http://localhost:8000/health"
echo "   Model Info: http://localhost:8000/model/info"
echo "   API Docs:   http://localhost:8000/docs"
echo "   Root:       http://localhost:8000/"
echo ""
echo "📝 View logs:  tail -f deployment/api.log"
echo "🛑 Stop API:   kill $API_PID"
echo "======================================================================"