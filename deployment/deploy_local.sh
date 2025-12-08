#!/bin/bash

set -e

echo "======================================================================"
echo "🚀 STARTING LOCAL MLOPS DEPLOYMENT"
echo "======================================================================"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# -------------------------------------------------------
# STEP 1: Environment Check & Load Secrets
# -------------------------------------------------------

echo -e "\n${BLUE}[STEP 1/6]${NC} Checking environment..."

if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found${NC}"
    echo "Creating .env from template..."

    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo -e "${YELLOW}⚠️  Please update .env with actual secrets first${NC}"
        exit 1
    else
        echo -e "${RED}❌ .env.template not found${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Environment file found${NC}"
source .env

# --- 🔥 NEW SECRET HANDLING ----------------------------------------

# if user is using secret mapping format → map values properly
if [ "$USE_DATABRICKS_SECRETS" = "true" ]; then
    echo -e "${BLUE}🔐 Using secret-mapped configuration from .env${NC}"

    if [ -z "$SECRET_HOST_KEY" ] || [ -z "$SECRET_TOKEN_KEY" ]; then
        echo -e "${RED}❌ SECRET_HOST_KEY or SECRET_TOKEN_KEY missing in .env${NC}"
        exit 1
    fi

    # map variable names
    export DATABRICKS_HOST="${SECRET_HOST_KEY}"
    export DATABRICKS_TOKEN="${SECRET_TOKEN_KEY}"
else
    echo -e "${BLUE}ℹ️ Using direct env credentials${NC}"
fi

# Validate
if [ -z "$DATABRICKS_HOST" ] || [ -z "$DATABRICKS_TOKEN" ]; then
    echo -e "${RED}❌ Missing required Databricks credentials (HOST or TOKEN)${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Databricks credentials set${NC}"

# -------------------------------------------------------
# STEP 2: Virtual Environment
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 2/6]${NC} Setting up Python environment..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# -------------------------------------------------------
# STEP 3: Install Dependencies
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 3/6]${NC} Installing dependencies..."

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}✅ Dependencies installed${NC}"

# -------------------------------------------------------
# STEP 4: Pull Model
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 4/6]${NC} Pulling production model..."

python scripts/pull_production_model.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Model pull failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Model pulled successfully${NC}"

# -------------------------------------------------------
# STEP 5: Verify Model
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 5/6]${NC} Verifying model..."

if [ ! -d "models/production_model" ]; then
    echo -e "${RED}❌ Model directory missing${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Model artifacts present${NC}"

# -------------------------------------------------------
# STEP 6: Start API
# -------------------------------------------------------
echo -e "\n${BLUE}[STEP 6/6]${NC} Starting FastAPI..."

if [ -f "api.pid" ]; then
    OLD_PID=$(cat api.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Stopping existing API (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
fi

nohup python app.py > api.log 2>&1 &
API_PID=$!
echo $API_PID > api.pid

sleep 5

if ps -p $API_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API running (PID: $API_PID)${NC}"
else
    echo -e "${RED}❌ API failed to start${NC}"
    exit 1
fi

echo -e "\n${BLUE}Performing health check...${NC}"
sleep 3

HEALTH_CHECK=$(curl -s http://localhost:8000/health || echo "failed")

if echo "$HEALTH_CHECK" | grep -q "healthy"; then
    echo -e "${GREEN}✅ API HEALTH OK${NC}"
else
    echo -e "${RED}❌ HEALTH CHECK FAILED${NC}"
fi

echo ""
echo "======================================================================"
echo -e "${GREEN}🚀 LOCAL DEPLOYMENT SUCCESSFUL${NC}"
echo "======================================================================"
