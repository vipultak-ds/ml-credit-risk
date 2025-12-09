#!/bin/bash

echo "🔍 DEBUGGING API STATUS (Model Registry Mode)"
echo "=============================================="

# Check if port 8000 is in use
echo ""
echo "1. Checking port 8000..."
if lsof -i :8000 2>/dev/null | grep -q LISTEN; then
    echo "✅ Port 8000 is in use"
    echo ""
    echo "Process details:"
    lsof -i :8000
else
    echo "❌ Port 8000 is NOT in use"
    echo "   → API is not running!"
fi

# Check if api.pid exists
echo ""
echo "2. Checking for api.pid file..."
if [ -f "deployment/api.pid" ]; then
    PID=$(cat deployment/api.pid)
    echo "✅ api.pid exists: $PID"
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Process $PID is running"
    else
        echo "❌ Process $PID is NOT running (stale PID file)"
    fi
else
    echo "❌ api.pid file not found"
fi

# Check API log
echo ""
echo "3. Checking API logs..."
if [ -f "deployment/api.log" ]; then
    echo "✅ api.log exists"
    echo ""
    echo "Last 30 lines of api.log:"
    echo "------------------------"
    tail -30 deployment/api.log
else
    echo "❌ api.log not found"
fi

# Check if deployment files exist
echo ""
echo "4. Checking deployment files..."
REQUIRED_FILES=(
    "deployment/app.py"
    "deployment/.env"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file MISSING"
    fi
done

# Check .env configuration
echo ""
echo "5. Checking .env configuration..."
if [ -f "deployment/.env" ]; then
    echo "✅ .env file exists"
    echo ""
    echo "Environment variables (without values):"
    grep -E "^(DATABRICKS_HOST|DATABRICKS_TOKEN|MODEL_NAME|MODEL_ALIAS)=" deployment/.env | sed 's/=.*/=***/' || echo "   ⚠️ No variables found"
else
    echo "❌ .env not found in deployment/"
fi

# Check Python and dependencies
echo ""
echo "6. Checking Python environment..."
if command -v python &> /dev/null; then
    echo "✅ Python found: $(python --version)"
    
    # Check MLflow
    if python -c "import mlflow" 2>/dev/null; then
        echo "✅ MLflow installed"
    else
        echo "❌ MLflow NOT installed (pip install mlflow)"
    fi
    
    # Check FastAPI
    if python -c "import fastapi" 2>/dev/null; then
        echo "✅ FastAPI installed"
    else
        echo "❌ FastAPI NOT installed (pip install fastapi uvicorn)"
    fi
else
    echo "❌ Python not found!"
fi

echo ""
echo "=============================================="
echo "💡 Next Steps:"
echo "=============================================="
if lsof -i :8000 2>/dev/null | grep -q LISTEN; then
    echo "✅ API is running but not responding properly"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs above for errors"
    echo "  2. Verify Databricks credentials in .env"
    echo "  3. Test manually: cd deployment && python app.py"
    echo "  4. Check model exists: MODEL_NAME and MODEL_ALIAS in .env"
else
    echo "❌ API is NOT running"
    echo ""
    echo "To start API:"
    echo "  Option 1: bash start_api.sh"
    echo "  Option 2: cd deployment && python app.py"
    echo ""
    echo "Requirements:"
    echo "  ✓ .env file with DATABRICKS_HOST and DATABRICKS_TOKEN"
    echo "  ✓ Model exists in Databricks Model Registry"
    echo "  ✓ Network access to Databricks"
fi
echo "=============================================="