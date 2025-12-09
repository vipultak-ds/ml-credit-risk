#!/bin/bash

echo "🔍 DEBUGGING API STATUS"
echo "======================="

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
    echo "Last 20 lines of api.log:"
    echo "------------------------"
    tail -20 deployment/api.log
else
    echo "❌ api.log not found"
fi

# Check if deployment files exist
echo ""
echo "4. Checking deployment files..."
REQUIRED_FILES=(
    "deployment/app.py"
    "deployment/models/endpoint_config.json"
    "deployment/.env"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file MISSING"
    fi
done

echo ""
echo "======================="
echo "💡 Next Steps:"
echo "======================="
if lsof -i :8000 2>/dev/null | grep -q LISTEN; then
    echo "API is running but not responding:"
    echo "  → Check logs above for errors"
    echo "  → Try restarting: cd deployment && python app.py"
else
    echo "API is NOT running:"
    echo "  → Run: cd deployment && python app.py"
    echo "  → Or use: bash deployment/deploy_local.sh"
fi