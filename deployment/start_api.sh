#!/bin/bash

echo "🚀 STARTING API SERVER"
echo "======================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if already running
if lsof -i :8000 2>/dev/null | grep -q LISTEN; then
    echo -e "${YELLOW}⚠️ Port 8000 is already in use${NC}"
    echo ""
    echo "Processes on port 8000:"
    lsof -i :8000
    echo ""
    read -p "Kill existing process? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Killing process..."
        lsof -ti :8000 | xargs kill -9 2>/dev/null
        sleep 2
    else
        echo "Exiting..."
        exit 1
    fi
fi

# Check required files
echo -e "\n${BLUE}Checking required files...${NC}"

if [ ! -f "deployment/app.py" ]; then
    echo -e "${RED}❌ deployment/app.py not found!${NC}"
    exit 1
fi

if [ ! -f "deployment/models/endpoint_config.json" ]; then
    echo -e "${RED}❌ deployment/models/endpoint_config.json not found!${NC}"
    echo "Run the GitHub Actions workflow first to create this file."
    exit 1
fi

if [ ! -f "deployment/.env" ]; then
    echo -e "${YELLOW}⚠️ deployment/.env not found${NC}"
    
    # Check if root .env exists
    if [ -f ".env" ]; then
        echo "Copying .env to deployment/"
        cp .env deployment/.env
    else
        echo -e "${RED}❌ No .env file found!${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ All required files present${NC}"

# Start API
echo -e "\n${BLUE}Starting API server...${NC}"
cd deployment

# Kill old process if pid file exists
if [ -f "api.pid" ]; then
    OLD_PID=$(cat api.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Killing old process: $OLD_PID"
        kill $OLD_PID 2>/dev/null
        sleep 2
    fi
    rm api.pid
fi

# Remove old log
rm -f api.log

# Start in background
echo "Executing: python app.py"
nohup python app.py > api.log 2>&1 &
API_PID=$!
echo $API_PID > api.pid

echo -e "${GREEN}✅ API started with PID: $API_PID${NC}"

# Wait for startup
echo -e "\n${BLUE}Waiting for API to start...${NC}"
sleep 5

# Check if process is still running
if ! ps -p $API_PID > /dev/null 2>&1; then
    echo -e "${RED}❌ API process died immediately!${NC}"
    echo ""
    echo "Error logs:"
    cat api.log
    exit 1
fi

# Test health endpoint
echo -e "\n${BLUE}Testing health endpoint...${NC}"
for i in {1..10}; do
    RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q "healthy"; then
        echo -e "${GREEN}✅ API is healthy!${NC}"
        echo ""
        echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
        echo ""
        echo "======================"
        echo -e "${GREEN}🎉 API STARTED SUCCESSFULLY${NC}"
        echo "======================"
        echo ""
        echo "📋 Available URLs:"
        echo "  🌐 Health:     http://localhost:8000/health"
        echo "  🌐 Model Info: http://localhost:8000/model/info"
        echo "  🌐 API Docs:   http://localhost:8000/docs"
        echo "  🌐 Root:       http://localhost:8000/"
        echo ""
        echo "📝 Logs: tail -f deployment/api.log"
        echo "🛑 Stop:  kill $API_PID"
        exit 0
    fi
    
    echo "  Attempt $i/10... waiting"
    sleep 2
done

# If we got here, health check failed
echo -e "${RED}❌ API health check failed${NC}"
echo ""
echo "Last 30 lines of api.log:"
echo "========================"
tail -30 api.log

exit 1