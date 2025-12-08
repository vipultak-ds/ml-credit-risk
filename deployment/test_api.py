"""
🧪 API Testing Script
Tests the deployed FastAPI inference service
"""

import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# ==================================================
# CONFIG (Modified as per env_template)
# ==================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")      # default if missing
API_PORT = os.getenv("API_PORT", "8000")

# Dynamic API URL
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# Optional model info (not used directly but loaded as per requirement)
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# Test data (NO CHANGE)
SAMPLE_INPUT = {
    "checking_balance": "< 0 DM",
    "months_loan_duration": 6,
    "credit_history": "critical",
    "purpose": "car (new)",
    "amount": 1169,
    "savings_balance": "unknown",
    "employment_duration": "> 7 yrs",
    "percent_of_income": 4,
    "years_at_residence": 4,
    "age": 67,
    "other_credit": "none",
    "housing": "own",
    "existing_loans_count": 2,
    "job": "skilled employee",
    "dependents": 1,
    "phone": "yes"
}

BATCH_INPUT = {
    "inputs": [
        SAMPLE_INPUT,
        {
            "checking_balance": "1 - 200 DM",
            "months_loan_duration": 12,
            "credit_history": "repaid",
            "purpose": "furniture",
            "amount": 2500,
            "savings_balance": "> 1000 DM",
            "employment_duration": "1 - 4 yrs",
            "percent_of_income": 2,
            "years_at_residence": 3,
            "age": 45,
            "other_credit": "bank",
            "housing": "rent",
            "existing_loans_count": 1,
            "job": "skilled employee",
            "dependents": 2,
            "phone": "yes"
        }
    ]
}

# ==================================================
# ⚠️ BELOW THIS LINE — NO LOGIC CHANGED
# ==================================================

def print_header(title: str):
    print("\n" + "="*70)
    print(f"🧪 {title}")
    print("="*70)

def print_result(passed: bool, message: str):
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status}: {message}")

def test_health_check() -> bool:
    print_header("TEST 1: Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
            result = data.get("status") == "healthy"
            print_result(result, "API is healthy")
            return result
        else:
            print_result(False, "Unexpected response")
    except:
        print_result(False, "API connection failed")
    return False

def test_model_info() -> bool:
    print_header("TEST 2: Model Info")
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
            result = all(key in data for key in ["model_name", "model_version", "features"])
            print_result(result, "Model info retrieved")
            return result
        else:
            print_result(False, "Unexpected response")
    except Exception as e:
        print_result(False, f"Error: {e}")
    return False

def test_single_prediction() -> bool:
    print_header("TEST 3: Single Prediction")
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=SAMPLE_INPUT, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
            result = all(k in data for k in ["prediction","prediction_label","probability","risk_score"])
            print_result(result, "Single prediction works")
            return result
        else:
            print_result(False, "Unexpected response")
    except Exception as e:
        print_result(False, f"Error: {e}")
    return False

def test_batch_prediction() -> bool:
    print_header("TEST 4: Batch Prediction")
    try:
        response = requests.post(f"{API_BASE_URL}/predict/batch", json=BATCH_INPUT, timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            result = data["total_count"] == len(BATCH_INPUT["inputs"])
            print_result(result, "Batch prediction successful")
            return result
    except Exception as e:
        print_result(False, f"Error: {e}")
    return False

def test_error_handling() -> bool:
    print_header("TEST 5: Error Handling")
    response = requests.post(f"{API_BASE_URL}/predict", json={"wrong": "input"})
    result = response.status_code == 422
    print_result(result, "Validation error detected correctly")
    return result

def test_performance(num_requests: int = 5) -> bool:
    print_header("TEST 6: Performance Test")
    import time
    times = []
    for _ in range(num_requests):
        start = time.time()
        requests.post(f"{API_BASE_URL}/predict", json=SAMPLE_INPUT)
        times.append(time.time() - start)

    avg = sum(times)/len(times)
    result = avg < 0.5
    print_result(result, f"Avg response: {avg:.3f}s")
    return result

def run_all_tests():
    print("\n" + "="*70)
    print("🚀 STARTING API TEST SUITE")
    print("="*70)
    print(f"Using API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME} [{MODEL_ALIAS}]")

    tests = [
        test_health_check,
        test_model_info,
        test_single_prediction,
        test_batch_prediction,
        test_error_handling,
        test_performance
    ]

    results = [test() for test in tests]
    print("\n🔍 SUMMARY:", f"{results.count(True)}/{len(results)} passed")

if __name__ == "__main__":
    run_all_tests()
