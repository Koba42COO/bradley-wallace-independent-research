#!/usr/bin/env python3
"""
System Integration Test
Tests all features and ensures everything is working
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8080"

def test_health():
    """Test health endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✅ Health check passed")
    return True

def test_ai_generation():
    """Test AI generation"""
    print("Testing AI generation...")
    prompts = [
        "Explain the Wallace Transform",
        "What is prime aligned compute mathematics?",
        "How does breakthrough detection work?"
    ]
    
    for prompt in prompts:
        response = requests.post(
            f"{BASE_URL}/api/ai/generate",
            json={"prompt": prompt, "model": "prime aligned compute"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "prime_aligned_metrics" in data
        print(f"  ✅ Generated response for: {prompt[:30]}...")
    
    print("✅ AI generation passed")
    return True

def test_system_status():
    """Test system status"""
    print("Testing system status...")
    response = requests.get(f"{BASE_URL}/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "prime_aligned_level" in data
    print("✅ System status passed")
    return True

def test_validation():
    """Test prime aligned compute validation"""
    print("Testing prime aligned compute validation...")
    test_data = {
        "test_data": {
            "wallace_transform_input": [0.5, 1.0, 1.618],
            "f2_optimization_input": [1.0, 2.0, 3.0],
            "consciousness_rule_input": 0.5
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/prime aligned compute/validate",
        json=test_data
    )
    assert response.status_code == 200
    data = response.json()
    assert "prime_aligned_score" in data
    assert "results" in data
    print("✅ prime aligned compute validation passed")
    return True

def test_trajectory():
    """Test trajectory endpoint"""
    print("Testing trajectory...")
    response = requests.get(f"{BASE_URL}/api/prime aligned compute/trajectory")
    assert response.status_code == 200
    data = response.json()
    assert "trajectory" in data
    assert "current_score" in data
    print("✅ Trajectory passed")
    return True

def test_level_update():
    """Test level update"""
    print("Testing level update...")
    response = requests.post(
        f"{BASE_URL}/api/prime aligned compute/level",
        json={"level": 10}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert data["new_level"] == 10
    print("✅ Level update passed")
    return True

def main():
    print("\n" + "="*50)
    print("prime aligned compute MATHEMATICS SYSTEM TEST")
    print("="*50 + "\n")
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    for i in range(10):
        try:
            requests.get(f"{BASE_URL}/health", timeout=1)
            print("Server is ready!\n")
            break
        except:
            if i == 9:
                print("❌ Server is not responding. Please start the server first.")
                sys.exit(1)
            time.sleep(2)
    
    # Run all tests
    tests = [
        test_health,
        test_ai_generation,
        test_system_status,
        test_validation,
        test_trajectory,
        test_level_update
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed: {e}")
            failed += 1
    
    print("\n" + "="*50)
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
        print("System is fully functional")
    else:
        print(f"❌ {failed} TESTS FAILED")
        print("Please check the errors above")
    print("="*50 + "\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
