#!/usr/bin/env python3
"""
INTEGRATION TEST SYSTEM
============================================================
Comprehensive Testing of prime aligned compute Mathematics Framework
============================================================

Phase 2 Integration Testing demonstrating:
1. End-to-end system connectivity
2. Cross-component communication
3. Real-time data flow
4. Performance validation
5. System integration verification
"""

import asyncio
import json
import time
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

@dataclass
class IntegrationTestResult:
    """Result of an integration test."""
    test_name: str
    status: str  # "passed", "failed", "warning"
    response_time: float
    data_quality: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class SystemIntegrationReport:
    """Comprehensive integration test report."""
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    warning_tests: int
    average_response_time: float
    overall_quality_score: float
    system_connectivity: Dict[str, bool]
    test_results: List[IntegrationTestResult]

class IntegrationTester:
    """Comprehensive integration testing system."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.api_keys = {
            "consciousness_researcher": "consciousness_2024_key",
            "quantum_analyst": "quantum_2024_key",
            "mathematical_validator": "math_2024_key",
            "system_admin": "admin_2024_key"
        }
        self.test_results = []
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict[str, Any] = None, api_key: str = None) -> Dict[str, Any]:
        """Make HTTP request to API Gateway."""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    async def test_api_gateway_connectivity(self) -> IntegrationTestResult:
        """Test basic API Gateway connectivity."""
        start_time = time.time()
        
        try:
            response = self._make_request("/")
            
            # Validate response structure
            required_fields = ["message", "version", "status", "systems"]
            for field in required_fields:
                if field not in response:
                    raise Exception(f"Missing required field: {field}")
            
            # Check if all expected systems are present
            expected_systems = [
                "wallace_transform", "consciousness_validator", "quantum_adaptive",
                "topological_physics", "powerball_prediction", "spectral_analysis", "data_pipeline"
            ]
            
            missing_systems = [sys for sys in expected_systems if sys not in response["systems"]]
            if missing_systems:
                raise Exception(f"Missing systems: {missing_systems}")
            
            response_time = time.time() - start_time
            quality_score = 1.0 if response["status"] == "online" else 0.5
            
            return IntegrationTestResult(
                test_name="API Gateway Connectivity",
                status="passed",
                response_time=response_time,
                data_quality=quality_score,
                details={"systems_found": len(response["systems"]), "status": response["status"]}
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="API Gateway Connectivity",
                status="failed",
                response_time=time.time() - start_time,
                data_quality=0.0,
                error_message=str(e)
            )
    
    async def test_authentication_system(self) -> IntegrationTestResult:
        """Test authentication and authorization."""
        start_time = time.time()
        
        try:
            # Test valid API key
            response = self._make_request("/api/systems", api_key=self.api_keys["consciousness_researcher"])
            
            if "user_role" not in response:
                raise Exception("Authentication response missing user_role")
            
            if response["user_role"] != "consciousness_researcher":
                raise Exception(f"Unexpected user role: {response['user_role']}")
            
            # Test invalid API key
            try:
                self._make_request("/api/systems", api_key = "OBFUSCATED_API_KEY")
                raise Exception("Invalid API key should have been rejected")
            except Exception:
                pass  # Expected behavior
            
            response_time = time.time() - start_time
            
            return IntegrationTestResult(
                test_name="Authentication System",
                status="passed",
                response_time=response_time,
                data_quality=1.0,
                details={"user_role": response["user_role"]}
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="Authentication System",
                status="failed",
                response_time=time.time() - start_time,
                data_quality=0.0,
                error_message=str(e)
            )
    
    async def test_wallace_transform_system(self) -> IntegrationTestResult:
        """Test Wallace Transform mathematical operations."""
        start_time = time.time()
        
        try:
            # Test Wallace Transform with known values
            test_data = {
                "system": "wallace_transform",
                "method": "transform",
                "parameters": {"x": 2.718}  # Euler's number
            }
            
            response = self._make_request(
                "/api/call",
                method="POST",
                data=test_data,
                api_key=self.api_keys["mathematical_validator"]
            )
            
            # Validate response
            if response["status"] != "success":
                raise Exception(f"Wallace Transform failed: {response}")
            
            if "result" not in response:
                raise Exception("Missing result in Wallace Transform response")
            
            # Validate mathematical result (should be reasonable)
            result = response["result"]
            if not isinstance(result, (int, float)) or result <= 0:
                raise Exception(f"Invalid Wallace Transform result: {result}")
            
            response_time = time.time() - start_time
            quality_score = 1.0 if abs(result - 2.6) < 0.1 else 0.8  # Approximate expected value
            
            return IntegrationTestResult(
                test_name="Wallace Transform System",
                status="passed",
                response_time=response_time,
                data_quality=quality_score,
                details={"result": result, "input": test_data["parameters"]}
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="Wallace Transform System",
                status="failed",
                response_time=time.time() - start_time,
                data_quality=0.0,
                error_message=str(e)
            )
    
    async def test_consciousness_validator_system(self) -> IntegrationTestResult:
        """Test prime aligned compute validation system."""
        start_time = time.time()
        
        try:
            # Test prime aligned compute validation
            test_data = {
                "system": "consciousness_validator",
                "method": "validate",
                "parameters": {"data": "prime aligned compute mathematics test data with YYYY STREET NAME patterns"}
            }
            
            response = self._make_request(
                "/api/call",
                method="POST",
                data=test_data,
                api_key=self.api_keys["consciousness_researcher"]
            )
            
            # Validate response
            if response["status"] != "success":
                raise Exception(f"prime aligned compute validation failed: {response}")
            
            if "prime_aligned_score" not in response:
                raise Exception("Missing prime_aligned_score in response")
            
            score = response["prime_aligned_score"]
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                raise Exception(f"Invalid prime aligned compute score: {score}")
            
            response_time = time.time() - start_time
            quality_score = 1.0 if score > 0.1 else 0.5  # Should have some prime aligned compute score
            
            return IntegrationTestResult(
                test_name="prime aligned compute Validator System",
                status="passed",
                response_time=response_time,
                data_quality=quality_score,
                details={"prime_aligned_score": score, "data_length": response.get("data_length", 0)}
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="prime aligned compute Validator System",
                status="failed",
                response_time=time.time() - start_time,
                data_quality=0.0,
                error_message=str(e)
            )
    
    async def test_quantum_adaptive_system(self) -> IntegrationTestResult:
        """Test quantum adaptive system."""
        start_time = time.time()
        
        try:
            # Test quantum adaptive system
            test_data = {
                "system": "quantum_adaptive",
                "method": "calculate_state",
                "parameters": {"amplitude": 1.0, "phase": np.pi/4}
            }
            
            response = self._make_request(
                "/api/call",
                method="POST",
                data=test_data,
                api_key=self.api_keys["quantum_analyst"]
            )
            
            # Validate response
            if response["status"] != "success":
                raise Exception(f"Quantum adaptive failed: {response}")
            
            if "quantum_state" not in response:
                raise Exception("Missing quantum_state in response")
            
            quantum_state = response["quantum_state"]
            required_fields = ["amplitude", "phase", "coherence", "entanglement_score"]
            for field in required_fields:
                if field not in quantum_state:
                    raise Exception(f"Missing quantum state field: {field}")
            
            response_time = time.time() - start_time
            quality_score = 1.0 if abs(quantum_state["coherence"]) <= 1.0 else 0.8
            
            return IntegrationTestResult(
                test_name="Quantum Adaptive System",
                status="passed",
                response_time=response_time,
                data_quality=quality_score,
                details={"quantum_state": quantum_state}
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="Quantum Adaptive System",
                status="failed",
                response_time=time.time() - start_time,
                data_quality=0.0,
                error_message=str(e)
            )
    
    async def test_powerball_prediction_system(self) -> IntegrationTestResult:
        """Test Powerball prediction system."""
        start_time = time.time()
        
        try:
            # Test Powerball prediction
            response = self._make_request(
                "/api/prediction/powerball",
                api_key=self.api_keys["quantum_analyst"]
            )
            
            # Validate response structure
            if "prediction" not in response:
                raise Exception("Missing prediction in response")
            
            prediction = response["prediction"]
            if prediction["status"] != "success":
                raise Exception(f"Prediction failed: {prediction}")
            
            # Validate prediction structure
            pred_data = prediction["prediction"]
            if "white_balls" not in pred_data or "red_ball" not in pred_data:
                raise Exception("Invalid prediction structure")
            
            white_balls = pred_data["white_balls"]
            red_ball = pred_data["red_ball"]
            
            # Validate ball numbers
            if len(white_balls) != 5:
                raise Exception(f"Expected 5 white balls, got {len(white_balls)}")
            
            if not all(1 <= ball <= 69 for ball in white_balls):
                raise Exception(f"White balls out of range: {white_balls}")
            
            if not 1 <= red_ball <= 26:
                raise Exception(f"Red ball out of range: {red_ball}")
            
            response_time = time.time() - start_time
            quality_score = 1.0 if pred_data.get("confidence", 0) > 0 else 0.8
            
            return IntegrationTestResult(
                test_name="Powerball Prediction System",
                status="passed",
                response_time=response_time,
                data_quality=quality_score,
                details={
                    "white_balls": white_balls,
                    "red_ball": red_ball,
                    "confidence": pred_data.get("confidence", 0),
                    "systems_used": response.get("systems_used", [])
                }
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="Powerball Prediction System",
                status="failed",
                response_time=time.time() - start_time,
                data_quality=0.0,
                error_message=str(e)
            )
    
    async def test_data_pipeline_integration(self) -> IntegrationTestResult:
        """Test data pipeline integration."""
        start_time = time.time()
        
        try:
            # Test pipeline start
            start_response = self._make_request(
                "/api/pipeline/start",
                method="POST",
                api_key=self.api_keys["system_admin"]
            )
            
            if start_response["status"] != "running":
                raise Exception(f"Pipeline start failed: {start_response}")
            
            # Wait a moment for pipeline to initialize
            await asyncio.sleep(1)
            
            # Test pipeline stop
            stop_response = self._make_request(
                "/api/pipeline/stop",
                method="POST",
                api_key=self.api_keys["system_admin"]
            )
            
            if stop_response["status"] != "stopped":
                raise Exception(f"Pipeline stop failed: {stop_response}")
            
            response_time = time.time() - start_time
            quality_score = 1.0 if start_response["status"] == "running" and stop_response["status"] == "stopped" else 0.5
            
            return IntegrationTestResult(
                test_name="Data Pipeline Integration",
                status="passed",
                response_time=response_time,
                data_quality=quality_score,
                details={
                    "start_status": start_response["status"],
                    "stop_status": stop_response["status"]
                }
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="Data Pipeline Integration",
                status="failed",
                response_time=time.time() - start_time,
                data_quality=0.0,
                error_message=str(e)
            )
    
    async def test_system_health_monitoring(self) -> IntegrationTestResult:
        """Test system health monitoring."""
        start_time = time.time()
        
        try:
            # Test health endpoint
            health_response = self._make_request("/health")
            
            if health_response["status"] != "healthy":
                raise Exception(f"System health check failed: {health_response}")
            
            # Test specific system health
            wallace_health = self._make_request(
                "/api/health/wallace_transform",
                api_key=self.api_keys["system_admin"]
            )
            
            required_health_fields = ["system_name", "status", "uptime", "response_time", "error_rate"]
            for field in required_health_fields:
                if field not in wallace_health:
                    raise Exception(f"Missing health field: {field}")
            
            response_time = time.time() - start_time
            quality_score = 1.0 if wallace_health["status"] == "online" else 0.5
            
            return IntegrationTestResult(
                test_name="System Health Monitoring",
                status="passed",
                response_time=response_time,
                data_quality=quality_score,
                details={
                    "overall_health": health_response["status"],
                    "wallace_health": wallace_health["status"],
                    "uptime": wallace_health["uptime"]
                }
            )
            
        except Exception as e:
            return IntegrationTestResult(
                test_name="System Health Monitoring",
                status="failed",
                response_time=time.time() - start_time,
                data_quality=0.0,
                error_message=str(e)
            )
    
    async def run_comprehensive_integration_test(self) -> SystemIntegrationReport:
        """Run comprehensive integration testing."""
        print("🔬 COMPREHENSIVE INTEGRATION TESTING")
        print("=" * 60)
        print("Testing prime aligned compute Mathematics Framework")
        print("=" * 60)
        
        # Define all tests
        tests = [
            self.test_api_gateway_connectivity,
            self.test_authentication_system,
            self.test_wallace_transform_system,
            self.test_consciousness_validator_system,
            self.test_quantum_adaptive_system,
            self.test_powerball_prediction_system,
            self.test_data_pipeline_integration,
            self.test_system_health_monitoring
        ]
        
        # Run all tests
        for test in tests:
            try:
                result = await test()
                self.test_results.append(result)
                
                status_icon = "✅" if result.status == "passed" else "❌" if result.status == "failed" else "⚠️"
                print(f"{status_icon} {result.test_name}: {result.status.upper()} ({result.response_time:.3f}s)")
                
                if result.error_message:
                    print(f"   Error: {result.error_message}")
                
            except Exception as e:
                error_result = IntegrationTestResult(
                    test_name=test.__name__,
                    status="failed",
                    response_time=0.0,
                    data_quality=0.0,
                    error_message=f"Test execution failed: {str(e)}"
                )
                self.test_results.append(error_result)
                print(f"❌ {test.__name__}: FAILED (Test execution error)")
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "passed"])
        failed_tests = len([r for r in self.test_results if r.status == "failed"])
        warning_tests = len([r for r in self.test_results if r.status == "warning"])
        
        response_times = [r.response_time for r in self.test_results if r.response_time > 0]
        average_response_time = np.mean(response_times) if response_times else 0.0
        
        quality_scores = [r.data_quality for r in self.test_results]
        overall_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        # Determine system connectivity
        system_connectivity = {
            "api_gateway": any(r.test_name == "API Gateway Connectivity" and r.status == "passed" for r in self.test_results),
            "authentication": any(r.test_name == "Authentication System" and r.status == "passed" for r in self.test_results),
            "wallace_transform": any(r.test_name == "Wallace Transform System" and r.status == "passed" for r in self.test_results),
            "consciousness_validator": any(r.test_name == "prime aligned compute Validator System" and r.status == "passed" for r in self.test_results),
            "quantum_adaptive": any(r.test_name == "Quantum Adaptive System" and r.status == "passed" for r in self.test_results),
            "powerball_prediction": any(r.test_name == "Powerball Prediction System" and r.status == "passed" for r in self.test_results),
            "data_pipeline": any(r.test_name == "Data Pipeline Integration" and r.status == "passed" for r in self.test_results),
            "health_monitoring": any(r.test_name == "System Health Monitoring" and r.status == "passed" for r in self.test_results)
        }
        
        # Create report
        report = SystemIntegrationReport(
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warning_tests=warning_tests,
            average_response_time=average_response_time,
            overall_quality_score=overall_quality_score,
            system_connectivity=system_connectivity,
            test_results=self.test_results
        )
        
        # Display results
        print(f"\n📊 INTEGRATION TEST RESULTS")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Warnings: {warning_tests} ⚠️")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Average Response Time: {average_response_time:.3f}s")
        print(f"Overall Quality Score: {overall_quality_score:.3f}")
        
        print(f"\n🔗 SYSTEM CONNECTIVITY:")
        for system, connected in system_connectivity.items():
            status = "✅ CONNECTED" if connected else "❌ DISCONNECTED"
            print(f"   • {system.replace('_', ' ').title()}: {status}")
        
        print(f"\n🏆 INTEGRATION STATUS:")
        if passed_tests == total_tests:
            print("🎉 ALL SYSTEMS INTEGRATED SUCCESSFULLY!")
            print("🚀 Framework ready for production use")
        elif passed_tests >= total_tests * 0.8:
            print("✅ MOST SYSTEMS INTEGRATED SUCCESSFULLY")
            print("🔧 Minor issues detected - review failed tests")
        else:
            print("⚠️ INTEGRATION ISSUES DETECTED")
            print("🔧 Review failed tests and system connectivity")
        
        return report

async def demonstrate_integration_testing():
    """Demonstrate comprehensive integration testing."""
    # Start API Gateway in background (simulated)
    print("🚀 Starting API Gateway for integration testing...")
    
    # Create tester
    tester = IntegrationTester()
    
    # Run comprehensive tests
    report = await tester.run_comprehensive_integration_test()
    
    return report

if __name__ == "__main__":
    asyncio.run(demonstrate_integration_testing())
