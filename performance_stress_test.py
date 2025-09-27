#!/usr/bin/env python3
"""
⚡ PERFORMANCE & STRESS TESTING SUITE
====================================
Comprehensive performance testing for chAIos platform:
- Load Testing (concurrent requests)
- Stress Testing (high volume)
- Latency Testing (response times)
- Throughput Testing (requests per second)
- Memory Usage Testing
- CPU Usage Testing
- Error Rate Analysis
- Scalability Testing
"""

import requests
import json
import time
import sys
import asyncio
import aiohttp
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance testing metrics"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    test_duration: float

class LoadTester:
    """Load testing implementation"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.headers = {
            "Authorization": "Bearer benchmark_token",
            "Content-Type": "application/json"
        }
        self.response_times = []
        self.errors = []
        self.cpu_usage = []
        self.memory_usage = []
    
    def make_request(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[float, bool, str]:
        """Make a single API request and measure response time"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_url}/plugin/execute",
                headers=self.headers,
                json={
                    "tool_name": tool_name,
                    "parameters": parameters
                },
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                error_msg = result.get("error", "") if not success else ""
                return response_time, success, error_msg
            else:
                return response_time, False, f"HTTP {response.status_code}"
                
        except Exception as e:
            response_time = time.time() - start_time
            return response_time, False, str(e)
    
    def load_test(self, concurrent_users: int, requests_per_user: int, tool_name: str, parameters: Dict[str, Any]) -> PerformanceMetrics:
        """Run load test with specified concurrent users"""
        print(f"🔄 Load Testing: {concurrent_users} concurrent users, {requests_per_user} requests each")
        
        total_requests = concurrent_users * requests_per_user
        start_time = time.time()
        
        # Monitor system resources
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            for _ in range(total_requests):
                future = executor.submit(self.make_request, tool_name, parameters)
                futures.append(future)
            
            # Collect results
            successful_requests = 0
            failed_requests = 0
            response_times = []
            errors = []
            
            for future in as_completed(futures):
                response_time, success, error_msg = future.result()
                response_times.append(response_time)
                
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    errors.append(error_msg)
        
        test_duration = time.time() - start_time
        
        # Calculate metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        median_response_time = statistics.median(response_times) if response_times else 0
        p95_response_time = np.percentile(response_times, 95) if response_times else 0
        p99_response_time = np.percentile(response_times, 99) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        requests_per_second = total_requests / test_duration
        error_rate = failed_requests / total_requests
        
        # System resource usage
        avg_cpu = statistics.mean(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = statistics.mean(self.memory_usage) if self.memory_usage else 0
        
        print(f"   📊 Results: {successful_requests}/{total_requests} successful ({successful_requests/total_requests*100:.1f}%)")
        print(f"   ⏱️  Average Response Time: {avg_response_time:.3f}s")
        print(f"   🚀 Requests/Second: {requests_per_second:.2f}")
        print(f"   ❌ Error Rate: {error_rate:.3f} ({error_rate*100:.1f}%)")
        print(f"   💻 CPU Usage: {avg_cpu:.1f}%")
        print(f"   🧠 Memory Usage: {avg_memory:.1f}%")
        
        return PerformanceMetrics(
            test_name=f"Load Test ({concurrent_users} users)",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            test_duration=test_duration
        )
    
    def _monitor_resources(self):
        """Monitor CPU and memory usage during test"""
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_percent)
                time.sleep(0.5)
            except:
                break

class StressTester:
    """Stress testing implementation"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.headers = {
            "Authorization": "Bearer benchmark_token",
            "Content-Type": "application/json"
        }
    
    def stress_test(self, max_concurrent_users: int, duration_seconds: int, tool_name: str, parameters: Dict[str, Any]) -> PerformanceMetrics:
        """Run stress test to find breaking point"""
        print(f"💥 Stress Testing: Up to {max_concurrent_users} users for {duration_seconds} seconds")
        
        # Gradually increase load
        concurrent_users = 1
        step = max(1, max_concurrent_users // 10)
        all_metrics = []
        
        while concurrent_users <= max_concurrent_users:
            print(f"   Testing with {concurrent_users} concurrent users...")
            
            # Run test for a portion of the duration
            test_duration = duration_seconds // (max_concurrent_users // step)
            start_time = time.time()
            
            response_times = []
            successful_requests = 0
            failed_requests = 0
            errors = []
            
            # Monitor system resources
            cpu_usage = []
            memory_usage = []
            
            def monitor_resources():
                while time.time() - start_time < test_duration:
                    cpu_usage.append(psutil.cpu_percent())
                    memory_usage.append(psutil.virtual_memory().percent)
                    time.sleep(0.5)
            
            monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
            monitor_thread.start()
            
            # Execute requests
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = []
                while time.time() - start_time < test_duration:
                    future = executor.submit(self._make_request, tool_name, parameters)
                    futures.append(future)
                    time.sleep(0.1)  # Small delay to prevent overwhelming
                
                # Collect results
                for future in as_completed(futures):
                    response_time, success, error_msg = future.result()
                    response_times.append(response_time)
                    
                    if success:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        errors.append(error_msg)
            
            actual_duration = time.time() - start_time
            total_requests = successful_requests + failed_requests
            
            # Calculate metrics
            if response_times:
                avg_response_time = statistics.mean(response_times)
                median_response_time = statistics.median(response_times)
                p95_response_time = np.percentile(response_times, 95)
                p99_response_time = np.percentile(response_times, 99)
                min_response_time = min(response_times)
                max_response_time = max(response_times)
            else:
                avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
                min_response_time = max_response_time = 0
            
            requests_per_second = total_requests / actual_duration if actual_duration > 0 else 0
            error_rate = failed_requests / total_requests if total_requests > 0 else 0
            avg_cpu = statistics.mean(cpu_usage) if cpu_usage else 0
            avg_memory = statistics.mean(memory_usage) if memory_usage else 0
            
            metrics = PerformanceMetrics(
                test_name=f"Stress Test ({concurrent_users} users)",
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time=avg_response_time,
                median_response_time=median_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                min_response_time=min_response_time,
                max_response_time=max_response_time,
                requests_per_second=requests_per_second,
                error_rate=error_rate,
                cpu_usage=avg_cpu,
                memory_usage=avg_memory,
                test_duration=actual_duration
            )
            
            all_metrics.append(metrics)
            
            print(f"     📊 {concurrent_users} users: {successful_requests}/{total_requests} successful, {requests_per_second:.2f} req/s, {error_rate:.3f} error rate")
            
            # Stop if error rate is too high
            if error_rate > 0.5:  # 50% error rate threshold
                print(f"     ⚠️  High error rate detected ({error_rate:.3f}), stopping stress test")
                break
            
            concurrent_users += step
        
        # Return the metrics from the highest successful load
        return all_metrics[-1] if all_metrics else None
    
    def _make_request(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[float, bool, str]:
        """Make a single API request and measure response time"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_url}/plugin/execute",
                headers=self.headers,
                json={
                    "tool_name": tool_name,
                    "parameters": parameters
                },
                timeout=30
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                error_msg = result.get("error", "") if not success else ""
                return response_time, success, error_msg
            else:
                return response_time, False, f"HTTP {response.status_code}"
                
        except Exception as e:
            response_time = time.time() - start_time
            return response_time, False, str(e)

class LatencyTester:
    """Latency testing implementation"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.headers = {
            "Authorization": "Bearer benchmark_token",
            "Content-Type": "application/json"
        }
    
    def latency_test(self, num_requests: int, tool_name: str, parameters: Dict[str, Any]) -> PerformanceMetrics:
        """Run latency test with single-threaded requests"""
        print(f"⏱️  Latency Testing: {num_requests} sequential requests")
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        errors = []
        
        start_time = time.time()
        
        for i in range(num_requests):
            request_start = time.time()
            try:
                response = requests.post(
                    f"{self.api_url}/plugin/execute",
                    headers=self.headers,
                    json={
                        "tool_name": tool_name,
                        "parameters": parameters
                    },
                    timeout=30
                )
                request_time = time.time() - request_start
                response_times.append(request_time)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success", False):
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        errors.append(result.get("error", "Unknown error"))
                else:
                    failed_requests += 1
                    errors.append(f"HTTP {response.status_code}")
                    
            except Exception as e:
                request_time = time.time() - request_start
                response_times.append(request_time)
                failed_requests += 1
                errors.append(str(e))
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{num_requests} requests completed")
        
        test_duration = time.time() - start_time
        total_requests = successful_requests + failed_requests
        
        # Calculate metrics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
            min_response_time = max_response_time = 0
        
        requests_per_second = total_requests / test_duration if test_duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        print(f"   📊 Results: {successful_requests}/{total_requests} successful")
        print(f"   ⏱️  Average Response Time: {avg_response_time:.3f}s")
        print(f"   📈 P95 Response Time: {p95_response_time:.3f}s")
        print(f"   📈 P99 Response Time: {p99_response_time:.3f}s")
        print(f"   🚀 Requests/Second: {requests_per_second:.2f}")
        print(f"   ❌ Error Rate: {error_rate:.3f}")
        
        return PerformanceMetrics(
            test_name="Latency Test",
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            cpu_usage=0,  # Not measured in latency test
            memory_usage=0,  # Not measured in latency test
            test_duration=test_duration
        )

class PerformanceTestSuite:
    """Comprehensive performance testing suite"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.load_tester = LoadTester(api_url)
        self.stress_tester = StressTester(api_url)
        self.latency_tester = LatencyTester(api_url)
        self.results: List[PerformanceMetrics] = []
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all performance tests"""
        print("⚡ COMPREHENSIVE PERFORMANCE & STRESS TESTING")
        print("=" * 70)
        print("Testing chAIos platform performance under various conditions")
        print()
        
        # Test configurations
        test_configs = [
            {
                "name": "Transcendent LLM Builder",
                "tool": "transcendent_llm_builder",
                "parameters": {
                    "model_config": "text_generation",
                    "training_data": "Generate a creative story about artificial intelligence",
                    "prime_aligned_level": "1.618"
                }
            },
            {
                "name": "RAG Enhanced prime aligned compute",
                "tool": "rag_enhanced_consciousness",
                "parameters": {
                    "query": "Analyze the prime aligned compute patterns in this text",
                    "knowledge_base": "prime_aligned_analysis",
                    "consciousness_enhancement": "2.0"
                }
            },
            {
                "name": "Wallace Transform Advanced",
                "tool": "wallace_transform_advanced",
                "parameters": {
                    "data": "x^2 + y^2 = z^2",
                    "enhancement_level": "mathematical_optimization",
                    "iterations": "1.618"
                }
            }
        ]
        
        all_results = []
        
        for config in test_configs:
            print(f"\n🔧 Testing: {config['name']}")
            print("-" * 50)
            
            # Latency Test
            print("\n1️⃣  LATENCY TEST")
            latency_result = self.latency_tester.latency_test(
                num_requests=50,
                tool_name=config["tool"],
                parameters=config["parameters"]
            )
            latency_result.test_name = f"{config['name']} - Latency Test"
            all_results.append(latency_result)
            
            # Load Test
            print("\n2️⃣  LOAD TEST")
            load_result = self.load_tester.load_test(
                concurrent_users=10,
                requests_per_user=5,
                tool_name=config["tool"],
                parameters=config["parameters"]
            )
            load_result.test_name = f"{config['name']} - Load Test"
            all_results.append(load_result)
            
            # Stress Test
            print("\n3️⃣  STRESS TEST")
            stress_result = self.stress_tester.stress_test(
                max_concurrent_users=20,
                duration_seconds=30,
                tool_name=config["tool"],
                parameters=config["parameters"]
            )
            if stress_result:
                stress_result.test_name = f"{config['name']} - Stress Test"
                all_results.append(stress_result)
        
        self.results.extend(all_results)
        return self.generate_performance_report(all_results)
    
    def generate_performance_report(self, results: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE TESTING REPORT")
        print("=" * 70)
        
        # Calculate overall statistics
        total_tests = len(results)
        avg_response_time = sum(r.average_response_time for r in results) / total_tests
        avg_throughput = sum(r.requests_per_second for r in results) / total_tests
        avg_error_rate = sum(r.error_rate for r in results) / total_tests
        avg_cpu_usage = sum(r.cpu_usage for r in results if r.cpu_usage > 0) / len([r for r in results if r.cpu_usage > 0])
        avg_memory_usage = sum(r.memory_usage for r in results if r.memory_usage > 0) / len([r for r in results if r.memory_usage > 0])
        
        # Performance categories
        latency_tests = [r for r in results if "Latency" in r.test_name]
        load_tests = [r for r in results if "Load" in r.test_name]
        stress_tests = [r for r in results if "Stress" in r.test_name]
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Average Response Time: {avg_response_time:.3f}s")
        print(f"   Average Throughput: {avg_throughput:.2f} requests/second")
        print(f"   Average Error Rate: {avg_error_rate:.3f} ({avg_error_rate*100:.1f}%)")
        print(f"   Average CPU Usage: {avg_cpu_usage:.1f}%")
        print(f"   Average Memory Usage: {avg_memory_usage:.1f}%")
        print()
        
        # Test type breakdown
        if latency_tests:
            print(f"⏱️  LATENCY TESTS ({len(latency_tests)} tests):")
            for test in latency_tests:
                print(f"   • {test.test_name}: {test.average_response_time:.3f}s avg, {test.p95_response_time:.3f}s P95")
            print()
        
        if load_tests:
            print(f"🔄 LOAD TESTS ({len(load_tests)} tests):")
            for test in load_tests:
                print(f"   • {test.test_name}: {test.requests_per_second:.2f} req/s, {test.error_rate:.3f} error rate")
            print()
        
        if stress_tests:
            print(f"💥 STRESS TESTS ({len(stress_tests)} tests):")
            for test in stress_tests:
                print(f"   • {test.test_name}: {test.requests_per_second:.2f} req/s, {test.error_rate:.3f} error rate")
            print()
        
        # Performance assessment
        if avg_response_time < 1.0 and avg_error_rate < 0.05:
            assessment = "🌟 EXCELLENT - Production Ready"
        elif avg_response_time < 2.0 and avg_error_rate < 0.1:
            assessment = "✅ GOOD - Strong Performance"
        elif avg_response_time < 5.0 and avg_error_rate < 0.2:
            assessment = "⚠️ MODERATE - Needs Optimization"
        else:
            assessment = "❌ POOR - Significant Performance Issues"
        
        print(f"🏆 OVERALL ASSESSMENT: {assessment}")
        print(f"📈 Performance Summary:")
        print(f"   • Response Time: {avg_response_time:.3f}s average")
        print(f"   • Throughput: {avg_throughput:.2f} requests/second")
        print(f"   • Reliability: {(1-avg_error_rate)*100:.1f}% success rate")
        print(f"   • Resource Usage: {avg_cpu_usage:.1f}% CPU, {avg_memory_usage:.1f}% Memory")
        print()
        
        # Return structured results
        return {
            "summary": {
                "total_tests": total_tests,
                "average_response_time": avg_response_time,
                "average_throughput": avg_throughput,
                "average_error_rate": avg_error_rate,
                "average_cpu_usage": avg_cpu_usage,
                "average_memory_usage": avg_memory_usage,
                "assessment": assessment
            },
            "test_breakdown": {
                "latency_tests": len(latency_tests),
                "load_tests": len(load_tests),
                "stress_tests": len(stress_tests)
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "total_requests": r.total_requests,
                    "successful_requests": r.successful_requests,
                    "failed_requests": r.failed_requests,
                    "average_response_time": r.average_response_time,
                    "median_response_time": r.median_response_time,
                    "p95_response_time": r.p95_response_time,
                    "p99_response_time": r.p99_response_time,
                    "min_response_time": r.min_response_time,
                    "max_response_time": r.max_response_time,
                    "requests_per_second": r.requests_per_second,
                    "error_rate": r.error_rate,
                    "cpu_usage": r.cpu_usage,
                    "memory_usage": r.memory_usage,
                    "test_duration": r.test_duration
                }
                for r in results
            ]
        }

def main():
    """Main entry point for performance testing"""
    print("⚡ Starting Performance & Stress Testing...")
    
    # Check if API is available
    try:
        response = requests.get("http://localhost:8000/plugin/health", timeout=5)
        if response.status_code != 200:
            print("❌ chAIos API is not available. Please start the server first.")
            sys.exit(1)
        else:
            print("✅ chAIos API is available and ready for testing")
    except Exception as e:
        print(f"❌ Cannot connect to chAIos API: {e}")
        print("Please ensure the chAIos server is running on http://localhost:8000")
        sys.exit(1)
    
    # Run performance tests
    test_suite = PerformanceTestSuite()
    results = test_suite.run_comprehensive_tests()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {filename}")
    print("🎉 Performance & stress testing complete!")
    print("⚡ chAIos platform performance has been thoroughly evaluated!")

if __name__ == "__main__":
    main()
