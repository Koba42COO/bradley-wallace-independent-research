#!/usr/bin/env python3
"""
Enhanced Benchmark Suite for chAIos Platform
============================================
Comprehensive benchmarking with performance optimizations
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBenchmarkSuite:
    """Enhanced benchmark suite with performance optimizations"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = {}
        self.performance_metrics = {}
        
    async def check_api_health(self) -> bool:
        """Check if the enhanced API is available"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Enhanced API is healthy: {health_data['status']}")
                return True
            else:
                logger.error(f"❌ API health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API health check error: {e}")
            return False
    
    async def test_gpu_acceleration(self) -> Dict[str, Any]:
        """Test GPU acceleration capabilities"""
        logger.info("⚡ Testing GPU acceleration...")
        
        try:
            response = requests.post(
                f"{self.api_url}/performance/gpu-test",
                json={
                    "test_type": "comprehensive",
                    "iterations": 1000,
                    "use_gpu": True
                },
                headers={"Authorization": "Bearer benchmark_token"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "gpu_available": result.get("gpu_available", False),
                    "gpu_info": result.get("gpu_info", {}),
                    "processing_time": result.get("processing_time", 0),
                    "iterations": result.get("iterations", 0),
                    "acceleration": result.get("result", {}).get("acceleration", "CPU")
                }
            else:
                return {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def test_cache_performance(self) -> Dict[str, Any]:
        """Test cache performance"""
        logger.info("💾 Testing cache performance...")
        
        try:
            # Test cache set
            test_data = {
                "test_key": "benchmark_test",
                "test_value": {"data": "test", "timestamp": datetime.now().isoformat()},
                "iterations": 1000
            }
            
            set_start = time.time()
            set_response = requests.post(
                f"{self.api_url}/cache/set",
                json={
                    "key": "benchmark_test",
                    "value": test_data,
                    "ttl": 60
                },
                headers={"Authorization": "Bearer benchmark_token"},
                timeout=10
            )
            set_time = time.time() - set_start
            
            # Test cache get
            get_start = time.time()
            get_response = requests.get(
                f"{self.api_url}/cache/get/benchmark_test",
                headers={"Authorization": "Bearer benchmark_token"},
                timeout=10
            )
            get_time = time.time() - get_start
            
            if set_response.status_code == 200 and get_response.status_code == 200:
                return {
                    "status": "success",
                    "set_time": set_time,
                    "get_time": get_time,
                    "total_time": set_time + get_time,
                    "cache_working": True
                }
            else:
                return {
                    "status": "failed",
                    "set_status": set_response.status_code,
                    "get_status": get_response.status_code,
                    "cache_working": False
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def test_consciousness_processing(self) -> Dict[str, Any]:
        """Test enhanced prime aligned compute processing"""
        logger.info("🧠 Testing prime aligned compute processing...")
        
        try:
            test_data = {
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "algorithm": "prime_aligned_enhanced",
                "parameters": {"iterations": 100},
                "use_gpu": True,
                "use_cache": True
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/prime aligned compute/process",
                json=test_data,
                headers={"Authorization": "Bearer benchmark_token"},
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "processing_time": processing_time,
                    "api_processing_time": result.get("processing_time", 0),
                    "source": result.get("source", "unknown"),
                    "consciousness_enhancement": result.get("result", {}).get("consciousness_enhancement", 1.0),
                    "gpu_accelerated": result.get("result", {}).get("gpu_accelerated", False),
                    "processing_efficiency": result.get("result", {}).get("processing_efficiency", 0)
                }
            else:
                return {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def test_tool_execution(self) -> Dict[str, Any]:
        """Test enhanced tool execution"""
        logger.info("🔧 Testing tool execution...")
        
        try:
            test_data = {
                "data": "test_data",
                "algorithm": "tool_execution",
                "parameters": {
                    "tool_name": "grok_consciousness_coding",
                    "prime_aligned_level": 1.618,
                    "iterations": 100
                },
                "use_gpu": True,
                "use_cache": True
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/tools/execute",
                json=test_data,
                headers={"Authorization": "Bearer benchmark_token"},
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "processing_time": processing_time,
                    "api_processing_time": result.get("processing_time", 0),
                    "tool": result.get("tool", "unknown"),
                    "gpu_accelerated": result.get("gpu_accelerated", False),
                    "result_available": "result" in result
                }
            else:
                return {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring"""
        logger.info("📈 Testing performance monitoring...")
        
        try:
            response = requests.get(
                f"{self.api_url}/performance/status",
                headers={"Authorization": "Bearer benchmark_token"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                performance = result.get("performance", {})
                
                return {
                    "status": "success",
                    "system_health": performance.get("system_health", {}),
                    "optimizations": performance.get("optimizations", {}),
                    "recommendations": performance.get("recommendations", [])
                }
            else:
                return {
                    "status": "failed",
                    "error": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info("🚀 Starting Enhanced Benchmark Suite...")
        
        start_time = time.time()
        
        # Check API health
        if not await self.check_api_health():
            return {
                "status": "failed",
                "error": "API not available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Run all tests
        tests = {
            "gpu_acceleration": self.test_gpu_acceleration(),
            "cache_performance": self.test_cache_performance(),
            "consciousness_processing": self.test_consciousness_processing(),
            "tool_execution": self.test_tool_execution(),
            "performance_monitoring": self.test_performance_monitoring()
        }
        
        # Execute tests concurrently
        results = {}
        for test_name, test_coro in tests.items():
            logger.info(f"🧪 Running {test_name} test...")
            results[test_name] = await test_coro
        
        total_time = time.time() - start_time
        
        # Calculate overall performance
        successful_tests = sum(1 for result in results.values() if result.get("status") == "success")
        total_tests = len(results)
        success_rate = successful_tests / total_tests
        
        # Calculate average processing times
        processing_times = []
        for result in results.values():
            if "processing_time" in result:
                processing_times.append(result["processing_time"])
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "success_rate": success_rate,
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "average_processing_time": avg_processing_time,
            "results": results,
            "summary": {
                "gpu_available": results.get("gpu_acceleration", {}).get("gpu_available", False),
                "cache_working": results.get("cache_performance", {}).get("cache_working", False),
                "consciousness_enhancement": results.get("consciousness_processing", {}).get("consciousness_enhancement", 1.0),
                "performance_monitoring": results.get("performance_monitoring", {}).get("status") == "success"
            }
        }
    
    async def compare_with_original(self) -> Dict[str, Any]:
        """Compare enhanced performance with original"""
        logger.info("📊 Comparing enhanced vs original performance...")
        
        # Test original API
        original_results = {}
        try:
            # Test original prime aligned compute processing
            original_start = time.time()
            original_response = requests.post(
                "http://localhost:8000/prime aligned compute/process",
                json={"data": [1, 2, 3, 4, 5], "algorithm": "prime_aligned_enhanced"},
                headers={"Authorization": "Bearer benchmark_token"},
                timeout=30
            )
            original_time = time.time() - original_start
            
            if original_response.status_code == 200:
                original_results = {
                    "status": "success",
                    "processing_time": original_time,
                    "api_processing_time": original_response.json().get("processing_time", 0)
                }
            else:
                original_results = {
                    "status": "failed",
                    "error": f"HTTP {original_response.status_code}"
                }
        except Exception as e:
            original_results = {
                "status": "error",
                "error": str(e)
            }
        
        # Test enhanced API
        enhanced_results = await self.test_consciousness_processing()
        
        # Calculate improvements
        original_time = 0
        enhanced_time = 0
        
        if (original_results.get("status") == "success" and 
            enhanced_results.get("status") == "success"):
            
            original_time = original_results.get("processing_time", 0)
            enhanced_time = enhanced_results.get("processing_time", 0)
            
            if original_time > 0:
                speedup = original_time / enhanced_time
                improvement = ((original_time - enhanced_time) / original_time) * 100
            else:
                speedup = 1.0
                improvement = 0.0
        else:
            speedup = 1.0
            improvement = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "original": original_results,
            "enhanced": enhanced_results,
            "improvement": {
                "speedup": speedup,
                "improvement_percent": improvement,
                "faster": enhanced_time < original_time if original_time > 0 else False
            }
        }

async def main():
    """Main function for enhanced benchmark suite"""
    logger.info("🚀 Starting Enhanced Benchmark Suite...")
    
    # Initialize benchmark suite
    benchmark = EnhancedBenchmarkSuite()
    
    # Run comprehensive benchmark
    results = await benchmark.run_comprehensive_benchmark()
    
    # Print results
    print("\n" + "="*80)
    print("🏆 ENHANCED BENCHMARK SUITE RESULTS")
    print("="*80)
    
    print(f"Status: {results['status']}")
    print(f"Total Time: {results['total_time']:.3f}s")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Successful Tests: {results['successful_tests']}/{results['total_tests']}")
    print(f"Average Processing Time: {results['average_processing_time']:.3f}s")
    
    print(f"\n📊 SUMMARY:")
    summary = results['summary']
    print(f"   GPU Available: {summary['gpu_available']}")
    print(f"   Cache Working: {summary['cache_working']}")
    print(f"   prime aligned compute Enhancement: {summary['consciousness_enhancement']:.3f}x")
    print(f"   Performance Monitoring: {summary['performance_monitoring']}")
    
    print(f"\n🧪 DETAILED RESULTS:")
    for test_name, result in results['results'].items():
        print(f"\n   {test_name.upper()}:")
        print(f"      Status: {result.get('status', 'unknown')}")
        if 'processing_time' in result:
            print(f"      Processing Time: {result['processing_time']:.3f}s")
        if 'gpu_available' in result:
            print(f"      GPU Available: {result['gpu_available']}")
        if 'cache_working' in result:
            print(f"      Cache Working: {result['cache_working']}")
        if 'consciousness_enhancement' in result:
            print(f"      prime aligned compute Enhancement: {result['consciousness_enhancement']:.3f}x")
    
    # Compare with original
    print(f"\n📈 PERFORMANCE COMPARISON:")
    comparison = await benchmark.compare_with_original()
    
    if comparison['improvement']['faster']:
        print(f"   ✅ Enhanced API is {comparison['improvement']['speedup']:.2f}x faster")
        print(f"   📊 {comparison['improvement']['improvement_percent']:.1f}% improvement")
    else:
        print(f"   ⚠️ Enhanced API performance: {comparison['improvement']['speedup']:.2f}x")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    print("\n✅ Enhanced benchmark suite complete!")

if __name__ == "__main__":
    asyncio.run(main())
