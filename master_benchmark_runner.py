#!/usr/bin/env python3
"""
🎯 MASTER BENCHMARK RUNNER
==========================
Orchestrates comprehensive testing of chAIos platform:
- GLUE & SuperGLUE Benchmarks
- Comprehensive Benchmark Suite
- Performance & Stress Testing
- System Health Monitoring
- Results Aggregation & Analysis
- Report Generation
"""

import requests
import json
import time
import sys
import subprocess
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterBenchmarkRunner:
    """Master benchmark runner that orchestrates all testing"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def check_api_health(self) -> bool:
        """Check if chAIos API is available and healthy"""
        print("🔍 Checking chAIos API health...")
        
        try:
            # Check basic connectivity
            response = requests.get(f"{self.api_url}/plugin/health", timeout=10)
            if response.status_code != 200:
                print(f"❌ API health check failed: HTTP {response.status_code}")
                return False
            
            # Check if plugins are available
            response = requests.get(f"{self.api_url}/plugin/catalog", 
                                  headers={"Authorization": "Bearer benchmark_token"}, 
                                  timeout=10)
            if response.status_code != 200:
                print(f"❌ Plugin catalog check failed: HTTP {response.status_code}")
                return False
            
            catalog = response.json()
            plugins = catalog.get("tools", [])
            if not plugins:
                print("❌ No plugins available")
                return False
            
            print(f"✅ API is healthy with {len(plugins)} plugins available")
            return True
            
        except Exception as e:
            print(f"❌ API health check failed: {e}")
            return False
    
    def run_glue_superglue_benchmarks(self) -> Dict[str, Any]:
        """Run GLUE and SuperGLUE benchmarks"""
        print("\n🏆 RUNNING GLUE & SUPERGLUE BENCHMARKS")
        print("=" * 50)
        
        try:
            # Import and run GLUE/SuperGLUE benchmarks
            from glue_superglue_benchmark import BenchmarkRunner
            
            runner = BenchmarkRunner(self.api_url)
            results = runner.run_all_benchmarks()
            
            print("✅ GLUE & SuperGLUE benchmarks completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"GLUE/SuperGLUE benchmarks failed: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        print("\n🎯 RUNNING COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 50)
        
        try:
            # Import and run comprehensive benchmarks
            from comprehensive_benchmark_suite import ComprehensiveBenchmarkRunner
            
            runner = ComprehensiveBenchmarkRunner(self.api_url)
            results = runner.run_all_benchmarks()
            
            print("✅ Comprehensive benchmarks completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive benchmarks failed: {e}")
            return {"error": str(e)}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and stress tests"""
        print("\n⚡ RUNNING PERFORMANCE & STRESS TESTS")
        print("=" * 50)
        
        try:
            # Import and run performance tests
            from performance_stress_test import PerformanceTestSuite
            
            test_suite = PerformanceTestSuite(self.api_url)
            results = test_suite.run_comprehensive_tests()
            
            print("✅ Performance tests completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            return {"error": str(e)}
    
    def run_system_health_check(self) -> Dict[str, Any]:
        """Run system health and monitoring checks"""
        print("\n🏥 RUNNING SYSTEM HEALTH CHECK")
        print("=" * 50)
        
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connectivity
            network_status = "unknown"
            try:
                response = requests.get("https://www.google.com", timeout=5)
                network_status = "connected" if response.status_code == 200 else "limited"
            except:
                network_status = "disconnected"
            
            # API response time
            api_response_time = 0
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_url}/plugin/health", timeout=10)
                api_response_time = time.time() - start_time
            except:
                api_response_time = -1
            
            health_data = {
                "system": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_usage_percent": (disk.used / disk.total) * 100,
                    "disk_free_gb": disk.free / (1024**3)
                },
                "network": {
                    "status": network_status,
                    "api_response_time": api_response_time
                },
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"   💻 CPU Usage: {cpu_percent:.1f}%")
            print(f"   🧠 Memory Usage: {memory.percent:.1f}% ({memory.available / (1024**3):.1f} GB available)")
            print(f"   💾 Disk Usage: {(disk.used / disk.total) * 100:.1f}% ({disk.free / (1024**3):.1f} GB free)")
            print(f"   🌐 Network: {network_status}")
            print(f"   ⚡ API Response Time: {api_response_time:.3f}s")
            
            print("✅ System health check completed successfully")
            return health_data
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"error": str(e)}
    
    def generate_master_report(self) -> Dict[str, Any]:
        """Generate comprehensive master report"""
        print("\n📊 GENERATING MASTER BENCHMARK REPORT")
        print("=" * 50)
        
        # Calculate overall execution time
        total_execution_time = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        
        # Aggregate results
        master_report = {
            "execution_summary": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_execution_time_seconds": total_execution_time,
                "api_url": self.api_url,
                "timestamp": datetime.now().isoformat()
            },
            "benchmark_results": self.results,
            "overall_assessment": self._calculate_overall_assessment()
        }
        
        # Print summary
        print(f"📊 MASTER BENCHMARK SUMMARY")
        print(f"   Total Execution Time: {total_execution_time:.1f} seconds")
        print(f"   API URL: {self.api_url}")
        print(f"   Tests Completed: {len([k for k, v in self.results.items() if 'error' not in v])}")
        print(f"   Tests Failed: {len([k for k, v in self.results.items() if 'error' in v])}")
        
        assessment = master_report["overall_assessment"]
        print(f"   Overall Assessment: {assessment['grade']}")
        print(f"   Performance Score: {assessment['performance_score']:.1f}/100")
        print(f"   Reliability Score: {assessment['reliability_score']:.1f}/100")
        print(f"   prime aligned compute Enhancement: {assessment['consciousness_enhancement']:.3f}x")
        
        return master_report
    
    def _calculate_overall_assessment(self) -> Dict[str, Any]:
        """Calculate overall assessment from all benchmark results"""
        scores = []
        consciousness_enhancements = []
        error_rates = []
        response_times = []
        
        # Extract metrics from different benchmark types
        for benchmark_type, results in self.results.items():
            if "error" in results:
                continue
                
            if benchmark_type == "glue_superglue":
                if "summary" in results:
                    summary = results["summary"]
                    if "average_accuracy" in summary:
                        scores.append(summary["average_accuracy"] * 100)
                    if "average_enhancement" in summary:
                        consciousness_enhancements.append(summary["average_enhancement"])
            
            elif benchmark_type == "comprehensive":
                if "summary" in results:
                    summary = results["summary"]
                    if "average_accuracy" in summary:
                        scores.append(summary["average_accuracy"] * 100)
                    if "average_enhancement" in summary:
                        consciousness_enhancements.append(summary["average_enhancement"])
                    if "average_error_rate" in summary:
                        error_rates.append(summary["average_error_rate"] * 100)
            
            elif benchmark_type == "performance":
                if "summary" in results:
                    summary = results["summary"]
                    if "average_error_rate" in summary:
                        error_rates.append(summary["average_error_rate"] * 100)
                    if "average_response_time" in summary:
                        response_times.append(summary["average_response_time"])
        
        # Calculate overall scores
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_consciousness_enhancement = sum(consciousness_enhancements) / len(consciousness_enhancements) if consciousness_enhancements else 1.0
        avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Performance score (0-100)
        performance_score = avg_score
        
        # Reliability score (0-100, based on error rate)
        reliability_score = max(0, 100 - avg_error_rate)
        
        # Overall grade
        if performance_score >= 80 and reliability_score >= 90:
            grade = "🌟 EXCELLENT - Production Ready"
        elif performance_score >= 60 and reliability_score >= 80:
            grade = "✅ GOOD - Strong Performance"
        elif performance_score >= 40 and reliability_score >= 60:
            grade = "⚠️ MODERATE - Needs Improvement"
        else:
            grade = "❌ POOR - Significant Issues"
        
        return {
            "grade": grade,
            "performance_score": performance_score,
            "reliability_score": reliability_score,
            "consciousness_enhancement": avg_consciousness_enhancement,
            "average_error_rate": avg_error_rate,
            "average_response_time": avg_response_time,
            "metrics_analyzed": len(scores)
        }
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"master_benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filename
    
    def run_all_benchmarks(self, include_performance: bool = True, include_health: bool = True) -> Dict[str, Any]:
        """Run all benchmark suites"""
        print("🚀 MASTER BENCHMARK RUNNER")
        print("=" * 70)
        print("Comprehensive testing of chAIos platform with traditional AI benchmarks")
        print()
        
        self.start_time = datetime.now()
        
        # Check API health first
        if not self.check_api_health():
            print("❌ API health check failed. Cannot proceed with benchmarks.")
            return {"error": "API health check failed"}
        
        # Run GLUE & SuperGLUE benchmarks
        print("\n" + "=" * 70)
        self.results["glue_superglue"] = self.run_glue_superglue_benchmarks()
        
        # Run comprehensive benchmarks
        print("\n" + "=" * 70)
        self.results["comprehensive"] = self.run_comprehensive_benchmarks()
        
        # Run performance tests (optional)
        if include_performance:
            print("\n" + "=" * 70)
            self.results["performance"] = self.run_performance_tests()
        
        # Run system health check (optional)
        if include_health:
            print("\n" + "=" * 70)
            self.results["system_health"] = self.run_system_health_check()
        
        self.end_time = datetime.now()
        
        # Generate master report
        print("\n" + "=" * 70)
        master_report = self.generate_master_report()
        
        # Save results
        filename = self.save_results(master_report)
        
        print(f"\n💾 Master benchmark results saved to: {filename}")
        print("🎉 All benchmark testing complete!")
        print("🏆 chAIos platform has been thoroughly tested with traditional AI benchmarks!")
        
        return master_report

def main():
    """Main entry point for master benchmark runner"""
    parser = argparse.ArgumentParser(description="Master Benchmark Runner for chAIos Platform")
    parser.add_argument("--api-url", default="http://localhost:8000", help="chAIos API URL")
    parser.add_argument("--no-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--no-health", action="store_true", help="Skip system health check")
    parser.add_argument("--output", help="Output filename for results")
    
    args = parser.parse_args()
    
    # Create and run master benchmark runner
    runner = MasterBenchmarkRunner(args.api_url)
    
    try:
        results = runner.run_all_benchmarks(
            include_performance=not args.no_performance,
            include_health=not args.no_health
        )
        
        if args.output:
            runner.save_results(results, args.output)
            print(f"Results also saved to: {args.output}")
        
        # Exit with appropriate code
        if "error" in results:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Master benchmark runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
