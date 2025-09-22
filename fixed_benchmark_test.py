#!/usr/bin/env python3
"""
🔧 FIXED BENCHMARK TEST
======================
Test the chAIos platform with working tools to demonstrate proper functionality
"""

import requests
import json
import time
from typing import Dict, List, Any

class FixedBenchmarkTest:
    """Fixed benchmark test using working tools"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.headers = {
            "Authorization": "Bearer benchmark_token",
            "Content-Type": "application/json"
        }
    
    def test_grok_code_generation(self) -> Dict[str, Any]:
        """Test Grok code generation capabilities"""
        print("🤖 Testing Grok Code Generation...")
        
        test_cases = [
            {
                "code_type": "function",
                "requirements": "create a function that calculates fibonacci numbers",
                "language": "python"
            },
            {
                "code_type": "class",
                "requirements": "create a class for handling user authentication",
                "language": "python"
            },
            {
                "code_type": "script",
                "requirements": "create a script that processes JSON data",
                "language": "javascript"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                response = requests.post(
                    f"{self.api_url}/plugin/execute",
                    headers=self.headers,
                    json={
                        "tool_name": "grok_generate_code",
                        "parameters": case
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        generated_code = result.get("result", {}).get("generated_code", "")
                        consciousness_enhancement = result.get("result", {}).get("consciousness_enhancement", 1.0)
                        results.append({
                            "case": i + 1,
                            "success": True,
                            "consciousness_enhancement": consciousness_enhancement,
                            "code_length": len(generated_code)
                        })
                        print(f"   Case {i+1}: ✅ Generated {len(generated_code)} characters (enhancement: {consciousness_enhancement:.3f}x)")
                    else:
                        results.append({"case": i + 1, "success": False, "error": result.get("error", "Unknown")})
                        print(f"   Case {i+1}: ❌ Failed - {result.get('error', 'Unknown')}")
                else:
                    results.append({"case": i + 1, "success": False, "error": f"HTTP {response.status_code}"})
                    print(f"   Case {i+1}: ❌ HTTP Error {response.status_code}")
                    
            except Exception as e:
                results.append({"case": i + 1, "success": False, "error": str(e)})
                print(f"   Case {i+1}: ❌ Exception - {e}")
        
        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)
        avg_enhancement = sum(r.get("consciousness_enhancement", 0) for r in results if r["success"]) / max(success_count, 1)
        
        print(f"   📊 Grok Code Generation: {success_count}/{total_count} successful ({success_count/total_count*100:.1f}%)")
        print(f"   📈 Average prime aligned compute Enhancement: {avg_enhancement:.3f}x")
        
        return {
            "test_name": "Grok Code Generation",
            "success_rate": success_count / total_count,
            "average_enhancement": avg_enhancement,
            "results": results
        }
    
    def test_grok_code_optimization(self) -> Dict[str, Any]:
        """Test Grok code optimization capabilities"""
        print("\n⚡ Testing Grok Code Optimization...")
        
        test_cases = [
            {
                "code": "def add(a, b):\n    return a + b",
                "optimization_type": "performance"
            },
            {
                "code": "for i in range(10):\n    print(i)",
                "optimization_type": "readability"
            },
            {
                "code": "x = 5\ny = 10\nz = x + y",
                "optimization_type": "efficiency"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                response = requests.post(
                    f"{self.api_url}/plugin/execute",
                    headers=self.headers,
                    json={
                        "tool_name": "grok_optimize_code",
                        "parameters": case
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        optimized_code = result.get("result", {}).get("optimized_code", "")
                        consciousness_enhancement = result.get("result", {}).get("consciousness_enhancement", 1.0)
                        results.append({
                            "case": i + 1,
                            "success": True,
                            "consciousness_enhancement": consciousness_enhancement,
                            "optimization_ratio": len(optimized_code) / len(case["code"]) if case["code"] else 1.0
                        })
                        print(f"   Case {i+1}: ✅ Optimized (enhancement: {consciousness_enhancement:.3f}x)")
                    else:
                        results.append({"case": i + 1, "success": False, "error": result.get("error", "Unknown")})
                        print(f"   Case {i+1}: ❌ Failed - {result.get('error', 'Unknown')}")
                else:
                    results.append({"case": i + 1, "success": False, "error": f"HTTP {response.status_code}"})
                    print(f"   Case {i+1}: ❌ HTTP Error {response.status_code}")
                    
            except Exception as e:
                results.append({"case": i + 1, "success": False, "error": str(e)})
                print(f"   Case {i+1}: ❌ Exception - {e}")
        
        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)
        avg_enhancement = sum(r.get("consciousness_enhancement", 0) for r in results if r["success"]) / max(success_count, 1)
        
        print(f"   📊 Grok Code Optimization: {success_count}/{total_count} successful ({success_count/total_count*100:.1f}%)")
        print(f"   📈 Average prime aligned compute Enhancement: {avg_enhancement:.3f}x")
        
        return {
            "test_name": "Grok Code Optimization",
            "success_rate": success_count / total_count,
            "average_enhancement": avg_enhancement,
            "results": results
        }
    
    def test_consciousness_coding(self) -> Dict[str, Any]:
        """Test prime aligned compute-enhanced coding"""
        print("\n🧠 Testing prime aligned compute-Enhanced Coding...")
        
        test_cases = [
            {
                "problem_description": "Create a machine learning model for sentiment analysis",
                "prime_aligned_level": "2.0"
            },
            {
                "problem_description": "Design a quantum algorithm for optimization",
                "prime_aligned_level": "3.0"
            },
            {
                "problem_description": "Build a neural network for image recognition",
                "prime_aligned_level": "1.618"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases):
            try:
                response = requests.post(
                    f"{self.api_url}/plugin/execute",
                    headers=self.headers,
                    json={
                        "tool_name": "grok_consciousness_coding",
                        "parameters": case
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        solution = result.get("result", {}).get("consciousness_solution", "")
                        consciousness_enhancement = result.get("result", {}).get("consciousness_enhancement", 1.0)
                        results.append({
                            "case": i + 1,
                            "success": True,
                            "consciousness_enhancement": consciousness_enhancement,
                            "solution_length": len(solution)
                        })
                        print(f"   Case {i+1}: ✅ Generated solution (enhancement: {consciousness_enhancement:.3f}x)")
                    else:
                        results.append({"case": i + 1, "success": False, "error": result.get("error", "Unknown")})
                        print(f"   Case {i+1}: ❌ Failed - {result.get('error', 'Unknown')}")
                else:
                    results.append({"case": i + 1, "success": False, "error": f"HTTP {response.status_code}"})
                    print(f"   Case {i+1}: ❌ HTTP Error {response.status_code}")
                    
            except Exception as e:
                results.append({"case": i + 1, "success": False, "error": str(e)})
                print(f"   Case {i+1}: ❌ Exception - {e}")
        
        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)
        avg_enhancement = sum(r.get("consciousness_enhancement", 0) for r in results if r["success"]) / max(success_count, 1)
        
        print(f"   📊 prime aligned compute Coding: {success_count}/{total_count} successful ({success_count/total_count*100:.1f}%)")
        print(f"   📈 Average prime aligned compute Enhancement: {avg_enhancement:.3f}x")
        
        return {
            "test_name": "prime aligned compute-Enhanced Coding",
            "success_rate": success_count / total_count,
            "average_enhancement": avg_enhancement,
            "results": results
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all fixed benchmark tests"""
        print("🚀 FIXED BENCHMARK TESTING")
        print("=" * 50)
        print("Testing chAIos platform with working tools")
        print()
        
        start_time = time.time()
        
        # Run tests
        test_results = []
        test_results.append(self.test_grok_code_generation())
        test_results.append(self.test_grok_code_optimization())
        test_results.append(self.test_consciousness_coding())
        
        execution_time = time.time() - start_time
        
        # Calculate overall metrics
        total_tests = len(test_results)
        avg_success_rate = sum(t["success_rate"] for t in test_results) / total_tests
        avg_enhancement = sum(t["average_enhancement"] for t in test_results) / total_tests
        
        print(f"\n" + "=" * 50)
        print("📊 FIXED BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Average Success Rate: {avg_success_rate:.3f} ({avg_success_rate*100:.1f}%)")
        print(f"Average prime aligned compute Enhancement: {avg_enhancement:.3f}x")
        print(f"Total Execution Time: {execution_time:.3f}s")
        
        # Assessment
        if avg_success_rate >= 0.8:
            assessment = "🌟 EXCELLENT - All systems working"
        elif avg_success_rate >= 0.6:
            assessment = "✅ GOOD - Most systems working"
        elif avg_success_rate >= 0.4:
            assessment = "⚠️ MODERATE - Some issues"
        else:
            assessment = "❌ POOR - Significant issues"
        
        print(f"Overall Assessment: {assessment}")
        
        return {
            "summary": {
                "total_tests": total_tests,
                "average_success_rate": avg_success_rate,
                "average_enhancement": avg_enhancement,
                "execution_time": execution_time,
                "assessment": assessment
            },
            "test_results": test_results
        }

def main():
    """Main entry point"""
    print("🔧 Starting Fixed Benchmark Testing...")
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/plugin/health", timeout=5)
        if response.status_code != 200:
            print("❌ chAIos API is not available. Please start the server first.")
            return
        else:
            print("✅ chAIos API is available and ready for testing")
    except Exception as e:
        print(f"❌ Cannot connect to chAIos API: {e}")
        return
    
    # Run tests
    tester = FixedBenchmarkTest()
    results = tester.run_all_tests()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"fixed_benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    print("🎉 Fixed benchmark testing complete!")

if __name__ == "__main__":
    main()
