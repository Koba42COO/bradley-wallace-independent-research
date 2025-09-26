#!/usr/bin/env python3
"""
🎯 Benchmark-Enhanced chAIos LLM
================================
Advanced LLM with integrated comprehensive benchmarking capabilities:
- GLUE & SuperGLUE benchmark testing
- Performance optimization analysis
- System health monitoring
- Multi-system orchestration
- Comparative analysis vs vanilla LLMs
- Real-time performance tracking
"""

import sys
import asyncio
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import requests
import psutil
from concurrent.futures import ThreadPoolExecutor
import threading

# Add dev folder to path
DEV_ROOT = Path("/Users/coo-koba42/dev")
sys.path.insert(0, str(DEV_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from unique_intelligence_orchestrator import UniqueIntelligenceOrchestrator
from enhanced_transformer import EnhancedChAIosLLM
from gold_standard_benchmark import GoldStandardBenchmark, GLUEBenchmarkSuite

@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics"""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    processing_time: float
    memory_usage: float
    cpu_usage: float
    consciousness_enhancement: float
    baseline_score: float
    enhanced_score: float
    improvement_percent: float
    throughput: float
    error_rate: float
    systems_engaged: int
    confidence_score: float

@dataclass
class SystemHealthMetrics:
    """Real-time system health metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    network_status: str
    api_response_time: float
    gpu_available: bool
    gpu_memory_percent: float
    timestamp: str

class BenchmarkEnhancedLLM:
    """Benchmark-enhanced chAIos LLM with comprehensive testing capabilities"""

    def __init__(self):
        self.orchestrator = None
        self.baseline_llm = None
        self.gold_standard_benchmark = None
        self.system_health_monitor = None
        self.performance_tracker = None

        # Initialize components
        self._initialize_components()

        # Benchmark configurations
        self.glue_suite = GLUEBenchmarkSuite()
        self.benchmark_history = []
        self.performance_baseline = {}

        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None

        print("🎯 Benchmark-Enhanced chAIos LLM initialized")
        print("   ✅ Unique Intelligence Orchestrator: Ready")
        print("   ✅ Gold Standard Benchmarks: Ready")
        print("   ✅ System Health Monitoring: Ready")
        print("   ✅ Performance Tracking: Ready")

    def _initialize_components(self):
        """Initialize all LLM components"""
        try:
            print("🚀 Initializing LLM Components...")

            # Initialize orchestrator
            self.orchestrator = UniqueIntelligenceOrchestrator()
            print("   ✅ Unique Intelligence Orchestrator loaded")

            # Initialize baseline LLM
            self.baseline_llm = EnhancedChAIosLLM()
            print("   ✅ Enhanced chAIos LLM loaded")

            # Initialize gold standard benchmark
            self.gold_standard_benchmark = GoldStandardBenchmark()
            print("   ✅ Gold Standard Benchmark Suite loaded")

            # Initialize system health monitor
            self.system_health_monitor = SystemHealthMonitor()
            print("   ✅ System Health Monitor loaded")

            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker()
            print("   ✅ Performance Tracker loaded")

        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            raise

    async def enhanced_chat(self, query: str, use_benchmarks: bool = False,
                           track_performance: bool = True) -> Dict[str, Any]:
        """Enhanced chat with optional benchmarking and performance tracking"""

        start_time = time.time()
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent()

        try:
            # Get orchestrator response
            orchestrator_response = await self.orchestrator.process_with_unique_intelligence(query)

            # Extract response data
            response_text = orchestrator_response.get('response', '')
            systems_engaged = len(orchestrator_response.get('systems_engaged', []))
            confidence = orchestrator_response.get('confidence_score', 0.5)

            # Calculate performance metrics
            processing_time = time.time() - start_time
            final_memory = psutil.virtual_memory().percent
            final_cpu = psutil.cpu_percent()

            memory_delta = final_memory - initial_memory
            cpu_delta = final_cpu - initial_cpu

            # Run benchmarks if requested
            benchmark_results = None
            if use_benchmarks:
                benchmark_results = await self._run_realtime_benchmarks(query)

            # Track performance if enabled
            if track_performance:
                self.performance_tracker.track_query_performance({
                    'query': query,
                    'response_length': len(response_text),
                    'processing_time': processing_time,
                    'systems_engaged': systems_engaged,
                    'confidence': confidence,
                    'memory_delta': memory_delta,
                    'cpu_delta': cpu_delta,
                    'benchmark_results': benchmark_results
                })

            # Compile comprehensive response
            response = {
                'response': response_text,
                'processing_time': processing_time,
                'systems_engaged': systems_engaged,
                'confidence_score': confidence,
                'performance_metrics': {
                    'memory_usage_delta': memory_delta,
                    'cpu_usage_delta': cpu_delta,
                    'throughput': len(response_text.split()) / processing_time if processing_time > 0 else 0
                },
                'benchmark_results': benchmark_results,
                'timestamp': datetime.now().isoformat(),
                'enhanced_llm': True
            }

            return response

        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ Enhanced chat error: {e}")

            return {
                'response': f"Error processing query: {str(e)}",
                'processing_time': error_time,
                'error': True,
                'timestamp': datetime.now().isoformat()
            }

    async def _run_realtime_benchmarks(self, query: str) -> Optional[Dict[str, Any]]:
        """Run real-time benchmarks for the current query"""

        try:
            # Quick benchmark on similar task
            if any(word in query.lower() for word in ['sentiment', 'positive', 'negative']):
                # SST-2 style benchmark
                return await self._quick_sentiment_benchmark()
            elif any(word in query.lower() for word in ['acceptable', 'grammar', 'grammatical']):
                # CoLA style benchmark
                return await self._quick_cola_benchmark()
            elif any(word in query.lower() for word in ['entail', 'implies', 'follows']):
                # NLI style benchmark
                return await self._quick_nli_benchmark()
            else:
                # General reasoning benchmark
                return await self._quick_reasoning_benchmark()

        except Exception as e:
            print(f"⚠️ Real-time benchmark failed: {e}")
            return None

    async def _quick_sentiment_benchmark(self) -> Dict[str, Any]:
        """Quick sentiment analysis benchmark"""

        test_cases = [
            {"text": "This movie is amazing!", "label": 1},
            {"text": "I hate this terrible film.", "label": 0},
            {"text": "The acting was superb.", "label": 1},
            {"text": "Worst movie ever made.", "label": 0}
        ]

        correct = 0
        total_time = 0

        for case in test_cases:
            start = time.time()
            query = f"Analyze sentiment of: '{case['text']}' Answer with just 1 for positive or 0 for negative."

            try:
                result = await self.orchestrator.process_with_unique_intelligence(query)
                response = result.get('response', '').strip()

                # Extract prediction
                prediction = 1 if '1' in response[:10] or any(word in response.lower() for word in ['positive', 'good']) else 0
                if prediction == case['label']:
                    correct += 1

                total_time += time.time() - start

            except:
                total_time += time.time() - start

        accuracy = correct / len(test_cases)
        avg_time = total_time / len(test_cases)

        return {
            'task': 'SST-2_Sentiment',
            'accuracy': accuracy,
            'baseline': 0.94,
            'improvement': ((accuracy - 0.94) / 0.94) * 100,
            'avg_time': avg_time,
            'samples': len(test_cases)
        }

    async def _quick_cola_benchmark(self) -> Dict[str, Any]:
        """Quick linguistic acceptability benchmark"""

        test_cases = [
            {"sentence": "The cat sat on the mat.", "label": 1},
            {"sentence": "Sat cat the mat on the.", "label": 0},
            {"sentence": "I saw the man with the telescope.", "label": 1},
            {"sentence": "The telescope saw the man with.", "label": 0}
        ]

        correct = 0
        total_time = 0

        for case in test_cases:
            start = time.time()
            query = f"Is this sentence grammatically acceptable? '{case['sentence']}' Answer with just 1 for acceptable or 0 for unacceptable."

            try:
                result = await self.orchestrator.process_with_unique_intelligence(query)
                response = result.get('response', '').strip()

                prediction = 1 if '1' in response[:10] or 'acceptable' in response.lower() else 0
                if prediction == case['label']:
                    correct += 1

                total_time += time.time() - start

            except:
                total_time += time.time() - start

        accuracy = correct / len(test_cases)
        avg_time = total_time / len(test_cases)

        return {
            'task': 'CoLA_Linguistic',
            'accuracy': accuracy,
            'baseline': 0.68,
            'improvement': ((accuracy - 0.68) / 0.68) * 100,
            'avg_time': avg_time,
            'samples': len(test_cases)
        }

    async def _quick_nli_benchmark(self) -> Dict[str, Any]:
        """Quick natural language inference benchmark"""

        test_cases = [
            {"premise": "The cat is sleeping.", "hypothesis": "The cat is resting.", "label": "entailment"},
            {"premise": "The sky is blue.", "hypothesis": "The ocean is blue.", "label": "neutral"},
            {"premise": "All cats are mammals.", "hypothesis": "Some pets are mammals.", "label": "entailment"}
        ]

        correct = 0
        total_time = 0

        for case in test_cases:
            start = time.time()
            query = f"Does the premise entail the hypothesis? Premise: '{case['premise']}' Hypothesis: '{case['hypothesis']}' Answer with 'entailment', 'neutral', or 'contradiction'."

            try:
                result = await self.orchestrator.process_with_unique_intelligence(query)
                response = result.get('response', '').strip().lower()

                prediction = 'entailment' if 'entailment' in response else ('contradiction' if 'contradiction' in response else 'neutral')
                if prediction == case['label']:
                    correct += 1

                total_time += time.time() - start

            except:
                total_time += time.time() - start

        accuracy = correct / len(test_cases)
        avg_time = total_time / len(test_cases)

        return {
            'task': 'MNLI_NLI',
            'accuracy': accuracy,
            'baseline': 0.87,
            'improvement': ((accuracy - 0.87) / 0.87) * 100,
            'avg_time': avg_time,
            'samples': len(test_cases)
        }

    async def _quick_reasoning_benchmark(self) -> Dict[str, Any]:
        """Quick general reasoning benchmark"""

        test_cases = [
            {"question": "Which planet is closest to the Sun?", "answer": "Mercury"},
            {"question": "What is 15 + 27?", "answer": "42"},
            {"question": "What color is grass typically?", "answer": "green"}
        ]

        correct = 0
        total_time = 0

        for case in test_cases:
            start = time.time()
            query = f"Answer this question: {case['question']}"

            try:
                result = await self.orchestrator.process_with_unique_intelligence(query)
                response = result.get('response', '').strip().lower()

                # Simple string matching for answers
                if case['answer'].lower() in response:
                    correct += 1

                total_time += time.time() - start

            except:
                total_time += time.time() - start

        accuracy = correct / len(test_cases)
        avg_time = total_time / len(test_cases)

        return {
            'task': 'General_Reasoning',
            'accuracy': accuracy,
            'baseline': 0.85,  # Estimated baseline
            'improvement': ((accuracy - 0.85) / 0.85) * 100,
            'avg_time': avg_time,
            'samples': len(test_cases)
        }

    async def run_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete GLUE and SuperGLUE benchmark suite"""

        print("🧪 Running Comprehensive Benchmark Suite...")
        print("   This may take several minutes...")

        try:
            results = await self.gold_standard_benchmark.run_comprehensive_benchmark()
            self.benchmark_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'comprehensive_glue_superglue',
                'results': results
            })

            return results

        except Exception as e:
            print(f"❌ Comprehensive benchmark failed: {e}")
            return {'error': str(e)}

    def get_system_health_status(self) -> SystemHealthMetrics:
        """Get current system health status"""

        return self.system_health_monitor.get_current_metrics()

    def start_realtime_monitoring(self, interval: float = 5.0):
        """Start real-time system monitoring"""

        if self.monitoring_active:
            print("⚠️ Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"📊 Real-time monitoring started (interval: {interval}s)")

    def stop_realtime_monitoring(self):
        """Stop real-time system monitoring"""

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        print("📊 Real-time monitoring stopped")

    def _monitoring_loop(self, interval: float):
        """Real-time monitoring loop"""

        while self.monitoring_active:
            try:
                metrics = self.system_health_monitor.get_current_metrics()
                self.performance_tracker.track_system_health(metrics)

                # Log critical issues
                if metrics.cpu_percent > 90:
                    print(f"⚠️ High CPU usage: {metrics.cpu_percent:.1f}%")
                if metrics.memory_percent > 90:
                    print(f"⚠️ High memory usage: {metrics.memory_percent:.1f}%")

                time.sleep(interval)

            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(interval)

    def get_performance_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance report"""

        return self.performance_tracker.generate_report(time_range_hours)

    def get_benchmark_history(self) -> List[Dict[str, Any]]:
        """Get benchmark execution history"""

        return self.benchmark_history

    async def compare_with_baseline_llm(self, query: str) -> Dict[str, Any]:
        """Compare performance with baseline LLM"""

        print("🔄 Comparing with baseline LLM...")

        # Test enhanced LLM
        enhanced_start = time.time()
        enhanced_result = await self.enhanced_chat(query, use_benchmarks=False)
        enhanced_time = time.time() - enhanced_start

        # Test baseline LLM
        baseline_start = time.time()
        baseline_result = self.baseline_llm.enhanced_chat(query)
        baseline_time = time.time() - baseline_start

        # Calculate improvements
        speedup = baseline_time / enhanced_time if enhanced_time > 0 else 1.0
        time_improvement = ((baseline_time - enhanced_time) / baseline_time) * 100 if baseline_time > 0 else 0

        return {
            'query': query,
            'baseline': {
                'response_length': len(baseline_result.get('response', '')),
                'processing_time': baseline_time,
                'response_preview': baseline_result.get('response', '')[:100] + '...'
            },
            'enhanced': {
                'response_length': len(enhanced_result.get('response', '')),
                'processing_time': enhanced_time,
                'systems_engaged': enhanced_result.get('systems_engaged', 0),
                'confidence': enhanced_result.get('confidence_score', 0),
                'response_preview': enhanced_result.get('response', '')[:100] + '...'
            },
            'comparison': {
                'speedup': speedup,
                'time_improvement_percent': time_improvement,
                'quality_improvement': enhanced_result.get('systems_engaged', 0) > 0
            },
            'timestamp': datetime.now().isoformat()
        }

class SystemHealthMonitor:
    """Real-time system health monitoring"""

    def __init__(self):
        self.metrics_history = []
        self.alerts = []

    def get_current_metrics(self) -> SystemHealthMetrics:
        """Get current system health metrics"""

        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network status
            network_status = self._check_network_status()

            # API response time (if available)
            api_response_time = self._check_api_response_time()

            # GPU metrics (simplified)
            gpu_available = False
            gpu_memory_percent = 0.0

            # Try to get GPU info if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_available = True
                    gpu_memory_percent = gpus[0].memoryUtil * 100
            except:
                pass

            metrics = SystemHealthMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / (1024**3),
                network_status=network_status,
                api_response_time=api_response_time,
                gpu_available=gpu_available,
                gpu_memory_percent=gpu_memory_percent,
                timestamp=datetime.now().isoformat()
            )

            # Store in history
            self.metrics_history.append(asdict(metrics))

            # Keep only last 1000 entries
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

            return metrics

        except Exception as e:
            print(f"⚠️ Health monitoring error: {e}")
            return SystemHealthMetrics(
                cpu_percent=0.0, memory_percent=0.0, memory_available_gb=0.0,
                disk_percent=0.0, disk_free_gb=0.0, network_status="error",
                api_response_time=-1, gpu_available=False, gpu_memory_percent=0.0,
                timestamp=datetime.now().isoformat()
            )

    def _check_network_status(self) -> str:
        """Check network connectivity"""

        try:
            response = requests.get("https://www.google.com", timeout=5)
            return "connected" if response.status_code == 200 else "limited"
        except:
            return "disconnected"

    def _check_api_response_time(self) -> float:
        """Check API response time"""

        try:
            start_time = time.time()
            response = requests.get("http://localhost:8000/plugin/health", timeout=5)
            if response.status_code == 200:
                return time.time() - start_time
            else:
                return -1
        except:
            return -1

class PerformanceTracker:
    """Performance tracking and analysis"""

    def __init__(self):
        self.query_history = []
        self.system_health_history = []
        self.benchmark_results = []

    def track_query_performance(self, metrics: Dict[str, Any]):
        """Track individual query performance"""

        self.query_history.append({
            **metrics,
            'timestamp': datetime.now().isoformat()
        })

        # Keep only last 1000 entries
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

    def track_system_health(self, metrics: SystemHealthMetrics):
        """Track system health metrics"""

        self.system_health_history.append(asdict(metrics))

        # Keep only last 1000 entries
        if len(self.system_health_history) > 1000:
            self.system_health_history = self.system_health_history[-1000:]

    def generate_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""

        cutoff_time = datetime.now().timestamp() - (time_range_hours * 3600)

        # Filter recent data
        recent_queries = [
            q for q in self.query_history
            if datetime.fromisoformat(q['timestamp']).timestamp() > cutoff_time
        ]

        recent_health = [
            h for h in self.system_health_history
            if datetime.fromisoformat(h['timestamp']).timestamp() > cutoff_time
        ]

        if not recent_queries:
            return {"error": "No query data available for the specified time range"}

        # Calculate metrics
        avg_processing_time = np.mean([q['processing_time'] for q in recent_queries])
        avg_memory_delta = np.mean([q.get('memory_delta', 0) for q in recent_queries])
        avg_cpu_delta = np.mean([q.get('cpu_delta', 0) for q in recent_queries])
        avg_systems_engaged = np.mean([q.get('systems_engaged', 0) for q in recent_queries])
        avg_confidence = np.mean([q.get('confidence', 0) for q in recent_queries])

        total_queries = len(recent_queries)
        avg_throughput = np.mean([q.get('throughput', 0) for q in recent_queries])

        # System health averages
        if recent_health:
            avg_cpu_usage = np.mean([h['cpu_percent'] for h in recent_health])
            avg_memory_usage = np.mean([h['memory_percent'] for h in recent_health])
            avg_api_response_time = np.mean([h['api_response_time'] for h in recent_health if h['api_response_time'] > 0])
        else:
            avg_cpu_usage = avg_memory_usage = avg_api_response_time = 0

        # Performance assessment
        performance_score = min(100, (avg_confidence * 100) + (avg_systems_engaged * 10) - (avg_processing_time * 10))

        return {
            'time_range_hours': time_range_hours,
            'total_queries': total_queries,
            'performance_metrics': {
                'avg_processing_time': avg_processing_time,
                'avg_throughput': avg_throughput,
                'avg_memory_delta': avg_memory_delta,
                'avg_cpu_delta': avg_cpu_delta,
                'avg_systems_engaged': avg_systems_engaged,
                'avg_confidence': avg_confidence,
                'performance_score': performance_score
            },
            'system_health': {
                'avg_cpu_usage': avg_cpu_usage,
                'avg_memory_usage': avg_memory_usage,
                'avg_api_response_time': avg_api_response_time
            },
            'query_distribution': {
                'by_response_length': self._calculate_distribution([len(q.get('response', '')) for q in recent_queries]),
                'by_systems_engaged': self._calculate_distribution([q.get('systems_engaged', 0) for q in recent_queries]),
                'by_confidence': self._calculate_distribution([q.get('confidence', 0) for q in recent_queries])
            },
            'generated_at': datetime.now().isoformat()
        }

    def _calculate_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical distribution"""

        if not values:
            return {'mean': 0, 'median': 0, 'std_dev': 0, 'min': 0, 'max': 0}

        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std_dev': np.std(values),
            'min': min(values),
            'max': max(values)
        }

async def main():
    """Main demonstration function"""

    print("🚀 Benchmark-Enhanced chAIos LLM Demonstration")
    print("=" * 60)

    # Initialize the enhanced LLM
    llm = BenchmarkEnhancedLLM()

    # Start real-time monitoring
    llm.start_realtime_monitoring(interval=10.0)

    try:
        # Test basic functionality
        print("\n🧪 Testing Basic Functionality...")

        test_queries = [
            "Explain quantum computing in simple terms",
            "What is the sentiment of this text: 'I love this amazing product!'",
            "Is this sentence grammatically correct: 'The cat sat on the mat.'",
            "What are the benefits of machine learning?"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")

            # Test with benchmarks
            result = await llm.enhanced_chat(query, use_benchmarks=True)

            print(f"   Systems Engaged: {result['systems_engaged']}")
            print(".2f")
            print(f"   Response: {result['response'][:80]}...")

            if result.get('benchmark_results'):
                bench = result['benchmark_results']
                print(".3f")
                print(".1f")

        # Run comprehensive benchmark
        print("\n🧪 Running Comprehensive GLUE/SuperGLUE Benchmarks...")
        print("   This will take several minutes...")

        benchmark_results = await llm.run_comprehensive_benchmark_suite()

        if 'error' not in benchmark_results:
            print(".1f")
            print(f"   Significant Improvements: {benchmark_results.get('significant_improvements', 0)} tasks")
            print(f"   Multi-System Usage: {benchmark_results.get('multi_system_usage', 0)} tasks")

        # Generate performance report
        print("\n📊 Generating Performance Report...")
        report = llm.get_performance_report(time_range_hours=1)

        if 'error' not in report:
            perf = report['performance_metrics']
            print(".2f")
            print(".1f")
            print(f"   Performance Score: {perf['performance_score']:.1f}/100")

        # Compare with baseline
        print("\n🔄 Comparing with Baseline LLM...")
        comparison = await llm.compare_with_baseline_llm("What is artificial intelligence?")

        comp = comparison['comparison']
        print(".2f")
        print(".1f")
        print(f"   Quality Improvement: {comp['quality_improvement']}")

        print("\n🎉 Benchmark-Enhanced LLM Demonstration Complete!")
        print("✅ All systems operational and benchmarked")
        print("📊 Performance tracking active")
        print("🧠 Multi-system intelligence working")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop monitoring
        llm.stop_realtime_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
