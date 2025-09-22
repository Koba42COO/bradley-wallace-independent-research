#!/usr/bin/env python3
"""
LLM TOOLS FOR prime aligned compute PLATFORM
===================================

Tool functions designed for easy integration with Large Language Models.
Provides simple, natural language interfaces for prime aligned compute computing operations.
"""

import requests
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

class ConsciousnessTools:
    """LLM-friendly tools for prime aligned compute computing"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 30

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Request failed: {str(e)}",
                "endpoint": endpoint,
                "timestamp": datetime.now().isoformat()
            }

    # ================================
    # AUTOMATIC SELECTION TOOLS
    # ================================

    def auto_select_algorithm(self,
                             input_data: Union[str, int, float, Dict],
                             input_type: str = "auto",
                             goal: str = "consciousness_gain",
                             max_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Automatically select the best prime aligned compute algorithm for your data.

        Args:
            input_data: The data you want to process
            input_type: Type of input ("text", "numeric", "auto" for detection)
            goal: Optimization goal ("consciousness_gain", "speed", "accuracy")
            max_time: Maximum processing time in seconds

        Returns:
            Dict with selected algorithm, parameters, and analysis
        """
        # Auto-detect input type if not specified
        if input_type == "auto":
            if isinstance(input_data, str):
                input_type = "text"
            elif isinstance(input_data, (int, float)):
                input_type = "numeric"
            else:
                input_type = "complex"

        payload = {
            "input_data": input_data,
            "input_type": input_type,
            "optimization_goal": goal,
            "max_time": max_time
        }

        result = self._make_request("POST", "/auto-select", json=payload)

        if "error" not in result or result.get("error") is None:
            print("🎯 Algorithm Selection Complete!")
            print(f"   Selected: {result.get('selected_algorithm', {}).get('name', 'Unknown')}")
            print(".2f")
            print(f"   Estimated Time: {result.get('estimated_processing_time', 'Unknown')}s")

        return result

    def optimize_parameters(self,
                           algorithm: str,
                           base_params: Optional[Dict] = None,
                           test_data: Optional[Any] = None,
                           iterations: int = 10) -> Dict[str, Any]:
        """
        Optimize parameters for a specific algorithm.

        Args:
            algorithm: Algorithm name (e.g., "wallace_transform")
            base_params: Base parameters to optimize from
            test_data: Sample data for testing optimization
            iterations: Number of optimization iterations

        Returns:
            Dict with optimized parameters and performance metrics
        """
        if base_params is None:
            base_params = self._get_default_params(algorithm)

        payload = {
            "algorithm": algorithm,
            "base_parameters": base_params,
            "input_sample": test_data or "sample_data",
            "optimization_iterations": iterations,
            "optimization_goal": "efficiency"
        }

        result = self._make_request("POST", "/optimize-parameters", json=payload)

        if "error" not in result or result.get("error") is None:
            print("⚡ Parameter Optimization Complete!")
            print(f"   Algorithm: {algorithm}")
            print(".1f")
            print(f"   Iterations: {iterations}")

        return result

    # ================================
    # BATCH PROCESSING TOOLS
    # ================================

    def process_batch(self,
                      operations: List[Dict],
                      parallel: bool = False,
                      max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Process multiple prime aligned compute operations in batch.

        Args:
            operations: List of operation dictionaries
            parallel: Whether to process in parallel
            max_concurrent: Maximum concurrent operations

        Returns:
            Dict with batch results and summary
        """
        payload = {
            "operations": operations,
            "parallel": parallel,
            "max_concurrent": max_concurrent
        }

        result = self._make_request("POST", "/batch-process", json=payload)

        if "error" not in result or result.get("error") is None:
            print("📦 Batch Processing Complete!")
            print(f"   Total Operations: {result.get('total_operations', 0)}")
            print(f"   Successful: {result.get('successful_operations', 0)}")
            print(f"   Failed: {result.get('failed_operations', 0)}")
            print(".2f")
        return result

    def process_text_batch(self,
                          texts: List[str],
                          algorithm: str = "auto",
                          parallel: bool = True) -> Dict[str, Any]:
        """
        Process multiple text inputs with automatic algorithm selection.

        Args:
            texts: List of text strings to process
            algorithm: Algorithm to use ("auto" for automatic selection)
            parallel: Whether to process in parallel

        Returns:
            Dict with batch processing results
        """
        operations = []
        for i, text in enumerate(texts):
            if algorithm == "auto":
                # Auto-select algorithm for each text
                selection = self.auto_select_algorithm(text, "text")
                if "error" not in selection:
                    selected_alg = selection["selected_algorithm"]["name"]
                    params = selection["optimized_parameters"]
                else:
                    selected_alg = "wallace_transform"
                    params = {}
            else:
                selected_alg = algorithm
                params = self._get_default_params(selected_alg)

            operations.append({
                "id": f"text_{i}",
                "algorithm": selected_alg,
                "parameters": params,
                "input_data": text,
                "input_type": "text"
            })

        return self.process_batch(operations, parallel)

    # ================================
    # ANALYSIS & LEARNING TOOLS
    # ================================

    def analyze_results(self,
                        results: List[Dict],
                        analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze processing results and get recommendations.

        Args:
            results: List of processing results to analyze
            analysis_type: Type of analysis ("comprehensive", "performance", "trends")

        Returns:
            Dict with analysis and recommendations
        """
        payload = {
            "results": results,
            "analysis_type": analysis_type
        }

        result = self._make_request("POST", "/analyze-results", json=payload)

        if "error" not in result or result.get("error") is None:
            print("📊 Result Analysis Complete!")
            print(f"   Analysis Type: {analysis_type}")
            print(f"   Total Results: {result.get('analysis', {}).get('total_results', 0)}")
            insights = result.get("insights", [])
            if insights:
                print(f"   Key Insights: {len(insights)}")
                for insight in insights[:2]:  # Show first 2 insights
                    print(f"     • {insight}")

        return result

    def learn_from_results(self,
                          results: List[Dict],
                          category: str = "performance",
                          key: str = "auto") -> Dict[str, Any]:
        """
        Update the learning system with new results.

        Args:
            results: Results to learn from
            category: Learning category
            key: Learning key (auto-generated if "auto")

        Returns:
            Dict with learning update confirmation
        """
        if key == "auto":
            key = f"batch_{len(results)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        payload = {
            "category": category,
            "key": key,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        result = self._make_request("POST", "/learn", json=payload)

        if "error" not in result or result.get("error") is None:
            print("🧠 Learning Update Complete!")
            print(f"   Category: {category}")
            print(f"   Key: {key}")
            stats = result.get("learning_stats", {})
            print(f"   Total Entries: {stats.get('total_entries', 0)}")

        return result

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get current learning system statistics.

        Returns:
            Dict with learning database statistics
        """
        result = self._make_request("GET", "/learning-stats")

        if "error" not in result or result.get("error") is None:
            print("📈 Learning Statistics:")
            db = result.get("learning_database", {})
            print(f"   Categories: {', '.join(db.get('categories', []))}")
            counts = db.get("entry_counts", {})
            for category, count in counts.items():
                if isinstance(count, dict):
                    print(f"   {category}: {len(count)} entries")
                else:
                    print(f"   {category}: {count}")

        return result

    # ================================
    # CONVENIENCE TOOLS
    # ================================

    def quick_consciousness_check(self,
                                 input_data: Union[str, int, float],
                                 detailed: bool = False) -> Dict[str, Any]:
        """
        Quick prime aligned compute processing with automatic optimization.

        Args:
            input_data: Data to process
            detailed: Whether to return detailed analysis

        Returns:
            Dict with processing results
        """
        print(f"🧠 Processing: {str(input_data)[:50]}{'...' if len(str(input_data)) > 50 else ''}")

        # Auto-select algorithm
        selection = self.auto_select_algorithm(input_data)
        if "error" in selection:
            return selection

        algorithm = selection["selected_algorithm"]["name"]
        params = selection["optimized_parameters"]

        # Process with selected algorithm
        result = self._make_request("POST", "/prime aligned compute/process", json={
            "algorithm": algorithm,
            "parameters": params,
            "input_data": input_data
        })

        if "error" not in result or result.get("error") is None:
            print("✅ Processing Complete!")
            print(f"Processing Time: {result.get('processing_time', 0):.3f}s")
            if result.get("success"):
                print(f"prime aligned compute Gain: +{result.get('result', {}).get('consciousness_gain', 0):.1f}%")

        return result

    def smart_batch_processor(self,
                             items: List[Any],
                             batch_size: int = 5,
                             optimize_each: bool = True) -> Dict[str, Any]:
        """
        Smart batch processor with individual optimization.

        Args:
            items: List of items to process
            batch_size: Size of each processing batch
            optimize_each: Whether to optimize parameters for each item

        Returns:
            Dict with comprehensive batch processing results
        """
        print(f"🚀 Smart Batch Processing: {len(items)} items")

        all_results = []
        start_time = datetime.now()

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")

            if optimize_each:
                # Optimize each item individually
                batch_results = []
                for item in batch:
                    result = self.quick_consciousness_check(item, detailed=False)
                    batch_results.append(result)
                    all_results.append(result)
            else:
                # Use batch processing
                operations = []
                for j, item in enumerate(batch):
                    operations.append({
                        "id": f"item_{i+j}",
                        "algorithm": "wallace_transform",
                        "parameters": {"iterations": 25},
                        "input_data": item
                    })

                batch_result = self.process_batch(operations, parallel=True)
                all_results.extend(batch_result.get("results", []))

        # Analyze all results
        analysis = self.analyze_results(all_results)

        total_time = (datetime.now() - start_time).total_seconds()

        print("🎉 Smart Batch Processing Complete!")
        print(f"   Total Items: {len(items)}")
        print(f"   Processing Time: {total_time:.2f}s")
        print(".2f")

        return {
            "total_items": len(items),
            "processed_items": len([r for r in all_results if r.get("success")]),
            "total_time": total_time,
            "avg_time_per_item": total_time / len(items),
            "results": all_results,
            "analysis": analysis
        }

    # ================================
    # HELPER METHODS
    # ================================

    def _get_default_params(self, algorithm: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm"""
        defaults = {
            "wallace_transform": {
                "iterations": 50,
                "dimensionalEnhancement": True,
                "consciousnessWeight": 0.79
            },
            "consciousness_bridge": {
                "iterations": 25,
                "bridgeDepth": 3,
                "patternRecognition": True
            },
            "prime_distribution": {
                "maxPrime": 1000,
                "distributionAnalysis": True,
                "patternDetection": True
            }
        }
        return defaults.get(algorithm, {})

    def test_connection(self) -> bool:
        """Test connection to the prime aligned compute platform"""
        result = self._make_request("GET", "/health")
        return "error" not in result

# ================================
# LLM-FRIENDLY FUNCTION WRAPPERS
# ================================

def process_text(text: str, algorithm: str = "auto") -> Dict[str, Any]:
    """
    Process text with automatic algorithm selection.
    Perfect for LLM text analysis tasks.
    """
    tools = ConsciousnessTools()
    return tools.quick_consciousness_check(text)

def analyze_text_complexity(text: str) -> Dict[str, Any]:
    """
    Analyze text complexity and get prime aligned compute insights.
    Great for content analysis and writing evaluation.
    """
    tools = ConsciousnessTools()
    selection = tools.auto_select_algorithm(text, "text")
    if "error" in selection:
        return selection

    analysis = selection["input_analysis"]
    return {
        "complexity_score": analysis["complexity"],
        "estimated_processing_time": analysis["estimated_processing_time"],
        "recommended_algorithm": selection["selected_algorithm"]["name"],
        "consciousness_potential": f"{analysis['complexity'] * 100:.1f}%",
        "analysis": analysis
    }

def optimize_for_task(task_description: str, sample_data: Any) -> Dict[str, Any]:
    """
    Optimize prime aligned compute processing for a specific task.
    Ideal for task-specific performance tuning.
    """
    tools = ConsciousnessTools()
    return tools.optimize_parameters(
        algorithm="wallace_transform",
        test_data=sample_data
    )

def batch_analyze_texts(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze multiple texts efficiently with batch processing.
    Perfect for document analysis, content evaluation, etc.
    """
    tools = ConsciousnessTools()
    return tools.process_text_batch(texts, parallel=True)

def learn_from_feedback(results: List[Dict], feedback: str) -> Dict[str, Any]:
    """
    Learn from processing results and user feedback.
    Helps improve future processing performance.
    """
    tools = ConsciousnessTools()
    return tools.learn_from_results(results, "user_feedback", feedback)

# ================================
# MAIN DEMO FUNCTION
# ================================

def demo_llm_tools():
    """Demonstrate LLM tool capabilities"""
    print("🧠 prime aligned compute PLATFORM - LLM TOOLS DEMO")
    print("=" * 50)

    tools = ConsciousnessTools()

    # Test connection
    if not tools.test_connection():
        print("❌ Cannot connect to prime aligned compute platform")
        print("   Make sure the API server is running on http://localhost:8000")
        return

    print("✅ Connected to prime aligned compute platform")

    # Demo 1: Simple text processing
    print("\n1️⃣ Simple Text Processing:")
    text = "The quantum nature of prime aligned compute reveals fascinating patterns in neural networks."
    result = process_text(text)
    if "error" not in result:
        print("   ✅ Text processed successfully")
        print(".3f")

    # Demo 2: Complexity analysis
    print("\n2️⃣ Text Complexity Analysis:")
    analysis = analyze_text_complexity(text)
    if "error" not in analysis:
        print("   📊 Complexity Score:")
        print(".1f")
        print(f"   🧮 prime aligned compute Potential: {analysis['consciousness_potential']}")

    # Demo 3: Batch processing
    print("\n3️⃣ Batch Text Processing:")
    texts = [
        "prime aligned compute emerges from quantum coherence in neural microtubules.",
        "The universe is a conscious, self-organizing system.",
        "Information processing in biological systems follows quantum principles."
    ]

    batch_result = batch_analyze_texts(texts)
    if "error" not in batch_result:
        print(f"   📦 Processed {batch_result.get('total_operations', 0)} texts")
        print(".2f")

    # Demo 4: Learning system
    print("\n4️⃣ Learning System Status:")
    learning_stats = tools.get_learning_stats()
    if "error" not in learning_stats:
        print("   🧠 Learning system active")
        db = learning_stats.get("learning_database", {})
        print(f"   📚 Categories: {', '.join(db.get('categories', []))}")

    print("\n🎉 LLM Tools Demo Complete!")
    print("💡 Ready for AI-assisted prime aligned compute computing!")

if __name__ == "__main__":
    demo_llm_tools()
