#!/usr/bin/env python3
"""
Performance Monitoring System
============================
Comprehensive performance tracking and analytics for RAG/KAG system
Implements detailed metrics, trend analysis, and optimization recommendations
"""

import os
import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Core performance metrics collection"""

    def __init__(self):
        self.metrics = {
            "queries": [],
            "retrieval_performance": [],
            "quality_assessments": [],
            "system_performance": [],
            "user_feedback": []
        }

        self.baseline_metrics = {}
        self.current_session = datetime.now().isoformat()

    def record_query(self, query: str, domain: str, response_time: float,
                    results_count: int, quality_score: float):
        """Record query performance metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "session": self.current_session,
            "query": query,
            "domain": domain,
            "response_time": response_time,
            "results_count": results_count,
            "quality_score": quality_score,
            "success": quality_score > 0.5
        }

        self.metrics["queries"].append(metric)

    def record_retrieval_performance(self, query: str, retrieval_stats: Dict[str, Any]):
        """Record detailed retrieval performance"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "session": self.current_session,
            "query": query,
            **retrieval_stats
        }

        self.metrics["retrieval_performance"].append(metric)

    def record_quality_assessment(self, content_id: str, assessment: Dict[str, Any]):
        """Record content quality assessment"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "session": self.current_session,
            "content_id": content_id,
            **assessment
        }

        self.metrics["quality_assessments"].append(metric)

    def record_system_performance(self, component: str, metrics: Dict[str, Any]):
        """Record system component performance"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "session": self.current_session,
            "component": component,
            **metrics
        }

        self.metrics["system_performance"].append(metric)

    def set_baseline(self, domain: str, metric_name: str, value: float):
        """Set baseline performance metric"""
        if domain not in self.baseline_metrics:
            self.baseline_metrics[domain] = {}

        self.baseline_metrics[domain][metric_name] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }

class PerformanceAnalyzer:
    """Advanced performance analysis and trend detection"""

    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics

    def analyze_query_performance(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze query performance over time window"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

        recent_queries = [
            q for q in self.metrics.metrics["queries"]
            if datetime.fromisoformat(q["timestamp"]) > cutoff_time
        ]

        if not recent_queries:
            return {"error": "No recent queries found"}

        # Basic statistics
        response_times = [q["response_time"] for q in recent_queries]
        quality_scores = [q["quality_score"] for q in recent_queries]
        success_rate = sum(1 for q in recent_queries if q["success"]) / len(recent_queries)

        # Domain breakdown
        domain_stats = defaultdict(list)
        for query in recent_queries:
            domain_stats[query["domain"]].append(query)

        domain_performance = {}
        for domain, queries in domain_stats.items():
            domain_performance[domain] = {
                "query_count": len(queries),
                "avg_response_time": statistics.mean(q["response_time"] for q in queries),
                "avg_quality_score": statistics.mean(q["quality_score"] for q in queries),
                "success_rate": sum(1 for q in queries if q["success"]) / len(queries)
            }

        return {
            "time_window_hours": time_window_hours,
            "total_queries": len(recent_queries),
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "avg_quality_score": statistics.mean(quality_scores),
            "success_rate": success_rate,
            "domain_performance": domain_performance,
            "performance_trend": self._analyze_performance_trend(recent_queries)
        }

    def analyze_retrieval_effectiveness(self) -> Dict[str, Any]:
        """Analyze retrieval system effectiveness"""
        retrieval_data = self.metrics.metrics["retrieval_performance"]

        if not retrieval_data:
            return {"error": "No retrieval performance data available"}

        # Aggregate metrics
        avg_retrieval_count = statistics.mean(r.get("retrieval_count", 0) for r in retrieval_data)
        avg_similarity_score = statistics.mean(r.get("avg_similarity", 0) for r in retrieval_data)
        avg_quality_score = statistics.mean(r.get("avg_quality_score", 0) for r in retrieval_data)

        # Precision analysis (quality of retrieved documents)
        precision_scores = []
        for r in retrieval_data:
            if r.get("retrieval_count", 0) > 0:
                precision = r.get("quality_matches", 0) / r.get("retrieval_count", 0)
                precision_scores.append(precision)

        avg_precision = statistics.mean(precision_scores) if precision_scores else 0

        return {
            "total_retrievals_analyzed": len(retrieval_data),
            "avg_retrieval_count": avg_retrieval_count,
            "avg_similarity_score": avg_similarity_score,
            "avg_quality_score": avg_quality_score,
            "avg_precision": avg_precision,
            "retrieval_efficiency": self._calculate_retrieval_efficiency(retrieval_data)
        }

    def analyze_content_quality_trends(self) -> Dict[str, Any]:
        """Analyze content quality trends"""
        quality_data = self.metrics.metrics["quality_assessments"]

        if not quality_data:
            return {"error": "No quality assessment data available"}

        # Quality distribution
        quality_levels = Counter(q["quality_level"] for q in quality_data)
        quality_scores = [q["score"] for q in quality_data]

        # Domain quality analysis
        domain_quality = defaultdict(list)
        for assessment in quality_data:
            domain = assessment.get("domain", "unknown")
            domain_quality[domain].append(assessment["score"])

        domain_avg_quality = {
            domain: statistics.mean(scores) for domain, scores in domain_quality.items()
        }

        return {
            "total_assessments": len(quality_data),
            "quality_distribution": dict(quality_levels),
            "avg_quality_score": statistics.mean(quality_scores),
            "quality_variance": statistics.variance(quality_scores) if len(quality_scores) > 1 else 0,
            "domain_quality": domain_avg_quality,
            "quality_improvement": self._analyze_quality_improvement(quality_data)
        }

    def _analyze_performance_trend(self, queries: List[Dict[str, Any]]) -> str:
        """Analyze performance trend over time"""
        if len(queries) < 5:
            return "insufficient_data"

        # Sort by timestamp
        sorted_queries = sorted(queries, key=lambda x: x["timestamp"])

        # Split into halves for comparison
        midpoint = len(sorted_queries) // 2
        first_half = sorted_queries[:midpoint]
        second_half = sorted_queries[midpoint:]

        # Compare quality scores
        first_avg = statistics.mean(q["quality_score"] for q in first_half)
        second_avg = statistics.mean(q["quality_score"] for q in second_half)

        improvement = ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0

        if improvement > 5:
            return "improving"
        elif improvement < -5:
            return "declining"
        else:
            return "stable"

    def _calculate_retrieval_efficiency(self, retrieval_data: List[Dict[str, Any]]) -> float:
        """Calculate overall retrieval efficiency"""
        if not retrieval_data:
            return 0.0

        efficiency_scores = []
        for r in retrieval_data:
            # Efficiency = (quality_matches / retrieval_count) * (1 / response_time)
            quality_matches = r.get("quality_matches", 0)
            retrieval_count = r.get("retrieval_count", 1)
            response_time = r.get("response_time", 1)

            if retrieval_count > 0 and response_time > 0:
                precision = quality_matches / retrieval_count
                speed_factor = 1 / response_time
                efficiency = precision * speed_factor
                efficiency_scores.append(efficiency)

        return statistics.mean(efficiency_scores) if efficiency_scores else 0.0

    def _analyze_quality_improvement(self, quality_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality improvement over time"""
        if len(quality_data) < 10:
            return {"trend": "insufficient_data"}

        # Sort by timestamp
        sorted_data = sorted(quality_data, key=lambda x: x["timestamp"])

        # Moving average analysis
        window_size = min(10, len(sorted_data) // 3)
        moving_averages = []

        for i in range(window_size, len(sorted_data) + 1):
            window = sorted_data[i-window_size:i]
            avg_quality = statistics.mean(q["score"] for q in window)
            moving_averages.append(avg_quality)

        if len(moving_averages) >= 2:
            improvement = moving_averages[-1] - moving_averages[0]
            trend = "improving" if improvement > 0.05 else "declining" if improvement < -0.05 else "stable"
        else:
            trend = "insufficient_data"

        return {
            "trend": trend,
            "improvement": improvement if 'improvement' in locals() else 0,
            "moving_average_window": window_size
        }

class PerformanceDashboard:
    """Interactive performance dashboard and reporting"""

    def __init__(self, metrics: PerformanceMetrics, analyzer: PerformanceAnalyzer):
        self.metrics = metrics
        self.analyzer = analyzer
        self.reports_dir = Path("performance_reports")
        self.reports_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        print("📊 Generating Comprehensive Performance Report...")

        report = {
            "generated_at": datetime.now().isoformat(),
            "report_period": "last_24_hours",
            "summary": {},
            "detailed_analysis": {},
            "recommendations": [],
            "alerts": []
        }

        # Query performance analysis
        query_analysis = self.analyzer.analyze_query_performance(24)
        report["detailed_analysis"]["query_performance"] = query_analysis

        # Retrieval effectiveness
        retrieval_analysis = self.analyzer.analyze_retrieval_effectiveness()
        report["detailed_analysis"]["retrieval_effectiveness"] = retrieval_analysis

        # Content quality trends
        quality_analysis = self.analyzer.analyze_content_quality_trends()
        report["detailed_analysis"]["content_quality"] = quality_analysis

        # Generate summary
        report["summary"] = self._generate_summary(query_analysis, retrieval_analysis, quality_analysis)

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(
            query_analysis, retrieval_analysis, quality_analysis
        )

        # Check for alerts
        report["alerts"] = self._check_alerts(query_analysis, retrieval_analysis, quality_analysis)

        # Save report
        report_file = self.reports_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"💾 Report saved to: {report_file}")

        return report

    def _generate_summary(self, query_analysis: Dict, retrieval_analysis: Dict, quality_analysis: Dict) -> Dict[str, Any]:
        """Generate executive summary"""
        summary = {
            "overall_health": "unknown",
            "key_metrics": {},
            "performance_trend": "unknown",
            "critical_issues": []
        }

        # Determine overall health
        if "error" not in query_analysis:
            success_rate = query_analysis.get("success_rate", 0)
            avg_quality = query_analysis.get("avg_quality_score", 0)

            if success_rate > 0.8 and avg_quality > 0.7:
                summary["overall_health"] = "excellent"
            elif success_rate > 0.6 and avg_quality > 0.5:
                summary["overall_health"] = "good"
            elif success_rate > 0.4 and avg_quality > 0.3:
                summary["overall_health"] = "fair"
            else:
                summary["overall_health"] = "needs_improvement"

        # Key metrics
        summary["key_metrics"] = {
            "total_queries": query_analysis.get("total_queries", 0),
            "avg_response_time": query_analysis.get("avg_response_time", 0),
            "success_rate": query_analysis.get("success_rate", 0),
            "avg_quality_score": query_analysis.get("avg_quality_score", 0),
            "retrieval_efficiency": retrieval_analysis.get("retrieval_efficiency", 0),
            "content_quality": quality_analysis.get("avg_quality_score", 0)
        }

        # Performance trend
        if "performance_trend" in query_analysis:
            summary["performance_trend"] = query_analysis["performance_trend"]

        return summary

    def _generate_recommendations(self, query_analysis: Dict, retrieval_analysis: Dict, quality_analysis: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Query performance recommendations
        if query_analysis.get("avg_response_time", 0) > 2.0:
            recommendations.append("Optimize query processing pipeline to reduce response time")

        if query_analysis.get("success_rate", 0) < 0.7:
            recommendations.append("Improve retrieval quality and relevance scoring")

        # Retrieval effectiveness recommendations
        if retrieval_analysis.get("avg_precision", 0) < 0.6:
            recommendations.append("Enhance document ranking and filtering algorithms")

        if retrieval_analysis.get("retrieval_efficiency", 0) < 0.5:
            recommendations.append("Optimize retrieval indexing and search algorithms")

        # Content quality recommendations
        if quality_analysis.get("avg_quality_score", 0) < 0.6:
            recommendations.append("Implement stricter content quality filtering")

        quality_trend = quality_analysis.get("quality_improvement", {}).get("trend", "unknown")
        if quality_trend == "declining":
            recommendations.append("Review content acquisition sources and preprocessing")

        # Domain-specific recommendations
        domain_performance = query_analysis.get("domain_performance", {})
        for domain, stats in domain_performance.items():
            if stats.get("success_rate", 0) < 0.6:
                recommendations.append(f"Improve knowledge base for {domain} domain")

        # General recommendations
        recommendations.extend([
            "Implement continuous performance monitoring",
            "Add A/B testing for retrieval improvements",
            "Regular knowledge base quality audits",
            "User feedback integration for relevance assessment"
        ])

        return recommendations

    def _check_alerts(self, query_analysis: Dict, retrieval_analysis: Dict, quality_analysis: Dict) -> List[str]:
        """Check for critical performance alerts"""
        alerts = []

        # Critical alerts
        if query_analysis.get("success_rate", 1) < 0.3:
            alerts.append("CRITICAL: Query success rate below 30%")

        if query_analysis.get("avg_response_time", 0) > 10.0:
            alerts.append("CRITICAL: Average response time exceeds 10 seconds")

        if quality_analysis.get("avg_quality_score", 1) < 0.3:
            alerts.append("CRITICAL: Content quality score below acceptable threshold")

        # Warning alerts
        if query_analysis.get("performance_trend") == "declining":
            alerts.append("WARNING: Performance trending downward")

        if retrieval_analysis.get("avg_precision", 1) < 0.4:
            alerts.append("WARNING: Retrieval precision below 40%")

        return alerts

    def create_performance_visualization(self, report: Dict[str, Any]):
        """Create performance visualization charts"""
        try:
            # Create plots directory
            plots_dir = self.reports_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Domain performance chart
            domain_perf = report["detailed_analysis"]["query_performance"].get("domain_performance", {})
            if domain_perf:
                domains = list(domain_perf.keys())
                success_rates = [domain_perf[d]["success_rate"] for d in domains]
                quality_scores = [domain_perf[d]["avg_quality_score"] for d in domains]

                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.bar(domains, success_rates)
                plt.title("Success Rate by Domain")
                plt.xticks(rotation=45)
                plt.ylabel("Success Rate")

                plt.subplot(1, 2, 2)
                plt.bar(domains, quality_scores)
                plt.title("Quality Score by Domain")
                plt.xticks(rotation=45)
                plt.ylabel("Quality Score")

                plt.tight_layout()
                plt.savefig(plots_dir / "domain_performance.png", dpi=150, bbox_inches='tight')
                plt.close()

                print(f"📊 Performance visualization saved to: {plots_dir}/domain_performance.png")

        except Exception as e:
            print(f"⚠️ Failed to create visualizations: {e}")

class PerformanceMonitor:
    """Main performance monitoring orchestrator"""

    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.analyzer = PerformanceAnalyzer(self.metrics)
        self.dashboard = PerformanceDashboard(self.metrics, self.analyzer)

    def start_monitoring_session(self):
        """Start a new monitoring session"""
        self.metrics.current_session = datetime.now().isoformat()
        print(f"📊 Started performance monitoring session: {self.metrics.current_session}")

    def record_query_metrics(self, query: str, domain: str, start_time: float,
                           results_count: int, quality_score: float):
        """Record comprehensive query metrics"""
        response_time = time.time() - start_time

        self.metrics.record_query(
            query=query,
            domain=domain,
            response_time=response_time,
            results_count=results_count,
            quality_score=quality_score
        )

    def record_retrieval_metrics(self, query: str, retrieval_count: int,
                               avg_similarity: float, quality_matches: int,
                               response_time: float):
        """Record detailed retrieval metrics"""
        retrieval_stats = {
            "retrieval_count": retrieval_count,
            "avg_similarity": avg_similarity,
            "quality_matches": quality_matches,
            "response_time": response_time,
            "precision": quality_matches / retrieval_count if retrieval_count > 0 else 0
        }

        self.metrics.record_retrieval_performance(query, retrieval_stats)

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate and return comprehensive performance report"""
        return self.dashboard.generate_comprehensive_report()

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time performance statistics"""
        return {
            "current_session": self.metrics.current_session,
            "queries_today": len([
                q for q in self.metrics.metrics["queries"]
                if datetime.fromisoformat(q["timestamp"]).date() == datetime.now().date()
            ]),
            "avg_response_time": self._calculate_recent_avg("response_time", 1),
            "success_rate": self._calculate_recent_success_rate(1),
            "system_health": "operational"
        }

    def _calculate_recent_avg(self, metric_name: str, hours: int) -> float:
        """Calculate recent average for a metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_values = [
            q[metric_name] for q in self.metrics.metrics["queries"]
            if datetime.fromisoformat(q["timestamp"]) > cutoff_time
        ]

        return statistics.mean(recent_values) if recent_values else 0.0

    def _calculate_recent_success_rate(self, hours: int) -> float:
        """Calculate recent success rate"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_queries = [
            q for q in self.metrics.metrics["queries"]
            if datetime.fromisoformat(q["timestamp"]) > cutoff_time
        ]

        if not recent_queries:
            return 0.0

        successful = sum(1 for q in recent_queries if q["success"])
        return successful / len(recent_queries)

def main():
    """Demonstrate performance monitoring system"""
    print("📊 Performance Monitoring System Demonstration")
    print("=" * 60)

    # Initialize monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring_session()

    print("📝 Simulating performance data collection...")

    # Simulate some performance data
    test_queries = [
        ("What is machine learning?", "factual_knowledge", 1.2, 5, 0.8),
        ("How does sentiment analysis work?", "sentiment_analysis", 0.8, 3, 0.9),
        ("What are linguistic rules?", "linguistics", 1.5, 4, 0.7),
        ("How do computers understand meaning?", "semantic_analysis", 2.1, 2, 0.6),
        ("What causes emotions?", "sentiment_analysis", 0.9, 6, 0.8),
    ]

    for query, domain, response_time, results_count, quality_score in test_queries:
        monitor.record_query_metrics(query, domain, time.time() - response_time,
                                   results_count, quality_score)

        # Simulate retrieval metrics
        monitor.record_retrieval_metrics(
            query=query,
            retrieval_count=results_count,
            avg_similarity=0.75,
            quality_matches=int(results_count * quality_score),
            response_time=response_time
        )

        time.sleep(0.1)  # Small delay between recordings

    print("📊 Generating performance report...")

    # Generate comprehensive report
    report = monitor.generate_performance_report()

    # Display key insights
    summary = report["summary"]
    print(f"\n🏆 Overall Health: {summary['overall_health'].upper()}")
    print(f"📈 Performance Trend: {summary['performance_trend']}")
    print(f"🔢 Total Queries: {summary['key_metrics']['total_queries']}")
    print(".2f")
    print(".1%")
    print(".2f")

    if report["alerts"]:
        print(f"\n🚨 Alerts ({len(report['alerts'])}):")
        for alert in report["alerts"]:
            print(f"   • {alert}")

    print(f"\n💡 Recommendations ({len(report['recommendations'])}):")
    for rec in report["recommendations"][:5]:  # Show first 5
        print(f"   • {rec}")

    print("\n✅ Performance monitoring demonstration complete!")
    print("📈 System is now tracking performance metrics and generating optimization insights!")

if __name__ == "__main__":
    main()
