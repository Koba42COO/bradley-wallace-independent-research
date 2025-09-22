#!/usr/bin/env python3
"""
MONITORING SETUP
=================

Production monitoring system for the Enterprise prime aligned compute Platform.
Includes metrics collection, health checks, and alerting.
"""

import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

# Import centralized logging
try:
    from core_logging import get_platform_logger
    logger = get_platform_logger()
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

class SystemMetrics:
    """System metrics collection"""

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_health_check = None
        self.health_status = "unknown"

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total_bytes": psutil.virtual_memory().total,
                "available_bytes": psutil.virtual_memory().available,
                "used_bytes": psutil.virtual_memory().used,
                "usage_percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_bytes": psutil.disk_usage('/').total,
                "free_bytes": psutil.disk_usage('/').free,
                "used_bytes": psutil.disk_usage('/').used,
                "usage_percent": psutil.disk_usage('/').percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
                "packets_sent": psutil.net_io_counters().packets_sent,
                "packets_recv": psutil.net_io_counters().packets_recv
            },
            "application": {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "health_status": self.health_status
            }
        }

    def increment_request(self):
        """Increment request counter"""
        self.request_count += 1

    def increment_error(self):
        """Increment error counter"""
        self.error_count += 1

    def update_health_status(self, status: str):
        """Update health status"""
        self.health_status = status
        self.last_health_check = datetime.now()

class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self, metrics: SystemMetrics):
        self.metrics = metrics
        self.checks = []
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check("cpu_usage", self._check_cpu_usage)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_usage", self._check_disk_usage)
        self.register_check("network_connectivity", self._check_network_connectivity)
        self.register_check("application_health", self._check_application_health)

    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks.append({
            "name": name,
            "function": check_func,
            "last_result": None,
            "last_check": None
        })

    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage health"""
        cpu_percent = psutil.cpu_percent(interval=1)
        threshold = 80.0  # Configurable threshold

        return {
            "status": "healthy" if cpu_percent < threshold else "unhealthy",
            "value": cpu_percent,
            "threshold": threshold,
            "unit": "percent"
        }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage health"""
        memory = psutil.virtual_memory()
        threshold = 85.0  # Configurable threshold

        return {
            "status": "healthy" if memory.percent < threshold else "unhealthy",
            "value": memory.percent,
            "threshold": threshold,
            "unit": "percent",
            "details": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used
            }
        }

    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage health"""
        disk = psutil.disk_usage('/')
        threshold = 90.0  # Configurable threshold

        return {
            "status": "healthy" if disk.percent < threshold else "unhealthy",
            "value": disk.percent,
            "threshold": threshold,
            "unit": "percent",
            "details": {
                "total": disk.total,
                "free": disk.free,
                "used": disk.used
            }
        }

    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            # Simple connectivity check
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return {
                "status": "healthy",
                "value": "connected",
                "details": "DNS resolution successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "value": "disconnected",
                "details": str(e)
            }

    def _check_application_health(self) -> Dict[str, Any]:
        """Check application-specific health"""
        try:
            # Import and test core modules
            from enterprise_consciousness import ConsciousnessMathFramework
            cmf = ConsciousnessMathFramework()
            result = cmf.wallace_transform_proper(1.0)

            return {
                "status": "healthy",
                "value": "operational",
                "details": f"ConsciousnessMathFramework working, result: {result:.4f}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "value": "error",
                "details": str(e)
            }

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks asynchronously"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {},
            "duration_ms": 0
        }

        start_time = time.time()

        # Run checks concurrently
        tasks = []
        for check in self.checks:
            task = asyncio.create_task(self._run_single_check(check))
            tasks.append(task)

        check_results = await asyncio.gather(*tasks)

        # Process results
        for i, result in enumerate(check_results):
            check_name = self.checks[i]["name"]
            results["checks"][check_name] = result

            # Update check metadata
            self.checks[i]["last_result"] = result
            self.checks[i]["last_check"] = datetime.now()

            # Update overall status
            if result["status"] == "unhealthy":
                results["overall_status"] = "unhealthy"

        results["duration_ms"] = (time.time() - start_time) * 1000

        # Update metrics
        self.metrics.update_health_status(results["overall_status"])

        logger.info(f"Health check completed: {results['overall_status']} "
                   f"({len(results['checks'])} checks in {results['duration_ms']:.2f}ms)")

        return results

    async def _run_single_check(self, check: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single health check"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, check["function"]
            )
            return result
        except Exception as e:
            logger.error(f"Health check '{check['name']}' failed: {e}")
            return {
                "status": "error",
                "value": "check_failed",
                "details": str(e)
            }

class AlertManager:
    """Alert management system"""

    def __init__(self, metrics: SystemMetrics):
        self.metrics = metrics
        self.alerts = []
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 0.05  # 5% error rate
        }

    def check_alerts(self, health_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions and generate alerts"""
        new_alerts = []

        # Check system metrics
        for check_name, check_result in health_results["checks"].items():
            if check_result["status"] == "unhealthy":
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "system_alert",
                    "severity": "warning",
                    "component": check_name,
                    "message": f"{check_name} is unhealthy",
                    "details": check_result
                }
                new_alerts.append(alert)
                logger.warning(f"System alert: {check_name} is unhealthy")

        # Check error rate
        error_rate = self.metrics.error_count / max(self.metrics.request_count, 1)
        if error_rate > self.alert_thresholds["error_rate"]:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "application_alert",
                "severity": "error",
                "component": "error_rate",
                "message": ".2f",
                "details": {
                    "error_rate": error_rate,
                    "error_count": self.metrics.error_count,
                    "request_count": self.metrics.request_count
                }
            }
            new_alerts.append(alert)
            logger.error(".2f")

        self.alerts.extend(new_alerts)
        return new_alerts

class MonitoringSystem:
    """Complete monitoring system"""

    def __init__(self):
        self.metrics = SystemMetrics()
        self.health_checker = HealthChecker(self.metrics)
        self.alert_manager = AlertManager(self.metrics)
        self.is_running = False

    async def start_monitoring(self, interval_seconds: int = 30):
        """Start the monitoring system"""
        self.is_running = True
        logger.info(f"Starting monitoring system with {interval_seconds}s intervals")

        while self.is_running:
            try:
                # Collect metrics
                system_metrics = self.metrics.collect_system_metrics()

                # Run health checks
                health_results = await self.health_checker.run_health_checks()

                # Check for alerts
                alerts = self.alert_manager.check_alerts(health_results)

                # Log summary
                logger.info(f"Monitoring cycle completed - "
                           f"Health: {health_results['overall_status']}, "
                           f"Alerts: {len(alerts)}, "
                           f"CPU: {system_metrics['cpu']['usage_percent']:.1f}%, "
                           f"Memory: {system_metrics['memory']['usage_percent']:.1f}%")

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(interval_seconds)

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        logger.info("Monitoring system stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "is_running": self.is_running,
            "metrics": self.metrics.collect_system_metrics(),
            "active_checks": len(self.health_checker.checks),
            "active_alerts": len(self.alert_manager.alerts),
            "uptime_seconds": time.time() - self.metrics.start_time
        }

# Global monitoring instance
_monitoring_instance: Optional[MonitoringSystem] = None

def get_monitoring_system() -> MonitoringSystem:
    """Get global monitoring system instance"""
    global _monitoring_instance
    if _monitoring_instance is None:
        _monitoring_instance = MonitoringSystem()
    return _monitoring_instance

async def run_health_check() -> Dict[str, Any]:
    """Run a single health check"""
    monitoring = get_monitoring_system()
    return await monitoring.health_checker.run_health_checks()

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    monitoring = get_monitoring_system()
    return monitoring.metrics.collect_system_metrics()

def increment_request_count():
    """Increment request counter for metrics"""
    monitoring = get_monitoring_system()
    monitoring.metrics.increment_request()

def increment_error_count():
    """Increment error counter for metrics"""
    monitoring = get_monitoring_system()
    monitoring.metrics.increment_error()

if __name__ == "__main__":
    # Demo monitoring system
    async def demo():
        monitoring = get_monitoring_system()

        print("🩺 Enterprise prime aligned compute Platform - Monitoring System")
        print("=" * 60)

        # Run initial health check
        print("Running initial health check...")
        health_results = await run_health_check()

        print(f"Overall Health: {health_results['overall_status'].upper()}")
        print(f"Checks Run: {len(health_results['checks'])}")
        print(f"Duration: {health_results['duration_ms']:.2f}ms")
        print()

        # Show individual check results
        print("Individual Check Results:")
        for check_name, result in health_results['checks'].items():
            status_icon = "✅" if result['status'] == 'healthy' else "❌"
            print(f"  {status_icon} {check_name}: {result['status']}")

        print()
        print("System Metrics:")
        metrics = get_system_metrics()
        print(f"  CPU Usage: {metrics['cpu']['usage_percent']:.1f}%")
        print(f"  Memory Usage: {metrics['memory']['usage_percent']:.1f}%")
        print(f"  Disk Usage: {metrics['disk']['usage_percent']:.1f}%")
        print(f"  Uptime: {metrics['uptime_seconds']:.0f} seconds")

        print()
        print("🎯 Monitoring system demo completed!")
        print("Run with: python monitoring_setup.py")

    # Run demo
    asyncio.run(demo())
