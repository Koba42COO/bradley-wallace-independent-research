#!/usr/bin/env python3
"""
SquashPlot Monitoring and Observability
Provides metrics collection, logging, and health monitoring
"""

import os
import time
import psutil
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float]
    active_processes: int
    network_connections: int

@dataclass
class ApplicationMetrics:
    timestamp: datetime
    active_jobs: int
    pending_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_plots_created: int
    avg_plot_time_minutes: float
    error_rate_percent: float
    uptime_seconds: float

class MetricsCollector:
    """Collects and stores system and application metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.system_metrics: List[SystemMetrics] = []
        self.app_metrics: List[ApplicationMetrics] = []
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start metrics collection in background"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
    
    def _collect_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect metrics every 30 seconds
                self._collect_system_metrics()
                self._collect_app_metrics()
                self._cleanup_old_metrics()
                time.sleep(30)
            except Exception as e:
                print(f"Metrics collection error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(os.getloadavg())
            except (OSError, AttributeError):
                load_avg = [0.0, 0.0, 0.0]
            
            # Process count
            active_processes = len(psutil.pids())
            
            # Network connections
            try:
                network_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                network_connections = 0
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / (1024**3),
                load_average=load_avg,
                active_processes=active_processes,
                network_connections=network_connections
            )
            
            with self.lock:
                self.system_metrics.append(metrics)
                
        except Exception as e:
            print(f"System metrics collection error: {e}")
    
    def _collect_app_metrics(self):
        """Collect application-level metrics"""
        try:
            from job_queue import job_queue
            
            # Job statistics
            stats = job_queue.get_queue_stats()
            
            # Calculate averages and rates
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Get recent jobs for calculations
            recent_jobs = job_queue.get_jobs(limit=100)
            completed_jobs = [j for j in recent_jobs if j.status.value == 'completed']
            failed_jobs = [j for j in recent_jobs if j.status.value == 'failed']
            
            # Calculate average plot time
            avg_plot_time = 0.0
            if completed_jobs:
                plot_times = []
                for job in completed_jobs:
                    if job.start_time and job.end_time:
                        duration = (job.end_time - job.start_time).total_seconds() / 60
                        plot_times.append(duration)
                
                avg_plot_time = sum(plot_times) / len(plot_times) if plot_times else 0.0
            
            # Calculate error rate
            total_finished = len(completed_jobs) + len(failed_jobs)
            error_rate = (len(failed_jobs) / total_finished * 100) if total_finished > 0 else 0.0
            
            metrics = ApplicationMetrics(
                timestamp=datetime.now(),
                active_jobs=stats.get('running', 0),
                pending_jobs=stats.get('pending', 0),
                completed_jobs=stats.get('completed', 0),
                failed_jobs=stats.get('failed', 0),
                total_plots_created=stats.get('completed', 0),
                avg_plot_time_minutes=avg_plot_time,
                error_rate_percent=error_rate,
                uptime_seconds=uptime
            )
            
            with self.lock:
                self.app_metrics.append(metrics)
                
        except Exception as e:
            print(f"Application metrics collection error: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            self.system_metrics = [m for m in self.system_metrics if m.timestamp > cutoff]
            self.app_metrics = [m for m in self.app_metrics if m.timestamp > cutoff]
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics"""
        with self.lock:
            return self.system_metrics[-1] if self.system_metrics else None
    
    def get_latest_app_metrics(self) -> Optional[ApplicationMetrics]:
        """Get the most recent application metrics"""
        with self.lock:
            return self.app_metrics[-1] if self.app_metrics else None
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of recent metrics"""
        with self.lock:
            latest_system = self.system_metrics[-1] if self.system_metrics else None
            latest_app = self.app_metrics[-1] if self.app_metrics else None
            
            return {
                'system': asdict(latest_system) if latest_system else None,
                'application': asdict(latest_app) if latest_app else None,
                'collection_stats': {
                    'system_samples': len(self.system_metrics),
                    'app_samples': len(self.app_metrics),
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
                }
            }
    
    def get_time_series(self, hours: int = 1) -> Dict:
        """Get time series data for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_system = [asdict(m) for m in self.system_metrics if m.timestamp > cutoff]
            recent_app = [asdict(m) for m in self.app_metrics if m.timestamp > cutoff]
            
            # Convert datetime objects to ISO strings
            for metrics in recent_system:
                metrics['timestamp'] = metrics['timestamp'].isoformat()
            for metrics in recent_app:
                metrics['timestamp'] = metrics['timestamp'].isoformat()
            
            return {
                'system_metrics': recent_system,
                'application_metrics': recent_app,
                'time_range_hours': hours
            }

# Global metrics collector instance
metrics_collector = MetricsCollector()

def start_monitoring():
    """Start the monitoring system"""
    metrics_collector.start()
    print("✅ Monitoring system started")

def stop_monitoring():
    """Stop the monitoring system"""
    metrics_collector.stop()
    print("✅ Monitoring system stopped")

def get_health_status() -> Dict:
    """Get overall health status"""
    latest_system = metrics_collector.get_latest_system_metrics()
    latest_app = metrics_collector.get_latest_app_metrics()
    
    # Health checks
    health_issues = []
    
    if latest_system:
        if latest_system.cpu_percent > 90:
            health_issues.append("High CPU usage")
        if latest_system.memory_percent > 90:
            health_issues.append("High memory usage")
        if latest_system.disk_usage_percent > 95:
            health_issues.append("Low disk space")
    
    if latest_app:
        if latest_app.error_rate_percent > 20:
            health_issues.append("High error rate")
    
    # Overall status
    if not health_issues:
        status = "healthy"
    elif len(health_issues) <= 2:
        status = "warning"
    else:
        status = "critical"
    
    return {
        'status': status,
        'issues': health_issues,
        'system_metrics': asdict(latest_system) if latest_system else None,
        'app_metrics': asdict(latest_app) if latest_app else None,
        'timestamp': datetime.now().isoformat()
    }