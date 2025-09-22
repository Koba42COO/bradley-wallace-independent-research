#!/usr/bin/env python3
"""
Monitoring Dashboard - Real-time Analytics and System Monitoring
=================================================================
Comprehensive monitoring dashboard with real-time analytics, system metrics,
performance monitoring, and interactive visualizations for the chAIos platform.
"""

import os
import json
import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import deque
import psutil
import socket
import requests

from fastapi import FastAPI, Request, WebSocket, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import websockets

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """Real-time monitoring dashboard for the chAIos platform"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI(title="chAIos Monitoring Dashboard", version="1.0.0")

        # Data storage
        self.metrics_history = deque(maxlen=1000)  # Last 1000 data points
        self.alerts = []
        self.system_info = {}
        self.service_status = {}

        # Real-time connections
        self.websocket_connections = set()

        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None

        # Setup FastAPI
        self._setup_middleware()
        self._setup_routes()
        self._setup_static_files()

        # Initialize monitoring
        self._initialize_monitoring()

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Main dashboard page"""
            return self._get_dashboard_html()

        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current system metrics"""
            return self._get_current_metrics()

        @self.app.get("/api/metrics/history")
        async def get_metrics_history(limit: int = 100):
            """Get metrics history"""
            return list(self.metrics_history)[-limit:]

        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get active alerts"""
            return self.alerts[-50:]  # Last 50 alerts

        @self.app.get("/api/system/info")
        async def get_system_info():
            """Get system information"""
            return self.system_info

        @self.app.get("/api/services/status")
        async def get_services_status():
            """Get service status"""
            return self.service_status

        @self.app.websocket("/ws/metrics")
        async def websocket_metrics(websocket: WebSocket):
            """WebSocket for real-time metrics"""
            await websocket.accept()
            self.websocket_connections.add(websocket)
            try:
                while True:
                    # Send current metrics every second
                    metrics = self._get_current_metrics()
                    await websocket.send_json(metrics)
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_connections.remove(websocket)

        @self.app.post("/api/alerts/clear")
        async def clear_alerts():
            """Clear all alerts"""
            self.alerts.clear()
            return {"status": "cleared"}

        @self.app.get("/api/health")
        async def health_check():
            """Dashboard health check"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "connections": len(self.websocket_connections),
                "metrics_points": len(self.metrics_history)
            }

    def _setup_static_files(self):
        """Setup static file serving"""
        # Create static directory if it doesn't exist
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(static_dir, exist_ok=True)

        # Create CSS and JS files
        self._create_static_files(static_dir)

        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def _create_static_files(self, static_dir: str):
        """Create CSS and JavaScript files for the dashboard"""

        # CSS
        css_content = """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f23;
            color: #ffffff;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            border: 1px solid #16213e;
            transition: transform 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        .metric-title {
            font-size: 0.9em;
            color: #a0a0a0;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #00d4ff;
            margin-bottom: 5px;
        }

        .metric-subtitle {
            font-size: 0.8em;
            color: #888;
        }

        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .chart-card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            border: 1px solid #16213e;
        }

        .chart-title {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #00d4ff;
        }

        .chart-container {
            height: 300px;
            position: relative;
        }

        .services-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }

        .service-card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            border: 1px solid #16213e;
        }

        .service-name {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .service-status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }

        .status-healthy { background: #28a745; color: white; }
        .status-unhealthy { background: #dc3545; color: white; }
        .status-warning { background: #ffc107; color: #212529; }

        .alerts-container {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            border: 1px solid #16213e;
        }

        .alerts-title {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #00d4ff;
        }

        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }

        .alert-error { border-left-color: #dc3545; background: rgba(220, 53, 69, 0.1); }
        .alert-warning { border-left-color: #ffc107; background: rgba(255, 193, 7, 0.1); }
        .alert-info { border-left-color: #17a2b8; background: rgba(23, 162, 184, 0.1); }

        .alert-time {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 2px;
        }

        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
        }

        @media (max-width: 768px) {
            .charts-container {
                grid-template-columns: 1fr;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 2em;
            }
        }
        """

        with open(os.path.join(static_dir, "dashboard.css"), "w") as f:
            f.write(css_content)

        # JavaScript
        js_content = """
        class MonitoringDashboard {
            constructor() {
                this.charts = {};
                this.ws = null;
                this.init();
            }

            init() {
                this.connectWebSocket();
                this.loadInitialData();
                this.setupCharts();
                this.startPeriodicUpdates();
            }

            connectWebSocket() {
                this.ws = new WebSocket(`ws://${window.location.host}/ws/metrics`);

                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.updateDashboard(data);
                };

                this.ws.onclose = () => {
                    setTimeout(() => this.connectWebSocket(), 5000);
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }

            async loadInitialData() {
                try {
                    const [metrics, alerts, systemInfo, services] = await Promise.all([
                        fetch('/api/metrics').then(r => r.json()),
                        fetch('/api/alerts').then(r => r.json()),
                        fetch('/api/system/info').then(r => r.json()),
                        fetch('/api/services/status').then(r => r.json())
                    ]);

                    this.updateMetrics(metrics);
                    this.updateAlerts(alerts);
                    this.updateSystemInfo(systemInfo);
                    this.updateServices(services);
                } catch (error) {
                    console.error('Failed to load initial data:', error);
                }
            }

            setupCharts() {
                // CPU Chart
                this.charts.cpu = this.createChart('cpuChart', 'CPU Usage', '#00d4ff');

                // Memory Chart
                this.charts.memory = this.createChart('memoryChart', 'Memory Usage', '#28a745');

                // Network Chart
                this.charts.network = this.createChart('networkChart', 'Network I/O', '#ffc107');

                // Disk Chart
                this.charts.disk = this.createChart('diskChart', 'Disk I/O', '#dc3545');
            }

            createChart(containerId, label, color) {
                const ctx = document.getElementById(containerId);
                if (!ctx) return null;

                return new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: label,
                            data: [],
                            borderColor: color,
                            backgroundColor: color + '20',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: { color: '#333' },
                                ticks: { color: '#fff' }
                            },
                            x: {
                                grid: { color: '#333' },
                                ticks: { color: '#fff' }
                            }
                        },
                        plugins: {
                            legend: {
                                labels: { color: '#fff' }
                            }
                        }
                    }
                });
            }

            updateDashboard(data) {
                this.updateMetrics(data);
                this.updateCharts(data);
            }

            updateMetrics(data) {
                // Update metric values
                this.updateMetric('cpuValue', data.cpu_percent, '%');
                this.updateMetric('memoryValue', data.memory_percent, '%');
                this.updateMetric('diskValue', data.disk_usage, '%');
                this.updateMetric('networkValue', data.network_connections, '');

                // Update request metrics
                this.updateMetric('totalRequests', data.total_requests || 0, '');
                this.updateMetric('activeConnections', data.active_connections || 0, '');
                this.updateMetric('responseTime', data.avg_response_time || 0, 'ms');
            }

            updateMetric(elementId, value, unit) {
                const element = document.getElementById(elementId);
                if (element) {
                    element.textContent = typeof value === 'number' ? value.toFixed(1) + unit : value + unit;
                }
            }

            updateCharts(data) {
                const now = new Date().toLocaleTimeString();

                // Update CPU chart
                if (this.charts.cpu) {
                    this.updateChart(this.charts.cpu, now, data.cpu_percent);
                }

                // Update Memory chart
                if (this.charts.memory) {
                    this.updateChart(this.charts.memory, now, data.memory_percent);
                }
            }

            updateChart(chart, label, value) {
                chart.data.labels.push(label);
                chart.data.datasets[0].data.push(value);

                // Keep only last 20 points
                if (chart.data.labels.length > 20) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                }

                chart.update();
            }

            updateAlerts(alerts) {
                const container = document.getElementById('alertsContainer');
                if (!container) return;

                container.innerHTML = alerts.map(alert => `
                    <div class="alert-item alert-${alert.level || 'info'}">
                        <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                        <div>${alert.message}</div>
                    </div>
                `).join('');
            }

            updateSystemInfo(info) {
                const container = document.getElementById('systemInfo');
                if (!container) return;

                container.innerHTML = `
                    <div><strong>OS:</strong> ${info.os || 'Unknown'}</div>
                    <div><strong>Python:</strong> ${info.python_version || 'Unknown'}</div>
                    <div><strong>CPU Cores:</strong> ${info.cpu_count || 'Unknown'}</div>
                    <div><strong>Total Memory:</strong> ${info.total_memory || 'Unknown'}</div>
                `;
            }

            updateServices(services) {
                const container = document.getElementById('servicesContainer');
                if (!container) return;

                container.innerHTML = Object.entries(services).map(([name, service]) => `
                    <div class="service-card">
                        <div class="service-name">${name}</div>
                        <span class="service-status status-${service.status || 'unknown'}">
                            ${service.status || 'unknown'}
                        </span>
                        <div>Port: ${service.port || 'N/A'}</div>
                        <div>Requests: ${service.requests || 0}</div>
                    </div>
                `).join('');
            }

            startPeriodicUpdates() {
                // Update alerts and services every 30 seconds
                setInterval(async () => {
                    try {
                        const [alerts, services] = await Promise.all([
                            fetch('/api/alerts').then(r => r.json()),
                            fetch('/api/services/status').then(r => r.json())
                        ]);

                        this.updateAlerts(alerts);
                        this.updateServices(services);
                    } catch (error) {
                        console.error('Failed to update data:', error);
                    }
                }, 30000);
            }
        }

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new MonitoringDashboard();
        });
        """

        with open(os.path.join(static_dir, "dashboard.js"), "w") as f:
            f.write(js_content)

    def _get_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>chAIos Monitoring Dashboard</title>
    <link rel="stylesheet" href="/static/dashboard.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 chAIos Monitoring Dashboard</h1>
            <p>Real-time system monitoring and analytics</p>
        </div>

        <!-- System Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">CPU Usage</div>
                <div class="metric-value" id="cpuValue">--</div>
                <div class="metric-subtitle">Current utilization</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">Memory Usage</div>
                <div class="metric-value" id="memoryValue">--</div>
                <div class="metric-subtitle">RAM utilization</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">Disk Usage</div>
                <div class="metric-value" id="diskValue">--</div>
                <div class="metric-subtitle">Storage utilization</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">Network</div>
                <div class="metric-value" id="networkValue">--</div>
                <div class="metric-subtitle">Active connections</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">Total Requests</div>
                <div class="metric-value" id="totalRequests">--</div>
                <div class="metric-subtitle">Since startup</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">Active Connections</div>
                <div class="metric-value" id="activeConnections">--</div>
                <div class="metric-subtitle">Current sessions</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">Avg Response Time</div>
                <div class="metric-value" id="responseTime">--</div>
                <div class="metric-subtitle">API performance</div>
            </div>

            <div class="metric-card">
                <div class="metric-title">System Uptime</div>
                <div class="metric-value" id="uptime">--</div>
                <div class="metric-subtitle">Time since restart</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="charts-container">
            <div class="chart-card">
                <div class="chart-title">CPU Usage Over Time</div>
                <div class="chart-container">
                    <canvas id="cpuChart"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <div class="chart-title">Memory Usage Over Time</div>
                <div class="chart-container">
                    <canvas id="memoryChart"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <div class="chart-title">Network I/O</div>
                <div class="chart-container">
                    <canvas id="networkChart"></canvas>
                </div>
            </div>

            <div class="chart-card">
                <div class="chart-title">Disk I/O</div>
                <div class="chart-container">
                    <canvas id="diskChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Services Status -->
        <div class="services-grid" id="servicesContainer">
            <!-- Services will be loaded here -->
        </div>

        <!-- System Information and Alerts -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div class="chart-card">
                <div class="chart-title">System Information</div>
                <div id="systemInfo">
                    Loading system information...
                </div>
            </div>

            <div class="alerts-container">
                <div class="alerts-title">Recent Alerts</div>
                <div id="alertsContainer">
                    No recent alerts
                </div>
            </div>
        </div>

        <div class="footer">
            <p>chAIos Monitoring Dashboard • Real-time system monitoring</p>
        </div>
    </div>

    <script src="/static/dashboard.js"></script>
</body>
</html>
        """

    def _initialize_monitoring(self):
        """Initialize system monitoring"""
        self.system_info = self._get_system_info()
        self._start_monitoring()

    def _start_monitoring(self):
        """Start background monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_system_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Check for alerts
                self._check_alerts(metrics)

                # Update service status
                self._update_service_status()

                # Broadcast to WebSocket connections
                asyncio.run(self._broadcast_metrics(metrics))

                time.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_used': psutil.virtual_memory().used,
            'disk_total': psutil.disk_usage('/').total,
            'disk_used': psutil.disk_usage('/').used,
        }

        # Add network I/O
        net_io = psutil.net_io_counters()
        metrics.update({
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv,
            'network_packets_sent': net_io.packets_sent,
            'network_packets_recv': net_io.packets_recv,
        })

        # Add disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.update({
                'disk_read_bytes': disk_io.read_bytes,
                'disk_write_bytes': disk_io.write_bytes,
                'disk_read_count': disk_io.read_count,
                'disk_write_count': disk_io.write_count,
            })

        # Add platform-specific metrics
        try:
            metrics.update({
                'total_requests': getattr(self, 'total_requests', 0),
                'active_connections': len(getattr(self, 'websocket_connections', set())),
                'avg_response_time': getattr(self, 'avg_response_time', 0),
            })
        except:
            pass

        return metrics

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            import platform
            import sys

            return {
                'hostname': socket.gethostname(),
                'os': platform.system(),
                'os_version': platform.version(),
                'architecture': platform.machine(),
                'python_version': sys.version.split()[0],
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'total_memory': self._format_bytes(psutil.virtual_memory().total),
                'total_disk': self._format_bytes(psutil.disk_usage('/').total),
                'uptime': self._get_system_uptime(),
            }
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return ".1f"
            bytes_value /= 1024.0
        return ".1f"

    def _get_system_uptime(self) -> str:
        """Get system uptime"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            uptime_str = str(timedelta(seconds=int(uptime_seconds)))
            return uptime_str
        except:
            return "Unknown"

    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check for system alerts"""
        alerts = []

        # CPU usage alert
        if metrics.get('cpu_percent', 0) > 90:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'level': 'error',
                'message': '.1f'
            })

        # Memory usage alert
        if metrics.get('memory_percent', 0) > 90:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'level': 'error',
                'message': '.1f'
            })

        # Disk usage alert
        if metrics.get('disk_usage', 0) > 90:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'level': 'warning',
                'message': '.1f'
            })

        # Network connections alert
        if metrics.get('network_connections', 0) > 1000:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'level': 'warning',
                'message': f"High network connections: {metrics['network_connections']}"
            })

        # Add alerts to list
        self.alerts.extend(alerts)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

    def _update_service_status(self):
        """Update service status information"""
        # Check common service endpoints
        services_to_check = {
            'api_gateway': {'url': 'http://localhost:8000/health', 'port': 8000},
            'knowledge_system': {'url': 'http://localhost:8003/health', 'port': 8003},
            'polymath_brain': {'url': 'http://localhost:8004/health', 'port': 8004},
            'cudnt_accelerator': {'url': 'http://localhost:8005/health', 'port': 8005},
        }

        for service_name, config in services_to_check.items():
            status = self._check_service_health(config['url'])
            self.service_status[service_name] = {
                'name': service_name.replace('_', ' ').title(),
                'status': status,
                'port': config['port'],
                'url': config['url'],
                'requests': getattr(self, f'{service_name}_requests', 0),
                'last_check': datetime.now().isoformat()
            }

    def _check_service_health(self, url: str) -> str:
        """Check service health"""
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return 'healthy'
            else:
                return 'unhealthy'
        except:
            return 'unhealthy'

    async def _broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast metrics to WebSocket connections"""
        if not self.websocket_connections:
            return

        # Remove dead connections
        self.websocket_connections = {ws for ws in self.websocket_connections if not ws.closed}

        # Broadcast to active connections
        for websocket in self.websocket_connections.copy():
            try:
                await websocket.send_json(metrics)
            except Exception as e:
                logger.error(f"Failed to send metrics to websocket: {e}")
                self.websocket_connections.remove(websocket)

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        else:
            return self._collect_system_metrics()

    def run(self):
        """Run the monitoring dashboard"""
        logger.info(f"Starting Monitoring Dashboard on {self.host}:{self.port}")

        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("Monitoring Dashboard stopped")
        finally:
            self.monitoring_active = False

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='chAIos Monitoring Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')

    args = parser.parse_args()

    dashboard = MonitoringDashboard(host=args.host, port=args.port)
    dashboard.run()

if __name__ == "__main__":
    main()
