#!/usr/bin/env python3
"""
SquashPlot Dashboard - Web Monitoring Interface for Chia Farming
===============================================================

Real-time web dashboard for monitoring Chia blockchain farming operations.
Provides comprehensive analytics, performance monitoring, and management tools.

Features:
- Real-time farming statistics
- Plot distribution visualization
- Performance analytics
- Resource utilization monitoring
- Cost analysis and optimization
- Automated alerts and notifications

Author: Bradley Wallace (COO, Koba42 Corp)
Contact: user@domain.com
License: MIT License
"""

import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
import logging

# Import our farming manager
from squashplot_chia_system import ChiaFarmingManager, OptimizationMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('squashplot_dashboard')

class SquashPlotDashboard:
    """Web dashboard for SquashPlot farming management"""

    def __init__(self, farming_manager: ChiaFarmingManager):
        self.farming_manager = farming_manager
        self.app = Flask(__name__,
                        template_folder='templates',
                        static_folder='static')
        CORS(self.app)

        # Dashboard data
        self.dashboard_data = {}
        self.alerts = []
        self.performance_history = []

        # Setup routes
        self._setup_routes()

        # Start data collection thread
        self.data_thread = threading.Thread(target=self._collect_dashboard_data, daemon=True)
        self.data_thread.start()

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')

        @self.app.route('/api/dashboard')
        def get_dashboard():
            """Get complete dashboard data"""
            return jsonify(self._get_dashboard_data())

        @self.app.route('/api/stats')
        def get_stats():
            """Get farming statistics"""
            return jsonify(self.farming_manager.get_farming_report())

        @self.app.route('/api/plots')
        def get_plots():
            """Get plot information"""
            plots_data = [plot.__dict__ for plot in self.farming_manager.plots]
            return jsonify(plots_data)

        @self.app.route('/api/resources')
        def get_resources():
            """Get resource utilization"""
            resources = self.farming_manager.resource_monitor.get_resources()
            return jsonify(resources.__dict__)

        @self.app.route('/api/optimize', methods=['POST'])
        def optimize():
            """Change optimization mode"""
            data = request.get_json()
            mode = data.get('mode', 'middle')

            try:
                self.farming_manager.optimization_mode = OptimizationMode(mode)
                self.farming_manager._set_optimization_parameters()

                # Add alert
                self._add_alert(f"Optimization mode changed to {mode.upper()}", "info")

                return jsonify({
                    'status': 'success',
                    'message': f'Optimization mode changed to {mode}',
                    'mode': mode
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 400

        @self.app.route('/api/alerts')
        def get_alerts():
            """Get recent alerts"""
            return jsonify(self.alerts[-10:])  # Last 10 alerts

        @self.app.route('/api/performance')
        def get_performance():
            """Get performance history"""
            return jsonify(self.performance_history[-100:])  # Last 100 data points

        @self.app.route('/api/export')
        def export_data():
            """Export dashboard data"""
            data = self._get_dashboard_data()
            filename = f"squashplot_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            return Response(
                json.dumps(data, indent=2, default=str),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment;filename={filename}'}
            )

    def _collect_dashboard_data(self):
        """Collect dashboard data in background"""
        while True:
            try:
                # Update dashboard data
                self.dashboard_data = self._get_dashboard_data()

                # Check for alerts
                self._check_alerts()

                # Store performance history
                self._store_performance_data()

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Dashboard data collection error: {e}")
                time.sleep(60)

    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        farming_report = self.farming_manager.get_farming_report()

        return {
            'timestamp': datetime.now().isoformat(),
            'farming_stats': farming_report['farming_stats'],
            'system_resources': farming_report['system_resources'],
            'optimization_mode': self.farming_manager.optimization_mode.value,
            'plots_summary': {
                'total_plots': len(self.farming_manager.plots),
                'total_size_gb': sum(p.size_gb for p in self.farming_manager.plots),
                'active_plots': sum(1 for p in self.farming_manager.plots if p.farming_status == "active"),
                'avg_quality': sum(p.quality_score for p in self.farming_manager.plots) / len(self.farming_manager.plots) if self.farming_manager.plots else 0
            },
            'alerts': self.alerts[-5:],  # Last 5 alerts
            'performance_trends': self._calculate_performance_trends(),
            'recommendations': farming_report.get('recommendations', [])
        }

    def _check_alerts(self):
        """Check for system alerts"""
        resources = self.farming_manager.resource_monitor.get_resources()

        # CPU usage alert
        if resources.cpu_usage > 90:
            self._add_alert("High CPU usage detected (>90%)", "warning")

        # Memory usage alert
        if resources.memory_usage > 85:
            self._add_alert("High memory usage detected (>85%)", "warning")

        # Disk space alert
        for drive, usage in resources.disk_usage.items():
            if usage > 95:
                self._add_alert(f"Critical disk space on {drive} (>95%)", "critical")

        # Farming alerts
        farming_stats = self.farming_manager.farming_stats
        if farming_stats.total_plots == 0:
            self._add_alert("No plots found - farming not active", "warning")

        if farming_stats.proofs_found_24h == 0 and farming_stats.total_plots > 0:
            self._add_alert("No proofs found in last 24 hours", "info")

    def _add_alert(self, message: str, level: str):
        """Add an alert to the system"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'level': level,
            'id': len(self.alerts)
        }

        self.alerts.append(alert)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        logger.info(f"Alert added: {message} ({level})")

    def _store_performance_data(self):
        """Store performance data for trending"""
        data_point = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': self.farming_manager.resource_monitor.get_resources().cpu_usage,
            'memory_usage': self.farming_manager.resource_monitor.get_resources().memory_usage,
            'active_plots': sum(1 for p in self.farming_manager.plots if p.farming_status == "active"),
            'total_plots': len(self.farming_manager.plots)
        }

        self.performance_history.append(data_point)

        # Keep only last YYYY STREET NAME
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}

        recent = self.performance_history[-10:]  # Last 10 data points
        older = self.performance_history[-20:-10]  # Previous 10

        if not older:
            return {'trend': 'insufficient_data'}

        # Calculate trends
        cpu_trend = sum(r['cpu_usage'] for r in recent) / len(recent) - \
                   sum(o['cpu_usage'] for o in older) / len(older)

        memory_trend = sum(r['memory_usage'] for r in recent) / len(recent) - \
                      sum(o['memory_usage'] for o in older) / len(older)

        return {
            'cpu_trend': round(cpu_trend, 2),
            'memory_trend': round(memory_trend, 2),
            'trend_direction': 'improving' if cpu_trend < 0 else 'degrading',
            'data_points': len(self.performance_history)
        }

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard server"""
        logger.info(f"Starting SquashPlot Dashboard on {host}:{port}")

        # Create templates directory if it doesn't exist
        self._create_templates()

        self.app.run(host=host, port=port, debug=debug)

    def _create_templates(self):
        """Create HTML templates for the dashboard"""
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        static_dir = os.path.join(os.path.dirname(__file__), 'static')

        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)

        # Create dashboard template
        dashboard_html = self._create_dashboard_template()
        with open(os.path.join(templates_dir, 'dashboard.html'), 'w') as f:
            f.write(dashboard_html)

        # Create CSS
        dashboard_css = self._create_dashboard_css()
        with open(os.path.join(static_dir, 'dashboard.css'), 'w') as f:
            f.write(dashboard_css)

        # Create JavaScript
        dashboard_js = self._create_dashboard_js()
        with open(os.path.join(static_dir, 'dashboard.js'), 'w') as f:
            f.write(dashboard_js)

    def _create_dashboard_template(self) -> str:
        """Create the main dashboard HTML template"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SquashPlot - Chia Farming Dashboard</title>
    <link rel="stylesheet" href="/static/dashboard.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="dashboard">
        <header class="dashboard-header">
            <h1>🍃 SquashPlot Dashboard</h1>
            <p class="subtitle">Chia Blockchain Farming Management System</p>
            <div class="status-bar">
                <span id="last-update">Loading...</span>
                <span id="optimization-mode">Mode: --</span>
            </div>
        </header>

        <div class="dashboard-grid">
            <!-- Farming Statistics -->
            <div class="card">
                <h3>🌾 Farming Statistics</h3>
                <div class="stats-grid">
                    <div class="stat">
                        <span class="stat-label">Total Plots</span>
                        <span class="stat-value" id="total-plots">--</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Active Plots</span>
                        <span class="stat-value" id="active-plots">--</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Total Size</span>
                        <span class="stat-value" id="total-size">-- GB</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">24h Proofs</span>
                        <span class="stat-value" id="proofs-24h">--</span>
                    </div>
                </div>
            </div>

            <!-- System Resources -->
            <div class="card">
                <h3>💻 System Resources</h3>
                <div class="resource-bars">
                    <div class="resource-bar">
                        <span class="resource-label">CPU</span>
                        <div class="progress-bar">
                            <div class="progress-fill" id="cpu-bar" style="width: 0%"></div>
                        </div>
                        <span class="resource-value" id="cpu-value">--%</span>
                    </div>
                    <div class="resource-bar">
                        <span class="resource-label">Memory</span>
                        <div class="progress-bar">
                            <div class="progress-fill" id="memory-bar" style="width: 0%"></div>
                        </div>
                        <span class="resource-value" id="memory-value">--%</span>
                    </div>
                    <div class="resource-bar" id="gpu-resource" style="display: none;">
                        <span class="resource-label">GPU</span>
                        <div class="progress-bar">
                            <div class="progress-fill" id="gpu-bar" style="width: 0%"></div>
                        </div>
                        <span class="resource-value" id="gpu-value">--%</span>
                    </div>
                </div>
            </div>

            <!-- Optimization Controls -->
            <div class="card">
                <h3>⚡ Optimization Controls</h3>
                <div class="optimization-controls">
                    <button class="opt-btn active" data-mode="speed">🚀 Speed</button>
                    <button class="opt-btn" data-mode="middle">⚖️ Middle</button>
                    <button class="opt-btn" data-mode="cost">💰 Cost</button>
                </div>
                <div class="optimization-info">
                    <p id="current-mode">Current Mode: --</p>
                    <p id="mode-description">Select optimization mode above</p>
                </div>
            </div>

            <!-- Performance Chart -->
            <div class="card chart-card">
                <h3>📈 Performance Trends</h3>
                <canvas id="performanceChart" width="400" height="200"></canvas>
            </div>

            <!-- Alerts -->
            <div class="card">
                <h3>🚨 Alerts</h3>
                <div id="alerts-container">
                    <p class="no-alerts">No active alerts</p>
                </div>
            </div>

            <!-- Recommendations -->
            <div class="card">
                <h3>💡 Recommendations</h3>
                <ul id="recommendations-list">
                    <li>Loading recommendations...</li>
                </ul>
            </div>
        </div>

        <!-- Footer -->
        <footer class="dashboard-footer">
            <p>SquashPlot Dashboard | Bradley Wallace (Koba42 Corp) | Real-time Chia Farming Management</p>
        </footer>
    </div>

    <script src="/static/dashboard.js"></script>
</body>
</html>"""

    def _create_dashboard_css(self) -> str:
        """Create dashboard CSS"""
        return """/* SquashPlot Dashboard Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.dashboard {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.dashboard-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 30px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.dashboard-header h1 {
    color: #2c3e50;
    font-size: 2.5rem;
    margin-bottom: 10px;
}

.subtitle {
    color: #7f8c8d;
    font-size: 1.1rem;
    margin-bottom: 20px;
}

.status-bar {
    display: flex;
    justify-content: center;
    gap: 30px;
    font-size: 0.9rem;
    color: #34495e;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
}

.card h3 {
    color: #2c3e50;
    margin-bottom: 20px;
    font-size: 1.3rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.stat {
    text-align: center;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 10px;
}

.stat-label {
    display: block;
    color: #7f8c8d;
    font-size: 0.9rem;
    margin-bottom: 5px;
}

.stat-value {
    display: block;
    color: #2c3e50;
    font-size: 1.8rem;
    font-weight: bold;
}

.resource-bars {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.resource-bar {
    display: flex;
    align-items: center;
    gap: 15px;
}

.resource-label {
    min-width: 60px;
    font-weight: bold;
    color: #2c3e50;
}

.progress-bar {
    flex: 1;
    height: 8px;
    background: #ecf0f1;
    border-radius: 4px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #27ae60, #2ecc71);
    border-radius: 4px;
    transition: width 0.3s ease;
}

.resource-value {
    min-width: 50px;
    text-align: right;
    font-weight: bold;
    color: #2c3e50;
}

.optimization-controls {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.opt-btn {
    padding: 10px 20px;
    border: 2px solid #3498db;
    background: white;
    color: #3498db;
    border-radius: 25px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: bold;
}

.opt-btn:hover {
    background: #3498db;
    color: white;
}

.opt-btn.active {
    background: #3498db;
    color: white;
}

.optimization-info p {
    margin: 5px 0;
    color: #7f8c8d;
}

.chart-card {
    grid-column: span 2;
}

#alerts-container {
    max-height: 200px;
    overflow-y: auto;
}

.alert {
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
    border-left: 4px solid;
}

.alert.warning {
    background: #fff3cd;
    border-left-color: #ffc107;
    color: #856404;
}

.alert.critical {
    background: #f8d7da;
    border-left-color: #dc3545;
    color: #721c24;
}

.alert.info {
    background: #d1ecf1;
    border-left-color: #17a2b8;
    color: #0c5460;
}

.no-alerts {
    color: #7f8c8d;
    font-style: italic;
}

#recommendations-list {
    padding-left: 20px;
}

#recommendations-list li {
    margin: 8px 0;
    color: #555;
}

.dashboard-footer {
    text-align: center;
    padding: 20px;
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .dashboard-grid {
        grid-template-columns: 1fr;
    }

    .chart-card {
        grid-column: span 1;
    }

    .dashboard-header h1 {
        font-size: 2rem;
    }

    .stats-grid {
        grid-template-columns: 1fr;
    }
}

/* Loading Animation */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.loading {
    animation: pulse 1.5s infinite;
}"""

    def _create_dashboard_js(self) -> str:
        """Create dashboard JavaScript"""
        return """// SquashPlot Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initializeDashboard();

    // Set up optimization controls
    setupOptimizationControls();

    // Start data updates
    setInterval(updateDashboard, 5000); // Update every 5 seconds

    // Initial data load
    updateDashboard();
});

let performanceChart = null;
let performanceData = {
    labels: [],
    cpuData: [],
    memoryData: []
};

function initializeDashboard() {
    console.log('Initializing SquashPlot Dashboard...');

    // Initialize performance chart
    const ctx = document.getElementById('performanceChart').getContext('2d');
    performanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: performanceData.labels,
            datasets: [{
                label: 'CPU Usage (%)',
                data: performanceData.cpuData,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.4
            }, {
                label: 'Memory Usage (%)',
                data: performanceData.memoryData,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
}

function setupOptimizationControls() {
    const buttons = document.querySelectorAll('.opt-btn');

    buttons.forEach(button => {
        button.addEventListener('click', async function() {
            const mode = this.getAttribute('data-mode');

            // Update UI
            buttons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Send optimization request
            try {
                const response = await fetch('/api/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ mode: mode })
                });

                const result = await response.json();

                if (result.status === 'success') {
                    updateModeDisplay(mode);
                    showAlert('Optimization mode updated successfully', 'success');
                } else {
                    showAlert('Failed to update optimization mode', 'error');
                }
            } catch (error) {
                console.error('Optimization request failed:', error);
                showAlert('Network error - please try again', 'error');
            }
        });
    });
}

async function updateDashboard() {
    try {
        // Update timestamp
        document.getElementById('last-update').textContent =
            `Last Update: ${new Date().toLocaleTimeString()}`;

        // Fetch dashboard data
        const response = await fetch('/api/dashboard');
        const data = await response.json();

        // Update farming statistics
        updateFarmingStats(data);

        // Update system resources
        updateSystemResources(data);

        // Update optimization mode
        updateModeDisplay(data.optimization_mode);

        // Update alerts
        updateAlerts(data.alerts);

        // Update recommendations
        updateRecommendations(data.recommendations);

        // Update performance chart
        updatePerformanceChart(data.performance_trends);

    } catch (error) {
        console.error('Dashboard update failed:', error);
    }
}

function updateFarmingStats(data) {
    const stats = data.farming_stats;

    document.getElementById('total-plots').textContent = stats.total_plots || 0;
    document.getElementById('active-plots').textContent = stats.active_plots || 0;
    document.getElementById('total-size').textContent = `${stats.total_size_gb || 0} GB`;
    document.getElementById('proofs-24h').textContent = stats.proofs_found_24h || 0;
}

function updateSystemResources(data) {
    const resources = data.system_resources;

    // CPU
    const cpuValue = resources.cpu_usage || 0;
    document.getElementById('cpu-bar').style.width = `${cpuValue}%`;
    document.getElementById('cpu-value').textContent = `${cpuValue.toFixed(1)}%`;

    // Memory
    const memoryValue = resources.memory_usage || 0;
    document.getElementById('memory-bar').style.width = `${memoryValue}%`;
    document.getElementById('memory-value').textContent = `${memoryValue.toFixed(1)}%`;

    // GPU (if available)
    if (resources.gpu_usage !== undefined) {
        document.getElementById('gpu-resource').style.display = 'flex';
        const gpuValue = resources.gpu_usage || 0;
        document.getElementById('gpu-bar').style.width = `${gpuValue}%`;
        document.getElementById('gpu-value').textContent = `${gpuValue.toFixed(1)}%`;
    }
}

function updateModeDisplay(mode) {
    document.getElementById('optimization-mode').textContent = `Mode: ${mode.toUpperCase()}`;
    document.getElementById('current-mode').textContent = `Current Mode: ${mode.toUpperCase()}`;

    const descriptions = {
        'speed': 'Maximum plotting speed with GPU acceleration',
        'cost': 'Minimum resource usage for cost optimization',
        'middle': 'Balanced performance and cost optimization'
    };

    document.getElementById('mode-description').textContent =
        descriptions[mode] || 'Unknown optimization mode';
}

function updateAlerts(alerts) {
    const container = document.getElementById('alerts-container');

    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<p class="no-alerts">No active alerts</p>';
        return;
    }

    container.innerHTML = alerts.map(alert => `
        <div class="alert ${alert.level}">
            <strong>${alert.level.toUpperCase()}</strong>: ${alert.message}
            <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
        </div>
    `).join('');
}

function updateRecommendations(recommendations) {
    const list = document.getElementById('recommendations-list');

    if (!recommendations || recommendations.length === 0) {
        list.innerHTML = '<li>No recommendations at this time</li>';
        return;
    }

    list.innerHTML = recommendations.map(rec => `<li>${rec}</li>`).join('');
}

function updatePerformanceChart(trends) {
    if (!trends || trends.trend === 'insufficient_data') {
        return;
    }

    // Add new data point (simulate for demo)
    const now = new Date().toLocaleTimeString();
    performanceData.labels.push(now);

    // Get current resource values
    const cpuBar = document.getElementById('cpu-bar');
    const memoryBar = document.getElementById('memory-bar');

    const cpuValue = parseFloat(cpuBar.style.width) || 0;
    const memoryValue = parseFloat(memoryBar.style.width) || 0;

    performanceData.cpuData.push(cpuValue);
    performanceData.memoryData.push(memoryValue);

    // Keep only last 50 data points
    if (performanceData.labels.length > 50) {
        performanceData.labels.shift();
        performanceData.cpuData.shift();
        performanceData.memoryData.shift();
    }

    // Update chart
    performanceChart.update();
}

function showAlert(message, type) {
    // Create temporary alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert ${type}`;
    alertDiv.innerHTML = `<strong>${type.toUpperCase()}</strong>: ${message}`;

    const alertsContainer = document.getElementById('alerts-container');
    alertsContainer.insertBefore(alertDiv, alertsContainer.firstChild);

    // Remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 5000);
}

// Export functionality
document.addEventListener('keydown', function(event) {
    if (event.ctrlKey && event.key === 'e') {
        event.preventDefault();
        exportDashboard();
    }
});

async function exportDashboard() {
    try {
        const response = await fetch('/api/export');
        const blob = await response.blob();

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `squashplot_dashboard_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showAlert('Dashboard data exported successfully', 'success');
    } catch (error) {
        console.error('Export failed:', error);
        showAlert('Export failed - please try again', 'error');
    }
}

// Error handling
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showAlert('An error occurred - please refresh the page', 'error');
});

console.log('SquashPlot Dashboard loaded successfully');"""

def main():
    """Main dashboard application"""
    import argparse

    parser = argparse.ArgumentParser(description='SquashPlot Dashboard - Chia Farming Monitor')
    parser.add_argument('--chia-root', default='~/chia-blockchain',
                       help='Path to Chia blockchain installation')
    parser.add_argument('--plot-dirs', nargs='+',
                       help='Directories containing plot files')
    parser.add_argument('--mode', choices=['speed', 'cost', 'middle'],
                       default='middle', help='Optimization mode')
    parser.add_argument('--host', default='0.0.0.0', help='Dashboard host')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Initialize farming manager
    manager = ChiaFarmingManager(
        chia_root=args.chia_root,
        plot_directories=args.plot_dirs,
        optimization_mode=OptimizationMode(args.mode)
    )

    # Start monitoring
    manager.start_monitoring()

    try:
        print("🌐 Starting SquashPlot Dashboard...")
        print(f"Host: {args.host}:{args.port}")
        print(f"Optimization Mode: {args.mode.upper()}")
        print("Press Ctrl+C to stop")
        print("=" * 50)

        # Initialize and run dashboard
        dashboard = SquashPlotDashboard(manager)
        dashboard.run(host=args.host, port=args.port, debug=args.debug)

    except KeyboardInterrupt:
        print("\n🛑 Stopping SquashPlot Dashboard...")
    finally:
        manager.stop_monitoring()

if __name__ == '__main__':
    main()
