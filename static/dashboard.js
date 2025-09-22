// SquashPlot Dashboard JavaScript
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

console.log('SquashPlot Dashboard loaded successfully');