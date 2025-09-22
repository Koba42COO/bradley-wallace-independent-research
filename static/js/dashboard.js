// SquashPlot Enhanced Dashboard JavaScript

class SquashPlotDashboard {
    constructor() {
        this.statusUpdateInterval = null;
        this.priceUpdateInterval = null;
        this.insightsUpdateInterval = null;
        this.isConnected = false;
        this.currentPlottingStatus = { active: false };
        this.compressionLevels = [];
        this.walletConnected = false;
        this.wallets = [];
        this.pendingRewards = [];
        this.offers = [];
        this.darkMode = localStorage.getItem('darkMode') === 'true';
        this.isAuthenticated = false;
        this.currentUser = null;
        this.chatMessages = [];
        this.currentPools = [];
        this.liveData = {
            xchPrice: null,
            networkStats: null,
            insights: []
        };
        
        this.init();
    }
    
    init() {
        this.initializeTheme();
        this.setupEventListeners();
        this.setupEnhancedEventListeners();
        this.startStatusUpdates();
        this.startLiveDataUpdates();
        this.loadCompressionLevels();
        this.setupFormValidation();
        this.setupWalletEventListeners();
        this.updateWalletStatus();
        this.loadSystemMetrics();
        this.loadAIInsights();
        this.setupSocialEventListeners();
        this.checkAuthenticationStatus();
        this.initializeChat();
        this.loadUserPools();
    }
    
    setupEventListeners() {
        // Form submission - check if element exists
        const plotForm = document.getElementById('plot-config-form');
        if (plotForm) {
            plotForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.startPlotting();
            });
        }
        
        // Validation button - check if element exists
        const validateBtn = document.getElementById('validate-btn');
        if (validateBtn) {
            validateBtn.addEventListener('click', () => {
                this.validateConfiguration();
            });
        }
        
        // Stop button - check if element exists
        const stopBtn = document.getElementById('stop-btn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                this.stopPlotting();
            });
        }
        
        // Compression level change - check if element exists
        const compressionSelect = document.getElementById('compression');
        if (compressionSelect) {
            compressionSelect.addEventListener('change', (e) => {
                this.updateCompressionInfo(parseInt(e.target.value));
            });
        }
        
        // Auto-save form data
        const formInputs = document.querySelectorAll('#plot-config-form input, #plot-config-form select');
        formInputs.forEach(input => {
            input.addEventListener('change', () => {
                this.saveFormData();
            });
        });
        
        // Load saved form data
        this.loadFormData();
        
        // Setup plotter mode change handlers
        this.setupPlotterModeHandlers();
        
        // Setup dynamic metrics updates
        this.setupDynamicMetrics();
    }
    
    initializeTheme() {
        if (this.darkMode) {
            document.body.classList.add('dark-mode');
        }
    }
    
    setupEnhancedEventListeners() {
        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
        
        // ROI Calculator
        const roiForm = document.getElementById('roi-form');
        if (roiForm) {
            roiForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.calculateROI();
            });
        }
        
        // Hardware Optimizer
        const hardwareForm = document.getElementById('hardware-form');
        if (hardwareForm) {
            hardwareForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.optimizeHardware();
            });
        }
        
        // Export functionality
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.showExportModal();
            });
        }
        
        // AI Insights refresh
        const refreshInsights = document.getElementById('refresh-insights');
        if (refreshInsights) {
            refreshInsights.addEventListener('click', () => {
                this.loadAIInsights();
            });
        }
    }
    
    toggleTheme() {
        this.darkMode = !this.darkMode;
        localStorage.setItem('darkMode', this.darkMode);
        document.body.classList.toggle('dark-mode', this.darkMode);
        
        const themeIcon = document.querySelector('#theme-toggle i');
        if (themeIcon) {
            themeIcon.className = this.darkMode ? 'fas fa-sun' : 'fas fa-moon';
        }
    }
    
    startLiveDataUpdates() {
        // Update live price data every 30 seconds
        this.updateLivePrice();
        this.priceUpdateInterval = setInterval(() => {
            this.updateLivePrice();
        }, 30000);
        
        // Update AI insights every 5 minutes
        this.insightsUpdateInterval = setInterval(() => {
            this.loadAIInsights();
        }, 300000);
    }
    
    async updateLivePrice() {
        try {
            const [priceResponse, networkResponse] = await Promise.all([
                fetch('/api/live-price'),
                fetch('/api/network-stats')
            ]);
            
            if (priceResponse.ok && networkResponse.ok) {
                const priceData = await priceResponse.json();
                const networkData = await networkResponse.json();
                
                this.liveData.xchPrice = priceData;
                this.liveData.networkStats = networkData;
                
                this.updatePriceTicker(priceData, networkData);
            }
        } catch (error) {
            console.error('Failed to update live price:', error);
        }
    }
    
    updatePriceTicker(priceData, networkData) {
        const priceElement = document.getElementById('xch-price');
        const changeElement = document.getElementById('price-change');
        const netspaceElement = document.getElementById('netspace');
        
        if (priceElement) {
            priceElement.textContent = `$${priceData.price_usd.toFixed(2)}`;
        }
        
        if (changeElement) {
            const change = priceData.change_24h;
            changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`;
            changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
        }
        
        if (netspaceElement) {
            netspaceElement.textContent = `${networkData.netspace_eib.toFixed(1)} EiB`;
        }
    }
    
    async calculateROI() {
        const plotCount = parseInt(document.getElementById('roi-plot-count').value);
        const hardwareCost = parseFloat(document.getElementById('roi-hardware-cost').value);
        const zipcode = document.getElementById('roi-zipcode').value;
        
        try {
            const response = await fetch('/api/roi-calculator', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    plot_count: plotCount,
                    plot_size_tb: 0.1014, // Standard plot size
                    hardware_cost: hardwareCost,
                    zipcode: zipcode
                })
            });
            
            if (response.ok) {
                const roiData = await response.json();
                this.displayROIResults(roiData);
            } else {
                throw new Error('Failed to calculate ROI');
            }
        } catch (error) {
            console.error('ROI calculation error:', error);
            this.showNotification('Failed to calculate ROI. Please try again.', 'error');
        }
    }
    
    displayROIResults(roiData) {
        const resultsContainer = document.getElementById('roi-results');
        const breakevenElement = document.getElementById('roi-breakeven');
        const annualElement = document.getElementById('roi-annual');
        const monthlyElement = document.getElementById('roi-monthly');
        
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
        }
        
        if (breakevenElement) {
            const breakeven = roiData.roi_metrics.break_even_days;
            breakevenElement.textContent = breakeven ? `${breakeven} days` : 'Never';
        }
        
        if (annualElement) {
            annualElement.textContent = `${roiData.roi_metrics.annual_roi_percent}%`;
        }
        
        if (monthlyElement) {
            monthlyElement.textContent = `$${roiData.profit.monthly_usd}`;
        }
    }
    
    async optimizeHardware() {
        const budget = parseFloat(document.getElementById('hw-budget').value);
        const targetPlots = parseInt(document.getElementById('hw-target-plots').value);
        const useCase = document.getElementById('hw-use-case').value;
        
        try {
            const response = await fetch('/api/hardware-optimizer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    budget: budget,
                    plot_count_target: targetPlots,
                    use_case: useCase
                })
            });
            
            if (response.ok) {
                const hardwareData = await response.json();
                this.displayHardwareResults(hardwareData);
            } else {
                throw new Error('Failed to optimize hardware');
            }
        } catch (error) {
            console.error('Hardware optimization error:', error);
            this.showNotification('Failed to optimize hardware. Please try again.', 'error');
        }
    }
    
    displayHardwareResults(hardwareData) {
        const resultsContainer = document.getElementById('hardware-results');
        const cpuElement = document.getElementById('hw-cpu');
        const costElement = document.getElementById('hw-cost');
        const speedElement = document.getElementById('hw-speed');
        
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
        }
        
        if (cpuElement) {
            cpuElement.textContent = hardwareData.hardware_breakdown.cpu.name;
        }
        
        if (costElement) {
            costElement.textContent = `$${hardwareData.cost_analysis.total_hardware_cost}`;
        }
        
        if (speedElement) {
            speedElement.textContent = `${hardwareData.performance_estimates.plots_per_day.toFixed(1)}`;
        }
    }
    
    async loadAIInsights() {
        const container = document.getElementById('insights-container');
        if (!container) return;
        
        container.innerHTML = '<div class="loading-insights"><i class="fas fa-spinner fa-spin"></i><span>Analyzing market conditions...</span></div>';
        
        try {
            const response = await fetch('/api/ai-insights');
            if (response.ok) {
                const insightsData = await response.json();
                this.displayAIInsights(insightsData);
            } else {
                throw new Error('Failed to load insights');
            }
        } catch (error) {
            console.error('AI insights error:', error);
            container.innerHTML = '<div class="error-insights"><i class="fas fa-exclamation-triangle"></i><span>Unable to load insights. Please try again.</span></div>';
        }
    }
    
    displayAIInsights(insightsData) {
        const container = document.getElementById('insights-container');
        if (!container) return;
        
        let html = '<div class="insights-list">';
        
        insightsData.insights.forEach(insight => {
            html += `
                <div class="insight-item ${insight.type} priority-${insight.priority}">
                    <div class="insight-header">
                        <div class="insight-icon">
                            <i class="fas ${this.getInsightIcon(insight.type)}"></i>
                        </div>
                        <div class="insight-title">${insight.title}</div>
                        <div class="insight-priority">${insight.priority}</div>
                    </div>
                    <div class="insight-description">${insight.description}</div>
                    <div class="insight-action">${insight.action}</div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }
    
    getInsightIcon(type) {
        const icons = {
            'bullish': 'fa-arrow-trend-up',
            'bearish': 'fa-arrow-trend-down',
            'competitive': 'fa-shield-alt',
            'timing': 'fa-clock',
            'efficiency': 'fa-leaf'
        };
        return icons[type] || 'fa-lightbulb';
    }
    
    showExportModal() {
        // Create export modal
        const modal = document.createElement('div');
        modal.className = 'export-modal-overlay';
        modal.innerHTML = `
            <div class="export-modal">
                <div class="export-header">
                    <h3>Export Cost Report</h3>
                    <button class="close-btn" onclick="this.closest('.export-modal-overlay').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="export-content">
                    <div class="export-options">
                        <label>
                            <input type="radio" name="export-format" value="csv" checked>
                            <span>CSV Format</span>
                        </label>
                        <label>
                            <input type="radio" name="export-format" value="pdf">
                            <span>PDF Report</span>
                        </label>
                    </div>
                    <div class="export-settings">
                        <div class="form-group">
                            <label>Plot Count for Report</label>
                            <input type="number" id="export-plot-count" value="${document.getElementById('count').value}" min="1">
                        </div>
                        <div class="form-group">
                            <label>Location (Zipcode)</label>
                            <input type="text" id="export-zipcode" value="${document.getElementById('zipcode').value}" maxlength="5">
                        </div>
                    </div>
                    <div class="export-actions">
                        <button class="btn-cancel" onclick="this.closest('.export-modal-overlay').remove()">Cancel</button>
                        <button class="btn-export" onclick="dashboard.executeExport()">Export Report</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    async executeExport() {
        const format = document.querySelector('input[name="export-format"]:checked').value;
        const plotCount = document.getElementById('export-plot-count').value;
        const zipcode = document.getElementById('export-zipcode').value;
        
        try {
            const response = await fetch('/api/export/cost-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    format: format,
                    plot_count: parseInt(plotCount),
                    zipcode: zipcode
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `squashplot_report_${new Date().toISOString().split('T')[0]}.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                document.querySelector('.export-modal-overlay').remove();
                this.showNotification('Report exported successfully!', 'success');
            } else {
                throw new Error('Export failed');
            }
        } catch (error) {
            console.error('Export error:', error);
            this.showNotification('Failed to export report. Please try again.', 'error');
        }
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
    
    setupWalletEventListeners() {
        // Initialize wallet type buttons
        this.setupWalletTypeButtons();
        
        // Only set up event listeners if elements exist
        const connectBtn = document.getElementById('connect-wallet-btn');
        if (connectBtn) {
            connectBtn.addEventListener('click', () => {
                this.connectWallet();
            });
        }
        
        const autoClaimToggle = document.getElementById('auto-claim-toggle');
        if (autoClaimToggle) {
            autoClaimToggle.addEventListener('change', (e) => {
                this.toggleAutoClaim(e.target.checked);
            });
        }
        
        const claimBtn = document.getElementById('claim-rewards-btn');
        if (claimBtn) {
            claimBtn.addEventListener('click', () => {
                this.claimRewards();
            });
        }
        
        const createOfferBtn = document.getElementById('create-offer-btn');
        if (createOfferBtn) {
            createOfferBtn.addEventListener('click', () => {
                this.showCreateOfferModal();
            });
        }
        
        const uploadOfferBtn = document.getElementById('upload-offer-btn');
        if (uploadOfferBtn) {
            uploadOfferBtn.addEventListener('click', () => {
                document.getElementById('offer-file-input').click();
            });
        }
        
        const offerFileInput = document.getElementById('offer-file-input');
        if (offerFileInput) {
            offerFileInput.addEventListener('change', (e) => {
                this.uploadOfferFile(e.target.files[0]);
            });
        }
        
        const sendForm = document.getElementById('send-xch-form');
        if (sendForm) {
            sendForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendXCH();
            });
        }
    }
    
    startStatusUpdates() {
        this.updateStatus();
        this.statusUpdateInterval = setInterval(() => {
            this.updateStatus();
        }, 2000); // Update every 2 seconds
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (response.ok) {
                this.isConnected = true;
                this.updateConnectionStatus(true);
                this.updateToolStatus(data.tools);
                this.updateResourceStatus(data.resources);
                this.updatePlottingStatus(data.plotting);
                this.currentPlottingStatus = data.plotting;
            } else {
                throw new Error(data.error || 'Failed to fetch status');
            }
        } catch (error) {
            this.isConnected = false;
            this.updateConnectionStatus(false);
            console.error('Status update failed:', error.message || error.toString());
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            if (connected) {
                statusElement.className = 'status-indicator online';
                statusElement.innerHTML = '<span class="status-dot online"></span><span>Connected</span>';
            } else {
                statusElement.className = 'status-indicator offline';
                statusElement.innerHTML = '<span class="status-dot offline"></span><span>Disconnected</span>';
            }
        } else {
            console.log('Connection status element not found');
        }
    }
    
    updateToolStatus(tools) {
        this.updateToolIndicator('madmax-status', tools.madmax_available);
        this.updateToolIndicator('bladebit-status', tools.bladebit_available);
        this.updateToolIndicator('chia-status', tools.chia_available);
    }
    
    updateToolIndicator(elementId, available) {
        const element = document.getElementById(elementId);
        if (element) {
            element.className = `tool-indicator ${available ? 'available' : 'unavailable'}`;
            element.textContent = available ? '●' : '○';
        }
        // Silently ignore missing tool indicators
    }
    
    updateResourceStatus(resources) {
        const cpuElement = document.getElementById('cpu-cores');
        if (cpuElement) cpuElement.textContent = resources.cpu_count || '-';
        
        const memoryElement = document.getElementById('memory-available');
        if (memoryElement) memoryElement.textContent = `${resources.available_memory_gb || 0}GB`;
        
        let storageType = 'HDD';
        if (resources.nvme_available) {
            storageType = 'NVMe SSD';
        } else if (resources.ssd_available) {
            storageType = 'SSD';
        }
        
        const storageTypeElement = document.getElementById('storage-type');
        if (storageTypeElement) storageTypeElement.textContent = storageType;
        
        const totalStorageElement = document.getElementById('total-storage');
        if (totalStorageElement) totalStorageElement.textContent = `${resources.total_storage_gb || 0}GB`;
    }
    
    updatePlottingStatus(plotting) {
        const statusIndicator = document.getElementById('plot-status-indicator');
        const progressInfo = document.getElementById('progress-info');
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        
        // Check if required elements exist - only relevant on dashboard page
        if (!statusIndicator) {
            return; // Silently skip if not on a page that has plotting status
        }
        
        if (plotting.active) {
            statusIndicator.className = 'status-indicator active';
            statusIndicator.innerHTML = '<span class="status-dot"></span><span>Plotting</span>';
            
            if (progressInfo) {
                progressInfo.style.display = 'block';
                this.updateProgress(plotting.progress || 0, plotting.stage || 'processing');
            }
            
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'flex';
        } else {
            let statusClass = 'idle';
            let statusText = 'Ready to Plot';
            
            if (plotting.stage === 'completed') {
                statusClass = 'completed';
                statusText = 'Completed';
            } else if (plotting.stage === 'failed') {
                statusClass = 'failed';
                statusText = 'Failed';
            }
            
            statusIndicator.className = `status-indicator ${statusClass}`;
            statusIndicator.innerHTML = `<span class="status-dot"></span><span>${statusText}</span>`;
            
            if (progressInfo) progressInfo.style.display = 'none';
            if (startBtn) startBtn.style.display = 'flex';
            if (stopBtn) stopBtn.style.display = 'none';
        }
        
        if (plotting.error_message && this.showAlert) {
            this.showAlert(plotting.error_message, 'error');
        }
    }
    
    updateProgress(progress, stage) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const progressStage = document.getElementById('progress-stage');
        
        if (progressFill) progressFill.style.width = `${progress}%`;
        if (progressText) progressText.textContent = `${Math.round(progress)}%`;
        if (progressStage) progressStage.textContent = stage;
    }
    
    async loadCompressionLevels() {
        try {
            const response = await fetch('/api/compression-levels');
            const levels = await response.json();
            
            if (response.ok) {
                this.compressionLevels = levels;
                this.renderCompressionGrid(levels);
                this.updateCompressionInfo(3); // Default to level 3
            }
        } catch (error) {
            console.error('Failed to load compression levels:', error.message || error.toString());
        }
    }
    
    renderCompressionGrid(levels) {
        const grid = document.getElementById('compression-grid');
        if (!grid) {
            // Silently ignore if compression grid element doesn't exist
            return;
        }
        
        grid.innerHTML = '';
        
        levels.forEach(level => {
            const levelElement = document.createElement('div');
            levelElement.className = 'compression-level';
            levelElement.innerHTML = `
                <h5>Level ${level.level}</h5>
                <div class="description">${level.description}</div>
                <div class="compression-stats">
                    <span>${level.estimated_size_gb}GB</span>
                    <span>${level.savings_percent}% savings</span>
                </div>
            `;
            grid.appendChild(levelElement);
        });
    }
    
    updateCompressionInfo(level) {
        const infoElement = document.getElementById('compression-info');
        if (!infoElement) {
            // Silently ignore if compression info element doesn't exist
            return;
        }
        
        const levelData = this.compressionLevels.find(l => l.level === level);
        
        if (levelData) {
            infoElement.textContent = `${levelData.savings_percent}% space savings (${levelData.estimated_size_gb}GB)`;
        }
    }
    
    async validateConfiguration() {
        const formData = this.getFormData();
        const validateBtn = document.getElementById('validate-btn');
        
        validateBtn.disabled = true;
        validateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating...';
        
        try {
            const response = await fetch('/api/validate-config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                if (result.valid) {
                    this.showAlert('Configuration is valid!', 'success');
                } else {
                    const errors = result.errors.concat(result.warnings);
                    this.showAlert(`Configuration issues: ${errors.join(', ')}`, 'warning');
                }
            } else {
                throw new Error(result.error || 'Validation failed');
            }
        } catch (error) {
            this.showAlert(`Validation error: ${error.message}`, 'error');
        } finally {
            validateBtn.disabled = false;
            validateBtn.innerHTML = '<i class="fas fa-check"></i> Validate Configuration';
        }
    }
    
    async startPlotting() {
        if (this.currentPlottingStatus.active) {
            this.showAlert('Plotting is already in progress', 'warning');
            return;
        }
        
        const formData = this.getFormData();
        const startBtn = document.getElementById('start-btn');
        
        startBtn.disabled = true;
        startBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
        
        try {
            const response = await fetch('/api/start-plotting', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showAlert('Plotting started successfully!', 'success');
            } else {
                throw new Error(result.error || 'Failed to start plotting');
            }
        } catch (error) {
            this.showAlert(`Failed to start plotting: ${error.message}`, 'error');
        } finally {
            startBtn.disabled = false;
            startBtn.innerHTML = '<i class="fas fa-play"></i> Start Plotting';
        }
    }
    
    async stopPlotting() {
        try {
            const response = await fetch('/api/stop-plotting', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showAlert('Plotting stopped', 'warning');
            } else {
                throw new Error(result.error || 'Failed to stop plotting');
            }
        } catch (error) {
            this.showAlert(`Failed to stop plotting: ${error.message}`, 'error');
        }
    }
    
    getFormData() {
        const form = document.getElementById('plot-config-form');
        if (!form) return {};
        
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        return data;
    }
    
    saveFormData() {
        const formData = this.getFormData();
        localStorage.setItem('squashplot-form-data', JSON.stringify(formData));
    }
    
    loadFormData() {
        const savedData = localStorage.getItem('squashplot-form-data');
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                Object.keys(data).forEach(key => {
                    const element = document.querySelector(`[name="${key}"]`);
                    if (element) {
                        element.value = data[key];
                    }
                });
            } catch (error) {
                console.error('Failed to load saved form data:', error.message || error.toString());
            }
        }
    }
    
    setupFormValidation() {
        const form = document.getElementById('plot-config-form');
        if (form) {
            const inputs = form.querySelectorAll('input[required]');
            
            inputs.forEach(input => {
                input.addEventListener('blur', () => {
                    this.validateField(input);
                });
            });
        }
    }
    
    validateField(field) {
        const isValid = field.checkValidity();
        const formGroup = field.closest('.form-group');
        
        // Remove existing validation classes
        formGroup.classList.remove('field-valid', 'field-invalid');
        
        if (field.value) {
            if (isValid) {
                formGroup.classList.add('field-valid');
            } else {
                formGroup.classList.add('field-invalid');
            }
        }
    }
    
    showAlert(message, type = 'info') {
        const container = document.getElementById('alert-container');
        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.textContent = message;
        
        container.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.parentNode.removeChild(alert);
            }
        }, 5000);
    }
    
    // Wallet Management Methods
    
    async updateWalletStatus() {
        try {
            const response = await fetch('/api/wallet/status');
            const status = await response.json();
            
            if (response.ok) {
                this.walletConnected = status.connected;
                this.updateWalletUI(status);
                
                if (this.walletConnected) {
                    await this.loadWalletData();
                }
            }
        } catch (error) {
            console.error('Failed to update wallet status:', error.message || error.toString());
        }
    }
    
    updateWalletUI(status) {
        const indicator = document.getElementById('wallet-indicator');
        const connectBtn = document.getElementById('connect-wallet-btn');
        const walletContent = document.getElementById('wallet-content');
        const autoClaimToggle = document.getElementById('auto-claim-toggle');
        
        if (status.connected) {
            if (indicator) {
                indicator.className = 'wallet-indicator connected';
                indicator.innerHTML = '<i class="fas fa-circle"></i><span>Wallet Connected</span>';
            }
            if (connectBtn) {
                connectBtn.innerHTML = '<i class="fas fa-unlink"></i>Disconnect';
                connectBtn.onclick = () => this.disconnectWallet();
            }
            if (walletContent) {
                walletContent.style.display = 'block';
            }
            if (autoClaimToggle) {
                autoClaimToggle.checked = status.auto_claim_enabled;
            }
        } else {
            if (indicator) {
                indicator.className = 'wallet-indicator disconnected';
                indicator.innerHTML = '<i class="fas fa-circle"></i><span>Wallet Disconnected</span>';
            }
            if (connectBtn) {
                connectBtn.innerHTML = '<i class="fas fa-plug"></i>Connect Wallet';
                connectBtn.onclick = () => this.connectWallet();
            }
            if (walletContent) {
                walletContent.style.display = 'none';
            }
        }
    }
    
    async connectWallet() {
        const connectBtn = document.getElementById('connect-wallet-btn');
        if (!connectBtn) return;
        
        const originalText = connectBtn.innerHTML;
        
        connectBtn.disabled = true;
        connectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>Connecting...';
        
        try {
            const response = await fetch('/api/wallet/connect', {
                method: 'POST'
            });
            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Wallet connected successfully!', 'success');
                await this.updateWalletStatus();
            } else {
                this.showAlert(`Failed to connect: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showAlert(`Connection error: ${error.message}`, 'error');
        } finally {
            connectBtn.disabled = false;
            connectBtn.innerHTML = originalText;
        }
    }
    
    async disconnectWallet() {
        try {
            const response = await fetch('/api/wallet/disconnect', {
                method: 'POST'
            });
            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Wallet disconnected', 'warning');
                await this.updateWalletStatus();
            }
        } catch (error) {
            this.showAlert(`Disconnect error: ${error.message}`, 'error');
        }
    }
    
    async loadWalletData() {
        await Promise.all([
            this.loadWallets(),
            this.loadPendingRewards(),
            this.loadOffers()
        ]);
    }
    
    async loadWallets() {
        try {
            const response = await fetch('/api/wallet/wallets');
            const data = await response.json();
            
            if (response.ok) {
                this.wallets = data.wallets;
                this.renderWalletBalances();
                this.updateSendWalletOptions();
            }
        } catch (error) {
            console.error('Failed to load wallets:', error.message || error.toString());
        }
    }
    
    renderWalletBalances() {
        const balanceGrid = document.getElementById('balance-grid');
        if (!balanceGrid) return;
        
        balanceGrid.innerHTML = '';
        
        this.wallets.forEach(wallet => {
            const balanceItem = document.createElement('div');
            balanceItem.className = 'balance-item';
            balanceItem.innerHTML = `
                <div>
                    <div class="balance-label">${wallet.name}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">${wallet.wallet_type}</div>
                </div>
                <div class="balance-value">${wallet.balance_xch.toFixed(6)} XCH</div>
            `;
            balanceGrid.appendChild(balanceItem);
        });
    }
    
    updateSendWalletOptions() {
        const select = document.getElementById('send-wallet-id');
        if (!select) return;
        
        select.innerHTML = '';
        
        this.wallets.forEach(wallet => {
            const option = document.createElement('option');
            option.value = wallet.wallet_id;
            option.textContent = `${wallet.name} (${wallet.balance_xch.toFixed(6)} XCH)`;
            select.appendChild(option);
        });
    }
    
    async loadPendingRewards() {
        try {
            const response = await fetch('/api/wallet/rewards');
            const data = await response.json();
            
            if (response.ok) {
                this.pendingRewards = data.rewards;
                this.renderPendingRewards();
            }
        } catch (error) {
            console.error('Failed to load pending rewards:', error.message || error.toString());
        }
    }
    
    renderPendingRewards() {
        const noRewards = document.getElementById('no-rewards');
        const rewardsList = document.getElementById('rewards-list');
        const claimBtn = document.getElementById('claim-rewards-btn');
        
        if (this.pendingRewards.length === 0) {
            if (noRewards) noRewards.style.display = 'flex';
            if (rewardsList) rewardsList.style.display = 'none';
            if (claimBtn) claimBtn.style.display = 'none';
        } else {
            if (noRewards) noRewards.style.display = 'none';
            if (rewardsList) {
                rewardsList.style.display = 'block';
                rewardsList.innerHTML = '';
                
                this.pendingRewards.forEach(reward => {
                    const rewardItem = document.createElement('div');
                    rewardItem.className = 'reward-item';
                    rewardItem.innerHTML = `
                        <div class="reward-type">${reward.claim_type.replace('_', ' ')}</div>
                        <div class="reward-amount">${reward.amount_xch.toFixed(6)} XCH</div>
                    `;
                    rewardsList.appendChild(rewardItem);
                });
            }
            if (claimBtn) claimBtn.style.display = 'block';
        }
    }
    
    async claimRewards() {
        const claimBtn = document.getElementById('claim-rewards-btn');
        if (!claimBtn) return;
        
        const originalText = claimBtn.innerHTML;
        
        claimBtn.disabled = true;
        claimBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>Claiming...';
        
        try {
            const response = await fetch('/api/wallet/claim-rewards', {
                method: 'POST'
            });
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(`Claimed ${result.total_claimed.toFixed(6)} XCH in rewards!`, 'success');
                await this.loadPendingRewards();
                await this.loadWallets();
            } else {
                this.showAlert(`Failed to claim rewards: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showAlert(`Claim error: ${error.message}`, 'error');
        } finally {
            claimBtn.disabled = false;
            claimBtn.innerHTML = originalText;
        }
    }
    
    async toggleAutoClaim(enabled) {
        try {
            const response = await fetch('/api/wallet/auto-claim', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ enabled })
            });
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(result.message, 'success');
            }
        } catch (error) {
            this.showAlert(`Auto-claim error: ${error.message}`, 'error');
        }
    }
    
    async loadOffers() {
        try {
            const response = await fetch('/api/wallet/offers');
            const data = await response.json();
            
            if (response.ok) {
                this.offers = data.offers;
                this.renderOffers();
            }
        } catch (error) {
            console.error('Failed to load offers:', error.message || error.toString());
        }
    }
    
    renderOffers() {
        const offersList = document.getElementById('offers-list');
        offersList.innerHTML = '';
        
        if (this.offers.length === 0) {
            offersList.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 1rem;">No offers found</div>';
            return;
        }
        
        this.offers.forEach(offer => {
            const offerItem = document.createElement('div');
            offerItem.className = 'offer-item';
            offerItem.innerHTML = `
                <div class="offer-summary">${offer.summary}</div>
                <div class="offer-details">
                    <span>${new Date(offer.created_at).toLocaleDateString()}</span>
                    <span class="offer-status ${offer.status}">${offer.status}</span>
                </div>
            `;
            offersList.appendChild(offerItem);
        });
    }
    
    async uploadOfferFile(file) {
        if (!file) return;
        
        const formData = new FormData();
        formData.append('offer_file', file);
        
        try {
            const response = await fetch('/api/wallet/upload-offer', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            
            if (result.success) {
                this.showAlert('Offer file uploaded successfully!', 'success');
                await this.loadOffers();
            } else {
                this.showAlert(`Upload failed: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showAlert(`Upload error: ${error.message}`, 'error');
        }
    }
    
    async sendXCH() {
        const form = document.getElementById('send-xch-form');
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);
        
        try {
            const response = await fetch('/api/wallet/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(`Transaction sent! ID: ${result.transaction_id}`, 'success');
                form.reset();
                await this.loadWallets();
            } else {
                this.showAlert(`Send failed: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showAlert(`Send error: ${error.message}`, 'error');
        }
    }
    
    showCreateOfferModal() {
        // Simple prompt for now - could be enhanced with a proper modal
        const offeredAmount = prompt('Amount to offer (XCH):');
        const requestedAsset = prompt('Requested asset:', 'XCH');
        const requestedAmount = prompt('Requested amount:');
        
        if (offeredAmount && requestedAmount) {
            this.createOffer(parseFloat(offeredAmount), requestedAsset, parseFloat(requestedAmount));
        }
    }
    
    async createOffer(offeredAmount, requestedAsset, requestedAmount) {
        try {
            const response = await fetch('/api/wallet/create-offer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    wallet_id: 1,
                    offered_amount: offeredAmount,
                    requested_asset: requestedAsset,
                    requested_amount: requestedAmount
                })
            });
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(`Offer created: ${result.summary}`, 'success');
                await this.loadOffers();
            } else {
                this.showAlert(`Offer creation failed: ${result.error}`, 'error');
            }
        } catch (error) {
            this.showAlert(`Offer error: ${error.message}`, 'error');
        }
    }
    
    // System Metrics Methods
    
    async loadSystemMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const metrics = await response.json();
            
            if (response.ok) {
                this.updateMetricsDisplay(metrics);
            }
        } catch (error) {
            console.error('Failed to load system metrics:', error.message || error.toString());
        }
    }
    
    updateMetricsDisplay(metrics) {
        // Update predicted time
        const predictedTimeEl = document.getElementById('predicted-time');
        if (predictedTimeEl) {
            predictedTimeEl.textContent = metrics.predicted_time || '--';
            predictedTimeEl.title = `Traditional: ${metrics.traditional_time}, Saved: ${metrics.time_saved}`;
        }
        
        // Update cost savings
        const costSavingsEl = document.getElementById('cost-savings');
        if (costSavingsEl) {
            costSavingsEl.textContent = metrics.cost_savings || '--';
            costSavingsEl.title = `Traditional: ${metrics.traditional_cost}, SquashPlot: ${metrics.cost_savings}`;
        }
        
        // Update energy use
        const energyUseEl = document.getElementById('energy-use');
        if (energyUseEl) {
            energyUseEl.textContent = metrics.energy_use || '--';
            energyUseEl.title = `Traditional: ${metrics.traditional_energy}, Saved: ${metrics.energy_saved}`;
        }
        
        // Update efficiency gain
        const efficiencyGainEl = document.getElementById('efficiency-gain');
        if (efficiencyGainEl) {
            efficiencyGainEl.textContent = metrics.efficiency_gain || '--';
        }
    }
    
    setupWalletTypeButtons() {
        // Set up wallet connection buttons
        document.querySelectorAll('.wallet-type-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const walletType = btn.dataset.wallet;
                
                try {
                    // Show loading state
                    btn.style.opacity = '0.6';
                    btn.style.pointerEvents = 'none';
                    
                    // Call the appropriate wallet connection method
                    const success = await this.connectSpecificWallet(walletType);
                    
                    if (success) {
                        console.log(`${walletType} wallet connected successfully`);
                    }
                    
                } catch (error) {
                    console.error(`Failed to connect ${walletType} wallet:`, error);
                    this.showAlert(`Failed to connect ${walletType} wallet: ${error.message}`, 'error');
                } finally {
                    // Reset button state
                    btn.style.opacity = '';
                    btn.style.pointerEvents = '';
                }
            });
        });
    }
    
    async connectSpecificWallet(walletType) {
        try {
            switch(walletType) {
                case 'goby':
                    return await this.connectGobyWallet();
                case 'sage':
                    return await this.connectSageWallet();
                case 'native':
                    return await this.connectNativeWallet();
                default:
                    throw new Error(`Unknown wallet type: ${walletType}`);
            }
        } catch (error) {
            throw error;
        }
    }
    
    async connectGobyWallet() {
        // Check if Goby is available
        if (typeof window.chia === 'undefined') {
            this.showAlert('Goby wallet extension not found. Please install Goby from the Chrome Web Store.', 'error');
            return false;
        }
        
        try {
            // Request connection to Goby
            const response = await window.chia.request({
                method: 'chia_logIn',
                params: {}
            });
            
            if (response && response.length > 0) {
                this.updateWalletConnected('goby', response[0]);
                this.showAlert('Goby wallet connected successfully!', 'success');
                return true;
            } else {
                throw new Error('No keys returned from Goby wallet');
            }
        } catch (error) {
            throw new Error(`Goby connection failed: ${error.message}`);
        }
    }
    
    async connectSageWallet() {
        // For Sage, show a modal with WalletConnect instructions
        this.showWalletConnectModal('sage');
        
        // Simulate successful connection after a delay (for demo)
        setTimeout(() => {
            this.updateWalletConnected('sage', 'xch1sage_demo_address');
            this.showAlert('Sage wallet connected successfully!', 'success');
            this.closeWalletConnectModal();
        }, 3000);
        
        return true;
    }
    
    async connectNativeWallet() {
        // For Native Chia, show a modal with WalletConnect instructions
        this.showWalletConnectModal('native');
        
        // Simulate successful connection after a delay (for demo)
        setTimeout(() => {
            this.updateWalletConnected('native', 'xch1native_demo_address');
            this.showAlert('Native Chia wallet connected successfully!', 'success');
            this.closeWalletConnectModal();
        }, 3000);
        
        return true;
    }
    
    updateWalletConnected(walletType, address) {
        // Update UI for connected state
        const walletIndicator = document.getElementById('wallet-indicator');
        if (walletIndicator) {
            walletIndicator.classList.remove('disconnected');
            walletIndicator.classList.add('connected');
            walletIndicator.innerHTML = `<i class="fas fa-circle"></i><span>${walletType.charAt(0).toUpperCase() + walletType.slice(1)} Connected</span>`;
        }
        
        // Highlight the connected wallet button
        document.querySelectorAll('.wallet-type-btn').forEach(btn => {
            btn.classList.remove('connected');
            if (btn.dataset.wallet === walletType) {
                btn.classList.add('connected');
            }
        });
        
        // Show wallet content
        const walletContent = document.getElementById('wallet-content');
        if (walletContent) {
            walletContent.style.display = 'block';
            
            // Update wallet info
            const addressElement = document.getElementById('wallet-address');
            const balanceElement = document.getElementById('wallet-balance');
            
            if (addressElement) {
                addressElement.textContent = address || 'Connected';
            }
            
            if (balanceElement) {
                balanceElement.textContent = '0.00 XCH';
            }
        }
    }
    
    showWalletConnectModal(walletType) {
        const modal = document.createElement('div');
        modal.className = 'wallet-connect-modal';
        modal.id = 'wallet-connect-modal';
        modal.innerHTML = `
            <div class="modal-overlay">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>Connect ${walletType.charAt(0).toUpperCase() + walletType.slice(1)} Wallet</h3>
                        <button class="modal-close" onclick="this.closest('.wallet-connect-modal').remove()">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="modal-body">
                        <div class="qr-code-placeholder">
                            <i class="fas fa-qrcode"></i>
                            <p>Connecting to ${walletType} wallet...</p>
                            <small>Please approve the connection in your wallet</small>
                        </div>
                        <div class="connection-instructions">
                            <h4>Instructions:</h4>
                            <ol>
                                <li>Open your ${walletType} wallet</li>
                                <li>Navigate to WalletConnect or dApp connection</li>
                                <li>Approve the connection request</li>
                                <li>Your wallet will be connected automatically</li>
                            </ol>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="this.closest('.wallet-connect-modal').remove()">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    closeWalletConnectModal() {
        const modal = document.getElementById('wallet-connect-modal');
        if (modal) {
            modal.remove();
        }
    }
    
    setupPlotterModeHandlers() {
        // Handle plotter mode changes
        document.querySelectorAll('input[name="plotter-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                const mode = e.target.value;
                this.handlePlotterModeChange(mode);
            });
        });
    }
    
    handlePlotterModeChange(mode) {
        console.log(`Plotter mode changed to: ${mode}`);
        
        // Update metrics immediately when mode changes
        this.updatePlotterModeMetrics(mode);
        
        // Update compression options based on mode
        const compressionSelect = document.getElementById('compression');
        const compressionInfo = document.getElementById('compression-info');
        
        if (compressionSelect && compressionInfo) {
            switch(mode) {
                case 'madmax':
                    // Mad Max: Focus on speed, basic compression
                    compressionSelect.innerHTML = `
                        <option value="0" selected>Level 0 - No compression (108GB) - Maximum Speed</option>
                        <option value="1">Level 1 - Light compression (94GB) - Slight Speed Trade-off</option>
                        <option value="2">Level 2 - Medium compression (91GB) - Balanced</option>
                    `;
                    if (compressionInfo) compressionInfo.textContent = 'Mad Max optimized for maximum plotting speed';
                    break;
                    
                case 'bladebit':
                    // BladeBit: Focus on compression
                    compressionSelect.innerHTML = `
                        <option value="0">Level 0 - No compression (108GB)</option>
                        <option value="1">Level C1 - BladeBit compression (87GB)</option>
                        <option value="2">Level C2 - BladeBit compression (84GB)</option>
                        <option value="3">Level C3 - BladeBit compression (81GB)</option>
                        <option value="4" selected>Level C4 - BladeBit compression (78GB)</option>
                        <option value="5">Level C5 - BladeBit compression (75GB)</option>
                        <option value="6">Level C6 - BladeBit compression (72GB)</option>
                        <option value="7">Level C7 - BladeBit compression (69GB)</option>
                    `;
                    if (compressionInfo) compressionInfo.textContent = 'BladeBit optimized for maximum compression';
                    break;
                    
                case 'hybrid':
                    // Hybrid: Best of both worlds
                    compressionSelect.innerHTML = `
                        <option value="0">Level 0 - No compression (108GB)</option>
                        <option value="1">Level 1 - Light hybrid (94GB)</option>
                        <option value="2">Level 2 - Medium hybrid (91GB)</option>
                        <option value="3" selected>Level 3 - Balanced hybrid (87GB)</option>
                        <option value="4">Level 4 - Strong hybrid (84GB)</option>
                        <option value="5">Level 5 - Maximum hybrid (81GB)</option>
                    `;
                    if (compressionInfo) compressionInfo.textContent = 'Hybrid mode: Mad Max speed + BladeBit compression';
                    break;
            }
        }
        
        // Show mode-specific settings
        this.updateModeSpecificSettings(mode);
        
        // Update metrics display
        this.updatePlotterModeMetrics(mode);
    }
    
    updateModeSpecificSettings(mode) {
        // Update UI elements based on selected mode
        const threadsInput = document.getElementById('threads');
        const bucketsSelect = document.getElementById('buckets');
        const cacheSelect = document.getElementById('cache-size');
        
        if (mode === 'madmax') {
            // Mad Max optimizations
            if (threadsInput) threadsInput.placeholder = 'Recommended: CPU cores + 2';
            if (bucketsSelect) bucketsSelect.value = '256'; // Lower memory usage
        } else if (mode === 'bladebit') {
            // BladeBit optimizations  
            if (threadsInput) threadsInput.placeholder = 'Recommended: CPU cores';
            if (cacheSelect) cacheSelect.value = '32G'; // Higher cache for compression
        } else if (mode === 'hybrid') {
            // Hybrid optimizations
            if (threadsInput) threadsInput.placeholder = 'Recommended: CPU cores + 1';
            if (bucketsSelect) bucketsSelect.value = '512'; // Balanced memory
        }
    }
    
    updatePlotterModeMetrics(mode) {
        // Get user inputs for dynamic calculations
        const plotCount = parseInt(document.getElementById('count')?.value || 1);
        const compressionLevel = parseInt(document.getElementById('compression')?.value || 0);
        const threads = parseInt(document.getElementById('threads')?.value || 4);
        
        // Base metrics per mode (hardware depreciation + base costs)
        const baseMetrics = {
            madmax: {
                timePerPlot: 2.5,
                baseCostPerPlot: 8.50,  // Hardware wear + misc
                energyPerPlot: 3.2,
                efficiency: 95,
                type: 'speed'
            },
            bladebit: {
                timePerPlot: 3.2,
                baseCostPerPlot: 6.25,  // Lower hardware stress
                energyPerPlot: 2.8,
                efficiency: 85,
                type: 'space'
            },
            hybrid: {
                timePerPlot: 2.8,
                baseCostPerPlot: 7.40,  // Balanced
                energyPerPlot: 3.0,
                efficiency: 90,
                type: 'optimal'
            }
        };
        
        const metrics = baseMetrics[mode] || baseMetrics['hybrid'];
        
        // Calculate actual costs based on location
        const electricityRate = this.getElectricityRate();
        const compressionMultiplier = 1 + (compressionLevel * 0.15); // Higher compression = more time
        const threadMultiplier = Math.max(0.7, 1 - (threads - 4) * 0.05); // More threads = less time
        
        const calculatedTime = (metrics.timePerPlot * compressionMultiplier * threadMultiplier * plotCount);
        const calculatedEnergy = metrics.energyPerPlot * plotCount * threadMultiplier;
        const electricityCost = calculatedEnergy * electricityRate;
        const hardwareCost = metrics.baseCostPerPlot * plotCount;
        const totalCost = electricityCost + hardwareCost;
        const calculatedEfficiency = Math.min(98, metrics.efficiency + compressionLevel * 2);
        
        // Animate value changes
        this.animateMetricUpdate('predicted-time', this.formatTime(calculatedTime), 'vs Traditional');
        this.animateMetricUpdate('estimated-cost', `$${totalCost.toFixed(2)}`, 'Per Plot Total');
        this.animateMetricUpdate('energy-use', `${calculatedEnergy.toFixed(1)} kWh`, `@ $${electricityRate}/kWh`);
        this.animateMetricUpdate('efficiency-gain', `${Math.round(calculatedEfficiency)}% ${metrics.type}`, 'Speed vs Size');
    }
    
    animateMetricUpdate(elementId, newValue, subtitle) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        // Add updating class for animation
        element.classList.add('metric-updating');
        
        // Update after brief delay for animation
        setTimeout(() => {
            element.textContent = newValue;
            element.classList.remove('metric-updating');
            element.classList.add('metric-updated');
            
            // Remove animation class
            setTimeout(() => {
                element.classList.remove('metric-updated');
            }, 500);
        }, 150);
    }
    
    formatTime(hours) {
        if (hours < 1) {
            return `${Math.round(hours * 60)} min`;
        } else if (hours < 24) {
            const h = Math.floor(hours);
            const m = Math.round((hours - h) * 60);
            return m > 0 ? `${h}.${Math.round(m/6)} hrs` : `${h} hrs`;
        } else {
            const days = Math.floor(hours / 24);
            const remainingHours = Math.round(hours % 24);
            return `${days}d ${remainingHours}h`;
        }
    }
    
    getElectricityRate() {
        const zipcode = document.getElementById('zipcode')?.value || '';
        
        // US average electricity rates by region (cents per kWh)
        const regionalRates = {
            // Northeast (high rates)
            '0': 0.22, '1': 0.20, '2': 0.18, '3': 0.19,
            // Southeast (moderate rates)
            '2': 0.11, '3': 0.12, '4': 0.13,
            // Midwest (low rates)
            '4': 0.13, '5': 0.12, '6': 0.11, '7': 0.12,
            // West (variable rates)
            '8': 0.15, '9': 0.18,
            // Pacific (high rates)
            '9': 0.22
        };
        
        if (zipcode && zipcode.length >= 1) {
            const firstDigit = zipcode.charAt(0);
            return regionalRates[firstDigit] || 0.14; // US average fallback
        }
        
        return 0.14; // US national average
    }
    
    setupDynamicMetrics() {
        // Add event listeners for dynamic metric updates
        const dynamicInputs = [
            'count', 'compression', 'threads', 
            'buckets', 'cache-size', 'temp-dir', 'zipcode'
        ];
        
        dynamicInputs.forEach(inputId => {
            const element = document.getElementById(inputId);
            if (element) {
                element.addEventListener('input', () => {
                    // Get current plotter mode
                    const selectedMode = document.querySelector('input[name="plotter-mode"]:checked');
                    if (selectedMode) {
                        this.updatePlotterModeMetrics(selectedMode.value);
                    }
                });
                
                element.addEventListener('change', () => {
                    const selectedMode = document.querySelector('input[name="plotter-mode"]:checked');
                    if (selectedMode) {
                        this.updatePlotterModeMetrics(selectedMode.value);
                    }
                });
            }
        });
        
        // Initial metrics calculation
        const selectedMode = document.querySelector('input[name="plotter-mode"]:checked');
        if (selectedMode) {
            this.updatePlotterModeMetrics(selectedMode.value);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new SquashPlotDashboard();
});