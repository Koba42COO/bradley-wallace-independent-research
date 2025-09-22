/**
 * SquashPlot Wallet Connection Manager
 * Supports Goby, Sage, and Native Chia wallet connections
 */

class ChiaWalletConnector {
    constructor() {
        this.walletType = null;
        this.isConnected = false;
        this.walletData = null;
        this.listeners = new Map();
        
        // Initialize wallet detection
        this.detectAvailableWallets();
    }
    
    detectAvailableWallets() {
        const availability = {
            goby: this.isGobyAvailable(),
            sage: true, // Always available via WalletConnect
            native: true // Always available via WalletConnect
        };
        
        // Update UI based on availability
        this.updateWalletButtonStates(availability);
        return availability;
    }
    
    isGobyAvailable() {
        return typeof window !== 'undefined' && !!window.chia;
    }
    
    updateWalletButtonStates(availability) {
        const gobyBtn = document.getElementById('goby-connect-btn');
        const sageBtn = document.getElementById('sage-connect-btn');
        const nativeBtn = document.getElementById('native-connect-btn');
        
        if (!availability.goby && gobyBtn) {
            gobyBtn.style.opacity = '0.5';
            gobyBtn.title = 'Goby extension not detected. Please install from Chrome Web Store.';
        }
    }
    
    async connectWallet(walletType) {
        try {
            switch (walletType) {
                case 'goby':
                    return await this.connectGoby();
                case 'sage':
                    return await this.connectSage();
                case 'native':
                    return await this.connectNative();
                default:
                    throw new Error(`Unknown wallet type: ${walletType}`);
            }
        } catch (error) {
            console.error(`Failed to connect ${walletType} wallet:`, error);
            this.emit('error', { walletType, error: error.message });
            return false;
        }
    }
    
    async connectGoby() {
        if (!this.isGobyAvailable()) {
            throw new Error('Goby wallet extension not found. Please install Goby from the Chrome Web Store.');
        }
        
        try {
            // Request connection to Goby
            const response = await window.chia.request({
                method: 'chia_logIn',
                params: {}
            });
            
            if (response && response.length > 0) {
                this.walletType = 'goby';
                this.isConnected = true;
                this.walletData = {
                    publicKeys: response,
                    address: response[0], // Use first public key as primary
                    walletType: 'goby'
                };
                
                // Get wallet balance
                await this.updateWalletBalance();
                
                this.emit('connected', this.walletData);
                return true;
            }
            
            throw new Error('No keys returned from Goby wallet');
            
        } catch (error) {
            throw new Error(`Goby connection failed: ${error.message}`);
        }
    }
    
    async connectSage() {
        try {
            // For Sage, we'll use WalletConnect
            const wcUri = await this.initiateWalletConnect('sage');
            
            // Show QR code or connection URI to user
            this.showWalletConnectModal(wcUri, 'sage');
            
            // Wait for connection
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Sage wallet connection timeout'));
                }, 60000); // 1 minute timeout
                
                this.once('walletconnect_connected', (data) => {
                    clearTimeout(timeout);
                    this.walletType = 'sage';
                    this.isConnected = true;
                    this.walletData = {
                        ...data,
                        walletType: 'sage'
                    };
                    this.emit('connected', this.walletData);
                    resolve(true);
                });
            });
            
        } catch (error) {
            throw new Error(`Sage connection failed: ${error.message}`);
        }
    }
    
    async connectNative() {
        try {
            // For Native Chia wallet, use WalletConnect
            const wcUri = await this.initiateWalletConnect('native');
            
            // Show QR code or connection URI to user
            this.showWalletConnectModal(wcUri, 'native');
            
            // Wait for connection
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Native wallet connection timeout'));
                }, 60000); // 1 minute timeout
                
                this.once('walletconnect_connected', (data) => {
                    clearTimeout(timeout);
                    this.walletType = 'native';
                    this.isConnected = true;
                    this.walletData = {
                        ...data,
                        walletType: 'native'
                    };
                    this.emit('connected', this.walletData);
                    resolve(true);
                });
            });
            
        } catch (error) {
            throw new Error(`Native wallet connection failed: ${error.message}`);
        }
    }
    
    async initiateWalletConnect(walletType) {
        // This would integrate with WalletConnect protocol
        // For now, return a mock URI - in production, this would generate a real WalletConnect URI
        const mockUri = `wc:chia-${walletType}-${Date.now()}@2?relay-protocol=irn&symKey=mock`;
        return mockUri;
    }
    
    showWalletConnectModal(uri, walletType) {
        // Create modal for showing QR code and instructions
        const modal = document.createElement('div');
        modal.className = 'wallet-connect-modal';
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
                            <p>QR Code would be generated here</p>
                            <small>URI: ${uri}</small>
                        </div>
                        <div class="connection-instructions">
                            <h4>Instructions:</h4>
                            <ol>
                                <li>Open your ${walletType} wallet</li>
                                <li>Navigate to WalletConnect or dApp connection</li>
                                <li>Scan this QR code or copy the URI above</li>
                                <li>Approve the connection in your wallet</li>
                            </ol>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-secondary" onclick="this.closest('.wallet-connect-modal').remove()">
                            Cancel
                        </button>
                        <button class="btn btn-primary" onclick="window.walletConnector.simulateConnection('${walletType}')">
                            Simulate Connection (Demo)
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    // Demo function to simulate successful connection
    simulateConnection(walletType) {
        setTimeout(() => {
            const mockData = {
                address: `xch1${Math.random().toString(36).substring(2, 60)}`,
                publicKeys: [`${Math.random().toString(36).substring(2, 60)}`],
                balance: Math.random() * 100
            };
            
            this.emit('walletconnect_connected', mockData);
            
            // Close modal
            const modal = document.querySelector('.wallet-connect-modal');
            if (modal) modal.remove();
        }, 1000);
    }
    
    async updateWalletBalance() {
        if (!this.isConnected || !this.walletData) return;
        
        try {
            switch (this.walletType) {
                case 'goby':
                    // Get balance from Goby
                    const balance = await window.chia.request({
                        method: 'chia_getWalletBalance',
                        params: { walletId: 1 }
                    });
                    this.walletData.balance = balance;
                    break;
                    
                case 'sage':
                case 'native':
                    // Balance would be retrieved via WalletConnect
                    // For demo, keep existing balance
                    break;
            }
            
            this.emit('balance_updated', this.walletData);
            
        } catch (error) {
            console.error('Failed to update wallet balance:', error);
        }
    }
    
    async sendTransaction(recipient, amount, fee = 0.00001) {
        if (!this.isConnected) {
            throw new Error('Wallet not connected');
        }
        
        try {
            switch (this.walletType) {
                case 'goby':
                    return await window.chia.request({
                        method: 'chia_sendTransaction',
                        params: {
                            walletId: 1,
                            address: recipient,
                            amount: Math.floor(amount * 1000000000000), // Convert to mojos
                            fee: Math.floor(fee * 1000000000000)
                        }
                    });
                    
                case 'sage':
                case 'native':
                    // Would use WalletConnect to send transaction
                    throw new Error('WalletConnect transactions not implemented in demo');
                    
                default:
                    throw new Error(`Unknown wallet type: ${this.walletType}`);
            }
        } catch (error) {
            throw new Error(`Transaction failed: ${error.message}`);
        }
    }
    
    disconnect() {
        this.walletType = null;
        this.isConnected = false;
        this.walletData = null;
        
        this.emit('disconnected');
    }
    
    // Event system
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }
    
    once(event, callback) {
        const onceCallback = (...args) => {
            callback(...args);
            this.off(event, onceCallback);
        };
        this.on(event, onceCallback);
    }
    
    off(event, callback) {
        if (this.listeners.has(event)) {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }
}

// Global wallet connector instance
window.walletConnector = new ChiaWalletConnector();