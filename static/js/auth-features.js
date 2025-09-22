// SquashPlot Enhanced Authentication & Social Features

// Authentication Methods
SquashPlotDashboard.prototype.checkAuthenticationStatus = async function() {
    try {
        const response = await fetch('/api/wallet/status');
        const data = await response.json();
        this.isAuthenticated = data.authenticated || false;
        
        if (this.isAuthenticated) {
            await this.loadUserProfile();
        } else {
            this.showUnauthenticatedState();
        }
    } catch (error) {
        console.error('Failed to check authentication status:', error);
        this.showUnauthenticatedState();
    }
};

SquashPlotDashboard.prototype.loadUserProfile = async function() {
    try {
        const response = await fetch('/api/user/profile');
        if (response.ok) {
            this.currentUser = await response.json();
            this.updateUserInterface();
        }
    } catch (error) {
        console.error('Failed to load user profile:', error);
    }
};

SquashPlotDashboard.prototype.updateUserInterface = function() {
    if (!this.currentUser) return;
    
    // Update user avatar and name
    const userAvatar = document.getElementById('user-avatar');
    const userName = document.getElementById('user-name');
    const userFarm = document.getElementById('user-farm');
    
    if (userAvatar && this.currentUser.profile_image_url) {
        userAvatar.src = this.currentUser.profile_image_url;
        userAvatar.style.display = 'block';
        userAvatar.nextElementSibling.style.display = 'none';
    }
    
    if (userName) {
        userName.textContent = this.currentUser.display_name || 'Farmer';
    }
    
    if (userFarm) {
        userFarm.textContent = this.currentUser.farm_name || 'No farm name set';
    }
    
    // Update profile stats
    const profileFarmName = document.getElementById('profile-farm-name');
    const profileSince = document.getElementById('profile-since');
    
    if (profileFarmName) {
        profileFarmName.textContent = this.currentUser.farm_name || 'Set farm name';
    }
    
    if (profileSince && this.currentUser.last_login) {
        const date = new Date(this.currentUser.last_login);
        profileSince.textContent = date.toLocaleDateString();
    }
};

SquashPlotDashboard.prototype.showUnauthenticatedState = function() {
    const userName = document.getElementById('user-name');
    const userFarm = document.getElementById('user-farm');
    
    if (userName) userName.textContent = 'Sign in to continue';
    if (userFarm) userFarm.textContent = 'Authentication required';
};

// Social Features Event Listeners
SquashPlotDashboard.prototype.setupSocialEventListeners = function() {
    // Chat functionality
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-message');
    const channelSelector = document.getElementById('chat-channel');
    
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendChatMessage();
            }
        });
    }
    
    if (sendButton) {
        sendButton.addEventListener('click', () => {
            this.sendChatMessage();
        });
    }
    
    if (channelSelector) {
        channelSelector.addEventListener('change', (e) => {
            this.switchChatChannel(e.target.value);
        });
    }
    
    // Profile form
    const profileForm = document.getElementById('profile-form');
    if (profileForm) {
        profileForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.updateUserProfile();
        });
    }
    
    // Modal controls
    this.setupModalControls();
};

SquashPlotDashboard.prototype.setupModalControls = function() {
    // Modal close buttons
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const modal = e.target.closest('.modal');
            if (modal) {
                this.closeModal(modal.id);
            }
        });
    });
    
    // Close modals on outside click
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeModal(modal.id);
            }
        });
    });
};

// Chat Functions
SquashPlotDashboard.prototype.initializeChat = async function() {
    if (!this.isAuthenticated) return;
    
    await this.loadChatMessages('general');
    this.startChatPolling();
};

SquashPlotDashboard.prototype.loadChatMessages = async function(channel = 'general') {
    try {
        const response = await fetch(`/api/chat/messages?channel=${channel}&limit=50`);
        const data = await response.json();
        
        this.chatMessages = data.messages || [];
        this.renderChatMessages();
    } catch (error) {
        console.error('Failed to load chat messages:', error);
    }
};

SquashPlotDashboard.prototype.renderChatMessages = function() {
    const chatContainer = document.getElementById('chat-messages');
    if (!chatContainer) return;
    
    chatContainer.innerHTML = '';
    
    this.chatMessages.forEach(message => {
        const messageEl = document.createElement('div');
        messageEl.className = 'chat-message';
        
        messageEl.innerHTML = `
            <div class="message-header">
                <img src="${message.user_avatar || ''}" alt="${message.user_name}" class="message-avatar" onerror="this.style.display='none'">
                <span class="message-author">${message.user_name}</span>
                <span class="message-time">${this.formatTime(message.timestamp)}</span>
            </div>
            <div class="message-content">${this.escapeHtml(message.message)}</div>
        `;
        
        chatContainer.appendChild(messageEl);
    });
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
};

SquashPlotDashboard.prototype.sendChatMessage = async function() {
    const chatInput = document.getElementById('chat-input');
    const channelSelector = document.getElementById('chat-channel');
    
    if (!chatInput || !this.isAuthenticated) return;
    
    const message = chatInput.value.trim();
    const channel = channelSelector ? channelSelector.value : 'general';
    
    if (!message) return;
    
    try {
        const response = await fetch('/api/chat/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                channel: channel
            })
        });
        
        if (response.ok) {
            chatInput.value = '';
            await this.loadChatMessages(channel);
        }
    } catch (error) {
        console.error('Failed to send message:', error);
    }
};

SquashPlotDashboard.prototype.switchChatChannel = function(channel) {
    this.loadChatMessages(channel);
};

SquashPlotDashboard.prototype.startChatPolling = function() {
    // Poll for new messages every 5 seconds
    setInterval(() => {
        if (this.isAuthenticated) {
            const channelSelector = document.getElementById('chat-channel');
            const channel = channelSelector ? channelSelector.value : 'general';
            this.loadChatMessages(channel);
        }
    }, 5000);
};

// Pool Management
SquashPlotDashboard.prototype.loadUserPools = async function() {
    if (!this.isAuthenticated) return;
    
    try {
        const response = await fetch('/api/user/pools');
        const data = await response.json();
        
        this.currentPools = data.memberships || [];
        this.renderCurrentPools();
        
        // Load recommended pools
        await this.loadRecommendedPools();
    } catch (error) {
        console.error('Failed to load user pools:', error);
    }
};

SquashPlotDashboard.prototype.loadRecommendedPools = async function() {
    try {
        const response = await fetch('/api/pools');
        const data = await response.json();
        
        this.renderRecommendedPools(data.pools || []);
    } catch (error) {
        console.error('Failed to load recommended pools:', error);
    }
};

SquashPlotDashboard.prototype.renderCurrentPools = function() {
    const container = document.getElementById('current-pools');
    if (!container) return;
    
    if (this.currentPools.length === 0) {
        container.innerHTML = '<p class="no-pools">No pools joined yet</p>';
        return;
    }
    
    container.innerHTML = this.currentPools.map(pool => `
        <div class="pool-item">
            <div class="pool-info">
                <span class="pool-name">${pool.pool_name}</span>
                <span class="pool-stats">${pool.plot_count} plots • ${pool.effective_space_tb.toFixed(1)} TB</span>
            </div>
            <div class="pool-rewards">
                <span class="reward-label">24h Rewards</span>
                <span class="reward-value">${pool.rewards_24h_xch.toFixed(4)} XCH</span>
            </div>
        </div>
    `).join('');
};

SquashPlotDashboard.prototype.renderRecommendedPools = function(pools) {
    const container = document.getElementById('recommended-pools');
    if (!container) return;
    
    container.innerHTML = pools.slice(0, 3).map(pool => `
        <div class="recommended-pool">
            <div class="pool-header">
                <span class="pool-name">${pool.pool_name}</span>
                <span class="pool-fee">${pool.fee_percent}% fee</span>
            </div>
            <div class="pool-stats">
                <span>Space: ${pool.pool_space_tb.toFixed(0)} TB</span>
                <span>Farmers: ${pool.total_farmers}</span>
            </div>
            <button class="btn-primary btn-sm" onclick="window.dashboard.joinPool(${pool.id})">Join Pool</button>
        </div>
    `).join('');
};

SquashPlotDashboard.prototype.joinPool = async function(poolId) {
    if (!this.isAuthenticated) {
        alert('Please sign in to join a pool');
        return;
    }
    
    try {
        const response = await fetch('/api/pools/join', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                pool_id: poolId,
                farmer_id: this.currentUser?.wallet_address || '',
                payout_address: this.currentUser?.wallet_address || ''
            })
        });
        
        if (response.ok) {
            alert('Successfully joined pool!');
            await this.loadUserPools();
            this.closeModal('pool-modal');
        } else {
            const error = await response.json();
            alert(error.error || 'Failed to join pool');
        }
    } catch (error) {
        console.error('Failed to join pool:', error);
        alert('Failed to join pool');
    }
};

// Profile Management
SquashPlotDashboard.prototype.updateUserProfile = async function() {
    if (!this.isAuthenticated) return;
    
    const farmName = document.getElementById('edit-farm-name').value;
    const walletAddress = document.getElementById('edit-wallet-address').value;
    const preferredPool = document.getElementById('edit-preferred-pool').value;
    
    try {
        const response = await fetch('/api/user/profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                farm_name: farmName,
                wallet_address: walletAddress,
                preferred_pool: preferredPool
            })
        });
        
        if (response.ok) {
            alert('Profile updated successfully!');
            await this.loadUserProfile();
            this.closeModal('profile-modal');
        }
    } catch (error) {
        console.error('Failed to update profile:', error);
        alert('Failed to update profile');
    }
};

// Utility Functions
SquashPlotDashboard.prototype.formatTime = function(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
};

SquashPlotDashboard.prototype.escapeHtml = function(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
};

SquashPlotDashboard.prototype.closeModal = function(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
};

SquashPlotDashboard.prototype.showModal = function(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'flex';
    }
};

// Global functions for modal controls
window.showProfileModal = function() {
    if (window.dashboard && window.dashboard.isAuthenticated) {
        // Pre-fill form with current user data
        if (window.dashboard.currentUser) {
            document.getElementById('edit-farm-name').value = window.dashboard.currentUser.farm_name || '';
            document.getElementById('edit-wallet-address').value = window.dashboard.currentUser.wallet_address || '';
            document.getElementById('edit-preferred-pool').value = window.dashboard.currentUser.preferred_pool || '';
        }
        window.dashboard.showModal('profile-modal');
    } else {
        alert('Please sign in to edit your profile');
    }
};

window.showPoolModal = function() {
    if (window.dashboard && window.dashboard.isAuthenticated) {
        window.dashboard.showModal('pool-modal');
        window.dashboard.loadRecommendedPools();
    } else {
        alert('Please sign in to join pools');
    }
};

window.showROIModal = function() {
    window.dashboard.showModal('roi-modal');
};

window.showAnalyticsModal = function() {
    alert('Detailed analytics coming soon!');
};

window.closeModal = function(modalId) {
    if (window.dashboard) {
        window.dashboard.closeModal(modalId);
    }
};