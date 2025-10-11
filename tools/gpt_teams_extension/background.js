// Background service worker for GPT Teams Archive Extension

class GPTArchiveBackground {
    constructor() {
        this.backendUrl = 'http://localhost:8765';
        this.init();
    }

    init() {
        // Handle extension installation
        chrome.runtime.onInstalled.addListener(() => {
            this.onInstalled();
        });

        // Handle messages from popup and content scripts
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep channel open for async responses
        });

        // Handle tab updates
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            if (changeInfo.status === 'complete' && tab.url?.includes('chatgpt.com')) {
                this.injectContentScript(tabId);
            }
        });

        console.log('🧠 GPT Archive Extension background service started');
    }

    onInstalled() {
        // Set default settings
        chrome.storage.sync.set({
            'gpt_exporter_settings': {
                limit: 100,
                since: '2024-01-01',
                includePersonal: false
            }
        });

        // Create context menu
        chrome.contextMenus.create({
            id: 'gpt-archive-export',
            title: 'Export GPT Conversation',
            contexts: ['page'],
            documentUrlPatterns: ['https://chatgpt.com/*']
        });
    }

    async handleMessage(request, sender, sendResponse) {
        try {
            switch (request.action) {
                case 'start_export':
                    const result = await this.startExport(request.data);
                    sendResponse({ success: true, result });
                    break;

                case 'check_backend':
                    const status = await this.checkBackendStatus();
                    sendResponse({ success: true, status });
                    break;

                case 'get_conversations':
                    const conversations = await this.getConversations();
                    sendResponse({ success: true, conversations });
                    break;

                case 'extract_conversation':
                    const data = await this.extractConversation(request.conversationId);
                    sendResponse({ success: true, data });
                    break;

                default:
                    sendResponse({ success: false, error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Background message error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    async checkBackendStatus() {
        try {
            const response = await fetch(`${this.backendUrl}/status`);
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    async startExport(config) {
        try {
            // Build export parameters
            const params = new URLSearchParams({
                dry_run: config.dry_run || false,
                limit: config.limit || 100,
                since: config.since || '2024-01-01',
                include_personal: config.include_personal || false
            });

            const response = await fetch(`${this.backendUrl}/export?${params}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            return result;

        } catch (error) {
            console.error('Export start error:', error);
            throw error;
        }
    }

    async getConversations() {
        // Get the active ChatGPT tab
        const tabs = await chrome.tabs.query({
            url: 'https://chatgpt.com/*',
            active: true
        });

        if (tabs.length === 0) {
            throw new Error('No active ChatGPT tab found');
        }

        // Send message to content script
        const response = await chrome.tabs.sendMessage(tabs[0].id, {
            action: 'get_conversation_list'
        });

        return response.conversations || [];
    }

    async extractConversation(conversationId) {
        // Get the active ChatGPT tab
        const tabs = await chrome.tabs.query({
            url: 'https://chatgpt.com/*'
        });

        if (tabs.length === 0) {
            throw new Error('No ChatGPT tab found');
        }

        // Try to navigate to the conversation first
        if (conversationId) {
            await chrome.tabs.update(tabs[0].id, {
                url: `https://chatgpt.com/c/${conversationId}`
            });

            // Wait for page to load
            await new Promise(resolve => setTimeout(resolve, 2000));
        }

        // Extract conversation data
        const response = await chrome.tabs.sendMessage(tabs[0].id, {
            action: 'extract_conversation'
        });

        return response.data;
    }

    async injectContentScript(tabId) {
        try {
            await chrome.scripting.executeScript({
                target: { tabId },
                files: ['content.js']
            });
        } catch (error) {
            // Content script might already be injected
            console.log('Content script injection skipped:', error.message);
        }
    }

    // Context menu handler
    chrome.contextMenus.onClicked.addListener((info, tab) => {
        if (info.menuItemId === 'gpt-archive-export') {
            this.handleContextMenuExport(tab);
        }
    });

    async handleContextMenuExport(tab) {
        try {
            // Extract current conversation
            const conversationData = await this.extractConversation();

            // Send to backend for processing
            const result = await this.startExport({
                conversation_data: conversationData,
                single_conversation: true
            });

            // Show notification
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/icon48.png',
                title: 'GPT Conversation Exported',
                message: `Successfully exported conversation to AIVA memory`
            });

        } catch (error) {
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icons/icon48.png',
                title: 'Export Failed',
                message: error.message
            });
        }
    }
}

// Initialize background service
new GPTArchiveBackground();
