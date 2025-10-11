// GPT Teams Archive Extension Popup
class GPTExporterPopup {
    constructor() {
        this.backendUrl = 'http://localhost:8767';
        this.init();
    }

    async init() {
        this.bindElements();
        this.bindEvents();
        await this.checkBackendConnection();

        // Load saved settings
        await this.loadSettings();

        // Check if we're on a ChatGPT page
        await this.checkChatGPTPage();
    }

    bindElements() {
        this.status = document.getElementById('status');
        this.statusText = document.getElementById('status-text');
        this.pasteBtn = document.getElementById('paste-btn');
        this.captureBtn = document.getElementById('capture-btn');
        this.startBtn = document.getElementById('start-btn');
        this.dryRunBtn = document.getElementById('dry-run-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.progress = document.getElementById('progress');
        this.progressFill = document.querySelector('.progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.limitInput = document.getElementById('limit-input');
        this.sinceInput = document.getElementById('since-input');
        this.includePersonal = document.getElementById('include-personal');

        // Paste dialog elements
        this.pasteDialog = document.getElementById('paste-dialog');
        this.closeDialogBtn = document.getElementById('close-dialog');
        this.conversationText = document.getElementById('conversation-text');
        this.processPasteBtn = document.getElementById('process-paste');
        this.cancelPasteBtn = document.getElementById('cancel-paste');
    }

    bindEvents() {
        this.pasteBtn.addEventListener('click', () => this.showPasteDialog());
        this.captureBtn.addEventListener('click', () => this.captureCurrentConversation());
        this.startBtn.addEventListener('click', () => this.startExport(false));
        this.dryRunBtn.addEventListener('click', () => this.startExport(true));
        this.stopBtn.addEventListener('click', () => this.stopExport());

        // Paste dialog events
        this.closeDialogBtn.addEventListener('click', () => this.hidePasteDialog());
        this.cancelPasteBtn.addEventListener('click', () => this.hidePasteDialog());
        this.processPasteBtn.addEventListener('click', () => this.processPastedConversation());

        // Close dialog when clicking outside
        this.pasteDialog.addEventListener('click', (e) => {
            if (e.target === this.pasteDialog) {
                this.hidePasteDialog();
            }
        });

        // Save settings on change
        this.limitInput.addEventListener('change', () => this.saveSettings());
        this.sinceInput.addEventListener('change', () => this.saveSettings());
        this.includePersonal.addEventListener('change', () => this.saveSettings());
    }

    async checkBackendConnection() {
        try {
            const response = await fetch(`${this.backendUrl}/status`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (response.ok) {
                this.setStatus('connected', '✅ Backend Connected');
                this.enableButtons(true);
            } else {
                throw new Error('Backend not responding');
            }
        } catch (error) {
            this.setStatus('disconnected', '❌ Backend Not Found - Start Python server first');
            this.enableButtons(false);
        }
    }

    async checkChatGPTPage() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            if (tab && tab.url && tab.url.includes('chatgpt.com')) {
                this.statusText.textContent += ' • On ChatGPT Page ✅';
            } else {
                this.statusText.textContent += ' • Not on ChatGPT Page ⚠️';
                this.showNotification('Please navigate to chatgpt.com first', 'warning');
            }
        } catch (error) {
            console.error('Error checking page:', error);
        }
    }

    setStatus(type, message) {
        this.status.className = `status ${type}`;
        this.statusText.textContent = message;
    }

    enableButtons(enabled) {
        this.startBtn.disabled = !enabled;
        this.dryRunBtn.disabled = !enabled;
    }

    async captureCurrentConversation() {
        try {
            // Check if we're on a ChatGPT page
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab.url || !tab.url.includes('chatgpt.com')) {
                throw new Error('Please navigate to a ChatGPT conversation first');
            }

            this.setStatus('connected', '📋 Capturing conversation...');
            this.captureBtn.disabled = true;
            this.captureBtn.textContent = '⏳ Capturing...';

            // Send message to content script to capture conversation
            const captureResult = await chrome.tabs.sendMessage(tab.id, {
                action: 'capture_conversation'
            });

            if (!captureResult || !captureResult.success) {
                throw new Error(captureResult?.error || 'Failed to capture conversation');
            }

            // Send captured content to backend
            const metadata = {
                url: tab.url,
                title: captureResult.title || 'Untitled Conversation',
                captured_at: new Date().toISOString(),
                content: captureResult.content,
                message_count: captureResult.message_count || 1,
                word_count: captureResult.word_count || 0
            };

            const response = await fetch(`${this.backendUrl}/capture`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(metadata)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            // Show success
            this.setStatus('connected', '✅ Conversation captured and saved!');
            this.showNotification('Conversation captured successfully!', 'success');

            // Show results
            this.showCaptureResults(result);

        } catch (error) {
            console.error('Capture error:', error);
            this.showNotification(`Capture failed: ${error.message}`, 'error');
            this.setStatus('connected', '❌ Capture failed');
        } finally {
            this.captureBtn.disabled = false;
            this.captureBtn.textContent = '📋 Capture Current Conversation';
        }
    }

    showPasteDialog() {
        this.pasteDialog.style.display = 'flex';
        this.conversationText.focus();
        // Auto-select placeholder text for easy replacement
        setTimeout(() => {
            this.conversationText.select();
        }, 100);
    }

    hidePasteDialog() {
        this.pasteDialog.style.display = 'none';
        this.conversationText.value = '';
    }

    async processPastedConversation() {
        const conversationText = this.conversationText.value.trim();

        if (!conversationText) {
            this.showNotification('Please paste some conversation text first', 'warning');
            return;
        }

        try {
            this.setStatus('connected', '🔄 Processing pasted conversation...');
            this.processPasteBtn.disabled = true;
            this.processPasteBtn.textContent = '⏳ Processing...';

            // Parse the conversation text
            const parsedData = this.parseConversationText(conversationText);

            // Send to backend for processing (uses /capture endpoint)
            const response = await fetch(`${this.backendUrl}/capture`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(parsedData)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            // Hide dialog and show success
            this.hidePasteDialog();

            // Show results
            this.setStatus('connected', '✅ Conversation processed and stored!');
            this.showNotification('Conversation processed successfully!', 'success');
            this.showPasteResults(result);

        } catch (error) {
            console.error('Paste processing error:', error);
            this.showNotification(`Processing failed: ${error.message}`, 'error');
            this.setStatus('connected', '❌ Processing failed');
        } finally {
            this.processPasteBtn.disabled = false;
            this.processPasteBtn.textContent = '🚀 Process & Store';
        }
    }

    parseConversationText(text) {
        // Parse the pasted conversation text into structured format
        const lines = text.split('\n').filter(line => line.trim());
        const messages = [];
        let currentMessage = '';
        let isUserMessage = false;

        for (const line of lines) {
            // Check for new message indicators
            const userMatch = line.match(/^(You|User|Human):\s*(.+)?$/i);
            const assistantMatch = line.match(/^(Assistant|GPT|AI|ChatGPT):\s*(.+)?$/i);

            if (userMatch) {
                // Save previous message if exists
                if (currentMessage.trim()) {
                    messages.push({
                        role: isUserMessage ? 'user' : 'assistant',
                        content: currentMessage.trim()
                    });
                }
                // Start new user message
                currentMessage = userMatch[2] || '';
                isUserMessage = true;
            } else if (assistantMatch) {
                // Save previous message if exists
                if (currentMessage.trim()) {
                    messages.push({
                        role: isUserMessage ? 'user' : 'assistant',
                        content: currentMessage.trim()
                    });
                }
                // Start new assistant message
                currentMessage = assistantMatch[2] || '';
                isUserMessage = false;
            } else {
                // Continue current message
                if (currentMessage) {
                    currentMessage += '\n' + line;
                } else {
                    currentMessage = line;
                }
            }
        }

        // Add the last message
        if (currentMessage.trim()) {
            messages.push({
                role: isUserMessage ? 'user' : 'assistant',
                content: currentMessage.trim()
            });
        }

        // Extract title from first few messages
        const title = this.extractConversationTitle(messages);

        // Calculate stats
        const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
        const messageCount = messages.length;

        return {
            title: title,
            content: text,
            messages: messages,
            message_count: messageCount,
            word_count: wordCount,
            pasted_at: new Date().toISOString()
        };
    }

    extractConversationTitle(messages) {
        // Try to extract a meaningful title from the first user message
        if (messages.length > 0 && messages[0].role === 'user') {
            const firstMessage = messages[0].content;
            // Take first 50 characters or first sentence
            const title = firstMessage.split(/[.!?]/)[0].trim();
            return title.length > 50 ? title.substring(0, 47) + '...' : title;
        }
        return 'Pasted Conversation';
    }

    showPasteResults(result) {
        // Create a results display for pasted conversations
        const resultsDiv = document.createElement('div');
        resultsDiv.style.cssText = `
            margin-top: 15px;
            padding: 15px;
            background: rgba(76, 175, 80, 0.1);
            border: 1px solid #4CAF50;
            border-radius: 6px;
            font-size: 12px;
            max-height: 150px;
            overflow-y: auto;
        `;

        try {
            resultsDiv.innerHTML = `
                <strong style="color: #4CAF50;">✅ Conversation Processed!</strong><br><br>
                <strong>Title:</strong> ${result.title || 'N/A'}<br>
                <strong>Messages:</strong> ${result.message_count || 1}<br>
                <strong>Words:</strong> ${result.word_count || 0}<br>
                <strong>Classified as:</strong> ${result.classification || 'processing'}<br>
                <strong>Stored in AIVA memory</strong>
            `;
        } catch (e) {
            resultsDiv.innerHTML = '<strong style="color: #4CAF50;">✅ Conversation processed and stored in AIVA memory!</strong>';
        }

        resultsDiv.className = 'paste-results';

        // Remove any existing results
        const existingResults = document.querySelector('.paste-results');
        if (existingResults) {
            existingResults.remove();
        }

        // Insert after controls
        const controls = document.querySelector('.controls');
        controls.parentNode.insertBefore(resultsDiv, controls.nextSibling);

        // Auto-remove after 8 seconds
        setTimeout(() => {
            if (resultsDiv.parentNode) {
                resultsDiv.remove();
            }
        }, 8000);
    }

    async startExport(dryRun = false) {
        try {
            const settings = this.getSettings();

            const requestData = {
                dry_run: dryRun,
                limit: settings.limit,
                since: settings.since,
                include_personal: settings.includePersonal
            };

            this.setStatus('connected', dryRun ? '🔍 Running Dry Run...' : '🚀 Starting Export...');
            this.showProgress(true);
            this.startBtn.style.display = 'none';
            this.dryRunBtn.style.display = 'none';
            this.stopBtn.style.display = 'block';

            const response = await fetch(`${this.backendUrl}/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();

            if (result.status === 'started') {
                this.startProgressPolling();
            } else {
                throw new Error('Failed to start export');
            }

        } catch (error) {
            console.error('Export start error:', error);
            this.showNotification(`Export failed: ${error.message}`, 'error');
            this.resetButtons();
        }
    }

    async stopExport() {
        try {
            const response = await fetch(`${this.backendUrl}/stop`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (response.ok) {
                this.setStatus('connected', '⏹️ Export Stopped');
                this.showProgress(false);
                this.resetButtons();
            }
        } catch (error) {
            console.error('Stop export error:', error);
            this.showNotification('Failed to stop export', 'error');
        }
    }

    startProgressPolling() {
        this.progressInterval = setInterval(async () => {
            try {
                const response = await fetch(`${this.backendUrl}/progress`);

                if (response.ok) {
                    const progress = await response.json();

                    if (progress.running) {
                        this.updateProgress(progress);
                    } else {
                        // Export completed
                        clearInterval(this.progressInterval);
                        this.showProgress(false);
                        this.setStatus('connected', '✅ Export Completed!');
                        this.resetButtons();

                        // Show results
                        if (progress.last_run) {
                            this.showResults(progress.last_run);
                        }
                    }
                }
            } catch (error) {
                console.error('Progress check error:', error);
            }
        }, 1000);
    }

    updateProgress(progress) {
        const percentage = progress.total > 0 ?
            Math.round((progress.current / progress.total) * 100) : 0;

        this.progressFill.style.width = `${percentage}%`;
        this.progressText.textContent = progress.message ||
            `Processing: ${progress.current}/${progress.total}`;
    }

    showProgress(show) {
        this.progress.style.display = show ? 'block' : 'none';
    }

    resetButtons() {
        this.startBtn.style.display = 'block';
        this.dryRunBtn.style.display = 'block';
        this.stopBtn.style.display = 'none';

        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
    }

    getSettings() {
        return {
            limit: parseInt(this.limitInput.value) || 100,
            since: this.sinceInput.value || '2024-01-01',
            includePersonal: this.includePersonal.checked
        };
    }

    async saveSettings() {
        const settings = this.getSettings();

        try {
            await chrome.storage.sync.set({
                'gpt_exporter_settings': settings
            });
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.sync.get(['gpt_exporter_settings']);
            const settings = result.gpt_exporter_settings || {};

            this.limitInput.value = settings.limit || 100;
            this.sinceInput.value = settings.since || '2024-01-01';
            this.includePersonal.checked = settings.includePersonal || false;
        } catch (error) {
            console.error('Failed to load settings:', error);
        }
    }

    showCaptureResults(result) {
        // Create a results display for captured conversations
        const resultsDiv = document.createElement('div');
        resultsDiv.style.cssText = `
            margin-top: 15px;
            padding: 15px;
            background: rgba(76, 175, 80, 0.1);
            border: 1px solid #4CAF50;
            border-radius: 6px;
            font-size: 12px;
            max-height: 150px;
            overflow-y: auto;
        `;

        try {
            resultsDiv.innerHTML = `
                <strong style="color: #4CAF50;">✅ Capture Successful!</strong><br><br>
                <strong>Title:</strong> ${result.title || 'N/A'}<br>
                <strong>Messages:</strong> ${result.message_count || 1}<br>
                <strong>Words:</strong> ${result.word_count || 0}<br>
                <strong>Captured:</strong> ${new Date(result.captured_at).toLocaleString()}<br>
                <strong>Classification:</strong> ${result.classification || 'processing'}<br>
                ${result.file_path ? `<strong>File:</strong> ${result.file_path.split('/').pop()}` : ''}
            `;
        } catch (e) {
            resultsDiv.innerHTML = '<strong style="color: #4CAF50;">✅ Conversation captured and saved to AIVA memory!</strong>';
        }

        resultsDiv.className = 'capture-results';

        // Remove any existing results
        const existingResults = document.querySelector('.capture-results');
        if (existingResults) {
            existingResults.remove();
        }

        // Insert after controls
        const controls = document.querySelector('.controls');
        controls.parentNode.insertBefore(resultsDiv, controls.nextSibling);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (resultsDiv.parentNode) {
                resultsDiv.remove();
            }
        }, 10000);
    }

    showResults(results) {
        // Legacy method for full exports - redirect to capture results for now
        this.showCaptureResults(results);
    }

    showNotification(message, type = 'info') {
        // Use Chrome notifications API
        chrome.notifications.create({
            type: 'basic',
            iconUrl: 'icons/icon48.png',
            title: 'GPT Archive Exporter',
            message: message
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new GPTExporterPopup();
});
