(function() {
    const vscode = acquireVsCodeApi();

    // DOM elements
    const messagesContainer = document.getElementById('messages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendBtn');
    const clearButton = document.getElementById('clearBtn');

    let isLoading = false;

    // Initialize
    function init() {
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        clearButton.addEventListener('click', clearChat);
        messageInput.addEventListener('keydown', handleKeyDown);

        // Load initial messages
        vscode.postMessage({ type: 'getMessages' });
    }

    // Handle keyboard shortcuts
    function handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    }

    // Send message to AIVA
    function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isLoading) return;

        // Add user message immediately
        addMessage('user', text);
        messageInput.value = '';
        setLoading(true);

        // Send to extension
        vscode.postMessage({
            type: 'sendMessage',
            text: text
        });
    }

    // Clear chat
    function clearChat() {
        messagesContainer.innerHTML = '';
        vscode.postMessage({ type: 'clearChat' });
    }

    // Add message to UI
    function addMessage(role, content, type = 'normal') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        if (type === 'error') {
            messageDiv.classList.add('error');
        }

        // Convert markdown-like code blocks to HTML
        let formattedContent = content
            .replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, lang, code) => {
                return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
            })
            .replace(/`([^`]+)`/g, (match, code) => {
                return `<code>${escapeHtml(code)}</code>`;
            })
            .replace(/\n/g, '<br>');

        messageDiv.innerHTML = formattedContent;
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Escape HTML entities
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Set loading state
    function setLoading(loading) {
        isLoading = loading;
        sendButton.disabled = loading;
        sendButton.textContent = loading ? '...' : 'Send';

        if (loading) {
            sendButton.classList.add('loading');
        } else {
            sendButton.classList.remove('loading');
        }
    }

    // Handle messages from extension
    window.addEventListener('message', event => {
        const message = event.data;

        switch (message.type) {
            case 'updateMessages':
                // Clear and re-render all messages
                messagesContainer.innerHTML = '';
                message.messages.forEach(msg => {
                    addMessage(msg.role, msg.content);
                });
                setLoading(false);
                break;

            case 'addMessage':
                addMessage(message.role, message.content, message.messageType);
                if (message.role === 'assistant') {
                    setLoading(false);
                }
                break;

            case 'error':
                addMessage('assistant', message.text, 'error');
                setLoading(false);
                break;

            case 'loading':
                setLoading(message.loading);
                break;
        }
    });

    // Auto-focus input when view becomes visible
    document.addEventListener('focusin', () => {
        if (!isLoading) {
            messageInput.focus();
        }
    });

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
