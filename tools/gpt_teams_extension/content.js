// Content script for ChatGPT pages
// Provides DOM access for conversation extraction

class ChatGPTContentScript {
    constructor() {
        this.init();
    }

    init() {
        // Listen for messages from popup/background
        HOST_REDACTED_23((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep message channel open for async responses
        });

        // Inject a small indicator that extension is active
        this.injectActiveIndicator();

        console.log('🤖 GPT Archive Extension loaded on ChatGPT page');
    }

    injectActiveIndicator() {
        // Add a small indicator in the top-right corner
        const indicator = document.createElement('div');
        indicator.id = 'gpt-archive-indicator';
        indicator.innerHTML = '🧠';
        indicator.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            cursor: pointer;
            z-index: 10000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            opacity: 0.7;
            transition: opacity 0.3s;
        `;

        indicator.title = 'GPT Archive Extension Active';
        indicator.addEventListener('mouseenter', () => indicator.style.opacity = '1');
        indicator.addEventListener('mouseleave', () => indicator.style.opacity = '0.7');

        document.body.appendChild(indicator);
    }

    async handleMessage(request, sender, sendResponse) {
        try {
            switch (request.action) {
                case 'extract_conversation':
                    const data = await this.extractCurrentConversation();
                    sendResponse({ success: true, data });
                    break;

                case 'capture_conversation':
                    const captureData = await this.captureVisibleConversation();
                    sendResponse(captureData);
                    break;

                case 'get_conversation_list':
                    const conversations = await this.getConversationList();
                    sendResponse({ success: true, conversations });
                    break;

                case 'check_login_status':
                    const isLoggedIn = this.checkLoginStatus();
                    sendResponse({ success: true, loggedIn: isLoggedIn });
                    break;

                case 'get_access_token':
                    const token = await this.getAccessToken();
                    sendResponse({ success: true, token });
                    break;

                default:
                    sendResponse({ success: false, error: 'Unknown action' });
            }
        } catch (error) {
            console.error('Content script error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }

    async extractCurrentConversation() {
        // Try multiple methods to extract conversation data

        // Method 1: Check for React state or global variables
        const reactData = this.extractFromReact();
        if (reactData) {
            return reactData;
        }

        // Method 2: Extract from DOM
        const domData = this.extractFromDOM();
        if (domData) {
            return domData;
        }

        // Method 3: Extract from URL and basic page info
        return this.extractBasicInfo();
    }

    extractFromReact() {
        try {
            // Look for conversation data in React state or global variables
            const keys = Object.keys(window);

            for (const key of keys) {
                if (key.includes('__NEXT_DATA__') || key.includes('conversation')) {
                    try {
                        const data = window[key];
                        if (data && typeof data === 'object') {
                            if (data.conversation) {
                                return data.conversation;
                            }
                            if (data.props?.pageProps?.conversation) {
                                return HOST_REDACTED_27;
                            }
                        }
                    } catch (e) {
                        // Continue searching
                    }
                }
            }
        } catch (error) {
            console.warn('React extraction failed:', error);
        }

        return null;
    }

    extractFromDOM() {
        try {
            const conversation = {
                id: this.getConversationId(),
                title: this.getConversationTitle(),
                messages: [],
                created_time: Date.now() / 1000
            };

            // Extract messages from DOM
            const messageElements = document.querySelectorAll('[data-message-id], [data-testid*="message"]');

            messageElements.forEach((element, index) => {
                try {
                    const message = this.extractMessageFromElement(element, index);
                    if (message) {
                        conversation.messages.push(message);
                    }
                } catch (error) {
                    console.warn('Failed to extract message:', error);
                }
            });

            // Convert to expected format
            if (conversation.messages.length > 0) {
                conversation.mapping = {};
                conversation.messages.forEach((msg, index) => {
                    conversation.mapping[`msg_${index}`] = {
                        message: msg
                    };
                });

                return conversation;
            }

        } catch (error) {
            console.warn('DOM extraction failed:', error);
        }

        return null;
    }

    extractMessageFromElement(element, index) {
        try {
            // Determine if this is a user or assistant message
            const isUser = element.closest('[data-testid*="user"], [class*="user"]') ||
                          element.textContent.includes('You said:') ||
                          element.previousElementSibling?.textContent?.includes('You');

            const role = isUser ? 'user' : 'assistant';

            // Extract text content
            let content = element.textContent || '';

            // Clean up common prefixes
            content = content.replace(/^You said:\s*/i, '');
            content = content.replace(/^Assistant:\s*/i, '');
            content = content.replace(/^GPT:\s*/i, '');

            if (content.trim()) {
                return {
                    id: `msg_${index}_${Date.now()}`,
                    author: { role },
                    content: { parts: [content.trim()] },
                    create_time: Date.now() / 1000
                };
            }
        } catch (error) {
            console.warn('Message extraction failed:', error);
        }

        return null;
    }

    extractBasicInfo() {
        // Fallback: extract basic info from page
        const url = window.location.href;
        const conversationId = url.split('/').pop();

        return {
            id: conversationId || 'unknown',
            title: document.title || 'Untitled Conversation',
            messages: [],
            url: url,
            extracted_at: new Date().toISOString()
        };
    }

    getConversationId() {
        // Try to extract conversation ID from URL
        const url = window.location.href;
        const match = url.match(/\/c\/([a-f0-9-]+)/);
        return match ? match[1] : null;
    }

    getConversationTitle() {
        // Try to get title from page title or DOM
        const titleElement = document.querySelector('[data-testid="conversation-title"], h1, title');
        if (titleElement) {
            return titleElement.textContent?.trim() || document.title;
        }
        return document.title || 'Untitled Conversation';
    }

    async getConversationList() {
        // Try to extract conversation list from sidebar
        const conversations = [];

        try {
            const convElements = document.querySelectorAll('[data-testid="conversation-item"], [href*="/c/"]');

            for (const element of convElements) {
                try {
                    const link = element.tagName === 'A' ? element : element.querySelector('a');
                    if (link) {
                        const href = link.getAttribute('href');
                        const id = href.split('/').pop();

                        const titleElement = element.querySelector('[data-testid="conversation-title"]') ||
                                           element.querySelector('span, div');
                        const title = titleElement?.textContent?.trim() || `Conversation ${id?.slice(0, 8)}`;

                        conversations.push({
                            id,
                            title,
                            url: `https://chatgpt.com${href}`
                        });
                    }
                } catch (error) {
                    console.warn('Failed to extract conversation from list:', error);
                }
            }
        } catch (error) {
            console.warn('Conversation list extraction failed:', error);
        }

        return conversations;
    }

    checkLoginStatus() {
        // Check various indicators of login status
        const loginIndicators = [
            '[data-testid="login-button"]',
            'button:contains("Log in")',
            'input[type="email"]',
            '[data-testid="email-input"]'
        ];

        for (const selector of loginIndicators) {
            if (document.querySelector(selector)) {
                return false; // Login elements present = not logged in
            }
        }

        // Check for logged-in indicators
        const loggedInIndicators = [
            '[data-testid="conversation-list"]',
            '[data-testid="new-chat-button"]',
            '.text-token-text-primary'
        ];

        for (const selector of loggedInIndicators) {
            if (document.querySelector(selector)) {
                return true; // Logged-in elements present
            }
        }

        return false; // Uncertain
    }

    async getAccessToken() {
        // Try to extract access token from various sources
        try {
            // Check localStorage
            const localToken = localStorage.getItem('accessToken') ||
                              localStorage.getItem('access_token') ||
                              localStorage.getItem('__Secure-accessToken');

            if (localToken) {
                return localToken;
            }

            // Check sessionStorage
            const sessionToken = sessionStorage.getItem('accessToken') ||
                                sessionStorage.getItem('access_token');

            if (sessionToken) {
                return sessionToken;
            }

            // Try to extract from cookies
            const cookies = document.cookie.split(';');
            for (const cookie of cookies) {
                const [name, value] = cookie.trim().split('=');
                if (name.includes('access') && name.includes('token')) {
                    return value;
                }
            }

            // Try to extract from fetch requests (advanced)
            // This would require intercepting network requests

        } catch (error) {
            console.warn('Token extraction failed:', error);
        }

        return null;
    }

    async captureVisibleConversation() {
        try {
            // Get conversation title
            const title = this.getConversationTitle();

            // Select all visible conversation text
            const conversationElements = document.querySelectorAll('[data-message-id], [data-testid*="message"], .message, .conversation-message');

            let fullContent = '';
            let messageCount = 0;

            for (const element of conversationElements) {
                // Skip system messages, buttons, etc.
                if (element.querySelector('button, .button, [role="button"]')) {
                    continue;
                }

                const text = element.textContent?.trim();
                if (text && text.length > 10) { // Skip very short fragments
                    fullContent += text + '\n\n---\n\n';
                    messageCount++;
                }
            }

            // If no structured messages found, try to get all text from main content area
            if (messageCount === 0) {
                const mainContent = document.querySelector('[data-testid="conversation-main"], main, [role="main"]');
                if (mainContent) {
                    fullContent = mainContent.textContent?.trim() || '';
                    // Estimate message count based on content
                    messageCount = Math.max(1, Math.floor(fullContent.split('\n').length / 20));
                }
            }

            // Clean up the content
            fullContent = fullContent.replace(/\n{3,}/g, '\n\n'); // Remove excessive newlines
            fullContent = fullContent.trim();

            const wordCount = fullContent.split(/\s+/).filter(word => word.length > 0).length;

            return {
                success: true,
                title: title,
                content: fullContent,
                message_count: messageCount,
                word_count: wordCount,
                url: window.location.href,
                captured_at: new Date().toISOString()
            };

        } catch (error) {
            console.error('Conversation capture error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }
}

// Initialize content script
new ChatGPTContentScript();
