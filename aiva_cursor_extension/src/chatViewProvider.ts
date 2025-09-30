import * as vscode from 'vscode';
import { AIVAClient, ChatMessage } from './aivaClient';

export class ChatViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'aiva-chat';
    private _view?: vscode.WebviewView;
    private _messages: ChatMessage[] = [];

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _aivaClient: AIVAClient
    ) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                this._extensionUri
            ]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.type) {
                    case 'sendMessage':
                        await this._handleUserMessage(message.text);
                        break;
                    case 'clearChat':
                        this._messages = [];
                        this._updateWebview();
                        break;
                }
            },
            undefined,
            context.subscriptions
        );

        // Initialize with welcome message
        if (this._messages.length === 0) {
            this._messages.push({
                role: 'assistant',
                content: 'Hello! I\'m AIVA, your local AI assistant. I\'m running entirely on your machine for maximum privacy. How can I help you with your code today?'
            });
            this._updateWebview();
        }
    }

    public addMessage(role: 'user' | 'assistant', content: string): void {
        this._messages.push({ role, content });
        this._updateWebview();
    }

    private async _handleUserMessage(text: string): Promise<void> {
        // Add user message
        this.addMessage('user', text);

        try {
            // Get AIVA response
            const response = await this._aivaClient.chatCompletion(this._messages);

            // Add assistant response
            this.addMessage('assistant', response);
        } catch (error: any) {
            this.addMessage('assistant', `❌ Error: ${error.message}`);
        }
    }

    private _updateWebview(): void {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'updateMessages',
                messages: this._messages
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.css'));

        const nonce = getNonce();

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link href="${styleUri}" rel="stylesheet">
                <title>AIVA Chat</title>
            </head>
            <body>
                <div class="chat-container">
                    <div class="chat-header">
                        <h3>🤖 AIVA Local Assistant</h3>
                        <div class="chat-controls">
                            <button id="clearBtn" title="Clear chat">🗑️</button>
                        </div>
                    </div>
                    <div id="messages" class="messages"></div>
                    <div class="input-container">
                        <textarea
                            id="messageInput"
                            placeholder="Ask AIVA anything about your code..."
                            rows="2"
                        ></textarea>
                        <button id="sendBtn">Send</button>
                    </div>
                </div>
                <script nonce="${nonce}" src="${scriptUri}"></script>
            </body>
            </html>`;
    }
}

function getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
