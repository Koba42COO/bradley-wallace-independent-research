import * as vscode from 'vscode';
import axios, { AxiosResponse } from 'axios';
import { AIVAClient } from './aivaClient';
import { ChatViewProvider } from './chatViewProvider';
import { InlineCompletionProvider } from './inlineCompletionProvider';

let aivaClient: AIVAClient;
let chatViewProvider: ChatViewProvider;
let inlineCompletionProvider: InlineCompletionProvider;

export async function activate(context: vscode.ExtensionContext) {
    console.log('AIVA Local Assistant is now active!');

    // Initialize AIVA client
    const config = vscode.workspace.getConfiguration('aiva');
    aivaClient = new AIVAClient(
        config.get('server.url', 'http://localhost:8000'),
        config.get('model.name', 'mixtral-8x7b-instruct')
    );

    // Test connection to AIVA server
    try {
        await aivaClient.healthCheck();
        vscode.window.showInformationMessage('AIVA Local Assistant connected successfully!');
    } catch (error) {
        vscode.window.showErrorMessage(`AIVA Local Assistant: Could not connect to server. ${error}`);
        return;
    }

    // Register chat view provider
    chatViewProvider = new ChatViewProvider(context.extensionUri, aivaClient);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, chatViewProvider)
    );

    // Register inline completion provider
    if (config.get('enableInlineSuggestions', true)) {
        inlineCompletionProvider = new InlineCompletionProvider(aivaClient);
        context.subscriptions.push(
            vscode.languages.registerInlineCompletionItemProvider(
                { pattern: '**' },
                inlineCompletionProvider
            )
        );
    }

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('aiva.chat.open', () => {
            vscode.commands.executeCommand('aiva-chat.focus');
        }),

        vscode.commands.registerCommand('aiva.code.complete', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor');
                return;
            }

            const document = editor.document;
            const selection = editor.selection;
            const cursorPosition = selection.active;

            // Get context around cursor
            const contextRange = new vscode.Range(
                Math.max(0, cursorPosition.line - 10),
                0,
                cursorPosition.line,
                cursorPosition.character
            );

            const context = document.getText(contextRange);

            try {
                const completion = await aivaClient.completeCode(context, document.languageId);
                if (completion) {
                    editor.edit(editBuilder => {
                        editBuilder.insert(cursorPosition, completion);
                    });
                }
            } catch (error) {
                vscode.window.showErrorMessage(`AIVA completion failed: ${error}`);
            }
        }),

        vscode.commands.registerCommand('aiva.code.explain', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showErrorMessage('Please select code to explain');
                return;
            }

            const selectedText = editor.document.getText(editor.selection);
            const language = editor.document.languageId;

            try {
                const explanation = await aivaClient.explainCode(selectedText, language);
                chatViewProvider.addMessage('user', `Explain this ${language} code:\n\`\`\`${language}\n${selectedText}\n\`\`\``);
                chatViewProvider.addMessage('assistant', explanation);
                vscode.commands.executeCommand('aiva-chat.focus');
            } catch (error) {
                vscode.window.showErrorMessage(`AIVA explanation failed: ${error}`);
            }
        }),

        vscode.commands.registerCommand('aiva.code.refactor', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showErrorMessage('Please select code to refactor');
                return;
            }

            const selectedText = editor.document.getText(editor.selection);
            const language = editor.document.languageId;

            try {
                const refactored = await aivaClient.refactorCode(selectedText, language);
                editor.edit(editBuilder => {
                    editBuilder.replace(editor.selection, refactored);
                });
                vscode.window.showInformationMessage('Code refactored by AIVA!');
            } catch (error) {
                vscode.window.showErrorMessage(`AIVA refactoring failed: ${error}`);
            }
        }),

        vscode.commands.registerCommand('aiva.code.optimize', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showErrorMessage('Please select code to optimize');
                return;
            }

            const selectedText = editor.document.getText(editor.selection);
            const language = editor.document.languageId;

            try {
                const optimized = await aivaClient.optimizeCode(selectedText, language);
                editor.edit(editBuilder => {
                    editBuilder.replace(editor.selection, optimized);
                });
                vscode.window.showInformationMessage('Code optimized by AIVA!');
            } catch (error) {
                vscode.window.showErrorMessage(`AIVA optimization failed: ${error}`);
            }
        }),

        vscode.commands.registerCommand('aiva.code.debug', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || editor.selection.isEmpty) {
                vscode.window.showErrorMessage('Please select code to debug');
                return;
            }

            const selectedText = editor.document.getText(editor.selection);
            const language = editor.document.languageId;

            try {
                const debugHelp = await aivaClient.debugCode(selectedText, language);
                chatViewProvider.addMessage('user', `Help debug this ${language} code:\n\`\`\`${language}\n${selectedText}\n\`\`\``);
                chatViewProvider.addMessage('assistant', debugHelp);
                vscode.commands.executeCommand('aiva-chat.focus');
            } catch (error) {
                vscode.window.showErrorMessage(`AIVA debug help failed: ${error}`);
            }
        }),

        vscode.commands.registerCommand('aiva.project.analyze', async () => {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                vscode.window.showErrorMessage('No workspace folder open');
                return;
            }

            try {
                const analysis = await aivaClient.analyzeProject(workspaceFolder.uri.fsPath);
                chatViewProvider.addMessage('user', `Analyze my project: ${workspaceFolder.name}`);
                chatViewProvider.addMessage('assistant', analysis);
                vscode.commands.executeCommand('aiva-chat.focus');
            } catch (error) {
                vscode.window.showErrorMessage(`AIVA project analysis failed: ${error}`);
            }
        })
    );

    // Listen for configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('aiva')) {
                const newConfig = vscode.workspace.getConfiguration('aiva');
                aivaClient.updateConfig(
                    newConfig.get('server.url', 'http://localhost:8000'),
                    newConfig.get('model.name', 'mixtral-8x7b-instruct')
                );
            }
        })
    );

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.command = 'aiva.chat.open';
    statusBarItem.text = '$(robot) AIVA';
    statusBarItem.tooltip = 'Open AIVA Chat';
    context.subscriptions.push(statusBarItem);
    statusBarItem.show();

    console.log('AIVA Local Assistant extension fully activated');
}

export function deactivate() {
    console.log('AIVA Local Assistant is deactivating');
}
