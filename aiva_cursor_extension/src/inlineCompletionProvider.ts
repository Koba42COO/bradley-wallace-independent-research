import * as vscode from 'vscode';
import { AIVAClient } from './aivaClient';

export class InlineCompletionProvider implements vscode.InlineCompletionItemProvider {
    constructor(private readonly aivaClient: AIVAClient) {}

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[] | vscode.InlineCompletionList> {
        // Don't provide completions if there's already text after the cursor
        const line = document.lineAt(position.line);
        if (position.character < line.text.length && !line.text.substring(position.character).trim().match(/^[\)\]\}\s]*$/)) {
            return [];
        }

        // Get context around the cursor
        const config = vscode.workspace.getConfiguration('aiva');
        const contextLines = 10;

        const startLine = Math.max(0, position.line - contextLines);
        const contextRange = new vscode.Range(startLine, 0, position.line, position.character);
        const contextText = document.getText(contextRange);

        // Don't provide completions for very short context
        if (contextText.trim().length < 10) {
            return [];
        }

        try {
            // Get completion from AIVA
            const completion = await this.aivaClient.completeCode(
                contextText,
                document.languageId,
                {
                    maxTokens: 50, // Shorter completions for inline
                    temperature: 0.3,
                    stop: ['\n', ';', '{', '}']
                }
            );

            if (!completion || completion.trim().length === 0) {
                return [];
            }

            // Clean up completion
            const cleanCompletion = this._cleanCompletion(completion, document.languageId);

            if (!cleanCompletion) {
                return [];
            }

            const completionItem = new vscode.InlineCompletionItem(
                cleanCompletion,
                new vscode.Range(position, position)
            );

            // Add metadata for better UX
            completionItem.filterText = cleanCompletion;
            completionItem.insertText = cleanCompletion;
            completionItem.documentation = new vscode.MarkdownString(
                `*AIVA Local Assistant* - AI-generated completion`
            );

            return [completionItem];

        } catch (error) {
            console.warn('AIVA inline completion failed:', error);
            return [];
        }
    }

    private _cleanCompletion(completion: string, languageId: string): string {
        if (!completion) return '';

        let cleaned = completion.trim();

        // Remove common unwanted patterns
        const unwantedPatterns = [
            /^```[\w]*\n?/,
            /\n```$/,
            /^[\s]*```[\s]*$/,
            /\/\/.*$/,  // Remove end-of-line comments
            /\/\*.*?\*\//g,  // Remove block comments
        ];

        for (const pattern of unwantedPatterns) {
            cleaned = cleaned.replace(pattern, '');
        }

        // Language-specific cleaning
        switch (languageId) {
            case 'python':
                // Stop at class/function definitions, imports, etc.
                cleaned = cleaned.split('\nclass ')[0];
                cleaned = cleaned.split('\ndef ')[0];
                cleaned = cleaned.split('\nimport ')[0];
                cleaned = cleaned.split('\nfrom ')[0];
                break;

            case 'javascript':
            case 'typescript':
                // Stop at function declarations, imports, etc.
                cleaned = cleaned.split('\nfunction ')[0];
                cleaned = cleaned.split('\nconst ')[0];
                cleaned = cleaned.split('\nlet ')[0];
                cleaned = cleaned.split('\nvar ')[0];
                cleaned = cleaned.split('\nimport ')[0];
                cleaned = cleaned.split('\nexport ')[0];
                break;

            case 'java':
            case 'csharp':
                // Stop at method/class declarations
                cleaned = cleaned.split('\npublic ')[0];
                cleaned = cleaned.split('\nprivate ')[0];
                cleaned = cleaned.split('\nprotected ')[0];
                cleaned = cleaned.split('\nclass ')[0];
                break;
        }

        // Remove trailing punctuation that might break syntax
        cleaned = cleaned.replace(/[;,]\s*$/, '');

        // Limit length for inline completions
        if (cleaned.length > 100) {
            cleaned = cleaned.substring(0, 100);
            // Try to cut at word boundary
            const lastSpace = cleaned.lastIndexOf(' ');
            if (lastSpace > 50) {
                cleaned = cleaned.substring(0, lastSpace);
            }
        }

        return cleaned.trim();
    }
}
